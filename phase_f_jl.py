#!/usr/bin/env python3
"""Phase F/JL — Between-Concept Principal Angles & JL Distance Preservation.

Phase F computes principal angles between every pair of concept subspaces
(from Phase D merged bases) to detect superposition — shared dimensions
between concepts. Phase JL checks whether pairwise distances are preserved
under projection to the union subspace (from Phase E).

Key design:
  - Phase F: SVD of V_A @ V_B^T for all concept pairs per slice
  - Empirical random baseline (200 trials) for null comparison
  - JL: ALL pairs for every sample — no subsampling anywhere
  - Row-by-row distance computation for large N (>50K) to avoid OOM on pair indices
  - float32 for distances, float64 subsample for Pythagorean validation
  - Resume logic via metadata.json per slice (preemption-safe)

Outputs:
  /data/.../phase_f/
    principal_angles/L{level}/layer_{layer:02d}/{pop}/
      pairwise_angles.csv          all concept pair angles
      metadata.json
    jl_check/L{level}/layer_{layer:02d}/{pop}/
      jl_results.json              distance preservation metrics
      metadata.json
    summary/
      phase_f_principal_angles.csv
      superposition_summary.csv
      redundancy_decomposition.csv
      jl_distance_preservation.csv
  plots/phase_f/
    superposition_heatmaps/
    angle_distributions/
    cross_layer_superposition/
    tier_boxplots/
    jl_scatter/
    jl_trajectories/
    variance_budget/

Usage:
  python phase_f_jl.py --config config.yaml       # Full run
  python phase_f_jl.py --pilot                     # Smoke test (L3/layer16/all)
  python phase_f_jl.py --phase-f-only              # Angles only, no activations
  python phase_f_jl.py --jl-only                   # JL only
  python phase_f_jl.py --plots-only                # Regenerate plots from CSVs
"""

import argparse
import json
import logging
import os
import tempfile
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

# ─── GPU (CuPy) ──────────────────────────────────────────────────────────────
try:
    import cupy as cp
    _CUPY_AVAILABLE = cp.cuda.is_available()
except (ImportError, Exception):
    _CUPY_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

LEVELS = [1, 2, 3, 4, 5]          # L1 included for Phase F (angles only)
LEVELS_JL = [2, 3, 4, 5]          # L1 excluded from JL (N=64, useless)
LAYERS = [4, 6, 8, 12, 16, 20, 24, 28, 31]
MIN_POPULATION = 30
PLOT_LEVELS = [3, 4, 5]
PLOT_LAYERS = [4, 16, 31]

# Phase F
SUPERPOSITION_MARGIN_DEG = 10.0    # angle_1 < p5 - margin → flag
SELF_ANGLE_TOLERANCE_DEG = 1.0     # self-test: all angles should be < this
N_RANDOM_BASELINE_TRIALS = 200     # empirical null trials
HIDDEN_DIM = 4096                  # model hidden dimension

# JL
JL_LARGE_N_THRESHOLD = 50000       # above this, use row-by-row distance computation
JL_RANDOM_SEED = 42
JL_BATCH_SIZE = 50000              # batch size for distance computation
JL_PYTH_SUBSAMPLE = 1000           # float64 Pythagorean validation sample

# L5 carry binning (must match Phase C/D/E)
L5_CARRY_BIN_THRESHOLDS = {
    "carry_0": None,
    "carry_1": 12,
    "carry_2": 13,
    "carry_3": 9,
    "carry_4": 5,
}


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def derive_paths(cfg):
    ws = Path(cfg["paths"]["workspace"])
    dr = Path(cfg["paths"]["data_root"])
    return {
        "workspace": ws,
        "data_root": dr,
        "act_dir": dr / "activations",
        "coloring_dir": dr / "phase_a" / "coloring_dfs",
        "residualized_dir": dr / "phase_c" / "residualized",
        "phase_c_subspaces": dr / "phase_c" / "subspaces",
        "phase_d_subspaces": dr / "phase_d" / "subspaces",
        "phase_e_data": dr / "phase_e",
        "union_bases_dir": dr / "phase_e" / "union_bases",
        "phase_e_summary": dr / "phase_e" / "summary",
        # Phase F outputs
        "phase_f_data": dr / "phase_f",
        "pa_dir": dr / "phase_f" / "principal_angles",
        "jl_dir": dr / "phase_f" / "jl_check",
        "phase_f_summary": dr / "phase_f" / "summary",
        "phase_f_plots": ws / "plots" / "phase_f",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(workspace):
    log_dir = workspace / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase_f")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fh = RotatingFileHandler(log_dir / "phase_f_jl.log",
                             maxBytes=10_000_000, backupCount=3)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s",
                            datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_coloring_df(level, coloring_dir):
    path = coloring_dir / f"L{level}_coloring.pkl"
    return pd.read_pickle(path)


def load_residualized(level, layer, residualized_dir):
    """Load product-residualized activations. Note: NOT zero-padded."""
    return np.load(residualized_dir / f"level{level}_layer{layer}.npy")


def get_populations(df):
    pops = {"all": df}
    correct_df = df[df["correct"] == True]
    wrong_df = df[df["correct"] == False]
    if len(correct_df) >= MIN_POPULATION:
        pops["correct"] = correct_df
    if len(wrong_df) >= MIN_POPULATION:
        pops["wrong"] = wrong_df
    return pops


def load_merged_basis(level, layer, pop, concept_name, paths):
    """Load Phase D merged basis for one concept. Returns (dim, 4096) or (0, 4096)."""
    concept_dir = (paths["phase_d_subspaces"] / f"L{level}" /
                   f"layer_{layer:02d}" / pop / concept_name)
    basis_path = concept_dir / "merged_basis.npy"
    if not basis_path.exists():
        return np.empty((0, HIDDEN_DIM))
    return np.load(basis_path)


def load_concept_tier(level, layer, pop, concept_name, paths):
    """Load tier from Phase D metadata.json for a concept."""
    meta_path = (paths["phase_d_subspaces"] / f"L{level}" /
                 f"layer_{layer:02d}" / pop / concept_name / "metadata.json")
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                return json.load(f).get("tier", 0)
        except (json.JSONDecodeError, KeyError):
            pass
    return 0


def load_union_basis(level, layer, pop, paths):
    """Load Phase E union basis. Returns (k, 4096)."""
    path = (paths["union_bases_dir"] / f"L{level}" /
            f"layer_{layer:02d}" / pop / "union_basis.npy")
    return np.load(path)


# ═══════════════════════════════════════════════════════════════════════════════
# CONCEPT REGISTRY (replicated from Phase E for standalone execution)
# ═══════════════════════════════════════════════════════════════════════════════

def get_concept_registry(level, df, pop_name="all"):
    """Return concept dicts matching Phase C's registry exactly."""
    cols = set(df.columns)
    concepts = []

    # Tier 1: Input digits
    digit_cols = ["a_units", "a_tens", "a_hundreds",
                  "b_units", "b_tens", "b_hundreds"]
    for col in digit_cols:
        if col in cols:
            concepts.append({"name": col, "column": col, "tier": 1,
                             "preprocess": None})

    # Tier 1: Answer digits
    ad_idx = 0
    while f"ans_digit_{ad_idx}_msf" in cols:
        col = f"ans_digit_{ad_idx}_msf"
        concepts.append({"name": col, "column": col, "tier": 1,
                         "preprocess": None})
        ad_idx += 1

    # Tier 2: Carries
    carry_idx = 0
    while f"carry_{carry_idx}" in cols:
        col = f"carry_{carry_idx}"
        bin_thresh = L5_CARRY_BIN_THRESHOLDS.get(col) if level == 5 else None
        if bin_thresh is not None:
            concepts.append({"name": col, "column": col, "tier": 2,
                             "preprocess": "bin_carry_tail",
                             "bin_threshold": bin_thresh})
        else:
            concepts.append({"name": col, "column": col, "tier": 2,
                             "preprocess": "filter_min_group"})
        carry_idx += 1

    # Tier 2: Column sums
    cs_idx = 0
    while f"col_sum_{cs_idx}" in cols:
        col = f"col_sum_{cs_idx}"
        concepts.append({"name": col, "column": col, "tier": 2,
                         "preprocess": "bin_deciles"})
        cs_idx += 1

    # Tier 3: Derived
    if pop_name == "all" and "correct" in cols:
        concepts.append({"name": "correct", "column": "correct",
                         "tier": 3, "preprocess": None})
    if "n_nonzero_carries" in cols:
        concepts.append({"name": "n_nonzero_carries",
                         "column": "n_nonzero_carries",
                         "tier": 3, "preprocess": None})
    if "total_carry_sum" in cols:
        concepts.append({"name": "total_carry_sum",
                         "column": "total_carry_sum",
                         "tier": 3, "preprocess": "filter_min_group"})
    if "max_carry_value" in cols:
        concepts.append({"name": "max_carry_value",
                         "column": "max_carry_value",
                         "tier": 3, "preprocess": "filter_min_group"})
    if "n_answer_digits" in cols:
        concepts.append({"name": "n_answer_digits",
                         "column": "n_answer_digits",
                         "tier": 3, "preprocess": None})
    if "product" in cols:
        concepts.append({"name": "product_binned", "column": "product",
                         "tier": 3, "preprocess": "bin_deciles"})

    # Tier 3: Per-digit correctness (only in "all" population)
    if pop_name == "all":
        dc_idx = 0
        while f"digit_correct_pos{dc_idx}" in cols:
            col = f"digit_correct_pos{dc_idx}"
            concepts.append({"name": col, "column": col, "tier": 3,
                             "preprocess": None})
            dc_idx += 1

    # Tier 4: Partial products
    pp_cols = sorted([c for c in cols if c.startswith("pp_")])
    for col in pp_cols:
        concepts.append({"name": col, "column": col, "tier": 4,
                         "preprocess": "bin_9"})

    return concepts


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE F: PRINCIPAL ANGLES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_principal_angles(basis_a, basis_b):
    """Compute principal angles between two subspaces in degrees.

    Args:
        basis_a: (d_a, D) orthonormal basis (rows)
        basis_b: (d_b, D) orthonormal basis (rows)

    Returns:
        angles_deg: (min(d_a, d_b),) principal angles in degrees, ascending
    """
    if basis_a.shape[0] == 0 or basis_b.shape[0] == 0:
        return np.array([])
    M = basis_a @ basis_b.T
    S = np.linalg.svd(M, compute_uv=False)
    S = np.clip(S, -1.0, 1.0)
    return np.degrees(np.arccos(S))


# Cache for empirical random baselines keyed by (dim_a, dim_b)
_random_baseline_cache = {}


def compute_random_baseline(dim_a, dim_b, d=HIDDEN_DIM,
                            n_trials=N_RANDOM_BASELINE_TRIALS):
    """Empirical null distribution for θ₁ between random subspaces.

    Generates n_trials pairs of random orthonormal subspaces of dimensions
    dim_a and dim_b in R^d, computes θ₁ for each, returns statistics.

    Returns:
        dict with keys: mean, std, p5 (5th percentile), p1 (1st percentile)
    """
    key = (min(dim_a, dim_b), max(dim_a, dim_b))
    if key in _random_baseline_cache:
        return _random_baseline_cache[key]

    if dim_a == 0 or dim_b == 0:
        result = {"mean": 90.0, "std": 0.0, "p5": 90.0, "p1": 90.0}
        _random_baseline_cache[key] = result
        return result

    rng = np.random.RandomState(12345)  # fixed seed for reproducibility
    theta1_values = np.empty(n_trials)

    for i in range(n_trials):
        # Random orthonormal subspaces via QR decomposition
        A = rng.randn(d, dim_a)
        Q_a, _ = np.linalg.qr(A)
        basis_a = Q_a[:, :dim_a].T  # (dim_a, d)

        B = rng.randn(d, dim_b)
        Q_b, _ = np.linalg.qr(B)
        basis_b = Q_b[:, :dim_b].T  # (dim_b, d)

        M = basis_a @ basis_b.T
        S = np.linalg.svd(M, compute_uv=False)
        theta1_values[i] = np.degrees(np.arccos(np.clip(S[0], -1.0, 1.0)))

    result = {
        "mean": float(np.mean(theta1_values)),
        "std": float(np.std(theta1_values)),
        "p5": float(np.percentile(theta1_values, 5)),
        "p1": float(np.percentile(theta1_values, 1)),
    }
    _random_baseline_cache[key] = result
    return result


def compute_pairwise_angles_for_slice(level, layer, pop, concept_list,
                                      paths, logger):
    """Compute principal angles for ALL concept pairs in one slice.

    Returns:
        rows: list of dicts (one per pair) for DataFrame
        n_valid_concepts: number of concepts with non-empty bases
    """
    slice_id = f"L{level}/layer{layer:02d}/{pop}"

    # Load all merged bases, filter to non-empty
    # Use tier from concept_dict directly (avoids per-concept disk reads)
    bases = {}
    for concept_dict in concept_list:
        name = concept_dict["name"]
        basis = load_merged_basis(level, layer, pop, name, paths)
        if basis.shape[0] > 0:
            tier = concept_dict.get("tier", 0)
            bases[name] = (basis, tier)

    concept_names = sorted(bases.keys())
    n_valid = len(concept_names)
    n_total = len(concept_list)

    if n_valid < 2:
        logger.warning(f"  [F {slice_id}] Only {n_valid} concepts with "
                       f"bases (of {n_total}) — skipping")
        return [], n_valid

    logger.debug(f"  [F {slice_id}] {n_valid}/{n_total} concepts with bases")

    rows = []
    n_self_warnings = 0

    for i in range(n_valid):
        for j in range(i, n_valid):
            name_a = concept_names[i]
            name_b = concept_names[j]
            basis_a, tier_a = bases[name_a]
            basis_b, tier_b = bases[name_b]

            angles = compute_principal_angles(basis_a, basis_b)
            dim_a, dim_b = basis_a.shape[0], basis_b.shape[0]
            is_self = (i == j)

            # Self-pair validation
            if is_self:
                if len(angles) > 0 and angles[0] > SELF_ANGLE_TOLERANCE_DEG:
                    logger.warning(
                        f"  [F {slice_id}] Self-angle {name_a} = "
                        f"{angles[0]:.2f}° (expected ~0°)")
                    n_self_warnings += 1
                continue  # don't include self-pairs in output

            # Random baseline
            baseline = compute_random_baseline(dim_a, dim_b)
            superposition_flag = (
                len(angles) > 0 and
                angles[0] < baseline["p5"] - SUPERPOSITION_MARGIN_DEG
            )

            row = {
                "concept_a": name_a,
                "concept_b": name_b,
                "tier_a": tier_a,
                "tier_b": tier_b,
                "dim_a": dim_a,
                "dim_b": dim_b,
                "n_angles": len(angles),
                "angle_1": float(angles[0]) if len(angles) > 0 else np.nan,
                "angle_2": float(angles[1]) if len(angles) > 1 else np.nan,
                "angle_3": float(angles[2]) if len(angles) > 2 else np.nan,
                "angle_4": float(angles[3]) if len(angles) > 3 else np.nan,
                "angle_5": float(angles[4]) if len(angles) > 4 else np.nan,
                "angle_median": float(np.median(angles)) if len(angles) > 0
                    else np.nan,
                "angle_max": float(angles[-1]) if len(angles) > 0
                    else np.nan,
                "random_baseline_mean": baseline["mean"],
                "random_baseline_p5": baseline["p5"],
                "superposition_flag": superposition_flag,
            }
            rows.append(row)

    if n_self_warnings > 0:
        logger.warning(f"  [F {slice_id}] {n_self_warnings} self-angle "
                       f"warnings (should be 0)")

    return rows, n_valid


# ═══════════════════════════════════════════════════════════════════════════════
# JL: DISTANCE PRESERVATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_all_pairs(N):
    """Generate all unique (i, j) pair indices with i < j.

    For N <= JL_LARGE_N_THRESHOLD: returns (n_pairs, 2) int array.
    For N > JL_LARGE_N_THRESHOLD: returns None (caller uses row-by-row).
    """
    n_pairs = N * (N - 1) // 2
    meta = {"sampling_method": "all_pairs", "n_pairs": n_pairs, "N": N}

    if N <= JL_LARGE_N_THRESHOLD:
        ii, jj = np.triu_indices(N, k=1)
        pairs = np.column_stack([ii, jj])
        return pairs, meta

    # Too large for explicit pair indices (would need >100GB for N=122K)
    meta["sampling_method"] = "all_pairs_rowwise"
    return None, meta


def compute_jl_distances(X, V_all, pairs, batch_size=JL_BATCH_SIZE):
    """Compute full-space and projected distances for pairs (N <= JL_LARGE_N_THRESHOLD).

    X: (N, D) float32 residualized activations (sliced to population)
    V_all: (k, D) float32/64 union basis
    pairs: (n_pairs, 2) index pairs into X

    Returns: d_full, d_proj — both (n_pairs,) float32
    """
    V_all = V_all.astype(np.float32)
    if X.dtype != np.float32:
        X = X.astype(np.float32)

    if _CUPY_AVAILABLE:
        X_g = cp.asarray(X)
        V_g = cp.asarray(V_all)
        X_proj_g = (X_g @ V_g.T) @ V_g

        idx_i = pairs[:, 0]
        idx_j = pairs[:, 1]

        d_full_parts = []
        d_proj_parts = []

        for start in range(0, len(pairs), batch_size):
            end = min(start + batch_size, len(pairs))
            bi, bj = idx_i[start:end], idx_j[start:end]

            diff_full = X_g[bi] - X_g[bj]
            diff_proj = X_proj_g[bi] - X_proj_g[bj]

            d_full_parts.append(
                cp.asnumpy(cp.linalg.norm(diff_full, axis=1)))
            d_proj_parts.append(
                cp.asnumpy(cp.linalg.norm(diff_proj, axis=1)))

            del diff_full, diff_proj

        del X_g, V_g, X_proj_g
        cp.get_default_memory_pool().free_all_blocks()

        d_full = np.concatenate(d_full_parts).astype(np.float32)
        d_proj = np.concatenate(d_proj_parts).astype(np.float32)

    else:
        X_proj = (X @ V_all.T) @ V_all

        idx_i = pairs[:, 0]
        idx_j = pairs[:, 1]

        d_full_parts = []
        d_proj_parts = []

        for start in range(0, len(pairs), batch_size):
            end = min(start + batch_size, len(pairs))
            bi, bj = idx_i[start:end], idx_j[start:end]

            d_full_parts.append(
                np.linalg.norm(X[bi] - X[bj], axis=1))
            d_proj_parts.append(
                np.linalg.norm(X_proj[bi] - X_proj[bj], axis=1))

        del X_proj

        d_full = np.concatenate(d_full_parts).astype(np.float32)
        d_proj = np.concatenate(d_proj_parts).astype(np.float32)

    return d_full, d_proj


def compute_jl_distances_rowwise(X, V_all, logger):
    """Compute ALL pairwise distances for large N using row-by-row iteration.

    Memory-efficient: does NOT allocate pair index arrays (which would be
    >100GB for N=122K). Pre-allocates output arrays d_full, d_proj.

    X: (N, D) float32 activations
    V_all: (k, D) float32/64 union basis

    Returns: d_full, d_proj — both (n_pairs,) float32
    """
    N, D = X.shape
    V_all = V_all.astype(np.float32)
    if X.dtype != np.float32:
        X = X.astype(np.float32)

    n_pairs = N * (N - 1) // 2
    logger.info(f"    rowwise: allocating 2 × {n_pairs:,} float32 arrays "
                f"({n_pairs * 4 * 2 / 1e9:.1f} GB)")
    d_full = np.empty(n_pairs, dtype=np.float32)
    d_proj = np.empty(n_pairs, dtype=np.float32)

    if _CUPY_AVAILABLE:
        X_g = cp.asarray(X)
        V_g = cp.asarray(V_all)
        X_proj_g = (X_g @ V_g.T) @ V_g
        del V_g

        offset = 0
        for i in range(N):
            n_j = N - i - 1
            if n_j == 0:
                break

            diff = X_g[i + 1:] - X_g[i]
            d_full[offset:offset + n_j] = cp.asnumpy(
                cp.linalg.norm(diff, axis=1))

            diff = X_proj_g[i + 1:] - X_proj_g[i]
            d_proj[offset:offset + n_j] = cp.asnumpy(
                cp.linalg.norm(diff, axis=1))

            offset += n_j

            if i % 10000 == 0 and i > 0:
                logger.debug(f"    rowwise: {i}/{N} rows, "
                             f"{offset:,}/{n_pairs:,} pairs")

        del X_g, X_proj_g
        cp.get_default_memory_pool().free_all_blocks()

    else:
        X_proj = (X @ V_all.T) @ V_all

        offset = 0
        for i in range(N):
            n_j = N - i - 1
            if n_j == 0:
                break

            d_full[offset:offset + n_j] = np.linalg.norm(
                X[i + 1:] - X[i], axis=1)
            d_proj[offset:offset + n_j] = np.linalg.norm(
                X_proj[i + 1:] - X_proj[i], axis=1)

            offset += n_j

            if i % 5000 == 0 and i > 0:
                logger.debug(f"    rowwise: {i}/{N} rows, "
                             f"{offset:,}/{n_pairs:,} pairs")

        del X_proj

    return d_full, d_proj


def compute_pythagorean_check(X, V_all, n_check=JL_PYTH_SUBSAMPLE):
    """Pythagorean validation on random pairs — memory-efficient.

    Computes d_resid for only n_check random pairs (not all pairs).
    Uses float64 for numerical accuracy.
    """
    N = X.shape[0]
    n_total = N * (N - 1) // 2
    n_check = min(n_check, n_total)

    rng = np.random.RandomState(JL_RANDOM_SEED + 2)

    # Generate random (i, j) pairs with i < j
    check_pairs = set()
    while len(check_pairs) < n_check:
        batch_n = min(n_check * 3, 1000000)
        ia = rng.randint(0, N, size=batch_n)
        ib = rng.randint(0, N, size=batch_n)
        for a, b in zip(ia, ib):
            if a != b:
                check_pairs.add((min(int(a), int(b)), max(int(a), int(b))))
            if len(check_pairs) >= n_check:
                break
    check_pairs = np.array(list(check_pairs)[:n_check])

    V = V_all.astype(np.float64)
    diff = X[check_pairs[:, 0]].astype(np.float64) - \
        X[check_pairs[:, 1]].astype(np.float64)

    d_full = np.linalg.norm(diff, axis=1)
    proj = (diff @ V.T) @ V
    d_proj = np.linalg.norm(proj, axis=1)
    d_resid = np.linalg.norm(diff - proj, axis=1)

    pyth_residual = np.abs(d_full**2 - d_proj**2 - d_resid**2)
    pyth_max_error = float(np.max(
        pyth_residual / np.maximum(d_full**2, 1e-30)))

    return pyth_max_error


def _rank_array(a):
    """Assign ranks to array a, memory-efficient (chunked arange assignment).

    Returns rank array (float64, same length as a).
    Peak memory: len(a)*8 (order) + len(a)*8 (rank) + small chunk.
    """
    n = len(a)
    order = np.argsort(a)
    rank = np.empty(n, dtype=np.float64)
    chunk = 10_000_000
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        rank[order[s:e]] = np.arange(s, e, dtype=np.float64)
    del order
    return rank


def memory_efficient_spearman(a, b, logger=None):
    """Compute Spearman correlation with controlled peak memory.

    Ranks arrays one at a time and computes dot product in chunks.
    Peak memory: 2 × (N × 8 bytes for rank) + (N × 8 bytes for argsort temp).
    For N=7.5B: peak ~180GB (vs scipy's ~300GB+).
    """
    n = len(a)
    if logger:
        logger.info(f"    Spearman: ranking {n:,} values (pass 1/2)...")

    rank_a = _rank_array(a)

    if logger:
        logger.info(f"    Spearman: ranking {n:,} values (pass 2/2)...")

    rank_b = _rank_array(b)

    # Dot product in chunks (avoids creating N×float64 temporaries)
    if logger:
        logger.info("    Spearman: computing correlation from ranks...")
    chunk = 50_000_000
    sum_ab = 0.0
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sum_ab += np.dot(rank_a[s:e], rank_b[s:e])

    mean_r = (n - 1) / 2.0
    var_rank = n * (n - 1) * (2 * n - 1) / (6.0 * n) - mean_r**2
    cov = sum_ab / n - mean_r**2

    del rank_a, rank_b

    if var_rank > 0:
        return float(cov / var_rank)
    return 0.0


def _chunked_pearson(a, b, chunk_size=10_000_000):
    """Compute Pearson r in chunks to avoid 2×N float64 temporaries."""
    n = len(a)
    sx = sy = sxy = sx2 = sy2 = 0.0
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        ac = a[s:e].astype(np.float64)
        bc = b[s:e].astype(np.float64)
        sx += ac.sum()
        sy += bc.sum()
        sxy += np.dot(ac, bc)
        sx2 += np.dot(ac, ac)
        sy2 += np.dot(bc, bc)
    mx, my = sx / n, sy / n
    num = sxy / n - mx * my
    den = np.sqrt((sx2 / n - mx**2) * (sy2 / n - my**2))
    return float(num / den) if den > 0 else 0.0


def _chunked_rel_errors(a, b, chunk_size=50_000_000):
    """Compute mean and max relative error in chunks."""
    n = len(a)
    total = 0.0
    mx = 0.0
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        rel = np.abs(a[s:e] - b[s:e]) / np.maximum(a[s:e], 1e-12)
        total += rel.sum()
        mx = max(mx, float(rel.max()))
    return total / n, mx


def _chunked_dist_var_explained(d_full, d_proj, chunk_size=10_000_000):
    """Compute distance variance explained in chunks (float64 accumulation)."""
    n = len(d_full)
    # Two-pass: first compute means, then variance
    sum_fsq = sum_diff = 0.0
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        fsq = d_full[s:e].astype(np.float64) ** 2
        psq = d_proj[s:e].astype(np.float64) ** 2
        sum_fsq += fsq.sum()
        sum_diff += (fsq - psq).sum()
    mean_fsq = sum_fsq / n
    mean_diff = sum_diff / n

    var_fsq = var_diff = 0.0
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        fsq = d_full[s:e].astype(np.float64) ** 2
        psq = d_proj[s:e].astype(np.float64) ** 2
        var_fsq += ((fsq - mean_fsq) ** 2).sum()
        var_diff += ((fsq - psq - mean_diff) ** 2).sum()
    var_fsq /= n
    var_diff /= n

    if var_fsq > 0:
        return 1.0 - var_diff / var_fsq
    return 0.0


def compute_jl_metrics(d_full, d_proj, pyth_max_error, large_n=False,
                       logger=None):
    """Compute all JL quality metrics.

    d_full, d_proj: (n_pairs,) float32 distance arrays.
    pyth_max_error: pre-computed Pythagorean validation error (float64 subsample).
    large_n: if True, use chunked computations to control peak memory.
             Critical for billions of pairs where float64 temporaries
             would exceed available RAM.

    Returns dict with overall metrics.
    """
    n_pairs = len(d_full)

    if large_n:
        # Memory-efficient path: all computations chunked
        if logger:
            logger.info(f"    Large-N metrics: {n_pairs:,} pairs, "
                        "using chunked computations")
        pe_r = _chunked_pearson(d_full, d_proj)
        mean_rel_error, max_rel_error = _chunked_rel_errors(d_full, d_proj)
        dist_var_explained = _chunked_dist_var_explained(d_full, d_proj)
        sp_rho = memory_efficient_spearman(d_full, d_proj, logger=logger)
    else:
        # Standard path: scipy functions (fine for <50M pairs)
        sp_rho, _ = spearmanr(d_full, d_proj)
        sp_rho = float(sp_rho)
        pe_r, _ = pearsonr(d_full.astype(np.float64),
                           d_proj.astype(np.float64))
        rel_errors = np.abs(d_full - d_proj) / np.maximum(d_full, 1e-12)
        mean_rel_error = float(np.mean(rel_errors))
        max_rel_error = float(np.max(rel_errors))
        d_full_sq = d_full.astype(np.float64) ** 2
        d_proj_sq = d_proj.astype(np.float64) ** 2
        var_full_sq = np.var(d_full_sq)
        if var_full_sq > 0:
            dist_var_explained = float(
                1.0 - np.var(d_full_sq - d_proj_sq) / var_full_sq)
        else:
            dist_var_explained = 0.0

    return {
        "spearman_rho": float(sp_rho),
        "pearson_r": float(pe_r),
        "mean_rel_error": float(mean_rel_error),
        "max_rel_error": float(max_rel_error),
        "distance_var_explained": float(dist_var_explained),
        "pythagorean_max_error_f64": pyth_max_error,
        "n_pairs_total": n_pairs,
        "n_pairs_spearman": n_pairs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ATOMIC I/O
# ═══════════════════════════════════════════════════════════════════════════════

def atomic_json_write(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".json")
    os.close(fd)
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp_path, path)


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATORS
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase_f_slice(level, layer, pop, df_full, pop_df, paths, logger):
    """Run Phase F (principal angles) for one slice.

    Returns list of row dicts for the summary CSV.
    """
    slice_id = f"L{level}/layer{layer:02d}/{pop}"

    # Resume check
    out_dir = paths["pa_dir"] / f"L{level}" / f"layer_{layer:02d}" / pop
    meta_path = out_dir / "metadata.json"
    csv_path = out_dir / "pairwise_angles.csv"

    if meta_path.exists():
        try:
            with open(meta_path) as f:
                cached = json.load(f)
            if cached.get("computation_status") == "complete":
                logger.debug(f"  [F {slice_id}] cached — skipping")
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    return df.to_dict("records")
                return []
        except (json.JSONDecodeError, KeyError):
            pass

    t_start = time.time()
    concept_list = get_concept_registry(level, df_full, pop)
    rows, n_valid = compute_pairwise_angles_for_slice(
        level, layer, pop, concept_list, paths, logger)

    n_superposition = sum(1 for r in rows if r.get("superposition_flag"))
    elapsed = time.time() - t_start

    logger.info(f"  [F {slice_id}] {len(rows)} pairs from {n_valid} concepts, "
                f"{n_superposition} superposition flags, {elapsed:.1f}s")

    # Save
    if rows:
        angles_df = pd.DataFrame(rows)
        out_dir.mkdir(parents=True, exist_ok=True)
        angles_df.to_csv(csv_path, index=False)

    meta = {
        "level": level, "layer": layer, "population": pop,
        "n_concepts_total": len(concept_list),
        "n_concepts_with_basis": n_valid,
        "n_pairs": len(rows),
        "n_superposition_flags": n_superposition,
        "computation_time_s": elapsed,
        "computation_status": "complete",
    }
    atomic_json_write(meta, meta_path)

    return rows


def run_jl_slice(level, layer, pop, X, V_all, pop_df, paths, logger,
                 save_distances=False):
    """Run JL distance preservation check for one slice.

    X: (N_pop, D) float32 activations already sliced to population
    V_all: (k, D) union basis

    Returns summary row dict.
    """
    slice_id = f"L{level}/layer{layer:02d}/{pop}"

    # Resume check
    out_dir = paths["jl_dir"] / f"L{level}" / f"layer_{layer:02d}" / pop
    meta_path = out_dir / "metadata.json"

    if meta_path.exists():
        try:
            with open(meta_path) as f:
                cached = json.load(f)
            if cached.get("computation_status") == "complete":
                logger.debug(f"  [JL {slice_id}] cached — skipping")
                return cached.get("summary_row", None)
        except (json.JSONDecodeError, KeyError):
            pass

    t_start = time.time()
    N = X.shape[0]
    k = V_all.shape[0]
    d_residual = HIDDEN_DIM - k
    large_n = N > JL_LARGE_N_THRESHOLD

    # All pairs — no sampling
    pairs, sampling_meta = generate_all_pairs(N)
    n_pairs = sampling_meta["n_pairs"]
    logger.info(f"  [JL {slice_id}] N={N}, k={k}, "
                f"{sampling_meta['sampling_method']}, {n_pairs:,} pairs")

    # Pythagorean check (always done on 1000-pair float64 subsample)
    pyth_max_error = compute_pythagorean_check(X, V_all)

    # Distances — dispatch based on N
    if large_n:
        d_full, d_proj = compute_jl_distances_rowwise(X, V_all, logger)
    else:
        d_full, d_proj = compute_jl_distances(X, V_all, pairs)

    # Metrics
    jl_metrics = compute_jl_metrics(d_full, d_proj, pyth_max_error,
                                    large_n=large_n, logger=logger)

    elapsed = time.time() - t_start
    logger.info(
        f"  [JL {slice_id}] Spearman={jl_metrics['spearman_rho']:.4f}, "
        f"Pearson={jl_metrics['pearson_r']:.4f}, "
        f"mean_rel_err={jl_metrics['mean_rel_error']:.4f}, "
        f"dist_var_expl={jl_metrics['distance_var_explained']:.4f}, "
        f"pyth_err={jl_metrics['pythagorean_max_error_f64']:.2e}, "
        f"{elapsed:.1f}s")

    if jl_metrics["pythagorean_max_error_f64"] > 1e-6:
        logger.warning(f"  [JL {slice_id}] Pythagorean error "
                       f"{jl_metrics['pythagorean_max_error_f64']:.2e} > 1e-6")

    # Phase E cross-check
    var_explained_activations = np.nan
    pe_csv = paths["phase_e_summary"] / "phase_e_results.csv"
    if pe_csv.exists():
        try:
            pe_df = pd.read_csv(pe_csv)
            match = pe_df[(pe_df["level"] == level) &
                          (pe_df["layer"] == layer) &
                          (pe_df["population"] == pop)]
            if not match.empty:
                var_explained_activations = float(
                    match.iloc[0]["var_explained"])
        except Exception:
            pass

    summary_row = {
        "level": level, "layer": layer, "population": pop,
        "N": N, "k": k, "d_residual": d_residual,
        "var_explained_activations": var_explained_activations,
        "spearman_full": jl_metrics["spearman_rho"],
        "pearson_full": jl_metrics["pearson_r"],
        "mean_rel_error": jl_metrics["mean_rel_error"],
        "max_rel_error": jl_metrics["max_rel_error"],
        "distance_var_explained": jl_metrics["distance_var_explained"],
        "pythagorean_max_error_f64": jl_metrics["pythagorean_max_error_f64"],
        "n_pairs_total": jl_metrics["n_pairs_total"],
        "n_pairs_spearman": jl_metrics["n_pairs_spearman"],
        "sampling_method": sampling_meta["sampling_method"],
        "computation_time_s": elapsed,
    }

    # Per-stratum Spearman (only for small N where pair indices are available)
    for stratum in ["correct_correct", "correct_wrong", "wrong_wrong"]:
        summary_row[f"spearman_{stratum}"] = np.nan

    if not large_n and pairs is not None and "correct" in pop_df.columns:
        correct_vals = pop_df["correct"].values.astype(bool)
        ci = correct_vals[pairs[:, 0]]
        cj = correct_vals[pairs[:, 1]]
        strata_masks = {
            "correct_correct": ci & cj,
            "correct_wrong": ci ^ cj,
            "wrong_wrong": ~ci & ~cj,
        }
        for stratum, mask in strata_masks.items():
            if mask.sum() >= 10:
                s_rho, _ = spearmanr(d_full[mask], d_proj[mask])
                summary_row[f"spearman_{stratum}"] = float(s_rho)

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    jl_result_data = {
        **jl_metrics, **sampling_meta,
        "level": level, "layer": layer, "population": pop,
    }
    atomic_json_write(jl_result_data, out_dir / "jl_results.json")

    save_meta = {**summary_row, "computation_status": "complete",
                 "summary_row": summary_row}
    atomic_json_write(save_meta, meta_path)

    # Save distance subsample for scatter plots
    if save_distances:
        rng = np.random.RandomState(42)
        n_save = min(10000, len(d_full))
        idx = rng.choice(len(d_full), n_save, replace=False)
        dist_data = {"d_full": d_full[idx], "d_proj": d_proj[idx]}
        # Generate strata labels for saved subsample
        if pairs is not None and "correct" in pop_df.columns:
            correct_vals = pop_df["correct"].values.astype(bool)
            ci = correct_vals[pairs[idx, 0]]
            cj = correct_vals[pairs[idx, 1]]
            strata_arr = np.where(
                ci & cj, "correct_correct",
                np.where(~ci & ~cj, "wrong_wrong", "correct_wrong"))
            dist_data["strata"] = strata_arr
        np.savez_compressed(out_dir / "distances.npz", **dist_data)

    del d_full, d_proj

    return summary_row


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_superposition_heatmap(angles_df, level, layer, pop, plot_dir, logger):
    """N×N heatmap of angle_1 between all concept pairs."""
    sub = angles_df[(angles_df["level"] == level) &
                    (angles_df["layer"] == layer) &
                    (angles_df["population"] == pop)].copy()
    if sub.empty:
        return

    # Get unique concepts, ordered by tier then name
    concepts_a = sub[["concept_a", "tier_a"]].rename(
        columns={"concept_a": "c", "tier_a": "t"})
    concepts_b = sub[["concept_b", "tier_b"]].rename(
        columns={"concept_b": "c", "tier_b": "t"})
    all_concepts = pd.concat([concepts_a, concepts_b]).drop_duplicates("c")
    all_concepts = all_concepts.sort_values(["t", "c"])
    concept_order = all_concepts["c"].tolist()
    n = len(concept_order)
    idx_map = {c: i for i, c in enumerate(concept_order)}

    # Fill matrix
    mat = np.full((n, n), np.nan)
    np.fill_diagonal(mat, 0.0)
    for _, row in sub.iterrows():
        i = idx_map.get(row["concept_a"])
        j = idx_map.get(row["concept_b"])
        if i is not None and j is not None:
            mat[i, j] = row["angle_1"]
            mat[j, i] = row["angle_1"]

    fig, ax = plt.subplots(figsize=(max(10, n * 0.35), max(9, n * 0.32)))
    cmap = plt.cm.RdYlBu
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=90, aspect="equal")
    plt.colorbar(im, ax=ax, label="θ₁ (degrees)", shrink=0.8)

    ax.set_xticks(range(n))
    ax.set_xticklabels(concept_order, rotation=90, fontsize=6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(concept_order, fontsize=6)
    ax.set_title(f"Phase F: Min Principal Angle θ₁ — L{level}/layer{layer}/{pop}",
                 fontsize=11)

    # Mark superposition flags
    for _, row in sub[sub["superposition_flag"] == True].iterrows():
        i = idx_map.get(row["concept_a"])
        j = idx_map.get(row["concept_b"])
        if i is not None and j is not None:
            ax.plot(j, i, "k*", markersize=5)
            ax.plot(i, j, "k*", markersize=5)

    out = plot_dir / "superposition_heatmaps"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"L{level}_layer{layer:02d}_{pop}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_angle_distribution(angles_df, level, plot_dir, logger):
    """Histogram of angle_1 across all pairs at layer 16."""
    sub = angles_df[(angles_df["level"] == level) &
                    (angles_df["layer"] == 16) &
                    (angles_df["population"] == "all")]
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    vals = sub["angle_1"].dropna()
    ax.hist(vals, bins=50, edgecolor="black", alpha=0.7, color="steelblue")

    # Random baseline lines
    if "random_baseline_mean" in sub.columns:
        mean_bl = sub["random_baseline_mean"].median()
        p5_bl = sub["random_baseline_p5"].median()
        ax.axvline(mean_bl, color="red", ls="--", lw=1.5,
                   label=f"Random mean = {mean_bl:.1f}°")
        ax.axvline(p5_bl, color="orange", ls=":", lw=1.5,
                   label=f"Random p5 = {p5_bl:.1f}°")
        ax.legend(fontsize=9)

    ax.set_xlabel("θ₁ (minimum principal angle, degrees)")
    ax.set_ylabel("Count")
    ax.set_title(f"Phase F: θ₁ Distribution — L{level}/layer16/all "
                 f"(n={len(vals)})")

    out = plot_dir / "angle_distributions"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"L{level}_angle1_distribution.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cross_layer_superposition(angles_df, level, plot_dir, logger):
    """Track angle_1 across layers for top superposition pairs."""
    ref = angles_df[(angles_df["level"] == level) &
                    (angles_df["layer"] == 16) &
                    (angles_df["population"] == "all") &
                    (angles_df["superposition_flag"] == True)]
    if ref.empty:
        return

    # Top 5 by smallest angle_1
    top_pairs = ref.nsmallest(5, "angle_1")[["concept_a", "concept_b"]]

    for _, pair_row in top_pairs.iterrows():
        ca, cb = pair_row["concept_a"], pair_row["concept_b"]
        sub = angles_df[(angles_df["level"] == level) &
                        (angles_df["population"] == "all") &
                        (((angles_df["concept_a"] == ca) &
                          (angles_df["concept_b"] == cb)) |
                         ((angles_df["concept_a"] == cb) &
                          (angles_df["concept_b"] == ca)))]
        if sub.empty:
            continue

        sub = sub.sort_values("layer")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sub["layer"], sub["angle_1"], "o-", color="steelblue",
                markersize=6)
        if "random_baseline_p5" in sub.columns:
            ax.plot(sub["layer"], sub["random_baseline_p5"], "--",
                    color="red", alpha=0.6, label="Random p5")
            ax.legend(fontsize=9)
        ax.set_xlabel("Layer")
        ax.set_ylabel("θ₁ (degrees)")
        ax.set_title(f"Superposition: {ca} vs {cb} — L{level}/all")
        ax.set_xticks(LAYERS)

        out = plot_dir / "cross_layer_superposition"
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"L{level}_{ca}_vs_{cb}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_tier_boxplot(angles_df, level, pop, layer, plot_dir, logger):
    """Box plot of angle_1 grouped by tier pair."""
    sub = angles_df[(angles_df["level"] == level) &
                    (angles_df["layer"] == layer) &
                    (angles_df["population"] == pop)].copy()
    if sub.empty:
        return

    sub["tier_pair"] = ("T" + sub["tier_a"].astype(int).astype(str) +
                        "×T" + sub["tier_b"].astype(int).astype(str))
    # Normalize: make T1×T2 = T2×T1
    def normalize_tp(tp):
        parts = tp.split("×")
        return "×".join(sorted(parts))
    sub["tier_pair"] = sub["tier_pair"].apply(normalize_tp)

    groups = sorted(sub["tier_pair"].unique())
    data = [sub[sub["tier_pair"] == g]["angle_1"].dropna().values
            for g in groups]

    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 0.8), 5))
    bp = ax.boxplot(data, labels=groups, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.6)
    ax.set_xlabel("Tier Pair")
    ax.set_ylabel("θ₁ (degrees)")
    ax.set_title(f"Phase F: θ₁ by Tier Pair — L{level}/{pop}/layer{layer}")
    ax.tick_params(axis="x", rotation=45)

    out = plot_dir / "tier_boxplots"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"L{level}_{pop}_layer{layer:02d}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_jl_scatter(level, layer, pop, paths, plot_dir, logger):
    """d_proj vs d_full scatter with identity line."""
    dist_path = (paths["jl_dir"] / f"L{level}" / f"layer_{layer:02d}" /
                 pop / "distances.npz")
    if not dist_path.exists():
        return

    data = np.load(dist_path, allow_pickle=True)
    d_full = data["d_full"]
    d_proj = data["d_proj"]
    strata = data.get("strata", None)

    fig, ax = plt.subplots(figsize=(7, 7))

    if strata is not None:
        colors = {"correct_correct": "green", "correct_wrong": "orange",
                  "wrong_wrong": "red"}
        for s in np.unique(strata):
            mask = strata == s
            ax.scatter(d_full[mask], d_proj[mask], alpha=0.15, s=4,
                       color=colors.get(s, "gray"), label=s, rasterized=True)
        ax.legend(fontsize=8, markerscale=4)
    else:
        ax.scatter(d_full, d_proj, alpha=0.15, s=4, color="steelblue",
                   rasterized=True)

    lims = [0, max(d_full.max(), d_proj.max()) * 1.05]
    ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5, label="y=x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("d_full (full 4096D)")
    ax.set_ylabel("d_proj (projected)")
    ax.set_title(f"JL Distance Preservation — L{level}/layer{layer}/{pop}")
    ax.set_aspect("equal")

    out = plot_dir / "jl_scatter"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"L{level}_layer{layer:02d}_{pop}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_jl_spearman_trajectory(jl_df, level, plot_dir, logger):
    """Spearman across layers, separate lines per population."""
    sub = jl_df[jl_df["level"] == level].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"all": "steelblue", "correct": "green", "wrong": "red"}
    for pop in ["all", "correct", "wrong"]:
        pop_sub = sub[sub["population"] == pop].sort_values("layer")
        if pop_sub.empty:
            continue
        ax.plot(pop_sub["layer"], pop_sub["spearman_full"], "o-",
                color=colors.get(pop, "gray"), label=pop, markersize=5)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman ρ (d_full vs d_proj)")
    ax.set_title(f"JL Distance Preservation — L{level}")
    ax.set_xticks(LAYERS)
    ax.set_ylim(0.8, 1.0)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    out = plot_dir / "jl_trajectories"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"L{level}_spearman.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_variance_budget(jl_df, level, plot_dir, logger):
    """Stacked bar: concept variance + JL Spearman at layer 16."""
    sub = jl_df[(jl_df["level"] == level) &
                (jl_df["layer"] == 16) &
                (jl_df["population"] == "all")]
    if sub.empty:
        return

    row = sub.iloc[0]
    var_expl = row.get("var_explained_activations", np.nan)
    dist_var = row.get("distance_var_explained", np.nan)
    sp = row.get("spearman_full", np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5),
                             gridspec_kw={"width_ratios": [2, 1]})

    # Left: variance budget bar
    ax = axes[0]
    if not np.isnan(var_expl):
        bars = [var_expl, 1.0 - var_expl]
        labels = [f"Concepts ({var_expl:.1%})",
                  f"Residual ({1-var_expl:.1%})"]
        colors = ["steelblue", "lightcoral"]
        ax.bar(["Variance"], [bars[0]], color=colors[0], label=labels[0])
        ax.bar(["Variance"], [bars[1]], bottom=[bars[0]], color=colors[1],
               label=labels[1])
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Fraction of total variance")
        ax.legend(fontsize=9)
    ax.set_title(f"Variance Budget — L{level}/layer16/all")

    # Right: JL metrics
    ax = axes[1]
    metrics = {}
    if not np.isnan(sp):
        metrics["Spearman ρ"] = sp
    if not np.isnan(dist_var):
        metrics["Dist Var Expl"] = dist_var
    if metrics:
        bars = ax.barh(list(metrics.keys()), list(metrics.values()),
                       color="steelblue", alpha=0.7)
        ax.set_xlim(0.7, 1.0)
        for bar, val in zip(bars, metrics.values()):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{val:.4f}", va="center", fontsize=9)
    ax.set_title("JL Metrics")

    fig.tight_layout()
    out = plot_dir / "variance_budget"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"L{level}_layer16.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_plots(angles_df, jl_df, paths, logger):
    """Generate all 7 plot types."""
    plot_dir = paths["phase_f_plots"]
    plot_dir.mkdir(parents=True, exist_ok=True)
    n_plots = 0

    logger.info("Generating plots...")

    # 1. Superposition heatmaps
    for level in PLOT_LEVELS:
        for layer in PLOT_LAYERS:
            plot_superposition_heatmap(angles_df, level, layer, "all",
                                      plot_dir, logger)
            n_plots += 1

    # 2. Angle distributions (all levels at layer 16)
    for level in LEVELS:
        if level == 1:
            continue  # L1 too few concepts for meaningful histogram
        plot_angle_distribution(angles_df, level, plot_dir, logger)
        n_plots += 1

    # 3. Cross-layer superposition trajectories
    for level in PLOT_LEVELS:
        plot_cross_layer_superposition(angles_df, level, plot_dir, logger)
        n_plots += 5  # up to 5 per level

    # 4. Tier boxplots
    for level in PLOT_LEVELS:
        for layer in PLOT_LAYERS:
            plot_tier_boxplot(angles_df, level, "all", layer,
                              plot_dir, logger)
            n_plots += 1

    # 5. JL scatter
    for level in PLOT_LEVELS:
        for layer in PLOT_LAYERS:
            plot_jl_scatter(level, layer, "all", paths, plot_dir, logger)
            n_plots += 1

    # 6. JL Spearman trajectories
    if not jl_df.empty:
        for level in LEVELS_JL:
            plot_jl_spearman_trajectory(jl_df, level, plot_dir, logger)
            n_plots += 1

    # 7. Variance budget
    if not jl_df.empty:
        for level in PLOT_LEVELS:
            plot_variance_budget(jl_df, level, plot_dir, logger)
            n_plots += 1

    logger.info(f"  Generated up to {n_plots} plots in {plot_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_phase_f_summaries(all_angle_rows, jl_results, paths, logger):
    """Generate all 4 summary CSVs."""
    summary_dir = paths["phase_f_summary"]
    summary_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full principal angles CSV
    if all_angle_rows:
        angles_df = pd.DataFrame(all_angle_rows)
        angles_df.to_csv(summary_dir / "phase_f_principal_angles.csv",
                         index=False)
        logger.info(f"  Saved phase_f_principal_angles.csv "
                    f"({len(angles_df)} rows)")

        # 2. Superposition summary
        flagged = angles_df[angles_df["superposition_flag"] == True].copy()
        if not flagged.empty:
            # Count angles below 45° for each flagged pair
            angle_cols = [c for c in angles_df.columns
                          if c.startswith("angle_") and c[6:].isdigit()]
            if angle_cols:
                below_45 = (flagged[angle_cols] < 45.0).sum(axis=1)
                flagged["n_angles_below_45deg"] = below_45
            flagged.to_csv(summary_dir / "superposition_summary.csv",
                           index=False)
            logger.info(f"  Saved superposition_summary.csv "
                        f"({len(flagged)} rows)")
        else:
            pd.DataFrame().to_csv(
                summary_dir / "superposition_summary.csv", index=False)
            logger.info("  No superposition flags — empty summary")

        # 3. Redundancy decomposition
        # Pairs where angle_1 is below the random p5 threshold
        if "random_baseline_p5" in angles_df.columns:
            redundant = angles_df[
                angles_df["angle_1"] < angles_df["random_baseline_p5"]
            ].copy()
            if not redundant.empty:
                # Count how many angles are below the random p5
                if angle_cols:
                    n_below = []
                    for _, row in redundant.iterrows():
                        p5 = row["random_baseline_p5"]
                        below = sum(1 for c in angle_cols
                                    if not np.isnan(row.get(c, np.nan))
                                    and row[c] < p5)
                        n_below.append(below)
                    redundant["n_angles_below_random_p5"] = n_below
                    redundant["estimated_shared_dims"] = n_below
            else:
                redundant = pd.DataFrame()
            redundant.to_csv(
                summary_dir / "redundancy_decomposition.csv", index=False)
            logger.info(f"  Saved redundancy_decomposition.csv "
                        f"({len(redundant)} rows)")
    else:
        angles_df = pd.DataFrame()

    # 4. JL distance preservation CSV
    if jl_results:
        jl_df = pd.DataFrame(jl_results)
        jl_df.to_csv(summary_dir / "jl_distance_preservation.csv",
                     index=False)
        logger.info(f"  Saved jl_distance_preservation.csv "
                    f"({len(jl_df)} rows)")
    else:
        jl_df = pd.DataFrame()

    return angles_df, jl_df


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase F/JL — Between-Concept Angles & "
                    "JL Distance Preservation")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--pilot", action="store_true",
                        help="Smoke test: L3, layer 16, all population")
    parser.add_argument("--level", type=int, nargs="*",
                        help="Specific levels (default: 1-5 for F, 2-5 for JL)")
    parser.add_argument("--layer", type=int, nargs="*",
                        help="Specific layers (default: all 9)")
    parser.add_argument("--population", nargs="*",
                        help="Specific populations (default: all viable)")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--plots-only", action="store_true",
                        help="Regenerate plots from saved CSVs")
    parser.add_argument("--phase-f-only", action="store_true",
                        help="Skip JL, run Phase F only (no activation loading)")
    parser.add_argument("--jl-only", action="store_true",
                        help="Skip Phase F, run JL only")
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    cfg = load_config(args.config)
    paths = derive_paths(cfg)
    logger = setup_logging(paths["workspace"])

    # Banner
    logger.info("=" * 70)
    logger.info("Phase F/JL — Between-Concept Angles & JL Distance Check")
    logger.info("=" * 70)

    # Scope
    if args.pilot:
        levels_f = [3]
        levels_jl = [3]
        layers = [16]
        populations_filter = ["all"]
        logger.info("PILOT MODE: L3, layer 16, all population")
    else:
        levels_f = args.level if args.level else LEVELS
        levels_jl = [l for l in (args.level if args.level else LEVELS_JL)
                     if l in LEVELS_JL]
        layers = args.layer if args.layer else LAYERS
        populations_filter = args.population if args.population else None

    logger.info(f"Phase F levels: {levels_f}")
    logger.info(f"JL levels: {levels_jl}")
    logger.info(f"Layers: {layers}")
    logger.info(f"Populations: {populations_filter or 'all viable'}")
    logger.info(f"GPU (CuPy): "
                f"{'available' if _CUPY_AVAILABLE else 'CPU fallback'}")
    logger.info(f"Phase F only: {args.phase_f_only}")
    logger.info(f"JL only: {args.jl_only}")

    # Pre-flight checks
    missing = []
    all_levels = set(levels_f) | set(levels_jl)
    for level in all_levels:
        pkl_path = paths["coloring_dir"] / f"L{level}_coloring.pkl"
        if not pkl_path.exists():
            missing.append(str(pkl_path))
    if not args.phase_f_only:
        for level in levels_jl:
            for layer in layers:
                resid_path = (paths["residualized_dir"] /
                              f"level{level}_layer{layer}.npy")
                if not resid_path.exists():
                    missing.append(str(resid_path))

    if missing:
        logger.error(f"Missing {len(missing)} input files. First 5:")
        for p in missing[:5]:
            logger.error(f"  {p}")
        logger.error("Run Phase C/D/E first.")
        return

    # Phase E data integrity spot-check
    if not args.phase_f_only:
        spot_check_path = (paths["union_bases_dir"] / "L3" /
                           "layer_16" / "all" / "union_basis.npy")
        if spot_check_path.exists():
            ub = np.load(spot_check_path)
            if ub.shape[0] == 0 or ub.shape[1] != HIDDEN_DIM:
                logger.error(f"Phase E union basis has bad shape: {ub.shape}")
                return
            logger.info(f"Phase E spot-check OK: {spot_check_path.name} "
                        f"shape {ub.shape}")
            del ub
        else:
            logger.warning(f"Cannot spot-check Phase E: {spot_check_path}")

    # Phase D metadata count
    phase_d_meta_count = sum(
        1 for _ in paths["phase_d_subspaces"].rglob("metadata.json"))
    logger.info(f"Phase D metadata files: {phase_d_meta_count}")

    # Create output directories
    for key in ["phase_f_data", "pa_dir", "jl_dir",
                "phase_f_summary", "phase_f_plots"]:
        paths[key].mkdir(parents=True, exist_ok=True)

    # --plots-only mode
    if args.plots_only:
        logger.info("PLOTS-ONLY MODE: loading from saved CSVs")
        angles_csv = paths["phase_f_summary"] / "phase_f_principal_angles.csv"
        jl_csv = paths["phase_f_summary"] / "jl_distance_preservation.csv"
        angles_df = pd.read_csv(angles_csv) if angles_csv.exists() \
            else pd.DataFrame()
        jl_df = pd.read_csv(jl_csv) if jl_csv.exists() else pd.DataFrame()
        logger.info(f"  Loaded {len(angles_df)} angle rows, "
                    f"{len(jl_df)} JL rows")
        generate_plots(angles_df, jl_df, paths, logger)
        elapsed = time.time() - t0
        logger.info(f"Plots-only complete: {elapsed:.0f}s")
        return

    # ── Main loop ─────────────────────────────────────────────────────────
    all_angle_rows = []
    jl_results = []
    n_f_slices = 0
    n_jl_slices = 0

    # Determine which levels to iterate
    if args.jl_only:
        iter_levels = sorted(set(levels_jl))
    elif args.phase_f_only:
        iter_levels = sorted(set(levels_f))
    else:
        iter_levels = sorted(set(levels_f) | set(levels_jl))

    for level in iter_levels:
        df_full = load_coloring_df(level, paths["coloring_dir"])
        pops = get_populations(df_full)
        # L1 is 100% correct — "correct" pop is identical to "all", skip it
        if level == 1 and "correct" in pops:
            del pops["correct"]
        logger.info(f"L{level}: N={len(df_full)}, populations={list(pops.keys())}")

        for layer in tqdm(layers, desc=f"L{level}", leave=False):

            # Load activations ONCE per (level, layer) — only for JL
            X_resid_full = None
            do_jl = (not args.phase_f_only and level in levels_jl)
            if do_jl:
                X_resid_full = load_residualized(
                    level, layer, paths["residualized_dir"])

            for pop_name, pop_df in pops.items():
                if populations_filter and pop_name not in populations_filter:
                    continue

                # Phase F
                if not args.jl_only and level in levels_f:
                    rows = run_phase_f_slice(
                        level, layer, pop_name, df_full, pop_df,
                        paths, logger)
                    for r in rows:
                        r["level"] = level
                        r["layer"] = layer
                        r["population"] = pop_name
                    all_angle_rows.extend(rows)
                    n_f_slices += 1

                # JL
                if do_jl and X_resid_full is not None:
                    pop_indices = pop_df.index.values
                    X = X_resid_full[pop_indices]

                    # Load union basis for this population
                    ub_path = (paths["union_bases_dir"] / f"L{level}" /
                               f"layer_{layer:02d}" / pop_name /
                               "union_basis.npy")
                    if ub_path.exists():
                        V_all = np.load(ub_path)
                        # Save distances for scatter plots at select slices
                        save_dist = (level in PLOT_LEVELS and
                                     layer in PLOT_LAYERS and
                                     pop_name == "all")
                        jl_row = run_jl_slice(
                            level, layer, pop_name, X, V_all, pop_df,
                            paths, logger, save_distances=save_dist)
                        if jl_row:
                            jl_results.append(jl_row)
                            n_jl_slices += 1
                    else:
                        logger.warning(
                            f"  [JL] Missing union basis: {ub_path}")

            if X_resid_full is not None:
                del X_resid_full

    logger.info("")
    logger.info(f"Computation complete: {n_f_slices} Phase F slices, "
                f"{n_jl_slices} JL slices")

    # ── Summaries ─────────────────────────────────────────────────────────
    logger.info("")
    logger.info("Generating summary CSVs...")
    angles_df, jl_df = generate_phase_f_summaries(
        all_angle_rows, jl_results, paths, logger)

    # ── Plots ─────────────────────────────────────────────────────────────
    if not args.skip_plots:
        logger.info("")
        generate_plots(angles_df, jl_df, paths, logger)

    # ── Final summary ─────────────────────────────────────────────────────
    elapsed = time.time() - t0
    total_f_time = sum(r.get("computation_time_s", 0)
                       for r in all_angle_rows[:1])  # metadata has it
    total_jl_time = sum(r.get("computation_time_s", 0) for r in jl_results)

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"Phase F/JL complete: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    logger.info(f"  Phase F: {n_f_slices} slices, "
                f"{len(all_angle_rows)} total pairs")
    logger.info(f"  JL: {n_jl_slices} slices, "
                f"total compute {total_jl_time:.0f}s")
    if jl_results:
        sp_vals = [r["spearman_full"] for r in jl_results
                   if not np.isnan(r.get("spearman_full", np.nan))]
        if sp_vals:
            logger.info(f"  JL Spearman range: "
                        f"{min(sp_vals):.4f} — {max(sp_vals):.4f}")
    if all_angle_rows:
        n_flags = sum(1 for r in all_angle_rows
                      if r.get("superposition_flag"))
        logger.info(f"  Superposition flags: {n_flags}/{len(all_angle_rows)}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
