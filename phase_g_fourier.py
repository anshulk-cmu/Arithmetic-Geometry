#!/usr/bin/env python3
"""
Phase G: Fourier Screening for Periodic Structure in Concept Subspaces.

Tests whether digit-like concepts are arranged on circles or helices
(periodic structure) within their linear subspaces found in Phases C/D.
Uses explicit DFT at specified periods, permutation null for significance,
and separate circle/helix detection with FDR correction.
"""

import sys
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from pathlib import Path
import json
import argparse
import logging
from logging.handlers import RotatingFileHandler
import math
import time
import pickle
import yaml
from collections import defaultdict
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle as MplCircle

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

LEVELS = [2, 3, 4, 5]
LAYERS = [4, 6, 8, 12, 16, 20, 24, 28, 31]
MIDDLE_LAYERS = [8, 12, 16, 20, 24]
POPULATIONS = ["all", "correct", "wrong"]

N_PERMUTATIONS = 1000
PERM_ALPHA = 0.01
DIGIT_PERIOD = 10
FDR_THRESHOLD = 0.05
LINEAR_P_THRESHOLD = 0.01
COORD_P_THRESHOLD = 0.01
ZERO_POWER_THRESHOLD = 1e-12
SPOT_CHECK_ATOL = 1e-4
SPOT_CHECK_RTOL = 1e-3
MIN_CARRY_MOD10_VALUES = 6
MIN_CARRY_RAW_VALUES = 6
MIN_POPULATION = 30

# Digit concepts available at each level (from verified coloring DFs)
DIGIT_CONCEPTS_BY_LEVEL = {
    2: ["a_units", "a_tens", "b_units",
        "ans_digit_0_msf", "ans_digit_1_msf", "ans_digit_2_msf"],
    3: ["a_units", "a_tens", "b_units", "b_tens",
        "ans_digit_0_msf", "ans_digit_1_msf", "ans_digit_2_msf", "ans_digit_3_msf"],
    4: ["a_units", "a_tens", "a_hundreds", "b_units", "b_tens",
        "ans_digit_0_msf", "ans_digit_1_msf", "ans_digit_2_msf",
        "ans_digit_3_msf", "ans_digit_4_msf"],
    5: ["a_units", "a_tens", "a_hundreds", "b_units", "b_tens", "b_hundreds",
        "ans_digit_0_msf", "ans_digit_1_msf", "ans_digit_2_msf",
        "ans_digit_3_msf", "ans_digit_4_msf", "ans_digit_5_msf"],
}

CARRY_CONCEPTS_BY_LEVEL = {
    2: ["carry_0"],
    3: ["carry_0", "carry_1"],
    4: ["carry_0", "carry_1", "carry_2"],
    5: ["carry_0", "carry_1", "carry_2", "carry_3", "carry_4"],
}

PERIOD_SPECS = {"digit", "carry_binned", "carry_mod10", "carry_raw"}

# ═══════════════════════════════════════════════════════════════════════════════
# DECISION RULE (pre-registered, printed in summary)
# ═══════════════════════════════════════════════════════════════════════════════

DECISION_RULE = """\
Periodic structure is confirmed for a concept class (input digits, answer digits,
carries) if >=3 concept-layer cells are significant after FDR correction (q < 0.05),
spanning >=2 distinct concepts and >=2 distinct layers in {8, 12, 16, 20, 24}
(5 middle layers), in the `all` population. A concept-layer cell is significant if
geometry_detected != "none" in EITHER basis.

Geometry classification (hierarchical):
  - geometry_detected = "helix"  if helix_detected = True
  - geometry_detected = "circle" if circle_detected = True and helix_detected = False
  - geometry_detected = "none"   otherwise

Circle detection:  p_two_axis < 0.01 (pre-FDR) AND both best coords have p_coord < 0.01.
Helix detection:   p_helix < 0.01 (pre-FDR) AND both best Fourier coords have p_coord < 0.01
                   AND helix_linear_coord has p_linear < 0.01.

Phase C vs Phase D: Significance in EITHER basis suffices. The agreement column records
which basis(es) passed.

Correct vs wrong population comparisons are EXPLORATORY (no pre-registered threshold).
"""

# ═══════════════════════════════════════════════════════════════════════════════
# NUMBER-TOKEN FRAMING (Fix 10: honest framing, not "stronger test")
# ═══════════════════════════════════════════════════════════════════════════════

NUMBER_TOKEN_FRAMING = """\
We probe at the = token because our broader subspace pipeline (Phases A-F) uses
activations at that position, and we want Phase G to sit in the same frame as the
rest of the paper. The number-token probe provides a literature-grounded comparison
point. A positive at = is a stronger claim than K&T made; a null at = combined with
a positive at the number token matches K&T's finding and does not contradict it.
The = probe and the number-token probe test different hypotheses: whether Fourier
features exist at the computation position vs. whether they exist at the input position.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# RUNTIME ESTIMATE (Fix 7: realistic helix overhead)
# ═══════════════════════════════════════════════════════════════════════════════

RUNTIME_ESTIMATE = """\
Runtime estimate for main job:
  ~3,348 analyses x ~4s avg (centroids + Fourier)  = ~3.7 hours
  + helix computation (~30-50% overhead)            = ~1.5 hours
  + I/O overhead                                    = ~0.5 hours
  + plotting                                        = ~0.3 hours
  Total                                             ~ 6.5 hours
Number-token probe (separate job): ~4 hours additional.
SLURM allocation: 10 hours (safety margin).
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


def load_config(path):
    """Load YAML configuration file."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def derive_paths(cfg):
    """Derive all filesystem paths from config."""
    workspace = Path(cfg["paths"]["workspace"])
    data_root = Path(cfg["paths"]["data_root"])
    paths = {
        "workspace": workspace,
        "data_root": data_root,
        "coloring_dfs": data_root / "phase_a" / "coloring_dfs",
        "phase_c_subspaces": data_root / "phase_c" / "subspaces",
        "phase_c_residualized": data_root / "phase_c" / "residualized",
        "phase_d_subspaces": data_root / "phase_d" / "subspaces",
        "activations": data_root / "activations",
        "phase_g_output": data_root / "phase_g" / "fourier",
        "phase_g_summary": data_root / "phase_g" / "summary",
        "plots": workspace / "plots" / "phase_g",
        "logs": workspace / "logs",
    }
    return paths


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════


def setup_logging(workspace):
    """Configure rotating file + console logging for Phase G."""
    log_dir = Path(workspace) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("phase_g")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S"
    )

    fh = RotatingFileHandler(
        log_dir / "phase_g_fourier.log", maxBytes=10_000_000, backupCount=3
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════


def load_coloring_df(level, paths, logger):
    """Load the coloring DataFrame for a level."""
    pkl_path = paths["coloring_dfs"] / f"L{level}_coloring.pkl"
    logger.info("Loading coloring DataFrame: %s", pkl_path)
    t0 = time.time()
    df = pd.read_pickle(pkl_path)
    logger.debug(
        "  Loaded L%d coloring: %d rows, %d columns in %.2fs",
        level, len(df), len(df.columns), time.time() - t0,
    )
    return df


def get_population_mask(df, pop_name, logger):
    """Return boolean mask for the given population."""
    if pop_name == "all":
        mask = np.ones(len(df), dtype=bool)
    elif pop_name == "correct":
        mask = df["correct"].values.astype(bool)
    elif pop_name == "wrong":
        mask = ~df["correct"].values.astype(bool)
    else:
        raise ValueError(f"Unknown population: {pop_name}")
    n = mask.sum()
    logger.debug("  Population '%s': N=%d", pop_name, n)
    return mask


def load_residualized(level, layer, paths, logger):
    """Load residualized activations for a (level, layer) pair."""
    act_path = paths["phase_c_residualized"] / f"level{level}_layer{layer}.npy"
    logger.debug("Loading residualized activations: %s", act_path)
    t0 = time.time()
    acts = np.load(act_path)
    logger.debug(
        "  Loaded residualized: shape=%s, dtype=%s in %.2fs",
        acts.shape, acts.dtype, time.time() - t0,
    )
    return acts


def load_raw_activations(level, layer, paths, logger):
    """Load raw (non-residualized) activations for a (level, layer) pair."""
    act_path = paths["activations"] / f"level{level}_layer{layer}.npy"
    logger.debug("Loading raw activations: %s", act_path)
    t0 = time.time()
    acts = np.load(act_path)
    logger.debug(
        "  Loaded raw activations: shape=%s, dtype=%s in %.2fs",
        acts.shape, acts.dtype, time.time() - t0,
    )
    return acts


def load_phase_c_projected(level, layer, pop, concept, paths, logger):
    """Load Phase C projected activations. Returns None if not found."""
    proj_path = (
        paths["phase_c_subspaces"]
        / f"L{level}"
        / f"layer_{layer:02d}"
        / pop
        / concept
        / "projected_all.npy"
    )
    if not proj_path.exists():
        logger.debug("  Phase C projected not found: %s", proj_path)
        return None
    logger.debug("  Loading Phase C projected: %s", proj_path)
    data = np.load(proj_path)
    logger.debug("    Shape: %s", data.shape)
    return data


def load_phase_c_metadata(level, layer, pop, concept, paths, logger):
    """Load Phase C metadata JSON. Returns None if not found."""
    meta_path = (
        paths["phase_c_subspaces"]
        / f"L{level}"
        / f"layer_{layer:02d}"
        / pop
        / concept
        / "metadata.json"
    )
    if not meta_path.exists():
        logger.debug("  Phase C metadata not found: %s", meta_path)
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    logger.debug(
        "  Phase C metadata for %s: n_groups=%s, dim_consensus=%s, dim_perm=%s",
        concept, meta.get("n_groups"), meta.get("dim_consensus"), meta.get("dim_perm"),
    )
    return meta


def load_phase_c_eigenvalues(level, layer, pop, concept, paths, logger):
    """Load Phase C eigenvalues. Returns None if not found."""
    eig_path = (
        paths["phase_c_subspaces"]
        / f"L{level}"
        / f"layer_{layer:02d}"
        / pop
        / concept
        / "eigenvalues.npy"
    )
    if not eig_path.exists():
        logger.debug("  Phase C eigenvalues not found: %s", eig_path)
        return None
    eig = np.load(eig_path)
    logger.debug("  Phase C eigenvalues: shape=%s", eig.shape)
    return eig


def load_phase_d_merged_basis(level, layer, pop, concept, paths, logger):
    """Load Phase D merged basis. Returns None if not found."""
    basis_path = (
        paths["phase_d_subspaces"]
        / f"L{level}"
        / f"layer_{layer:02d}"
        / pop
        / concept
        / "merged_basis.npy"
    )
    if not basis_path.exists():
        logger.debug("  Phase D merged basis not found: %s", basis_path)
        return None
    basis = np.load(basis_path)
    logger.debug("  Phase D merged basis: shape=%s", basis.shape)
    return basis


def count_phase_d_bases(paths, logger):
    """Walk filesystem to count all Phase D merged_basis.npy files."""
    base = paths["phase_d_subspaces"]
    count = 0
    per_level = defaultdict(int)
    for f in base.rglob("merged_basis.npy"):
        count += 1
        parts = f.relative_to(base).parts
        if parts:
            per_level[parts[0]] += 1
    logger.info(
        "Phase D merged bases found: %d total (%s)",
        count, dict(per_level),
    )
    return count


# ═══════════════════════════════════════════════════════════════════════════════
# CONCEPT REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════


def _get_phase_c_group_labels(level, layer, pop, concept, paths, logger):
    """Read Phase C metadata to get the group labels used during Phase C."""
    meta = load_phase_c_metadata(level, layer, pop, concept, paths, logger)
    if meta is None:
        return None, None
    if meta.get("dim_perm", 0) == 0:
        logger.debug("  Concept %s has dim_perm=0 at L%d/layer%d/%s, skipping",
                      concept, level, layer, pop)
        return None, None
    group_sizes = meta["group_sizes"]
    groups = sorted(int(k) for k in group_sizes.keys())
    n_groups = len(groups)
    logger.debug("  Phase C groups for %s: %d groups, labels=%s",
                 concept, n_groups, groups)
    return groups, meta


def compute_labels_and_linear_values(raw_col, phase_c_groups):
    """
    Given raw column values and Phase C group labels, compute:
    - labels: array mapping each sample to its group label
    - unique_groups: sorted array of group labels
    - v_linear: linear axis values for each group (Fix 2: mean raw value per bin)

    For non-tail groups: v_linear = group label (the integer itself).
    For tail bin: v_linear = mean of raw values in that bin FROM THIS POPULATION.
    (Fix #2 from review: never mix population slices — compute from raw_col only.)
    """
    unique_groups = np.array(sorted(phase_c_groups))
    max_group = unique_groups.max()
    max_raw = int(raw_col.max()) if len(raw_col) > 0 else max_group

    if max_raw > max_group:
        # Tail binning: values > max_group are clipped to max_group
        labels = np.minimum(raw_col.astype(int), max_group)
        v_linear = unique_groups.astype(float).copy()
        # For the tail bin, use mean of the raw values in the bin
        # from THIS population only (not cross-population)
        tail_mask = raw_col >= max_group
        if tail_mask.sum() > 0:
            v_linear[-1] = float(raw_col[tail_mask].mean())
    else:
        labels = raw_col.astype(int)
        v_linear = unique_groups.astype(float).copy()

    return labels, unique_groups, v_linear


def get_fourier_concepts(level, coloring_df, logger):
    """
    Build the list of Fourier-eligible concepts for a given level.

    Reads actual unique values from the coloring DataFrame. Returns a list of
    concept info dicts, each containing period_specs to test.
    """
    concepts = []

    # --- Digit concepts ---
    for name in DIGIT_CONCEPTS_BY_LEVEL.get(level, []):
        if name not in coloring_df.columns:
            logger.debug("  Digit concept '%s' not in L%d coloring DF, skipping", name, level)
            continue
        raw_vals = coloring_df[name].dropna().astype(int)
        unique_vals = sorted(raw_vals.unique())
        n_groups = len(unique_vals)
        if n_groups < 3:
            logger.debug("  Digit concept '%s' at L%d has only %d groups, skipping",
                         name, level, n_groups)
            continue
        concepts.append({
            "name": name,
            "column": name,
            "tier": "A",
            "is_carry": False,
            "period_specs": [
                {
                    "period": DIGIT_PERIOD,
                    "spec_name": "digit",
                    "values": np.array(unique_vals),
                    "v_linear": np.array(unique_vals, dtype=float),
                }
            ],
        })
        logger.debug(
            "  Registered digit concept '%s': values=%s, n_groups=%d",
            name, unique_vals, n_groups,
        )

    # --- Carry concepts ---
    for name in CARRY_CONCEPTS_BY_LEVEL.get(level, []):
        if name not in coloring_df.columns:
            logger.debug("  Carry concept '%s' not in L%d coloring DF, skipping", name, level)
            continue
        raw_vals = coloring_df[name].dropna().astype(int)
        unique_raw = sorted(raw_vals.unique())
        n_raw = len(unique_raw)
        if n_raw < 3:
            logger.debug("  Carry concept '%s' at L%d has only %d raw values, skipping",
                         name, level, n_raw)
            continue

        period_specs = []

        # carry_binned: period = n_groups (after Phase C binning, determined at runtime)
        # Values and n_groups will be set per (layer, pop) from Phase C metadata.
        # We store a placeholder here; the actual values come from Phase C metadata.
        period_specs.append({
            "period": None,  # set dynamically from Phase C metadata
            "spec_name": "carry_binned",
            "values": None,
            "v_linear": None,
            "needs_phase_c_groups": True,
        })

        # carry_mod10: use only values 0-9 that exist as individual groups
        mod10_vals = [v for v in unique_raw if 0 <= v <= 9]
        if len(mod10_vals) >= MIN_CARRY_MOD10_VALUES:
            period_specs.append({
                "period": 10,
                "spec_name": "carry_mod10",
                "values": np.array(mod10_vals),
                "v_linear": np.array(mod10_vals, dtype=float),
            })
            logger.debug(
                "  Carry '%s' carry_mod10: values=%s (n=%d)",
                name, mod10_vals, len(mod10_vals),
            )
        else:
            logger.debug(
                "  Carry '%s' carry_mod10 skipped: only %d values in 0-9",
                name, len(mod10_vals),
            )

        # carry_raw: use un-binned values, only if n_raw >= MIN and different from binned
        if n_raw >= MIN_CARRY_RAW_VALUES:
            period_specs.append({
                "period": n_raw,
                "spec_name": "carry_raw",
                "values": np.array(unique_raw),
                "v_linear": np.array(unique_raw, dtype=float),
            })
            logger.debug(
                "  Carry '%s' carry_raw: period=%d, values=%s",
                name, n_raw, unique_raw,
            )

        concepts.append({
            "name": name,
            "column": name,
            "tier": "B",
            "is_carry": True,
            "period_specs": period_specs,
        })
        logger.debug(
            "  Registered carry concept '%s': %d period_specs",
            name, len(period_specs),
        )

    logger.info(
        "  L%d concept registry: %d digit concepts, %d carry concepts",
        level,
        sum(1 for c in concepts if not c["is_carry"]),
        sum(1 for c in concepts if c["is_carry"]),
    )
    return concepts


def resolve_carry_binned_spec(concept_info, phase_c_groups, raw_col, logger):
    """
    Resolve the carry_binned period_spec using Phase C group labels.

    Fix #2: Uses only population-local raw_col for tail-bin mean computation,
    never cross-population data.

    Returns an updated period_spec dict, or None if n_groups < 3.
    """
    unique_groups = np.array(sorted(phase_c_groups))
    n_groups = len(unique_groups)
    if n_groups < 3:
        logger.debug("  carry_binned for '%s': only %d groups, skipping",
                      concept_info["name"], n_groups)
        return None

    _, _, v_linear = compute_labels_and_linear_values(raw_col, phase_c_groups)

    spec = {
        "period": n_groups,
        "spec_name": "carry_binned",
        "values": unique_groups,
        "v_linear": v_linear,
    }
    logger.debug(
        "  Resolved carry_binned for '%s': period=%d, values=%s, v_linear=%s",
        concept_info["name"], n_groups, unique_groups.tolist(),
        [f"{v:.2f}" for v in v_linear],
    )
    return spec


# ═══════════════════════════════════════════════════════════════════════════════
# CORE FOURIER MATH
# ═══════════════════════════════════════════════════════════════════════════════


def compute_freq_range(period):
    """
    Compute the number of Fourier frequencies K for a given period P.

    Odd P:  K = (P-1) // 2.  All frequencies have 2 DOF (cos + sin).
    Even P: K = P // 2.       Includes Nyquist (1 DOF, rescaled by 2x).
    """
    if period % 2 == 0:
        return period // 2
    else:
        return (period - 1) // 2


def is_nyquist(k, period):
    """True if frequency k is the Nyquist frequency for even-period P."""
    return period % 2 == 0 and k == period // 2


def fourier_single_coordinate(signal, values, period):
    """
    Compute Fourier power spectrum for a single coordinate signal.

    Args:
        signal: (m,) array of centroid values for one coordinate.
        values: (m,) array of integer concept values.
        period: integer period P.

    Returns dict with:
        per_freq_power: (K,) power at each frequency (Nyquist rescaled).
        fcr_top1: fraction of power in the dominant frequency.
        fcr_top2: fraction of power in top-2 frequencies.
        dominant_freq: 1-indexed dominant frequency.
        total_power: sum of all frequency powers.
    """
    K = compute_freq_range(period)
    powers = np.zeros(K)

    for ki in range(K):
        k = ki + 1  # 1-indexed frequencies
        angles = 2.0 * np.pi * k * values.astype(float) / period
        a_k = np.sum(signal * np.cos(angles))
        b_k = np.sum(signal * np.sin(angles))

        if is_nyquist(k, period):
            # Nyquist: sin component is identically zero, rescale by 2x
            powers[ki] = 2.0 * a_k ** 2
        else:
            powers[ki] = a_k ** 2 + b_k ** 2

    total_power = powers.sum()

    if total_power < ZERO_POWER_THRESHOLD:
        return {
            "per_freq_power": powers,
            "fcr_top1": 0.0,
            "fcr_top2": 0.0,
            "dominant_freq": 1,
            "total_power": total_power,
        }

    sorted_powers = np.sort(powers)[::-1]
    fcr_top1 = sorted_powers[0] / total_power
    fcr_top2 = (sorted_powers[0] + sorted_powers[1]) / total_power if K >= 2 else fcr_top1
    dominant_freq = int(np.argmax(powers)) + 1

    return {
        "per_freq_power": powers,
        "fcr_top1": fcr_top1,
        "fcr_top2": fcr_top2,
        "dominant_freq": dominant_freq,
        "total_power": total_power,
    }


def fourier_all_coordinates(centroids, values, period, logger=None):
    """
    Compute Fourier analysis across all subspace coordinates.

    Args:
        centroids: (m, d) DC-removed centroids.
        values: (m,) integer concept values.
        period: integer period P.

    Returns dict with two_axis_fcr, uniform_fcr_top1, per-coord stats, etc.
    """
    m, d = centroids.shape
    K = compute_freq_range(period)

    if logger:
        logger.debug("    Fourier: m=%d values, d=%d coords, P=%d, K=%d freqs",
                      m, d, period, K)

    # Per-coordinate Fourier analysis
    coord_results = []
    per_coord_per_freq_power = np.zeros((d, K))
    per_coord_fcr_top1 = np.zeros(d)
    per_coord_dominant_freq = np.zeros(d, dtype=int)

    for j in range(d):
        res = fourier_single_coordinate(centroids[:, j], values, period)
        coord_results.append(res)
        per_coord_per_freq_power[j] = res["per_freq_power"]
        per_coord_fcr_top1[j] = res["fcr_top1"]
        per_coord_dominant_freq[j] = res["dominant_freq"]

    # Total power across all coordinates and frequencies
    total_power = per_coord_per_freq_power.sum()

    if total_power < ZERO_POWER_THRESHOLD:
        if logger:
            logger.warning("    Total power < %.1e: all FCR set to 0.0", ZERO_POWER_THRESHOLD)
        return {
            "two_axis_fcr": 0.0,
            "two_axis_best_freq": 1,
            "two_axis_coord_a": 0,
            "two_axis_coord_b": 1 if d > 1 else 0,
            "uniform_fcr_top1": 0.0,
            "per_coord_fcr_top1": per_coord_fcr_top1,
            "per_coord_dominant_freq": per_coord_dominant_freq,
            "per_coord_per_freq_power": per_coord_per_freq_power,
            "per_freq_two_axis_power": np.zeros(K),
            "per_freq_top2_coords": np.zeros((K, 2), dtype=int),
            "total_power": total_power,
            "fcr_top1_max": 0.0,
            "fcr_top1_max_coord": 0,
            "fcr_top1_max_freq": 1,
            "dominant_freq_mode": 1,
            "n_sig_coords_at_mode_freq": 0,
        }

    # two_axis_fcr: for each frequency, take top-2 coords by power
    per_freq_two_axis_power = np.zeros(K)
    per_freq_top2_coords = np.zeros((K, 2), dtype=int)

    for ki in range(K):
        freq_power = per_coord_per_freq_power[:, ki]
        sorted_idx = np.argsort(freq_power)[::-1]
        top2 = sorted_idx[:min(2, d)]
        per_freq_two_axis_power[ki] = freq_power[top2].sum()
        per_freq_top2_coords[ki, :len(top2)] = top2

    best_freq_idx = int(np.argmax(per_freq_two_axis_power))
    two_axis_fcr = per_freq_two_axis_power[best_freq_idx] / total_power
    two_axis_best_freq = best_freq_idx + 1
    two_axis_coord_a = int(per_freq_top2_coords[best_freq_idx, 0])
    two_axis_coord_b = int(per_freq_top2_coords[best_freq_idx, 1]) if d > 1 else 0

    # Uniform FCR (mean of per-coord FCR_top1)
    uniform_fcr_top1 = float(per_coord_fcr_top1.mean())

    # Max per-coord stats
    fcr_top1_max_coord = int(np.argmax(per_coord_fcr_top1))
    fcr_top1_max = float(per_coord_fcr_top1[fcr_top1_max_coord])
    fcr_top1_max_freq = int(per_coord_dominant_freq[fcr_top1_max_coord])

    # Dominant frequency mode across coords
    freq_counts = np.bincount(per_coord_dominant_freq, minlength=K + 1)
    dominant_freq_mode = int(np.argmax(freq_counts[1:])) + 1 if K > 0 else 1
    n_sig_coords_at_mode_freq = int(freq_counts[dominant_freq_mode])

    if logger:
        logger.debug(
            "    two_axis_fcr=%.4f, best_freq=%d, coords=(%d,%d), "
            "uniform_fcr=%.4f, total_power=%.6f",
            two_axis_fcr, two_axis_best_freq,
            two_axis_coord_a, two_axis_coord_b,
            uniform_fcr_top1, total_power,
        )

    return {
        "two_axis_fcr": float(two_axis_fcr),
        "two_axis_best_freq": two_axis_best_freq,
        "two_axis_coord_a": two_axis_coord_a,
        "two_axis_coord_b": two_axis_coord_b,
        "uniform_fcr_top1": float(uniform_fcr_top1),
        "per_coord_fcr_top1": per_coord_fcr_top1,
        "per_coord_dominant_freq": per_coord_dominant_freq,
        "per_coord_per_freq_power": per_coord_per_freq_power,
        "per_freq_two_axis_power": per_freq_two_axis_power,
        "per_freq_top2_coords": per_freq_top2_coords,
        "total_power": float(total_power),
        "fcr_top1_max": float(fcr_top1_max),
        "fcr_top1_max_coord": fcr_top1_max_coord,
        "fcr_top1_max_freq": fcr_top1_max_freq,
        "dominant_freq_mode": dominant_freq_mode,
        "n_sig_coords_at_mode_freq": n_sig_coords_at_mode_freq,
    }


def compute_linear_power(centroids, v_linear, group_sizes=None):
    """
    Compute per-coordinate linear power (unnormalized, Fix 1).

    Uses centered values to match Fourier unnormalized-sum convention.
    linear_power_j = (sum_v (v_centered * c_v[j]))^2

    Fix #4 from review: uses sample-weighted mean for centering when group_sizes
    are available, so that the linear projection isn't biased toward unbalanced groups.

    Args:
        centroids: (m, d) DC-removed centroids.
        v_linear: (m,) linear axis values (Fix 2: mean raw value for tail bins).
        group_sizes: (m,) optional sample counts per group for weighted centering.

    Returns: (d,) array of linear power per coordinate.
    """
    if group_sizes is not None and len(group_sizes) == len(v_linear):
        weights = group_sizes.astype(float)
        v_weighted_mean = np.sum(weights * v_linear) / np.sum(weights)
        v_centered = v_linear - v_weighted_mean
    else:
        v_centered = v_linear - v_linear.mean()
    # (m,) @ (m, d) -> (d,)  then square
    projections = v_centered @ centroids  # (d,)
    linear_power = projections ** 2
    return linear_power


def compute_helix_fcr(fourier_results, linear_power, v_linear=None,
                      group_sizes=None, logger=None):
    """
    Compute helix FCR: two Fourier axes + one linear axis (Fix 9).

    The linear axis is chosen from coordinates NOT in the top-2 Fourier coords
    at the best frequency, ensuring no double-dipping.

    Fix #5 from review: rescale linear power to match Fourier DOF scale before
    pooling into the helix denominator. Under Gaussian null, a Fourier bin (2 DOF)
    has E[P_k] proportional to ||cos_basis||^2 + ||sin_basis||^2 ≈ m, while a
    linear bin (1 DOF) has E[P_lin] proportional to sum(v_centered^2). We rescale
    linear power by m / (2 * sum(v_centered^2)) so E[rescaled] ≈ E[Fourier_k].

    Args:
        fourier_results: dict from fourier_all_coordinates.
        linear_power: (d,) from compute_linear_power (raw, unrescaled).
        v_linear: (m,) linear values used (needed for rescaling).
        group_sizes: (m,) optional sample counts for weighted centering.

    Returns dict with helix_fcr, helix_best_freq, helix_linear_coord, etc.
    """
    d = len(linear_power)
    K = fourier_results["per_coord_per_freq_power"].shape[1]
    per_freq_two_axis_power = fourier_results["per_freq_two_axis_power"]
    per_freq_top2_coords = fourier_results["per_freq_top2_coords"]
    total_fourier_power = fourier_results["total_power"]

    # Rescale linear power to match Fourier scale (Fix #5)
    if v_linear is not None and len(v_linear) > 1:
        if group_sizes is not None and len(group_sizes) == len(v_linear):
            weights = group_sizes.astype(float)
            v_mean = np.sum(weights * v_linear) / np.sum(weights)
        else:
            v_mean = v_linear.mean()
        v_centered = v_linear - v_mean
        v_norm_sq = np.sum(v_centered ** 2)
        m = len(v_linear)
        if v_norm_sq > ZERO_POWER_THRESHOLD:
            # Rescale so E[linear_rescaled] ≈ E[Fourier_k] under null
            scale_factor = m / (2.0 * v_norm_sq)
            linear_power_rescaled = linear_power * scale_factor
        else:
            linear_power_rescaled = linear_power
    else:
        linear_power_rescaled = linear_power

    if total_fourier_power < ZERO_POWER_THRESHOLD:
        if logger:
            logger.debug("    Helix: total_fourier_power < threshold, helix_fcr=0.0")
        return {
            "helix_fcr": 0.0,
            "helix_best_freq": 1,
            "helix_linear_coord": 0,
            "helix_linear_power": 0.0,
            "total_power_helix": 0.0,
        }

    # For each frequency k, compute helix_power_k = two_axis_power_k + best_linear_power
    # where best_linear is the max linear_power among coords NOT in top-2 at freq k.
    helix_power_per_freq = np.zeros(K)
    helix_linear_coord_per_freq = np.zeros(K, dtype=int)
    helix_linear_power_per_freq = np.zeros(K)

    for ki in range(K):
        excluded = set(per_freq_top2_coords[ki].tolist())
        # Find best linear coord excluding top-2 Fourier coords
        best_linear_power = -1.0
        best_linear_coord = -1
        for j in range(d):
            if j not in excluded and linear_power_rescaled[j] > best_linear_power:
                best_linear_power = linear_power_rescaled[j]
                best_linear_coord = j
        if best_linear_coord == -1:
            if d == 0:
                # No coordinates at all — helix is undefined, use zero power
                best_linear_coord = 0
                best_linear_power = 0.0
                if logger:
                    logger.debug("    Helix: d=0, no coords available, using zero power")
            else:
                # All coords used by Fourier (d <= 2), use whatever is available
                best_linear_coord = 0
                best_linear_power = linear_power_rescaled[0]
                if logger:
                    logger.debug("    Helix: d=%d, cannot exclude top-2, using coord 0", d)

        helix_power_per_freq[ki] = per_freq_two_axis_power[ki] + best_linear_power
        helix_linear_coord_per_freq[ki] = best_linear_coord
        helix_linear_power_per_freq[ki] = best_linear_power

    best_freq_idx = int(np.argmax(helix_power_per_freq))
    # Denominator uses only the chosen linear coord (consistent with numerator)
    best_linear_rescaled = float(helix_linear_power_per_freq[best_freq_idx])
    total_power_helix = total_fourier_power + best_linear_rescaled
    helix_fcr = float(helix_power_per_freq[best_freq_idx] / total_power_helix)
    helix_best_freq = best_freq_idx + 1
    helix_linear_coord = int(helix_linear_coord_per_freq[best_freq_idx])
    helix_linear_power = float(linear_power[helix_linear_coord])  # raw (unrescaled) for p-value

    if logger:
        logger.debug(
            "    helix_fcr=%.4f, best_freq=%d, linear_coord=%d, "
            "linear_power=%.6f, total_helix_power=%.6f",
            helix_fcr, helix_best_freq, helix_linear_coord,
            helix_linear_power, total_power_helix,
        )

    return {
        "helix_fcr": helix_fcr,
        "helix_best_freq": helix_best_freq,
        "helix_linear_coord": helix_linear_coord,
        "helix_linear_power": helix_linear_power,
        "total_power_helix": total_power_helix,
    }


def compute_centroids_grouped(projected, labels, unique_values):
    """
    Compute group centroids from projected activations.

    Args:
        projected: (N, d) projected activations.
        labels: (N,) integer group labels.
        unique_values: (m,) sorted unique group labels.

    Returns:
        centroids: (m, d) group centroids.
        group_sizes: (m,) number of samples per group.
    """
    m = len(unique_values)
    d = projected.shape[1]
    centroids = np.zeros((m, d))
    group_sizes = np.zeros(m, dtype=int)

    for i, v in enumerate(unique_values):
        mask = labels == v
        group_sizes[i] = mask.sum()
        if group_sizes[i] > 0:
            centroids[i] = projected[mask].mean(axis=0)

    return centroids, group_sizes


def compute_eigenvalue_weighted_fcr(per_coord_fcr, eigenvalues):
    """Compute eigenvalue-weighted FCR (secondary statistic)."""
    if eigenvalues is None or len(eigenvalues) == 0:
        return float(per_coord_fcr.mean())
    d = min(len(per_coord_fcr), len(eigenvalues))
    weights = eigenvalues[:d].copy()
    weight_sum = weights.sum()
    if weight_sum < 1e-12:
        return float(per_coord_fcr[:d].mean())
    weights = weights / weight_sum
    return float(np.sum(weights * per_coord_fcr[:d]))


# ═══════════════════════════════════════════════════════════════════════════════
# PERMUTATION NULL
# ═══════════════════════════════════════════════════════════════════════════════


def permutation_null(projected, labels, unique_values, period, v_linear,
                     n_perms, rng, logger):
    """
    Compute permutation null distribution for Fourier statistics.

    Shuffles per-sample labels (preserving group sizes), recomputes centroids,
    and runs full Fourier + helix analysis for each permutation.

    Args:
        projected: (N, d) projected activations.
        labels: (N,) integer group labels.
        unique_values: (m,) sorted unique group labels.
        period: integer period P.
        v_linear: (m,) linear axis values for helix test.
        n_perms: number of permutations.
        rng: numpy random generator.

    Returns dict with null distributions and p-values.
    """
    N, d = projected.shape
    m = len(unique_values)
    K = compute_freq_range(period)

    logger.debug(
        "    Permutation null: N=%d, d=%d, m=%d, P=%d, K=%d, n_perms=%d",
        N, d, m, period, K, n_perms,
    )

    # Pre-compute group sizes for efficient shuffling
    # Note: the permutation preserves group sizes exactly (conditioned null,
    # slightly more conservative than Phase C's unconditioned null — acceptable).
    group_sizes_arr = np.array([np.sum(labels == v) for v in unique_values])
    cum_sizes = np.concatenate([[0], np.cumsum(group_sizes_arr)])

    # Verify all samples accounted for
    assert cum_sizes[-1] == N, (
        f"Group sizes sum to {cum_sizes[-1]}, expected {N}"
    )
    logger.debug("    Group sizes: %s (sum=%d)", group_sizes_arr.tolist(), N)

    # Null distribution arrays
    null_two_axis_fcr = np.zeros(n_perms)
    null_helix_fcr = np.zeros(n_perms)
    null_uniform_fcr = np.zeros(n_perms)
    null_coord_fcr = np.zeros((n_perms, d))
    null_linear_power = np.zeros((n_perms, d))

    all_idx = np.arange(N)
    t0 = time.time()
    log_interval = max(1, n_perms // 10)

    for p in range(n_perms):
        # Shuffle sample indices
        rng.shuffle(all_idx)

        # Compute null centroids by splitting shuffled indices into groups
        null_centroids = np.zeros((m, d))
        for i in range(m):
            idx = all_idx[cum_sizes[i]:cum_sizes[i + 1]]
            null_centroids[i] = projected[idx].mean(axis=0)

        # DC removal
        null_centroids -= null_centroids.mean(axis=0)

        # Fourier analysis
        null_fourier = fourier_all_coordinates(null_centroids, unique_values, period)
        null_two_axis_fcr[p] = null_fourier["two_axis_fcr"]
        null_uniform_fcr[p] = null_fourier["uniform_fcr_top1"]
        null_coord_fcr[p] = null_fourier["per_coord_fcr_top1"]

        # Linear power and helix (group_sizes preserved by shuffle)
        null_lp = compute_linear_power(null_centroids, v_linear, group_sizes_arr)
        null_helix = compute_helix_fcr(null_fourier, null_lp, v_linear, group_sizes_arr)
        null_helix_fcr[p] = null_helix["helix_fcr"]
        null_linear_power[p] = null_lp

        if (p + 1) % log_interval == 0:
            elapsed = time.time() - t0
            logger.debug(
                "    Perm %d/%d (%.1fs elapsed, %.1fs/perm)",
                p + 1, n_perms, elapsed, elapsed / (p + 1),
            )

    elapsed = time.time() - t0
    logger.debug(
        "    Permutation null complete: %.1fs total (%.3fs/perm)",
        elapsed, elapsed / n_perms if n_perms > 0 else 0.0,
    )

    # p-value floor: 1 / min(n_perms + 1, m!)
    m_factorial = math.factorial(m)
    p_value_floor = 1.0 / min(n_perms + 1, m_factorial)
    logger.debug(
        "    p_value_floor = 1/min(%d, %d!) = 1/min(%d, %d) = %.6f",
        n_perms + 1, m, n_perms + 1, m_factorial, p_value_floor,
    )

    return {
        "null_two_axis_fcr": null_two_axis_fcr,
        "null_helix_fcr": null_helix_fcr,
        "null_uniform_fcr": null_uniform_fcr,
        "null_coord_fcr": null_coord_fcr,
        "null_linear_power": null_linear_power,
        "p_value_floor": p_value_floor,
        "m_factorial": m_factorial,
        "elapsed_seconds": elapsed,
    }


def compute_pvalues(observed, null_dist):
    """Compute conservative p-value: (count(null >= observed) + 1) / (n + 1)."""
    n = len(null_dist)
    count = np.sum(null_dist >= observed)
    return (count + 1) / (n + 1)


def compute_pvalues_array(observed, null_dist_2d):
    """Compute per-element p-values for array statistics. null_dist_2d: (n_perms, d)."""
    n = null_dist_2d.shape[0]
    counts = np.sum(null_dist_2d >= observed[None, :], axis=0)
    return (counts + 1) / (n + 1)


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_one(level, layer, pop_name, concept_info, period, period_spec,
                values, v_linear, subspace_type, projected_data, labels,
                unique_values, eigenvalues, n_perms, rng, logger):
    """
    Run the full Fourier screening pipeline for one analysis cell.

    Returns a result dict suitable for one row of the summary CSV.
    """
    concept_name = concept_info["name"]
    tier = concept_info["tier"]
    m = len(unique_values)
    N = len(labels)
    d = projected_data.shape[1]

    logger.info(
        "  ANALYZE: %s / L%d / layer%d / %s / %s / %s (P=%d, m=%d, d=%d, N=%d)",
        concept_name, level, layer, pop_name, subspace_type, period_spec,
        period, m, d, N,
    )

    t_start = time.time()

    # Step 1: Compute centroids
    logger.debug("    Step 1: Computing centroids from %d samples", N)
    centroids, group_sizes = compute_centroids_grouped(projected_data, labels, unique_values)
    total_samples = group_sizes.sum()
    assert total_samples == N, (
        f"Sample count mismatch: group_sizes sum to {total_samples}, expected {N}"
    )
    logger.debug("    Centroids shape: %s, group_sizes: %s", centroids.shape, group_sizes.tolist())

    # Step 3: DC removal
    dc_offset = centroids.mean(axis=0)
    centroids -= dc_offset
    logger.debug("    DC offset norm: %.6f", np.linalg.norm(dc_offset))

    # Step 4-6: Fourier analysis
    logger.debug("    Steps 4-6: Fourier analysis (period=%d)", period)
    fourier_res = fourier_all_coordinates(centroids, values, period, logger)

    # Step 6b: Linear power and helix FCR
    logger.debug("    Step 6b: Linear power and helix FCR")
    linear_power = compute_linear_power(centroids, v_linear, group_sizes)
    helix_res = compute_helix_fcr(fourier_res, linear_power, v_linear, group_sizes, logger)

    # Step 7: Eigenvalue-weighted FCR
    eig_fcr = compute_eigenvalue_weighted_fcr(fourier_res["per_coord_fcr_top1"], eigenvalues)
    logger.debug("    Eigenvalue-weighted FCR: %.4f", eig_fcr)

    # Step 8: Permutation null
    logger.debug("    Step 8: Permutation null (%d permutations)", n_perms)
    null_res = permutation_null(
        projected_data, labels, unique_values, period, v_linear, n_perms, rng, logger
    )

    # Compute p-values
    p_two_axis = compute_pvalues(fourier_res["two_axis_fcr"], null_res["null_two_axis_fcr"])
    p_helix = compute_pvalues(helix_res["helix_fcr"], null_res["null_helix_fcr"])
    p_uniform = compute_pvalues(fourier_res["uniform_fcr_top1"], null_res["null_uniform_fcr"])
    p_coord = compute_pvalues_array(
        fourier_res["per_coord_fcr_top1"], null_res["null_coord_fcr"]
    )
    p_linear = compute_pvalues_array(linear_power, null_res["null_linear_power"])

    # Check p-value saturation
    p_saturated = (p_two_axis <= null_res["p_value_floor"] + 1e-10 or
                   p_helix <= null_res["p_value_floor"] + 1e-10)

    logger.info(
        "    p_two_axis=%.4f, p_helix=%.4f, p_uniform=%.4f, p_saturated=%s",
        p_two_axis, p_helix, p_uniform, p_saturated,
    )
    logger.debug("    p_coord: %s", [f"{p:.4f}" for p in p_coord])
    logger.debug("    p_linear: %s", [f"{p:.4f}" for p in p_linear])

    # Step 9: Circle and helix detection (pre-FDR)
    coord_a = fourier_res["two_axis_coord_a"]
    coord_b = fourier_res["two_axis_coord_b"]
    if d == 0:
        # Zero-dimensional subspace: no geometry possible
        circle_detected = False
        helix_detected = False
        helix_linear_coord = 0
        helix_coord_a = 0
        helix_coord_b = 0
    else:
        circle_detected = (
            p_two_axis < PERM_ALPHA
            and p_coord[coord_a] < COORD_P_THRESHOLD
            and p_coord[coord_b] < COORD_P_THRESHOLD
        )

        helix_linear_coord = helix_res["helix_linear_coord"]
        # For helix, find the top-2 Fourier coords at the helix's best frequency
        helix_best_freq_idx = helix_res["helix_best_freq"] - 1
        helix_fourier_top2 = fourier_res["per_freq_top2_coords"][helix_best_freq_idx]
        helix_coord_a = int(helix_fourier_top2[0])
        helix_coord_b = int(helix_fourier_top2[1]) if d > 1 else 0
        helix_detected = (
            d >= 2
            and p_helix < PERM_ALPHA
            and p_coord[helix_coord_a] < COORD_P_THRESHOLD
            and p_coord[helix_coord_b] < COORD_P_THRESHOLD
            and p_linear[helix_linear_coord] < LINEAR_P_THRESHOLD
        )

    # Fix 3: Hierarchical geometry classifier
    if helix_detected:
        geometry_detected = "helix"
    elif circle_detected:
        geometry_detected = "circle"
    else:
        geometry_detected = "none"

    logger.info(
        "    circle_detected=%s, helix_detected=%s, geometry_detected=%s",
        circle_detected, helix_detected, geometry_detected,
    )

    # Step 10: Multi-frequency pattern
    multi_freq_pattern = _classify_multi_freq(fourier_res, p_coord, period, logger)

    elapsed = time.time() - t_start
    logger.debug("    Analysis complete in %.2fs", elapsed)

    # Build result dict
    result = {
        "concept": concept_name,
        "tier": tier,
        "level": level,
        "layer": layer,
        "population": pop_name,
        "subspace_type": subspace_type,
        "period_spec": period_spec,
        "n_groups": m,
        "period": period,
        "values_tested": json.dumps([int(v) for v in values]),
        "d_sub": d,
        "n_samples_used": int(N),
        "n_perms_used": int(n_perms),
        # Two-axis (circle) statistics
        "two_axis_fcr": fourier_res["two_axis_fcr"],
        "two_axis_best_freq": fourier_res["two_axis_best_freq"],
        "two_axis_coord_a": coord_a,
        "two_axis_coord_b": coord_b,
        "two_axis_p_value": float(p_two_axis),
        "two_axis_q_value": np.nan,  # filled in by FDR step
        # Uniform FCR
        "uniform_fcr_top1": fourier_res["uniform_fcr_top1"],
        "uniform_fcr_p_value": float(p_uniform),
        # Eigenvalue-weighted
        "eigenvalue_fcr_top1": eig_fcr,
        # Per-coord max
        "fcr_top1_max": fourier_res["fcr_top1_max"],
        "fcr_top1_max_coord": fourier_res["fcr_top1_max_coord"],
        "fcr_top1_max_freq": fourier_res["fcr_top1_max_freq"],
        # Dominant frequency
        "dominant_freq_mode": fourier_res["dominant_freq_mode"],
        "n_sig_coords_at_mode_freq": fourier_res["n_sig_coords_at_mode_freq"],
        # Detection
        "circle_detected": bool(circle_detected),
        "helix_detected": bool(helix_detected),
        "geometry_detected": geometry_detected,
        "multi_freq_pattern": multi_freq_pattern,
        # Helix statistics
        "helix_fcr": helix_res["helix_fcr"],
        "helix_best_freq": helix_res["helix_best_freq"],
        "helix_linear_coord": helix_linear_coord,
        "helix_p_value": float(p_helix),
        "helix_q_value": np.nan,  # filled in by FDR step
        # Floors and flags
        "p_value_floor": null_res["p_value_floor"],
        "p_saturated": bool(p_saturated),
        # Agreement placeholder (filled in later)
        "agreement": "",
        # Eigenvalue info (for reference)
        "eigenvalue_top1": float(eigenvalues[0]) if eigenvalues is not None and len(eigenvalues) > 0 else np.nan,
        "eigenvalue_top2": float(eigenvalues[1]) if eigenvalues is not None and len(eigenvalues) > 1 else np.nan,
        "eigenvalue_top3": float(eigenvalues[2]) if eigenvalues is not None and len(eigenvalues) > 2 else np.nan,
        # Internal (for per-concept saving, not in summary CSV)
        "_centroids": centroids,
        "_group_sizes": group_sizes,
        "_fourier_res": fourier_res,
        "_helix_res": helix_res,
        "_p_coord": p_coord,
        "_p_linear": p_linear,
        "_linear_power": linear_power,
    }

    return result


def _classify_multi_freq(fourier_res, p_coord, period, logger):
    """
    Classify multi-frequency pattern after circle detection.

    EXPLORATORY ONLY — do not trust automatic classification for the paper's
    pentagonal-prism story. Proper prism detection requires checking whether
    the same two axes have significant power at multiple frequencies, not just
    whether different coordinates are dominant at different frequencies.
    Manual review of power spectra plots is required for multi-frequency claims.
    """
    K = compute_freq_range(period)
    per_coord_dominant_freq = fourier_res["per_coord_dominant_freq"]

    # Find significant frequencies (coords with p < 0.01)
    sig_mask = p_coord < COORD_P_THRESHOLD
    if not sig_mask.any():
        return "none"
    sig_freqs = set(per_coord_dominant_freq[sig_mask].tolist())

    # Classify
    if sig_freqs == {1}:
        pattern = "{1}"
    elif sig_freqs == {1, 2}:
        pattern = "{1,2}"
    elif sig_freqs == {1, 2, 5}:
        pattern = "{1,2,5}"
    else:
        pattern = str(sorted(sig_freqs))

    logger.debug("    Multi-freq pattern (exploratory): %s (sig_freqs=%s)", pattern, sig_freqs)
    return pattern


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════


def process_level_layer_pop(level, layer, pop_name, paths, coloring_df,
                            fourier_concepts, n_perms, rng, logger):
    """
    Process all Fourier concepts for a single (level, layer, population) slice.

    Returns a list of result dicts.
    """
    results = []
    pop_mask = get_population_mask(coloring_df, pop_name, logger)
    N_pop = pop_mask.sum()

    if N_pop < MIN_POPULATION:
        logger.info(
            "  Skipping L%d/layer%d/%s: N=%d < MIN_POPULATION=%d",
            level, layer, pop_name, N_pop, MIN_POPULATION,
        )
        return results

    logger.info(
        "=== Processing L%d / layer_%02d / %s (N=%d) ===",
        level, layer, pop_name, N_pop,
    )

    # Load residualized activations for Phase D path (once per slice)
    resid_acts = None  # lazy load

    for concept_info in fourier_concepts:
        concept_name = concept_info["name"]
        concept_col = concept_info["column"]

        # Get raw labels from coloring DF
        raw_labels_pop = coloring_df[concept_col].values[pop_mask]

        # Fix #8: Actually apply NaN mask to filter labels and track valid indices
        if np.issubdtype(raw_labels_pop.dtype, np.floating):
            valid_mask = ~np.isnan(raw_labels_pop)
        else:
            valid_mask = np.ones(N_pop, dtype=bool)

        n_valid = valid_mask.sum()
        if n_valid < MIN_POPULATION:
            logger.debug("  Skipping %s: too few valid samples (%d)", concept_name, n_valid)
            continue

        # Apply NaN filter — all downstream uses work on valid-only data
        raw_labels_valid = raw_labels_pop[valid_mask].astype(int)
        pop_valid_indices = np.where(pop_mask)[0][valid_mask]
        logger.debug("  Concept %s: %d valid samples (of %d pop)", concept_name, n_valid, N_pop)

        for spec in concept_info["period_specs"]:
            spec_name = spec["spec_name"]

            # ═══ Get Phase C group info (shared by both paths) ═══
            phase_c_groups, phase_c_meta = _get_phase_c_group_labels(
                level, layer, pop_name, concept_name, paths, logger
            )

            # Fix #1: If carry_binned needs Phase C groups and we don't have them,
            # skip BOTH Phase C and Phase D (not just Phase C).
            if spec.get("needs_phase_c_groups") and phase_c_groups is None:
                logger.debug("  Skipping %s/%s: carry_binned requires Phase C metadata "
                             "but none available (dim_perm=0 or missing)",
                             concept_name, spec_name)
                continue

            # Resolve period/values/v_linear for this spec
            if spec.get("needs_phase_c_groups"):
                resolved_spec = resolve_carry_binned_spec(
                    concept_info, phase_c_groups, raw_labels_valid,
                    logger,
                )
                if resolved_spec is None:
                    logger.debug("  Skipping carry_binned for %s (too few groups)", concept_name)
                    continue
                period = resolved_spec["period"]
                values = resolved_spec["values"]
                v_linear = resolved_spec["v_linear"]
            else:
                period = spec["period"]
                values = spec["values"]
                v_linear = spec["v_linear"]

            # Compute labels using Phase C group mapping (or raw if no binning)
            if phase_c_groups is not None:
                labels, unique_vals, v_linear_mapped = compute_labels_and_linear_values(
                    raw_labels_valid, phase_c_groups
                )
            else:
                labels = raw_labels_valid
                unique_vals = np.array(sorted(np.unique(labels)))
                v_linear_mapped = unique_vals.astype(float)

            # For non-carry_binned, filter to the spec's values
            if not spec.get("needs_phase_c_groups"):
                # carry_mod10 values may be absent if Phase C merged into tail bins
                if spec_name == "carry_mod10" and concept_info["is_carry"]:
                    present_values = np.array([v for v in values if v in unique_vals])
                    if len(present_values) < MIN_CARRY_MOD10_VALUES:
                        logger.debug(
                            "  Skipping carry_mod10 for %s: only %d/%d values "
                            "present in this slice (need %d)",
                            concept_name, len(present_values), len(values),
                            MIN_CARRY_MOD10_VALUES,
                        )
                        continue
                    if len(present_values) < len(values):
                        logger.debug(
                            "  carry_mod10 for %s: using %d/%d values present "
                            "in this slice: %s",
                            concept_name, len(present_values), len(values),
                            present_values.tolist(),
                        )
                        values = present_values
                        v_linear = present_values.astype(float)
                keep_mask_labels = np.isin(labels, values)
            else:
                keep_mask_labels = np.ones(len(labels), dtype=bool)
                # For carry_binned, use the mapped v_linear
                v_linear = v_linear_mapped

            n_after_filter = keep_mask_labels.sum()

            # ═══ Phase C path ═══
            if phase_c_groups is not None:
                projected_c = load_phase_c_projected(
                    level, layer, pop_name, concept_name, paths, logger
                )
                eigenvalues_c = load_phase_c_eigenvalues(
                    level, layer, pop_name, concept_name, paths, logger
                )

                if projected_c is not None:
                    # Apply both valid_mask and value filter
                    proj_c_valid = projected_c[valid_mask]
                    proj_c_filtered = proj_c_valid[keep_mask_labels]
                    labels_c_filtered = labels[keep_mask_labels]

                    if len(proj_c_filtered) < MIN_POPULATION:
                        logger.debug("  Skipping Phase C %s/%s: %d samples after filter < MIN_POPULATION",
                                     concept_name, spec_name, len(proj_c_filtered))
                    else:
                        unique_filtered = values if not spec.get("needs_phase_c_groups") else unique_vals
                        v_linear_filtered = v_linear

                        result_c = analyze_one(
                            level, layer, pop_name, concept_info,
                            period, spec_name, unique_filtered, v_linear_filtered,
                            "phase_c", proj_c_filtered, labels_c_filtered,
                            unique_filtered, eigenvalues_c, n_perms, rng, logger,
                        )
                        results.append(result_c)

            # ═══ Phase D path ═══
            merged_basis = load_phase_d_merged_basis(
                level, layer, pop_name, concept_name, paths, logger
            )

            if merged_basis is not None and merged_basis.shape[0] > 0:
                # Load residualized activations (lazy, once per slice)
                if resid_acts is None:
                    resid_acts = load_residualized(level, layer, paths, logger)

                # Project valid population into merged subspace
                logger.debug("  Projecting into Phase D subspace for %s (d_merged=%d)",
                             concept_name, merged_basis.shape[0])
                pop_acts_valid = resid_acts[pop_valid_indices]
                projected_d = pop_acts_valid @ merged_basis.T

                # Apply value filter (Fix #7: MIN_POPULATION check)
                proj_d_filtered = projected_d[keep_mask_labels]
                labels_d_filtered = labels[keep_mask_labels]

                if len(proj_d_filtered) < MIN_POPULATION:
                    logger.debug("  Skipping Phase D %s/%s: %d samples after filter < MIN_POPULATION",
                                 concept_name, spec_name, len(proj_d_filtered))
                    continue

                unique_filtered = values if not spec.get("needs_phase_c_groups") else unique_vals
                v_linear_filtered = v_linear

                result_d = analyze_one(
                    level, layer, pop_name, concept_info,
                    period, spec_name, unique_filtered, v_linear_filtered,
                    "phase_d_merged", proj_d_filtered, labels_d_filtered,
                    unique_filtered, None, n_perms, rng, logger,
                )
                results.append(result_d)

    logger.info(
        "  L%d/layer%d/%s complete: %d analyses",
        level, layer, pop_name, len(results),
    )
    return results


def run_all(paths, levels, layers, n_perms, logger):
    """Run the full Phase G Fourier screening across all levels, layers, populations."""
    logger.info("=" * 80)
    logger.info("PHASE G: FOURIER SCREENING")
    logger.info("=" * 80)
    logger.info("Levels: %s", levels)
    logger.info("Layers: %s", layers)
    logger.info("Populations: %s", POPULATIONS)
    logger.info("N_PERMUTATIONS: %d", n_perms)
    logger.info("")
    logger.info(RUNTIME_ESTIMATE)
    logger.info("")
    logger.info("Decision rule:\n%s", DECISION_RULE)

    # Pre-launch check: count Phase D bases
    n_bases = count_phase_d_bases(paths, logger)
    if n_bases < 2844:
        logger.error(
            "Expected >= 2844 Phase D merged bases, found %d. "
            "Check that Phase D completed successfully.", n_bases
        )
        sys.exit(1)
    logger.info("Phase D check passed: %d bases >= 2844", n_bases)

    all_results = []
    rng = np.random.default_rng(seed=42)
    checkpoint_path = paths["phase_g_summary"] / "checkpoint_results.pkl"
    paths["phase_g_summary"].mkdir(parents=True, exist_ok=True)

    total_t0 = time.time()

    for level in levels:
        logger.info("=" * 60)
        logger.info("LEVEL %d", level)
        logger.info("=" * 60)

        coloring_df = load_coloring_df(level, paths, logger)
        fourier_concepts = get_fourier_concepts(level, coloring_df, logger)

        for layer in layers:
            for pop_name in POPULATIONS:
                results = process_level_layer_pop(
                    level, layer, pop_name, paths, coloring_df,
                    fourier_concepts, n_perms, rng, logger,
                )
                all_results.extend(results)

        # Per-level checkpoint
        with open(checkpoint_path, "wb") as f:
            pickle.dump(all_results, f)
        logger.info("  Checkpoint saved: %d results after L%d", len(all_results), level)

    total_elapsed = time.time() - total_t0
    logger.info("=" * 60)
    logger.info("ALL ANALYSES COMPLETE: %d results in %.1f minutes",
                len(all_results), total_elapsed / 60)
    logger.info("=" * 60)

    if not all_results:
        logger.warning("No results generated!")
        return pd.DataFrame(), []

    # Build DataFrame
    df = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in all_results
    ])
    logger.info("Results DataFrame: %d rows, %d columns", len(df), len(df.columns))

    return df, all_results


# ═══════════════════════════════════════════════════════════════════════════════
# FDR CORRECTION, DETECTION RULES, AND AGREEMENT
# ═══════════════════════════════════════════════════════════════════════════════


def apply_fdr(results_df, logger):
    """Apply Benjamini-Hochberg FDR correction to p-values."""
    logger.info("Applying FDR correction (BH procedure)...")
    n = len(results_df)
    if n == 0:
        return results_df

    # Fix #15: Handle NaN p-values before FDR — assign q=1.0 for NaN rows
    # FDR for two_axis p-values
    p_vals = results_df["two_axis_p_value"].values.copy()
    nan_mask = np.isnan(p_vals)
    if nan_mask.any():
        logger.warning("  %d NaN two_axis_p_values found — assigning q=1.0", nan_mask.sum())
        p_vals[nan_mask] = 1.0
    q_vals = _benjamini_hochberg(p_vals)
    q_vals[nan_mask] = 1.0
    results_df["two_axis_q_value"] = q_vals
    n_sig_circle = (q_vals < FDR_THRESHOLD).sum()
    logger.info("  two_axis FDR: %d / %d significant at q < %.2f",
                n_sig_circle, n, FDR_THRESHOLD)

    # FDR for helix p-values
    p_vals_h = results_df["helix_p_value"].values.copy()
    nan_mask_h = np.isnan(p_vals_h)
    if nan_mask_h.any():
        logger.warning("  %d NaN helix_p_values found — assigning q=1.0", nan_mask_h.sum())
        p_vals_h[nan_mask_h] = 1.0
    q_vals_h = _benjamini_hochberg(p_vals_h)
    q_vals_h[nan_mask_h] = 1.0
    results_df["helix_q_value"] = q_vals_h
    n_sig_helix = (q_vals_h < FDR_THRESHOLD).sum()
    logger.info("  helix FDR: %d / %d significant at q < %.2f",
                n_sig_helix, n, FDR_THRESHOLD)

    return results_df


def _benjamini_hochberg(p_values):
    """Compute BH-adjusted q-values."""
    n = len(p_values)
    if n == 0:
        return np.array([])
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    q_values = np.zeros(n)
    ranks = np.arange(1, n + 1)
    adjusted = sorted_p * n / ranks
    # Enforce monotonicity from the right
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    q_values[sorted_idx] = adjusted
    return q_values


def compute_agreement(results_df, logger):
    """
    Compute agreement column: how Phase C and Phase D compare per analysis cell.

    Groups by (concept, level, layer, population, period_spec).
    """
    logger.info("Computing Phase C / Phase D agreement...")
    group_cols = ["concept", "level", "layer", "population", "period_spec"]

    agreements = []
    for key, grp in results_df.groupby(group_cols):
        has_c = grp[grp["subspace_type"] == "phase_c"]
        has_d = grp[grp["subspace_type"] == "phase_d_merged"]

        c_detected = (
            len(has_c) > 0 and has_c["geometry_detected"].values[0] != "none"
        )
        d_detected = (
            len(has_d) > 0 and has_d["geometry_detected"].values[0] != "none"
        )

        if c_detected and d_detected:
            agreement = "both"
        elif c_detected:
            agreement = "phase_c_only"
        elif d_detected:
            agreement = "phase_d_only"
        else:
            agreement = "neither"

        for idx in grp.index:
            agreements.append((idx, agreement))

    for idx, agr in agreements:
        results_df.at[idx, "agreement"] = agr

    # Summary
    if len(agreements) > 0:
        counts = results_df["agreement"].value_counts()
        logger.info("  Agreement distribution:\n%s", counts.to_string())

    return results_df


def apply_decision_rule(results_df, logger):
    """
    Apply the pre-registered decision rule and report class-level results.

    Returns a dict summarizing class-level decisions.
    """
    logger.info("=" * 60)
    logger.info("APPLYING DECISION RULE")
    logger.info("=" * 60)
    logger.info(DECISION_RULE)

    # Filter to 'all' population, middle layers, either basis significant
    mask_pop = results_df["population"] == "all"
    mask_layers = results_df["layer"].isin(MIDDLE_LAYERS)
    mask_sig = (
        (results_df["two_axis_q_value"] < FDR_THRESHOLD)
        | (results_df["helix_q_value"] < FDR_THRESHOLD)
    )

    sig_df = results_df[mask_pop & mask_layers & mask_sig].copy()

    # Group cells by concept and layer (collapse across subspace_type and period_spec)
    # A cell is "significant" if ANY row for that (concept, layer) is significant
    cell_df = sig_df.groupby(["concept", "layer"]).agg(
        geometry_any=("geometry_detected", lambda x: any(g != "none" for g in x)),
    ).reset_index()

    cell_df = cell_df.loc[cell_df["geometry_any"]].copy()
    logger.info("Significant concept-layer cells: %d", len(cell_df))

    # Class-level decisions
    concept_classes = {
        "input_digits": [c for c in DIGIT_CONCEPTS_BY_LEVEL.get(5, [])
                         if c.startswith("a_") or c.startswith("b_")],
        "answer_digits": [c for c in DIGIT_CONCEPTS_BY_LEVEL.get(5, [])
                          if c.startswith("ans_")],
        "carries": [c for c in CARRY_CONCEPTS_BY_LEVEL.get(5, [])],
    }

    decisions = {}
    for class_name, class_concepts in concept_classes.items():
        class_cells = cell_df[cell_df["concept"].isin(class_concepts)]
        n_cells = len(class_cells)
        n_concepts = class_cells["concept"].nunique()
        n_layers = class_cells["layer"].nunique()

        confirmed = n_cells >= 3 and n_concepts >= 2 and n_layers >= 2
        decisions[class_name] = {
            "confirmed": confirmed,
            "n_significant_cells": n_cells,
            "n_concepts": n_concepts,
            "n_layers": n_layers,
            "concepts": sorted(class_cells["concept"].unique().tolist()),
            "layers": sorted(class_cells["layer"].unique().tolist()),
        }
        status = "CONFIRMED" if confirmed else "NOT CONFIRMED"
        logger.info(
            "  %s: %s (cells=%d, concepts=%d, layers=%d)",
            class_name, status, n_cells, n_concepts, n_layers,
        )

    # Separate circle vs helix counts
    for geom in ["circle", "helix"]:
        geom_mask = sig_df["geometry_detected"] == geom
        n_geom = geom_mask.sum()
        logger.info("  %s detections: %d", geom, n_geom)

    return decisions


# ═══════════════════════════════════════════════════════════════════════════════
# SAVING
# ═══════════════════════════════════════════════════════════════════════════════


def save_per_concept_results(result, paths, logger):
    """Save per-concept Fourier results (JSON + centroids.npy)."""
    level = result["level"]
    layer = result["layer"]
    pop = result["population"]
    concept = result["concept"]
    subspace_type = result["subspace_type"]
    period_spec = result["period_spec"]

    subdir = "phase_c" if subspace_type == "phase_c" else "phase_d_merged"
    out_dir = (
        paths["phase_g_output"]
        / f"L{level}"
        / f"layer_{layer:02d}"
        / pop
        / concept
        / subdir
        / period_spec
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save centroids
    centroids = result.get("_centroids")
    if centroids is not None:
        np.save(out_dir / "centroids.npy", centroids)

    # Save JSON (exclude internal numpy arrays)
    json_result = {
        k: v for k, v in result.items()
        if not k.startswith("_") and not isinstance(v, np.ndarray)
    }
    # Convert numpy types to Python types
    for k, v in json_result.items():
        if isinstance(v, (np.integer,)):
            json_result[k] = int(v)
        elif isinstance(v, (np.floating,)):
            json_result[k] = float(v)
        elif isinstance(v, (np.bool_,)):
            json_result[k] = bool(v)

    with open(out_dir / "fourier_results.json", "w") as f:
        json.dump(json_result, f, indent=2, default=str)

    logger.debug("  Saved per-concept results to %s", out_dir)


def save_summary_csvs(results_df, paths, logger):
    """Save summary CSV files."""
    summary_dir = paths["phase_g_summary"]
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Full results CSV
    csv_cols = [
        "concept", "tier", "level", "layer", "population", "subspace_type",
        "period_spec", "n_groups", "period", "values_tested", "d_sub",
        "n_samples_used", "n_perms_used",
        "two_axis_fcr", "two_axis_best_freq", "two_axis_coord_a", "two_axis_coord_b",
        "two_axis_p_value", "two_axis_q_value",
        "uniform_fcr_top1", "uniform_fcr_p_value",
        "eigenvalue_fcr_top1",
        "fcr_top1_max", "fcr_top1_max_coord", "fcr_top1_max_freq",
        "dominant_freq_mode", "n_sig_coords_at_mode_freq",
        "circle_detected", "helix_detected", "geometry_detected", "multi_freq_pattern",
        "helix_fcr", "helix_best_freq", "helix_linear_coord",
        "helix_p_value", "helix_q_value",
        "p_value_floor", "p_saturated",
        "agreement",
        "eigenvalue_top1", "eigenvalue_top2", "eigenvalue_top3",
    ]
    available_cols = [c for c in csv_cols if c in results_df.columns]
    results_df[available_cols].to_csv(summary_dir / "phase_g_results.csv", index=False)
    logger.info("Saved phase_g_results.csv: %d rows", len(results_df))

    # Circles-only CSV
    circles = results_df[results_df["circle_detected"] == True]
    if len(circles) > 0:
        circles[available_cols].to_csv(summary_dir / "phase_g_circles.csv", index=False)
        logger.info("Saved phase_g_circles.csv: %d rows", len(circles))

    # Helices-only CSV
    helices = results_df[results_df["helix_detected"] == True]
    if len(helices) > 0:
        helices[available_cols].to_csv(summary_dir / "phase_g_helices.csv", index=False)
        logger.info("Saved phase_g_helices.csv: %d rows", len(helices))

    # Agreement CSV
    agreement_cols = [
        "concept", "level", "layer", "population", "period_spec",
        "subspace_type", "geometry_detected", "agreement",
        "two_axis_q_value", "helix_q_value",
    ]
    avail_agr = [c for c in agreement_cols if c in results_df.columns]
    results_df[avail_agr].to_csv(summary_dir / "phase_g_agreement.csv", index=False)
    logger.info("Saved phase_g_agreement.csv: %d rows", len(results_df))


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════


def plot_fcr_heatmaps(results_df, paths, logger):
    """Plot FCR heatmaps: concept x layer for each (level, population, subspace_type)."""
    plot_dir = paths["plots"] / "fcr_heatmaps"
    plot_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating FCR heatmaps...")

    for (level, pop, stype, pspec), grp in results_df.groupby(
        ["level", "population", "subspace_type", "period_spec"]
    ):
        pivot = grp.pivot_table(
            index="concept", columns="layer", values="two_axis_fcr", aggfunc="first"
        )
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, max(4, len(pivot) * 0.4)))
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Concept")
        ax.set_title(f"two_axis_fcr — L{level} / {pop} / {stype} / {pspec}")
        plt.colorbar(im, ax=ax, label="two_axis_fcr")

        # Annotate with values
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

        fname = f"fcr_heatmap_L{level}_{pop}_{stype}_{pspec}.png"
        fig.tight_layout()
        fig.savefig(plot_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.debug("  Saved %s", fname)

    logger.info("  FCR heatmaps complete")


def plot_centroid_circles(results_df, paths, all_results, logger):
    """Plot centroid projections on top-2 coordinates with unit circle overlay."""
    plot_dir = paths["plots"] / "centroid_circles"
    plot_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating centroid circle plots...")

    n_plotted = 0
    for result in all_results:
        if result.get("geometry_detected") == "none":
            continue
        centroids = result.get("_centroids")
        if centroids is None:
            continue

        coord_a = result["two_axis_coord_a"]
        coord_b = result["two_axis_coord_b"]
        if coord_a >= centroids.shape[1] or coord_b >= centroids.shape[1]:
            continue

        x = centroids[:, coord_a]
        y = centroids[:, coord_b]

        fig, ax = plt.subplots(figsize=(6, 6))
        # Normalize to unit circle for overlay
        r_max = max(np.max(np.abs(x)), np.max(np.abs(y)), 1e-10)
        x_norm = x / r_max
        y_norm = y / r_max

        # Unit circle
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3, linewidth=1)

        # Centroids colored by value
        values = json.loads(result["values_tested"])
        scatter = ax.scatter(x_norm, y_norm, c=values, cmap="tab10",
                             s=80, edgecolors="k", zorder=5)
        for i, v in enumerate(values):
            ax.annotate(str(v), (x_norm[i], y_norm[i]),
                        fontsize=8, ha="center", va="bottom", xytext=(0, 5),
                        textcoords="offset points")

        # RMS residual from unit circle
        radii = np.sqrt(x_norm ** 2 + y_norm ** 2)
        rms_resid = np.sqrt(np.mean((radii - 1.0) ** 2))

        ax.set_aspect("equal")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_title(
            f"{result['concept']} L{result['level']}/layer{result['layer']}/{result['population']}\n"
            f"{result['subspace_type']}/{result['period_spec']} "
            f"(FCR={result['two_axis_fcr']:.3f}, RMS={rms_resid:.3f})"
        )
        ax.set_xlabel(f"Coord {coord_a}")
        ax.set_ylabel(f"Coord {coord_b}")

        fname = (
            f"circle_{result['concept']}_L{result['level']}_layer{result['layer']}_"
            f"{result['population']}_{result['subspace_type']}_{result['period_spec']}.png"
        )
        fig.tight_layout()
        fig.savefig(plot_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        n_plotted += 1

    logger.info("  Centroid circle plots: %d saved", n_plotted)


def plot_power_spectra(results_df, paths, all_results, logger):
    """Plot Fourier power spectra for detected concepts."""
    plot_dir = paths["plots"] / "frequency_spectra"
    plot_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating power spectrum plots...")

    n_plotted = 0
    for result in all_results:
        fourier_res = result.get("_fourier_res")
        if fourier_res is None:
            continue
        if result.get("geometry_detected") == "none":
            continue

        per_freq_power = fourier_res["per_freq_two_axis_power"]
        K = len(per_freq_power)
        freqs = np.arange(1, K + 1)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(freqs, per_freq_power, color="steelblue", edgecolor="k")
        ax.set_xlabel("Frequency k")
        ax.set_ylabel("Two-axis power")
        ax.set_title(
            f"{result['concept']} L{result['level']}/layer{result['layer']}/{result['population']}\n"
            f"{result['subspace_type']}/{result['period_spec']} "
            f"(two_axis_fcr={result['two_axis_fcr']:.3f})"
        )
        ax.set_xticks(freqs)

        fname = (
            f"spectrum_{result['concept']}_L{result['level']}_layer{result['layer']}_"
            f"{result['population']}_{result['subspace_type']}_{result['period_spec']}.png"
        )
        fig.tight_layout()
        fig.savefig(plot_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        n_plotted += 1

    logger.info("  Power spectrum plots: %d saved", n_plotted)


def plot_pvalue_trajectories(results_df, paths, logger):
    """Plot p-value vs layer trajectories for each concept."""
    plot_dir = paths["plots"] / "pvalue_trajectories"
    plot_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating p-value trajectory plots...")

    for (concept, pop, stype, pspec), grp in results_df.groupby(
        ["concept", "population", "subspace_type", "period_spec"]
    ):
        grp_sorted = grp.sort_values("layer")
        layers = grp_sorted["layer"].values
        p_circle = grp_sorted["two_axis_p_value"].values
        p_helix = grp_sorted["helix_p_value"].values

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.semilogy(layers, p_circle, "o-", label="two_axis (circle)", color="blue")
        ax.semilogy(layers, p_helix, "s--", label="helix", color="red")
        ax.axhline(PERM_ALPHA, color="gray", linestyle=":", label=f"alpha={PERM_ALPHA}")
        ax.axhline(FDR_THRESHOLD, color="orange", linestyle=":", label=f"FDR={FDR_THRESHOLD}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("p-value")
        ax.set_title(f"{concept} / {pop} / {stype} / {pspec}")
        ax.legend(fontsize=8)
        ax.set_xticks(LAYERS)

        levels = grp_sorted["level"].unique()
        level_str = "_".join(str(l) for l in levels)
        fname = f"pval_{concept}_L{level_str}_{pop}_{stype}_{pspec}.png"
        fig.tight_layout()
        fig.savefig(plot_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info("  P-value trajectory plots complete")


def plot_single_example_projections(results_df, paths, all_results, logger):
    """
    Plot individual activation projections for concepts where both tests are null.

    Fix 6: No manifold_visual CSV flag. Plots are purely visual deliverables
    for human inspection.
    """
    plot_dir = paths["plots"] / "single_example_projections"
    plot_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating single-example projection plots for null concepts...")

    # Fix #9: Cache coloring DFs to avoid reloading per result
    coloring_cache = {}

    n_plotted = 0
    for result in all_results:
        if result.get("geometry_detected") != "none":
            continue
        if result["subspace_type"] != "phase_c":
            continue  # Only Phase C has projected_all on disk

        level = result["level"]
        layer = result["layer"]
        pop = result["population"]
        concept = result["concept"]

        # Load projected_all.npy
        proj_path = (
            paths["phase_c_subspaces"]
            / f"L{level}"
            / f"layer_{layer:02d}"
            / pop
            / concept
            / "projected_all.npy"
        )
        if not proj_path.exists():
            continue

        projected = np.load(proj_path)
        coord_a = result["two_axis_coord_a"]
        coord_b = result["two_axis_coord_b"]
        if coord_a >= projected.shape[1] or coord_b >= projected.shape[1]:
            continue

        # Load labels (cached)
        if level not in coloring_cache:
            coloring_cache[level] = load_coloring_df(level, paths, logger)
        coloring_df = coloring_cache[level]
        pop_mask = get_population_mask(coloring_df, pop, logger)
        raw_labels = coloring_df[result["concept"]].values[pop_mask]

        x = projected[:, coord_a]
        y = projected[:, coord_b]

        fig, ax = plt.subplots(figsize=(8, 8))
        scatter = ax.scatter(x, y, c=raw_labels, cmap="tab10", s=3, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label="value")
        ax.set_xlabel(f"Coord {coord_a}")
        ax.set_ylabel(f"Coord {coord_b}")
        ax.set_title(
            f"{concept} L{level}/layer{layer}/{pop} (NULL)\n"
            f"Individual activations projected onto top-2 coords"
        )

        fname = (
            f"single_{concept}_L{level}_layer{layer}_{pop}.png"
        )
        fig.tight_layout()
        fig.savefig(plot_dir / fname, dpi=100, bbox_inches="tight")
        plt.close(fig)
        n_plotted += 1

    logger.info("  Single-example projection plots: %d saved", n_plotted)


def generate_all_plots(results_df, paths, all_results, logger):
    """Generate all Phase G plots."""
    logger.info("=" * 40)
    logger.info("GENERATING PLOTS")
    logger.info("=" * 40)

    # Fix #19: Wrap each plot type in try/except so one failure doesn't
    # crash the entire plotting step.
    plot_funcs = [
        ("FCR heatmaps", lambda: plot_fcr_heatmaps(results_df, paths, logger)),
        ("centroid circles", lambda: plot_centroid_circles(results_df, paths, all_results, logger)),
        ("power spectra", lambda: plot_power_spectra(results_df, paths, all_results, logger)),
        ("p-value trajectories", lambda: plot_pvalue_trajectories(results_df, paths, logger)),
        ("single-example projections", lambda: plot_single_example_projections(results_df, paths, all_results, logger)),
    ]
    for name, func in plot_funcs:
        try:
            func()
        except Exception:
            logger.exception("  Plot generation FAILED for %s — continuing", name)

    logger.info("All plot types attempted.")


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC PILOT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


def run_synthetic_pilot(logger):
    """
    Run synthetic pilot tests to validate Fourier machinery.

    Returns True if all tests pass.
    """
    logger.info("=" * 60)
    logger.info("SYNTHETIC PILOT TESTS")
    logger.info("=" * 60)

    all_passed = True
    rng = np.random.default_rng(seed=12345)

    # Test 1: Perfect circle (P=10, v=0..9)
    logger.info("Test 1: Perfect circle (P=10)")
    values = np.arange(10)
    v_linear = values.astype(float)
    centroids = np.zeros((10, 9))
    centroids[:, 0] = np.cos(2 * np.pi * values / 10)
    centroids[:, 1] = np.sin(2 * np.pi * values / 10)
    centroids -= centroids.mean(axis=0)
    res = fourier_all_coordinates(centroids, values, 10, logger)
    lp = compute_linear_power(centroids, v_linear)
    hx = compute_helix_fcr(res, lp, v_linear, logger=logger)
    if res["two_axis_fcr"] < 0.95:
        logger.error("  FAIL: two_axis_fcr = %.4f (expected ~1.0)", res["two_axis_fcr"])
        all_passed = False
    else:
        logger.info("  PASS: two_axis_fcr = %.4f", res["two_axis_fcr"])

    # Test 2: Random noise
    logger.info("Test 2: Random noise")
    centroids_noise = rng.standard_normal((10, 9))
    centroids_noise -= centroids_noise.mean(axis=0)
    res_noise = fourier_all_coordinates(centroids_noise, values, 10, logger)
    K = compute_freq_range(10)
    null_expectation = 1.0 / K  # ~0.2 for K=5
    if res_noise["two_axis_fcr"] > 0.8:
        logger.error("  FAIL: noise two_axis_fcr = %.4f (expected < 0.8)", res_noise["two_axis_fcr"])
        all_passed = False
    else:
        logger.info("  PASS: noise two_axis_fcr = %.4f (null expectation ~%.2f)",
                     res_noise["two_axis_fcr"], null_expectation)

    # Test 3: Linear/quadratic (no circle). Parabola has genuine Fourier
    # structure (~0.6 FCR) but well below true circles (>0.95).
    logger.info("Test 3: Linear/quadratic signal")
    centroids_lin = np.zeros((10, 9))
    centroids_lin[:, 0] = values / 9.0
    centroids_lin[:, 1] = (values / 9.0) ** 2
    centroids_lin -= centroids_lin.mean(axis=0)
    res_lin = fourier_all_coordinates(centroids_lin, values, 10, logger)
    if res_lin["two_axis_fcr"] > 0.65:
        logger.error("  FAIL: linear/quadratic two_axis_fcr = %.4f (expected < 0.65)",
                      res_lin["two_axis_fcr"])
        all_passed = False
    else:
        logger.info("  PASS: linear/quadratic two_axis_fcr = %.4f (< 0.65, not circular)",
                     res_lin["two_axis_fcr"])

    # Test 4: Incomplete grid (v=1..9, P=10)
    logger.info("Test 4: Incomplete grid (v=1..9)")
    values_inc = np.arange(1, 10)
    centroids_inc = np.zeros((9, 9))
    centroids_inc[:, 0] = np.cos(2 * np.pi * values_inc / 10)
    centroids_inc[:, 1] = np.sin(2 * np.pi * values_inc / 10)
    centroids_inc -= centroids_inc.mean(axis=0)
    res_inc = fourier_all_coordinates(centroids_inc, values_inc, 10, logger)
    if res_inc["two_axis_fcr"] < 0.7:
        logger.error("  FAIL: incomplete grid two_axis_fcr = %.4f (expected > 0.7)",
                      res_inc["two_axis_fcr"])
        all_passed = False
    else:
        logger.info("  PASS: incomplete grid two_axis_fcr = %.4f", res_inc["two_axis_fcr"])

    # Test 5: Pure DC offset (zero signal after removal)
    logger.info("Test 5: Pure DC offset")
    centroids_dc = np.full((10, 9), 5.0)
    centroids_dc -= centroids_dc.mean(axis=0)  # All zeros
    res_dc = fourier_all_coordinates(centroids_dc, values, 10, logger)
    if res_dc["two_axis_fcr"] != 0.0:
        logger.error("  FAIL: DC offset two_axis_fcr = %.4f (expected 0.0)", res_dc["two_axis_fcr"])
        all_passed = False
    else:
        logger.info("  PASS: DC offset two_axis_fcr = 0.0 (zero power, no crash)")

    # Test 6: P=9 conjugate test
    logger.info("Test 6: P=9 conjugate test")
    values_9 = np.arange(9)
    centroids_9 = np.zeros((9, 9))
    centroids_9[:, 0] = np.cos(2 * np.pi * values_9 / 9)
    centroids_9[:, 1] = np.sin(2 * np.pi * values_9 / 9)
    centroids_9 -= centroids_9.mean(axis=0)
    K_9 = compute_freq_range(9)
    res_9 = fourier_all_coordinates(centroids_9, values_9, 9, logger)
    if K_9 != 4:
        logger.error("  FAIL: K=%d for P=9 (expected 4)", K_9)
        all_passed = False
    elif res_9["two_axis_fcr"] < 0.95:
        logger.error("  FAIL: P=9 two_axis_fcr = %.4f (expected ~1.0)", res_9["two_axis_fcr"])
        all_passed = False
    else:
        logger.info("  PASS: P=9 K=%d, two_axis_fcr = %.4f", K_9, res_9["two_axis_fcr"])

    # Test 7: Convention spot-check (skipped in synthetic-only mode)
    logger.info("Test 7: Convention spot-check (requires real data, skipped in synthetic pilot)")

    # Test 8: Nyquist parity test
    logger.info("Test 8: Nyquist parity test ((-1)^v at P=10)")
    centroids_nyq = np.zeros((10, 9))
    centroids_nyq[:, 0] = (-1.0) ** values
    centroids_nyq -= centroids_nyq.mean(axis=0)
    res_nyq = fourier_all_coordinates(centroids_nyq, values, 10, logger)
    # The signal is 1D (coord 0 only), so two_axis_fcr will capture one coord with
    # Nyquist power and one with ~0 power.
    coord0_res = fourier_single_coordinate(centroids_nyq[:, 0], values, 10)
    nyquist_frac = coord0_res["per_freq_power"][-1] / coord0_res["total_power"] \
        if coord0_res["total_power"] > ZERO_POWER_THRESHOLD else 0.0
    if nyquist_frac < 0.95:
        logger.error("  FAIL: Nyquist fraction in coord 0 = %.4f (expected ~1.0)", nyquist_frac)
        all_passed = False
    else:
        logger.info("  PASS: Nyquist fraction = %.4f (2x rescaling working)", nyquist_frac)

    # Test 9: Helix test (circle + linear axis)
    logger.info("Test 9: Helix test (cos, sin, linear)")
    centroids_helix = np.zeros((10, 9))
    centroids_helix[:, 0] = np.cos(2 * np.pi * values / 10)
    centroids_helix[:, 1] = np.sin(2 * np.pi * values / 10)
    centroids_helix[:, 2] = values / 9.0
    centroids_helix -= centroids_helix.mean(axis=0)
    res_h = fourier_all_coordinates(centroids_helix, values, 10, logger)
    lp_h = compute_linear_power(centroids_helix, v_linear)
    hx_h = compute_helix_fcr(res_h, lp_h, v_linear, logger=logger)
    if hx_h["helix_fcr"] <= res_h["two_axis_fcr"]:
        logger.error("  FAIL: helix_fcr (%.4f) <= two_axis_fcr (%.4f)",
                      hx_h["helix_fcr"], res_h["two_axis_fcr"])
        all_passed = False
    elif hx_h["helix_linear_coord"] != 2:
        logger.error("  FAIL: helix_linear_coord = %d (expected 2)", hx_h["helix_linear_coord"])
        all_passed = False
    else:
        logger.info("  PASS: helix_fcr = %.4f > two_axis_fcr = %.4f, linear_coord = %d",
                     hx_h["helix_fcr"], res_h["two_axis_fcr"], hx_h["helix_linear_coord"])

    # Test 10: Pure linear ramp (Fix 1 verification)
    logger.info("Test 10: Pure linear ramp (no Fourier, verify linear power scaling)")
    centroids_ramp = np.zeros((10, 9))
    centroids_ramp[:, 0] = values - 4.5  # centered linear ramp
    centroids_ramp -= centroids_ramp.mean(axis=0)
    res_ramp = fourier_all_coordinates(centroids_ramp, values, 10, logger)
    lp_ramp = compute_linear_power(centroids_ramp, v_linear)
    total_fourier = res_ramp["total_power"]
    total_linear = lp_ramp.sum()
    # Linear power should dominate for a pure ramp
    if total_linear < total_fourier:
        logger.error(
            "  FAIL: linear power (%.4f) < Fourier power (%.4f) for pure ramp",
            total_linear, total_fourier,
        )
        all_passed = False
    else:
        logger.info(
            "  PASS: linear power (%.4f) >= Fourier power (%.4f) for pure ramp",
            total_linear, total_fourier,
        )

    logger.info("=" * 60)
    if all_passed:
        logger.info("ALL SYNTHETIC TESTS PASSED")
    else:
        logger.error("SOME SYNTHETIC TESTS FAILED — investigate before proceeding")
    logger.info("=" * 60)

    return all_passed


# ═══════════════════════════════════════════════════════════════════════════════
# PILOT DATA RUN
# ═══════════════════════════════════════════════════════════════════════════════


def run_pilot_data(paths, logger):
    """
    Run a pilot data analysis on a small subset.

    L3 + L5, layer 16 only, all populations, a_units + a_tens + carry_0,
    both subspace types. ~30 analyses in ~2 minutes.
    """
    logger.info("=" * 60)
    logger.info("PILOT DATA RUN")
    logger.info("=" * 60)

    pilot_levels = [3, 5]
    pilot_layers = [16]
    pilot_concepts = {"a_units", "a_tens", "carry_0"}
    n_perms = 100  # Reduced for pilot speed

    all_results = []
    rng = np.random.default_rng(seed=42)

    for level in pilot_levels:
        coloring_df = load_coloring_df(level, paths, logger)
        fourier_concepts = get_fourier_concepts(level, coloring_df, logger)
        # Filter to pilot concepts
        fourier_concepts = [c for c in fourier_concepts if c["name"] in pilot_concepts]

        for layer in pilot_layers:
            for pop_name in POPULATIONS:
                results = process_level_layer_pop(
                    level, layer, pop_name, paths, coloring_df,
                    fourier_concepts, n_perms, rng, logger,
                )
                all_results.extend(results)

    if all_results:
        df = pd.DataFrame([
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in all_results
        ])
        logger.info("Pilot data run: %d analyses completed", len(df))
        logger.info("  Columns: %s", list(df.columns))
        logger.info("  Concepts: %s", df["concept"].unique().tolist())
        logger.info("  circle_detected: %s", df["circle_detected"].value_counts().to_dict())
        logger.info("  helix_detected: %s", df["helix_detected"].value_counts().to_dict())
        logger.info("  geometry_detected: %s", df["geometry_detected"].value_counts().to_dict())

        # Validate ranges
        assert df["two_axis_fcr"].between(0, 1).all(), "FCR out of range"
        assert df["two_axis_p_value"].between(0, 1).all(), "p-value out of range"
        assert not df["two_axis_fcr"].isna().any(), "NaN in FCR"
        logger.info("  Range validations PASSED")

        return df, all_results
    else:
        logger.warning("Pilot data run produced no results!")
        return pd.DataFrame(), []


def run_pilot_0b(paths, logger):
    """
    Pilot 0b: Raw vs. Residualized spot check.

    Compares FCR on residualized vs raw activations for a_units/L5/layer16/all.

    Fix 8: If >20% disagreement, main experiment should use RAW activations
    and report residualized as sanity check only (raw is ground truth in that case).
    """
    logger.info("=" * 60)
    logger.info("PILOT 0b: RAW vs RESIDUALIZED SPOT CHECK")
    logger.info("=" * 60)

    level, layer, pop_name, concept_name = 5, 16, "all", "a_units"

    coloring_df = load_coloring_df(level, paths, logger)
    pop_mask = get_population_mask(coloring_df, pop_name, logger)
    raw_labels = coloring_df[concept_name].values[pop_mask].astype(int)
    unique_vals = np.arange(10)  # a_units is 0-9
    v_linear = unique_vals.astype(float)

    # Load Phase C basis
    basis_path = (
        paths["phase_c_subspaces"]
        / f"L{level}" / f"layer_{layer:02d}" / pop_name / concept_name / "basis.npy"
    )
    if not basis_path.exists():
        logger.error("  Phase C basis not found: %s", basis_path)
        return {
            "status": "skipped",
            "reason": "no Phase C basis",
            "fcr_residualized": None,
            "fcr_raw": None,
            "fcr_disagreement_pct": None,
            "helix_residualized": None,
            "helix_raw": None,
            "helix_disagreement_pct": None,
            "use_raw": False,
        }
    basis = np.load(basis_path)
    logger.info("  Phase C basis shape: %s", basis.shape)

    # Residualized path
    logger.info("  Computing FCR on RESIDUALIZED activations...")
    resid_acts = load_residualized(level, layer, paths, logger)
    pop_resid = resid_acts[pop_mask]
    grand_mean_r = pop_resid.mean(axis=0)
    projected_r = (pop_resid - grand_mean_r) @ basis.T
    centroids_r, gsizes_r = compute_centroids_grouped(projected_r, raw_labels, unique_vals)
    centroids_r -= centroids_r.mean(axis=0)
    res_r = fourier_all_coordinates(centroids_r, unique_vals, 10, logger)
    lp_r = compute_linear_power(centroids_r, v_linear)
    hx_r = compute_helix_fcr(res_r, lp_r, v_linear, logger=logger)

    # Raw path
    logger.info("  Computing FCR on RAW activations...")
    raw_acts = load_raw_activations(level, layer, paths, logger)
    pop_raw = raw_acts[pop_mask]
    grand_mean_w = pop_raw.mean(axis=0)
    projected_w = (pop_raw - grand_mean_w) @ basis.T
    centroids_w, gsizes_w = compute_centroids_grouped(projected_w, raw_labels, unique_vals)
    centroids_w -= centroids_w.mean(axis=0)
    res_w = fourier_all_coordinates(centroids_w, unique_vals, 10, logger)
    lp_w = compute_linear_power(centroids_w, v_linear)
    hx_w = compute_helix_fcr(res_w, lp_w, v_linear, logger=logger)

    # Compare
    fcr_resid = res_r["two_axis_fcr"]
    fcr_raw = res_w["two_axis_fcr"]
    helix_resid = hx_r["helix_fcr"]
    helix_raw = hx_w["helix_fcr"]

    avg_fcr = (fcr_resid + fcr_raw) / 2 if (fcr_resid + fcr_raw) > 0 else 1.0
    fcr_disagreement = abs(fcr_resid - fcr_raw) / avg_fcr
    avg_helix = (helix_resid + helix_raw) / 2 if (helix_resid + helix_raw) > 0 else 1.0
    helix_disagreement = abs(helix_resid - helix_raw) / avg_helix

    logger.info("  two_axis_fcr: residualized=%.4f, raw=%.4f, disagreement=%.1f%%",
                fcr_resid, fcr_raw, fcr_disagreement * 100)
    logger.info("  helix_fcr: residualized=%.4f, raw=%.4f, disagreement=%.1f%%",
                helix_resid, helix_raw, helix_disagreement * 100)

    # Convention spot-check (Test 7): Phase C centroids via projected_all vs manual projection
    logger.info("  Convention spot-check (Test 7):")
    projected_c = load_phase_c_projected(level, layer, pop_name, concept_name, paths, logger)
    if projected_c is not None:
        centroids_c, _ = compute_centroids_grouped(projected_c, raw_labels, unique_vals)
        centroids_c -= centroids_c.mean(axis=0)
        centroids_r_check = centroids_r  # from residualized path, already DC-removed
        match = np.allclose(centroids_c, centroids_r_check, atol=SPOT_CHECK_ATOL, rtol=SPOT_CHECK_RTOL)
        logger.info("    np.allclose(projected_all centroids, manual centroids) = %s", match)
        if not match:
            max_diff = np.max(np.abs(centroids_c - centroids_r_check))
            logger.warning("    Max absolute difference: %.6f", max_diff)

    # Decision
    use_raw = fcr_disagreement > 0.2 or helix_disagreement > 0.2
    if use_raw:
        logger.warning(
            "  DISAGREEMENT > 20%%: Main experiment should use RAW activations. "
            "Residualized results reported as sanity check only."
        )
    else:
        logger.info("  Agreement within 20%%: Proceed with residualized activations.")

    return {
        "status": "completed",
        "fcr_residualized": fcr_resid,
        "fcr_raw": fcr_raw,
        "fcr_disagreement_pct": fcr_disagreement * 100,
        "helix_residualized": helix_resid,
        "helix_raw": helix_raw,
        "helix_disagreement_pct": helix_disagreement * 100,
        "use_raw": use_raw,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase G: Fourier Screening for Periodic Structure"
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--level", type=int, nargs="+", default=None,
        help="Specific levels to run (default: all)"
    )
    parser.add_argument(
        "--layer", type=int, nargs="+", default=None,
        help="Specific layers to run (default: all)"
    )
    parser.add_argument(
        "--pilot", action="store_true",
        help="Run synthetic pilot tests + pilot data run only"
    )
    parser.add_argument(
        "--pilot-0b", action="store_true",
        help="Run pilot 0b (raw vs residualized spot check) only"
    )
    parser.add_argument(
        "--n-perms", type=int, default=N_PERMUTATIONS,
        help=f"Number of permutations (default: {N_PERMUTATIONS})"
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Skip plot generation"
    )
    parser.add_argument(
        "--skip-null", action="store_true",
        help="Skip permutation null (fast debug mode)"
    )
    return parser.parse_args()


def main():
    """Main entry point for Phase G Fourier screening."""
    args = parse_args()

    cfg = load_config(args.config)
    paths = derive_paths(cfg)
    logger = setup_logging(paths["workspace"])

    logger.info("Phase G Fourier Screening started")
    logger.info("Config: %s", args.config)
    logger.info("Paths: workspace=%s, data_root=%s", paths["workspace"], paths["data_root"])

    t_total = time.time()

    if args.pilot:
        # Run synthetic pilot
        synthetic_passed = run_synthetic_pilot(logger)
        if not synthetic_passed:
            logger.error("Synthetic pilot FAILED. Aborting.")
            sys.exit(1)

        # Run pilot data
        pilot_df, pilot_results = run_pilot_data(paths, logger)
        if len(pilot_df) > 0:
            logger.info("Pilot data run PASSED.")
        else:
            logger.error("Pilot data run produced no results. Check data paths.")

        logger.info("Pilot complete in %.1f minutes", (time.time() - t_total) / 60)
        return

    if args.pilot_0b:
        result = run_pilot_0b(paths, logger)
        logger.info("Pilot 0b result: %s", result)
        return

    # Full run
    levels = args.level if args.level else LEVELS
    layers = args.layer if args.layer else LAYERS
    n_perms = 0 if args.skip_null else args.n_perms

    logger.info("Starting full run: levels=%s, layers=%s, n_perms=%d", levels, layers, n_perms)

    results_df, all_results = run_all(paths, levels, layers, n_perms, logger)

    if len(results_df) == 0:
        logger.error("No results generated! Exiting.")
        return

    # Step 11: FDR correction
    logger.info("Step 11: FDR correction")
    results_df = apply_fdr(results_df, logger)

    # Agreement
    results_df = compute_agreement(results_df, logger)

    # Decision rule
    decisions = apply_decision_rule(results_df, logger)

    # Save results
    logger.info("Saving results...")
    for result in all_results:
        save_per_concept_results(result, paths, logger)
    save_summary_csvs(results_df, paths, logger)

    # Save decision rule results
    decision_path = paths["phase_g_summary"] / "phase_g_decisions.json"
    with open(decision_path, "w") as f:
        json.dump(decisions, f, indent=2)
    logger.info("Saved decisions to %s", decision_path)

    # Plots
    if not args.skip_plots:
        generate_all_plots(results_df, paths, all_results, logger)

    total_elapsed = time.time() - t_total
    logger.info("=" * 60)
    logger.info("PHASE G COMPLETE: %.1f minutes (%.1f hours)",
                total_elapsed / 60, total_elapsed / 3600)
    logger.info("  Total analyses: %d", len(results_df))
    logger.info("  Circles detected: %d", (results_df["circle_detected"] == True).sum())
    logger.info("  Helices detected: %d", (results_df["helix_detected"] == True).sum())
    logger.info("  Geometry detected: %s", results_df["geometry_detected"].value_counts().to_dict())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
