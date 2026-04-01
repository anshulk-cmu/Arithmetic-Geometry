#!/usr/bin/env python3
"""Phase D — LDA Refinement of Phase C Concept Subspaces.

Fisher LDA finds low-variance-but-discriminative directions that Phase C's
centroid-based SVD misses. Uses S_T (total scatter) as the denominator —
invariant to label permutation — with a compact K×K formulation that
avoids the full 4096×4096 generalized eigenproblem.

Key design:
  - S_T shared across all concepts in the same (level, layer, population)
  - ST_inv precomputed on GPU for fast permutation null inner loop
  - dim_perm=0 in Phase C → novelty_ratio=1.0 (empty comparison subspace)
  - product_binned uses raw activations; all others use residualized
  - 5-fold stratified CV (skipped for K=2 binary concepts)
  - Gram-Schmidt merge of Phase C basis + novel LDA directions via SVD

Outputs:
  /data/.../phase_d/
    subspaces/L{level}/layer_{layer:02d}/{pop}/{concept}/
      lda_basis.npy            (n_sig, D) significant LDA directions
      merged_basis.npy         Phase C + novel LDA, orthogonalized
      lda_eigenvalues.npy      all K-1 LDA eigenvalues
      null_lda_eigenvalues.npy (n_perms, K-1) null eigenvalues
      metadata.json
    summary/
      phase_d_results.csv
      lda_novelty_summary.csv
  plots/phase_d/
    eigenvalue_spectra/
    novelty_heatmaps/
    n_sig_heatmaps/
    population_comparison/

Usage:
  python phase_d_lda.py --config config.yaml         # Full run
  python phase_d_lda.py --pilot                       # Smoke test: L3/layer16/Tier1
  python phase_d_lda.py --skip-null --skip-plots      # Fast debug
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import tempfile
import time
from itertools import combinations
from logging.handlers import RotatingFileHandler
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
import yaml
from sklearn.model_selection import StratifiedKFold
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
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

LEVELS = [1, 2, 3, 4, 5]
LAYERS = [4, 6, 8, 12, 16, 20, 24, 28, 31]
MIN_POPULATION = 30
MIN_GROUP_SIZE = 20
N_PERMUTATIONS = 1000
PERM_ALPHA = 0.01
NOVELTY_THRESHOLD = 0.5
REGULARIZATION_FRAC = 1e-4
N_CV_SPLITS = 5
PRODUCT_N_BINS = 10
PP_N_BINS = 9
PLOT_LAYERS = [4, 16, 31]

# L5 carry binning thresholds (see docs/datageneration_analysis.md §23)
L5_CARRY_BIN_THRESHOLDS = {
    "carry_0": None,   # all 9 values viable individually
    "carry_1": 12,     # values 12-17 → one group (13 classes)
    "carry_2": 13,     # values 13-26 → one group (14 classes)
    "carry_3": 9,      # values 9-18  → one group (10 classes)
    "carry_4": 5,      # values 5-9   → one group (6 classes)
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
        "phase_c_summary": dr / "phase_c" / "summary",
        "phase_d_data": dr / "phase_d",
        "subspaces_dir": dr / "phase_d" / "subspaces",
        "summary_dir": dr / "phase_d" / "summary",
        "phase_d_plots": ws / "plots" / "phase_d",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(workspace):
    log_dir = workspace / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase_d")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fh = RotatingFileHandler(log_dir / "phase_d_lda.log",
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

def load_activations(level, layer, act_dir):
    return np.load(act_dir / f"level{level}_layer{layer}.npy")


def load_coloring_df(level, coloring_dir):
    return pd.read_pickle(coloring_dir / f"L{level}_coloring.pkl")


def load_residualized(level, layer, residualized_dir):
    return np.load(residualized_dir / f"level{level}_layer{layer}.npy")


def get_populations(df):
    """Return dict of {pop_name: df_subset} for viable populations."""
    pops = {"all": df}
    correct_df = df[df["correct"] == True]
    wrong_df = df[df["correct"] == False]
    if len(correct_df) >= MIN_POPULATION:
        pops["correct"] = correct_df
    if len(wrong_df) >= MIN_POPULATION:
        pops["wrong"] = wrong_df
    return pops


# ═══════════════════════════════════════════════════════════════════════════════
# CONCEPT REGISTRY (replicated from Phase C for standalone execution)
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
            concepts.append({"name": col, "column": col,
                             "tier": 1, "preprocess": None})

    # Tier 1: Answer digits (MSF = most-significant-first)
    ad_idx = 0
    while f"ans_digit_{ad_idx}_msf" in cols:
        col = f"ans_digit_{ad_idx}_msf"
        concepts.append({"name": col, "column": col,
                         "tier": 1, "preprocess": None})
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
            concepts.append({"name": col, "column": col,
                             "tier": 3, "preprocess": None})
            dc_idx += 1

    # Tier 4: Partial products
    pp_cols = sorted([c for c in cols if c.startswith("pp_")])
    for col in pp_cols:
        concepts.append({"name": col, "column": col,
                         "tier": 4, "preprocess": "bin_9"})

    return concepts


# ═══════════════════════════════════════════════════════════════════════════════
# CONCEPT VALUE PREPROCESSING (replicated from Phase C)
# ═══════════════════════════════════════════════════════════════════════════════

def bin_partial_product(values, n_bins=PP_N_BINS):
    """Bin partial product values (0-81) into equal-width bins."""
    bins = np.linspace(-0.5, 81.5, n_bins + 1)
    return np.digitize(values, bins) - 1


def bin_product_deciles(values, n_bins=PRODUCT_N_BINS):
    """Bin continuous product values into quantile bins."""
    try:
        binned = pd.qcut(pd.Series(values), q=n_bins,
                         labels=False, duplicates="drop")
    except ValueError:
        binned = pd.cut(pd.Series(values), bins=n_bins, labels=False)
    return np.asarray(binned, dtype=np.float64)


def bin_carry_tail(values, threshold):
    """Merge carry values >= threshold into a single group."""
    result = np.asarray(values, dtype=np.float64).copy()
    result[result >= threshold] = float(threshold)
    return result


def preprocess_concept(values, preprocess_type, bin_threshold=None):
    """Apply preprocessing to concept values. Returns numpy array."""
    if preprocess_type is None:
        return np.asarray(values, dtype=np.float64)
    elif preprocess_type == "filter_min_group":
        return np.asarray(values, dtype=np.float64)
    elif preprocess_type == "bin_carry_tail":
        return bin_carry_tail(np.asarray(values, dtype=np.float64),
                              bin_threshold)
    elif preprocess_type == "bin_9":
        return bin_partial_product(
            np.asarray(values, dtype=np.float64)).astype(np.float64)
    elif preprocess_type == "bin_deciles":
        return bin_product_deciles(values).astype(np.float64)
    else:
        raise ValueError(f"Unknown preprocess type: {preprocess_type}")


def filter_concept_values(values, min_size=MIN_GROUP_SIZE):
    """Filter concept values by minimum group size.

    Values with fewer than min_size samples are dropped (set to NaN).
    Returns (filtered_values, metadata) or (None, metadata) if <2 groups.
    """
    unique_vals, counts = np.unique(values[~np.isnan(values)],
                                    return_counts=True)
    keep_mask = counts >= min_size
    kept_vals = set(unique_vals[keep_mask])
    dropped_vals = set(unique_vals[~keep_mask])
    dropped_count = int(counts[~keep_mask].sum())

    filtered = values.copy()
    for v in dropped_vals:
        filtered[filtered == v] = np.nan

    surviving = np.unique(filtered[~np.isnan(filtered)])
    n_groups = len(surviving)

    metadata = {
        "original_unique": len(unique_vals),
        "kept_values": sorted([int(v) for v in kept_vals]),
        "dropped_values": sorted([int(v) for v in dropped_vals]),
        "dropped_count": dropped_count,
        "surviving_groups": n_groups,
    }

    if n_groups < 2:
        return None, metadata

    return filtered, metadata


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE C RESULT LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_phase_c_result(level, layer, pop, concept, paths):
    """Load Phase C metadata and basis for one concept.

    Returns (metadata_dict, basis_array) or (None, None) if not found.
    """
    concept_dir = (paths["phase_c_subspaces"] / f"L{level}" /
                   f"layer_{layer:02d}" / pop / concept)
    meta_path = concept_dir / "metadata.json"
    basis_path = concept_dir / "basis.npy"

    if not meta_path.exists():
        return None, None

    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, KeyError):
        return None, None

    basis = None
    if basis_path.exists():
        basis = np.load(basis_path)

    return meta, basis


# ═══════════════════════════════════════════════════════════════════════════════
# SCATTER MATRICES AND REGULARIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_total_scatter(acts, mu):
    """Compute S_T = (acts - mu)^T @ (acts - mu).

    Uses GPU via CuPy when available. S_T depends only on activations
    and the overall mean — INVARIANT to label permutation.

    Args:
        acts: (N, d) float32 activations
        mu: (d,) float64 population mean

    Returns:
        S_T: (d, d) float64 on CPU
    """
    if _CUPY_AVAILABLE:
        acts_g = cp.asarray(acts, dtype=cp.float64)
        mu_g = cp.asarray(mu, dtype=cp.float64)
        centered = acts_g - mu_g
        S_T = cp.asnumpy(centered.T @ centered)
        del acts_g, mu_g, centered
        cp.get_default_memory_pool().free_all_blocks()
    else:
        centered = acts.astype(np.float64) - mu
        S_T = centered.T @ centered
    return S_T


def regularize_and_factor(S_T, alpha_frac=REGULARIZATION_FRAC):
    """Regularize S_T and compute Cholesky factor.

    S_T_reg = S_T + alpha * I, where alpha = alpha_frac * trace(S_T) / d.

    Returns:
        L_factor: (L, lower) tuple from scipy.linalg.cho_factor
        alpha: float, the regularization constant used
    """
    d = S_T.shape[0]
    alpha = alpha_frac * np.trace(S_T) / d
    S_T_reg = S_T + alpha * np.eye(d)
    L_factor = scipy.linalg.cho_factor(S_T_reg, lower=True)
    return L_factor, alpha


def compute_ST_inv(L_factor, d):
    """Precompute S_T_reg^{-1} via Cholesky solve.

    Returns (d, d) float64 inverse matrix. Kept on CPU; transferred
    to GPU inside permutation_null_lda when needed.
    """
    return scipy.linalg.cho_solve(L_factor, np.eye(d))


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS MEANS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_class_means(acts_v, values_v, unique_vals, mu):
    """Compute centered class means and counts.

    Args:
        acts_v: (N, d) activations (valid rows only)
        values_v: (N,) concept labels (no NaN)
        unique_vals: sorted unique values
        mu: (d,) mean used for centering (population mean)

    Returns:
        M: (K, d) centered class means (mu_k - mu)
        n_k: (K,) integer counts per class
    """
    K = len(unique_vals)
    n = len(values_v)
    val_to_idx = {v: i for i, v in enumerate(unique_vals)}
    indices = np.array([val_to_idx[v] for v in values_v])

    one_hot = np.zeros((K, n), dtype=np.float32)
    one_hot[indices, np.arange(n)] = 1.0
    n_k = one_hot.sum(axis=1).astype(np.int64)

    class_means = ((one_hot @ acts_v.astype(np.float64))
                   / np.maximum(n_k[:, None], 1).astype(np.float64))
    M = class_means - mu

    return M, n_k


def _class_means_gpu_from_indices(acts_g, perm_indices, K, n, mu_g):
    """GPU class means from permuted integer indices.

    Args:
        acts_g: (N, d) CuPy float64 array on GPU
        perm_indices: (N,) numpy int array of class assignments
        K: number of classes
        n: number of samples
        mu_g: (d,) CuPy float64 array, population mean

    Returns:
        M_g: (K, d) CuPy array, centered class means
    """
    indices_g = cp.asarray(perm_indices)
    one_hot_g = cp.zeros((K, n), dtype=cp.float32)
    one_hot_g[indices_g, cp.arange(n)] = 1.0
    counts_g = one_hot_g.sum(axis=1, keepdims=True)
    counts_g = cp.maximum(counts_g, 1.0)
    class_means_g = (one_hot_g @ acts_g) / counts_g
    M_g = class_means_g - mu_g
    return M_g


# ═══════════════════════════════════════════════════════════════════════════════
# LDA SOLVER (compact K×K formulation)
# ═══════════════════════════════════════════════════════════════════════════════

def solve_lda_directions(M, n_k, L_factor):
    """Solve Fisher LDA via compact K×K eigenproblem.

    The generalized eigenproblem S_B w = λ S_T w has rank ≤ K-1.
    Instead of the full d×d problem, we exploit the low-rank structure:

    1. Weight:  M_w = diag(sqrt(n_k)) @ M          (K, d)
    2. Solve:   X = S_T_reg^{-1} @ M_w^T           (d, K)
    3. K×K:     A = M_w @ X                         (K, K), symmetric
    4. Eigendecompose A → eigenvalues + eigenvectors
    5. Directions: W = X @ V, normalize to unit length

    Args:
        M: (K, d) centered class means
        n_k: (K,) integer counts per class
        L_factor: Cholesky factor of S_T_reg

    Returns:
        eigenvalues: (K-1,) LDA eigenvalues, descending
        W: (K-1, d) unit-norm discriminant directions (rows)
    """
    K, d = M.shape
    sqrt_nk = np.sqrt(n_k.astype(np.float64))
    M_w = sqrt_nk[:, None] * M  # (K, d)

    X = scipy.linalg.cho_solve(L_factor, M_w.T)  # (d, K)
    A = M_w @ X  # (K, K), symmetric

    eig_vals, eig_vecs = scipy.linalg.eigh(A)
    # eigh returns ascending; reverse to descending, take top K-1
    eig_vals = eig_vals[::-1][:K - 1]
    eig_vecs = eig_vecs[:, ::-1][:, :K - 1]  # (K, K-1)

    W = (X @ eig_vecs).T  # (K-1, d)

    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    W = W / norms

    eig_vals = np.maximum(eig_vals, 0.0)

    return eig_vals, W


# ═══════════════════════════════════════════════════════════════════════════════
# PERMUTATION NULL FOR LDA
# ═══════════════════════════════════════════════════════════════════════════════

def permutation_null_lda(acts_v, values_v, unique_vals, mu_pop,
                         L_factor, ST_inv, n_perms, rng):
    """Permutation null for LDA eigenvalues.

    Shuffles class labels, recomputes LDA eigenvalues under the null.
    S_T (and hence L_factor, ST_inv) is INVARIANT to label permutation,
    so only class means change per permutation.

    GPU path: precomputed ST_inv on GPU enables fast matmul per iteration.
    CPU path: falls back to cho_solve per permutation.

    Args:
        acts_v: (N, d) activations (valid rows only, float64)
        values_v: (N,) concept labels
        unique_vals: sorted unique values
        mu_pop: (d,) population mean (same used for S_T)
        L_factor: Cholesky factor of S_T_reg
        ST_inv: (d, d) S_T_reg^{-1}
        n_perms: number of permutations
        rng: numpy RandomState

    Returns:
        null_eigenvalues: (n_perms, K-1)
    """
    K = len(unique_vals)
    d = acts_v.shape[1]
    n = len(values_v)

    val_to_idx = {v: i for i, v in enumerate(unique_vals)}
    base_indices = np.array([val_to_idx[v] for v in values_v])
    n_k = np.bincount(base_indices, minlength=K).astype(np.float64)
    sqrt_nk = np.sqrt(n_k)

    null_eigenvalues = np.zeros((n_perms, K - 1))

    if _CUPY_AVAILABLE:
        ST_inv_g = cp.asarray(ST_inv, dtype=cp.float64)
        acts_g = cp.asarray(acts_v, dtype=cp.float64)
        mu_g = cp.asarray(mu_pop, dtype=cp.float64)
        sqrt_nk_g = cp.asarray(sqrt_nk, dtype=cp.float64)

        for i in range(n_perms):
            perm_indices = rng.permutation(base_indices)
            M_g = _class_means_gpu_from_indices(
                acts_g, perm_indices, K, n, mu_g)
            M_w_g = sqrt_nk_g[:, None] * M_g

            X_g = ST_inv_g @ M_w_g.T  # (d, K) GPU matmul
            A = cp.asnumpy(M_w_g @ X_g)  # (K, K) → CPU

            eig_vals = scipy.linalg.eigh(A, eigvals_only=True)
            null_eigenvalues[i] = np.maximum(eig_vals[::-1][:K - 1], 0.0)

        del ST_inv_g, acts_g, mu_g, sqrt_nk_g
        cp.get_default_memory_pool().free_all_blocks()
    else:
        for i in range(n_perms):
            perm_indices = rng.permutation(base_indices)

            one_hot = np.zeros((K, n), dtype=np.float32)
            one_hot[perm_indices, np.arange(n)] = 1.0
            counts = np.maximum(one_hot.sum(axis=1, keepdims=True), 1.0)
            class_means = (one_hot @ acts_v) / counts
            M = class_means - mu_pop
            M_w = sqrt_nk[:, None] * M

            X = scipy.linalg.cho_solve(L_factor, M_w.T)
            A = M_w @ X

            eig_vals = scipy.linalg.eigh(A, eigvals_only=True)
            null_eigenvalues[i] = np.maximum(eig_vals[::-1][:K - 1], 0.0)

    return null_eigenvalues


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNIFICANCE, NOVELTY, COHEN'S D
# ═══════════════════════════════════════════════════════════════════════════════

def find_significant_directions(eigenvalues, null_eigenvalues,
                                alpha=PERM_ALPHA):
    """Sequential stopping: count consecutive eigenvalues exceeding null.

    Returns n_sig (int): number of significant LDA directions.
    """
    if null_eigenvalues is None or len(null_eigenvalues) == 0:
        return len(eigenvalues)

    thresholds = np.percentile(null_eigenvalues,
                               100 * (1 - alpha), axis=0)
    n_sig = 0
    for j in range(min(len(eigenvalues), len(thresholds))):
        if eigenvalues[j] > thresholds[j]:
            n_sig += 1
        else:
            break
    return n_sig


def compute_novelty_ratios(W_sig, phase_c_dim_perm, phase_c_basis):
    """Compute novelty ratio for each significant LDA direction.

    novelty_ratio_j = ||w_j - V_c^T V_c w_j|| / ||w_j||

    If phase_c_dim_perm == 0, Phase C's basis reflects noise directions
    from cumvar/ratio consensus (not validated by permutation null).
    Treat as empty comparison subspace → novelty_ratio = 1.0.

    Args:
        W_sig: (n_sig, d) unit-norm LDA directions
        phase_c_dim_perm: int, Phase C's dim_perm value
        phase_c_basis: (dim_consensus, d) Phase C's saved basis, or None

    Returns:
        novelty_ratios: (n_sig,) values in [0, 1]
    """
    n_sig = W_sig.shape[0]

    if (phase_c_dim_perm == 0 or phase_c_basis is None
            or phase_c_basis.shape[0] == 0):
        return np.ones(n_sig)

    V_c = phase_c_basis  # (d_c, d), orthonormal rows
    novelty_ratios = np.zeros(n_sig)
    for j in range(n_sig):
        w = W_sig[j]
        proj = V_c.T @ (V_c @ w)  # project onto Phase C subspace
        resid = w - proj
        novelty_ratios[j] = np.linalg.norm(resid)  # ||w|| = 1
    return novelty_ratios


def compute_cohens_d_all_pairs(W_sig, acts_v, values_v, unique_vals):
    """Cohen's d for all class pairs along each LDA direction.

    d_{ij} = (mu_i·w - mu_j·w) / s_pooled
    s_pooled = sqrt(((n_i-1)*s_i^2 + (n_j-1)*s_j^2) / (n_i + n_j - 2))

    Returns:
        dict mapping direction index → list of {"pair": [i,j], "d": float}
    """
    if W_sig.shape[0] == 0:
        return {}

    projections = acts_v @ W_sig.T  # (N, n_sig)

    class_projs = {}
    for v in unique_vals:
        mask = values_v == v
        class_projs[v] = projections[mask]

    cohens_d = {}
    for j in range(W_sig.shape[0]):
        pairs = []
        for (v_i, v_k) in combinations(unique_vals, 2):
            p_i = class_projs[v_i][:, j]
            p_k = class_projs[v_k][:, j]
            n_i, n_k_val = len(p_i), len(p_k)
            if n_i < 2 or n_k_val < 2:
                continue
            s_i = p_i.var(ddof=1)
            s_k = p_k.var(ddof=1)
            s_pooled = np.sqrt(((n_i - 1) * s_i + (n_k_val - 1) * s_k)
                               / (n_i + n_k_val - 2))
            if s_pooled < 1e-12:
                continue
            d_val = (p_i.mean() - p_k.mean()) / s_pooled
            pairs.append({"pair": [int(v_i), int(v_k)],
                          "d": float(d_val)})
        cohens_d[j] = pairs
    return cohens_d


# ═══════════════════════════════════════════════════════════════════════════════
# GRAM-SCHMIDT MERGE VIA SVD
# ═══════════════════════════════════════════════════════════════════════════════

def gram_schmidt_merge(phase_c_basis, novel_dirs, phase_c_dim_perm):
    """Merge Phase C basis with novel LDA directions via SVD.

    If phase_c_dim_perm == 0, Phase C basis is unvalidated noise directions.
    In that case, merged basis = just the novel LDA directions.

    Args:
        phase_c_basis: (d_c, d) Phase C basis (rows = directions), or None
        novel_dirs: (n_novel, d) novel LDA directions to add
        phase_c_dim_perm: Phase C's dim_perm

    Returns:
        merged: (n_merged, d) orthonormalized basis (rows)
    """
    d = 4096  # default
    if phase_c_basis is not None:
        d = phase_c_basis.shape[1]
    elif novel_dirs is not None and novel_dirs.shape[0] > 0:
        d = novel_dirs.shape[1]

    parts = []
    if (phase_c_dim_perm > 0 and phase_c_basis is not None
            and phase_c_basis.shape[0] > 0):
        parts.append(phase_c_basis)
    if novel_dirs is not None and novel_dirs.shape[0] > 0:
        parts.append(novel_dirs)

    if not parts:
        return np.empty((0, d))

    stacked = np.vstack(parts)
    U, S, Vt = np.linalg.svd(stacked, full_matrices=False)

    tol = 1e-10 * S[0] if len(S) > 0 else 1e-10
    keep = S > tol
    return Vt[keep]


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def cross_validate_lda(acts_v, values_v, unique_vals, n_sig,
                       n_splits=N_CV_SPLITS):
    """5-fold stratified CV for LDA direction stability.

    For each fold: learn LDA on train, project test centroids,
    correlate pairwise distances in full space vs LDA space.

    Skipped for K < 3 (pairwise correlation undefined with 1 pair).

    Returns:
        dict with mean_corr, std_corr, per_fold
    """
    K = len(unique_vals)
    if K < 3 or n_sig < 1:
        return {"mean_corr": np.nan, "std_corr": np.nan, "per_fold": []}

    correlations = []
    try:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=42)
        int_values = values_v.astype(int)

        for train_idx, test_idx in skf.split(acts_v, int_values):
            train_acts = acts_v[train_idx]
            train_vals = values_v[train_idx]
            mu_train = train_acts.mean(axis=0)

            M_train, n_k_train = compute_class_means(
                train_acts, train_vals, unique_vals, mu_train)

            if np.any(n_k_train < 2):
                continue

            S_T_train = compute_total_scatter(train_acts, mu_train)
            L_train, _ = regularize_and_factor(S_T_train)
            _, W_train = solve_lda_directions(M_train, n_k_train, L_train)
            W_train = W_train[:n_sig]

            if W_train.shape[0] < 1:
                continue

            test_acts = acts_v[test_idx]
            test_vals = values_v[test_idx]

            test_centroids = []
            for v in unique_vals:
                mask = test_vals == v
                if mask.sum() < 1:
                    continue
                test_centroids.append(test_acts[mask].mean(axis=0))

            if len(test_centroids) < 3:
                continue
            test_centroids = np.array(test_centroids)

            d_full = pdist(test_centroids)
            d_lda = pdist(test_centroids @ W_train.T)

            if (len(d_full) >= 2 and np.std(d_full) > 1e-12
                    and np.std(d_lda) > 1e-12):
                corr, _ = pearsonr(d_full, d_lda)
                correlations.append(float(corr))
    except ValueError:
        pass

    if not correlations:
        return {"mean_corr": np.nan, "std_corr": np.nan, "per_fold": []}

    return {
        "mean_corr": float(np.mean(correlations)),
        "std_corr": float(np.std(correlations)),
        "per_fold": correlations,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PRINCIPAL ANGLES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_principal_angles(basis_a, basis_b):
    """Principal angles between two subspaces in degrees.

    Args:
        basis_a: (d_a, D) orthonormal basis (rows)
        basis_b: (d_b, D) orthonormal basis (rows)

    Returns:
        angles_deg: (min(d_a, d_b),) principal angles in degrees
    """
    if basis_a.shape[0] == 0 or basis_b.shape[0] == 0:
        return np.array([])
    M = basis_a @ basis_b.T
    S = np.linalg.svd(M, compute_uv=False)
    S = np.clip(S, -1.0, 1.0)
    return np.degrees(np.arccos(S))


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE CONCEPT LDA PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_single_concept_lda(acts_pop, values, concept_name, tier, output_dir,
                           mu_pop, L_factor, ST_inv, alpha,
                           phase_c_meta, phase_c_basis,
                           n_perms, skip_null, rng, logger,
                           novelty_threshold=NOVELTY_THRESHOLD):
    """Run full LDA refinement for one concept.

    Uses shared mu_pop, L_factor, and ST_inv from the parent
    (level, layer, pop) to avoid recomputing S_T for every concept.

    Args:
        acts_pop: (N_pop, d) activations for this population
        values: (N_pop,) preprocessed concept values (may contain NaN)
        concept_name: str
        tier: int
        output_dir: Path
        mu_pop: (d,) population mean (same used for S_T)
        L_factor: Cholesky factor of S_T_reg (shared)
        ST_inv: (d, d) S_T_reg^{-1} (shared), or None if skip_null
        alpha: float, regularization value
        phase_c_meta: dict from Phase C metadata.json, or None
        phase_c_basis: (d_c, d) Phase C basis, or None
        n_perms: int
        skip_null: bool
        rng: np.random.RandomState
        logger: Logger
        novelty_threshold: float

    Returns:
        result dict or None if concept not viable
    """
    # ── Resume logic ──────────────────────────────────────────────────────
    meta_path = output_dir / "metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                cached = json.load(f)
            if cached.get("denominator") != "S_T":
                logger.debug(f"  [{concept_name}] stale cache "
                             f"(denominator={cached.get('denominator')})")
            else:
                logger.debug(f"  [{concept_name}] cached "
                             f"(n_sig={cached.get('n_sig', '?')})")
                return cached
        except (json.JSONDecodeError, KeyError):
            pass

    # ── Filter concept values ─────────────────────────────────────────────
    filtered, filter_meta = filter_concept_values(values)
    if filtered is None:
        logger.debug(f"  [{concept_name}] skipped: "
                     f"{filter_meta['surviving_groups']} groups")
        return None

    valid = ~np.isnan(filtered)
    acts_v = acts_pop[valid].astype(np.float64)
    values_v = filtered[valid]
    unique_vals = np.unique(values_v)
    K = len(unique_vals)
    N = int(valid.sum())
    d = acts_v.shape[1]

    if K < 2:
        logger.debug(f"  [{concept_name}] skipped: K={K} < 2")
        return None

    # ── Class means (centered at population mean) ─────────────────────────
    M, n_k = compute_class_means(acts_v, values_v, unique_vals, mu_pop)

    # ── Solve LDA ─────────────────────────────────────────────────────────
    eigenvalues, W = solve_lda_directions(M, n_k, L_factor)

    # ── Permutation null ──────────────────────────────────────────────────
    null_eigenvalues = None
    n_sig = len(eigenvalues)  # default: all significant if no null

    if not skip_null:
        null_path = output_dir / "null_lda_eigenvalues.npy"
        if null_path.exists():
            null_eigenvalues = np.load(null_path)
            if null_eigenvalues.shape != (n_perms, K - 1):
                null_eigenvalues = None

        if null_eigenvalues is None:
            null_eigenvalues = permutation_null_lda(
                acts_v, values_v, unique_vals, mu_pop,
                L_factor, ST_inv, n_perms, rng)
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(null_path, null_eigenvalues)

        n_sig = find_significant_directions(eigenvalues, null_eigenvalues)

    W_sig = W[:n_sig]

    # ── Phase C comparison ────────────────────────────────────────────────
    phase_c_dim_perm = 0
    if phase_c_meta is not None:
        phase_c_dim_perm = phase_c_meta.get("dim_perm", 0)

    novelty_ratios = compute_novelty_ratios(
        W_sig, phase_c_dim_perm, phase_c_basis)

    novel_mask = novelty_ratios >= novelty_threshold
    n_novel = int(novel_mask.sum())
    novel_dirs = W_sig[novel_mask] if n_novel > 0 else np.empty((0, d))

    # ── Cohen's d ─────────────────────────────────────────────────────────
    cohens_d = compute_cohens_d_all_pairs(
        W_sig, acts_v, values_v, unique_vals)

    max_cohens_d = {}
    for j, pairs in cohens_d.items():
        if pairs:
            max_cohens_d[j] = float(max(abs(p["d"]) for p in pairs))

    # ── Gram-Schmidt merge ────────────────────────────────────────────────
    merged_basis = gram_schmidt_merge(
        phase_c_basis, novel_dirs, phase_c_dim_perm)

    # ── Cross-validation ──────────────────────────────────────────────────
    cv_results = cross_validate_lda(
        acts_v, values_v, unique_vals, n_sig)

    # ── Principal angles: Phase C vs LDA ──────────────────────────────────
    angle_c_lda = np.nan
    if (phase_c_basis is not None and phase_c_basis.shape[0] > 0
            and W_sig.shape[0] > 0):
        angles = compute_principal_angles(phase_c_basis, W_sig)
        if len(angles) > 0:
            angle_c_lda = float(angles[0])

    # ── Save artifacts ────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "lda_eigenvalues.npy", eigenvalues)
    np.save(output_dir / "lda_basis.npy", W_sig)
    np.save(output_dir / "merged_basis.npy", merged_basis)

    # ── Build result ──────────────────────────────────────────────────────
    result = {
        "concept": concept_name,
        "tier": tier,
        "K": int(K),
        "N": int(N),
        "n_lda_directions": int(K - 1),
        "n_sig": int(n_sig),
        "n_novel": int(n_novel),
        "eigenvalues": eigenvalues.tolist(),
        "novelty_ratios": novelty_ratios.tolist(),
        "novel_mask": novel_mask.tolist(),
        "max_cohens_d": {str(k): v for k, v in max_cohens_d.items()},
        "cohens_d_detail": {str(k): v for k, v in cohens_d.items()},
        "merged_dim": int(merged_basis.shape[0]),
        "phase_c_dim_perm": int(phase_c_dim_perm),
        "phase_c_dim_consensus": (int(phase_c_meta.get("dim_consensus", 0))
                                  if phase_c_meta else 0),
        "angle_phase_c_lda": angle_c_lda,
        "cv_mean_corr": cv_results["mean_corr"],
        "cv_std_corr": cv_results["std_corr"],
        "filter_meta": filter_meta,
        "alpha": float(alpha),
        "denominator": "S_T",
        "novelty_threshold": float(novelty_threshold),
    }

    # ── Atomic write ──────────────────────────────────────────────────────
    fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix=".json")
    os.close(fd)
    with open(tmp_path, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp_path, meta_path)

    logger.debug(
        f"  [{concept_name}] n_sig={n_sig} n_novel={n_novel} "
        f"merged={merged_basis.shape[0]} "
        f"pc_perm={phase_c_dim_perm} "
        + (f"angle={angle_c_lda:.1f}" if not np.isnan(angle_c_lda) else ""))

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def process_level_layer_lda(level, layer, paths, n_perms, skip_null,
                            max_tier, logger):
    """Process all populations and concepts for one (level, layer).

    KEY OPTIMIZATION: S_T is computed ONCE per (level, layer, population)
    and shared across all concepts in that population. S_T depends only on
    the activations (not on concept labels), saving ~43 redundant O(N*d^2)
    computations per pop. ST_inv is precomputed once for the GPU perm null.

    product_binned uses raw activations → gets its own S_T/L_factor/ST_inv.
    """
    results = []
    rng = np.random.RandomState(42 + level * 100 + layer)

    df = load_coloring_df(level, paths["coloring_dir"])
    populations = get_populations(df)

    acts_raw = load_activations(level, layer, paths["act_dir"])
    acts_resid = load_residualized(level, layer, paths["residualized_dir"])
    d = acts_resid.shape[1]

    for pop_name, pop_df in populations.items():
        pop_idx = pop_df.index.values
        acts_pop_resid = acts_resid[pop_idx]
        acts_pop_raw = acts_raw[pop_idx]

        # ── Shared S_T for residualized activations ───────────────────────
        mu_resid = acts_pop_resid.astype(np.float64).mean(axis=0)
        t_scatter = time.time()
        S_T_resid = compute_total_scatter(acts_pop_resid, mu_resid)
        L_factor_resid, alpha_resid = regularize_and_factor(S_T_resid)

        ST_inv_resid = None
        if not skip_null:
            ST_inv_resid = compute_ST_inv(L_factor_resid, d)

        logger.info(
            f"  L{level}/layer{layer:02d}/{pop_name}: "
            f"S_T computed ({time.time()-t_scatter:.1f}s, "
            f"N={len(pop_idx)}, alpha={alpha_resid:.2e})")

        # ── Shared S_T for raw activations (product_binned) ──────────────
        mu_raw = acts_pop_raw.astype(np.float64).mean(axis=0)
        S_T_raw = compute_total_scatter(acts_pop_raw, mu_raw)
        L_factor_raw, alpha_raw = regularize_and_factor(S_T_raw)
        ST_inv_raw = None
        if not skip_null:
            ST_inv_raw = compute_ST_inv(L_factor_raw, d)

        # ── Process concepts ──────────────────────────────────────────────
        concepts = get_concept_registry(level, pop_df, pop_name)
        if max_tier is not None:
            concepts = [c for c in concepts if c["tier"] <= max_tier]

        for concept in concepts:
            c_name = concept["name"]
            c_col = concept["column"]
            c_tier = concept["tier"]
            c_pre = concept["preprocess"]

            raw_values = pop_df[c_col].values
            values = preprocess_concept(raw_values, c_pre,
                                        bin_threshold=concept.get(
                                            "bin_threshold"))

            if c_name == "product_binned":
                acts_for_concept = acts_pop_raw
                mu_pop = mu_raw
                L_factor = L_factor_raw
                ST_inv = ST_inv_raw
                alpha = alpha_raw
            else:
                acts_for_concept = acts_pop_resid
                mu_pop = mu_resid
                L_factor = L_factor_resid
                ST_inv = ST_inv_resid
                alpha = alpha_resid

            phase_c_meta, phase_c_basis = load_phase_c_result(
                level, layer, pop_name, c_name, paths)

            output_dir = (paths["subspaces_dir"] / f"L{level}" /
                          f"layer_{layer:02d}" / pop_name / c_name)

            result = run_single_concept_lda(
                acts_for_concept, values, c_name, c_tier,
                output_dir, mu_pop, L_factor, ST_inv, alpha,
                phase_c_meta, phase_c_basis,
                n_perms, skip_null, rng, logger)

            if result is not None:
                result["level"] = level
                result["layer"] = layer
                result["population"] = pop_name
                results.append(result)

        # Free GPU memory between populations
        if _CUPY_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()

    return results


def _gpu_worker(gpu_id, work_queue, result_queue, paths, n_perms,
                skip_null, max_tier, workspace):
    """Worker process for multi-GPU Phase D.

    Each worker pins to one GPU via CUDA_VISIBLE_DEVICES, creates its
    own logger, and pulls (level, layer) pairs from the shared queue
    until it receives a None sentinel.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Re-initialize CuPy on this GPU (module-level init used parent's env)
    global _CUPY_AVAILABLE
    try:
        import cupy as _cp
        _cp.cuda.Device(0).use()  # device 0 = our CUDA_VISIBLE_DEVICES GPU
        _CUPY_AVAILABLE = True
        # Replace module-level cp reference
        import phase_d_lda
        phase_d_lda.cp = _cp
        phase_d_lda._CUPY_AVAILABLE = True
    except Exception:
        _CUPY_AVAILABLE = False

    # Per-worker logger
    logger = logging.getLogger(f"phase_d_gpu{gpu_id}")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        log_path = Path(workspace) / "logs" / f"phase_d_lda_gpu{gpu_id}.log"
        fh = RotatingFileHandler(log_path, maxBytes=10_000_000, backupCount=3)
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter(
            f"%(asctime)s [GPU{gpu_id}] %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)

    logger.info(f"Worker GPU {gpu_id} started, CuPy: {_CUPY_AVAILABLE}")

    while True:
        item = work_queue.get()
        if item is None:
            break
        level, layer = item
        logger.info(f"Processing L{level}/layer{layer:02d}")
        try:
            batch = process_level_layer_lda(
                level, layer, paths, n_perms, skip_null, max_tier, logger)
            result_queue.put(batch)
        except Exception as e:
            logger.error(f"FAILED L{level}/layer{layer:02d}: {e}",
                         exc_info=True)
            result_queue.put([])

    logger.info(f"Worker GPU {gpu_id} finished")


def run_all_lda(paths, levels, layers, n_perms, skip_null, max_tier,
                logger, n_gpus=1):
    """Run Phase D across all specified levels and layers.

    When n_gpus > 1, spawns worker processes that each pin to a
    different GPU and pull (level, layer) pairs from a shared queue.
    """
    pairs = [(level, layer) for level in levels for layer in layers]
    logger.info(f"Processing {len(pairs)} (level, layer) pairs across "
                f"{len(levels)} levels, {len(layers)} layers")

    if n_gpus <= 1:
        # Single-GPU: original sequential loop
        all_results = []
        for level, layer in tqdm(pairs, desc="Phase D"):
            batch = process_level_layer_lda(
                level, layer, paths, n_perms, skip_null, max_tier, logger)
            all_results.extend(batch)
        return all_results

    # Multi-GPU: spawn workers
    logger.info(f"Multi-GPU mode: {n_gpus} workers")
    ctx = mp.get_context("spawn")
    work_queue = ctx.Queue()
    result_queue = ctx.Queue()

    # Enqueue all (level, layer) pairs
    for pair in pairs:
        work_queue.put(pair)
    # Sentinel per worker
    for _ in range(n_gpus):
        work_queue.put(None)

    workers = []
    for gpu_id in range(n_gpus):
        p = ctx.Process(
            target=_gpu_worker,
            args=(gpu_id, work_queue, result_queue, paths,
                  n_perms, skip_null, max_tier, paths["workspace"]))
        p.start()
        workers.append(p)
        logger.info(f"  Started worker GPU {gpu_id} (pid={p.pid})")

    # Collect results while workers run (avoids deadlock on full queue)
    all_results = []
    n_done = 0
    n_total = len(pairs)
    while n_done < n_total:
        batch = result_queue.get()  # blocks until a worker puts results
        all_results.extend(batch)
        n_done += 1
        logger.info(f"  Collected {n_done}/{n_total} (level,layer) pairs, "
                    f"{len(all_results)} results so far")

    for p in workers:
        p.join()

    logger.info(f"Multi-GPU complete: {len(all_results)} results from "
                f"{n_gpus} workers")
    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_lda_eigenvalue_spectrum(eigenvalues, null_eigenvalues, n_sig,
                                title, save_path):
    """Scree plot for LDA eigenvalues with permutation null overlay."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    K_minus_1 = len(eigenvalues)
    x = np.arange(1, K_minus_1 + 1)

    eig_plot = np.maximum(eigenvalues, 1e-15)
    ax.plot(x, eig_plot, "o-", color="C0", linewidth=2, markersize=6,
            label="Observed", zorder=5)

    if null_eigenvalues is not None and len(null_eigenvalues) > 0:
        n_null = min(K_minus_1, null_eigenvalues.shape[1])
        null_50 = np.percentile(null_eigenvalues[:, :n_null], 50, axis=0)
        null_99 = np.percentile(null_eigenvalues[:, :n_null],
                                100 * (1 - PERM_ALPHA), axis=0)
        x_null = np.arange(1, n_null + 1)
        ax.fill_between(x_null, 0, null_99, alpha=0.2, color="gray",
                        label=f"Null {100*(1-PERM_ALPHA):.0f}th pctl")
        ax.plot(x_null, null_50, "--", color="gray", alpha=0.6,
                label="Null median")

    if 0 < n_sig <= K_minus_1:
        ax.axvline(n_sig + 0.5, color="C3", linestyle=":", alpha=0.7,
                   label=f"Sig. cutoff (n={n_sig})")

    ax.set_xlabel("LDA direction index")
    ax.set_ylabel("Generalized eigenvalue (S_B / S_T)")
    ax.set_title(title)
    if (eig_plot.max() > 0 and eig_plot[eig_plot > 0].min() > 0
            and eig_plot.max() > 10 * eig_plot[eig_plot > 0].min()):
        ax.set_yscale("log")
    ax.set_xlim(0.5, K_minus_1 + 0.5)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_novelty_heatmap(results_df, level, pop, save_path):
    """Heatmap: concepts x layers, cells = mean novelty ratio."""
    subset = results_df[(results_df["level"] == level) &
                        (results_df["population"] == pop) &
                        (results_df["n_sig"] > 0)]
    if subset.empty:
        return

    rows = []
    for _, r in subset.iterrows():
        ratios = r.get("novelty_ratios_raw")
        if ratios is not None and len(ratios) > 0:
            mean_nov = float(np.mean(ratios))
        else:
            mean_nov = np.nan
        rows.append({"concept": r["concept"], "layer": r["layer"],
                     "mean_novelty": mean_nov})

    if not rows:
        return

    plot_df = pd.DataFrame(rows)
    pivot = plot_df.pivot_table(index="concept", columns="layer",
                                values="mean_novelty", aggfunc="first")

    fig, ax = plt.subplots(1, 1,
                           figsize=(12, max(4, 0.4 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=1, interpolation="nearest")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Concept")
    ax.set_title(f"LDA Novelty Ratio — L{level} {pop}")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black")

    fig.colorbar(im, ax=ax, label="Mean novelty ratio", shrink=0.8)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_n_sig_heatmap(results_df, level, pop, save_path):
    """Heatmap: concepts x layers, cells = n_sig."""
    subset = results_df[(results_df["level"] == level) &
                        (results_df["population"] == pop)]
    if subset.empty:
        return

    pivot = subset.pivot_table(index="concept", columns="layer",
                               values="n_sig", aggfunc="first")

    fig, ax = plt.subplots(1, 1,
                           figsize=(12, max(4, 0.4 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis",
                   interpolation="nearest")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Concept")
    ax.set_title(f"Significant LDA Directions — L{level} {pop}")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = ("white"
                         if val > np.nanmean(pivot.values)
                         else "black")
                ax.text(j, i, f"{int(val)}", ha="center",
                        va="center", fontsize=8, color=color)

    fig.colorbar(im, ax=ax, label="n_sig", shrink=0.8)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_population_comparison(results_df, concept, level, save_path):
    """Compare n_sig, n_novel, merged_dim across populations."""
    subset = results_df[(results_df["concept"] == concept) &
                        (results_df["level"] == level)]
    if subset.empty or len(subset["population"].unique()) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, metric in enumerate(["n_sig", "n_novel", "merged_dim"]):
        ax = axes[i]
        for pop in ["all", "correct", "wrong"]:
            pop_data = subset[subset["population"] == pop]
            if pop_data.empty:
                continue
            pop_sorted = pop_data.sort_values("layer")
            ax.plot(pop_sorted["layer"], pop_sorted[metric],
                    "o-", label=pop)
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} — {concept} L{level}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(results_df, paths, logger):
    """Generate all Phase D plots."""
    plot_dir = paths["phase_d_plots"]
    n_plots = 0

    # 1. LDA eigenvalue spectra (Tier 1+2 at key layers)
    logger.info("Generating LDA eigenvalue spectrum plots...")
    for _, row in results_df[
            (results_df["tier"].isin([1, 2])) &
            (results_df["layer"].isin(PLOT_LAYERS))].iterrows():
        level = row["level"]
        layer = row["layer"]
        pop = row["population"]
        concept = row["concept"]

        eig_path = (paths["subspaces_dir"] / f"L{level}" /
                    f"layer_{layer:02d}" / pop / concept /
                    "lda_eigenvalues.npy")
        null_path = (paths["subspaces_dir"] / f"L{level}" /
                     f"layer_{layer:02d}" / pop / concept /
                     "null_lda_eigenvalues.npy")

        if not eig_path.exists():
            continue

        eigenvalues = np.load(eig_path)
        null_eig = np.load(null_path) if null_path.exists() else None

        save = (plot_dir / "eigenvalue_spectra" /
                f"L{level}_{pop}_{concept}_layer{layer:02d}.png")
        plot_lda_eigenvalue_spectrum(
            eigenvalues, null_eig, row["n_sig"],
            f"LDA {concept} — L{level} layer {layer} ({pop})", save)
        n_plots += 1

    # 2. Novelty heatmaps
    logger.info("Generating novelty heatmaps...")
    for level in results_df["level"].unique():
        for pop in (results_df[results_df["level"] == level]
                    ["population"].unique()):
            save = plot_dir / "novelty_heatmaps" / f"L{level}_{pop}.png"
            plot_novelty_heatmap(results_df, level, pop, save)
            n_plots += 1

    # 3. n_sig heatmaps
    logger.info("Generating n_sig heatmaps...")
    for level in results_df["level"].unique():
        for pop in (results_df[results_df["level"] == level]
                    ["population"].unique()):
            save = plot_dir / "n_sig_heatmaps" / f"L{level}_{pop}.png"
            plot_n_sig_heatmap(results_df, level, pop, save)
            n_plots += 1

    # 4. Population comparisons (concepts with novel directions)
    logger.info("Generating population comparison plots...")
    key_concepts = set()
    for _, row in results_df[results_df["n_novel"] > 0].iterrows():
        key_concepts.add((row["concept"], row["level"]))

    for concept, level in key_concepts:
        save = (plot_dir / "population_comparison" /
                f"L{level}_{concept}.png")
        plot_population_comparison(results_df, concept, level, save)
        n_plots += 1

    logger.info(f"Generated {n_plots} plots")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def generate_summary(results_df, paths, logger):
    """Generate Phase D summary CSVs."""
    summary_dir = paths["summary_dir"]
    summary_dir.mkdir(parents=True, exist_ok=True)

    # 1. Master results table (flatten for CSV)
    csv_rows = []
    for _, r in results_df.iterrows():
        row = {
            "concept": r["concept"],
            "tier": r["tier"],
            "level": r["level"],
            "layer": r["layer"],
            "population": r["population"],
            "K": r["K"],
            "N": r["N"],
            "n_lda_directions": r["n_lda_directions"],
            "n_sig": r["n_sig"],
            "n_novel": r["n_novel"],
            "merged_dim": r["merged_dim"],
            "phase_c_dim_perm": r["phase_c_dim_perm"],
            "phase_c_dim_consensus": r["phase_c_dim_consensus"],
            "angle_phase_c_lda": r["angle_phase_c_lda"],
            "cv_mean_corr": r["cv_mean_corr"],
            "cv_std_corr": r["cv_std_corr"],
            "alpha": r["alpha"],
            "denominator": r["denominator"],
        }
        # First 3 eigenvalues
        eigs = r.get("eigenvalues", [])
        for k in range(min(3, len(eigs))):
            row[f"lda_eig_{k+1}"] = eigs[k]
        # First 3 novelty ratios
        novs = r.get("novelty_ratios", [])
        for k in range(min(3, len(novs))):
            row[f"novelty_ratio_{k+1}"] = novs[k]
        # Max Cohen's d for first 3 directions
        mcd = r.get("max_cohens_d", {})
        for k in range(3):
            row[f"max_cohens_d_{k+1}"] = mcd.get(str(k), np.nan)
        csv_rows.append(row)

    master_df = pd.DataFrame(csv_rows)
    master_df.to_csv(summary_dir / "phase_d_results.csv", index=False)
    logger.info(f"Saved phase_d_results.csv ({len(master_df)} rows)")

    # 2. Novelty summary
    novel_rows = master_df[master_df["n_novel"] > 0].copy()
    if not novel_rows.empty:
        novel_rows.to_csv(summary_dir / "lda_novelty_summary.csv",
                          index=False)
        logger.info(f"Saved lda_novelty_summary.csv "
                    f"({len(novel_rows)} rows)")

    # 3. Significance tables per level/pop
    for level in master_df["level"].unique():
        for pop in (master_df[master_df["level"] == level]
                    ["population"].unique()):
            subset = master_df[(master_df["level"] == level) &
                               (master_df["population"] == pop)]
            if subset.empty:
                continue
            pivot = subset.pivot_table(
                index="concept", columns="layer",
                values="n_sig", aggfunc="first")
            pivot.to_csv(
                summary_dir / f"lda_significance_L{level}_{pop}.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI & MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase D — LDA Refinement of Phase C Concept Subspaces")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--pilot", action="store_true",
                        help="Smoke test: L3, layer 16, Tier 1 only")
    parser.add_argument("--level", type=int, nargs="*",
                        help="Specific levels (default: all)")
    parser.add_argument("--layer", type=int, nargs="*",
                        help="Specific layers (default: all)")
    parser.add_argument("--n-perms", type=int, default=N_PERMUTATIONS,
                        help=f"Permutation null iterations "
                             f"(default: {N_PERMUTATIONS})")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--skip-null", action="store_true",
                        help="Skip permutation null (fast debug)")
    parser.add_argument("--n-gpus", type=int, default=1,
                        help="Number of GPUs for parallel (level,layer) "
                             "processing (default: 1)")
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    cfg = load_config(args.config)
    paths = derive_paths(cfg)
    logger = setup_logging(paths["workspace"])

    # Banner
    logger.info("=" * 70)
    logger.info("Phase D — LDA Refinement")
    logger.info("=" * 70)

    # Scope
    max_tier = None
    if args.pilot:
        levels = [3]
        layers = [16]
        max_tier = 2
        logger.info("PILOT MODE: L3, layer 16, Tier 1-2 "
                     "(input digits + carries)")
    else:
        levels = args.level if args.level else LEVELS
        layers = args.layer if args.layer else LAYERS

    n_perms = args.n_perms
    logger.info(f"Levels: {levels}")
    logger.info(f"Layers: {layers}")
    logger.info(f"Max tier: {max_tier or 'all'}")
    logger.info(f"Permutations: {n_perms} "
                f"{'(SKIPPED)' if args.skip_null else ''}")
    n_gpus = args.n_gpus
    logger.info(f"GPUs: {n_gpus} "
                f"({'multi-GPU' if n_gpus > 1 else 'single-GPU'})")
    logger.info(f"GPU (CuPy): "
                f"{'available' if _CUPY_AVAILABLE else 'CPU fallback'}")
    logger.info(f"Novelty threshold: {NOVELTY_THRESHOLD}")
    logger.info(f"Regularization: {REGULARIZATION_FRAC}")
    logger.info(f"Denominator: S_T (total scatter — "
                f"invariant to label permutation)")

    # ── Pre-flight checks ─────────────────────────────────────────────────
    missing = []
    for level in levels:
        pkl_path = paths["coloring_dir"] / f"L{level}_coloring.pkl"
        if not pkl_path.exists():
            missing.append(str(pkl_path))
        for layer in layers:
            resid_path = (paths["residualized_dir"] /
                          f"level{level}_layer{layer}.npy")
            if not resid_path.exists():
                missing.append(str(resid_path))
            act_path = paths["act_dir"] / f"level{level}_layer{layer}.npy"
            if not act_path.exists():
                missing.append(str(act_path))

    if missing:
        logger.error(f"Missing {len(missing)} input files. First 5:")
        for p in missing[:5]:
            logger.error(f"  {p}")
        logger.error("Run Phase C first to generate residualized "
                     "activations and subspaces.")
        return

    phase_c_csv = paths["phase_c_summary"] / "phase_c_results.csv"
    if not phase_c_csv.exists():
        logger.warning(f"Phase C results CSV not found: {phase_c_csv}")
        logger.warning("Phase D will proceed but Phase C comparison "
                       "will use per-concept metadata only.")

    # ── Create output directories ─────────────────────────────────────────
    for key in ["phase_d_data", "subspaces_dir", "summary_dir",
                "phase_d_plots"]:
        paths[key].mkdir(parents=True, exist_ok=True)

    # ── Step 1: Core LDA computation ─────────────────────────────────────
    logger.info("")
    logger.info("Step 1: LDA Refinement")
    logger.info("-" * 40)
    t1 = time.time()

    all_results = run_all_lda(
        paths, levels, layers, n_perms,
        args.skip_null, max_tier, logger, n_gpus)

    logger.info(f"Step 1 complete: {len(all_results)} concept LDA results "
                f"({time.time()-t1:.1f}s)")

    if not all_results:
        logger.warning("No results — check data paths and Phase C outputs")
        return

    results_df = pd.DataFrame(all_results)
    results_df["novelty_ratios_raw"] = results_df["novelty_ratios"]

    # ── Step 2: Plots ─────────────────────────────────────────────────────
    if not args.skip_plots:
        logger.info("")
        logger.info("Step 2: Plot generation")
        logger.info("-" * 40)
        t2 = time.time()
        generate_all_plots(results_df, paths, logger)
        logger.info(f"Step 2 complete ({time.time()-t2:.1f}s)")

    # ── Step 3: Summary tables ────────────────────────────────────────────
    logger.info("")
    logger.info("Step 3: Summary tables")
    logger.info("-" * 40)
    generate_summary(results_df, paths, logger)

    # ── Done ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"Phase D complete: {len(all_results)} LDA results, "
                f"{elapsed:.0f}s ({elapsed/60:.1f} min)")
    logger.info("=" * 70)

    sig = results_df[results_df["n_sig"] > 0]
    novel = results_df[results_df["n_novel"] > 0]
    logger.info(f"Concepts with significant LDA: "
                f"{len(sig)} / {len(results_df)}")
    logger.info(f"Concepts with novel directions: "
                f"{len(novel)} / {len(results_df)}")
    if not novel.empty:
        logger.info(f"Mean novel directions (where >0): "
                    f"{novel['n_novel'].mean():.1f}")


if __name__ == "__main__":
    main()
