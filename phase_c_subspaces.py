#!/usr/bin/env python3
"""Phase C — Concept Subspace Identification via Conditional Covariance + SVD.

Identifies the linear subspaces in 4096-dimensional activation space that encode
each mathematical concept (input digits, carries, partial products). Uses full SVD
on small centroid matrices, permutation null for significance, cross-validation for
robustness, and cross-layer principal angles for tracking subspace evolution.

Produces:
  - Basis matrices (.npy) for each significant concept subspace
  - Eigenvalue spectra with permutation null thresholds
  - Projected activations for downstream Fourier screening / GP geometry
  - Dimensionality heatmaps, cross-layer trajectories, correct/wrong divergence
  - Summary CSVs: phase_c_results.csv, significance_table.csv, etc.

Usage:
  python phase_c_subspaces.py --config config.yaml --n-jobs 8    # Full run
  python phase_c_subspaces.py --pilot                             # Smoke test: L3/layer16/all/Tier1
  python phase_c_subspaces.py --skip-null --skip-plots            # Fast debug
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
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

# Optional parallelization
try:
    from joblib import Parallel, delayed
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

LEVELS = [1, 2, 3, 4, 5]
LAYERS = [4, 6, 8, 12, 16, 20, 24, 28, 31]
MIN_POPULATION = 30        # minimum viable population for correct/wrong split
MIN_GROUP_SIZE = 20        # minimum samples per concept value (else merge)
N_PERMUTATIONS = 1000      # validated: minimum for α=0.01 significance
N_CV_SPLITS = 5
CUMVAR_THRESHOLD = 0.95
RATIO_THRESHOLD = 5.0
PERM_ALPHA = 0.01          # 99th percentile for permutation null
PP_N_BINS = 9              # bins for partial product values (0-81 → 9 bins)
PRODUCT_N_BINS = 10        # decile bins for product magnitude

# Eigenvalue spectrum plot: only these layers (to avoid plot explosion)
PLOT_LAYERS = [4, 16, 31]


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
        "labels_dir": ws / "labels",
        "answers_dir": dr / "answers",
        "act_dir": dr / "activations",
        "coloring_dir": dr / "phase_a" / "coloring_dfs",
        "phase_c_data": dr / "phase_c",
        "residualized_dir": dr / "phase_c" / "residualized",
        "subspaces_dir": dr / "phase_c" / "subspaces",
        "summary_dir": dr / "phase_c" / "summary",
        "phase_c_plots": ws / "plots" / "phase_c",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(workspace):
    log_dir = workspace / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase_c")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fh = RotatingFileHandler(log_dir / "phase_c_subspaces.log",
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
    path = act_dir / f"level{level}_layer{layer}.npy"
    return np.load(path)


def load_coloring_df(level, coloring_dir):
    path = coloring_dir / f"L{level}_coloring.pkl"
    return pd.read_pickle(path)


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
# CONCEPT REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

def get_concept_registry(level, df, pop_name="all"):
    """Return list of concept dicts for this level/population.

    Each dict: {"name": str, "column": str, "tier": int, "preprocess": str|None}
    """
    cols = set(df.columns)
    concepts = []

    # Tier 1: Input digits
    digit_cols = ["a_units", "a_tens", "a_hundreds", "b_units", "b_tens", "b_hundreds"]
    for col in digit_cols:
        if col in cols:
            concepts.append({"name": col, "column": col, "tier": 1, "preprocess": None})

    # Tier 1: Answer digits (output targets — what the model is trying to produce)
    # Values 0-9, same structure as input digits. MSF = most-significant-first.
    ad_idx = 0
    while f"ans_digit_{ad_idx}_msf" in cols:
        col = f"ans_digit_{ad_idx}_msf"
        concepts.append({"name": col, "column": col, "tier": 1, "preprocess": None})
        ad_idx += 1

    # Tier 2: Carries
    carry_idx = 0
    while f"carry_{carry_idx}" in cols:
        col = f"carry_{carry_idx}"
        concepts.append({"name": col, "column": col, "tier": 2,
                         "preprocess": "filter_min_group"})
        carry_idx += 1

    # Tier 2: Column sums (bridge between partial products and carries —
    # the pre-carry total at each output position. Qian et al. 2024, He et al. 2025)
    cs_idx = 0
    while f"col_sum_{cs_idx}" in cols:
        col = f"col_sum_{cs_idx}"
        concepts.append({"name": col, "column": col, "tier": 2,
                         "preprocess": "bin_deciles"})
        cs_idx += 1

    # Tier 3: Derived
    if pop_name == "all" and "correct" in cols:
        concepts.append({"name": "correct", "column": "correct", "tier": 3,
                         "preprocess": None})
    if "n_nonzero_carries" in cols:
        concepts.append({"name": "n_nonzero_carries", "column": "n_nonzero_carries",
                         "tier": 3, "preprocess": None})
    if "total_carry_sum" in cols:
        concepts.append({"name": "total_carry_sum", "column": "total_carry_sum",
                         "tier": 3, "preprocess": "filter_min_group"})
    if "max_carry_value" in cols:
        concepts.append({"name": "max_carry_value", "column": "max_carry_value",
                         "tier": 3, "preprocess": "filter_min_group"})
    if "n_answer_digits" in cols:
        concepts.append({"name": "n_answer_digits", "column": "n_answer_digits",
                         "tier": 3, "preprocess": None})
    if "product" in cols:
        concepts.append({"name": "product_binned", "column": "product",
                         "tier": 3, "preprocess": "bin_deciles"})

    # Tier 3: Per-digit correctness (only in "all" population — constant in
    # correct/wrong pops. Shows which output positions the model gets wrong.)
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
# CONCEPT VALUE PREPROCESSING & FILTERING
# ═══════════════════════════════════════════════════════════════════════════════

def bin_partial_product(values, n_bins=PP_N_BINS):
    """Bin partial product values (0-81) into equal-width bins."""
    bins = np.linspace(-0.5, 81.5, n_bins + 1)
    return np.digitize(values, bins) - 1


def bin_product_deciles(values, n_bins=PRODUCT_N_BINS):
    """Bin continuous product values into quantile bins."""
    try:
        binned = pd.qcut(pd.Series(values), q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        # Fallback to equal-width if too few unique values
        binned = pd.cut(pd.Series(values), bins=n_bins, labels=False)
    return np.asarray(binned, dtype=np.float64)


def preprocess_concept(values, preprocess_type):
    """Apply preprocessing to concept values. Returns numpy array."""
    if preprocess_type is None:
        return np.asarray(values, dtype=np.float64)
    elif preprocess_type == "filter_min_group":
        return np.asarray(values, dtype=np.float64)
    elif preprocess_type == "bin_9":
        return bin_partial_product(np.asarray(values, dtype=np.float64)).astype(np.float64)
    elif preprocess_type == "bin_deciles":
        return bin_product_deciles(values).astype(np.float64)
    else:
        raise ValueError(f"Unknown preprocess type: {preprocess_type}")


def filter_concept_values(values, min_size=MIN_GROUP_SIZE):
    """Filter concept values by minimum group size.

    Values with fewer than min_size samples are dropped (set to NaN).
    No merging of semantically different values into a sentinel bin.
    Returns (filtered_values, metadata) or (None, metadata) if <2 groups survive.
    """
    unique_vals, counts = np.unique(values[~np.isnan(values)], return_counts=True)
    keep_mask = counts >= min_size
    kept_vals = set(unique_vals[keep_mask])
    dropped_vals = set(unique_vals[~keep_mask])
    dropped_count = int(counts[~keep_mask].sum())

    # Build filtered array: drop rare values by setting to NaN
    filtered = values.copy()
    for v in dropped_vals:
        filtered[filtered == v] = np.nan

    # Count surviving groups
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
# PRODUCT RESIDUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def residualize_product(acts, product_values, cache_path=None):
    """Remove the product-magnitude direction from activations via OLS.

    Projects activations onto the orthogonal complement of the product-value
    direction. This prevents product magnitude (the dominant axis per Phase A)
    from contaminating concept subspaces.

    Args:
        acts: (N, 4096) activation matrix
        product_values: (N,) product magnitudes
        cache_path: optional path to cache result

    Returns:
        (N, 4096) residualized activations (centered)
    """
    if cache_path is not None and cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape == acts.shape:
            return cached

    X_c = acts - acts.mean(axis=0)
    p_c = product_values.astype(np.float64) - product_values.mean()
    p_dot_p = p_c @ p_c
    if p_dot_p < 1e-12:
        # Product is constant (shouldn't happen but be safe)
        result = X_c
    else:
        beta = X_c.T @ p_c / p_dot_p   # (4096,)
        result = X_c - np.outer(p_c, beta)

    result = result.astype(np.float32)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: write to temp, then rename
        fd, tmp_path = tempfile.mkstemp(dir=cache_path.parent, suffix=".npy")
        os.close(fd)
        np.save(tmp_path, result)
        os.replace(tmp_path, cache_path)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# CORE ALGORITHM
# ═══════════════════════════════════════════════════════════════════════════════

def compute_centroids(acts, concept_values):
    """Compute equal-weight group centroids.

    Uses vectorized scatter via one-hot encoding for speed (~4ms at N=4000).

    Args:
        acts: (N, d) activation matrix (only rows where concept_values is not NaN)
        concept_values: (N,) concept labels

    Returns:
        centroids: (m, d) centroid matrix
        grand_mean: (d,) mean of centroids (equal-weight)
        group_sizes: dict {value: count}
        unique_vals: sorted array of unique values
    """
    valid = ~np.isnan(concept_values)
    acts_v = acts[valid]
    vals_v = concept_values[valid]

    unique_vals = np.unique(vals_v)
    m = len(unique_vals)
    d = acts_v.shape[1]

    # Vectorized centroid computation: one-hot @ acts / counts
    val_to_idx = {v: i for i, v in enumerate(unique_vals)}
    indices = np.array([val_to_idx[v] for v in vals_v])
    one_hot = np.zeros((m, len(vals_v)), dtype=np.float32)
    one_hot[indices, np.arange(len(vals_v))] = 1.0
    counts = one_hot.sum(axis=1, keepdims=True)  # (m, 1)
    centroids = (one_hot @ acts_v) / counts       # (m, d)

    grand_mean = centroids.mean(axis=0)  # equal-weight mean of centroids

    group_sizes = {v: int(c) for v, c in zip(unique_vals, counts.ravel())}

    return centroids, grand_mean, group_sizes, unique_vals


def find_subspace(centroids, grand_mean):
    """SVD on centered centroid matrix to find concept subspace.

    Args:
        centroids: (m, d) centroid matrix
        grand_mean: (d,) grand mean

    Returns:
        eigenvalues: (m,) eigenvalues of between-class scatter
        Vt: (m, d) right singular vectors (rows = basis directions)
        explained_variance: (m,) fraction of variance per component
    """
    m = centroids.shape[0]
    M_c = (centroids - grand_mean) / np.sqrt(m)
    U, S, Vt = np.linalg.svd(M_c, full_matrices=False)

    eigenvalues = S ** 2
    total = eigenvalues.sum()
    explained_variance = eigenvalues / total if total > 1e-12 else np.zeros_like(eigenvalues)

    return eigenvalues, Vt, explained_variance


def permutation_null(acts, concept_values, n_perms=N_PERMUTATIONS, rng=None):
    """Compute null eigenvalue distribution by shuffling concept labels.

    Args:
        acts: (N, d) activation matrix (valid rows only)
        concept_values: (N,) concept labels (no NaN)
        n_perms: number of permutations
        rng: numpy RandomState

    Returns:
        null_eigenvalues: (n_perms, m-1) null eigenvalue matrix
    """
    if rng is None:
        rng = np.random.RandomState(42)

    valid = ~np.isnan(concept_values)
    acts_v = acts[valid]
    vals_v = concept_values[valid].copy()

    unique_vals = np.unique(vals_v)
    m = len(unique_vals)
    d = acts_v.shape[1]
    n = len(vals_v)

    null_eigenvalues = np.zeros((n_perms, m - 1))

    # Pre-compute integer indices once, then permute those directly
    # (avoids 4M Python dict lookups: N × n_perms)
    val_to_idx = {v: i for i, v in enumerate(unique_vals)}
    base_indices = np.array([val_to_idx[v] for v in vals_v])
    col_idx = np.arange(n)

    for i in range(n_perms):
        shuffled_indices = rng.permutation(base_indices)
        one_hot = np.zeros((m, n), dtype=np.float32)
        one_hot[shuffled_indices, col_idx] = 1.0
        counts = one_hot.sum(axis=1, keepdims=True)
        centroids = (one_hot @ acts_v) / counts
        grand_mean = centroids.mean(axis=0)

        M_c = (centroids - grand_mean) / np.sqrt(m)
        S = np.linalg.svd(M_c, compute_uv=False)
        eigs = S ** 2
        null_eigenvalues[i] = eigs[:m - 1]

    return null_eigenvalues


def find_dimensionality(eigenvalues, null_eigenvalues=None):
    """Determine subspace dimensionality via three methods.

    Returns dict with dim_cumvar, dim_ratio, dim_perm, dim_consensus.
    """
    m = len(eigenvalues)
    total = eigenvalues.sum()

    # Method 1: Cumulative variance >= 95%
    if total > 1e-12:
        cumvar = np.cumsum(eigenvalues) / total
        dim_cumvar = int(np.searchsorted(cumvar, CUMVAR_THRESHOLD)) + 1
        dim_cumvar = min(dim_cumvar, m)
    else:
        cumvar = np.zeros(m)
        dim_cumvar = 0

    # Method 2: Ratio test — first j where λ_j / λ_{j+1} > threshold
    # Note: last eigenvalue is structurally ~0 (rank deficiency from centering),
    # so only search up to m-2 to avoid false cliff at the structural zero.
    dim_ratio = m - 1  # default if no cliff found (max possible rank)
    if m > 2:
        for j in range(m - 2):
            denom = eigenvalues[j + 1] if eigenvalues[j + 1] > 1e-12 else 1e-12
            if eigenvalues[j] / denom > RATIO_THRESHOLD:
                dim_ratio = j + 1
                break

    # Method 3: Permutation null with sequential stopping
    dim_perm = 0
    if null_eigenvalues is not None and len(null_eigenvalues) > 0:
        threshold_99 = np.percentile(null_eigenvalues, 100 * (1 - PERM_ALPHA), axis=0)
        for j in range(min(m, len(threshold_99))):
            if eigenvalues[j] > threshold_99[j]:
                dim_perm += 1
            else:
                break  # sequential stopping

    # Consensus: median of three methods (or two if no null)
    if null_eigenvalues is not None:
        dim_consensus = int(np.median([dim_cumvar, dim_ratio, dim_perm]))
    else:
        dim_consensus = int(np.median([dim_cumvar, dim_ratio]))
    dim_consensus = max(dim_consensus, 1)  # at least 1

    return {
        "dim_cumvar": int(dim_cumvar),
        "dim_ratio": int(dim_ratio),
        "dim_perm": int(dim_perm),
        "dim_consensus": int(dim_consensus),
        "eigenvalues": eigenvalues.tolist(),
        "explained_variance": (eigenvalues / total if total > 1e-12
                               else np.zeros(m)).tolist(),
        "cumulative_variance": cumvar.tolist() if total > 1e-12 else [],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def cross_validate_subspace(acts, concept_values, dim, n_splits=N_CV_SPLITS):
    """Cross-validate subspace via centroid distance preservation.

    Returns dict with mean/std of Pearson correlation between full-space
    and subspace pairwise centroid distances.
    """
    valid = ~np.isnan(concept_values)
    acts_v = acts[valid]
    vals_v = concept_values[valid].astype(int)

    unique_vals = np.unique(vals_v)
    if len(unique_vals) < 3 or dim < 1:
        return {"mean_corr": np.nan, "std_corr": np.nan, "per_fold": []}

    correlations = []
    try:
        splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2,
                                         random_state=42)
        for train_idx, test_idx in splitter.split(acts_v, vals_v):
            # Learn subspace on train
            train_centroids, train_gm, _, _ = compute_centroids(
                acts_v[train_idx], vals_v[train_idx].astype(np.float64))
            _, Vt_train, _ = find_subspace(train_centroids, train_gm)
            basis = Vt_train[:dim]  # (dim, d)

            # Test centroids
            test_centroids, _, _, _ = compute_centroids(
                acts_v[test_idx], vals_v[test_idx].astype(np.float64))

            if test_centroids.shape[0] < 2:
                continue

            # Pairwise distances in full space and subspace
            d_full = pdist(test_centroids)
            d_sub = pdist(test_centroids @ basis.T)

            if len(d_full) >= 2 and np.std(d_full) > 1e-12:
                corr, _ = pearsonr(d_full, d_sub)
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
# CROSS-LAYER ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

def compute_principal_angles(basis_a, basis_b):
    """Compute principal angles between two subspaces in degrees.

    Args:
        basis_a: (d_a, D) orthonormal basis
        basis_b: (d_b, D) orthonormal basis

    Returns:
        angles_deg: (min(d_a, d_b),) principal angles in degrees
    """
    if basis_a.shape[0] == 0 or basis_b.shape[0] == 0:
        return np.array([])
    M = basis_a @ basis_b.T  # (d_a, d_b)
    S = np.linalg.svd(M, compute_uv=False)
    S = np.clip(S, -1.0, 1.0)
    return np.degrees(np.arccos(S))


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE CONCEPT PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_single_concept(acts, concept_values, concept_name, tier, output_dir,
                       n_perms, skip_null, rng, logger):
    """Run full subspace identification for one concept.

    Returns result dict or None if concept is not viable.
    """
    # Resume logic: check for completed metadata
    meta_path = output_dir / "metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                cached = json.load(f)
            logger.debug(f"  [{concept_name}] cached (dim={cached.get('dim_consensus', '?')})")
            return cached
        except (json.JSONDecodeError, KeyError):
            pass  # corrupted cache — recompute

    # Filter concept values
    filtered, filter_meta = filter_concept_values(concept_values)
    if filtered is None:
        logger.debug(f"  [{concept_name}] skipped: {filter_meta['surviving_groups']} groups "
                     f"(need ≥2, had {filter_meta['original_unique']} unique)")
        return None

    # Compute centroids
    centroids, grand_mean, group_sizes, unique_vals = compute_centroids(acts, filtered)
    m = centroids.shape[0]

    # SVD
    eigenvalues, Vt, explained_variance = find_subspace(centroids, grand_mean)

    # Permutation null
    null_eigenvalues = None
    if not skip_null and m > 1:
        null_path = output_dir / "null_eigenvalues.npy"
        if null_path.exists():
            null_eigenvalues = np.load(null_path)
            if null_eigenvalues.shape != (n_perms, m - 1):
                null_eigenvalues = None  # shape mismatch — recompute
        if null_eigenvalues is None:
            null_eigenvalues = permutation_null(acts, filtered, n_perms, rng)
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(null_path, null_eigenvalues)

    # Dimensionality
    dim_results = find_dimensionality(eigenvalues, null_eigenvalues)
    dim = dim_results["dim_consensus"]

    # Cross-validation
    cv_results = cross_validate_subspace(acts, filtered, dim)

    # Save artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "eigenvalues.npy", eigenvalues)
    np.save(output_dir / "basis.npy", Vt[:dim])

    # Save projected activations: ALL samples in population (not just filtered).
    # Basis is learned from filtered subset, but any activation can be projected.
    # Downstream Fourier screening / GP geometry needs every sample.
    valid = ~np.isnan(filtered)
    centered = acts - grand_mean
    projected_all = centered @ Vt[:dim].T  # (N_total, dim)
    np.save(output_dir / "projected_all.npy", projected_all)

    # Build result
    result = {
        "concept": concept_name,
        "tier": tier,
        "n_groups": m,
        "n_samples": int(valid.sum()),
        "group_sizes": {str(int(k)): v for k, v in group_sizes.items()},
        "filter_meta": filter_meta,
        **dim_results,
        "cv_mean_corr": cv_results["mean_corr"],
        "cv_std_corr": cv_results["std_corr"],
    }

    # Atomic write of metadata
    fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix=".json")
    os.close(fd)
    with open(tmp_path, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp_path, meta_path)

    logger.debug(f"  [{concept_name}] dim={dim} (cv={dim_results['dim_cumvar']}/"
                 f"rt={dim_results['dim_ratio']}/pm={dim_results['dim_perm']}) "
                 f"cv_corr={cv_results['mean_corr']:.3f}" if not np.isnan(cv_results['mean_corr'])
                 else f"  [{concept_name}] dim={dim}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def process_level_layer(level, layer, paths, n_perms, skip_null, max_tier, logger):
    """Process all populations and concepts for one (level, layer) pair."""
    results = []
    rng = np.random.RandomState(42 + level * 100 + layer)

    df = load_coloring_df(level, paths["coloring_dir"])
    populations = get_populations(df)

    acts_raw = load_activations(level, layer, paths["act_dir"])

    # Residualize product once for this (level, layer)
    resid_path = paths["residualized_dir"] / f"level{level}_layer{layer}.npy"
    acts_resid = residualize_product(acts_raw, df["product"].values, resid_path)

    for pop_name, pop_df in populations.items():
        pop_idx = pop_df.index.values
        acts_pop_raw = acts_raw[pop_idx]
        acts_pop_resid = acts_resid[pop_idx]

        concepts = get_concept_registry(level, pop_df, pop_name)
        if max_tier is not None:
            concepts = [c for c in concepts if c["tier"] <= max_tier]

        for concept in concepts:
            c_name = concept["name"]
            c_col = concept["column"]
            c_tier = concept["tier"]
            c_pre = concept["preprocess"]

            # Extract and preprocess concept values
            raw_values = pop_df[c_col].values
            values = preprocess_concept(raw_values, c_pre)

            # Choose activations: raw for product, residualized for everything else
            if c_name == "product_binned":
                acts_for_concept = acts_pop_raw
            else:
                acts_for_concept = acts_pop_resid

            output_dir = (paths["subspaces_dir"] / f"L{level}" /
                         f"layer_{layer:02d}" / pop_name / c_name)

            result = run_single_concept(
                acts_for_concept, values, c_name, c_tier,
                output_dir, n_perms, skip_null, rng, logger)

            if result is not None:
                result["level"] = level
                result["layer"] = layer
                result["population"] = pop_name
                results.append(result)

    return results


def run_all(paths, levels, layers, n_perms, skip_null, n_jobs, max_tier, logger):
    """Run Phase C across all specified levels and layers."""
    all_results = []

    pairs = [(level, layer) for level in levels for layer in layers]
    logger.info(f"Processing {len(pairs)} (level, layer) pairs across "
                f"{len(levels)} levels, {len(layers)} layers")

    if n_jobs > 1 and _JOBLIB_AVAILABLE:
        logger.info(f"Parallel mode: {n_jobs} jobs")
        batch_results = Parallel(n_jobs=n_jobs)(
            delayed(process_level_layer)(level, layer, paths, n_perms, skip_null, max_tier, logger)
            for level, layer in tqdm(pairs, desc="Phase C")
        )
        for batch in batch_results:
            all_results.extend(batch)
    else:
        for level, layer in tqdm(pairs, desc="Phase C"):
            batch = process_level_layer(level, layer, paths, n_perms, skip_null, max_tier, logger)
            all_results.extend(batch)

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-LAYER ALIGNMENT (post-processing)
# ═══════════════════════════════════════════════════════════════════════════════

def run_cross_layer_alignment(results_df, paths, logger):
    """Compute principal angles between adjacent layers for each concept."""
    logger.info("Computing cross-layer alignment...")
    rows = []
    adjacent = list(zip(LAYERS[:-1], LAYERS[1:]))

    groups = results_df.groupby(["level", "population", "concept"])
    for (level, pop, concept), group in groups:
        layer_to_dim = {r["layer"]: r["dim_consensus"] for _, r in group.iterrows()}

        for layer_a, layer_b in adjacent:
            if layer_a not in layer_to_dim or layer_b not in layer_to_dim:
                continue
            dim_a = layer_to_dim[layer_a]
            dim_b = layer_to_dim[layer_b]
            if dim_a < 1 or dim_b < 1:
                continue

            basis_a_path = (paths["subspaces_dir"] / f"L{level}" /
                           f"layer_{layer_a:02d}" / pop / concept / "basis.npy")
            basis_b_path = (paths["subspaces_dir"] / f"L{level}" /
                           f"layer_{layer_b:02d}" / pop / concept / "basis.npy")

            if not basis_a_path.exists() or not basis_b_path.exists():
                continue

            basis_a = np.load(basis_a_path)
            basis_b = np.load(basis_b_path)
            angles = compute_principal_angles(basis_a, basis_b)

            row = {
                "level": level, "population": pop, "concept": concept,
                "layer_a": layer_a, "layer_b": layer_b,
                "dim_a": dim_a, "dim_b": dim_b,
                "n_angles": len(angles),
            }
            for k, angle in enumerate(angles[:5]):
                row[f"angle_{k+1}"] = float(angle)
            rows.append(row)

    alignment_df = pd.DataFrame(rows)
    return alignment_df


# ═══════════════════════════════════════════════════════════════════════════════
# CORRECT/WRONG DIVERGENCE (post-processing)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_correct_wrong_divergence(results_df, paths, logger):
    """Compare subspaces between correct and wrong populations."""
    logger.info("Computing correct/wrong divergence...")
    rows = []

    for (level, layer, concept), group in results_df.groupby(
            ["level", "layer", "concept"]):
        pop_dims = {r["population"]: r for _, r in group.iterrows()}
        if "correct" not in pop_dims or "wrong" not in pop_dims:
            continue

        r_corr = pop_dims["correct"]
        r_wrong = pop_dims["wrong"]

        # Principal angle between correct and wrong subspaces
        basis_corr_path = (paths["subspaces_dir"] / f"L{level}" /
                          f"layer_{layer:02d}" / "correct" / concept / "basis.npy")
        basis_wrong_path = (paths["subspaces_dir"] / f"L{level}" /
                           f"layer_{layer:02d}" / "wrong" / concept / "basis.npy")

        angle_1 = np.nan
        if basis_corr_path.exists() and basis_wrong_path.exists():
            basis_corr = np.load(basis_corr_path)
            basis_wrong = np.load(basis_wrong_path)
            if basis_corr.shape[0] > 0 and basis_wrong.shape[0] > 0:
                angles = compute_principal_angles(basis_corr, basis_wrong)
                if len(angles) > 0:
                    angle_1 = float(angles[0])

        rows.append({
            "level": level, "layer": layer, "concept": concept,
            "dim_correct": int(r_corr["dim_consensus"]),
            "dim_wrong": int(r_wrong["dim_consensus"]),
            "dim_all": int(pop_dims["all"]["dim_consensus"]) if "all" in pop_dims else np.nan,
            "dim_perm_correct": int(r_corr["dim_perm"]),
            "dim_perm_wrong": int(r_wrong["dim_perm"]),
            "angle_1_cw": angle_1,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_eigenvalue_spectrum(eigenvalues, null_eigenvalues, dim_results,
                             title, save_path):
    """Scree plot with permutation null overlay."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    m = len(eigenvalues)
    x = np.arange(1, m + 1)

    # Real eigenvalues (clip structural zero for log scale)
    eig_plot = np.maximum(eigenvalues, 1e-15)
    ax.plot(x, eig_plot, "o-", color="C0", linewidth=2, markersize=6,
            label="Observed", zorder=5)

    # Null overlay
    if null_eigenvalues is not None and len(null_eigenvalues) > 0:
        n_null = min(m - 1, null_eigenvalues.shape[1])
        null_50 = np.percentile(null_eigenvalues[:, :n_null], 50, axis=0)
        null_99 = np.percentile(null_eigenvalues[:, :n_null],
                                100 * (1 - PERM_ALPHA), axis=0)
        x_null = np.arange(1, n_null + 1)
        ax.fill_between(x_null, 0, null_99, alpha=0.2, color="gray",
                        label=f"Null {100*(1-PERM_ALPHA):.0f}th pctl")
        ax.plot(x_null, null_50, "--", color="gray", alpha=0.6,
                label="Null median")

    # Dimensionality markers
    colors = {"dim_cumvar": "C1", "dim_ratio": "C2", "dim_perm": "C3"}
    labels = {"dim_cumvar": f"CumVar≥95% (d={dim_results['dim_cumvar']})",
              "dim_ratio": f"Ratio>{RATIO_THRESHOLD} (d={dim_results['dim_ratio']})",
              "dim_perm": f"Perm null (d={dim_results['dim_perm']})"}
    for key in ["dim_cumvar", "dim_ratio", "dim_perm"]:
        d = dim_results[key]
        if 0 < d <= m:
            ax.axvline(d + 0.5, color=colors[key], linestyle=":", alpha=0.7,
                      label=labels[key])

    ax.set_xlabel("Component index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.set_xlim(0.5, m + 0.5)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dimensionality_heatmap(results_df, level, pop, save_path):
    """Heatmap: concepts × layers, cells = consensus dimensionality."""
    subset = results_df[(results_df["level"] == level) &
                        (results_df["population"] == pop)]
    if subset.empty:
        return

    pivot = subset.pivot_table(index="concept", columns="layer",
                               values="dim_consensus", aggfunc="first")
    # Sort by tier then name
    tier_map = subset.drop_duplicates("concept").set_index("concept")["tier"]
    sort_key = pivot.index.map(lambda c: (tier_map.get(c, 99), c))
    pivot = pivot.iloc[sort_key.argsort()]

    fig, ax = plt.subplots(1, 1, figsize=(12, max(4, 0.4 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis",
                   interpolation="nearest")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Concept")
    ax.set_title(f"Subspace Dimensionality — L{level} {pop}")

    # Annotate cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{int(val)}", ha="center", va="center",
                       fontsize=8, color="white" if val > pivot.values[~np.isnan(pivot.values)].mean() else "black")

    fig.colorbar(im, ax=ax, label="Dimensionality", shrink=0.8)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cross_layer_trajectory(alignment_df, concept, level, pop, save_path):
    """Plot principal angles across adjacent layer pairs."""
    subset = alignment_df[(alignment_df["concept"] == concept) &
                          (alignment_df["level"] == level) &
                          (alignment_df["population"] == pop)]
    if subset.empty:
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x_labels = [f"{r['layer_a']}→{r['layer_b']}" for _, r in subset.iterrows()]
    x = range(len(x_labels))

    # Plot up to 3 principal angles
    for k in range(1, 4):
        col = f"angle_{k}"
        if col in subset.columns:
            vals = subset[col].values
            valid = ~np.isnan(vals)
            if valid.any():
                ax.plot(np.array(list(x))[valid], vals[valid], "o-",
                       label=f"Angle {k}", markersize=5)

    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels, rotation=45, fontsize=9)
    ax.set_xlabel("Layer transition")
    ax.set_ylabel("Principal angle (degrees)")
    ax.set_title(f"Subspace alignment — {concept} L{level} {pop}")
    ax.set_ylim(-5, 95)
    ax.axhline(45, color="gray", linestyle="--", alpha=0.3)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correct_wrong_bars(divergence_df, concept, level, save_path):
    """Bar chart comparing dimensionality for correct vs wrong."""
    subset = divergence_df[(divergence_df["concept"] == concept) &
                           (divergence_df["level"] == level)]
    if subset.empty:
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    layers = subset["layer"].values
    x = np.arange(len(layers))
    w = 0.35

    ax.bar(x - w/2, subset["dim_correct"].values, w, label="Correct", color="C0")
    ax.bar(x + w/2, subset["dim_wrong"].values, w, label="Wrong", color="C3")

    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Consensus dimensionality")
    ax.set_title(f"Correct vs Wrong — {concept} L{level}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(results_df, alignment_df, divergence_df, paths, logger):
    """Generate all Phase C plots."""
    plot_dir = paths["phase_c_plots"]
    n_plots = 0

    # 1. Eigenvalue spectra — Tier 1+2 at key layers
    logger.info("Generating eigenvalue spectrum plots...")
    for _, row in results_df[
            (results_df["tier"].isin([1, 2])) &
            (results_df["layer"].isin(PLOT_LAYERS))].iterrows():
        level, layer, pop, concept = row["level"], row["layer"], row["population"], row["concept"]
        eig_path = (paths["subspaces_dir"] / f"L{level}" /
                   f"layer_{layer:02d}" / pop / concept / "eigenvalues.npy")
        null_path = (paths["subspaces_dir"] / f"L{level}" /
                    f"layer_{layer:02d}" / pop / concept / "null_eigenvalues.npy")
        if not eig_path.exists():
            continue
        eigenvalues = np.load(eig_path)
        null_eig = np.load(null_path) if null_path.exists() else None
        dim_results = {k: row[k] for k in ["dim_cumvar", "dim_ratio", "dim_perm"]}

        save = plot_dir / "eigenvalue_spectra" / f"L{level}_{pop}_{concept}_layer{layer:02d}.png"
        plot_eigenvalue_spectrum(eigenvalues, null_eig, dim_results,
                                f"{concept} — L{level} layer {layer} ({pop})", save)
        n_plots += 1

    # 2. Dimensionality heatmaps
    logger.info("Generating dimensionality heatmaps...")
    for level in results_df["level"].unique():
        for pop in results_df[results_df["level"] == level]["population"].unique():
            save = plot_dir / "dimensionality_heatmaps" / f"L{level}_{pop}.png"
            plot_dimensionality_heatmap(results_df, level, pop, save)
            n_plots += 1

    # 3. Cross-layer trajectories
    if not alignment_df.empty:
        logger.info("Generating cross-layer trajectory plots...")
        for (concept, level, pop), _ in alignment_df.groupby(
                ["concept", "level", "population"]):
            save = plot_dir / "cross_layer_trajectories" / f"L{level}_{pop}_{concept}.png"
            plot_cross_layer_trajectory(alignment_df, concept, level, pop, save)
            n_plots += 1

    # 4. Correct/wrong comparison
    if not divergence_df.empty:
        logger.info("Generating correct/wrong comparison plots...")
        for (concept, level), _ in divergence_df.groupby(["concept", "level"]):
            save = plot_dir / "correct_wrong_comparison" / f"L{level}_{concept}.png"
            plot_correct_wrong_bars(divergence_df, concept, level, save)
            n_plots += 1

    logger.info(f"Generated {n_plots} plots")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLES
# ═══════════════════════════════════════════════════════════════════════════════

def generate_summary(results_df, alignment_df, divergence_df, paths, logger):
    """Generate summary CSV files."""
    summary_dir = paths["summary_dir"]
    summary_dir.mkdir(parents=True, exist_ok=True)

    # 1. Master results table
    results_df.to_csv(summary_dir / "phase_c_results.csv", index=False)
    logger.info(f"Saved phase_c_results.csv ({len(results_df)} rows)")

    # 2. Significance table: concept × layer grid of dim_perm
    for level in results_df["level"].unique():
        for pop in results_df[results_df["level"] == level]["population"].unique():
            subset = results_df[(results_df["level"] == level) &
                                (results_df["population"] == pop)]
            if subset.empty:
                continue
            pivot = subset.pivot_table(index="concept", columns="layer",
                                       values="dim_perm", aggfunc="first")
            pivot.to_csv(summary_dir / f"significance_L{level}_{pop}.csv")

    # 3. Correct/wrong divergence
    if not divergence_df.empty:
        divergence_df.to_csv(summary_dir / "correct_wrong_divergence.csv", index=False)
        logger.info(f"Saved correct_wrong_divergence.csv ({len(divergence_df)} rows)")

    # 4. Alignment results
    if not alignment_df.empty:
        alignment_df.to_csv(summary_dir / "alignment_results.csv", index=False)
        logger.info(f"Saved alignment_results.csv ({len(alignment_df)} rows)")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI & MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase C — Concept Subspace Identification via "
                    "Conditional Covariance + SVD")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--pilot", action="store_true",
                        help="Smoke test: L3, layer 16, all population, Tier 1 only")
    parser.add_argument("--level", type=int, nargs="*",
                        help="Specific levels to run (default: all)")
    parser.add_argument("--layer", type=int, nargs="*",
                        help="Specific layers to run (default: all)")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel jobs for (level, layer) pairs")
    parser.add_argument("--n-perms", type=int, default=N_PERMUTATIONS,
                        help=f"Permutation null iterations (default: {N_PERMUTATIONS})")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--skip-null", action="store_true",
                        help="Skip permutation null (fast debug mode)")
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    cfg = load_config(args.config)
    paths = derive_paths(cfg)
    logger = setup_logging(paths["workspace"])

    # Banner
    logger.info("=" * 70)
    logger.info("Phase C — Concept Subspace Identification")
    logger.info("=" * 70)

    # Determine scope
    max_tier = None  # no tier filter by default
    if args.pilot:
        levels = [3]
        layers = [16]
        max_tier = 1
        logger.info("PILOT MODE: L3, layer 16, Tier 1 only")
    else:
        levels = args.level if args.level else LEVELS
        layers = args.layer if args.layer else LAYERS

    n_perms = args.n_perms
    logger.info(f"Levels: {levels}")
    logger.info(f"Layers: {layers}")
    logger.info(f"Max tier: {max_tier or 'all'}")
    logger.info(f"Permutations: {n_perms} {'(SKIPPED)' if args.skip_null else ''}")
    logger.info(f"Jobs: {args.n_jobs}")

    # Pre-flight checks: verify input data exists
    missing = []
    for level in levels:
        pkl_path = paths["coloring_dir"] / f"L{level}_coloring.pkl"
        if not pkl_path.exists():
            missing.append(str(pkl_path))
        for layer in layers:
            act_path = paths["act_dir"] / f"level{level}_layer{layer}.npy"
            if not act_path.exists():
                missing.append(str(act_path))
    if missing:
        logger.error(f"Missing {len(missing)} input files. First 5:")
        for p in missing[:5]:
            logger.error(f"  {p}")
        logger.error("Run Phase A first to generate coloring DFs, "
                      "and ensure activations exist.")
        return

    # Create output directories
    for key in ["phase_c_data", "residualized_dir", "subspaces_dir",
                "summary_dir", "phase_c_plots"]:
        paths[key].mkdir(parents=True, exist_ok=True)

    # ── Core computation ──────────────────────────────────────────────────
    logger.info("")
    logger.info("Step 1: Subspace identification")
    logger.info("-" * 40)
    t1 = time.time()

    all_results = run_all(paths, levels, layers, n_perms,
                          args.skip_null, args.n_jobs, max_tier, logger)

    logger.info(f"Step 1 complete: {len(all_results)} concept subspaces "
                f"({time.time()-t1:.1f}s)")

    if not all_results:
        logger.warning("No results produced — check data paths and concept availability")
        return

    results_df = pd.DataFrame(all_results)

    # ── Cross-layer alignment ─────────────────────────────────────────────
    logger.info("")
    logger.info("Step 2: Cross-layer alignment")
    logger.info("-" * 40)
    t2 = time.time()

    alignment_df = run_cross_layer_alignment(results_df, paths, logger)
    logger.info(f"Step 2 complete: {len(alignment_df)} alignment entries "
                f"({time.time()-t2:.1f}s)")

    # ── Correct/wrong divergence ──────────────────────────────────────────
    logger.info("")
    logger.info("Step 3: Correct/wrong divergence")
    logger.info("-" * 40)
    t3 = time.time()

    divergence_df = compute_correct_wrong_divergence(results_df, paths, logger)
    logger.info(f"Step 3 complete: {len(divergence_df)} divergence entries "
                f"({time.time()-t3:.1f}s)")

    # ── Plots ─────────────────────────────────────────────────────────────
    if not args.skip_plots:
        logger.info("")
        logger.info("Step 4: Plot generation")
        logger.info("-" * 40)
        t4 = time.time()

        generate_all_plots(results_df, alignment_df, divergence_df, paths, logger)
        logger.info(f"Step 4 complete ({time.time()-t4:.1f}s)")

    # ── Summary tables ────────────────────────────────────────────────────
    logger.info("")
    logger.info("Step 5: Summary tables")
    logger.info("-" * 40)

    generate_summary(results_df, alignment_df, divergence_df, paths, logger)

    # ── Done ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"Phase C complete: {len(all_results)} subspaces, "
                f"{elapsed:.0f}s ({elapsed/60:.1f} min)")
    logger.info("=" * 70)

    # Print key stats
    if "dim_perm" in results_df.columns:
        sig = results_df[results_df["dim_perm"] > 0]
        logger.info(f"Significant subspaces (perm null): {len(sig)} / {len(results_df)}")
    if not divergence_df.empty and "angle_1_cw" in divergence_df.columns:
        strong_div = divergence_df[divergence_df["angle_1_cw"] < 60]
        logger.info(f"Strong correct/wrong divergence (<60°): {len(strong_div)} entries")


if __name__ == "__main__":
    main()
