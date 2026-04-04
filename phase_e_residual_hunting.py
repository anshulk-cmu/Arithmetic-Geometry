#!/usr/bin/env python3
"""Phase E — Residual Hunting via PCA on Concept-Projected Residuals.

Projects out all known concept subspaces (Phase C + D merged bases),
then runs PCA on the residual to detect unknown structure. Uses the
Marchenko-Pastur distribution to separate signal from noise eigenvalues.

Key design:
  - Union subspace built per (level, layer, population) slice
  - SVD orthonormalization (order-independent, numerically stable)
  - Product β direction included in union for completeness
  - σ² estimated via trace normalization (no full eigendecomposition needed)
  - Correlation sweep uses both Spearman and Pearson
  - Resume logic via metadata.json per slice

Outputs:
  /data/.../phase_e/
    union_bases/L{level}/layer_{layer:02d}/{pop}/
      union_basis.npy            (k, D) orthonormalized union
      metadata.json
    pca/L{level}/layer_{layer:02d}/{pop}/
      eigenvalues.npy            top eigenvalues of residual covariance
      eigenvectors.npy           (n_components, D) PCA directions
      metadata.json
    correlations/L{level}/layer_{layer:02d}/{pop}/
      correlation_sweep.csv      direction × metadata correlations
    summary/
      phase_e_results.csv
      eigenvalue_cliff_summary.csv
      union_rank_by_layer.csv
      variance_explained.csv
      total_carry_sum_diagnostic.csv
      top_eigenvalues_all_slices.csv
  plots/phase_e/
    eigenvalue_spectra/
    mp_heatmaps/
    variance_explained_heatmaps/
    union_rank_trajectories/
    correlation_heatmaps/

Usage:
  python phase_e_residual_hunting.py --config config.yaml    # Full run
  python phase_e_residual_hunting.py --pilot                  # Smoke test
  python phase_e_residual_hunting.py --level 5 --layer 16     # Specific slice
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
from scipy.stats import spearmanr, pearsonr
from sklearn.utils.extmath import randomized_svd
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

LEVELS = [2, 3, 4, 5]  # L1 excluded (N=64, useless for PCA in ~4000D)
LAYERS = [4, 6, 8, 12, 16, 20, 24, 28, 31]
MIN_POPULATION = 30
MIN_GROUP_SIZE = 20
N_PCA_COMPONENTS = 500
SVD_TOLERANCE_FACTOR = 1e-10
PLOT_LEVELS = [3, 4, 5]
PLOT_LAYERS = [4, 16, 31]
CORR_FLAG_THRESHOLD = 0.15  # Spearman |ρ| to flag for investigation

# L5 carry binning thresholds (must match Phase C/D)
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
        "pca_dir": dr / "phase_e" / "pca",
        "correlations_dir": dr / "phase_e" / "correlations",
        "summary_dir": dr / "phase_e" / "summary",
        "phase_e_plots": ws / "plots" / "phase_e",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(workspace):
    log_dir = workspace / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase_e")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fh = RotatingFileHandler(log_dir / "phase_e_residual.log",
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


def load_raw_activations(level, layer, act_dir):
    return np.load(act_dir / f"level{level}_layer{layer}.npy")


def get_populations(df):
    pops = {"all": df}
    correct_df = df[df["correct"] == True]
    wrong_df = df[df["correct"] == False]
    if len(correct_df) >= MIN_POPULATION:
        pops["correct"] = correct_df
    if len(wrong_df) >= MIN_POPULATION:
        pops["wrong"] = wrong_df
    return pops


# ═══════════════════════════════════════════════════════════════════════════════
# CONCEPT REGISTRY (replicated from Phase C/D for standalone execution)
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
# CORE: BUILD UNION SUBSPACE
# ═══════════════════════════════════════════════════════════════════════════════

def load_merged_basis(level, layer, pop, concept_name, paths):
    """Load Phase D merged basis for one concept. Returns (dim, 4096) or (0, 4096)."""
    concept_dir = (paths["phase_d_subspaces"] / f"L{level}" /
                   f"layer_{layer:02d}" / pop / concept_name)
    basis_path = concept_dir / "merged_basis.npy"
    if not basis_path.exists():
        return np.empty((0, 4096))
    basis = np.load(basis_path)
    return basis


def compute_product_beta(level, layer, df, paths):
    """Recompute the product residualization direction β."""
    raw_acts = load_raw_activations(level, layer, paths["act_dir"])
    product_values = df["product"].values.astype(np.float64)

    X_c = raw_acts.astype(np.float64) - raw_acts.mean(axis=0, dtype=np.float64)
    p_c = product_values - product_values.mean()
    p_dot_p = p_c @ p_c
    if p_dot_p < 1e-12:
        return np.zeros(raw_acts.shape[1], dtype=np.float64)
    beta = X_c.T @ p_c / p_dot_p
    beta_hat = beta / np.linalg.norm(beta)
    del raw_acts, X_c
    return beta_hat.astype(np.float32)


def build_union_basis(level, layer, pop, df, concept_list, paths, logger):
    """Stack all merged bases + β, orthonormalize via SVD.

    Returns:
        V_all: (k, 4096) orthonormalized union basis
        metadata: dict with stacking details
    """
    bases = []
    concepts_included = []
    concepts_empty = []
    dim_per_concept = {}

    for concept_dict in concept_list:
        name = concept_dict["name"]
        basis = load_merged_basis(level, layer, pop, name, paths)
        dim_per_concept[name] = basis.shape[0]
        if basis.shape[0] > 0:
            bases.append(basis)
            concepts_included.append(name)
        else:
            concepts_empty.append(name)

    # Add product β direction
    beta_hat = compute_product_beta(level, layer, df, paths)
    bases.append(beta_hat.reshape(1, -1))

    stacked = np.vstack(bases)  # (total_stacked, 4096)
    stacked_dim = stacked.shape[0]

    # SVD orthonormalization
    U, S, Vt = np.linalg.svd(stacked, full_matrices=False)
    tol = SVD_TOLERANCE_FACTOR * S[0] if len(S) > 0 else 1e-10
    keep = S > tol
    V_all = Vt[keep]
    k = V_all.shape[0]

    logger.debug(f"    Union: {stacked_dim} stacked → rank {k} "
                 f"({len(concepts_included)} concepts + β, "
                 f"{len(concepts_empty)} empty)")

    # total_carry_sum diagnostic: recompute k without it
    k_without_tcs = k
    if "total_carry_sum" in concepts_included:
        bases_no_tcs = []
        for concept_dict in concept_list:
            name = concept_dict["name"]
            if name == "total_carry_sum":
                continue
            basis = load_merged_basis(level, layer, pop, name, paths)
            if basis.shape[0] > 0:
                bases_no_tcs.append(basis)
        bases_no_tcs.append(beta_hat.reshape(1, -1))
        stacked_no_tcs = np.vstack(bases_no_tcs)
        _, S_no_tcs, Vt_no_tcs = np.linalg.svd(stacked_no_tcs,
                                                 full_matrices=False)
        tol_no_tcs = SVD_TOLERANCE_FACTOR * S_no_tcs[0]
        k_without_tcs = int(np.sum(S_no_tcs > tol_no_tcs))
        logger.debug(f"    total_carry_sum diagnostic: k={k} → "
                     f"k_without_tcs={k_without_tcs} "
                     f"(Δ={k - k_without_tcs})")

    meta = {
        "n_concepts_total": len(concept_list),
        "n_concepts_nonzero": len(concepts_included),
        "concepts_included": concepts_included,
        "concepts_empty": concepts_empty,
        "dim_per_concept": dim_per_concept,
        "stacked_dim_before_svd": stacked_dim,
        "union_rank_k": k,
        "k_without_total_carry_sum": k_without_tcs,
        "svd_tolerance": float(tol),
        "product_beta_included": True,
    }
    return V_all, meta


# ═══════════════════════════════════════════════════════════════════════════════
# CORE: PROJECT AND PCA
# ═══════════════════════════════════════════════════════════════════════════════

def compute_residual(X, V_all):
    """Project out V_all from X. Returns residual and variance explained."""
    if _CUPY_AVAILABLE:
        X_g = cp.asarray(X)
        V_g = cp.asarray(V_all)
        coords = X_g @ V_g.T
        X_proj = coords @ V_g
        X_resid = X_g - X_proj
        var_orig = float(cp.sum(X_g ** 2))
        var_resid = float(cp.sum(X_resid ** 2))
        result = cp.asnumpy(X_resid)
        del X_g, V_g, coords, X_proj, X_resid
        cp.get_default_memory_pool().free_all_blocks()
    else:
        coords = X @ V_all.T
        X_proj = coords @ V_all
        result = X - X_proj
        var_orig = float(np.sum(X ** 2))
        var_resid = float(np.sum(result ** 2))

    var_explained = 1.0 - (var_resid / var_orig) if var_orig > 0 else 0.0
    return result, var_explained, var_orig, var_resid


def pca_with_mp(X_residual, n_components, d_residual, logger):
    """PCA on residual + Marchenko-Pastur comparison.

    Args:
        X_residual: (N, D) residual activations
        n_components: max PCA components to compute
        d_residual: effective dimensionality (D - k, where k = union rank)
        logger: logging instance

    Returns:
        eigenvalues: (n_components,)
        eigenvectors: (n_components, D)
        mp_info: dict with σ², λ_max, n_above, etc.
    """
    N, D = X_residual.shape
    X_centered = X_residual - X_residual.mean(axis=0)

    n_comp = min(n_components, N - 1, D - 1)
    if n_comp < 1:
        logger.warning(f"    PCA skipped: n_comp={n_comp} (N={N}, D={D})")
        return np.array([]), np.empty((0, D)), {"n_above_mp": 0}

    U_pca, S_pca, Vt_pca = randomized_svd(X_centered, n_components=n_comp,
                                            random_state=42)
    eigenvalues = (S_pca ** 2) / N

    # σ² via trace normalization (d_residual = effective rank of residual space)
    total_var = float(np.sum(X_centered ** 2)) / N
    sigma_sq = total_var / d_residual
    gamma = d_residual / N

    if gamma >= 1.0:
        # Underdetermined: MP upper edge formula still applies but is very wide
        lambda_max_mp = sigma_sq * (1 + np.sqrt(gamma)) ** 2
        lambda_min_mp = 0.0
    else:
        lambda_max_mp = sigma_sq * (1 + np.sqrt(gamma)) ** 2
        lambda_min_mp = sigma_sq * (1 - np.sqrt(gamma)) ** 2

    n_above = int(np.sum(eigenvalues > lambda_max_mp))

    mp_info = {
        "sigma_sq_trace": float(sigma_sq),
        "gamma": float(gamma),
        "lambda_max_mp": float(lambda_max_mp),
        "lambda_min_mp": float(lambda_min_mp),
        "n_above_mp": n_above,
        "d_residual_pca": d_residual,
        "N": N,
    }

    logger.debug(f"    PCA: {n_comp} components, σ²={sigma_sq:.6f}, "
                 f"γ={gamma:.4f}, λ_max_MP={lambda_max_mp:.6f}, "
                 f"n_above={n_above}, top_eig={eigenvalues[0]:.6f}")

    return eigenvalues, Vt_pca, mp_info


# ═══════════════════════════════════════════════════════════════════════════════
# CORE: CORRELATION SWEEP
# ═══════════════════════════════════════════════════════════════════════════════

def compute_derived_columns(df):
    """Compute interaction terms and derived quantities for correlation sweep."""
    derived = {}
    cols = set(df.columns)

    # Carry interactions
    carry_cols = sorted([c for c in cols if c.startswith("carry_")])
    for i in range(len(carry_cols)):
        for j in range(i + 1, len(carry_cols)):
            ci, cj = carry_cols[i], carry_cols[j]
            derived[f"{ci}_x_{cj}"] = df[ci].values * df[cj].values

    # Consecutive nonzero carry count (longest run)
    if carry_cols:
        carry_matrix = df[carry_cols].values
        nonzero = carry_matrix > 0
        max_runs = np.zeros(len(df), dtype=int)
        for row_idx in range(len(df)):
            run = 0
            best = 0
            for col_idx in range(nonzero.shape[1]):
                if nonzero[row_idx, col_idx]:
                    run += 1
                    best = max(best, run)
                else:
                    run = 0
            max_runs[row_idx] = best
        derived["consecutive_carry_run"] = max_runs

    # Predicted answer digit features
    if "predicted" in cols:
        pred = df["predicted"].values
        pred_abs = np.abs(pred)
        pred_str = pd.Series(pred_abs).astype(str)
        derived["n_digits_predicted"] = pred_str.str.len().values
        derived["leading_digit_predicted"] = pred_str.str[0].astype(int).values
        derived["last_digit_predicted"] = (pred_abs % 10).astype(int)

    return derived


def correlation_sweep(X_residual, signal_directions, signal_eigenvalues,
                      df, pop_indices, lambda_max_mp, logger):
    """Correlate signal directions with all metadata.

    Returns DataFrame with one row per (direction, metadata_column) pair.
    """
    if len(signal_directions) == 0:
        return pd.DataFrame()

    # Slice DF to population
    df_pop = df.iloc[pop_indices].reset_index(drop=True)

    # All coloring columns (numeric only)
    numeric_cols = df_pop.select_dtypes(include=[np.number]).columns.tolist()
    # Remove identity columns
    skip_cols = {"problem_idx"}
    corr_cols = {c: df_pop[c].values for c in numeric_cols if c not in skip_cols}

    # Derived columns
    derived = compute_derived_columns(df_pop)
    corr_cols.update(derived)

    results = []
    for dir_idx in range(len(signal_directions)):
        v = signal_directions[dir_idx]
        eig = signal_eigenvalues[dir_idx]
        scores = X_residual @ v
        eig_over_mp = eig / lambda_max_mp if lambda_max_mp > 0 else np.inf

        for col_name, col_values in corr_cols.items():
            # Skip if constant or has NaN
            valid = ~np.isnan(col_values.astype(float))
            if valid.sum() < 30:
                continue
            s = scores[valid]
            v_col = col_values[valid].astype(float)
            if np.std(v_col) < 1e-12:
                continue

            rho_s, p_s = spearmanr(s, v_col)
            rho_p, p_p = pearsonr(s, v_col)

            results.append({
                "direction_idx": dir_idx,
                "eigenvalue": float(eig),
                "eig_over_mp_ratio": float(eig_over_mp),
                "metadata_column": col_name,
                "spearman_rho": float(rho_s),
                "spearman_p": float(p_s),
                "pearson_r": float(rho_p),
                "pearson_p": float(p_p),
                "n_valid": int(valid.sum()),
                "flagged": abs(rho_s) > CORR_FLAG_THRESHOLD,
            })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
# ATOMIC I/O
# ═══════════════════════════════════════════════════════════════════════════════

def atomic_json_write(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".json")
    os.close(fd)
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp_path, path)


def atomic_npy_write(arr, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".npy")
    os.close(fd)
    np.save(tmp_path, arr)
    os.replace(tmp_path, path)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SLICE PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase_e_slice(level, layer, pop, df_full, pop_df, paths,
                      n_pca_components, logger):
    """Run Phase E for one (level, layer, population) slice.

    Returns result dict for summary CSV.
    """
    slice_id = f"L{level}/layer{layer:02d}/{pop}"
    t_start = time.time()

    # Check for cached result
    union_meta_path = (paths["union_bases_dir"] / f"L{level}" /
                       f"layer_{layer:02d}" / pop / "metadata.json")
    pca_meta_path = (paths["pca_dir"] / f"L{level}" /
                     f"layer_{layer:02d}" / pop / "metadata.json")
    if pca_meta_path.exists():
        try:
            with open(pca_meta_path) as f:
                cached = json.load(f)
            if cached.get("computation_status") == "complete":
                logger.debug(f"  [{slice_id}] cached — skipping")
                return cached.get("summary_row", None)
        except (json.JSONDecodeError, KeyError):
            pass

    N = len(pop_df)
    pop_indices = pop_df.index.values
    logger.info(f"  [{slice_id}] N={N}")

    # Get concept list
    concept_list = get_concept_registry(level, df_full, pop)

    # Build union subspace
    V_all, union_meta = build_union_basis(level, layer, pop, df_full,
                                          concept_list, paths, logger)
    k = union_meta["union_rank_k"]
    d_residual = 4096 - k
    union_meta["level"] = level
    union_meta["layer"] = layer
    union_meta["population"] = pop
    union_meta["d_residual"] = d_residual
    union_meta["N"] = N

    # Save union basis
    union_dir = (paths["union_bases_dir"] / f"L{level}" /
                 f"layer_{layer:02d}" / pop)
    atomic_npy_write(V_all, union_dir / "union_basis.npy")
    atomic_json_write(union_meta, union_dir / "metadata.json")

    # Load residualized activations and slice to population
    X_resid_full = load_residualized(level, layer, paths["residualized_dir"])
    X_resid = X_resid_full[pop_indices]
    del X_resid_full

    # Compute residual
    X_residual, var_explained, var_orig, var_resid = compute_residual(
        X_resid, V_all)
    del X_resid

    logger.info(f"    var_explained={var_explained:.4f} "
                f"(k={k}, d_resid={d_residual})")

    # Sanity check: residual variance should not exceed original
    if var_resid > var_orig * 1.001:
        logger.warning(f"    [{slice_id}] Residual variance exceeds original — "
                       f"potential projection error! "
                       f"(var_resid={var_resid:.2f}, var_orig={var_orig:.2f})")

    # PCA on residual
    eigenvalues, eigenvectors, mp_info = pca_with_mp(
        X_residual, n_pca_components, d_residual, logger)

    n_above = mp_info["n_above_mp"]
    lambda_max_mp = mp_info["lambda_max_mp"]

    # Save PCA results
    pca_dir = paths["pca_dir"] / f"L{level}" / f"layer_{layer:02d}" / pop
    if len(eigenvalues) > 0:
        atomic_npy_write(eigenvalues, pca_dir / "eigenvalues.npy")
        atomic_npy_write(eigenvectors, pca_dir / "eigenvectors.npy")

    # Correlation sweep (if signal found)
    corr_df = pd.DataFrame()
    top_corr_concept = ""
    top_corr_spearman = 0.0
    top_corr_pearson = 0.0

    if n_above > 0:
        logger.info(f"    {n_above} eigenvalue(s) above MP edge — "
                    f"running correlation sweep")
        signal_dirs = eigenvectors[:n_above]
        signal_eigs = eigenvalues[:n_above]
        corr_df = correlation_sweep(X_residual, signal_dirs, signal_eigs,
                                    df_full, pop_indices, lambda_max_mp,
                                    logger)
        if not corr_df.empty:
            corr_path = (paths["correlations_dir"] / f"L{level}" /
                         f"layer_{layer:02d}" / pop / "correlation_sweep.csv")
            corr_path.parent.mkdir(parents=True, exist_ok=True)
            corr_df.to_csv(corr_path, index=False)
            logger.info(f"    n_correlation_tests={len(corr_df)} "
                        f"({n_above} dirs × {len(corr_df)//max(n_above,1)} cols)")


            # Find top correlation
            best_row = corr_df.loc[corr_df["spearman_rho"].abs().idxmax()]
            top_corr_concept = best_row["metadata_column"]
            top_corr_spearman = best_row["spearman_rho"]
            top_corr_pearson = best_row["pearson_r"]
            logger.info(f"    Top correlation: {top_corr_concept} "
                        f"(ρ_s={top_corr_spearman:.4f}, "
                        f"r_p={top_corr_pearson:.4f})")

            # Log all flagged
            flagged = corr_df[corr_df["flagged"]]
            if not flagged.empty:
                for _, row in flagged.iterrows():
                    logger.info(f"    FLAGGED: dir{row['direction_idx']} × "
                                f"{row['metadata_column']} "
                                f"ρ_s={row['spearman_rho']:.4f}")
    else:
        logger.info(f"    No eigenvalues above MP edge — flat spectrum")

    del X_residual

    elapsed = time.time() - t_start
    top_eig = float(eigenvalues[0]) if len(eigenvalues) > 0 else 0.0
    top_eig_ratio = top_eig / lambda_max_mp if lambda_max_mp > 0 else 0.0

    # Build summary row
    summary_row = {
        "level": level,
        "layer": layer,
        "population": pop,
        "n_concepts_total": union_meta["n_concepts_total"],
        "n_concepts_nonzero": union_meta["n_concepts_nonzero"],
        "stacked_dim": union_meta["stacked_dim_before_svd"],
        "union_rank_k": k,
        "d_residual": d_residual,
        "N": N,
        "gamma": mp_info.get("gamma", 0),
        "sigma_sq_trace": mp_info.get("sigma_sq_trace", 0),
        "lambda_max_mp": lambda_max_mp,
        "n_above_mp": n_above,
        "top_eigenvalue": top_eig,
        "top_eig_over_mp_ratio": top_eig_ratio,
        "var_explained": var_explained,
        "var_residual": var_resid / var_orig if var_orig > 0 else 1.0,
        "k_without_total_carry_sum": union_meta["k_without_total_carry_sum"],
        "top_corr_concept": top_corr_concept,
        "top_corr_spearman": top_corr_spearman,
        "top_corr_pearson": top_corr_pearson,
        "computation_time_s": elapsed,
    }

    # Save PCA metadata
    pca_meta = {
        **summary_row,
        "eigenvalues_above_mp": [float(e) for e in eigenvalues[:n_above]],
        "top_20_eigenvalues": [float(e) for e in eigenvalues[:20]],
        "pca_n_components_computed": len(eigenvalues),
        "computation_status": "complete",
        "summary_row": summary_row,
    }
    atomic_json_write(pca_meta, pca_meta_path)

    return summary_row


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_eigenvalue_spectrum(eigenvalues, mp_info, level, layer, pop,
                             plot_dir, logger):
    """Scree plot with MP overlay."""
    fig, ax = plt.subplots(figsize=(10, 6))
    n_plot = min(100, len(eigenvalues))
    x = np.arange(1, n_plot + 1)
    ax.plot(x, eigenvalues[:n_plot], "b.-", label="Observed eigenvalues")
    ax.axhline(mp_info["lambda_max_mp"], color="r", linestyle="--",
               label=f"MP upper edge (λ_max={mp_info['lambda_max_mp']:.5f})")
    ax.axhline(mp_info["sigma_sq_trace"], color="gray", linestyle=":",
               alpha=0.5, label=f"σ² (trace) = {mp_info['sigma_sq_trace']:.5f}")

    n_above = mp_info["n_above_mp"]
    n_fill = min(n_above, n_plot)
    if n_fill > 0:
        ax.fill_between(x[:n_fill], eigenvalues[:n_fill],
                        mp_info["lambda_max_mp"],
                        alpha=0.3, color="red",
                        label=f"{n_above} above MP")

    ax.set_xlabel("Component index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"Phase E: L{level}/layer{layer:02d}/{pop} "
                 f"(k={mp_info.get('d_residual_pca', '?')}, "
                 f"N={mp_info['N']})")
    ax.legend(fontsize=8)
    ax.set_yscale("log")

    out_dir = plot_dir / "eigenvalue_spectra"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"L{level}_layer{layer:02d}_{pop}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(data_df, value_col, title, fname, plot_dir, logger):
    """Heatmap of a value across layers × populations."""
    for level in data_df["level"].unique():
        sub = data_df[data_df["level"] == level]
        pivot = sub.pivot_table(index="population", columns="layer",
                                values=value_col, aggfunc="first")
        fig, ax = plt.subplots(figsize=(12, 3))
        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Layer")
        ax.set_title(f"{title} — L{level}")
        plt.colorbar(im, ax=ax)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=7, color="white" if val > pivot.values.max() * 0.5 else "black")

        out_dir = plot_dir / fname
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"L{level}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_union_rank_trajectories(data_df, plot_dir, logger):
    """k vs layer curves per (level, population)."""
    fig, axes = plt.subplots(1, len(PLOT_LEVELS), figsize=(5 * len(PLOT_LEVELS), 5),
                              sharey=True)
    if len(PLOT_LEVELS) == 1:
        axes = [axes]
    for ax, level in zip(axes, PLOT_LEVELS):
        sub = data_df[data_df["level"] == level]
        for pop in ["all", "correct", "wrong"]:
            pop_data = sub[sub["population"] == pop].sort_values("layer")
            if not pop_data.empty:
                ax.plot(pop_data["layer"], pop_data["union_rank_k"],
                        "o-", label=pop)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Union rank k")
        ax.set_title(f"L{level}")
        ax.legend(fontsize=8)
    out_dir = plot_dir / "union_rank_trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "union_rank_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_plots(results_df, all_eigenvalues, paths, logger):
    """Generate plots for L3/L4/L5 at key layers only."""
    plot_dir = paths["phase_e_plots"]
    logger.info("Generating plots...")

    # Eigenvalue spectra — only for PLOT_LEVELS × PLOT_LAYERS
    for (level, layer, pop), eigs_mp in all_eigenvalues.items():
        if level in PLOT_LEVELS and layer in PLOT_LAYERS:
            eigenvalues, mp_info = eigs_mp
            if len(eigenvalues) > 0:
                plot_eigenvalue_spectrum(eigenvalues, mp_info, level, layer,
                                        pop, plot_dir, logger)

    # Heatmaps — only for PLOT_LEVELS
    plot_data = results_df[results_df["level"].isin(PLOT_LEVELS)]
    if not plot_data.empty:
        plot_heatmap(plot_data, "n_above_mp", "# Eigenvalues above MP",
                     "mp_heatmaps", plot_dir, logger)
        plot_heatmap(plot_data, "var_explained", "Variance Explained",
                     "variance_explained_heatmaps", plot_dir, logger)

    # Union rank trajectories
    plot_union_rank_trajectories(results_df, plot_dir, logger)

    n_plots = len([k for k in all_eigenvalues
                   if k[0] in PLOT_LEVELS and k[1] in PLOT_LAYERS])
    logger.info(f"  Generated {n_plots} eigenvalue spectra + heatmaps")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLES
# ═══════════════════════════════════════════════════════════════════════════════

def generate_summaries(results_df, all_eigenvalues, paths, logger):
    """Generate all 6 summary CSVs."""
    summary_dir = paths["summary_dir"]
    summary_dir.mkdir(parents=True, exist_ok=True)

    # 1. Master table
    results_df.to_csv(summary_dir / "phase_e_results.csv", index=False)
    logger.info(f"  Saved phase_e_results.csv ({len(results_df)} rows)")

    # 2. Eigenvalue cliff summary (only slices with signal)
    cliff = results_df[results_df["n_above_mp"] > 0]
    cliff.to_csv(summary_dir / "eigenvalue_cliff_summary.csv", index=False)
    logger.info(f"  Saved eigenvalue_cliff_summary.csv ({len(cliff)} rows)")

    # 3. Union rank by layer
    rank_cols = ["level", "layer", "population", "union_rank_k",
                 "k_without_total_carry_sum", "stacked_dim", "d_residual"]
    rank_df = results_df[rank_cols].copy()
    rank_df.to_csv(summary_dir / "union_rank_by_layer.csv", index=False)
    logger.info(f"  Saved union_rank_by_layer.csv")

    # 4. Variance explained
    var_cols = ["level", "layer", "population", "var_explained",
                "var_residual", "union_rank_k", "N"]
    var_df = results_df[var_cols].copy()
    var_df.to_csv(summary_dir / "variance_explained.csv", index=False)
    logger.info(f"  Saved variance_explained.csv")

    # 5. total_carry_sum diagnostic
    tcs_cols = ["level", "layer", "population", "union_rank_k",
                "k_without_total_carry_sum"]
    tcs_df = results_df[tcs_cols].copy()
    tcs_df["delta_k"] = tcs_df["union_rank_k"] - tcs_df["k_without_total_carry_sum"]
    tcs_df.to_csv(summary_dir / "total_carry_sum_diagnostic.csv", index=False)
    logger.info(f"  Saved total_carry_sum_diagnostic.csv")

    # 6. Top eigenvalues per slice
    top_eig_rows = []
    for (level, layer, pop), (eigs, mp) in all_eigenvalues.items():
        for i, e in enumerate(eigs[:20]):
            top_eig_rows.append({
                "level": level, "layer": layer, "population": pop,
                "component_idx": i, "eigenvalue": float(e),
                "lambda_max_mp": mp["lambda_max_mp"],
                "above_mp": float(e) > mp["lambda_max_mp"],
            })
    if top_eig_rows:
        top_eig_df = pd.DataFrame(top_eig_rows)
        top_eig_df.to_csv(summary_dir / "top_eigenvalues_all_slices.csv",
                          index=False)
        logger.info(f"  Saved top_eigenvalues_all_slices.csv "
                    f"({len(top_eig_df)} rows)")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase E — Residual Hunting via PCA")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--pilot", action="store_true",
                        help="Smoke test: L3, layer 16, all population")
    parser.add_argument("--level", type=int, nargs="*",
                        help="Specific levels (default: 2-5)")
    parser.add_argument("--layer", type=int, nargs="*",
                        help="Specific layers (default: all 9)")
    parser.add_argument("--population", nargs="*",
                        help="Specific populations (default: all)")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--plots-only", action="store_true",
                        help="Skip Steps 1-2, regenerate plots from saved CSVs")
    parser.add_argument("--n-pca-components", type=int,
                        default=N_PCA_COMPONENTS,
                        help=f"Max PCA components (default: {N_PCA_COMPONENTS})")
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    cfg = load_config(args.config)
    paths = derive_paths(cfg)
    logger = setup_logging(paths["workspace"])

    # Banner
    logger.info("=" * 70)
    logger.info("Phase E — Residual Hunting")
    logger.info("=" * 70)

    # Scope
    if args.pilot:
        levels = [3]
        layers = [16]
        populations_filter = ["all"]
        logger.info("PILOT MODE: L3, layer 16, all population")
    else:
        levels = args.level if args.level else LEVELS
        layers = args.layer if args.layer else LAYERS
        populations_filter = args.population if args.population else None

    logger.info(f"Levels: {levels}")
    logger.info(f"Layers: {layers}")
    logger.info(f"Populations: {populations_filter or 'all viable'}")
    logger.info(f"PCA components: {args.n_pca_components}")
    logger.info(f"GPU (CuPy): "
                f"{'available' if _CUPY_AVAILABLE else 'CPU fallback'}")

    # Pre-flight checks
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

    if missing:
        logger.error(f"Missing {len(missing)} input files. First 5:")
        for p in missing[:5]:
            logger.error(f"  {p}")
        logger.error("Run Phase C/D first.")
        return

    # Check Phase D subspaces exist
    phase_d_meta_count = sum(1 for _ in paths["phase_d_subspaces"].rglob(
        "metadata.json"))
    logger.info(f"Phase D metadata files found: {phase_d_meta_count}")
    if phase_d_meta_count < 100:
        logger.warning(f"Only {phase_d_meta_count} Phase D metadata files "
                       f"found (expected ~2800)")

    # Create output directories
    for key in ["phase_e_data", "union_bases_dir", "pca_dir",
                "correlations_dir", "summary_dir", "phase_e_plots"]:
        paths[key].mkdir(parents=True, exist_ok=True)

    # --plots-only: reload from saved CSVs and eigenvalue files, skip Steps 1-2
    if args.plots_only:
        logger.info("PLOTS-ONLY MODE: loading from saved outputs")
        csv_path = paths["summary_dir"] / "phase_e_results.csv"
        if not csv_path.exists():
            logger.error(f"Cannot find {csv_path} — run full pipeline first")
            return
        results_df = pd.read_csv(csv_path)
        logger.info(f"  Loaded {len(results_df)} rows from phase_e_results.csv")

        all_eigenvalues = {}
        for _, row in results_df.iterrows():
            lv, ly, pop = int(row["level"]), int(row["layer"]), row["population"]
            eig_path = (paths["pca_dir"] / f"L{lv}" /
                        f"layer_{ly:02d}" / pop / "eigenvalues.npy")
            meta_path = (paths["pca_dir"] / f"L{lv}" /
                         f"layer_{ly:02d}" / pop / "metadata.json")
            if eig_path.exists() and meta_path.exists():
                eigs = np.load(eig_path)
                with open(meta_path) as f:
                    meta = json.load(f)
                all_eigenvalues[(lv, ly, pop)] = (
                    eigs, {
                        "lambda_max_mp": meta.get("lambda_max_mp", 0),
                        "sigma_sq_trace": meta.get("sigma_sq_trace", 0),
                        "gamma": meta.get("gamma", 0),
                        "n_above_mp": meta.get("n_above_mp", 0),
                        "d_residual_pca": meta.get("d_residual", 0),
                        "N": meta.get("N", 0),
                    })
        logger.info(f"  Loaded eigenvalues for {len(all_eigenvalues)} slices")

        logger.info("")
        logger.info("Step 3: Plot generation (plots-only)")
        logger.info("-" * 40)
        generate_plots(results_df, all_eigenvalues, paths, logger)

        elapsed = time.time() - t0
        logger.info(f"Plots-only complete: {elapsed:.0f}s")
        return

    # Main loop
    logger.info("")
    logger.info("Step 1: Per-slice residual analysis")
    logger.info("-" * 40)

    all_results = []
    all_eigenvalues = {}  # (level, layer, pop) → (eigenvalues, mp_info)
    n_slices = 0

    for level in levels:
        df_full = load_coloring_df(level, paths["coloring_dir"])
        pops = get_populations(df_full)

        for pop_name, pop_df in pops.items():
            if populations_filter and pop_name not in populations_filter:
                continue

            for layer in tqdm(layers,
                              desc=f"L{level}/{pop_name}",
                              leave=False):
                result = run_phase_e_slice(
                    level, layer, pop_name, df_full, pop_df,
                    paths, args.n_pca_components, logger)

                if result is not None:
                    all_results.append(result)
                    n_slices += 1

                    # Load eigenvalues for plotting/summary
                    eig_path = (paths["pca_dir"] / f"L{level}" /
                                f"layer_{layer:02d}" / pop_name /
                                "eigenvalues.npy")
                    meta_path = (paths["pca_dir"] / f"L{level}" /
                                 f"layer_{layer:02d}" / pop_name /
                                 "metadata.json")
                    if eig_path.exists() and meta_path.exists():
                        eigs = np.load(eig_path)
                        with open(meta_path) as f:
                            meta = json.load(f)
                        all_eigenvalues[(level, layer, pop_name)] = (
                            eigs, {
                                "lambda_max_mp": meta.get("lambda_max_mp", 0),
                                "sigma_sq_trace": meta.get("sigma_sq_trace", 0),
                                "gamma": meta.get("gamma", 0),
                                "n_above_mp": meta.get("n_above_mp", 0),
                                "d_residual_pca": meta.get("d_residual", 0),
                                "N": meta.get("N", 0),
                            })

    logger.info(f"Step 1 complete: {n_slices} slices processed")

    if not all_results:
        logger.warning("No results — check data paths")
        return

    results_df = pd.DataFrame(all_results)

    # Step 2: Summary CSVs
    logger.info("")
    logger.info("Step 2: Summary tables")
    logger.info("-" * 40)
    generate_summaries(results_df, all_eigenvalues, paths, logger)

    # Step 3: Plots
    if not args.skip_plots:
        logger.info("")
        logger.info("Step 3: Plot generation")
        logger.info("-" * 40)
        generate_plots(results_df, all_eigenvalues, paths, logger)

    # Cross-layer consistency check
    logger.info("")
    logger.info("Step 4: Cross-layer consistency check")
    logger.info("-" * 40)
    for level in levels:
        for pop in results_df["population"].unique():
            sub = results_df[(results_df["level"] == level) &
                             (results_df["population"] == pop)]
            if len(sub) > 1:
                k_vals = sub["union_rank_k"].values
                k_mean = k_vals.mean()
                k_std = k_vals.std()
                k_range = k_vals.max() - k_vals.min()
                logger.info(f"  L{level}/{pop}: k mean={k_mean:.0f} "
                            f"std={k_std:.1f} range={k_range}")
                if k_range > 50:
                    logger.warning(f"  L{level}/{pop}: k range={k_range} "
                                   f"exceeds 50 — check for anomalies")

    # Done
    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"Phase E complete: {n_slices} slices, "
                f"{elapsed:.0f}s ({elapsed / 60:.1f} min)")
    logger.info("=" * 70)

    n_with_signal = int((results_df["n_above_mp"] > 0).sum())
    logger.info(f"Slices with signal above MP: {n_with_signal} / {n_slices}")
    if n_with_signal == 0:
        logger.info("All eigenvalue spectra are flat — completeness result.")
    else:
        logger.info(f"Signal detected in {n_with_signal} slices — "
                    f"check correlation_sweep CSVs")

    mean_var = results_df["var_explained"].mean()
    logger.info(f"Mean variance explained by known concepts: "
                f"{mean_var:.4f} ({mean_var*100:.1f}%)")


if __name__ == "__main__":
    main()
