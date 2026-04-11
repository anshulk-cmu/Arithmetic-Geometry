#!/usr/bin/env python3
"""
Phase G Number-Token Fourier Screening
=======================================
Tests whether Kantamneni & Tegmark (2025) Fourier digit circles persist
at the operand token position in multiplication prompts ("{a} * {b} =").

Design:
  - Loads pre-extracted number-token activations (float16, 4096-dim)
  - Groups by digit value from coloring DataFrames
  - PCA on centroids to extract between-group subspace, then Fourier analysis
  - Dual reporting: PCA-space (primary, with permutation null) and raw 4096-dim (secondary)
  - Imports all Fourier/helix/permutation math from phase_g_fourier.py (zero duplication)

Usage:
  python phase_g_numtok_fourier.py --config config.yaml                    # full run
  python phase_g_numtok_fourier.py --config config.yaml --pilot            # L3/layer16 only
  python phase_g_numtok_fourier.py --config config.yaml --n-perms 200      # quick test
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import false_discovery_control

# ---------------------------------------------------------------------------
# Import statistical core from main Phase G script (zero duplication)
# ---------------------------------------------------------------------------
from phase_g_fourier import (
    fourier_all_coordinates,
    compute_linear_power,
    compute_helix_fcr,
    compute_centroids_grouped,
    compute_pvalues,
    compute_pvalues_array,
    permutation_null,
    PERM_ALPHA,
    COORD_P_THRESHOLD,
    LINEAR_P_THRESHOLD,
    MIN_POPULATION,
    ZERO_POWER_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LEVELS = [2, 3, 4, 5]
NUMTOK_LAYERS = [4, 8, 12, 16, 20, 24]
DIGIT_PERIOD = 10
FDR_THRESHOLD = 0.05
DEFAULT_PCA_DIM = 20
DEFAULT_N_PERMS = 1000
CHECKPOINT_INTERVAL = 50

# Digit concept registry: (column_name, position, available_levels)
# Position refers to which operand's token position to use.
DIGIT_CONCEPTS = [
    ("a_units",    "a", [2, 3, 4, 5]),
    ("a_tens",     "a", [2, 3, 4, 5]),
    ("a_hundreds", "a", [4, 5]),
    ("b_units",    "b", [2, 3, 4, 5]),
    ("b_tens",     "b", [3, 4, 5]),
    ("b_hundreds", "b", [5]),
]


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def setup_logging(log_path):
    logger = logging.getLogger("numtok_fourier")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S"
    )

    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_numtok_activations(level, layer, pos, data_root):
    """Load number-token activations for a given level/layer/position.

    Returns (N, 4096) float32 array.
    """
    fname = f"level{level}_layer_{layer:02d}_pos_{pos}.npy"
    path = os.path.join(data_root, "activations_numtok", fname)
    acts = np.load(path).astype(np.float32)
    return acts


def load_coloring_df(level, data_root):
    """Load the coloring DataFrame for a given level."""
    path = os.path.join(
        data_root, "phase_a", "coloring_dfs", f"L{level}_coloring.pkl"
    )
    return pd.read_pickle(path)


def get_digit_concepts_for_level(level):
    """Return list of (col_name, position) for concepts available at this level."""
    return [
        (col, pos)
        for col, pos, levels in DIGIT_CONCEPTS
        if level in levels
    ]


# ---------------------------------------------------------------------------
# PCA on centroids
# ---------------------------------------------------------------------------
def pca_on_centroids(centroids, pca_dim):
    """PCA on group centroids to extract between-group subspace.

    Args:
        centroids: (m, D) array of group centroids in full space
        pca_dim: maximum number of PCA dimensions

    Returns:
        components: (k, D) PCA basis vectors (k = min(pca_dim, m-1))
        eigenvalues: (k,) eigenvalues
    """
    m, D = centroids.shape
    # Between-group structure lives in at most m-1 dimensions
    k = min(pca_dim, m - 1)
    if k == 0:
        return np.zeros((0, D)), np.zeros(0)

    # Center centroids
    mean = centroids.mean(axis=0)
    centered = centroids - mean

    # SVD on centered centroids (m << D, so this is fast)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = (S[:k] ** 2) / (m - 1)
    components = Vt[:k]  # (k, D)

    return components, eigenvalues


# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------
def analyze_numtok_concept(
    acts, labels, unique_vals, v_linear, period, pca_dim, n_perms, rng, logger
):
    """Run Fourier screening on number-token activations for one concept.

    Args:
        acts: (N, 4096) float32 activations
        labels: (N,) integer digit labels
        unique_vals: sorted unique digit values
        v_linear: (m,) linear axis values for helix test
        period: Fourier period (10 for digits)
        pca_dim: max PCA dimensions
        n_perms: number of permutations
        rng: numpy random generator
        logger: logger

    Returns:
        dict with all results (FCR, p-values, detection flags, etc.)
    """
    m = len(unique_vals)
    N = len(labels)

    # ------------------------------------------------------------------
    # Step 1: Centroids in raw 4096-dim space (secondary reporting)
    # ------------------------------------------------------------------
    raw_centroids, group_sizes = compute_centroids_grouped(acts, labels, unique_vals)
    raw_dc = raw_centroids.mean(axis=0)
    raw_centroids_dc = raw_centroids - raw_dc

    raw_fourier = fourier_all_coordinates(raw_centroids_dc, unique_vals, period, logger=None)
    raw_linear_power = compute_linear_power(raw_centroids_dc, v_linear, group_sizes)
    raw_helix = compute_helix_fcr(raw_fourier, raw_linear_power, v_linear, group_sizes, logger=None)

    raw_two_axis_fcr = raw_fourier["two_axis_fcr"]
    raw_helix_fcr = raw_helix["helix_fcr"]

    logger.debug(
        f"  Raw 4096-dim: two_axis_fcr={raw_two_axis_fcr:.4f}, helix_fcr={raw_helix_fcr:.4f}"
    )

    # ------------------------------------------------------------------
    # Step 2: PCA on centroids → project to between-group subspace
    # ------------------------------------------------------------------
    components, eigenvalues = pca_on_centroids(raw_centroids, pca_dim)
    d = components.shape[0]  # actual dimensionality (min(pca_dim, m-1))

    if d == 0:
        logger.warning("  PCA dimension is 0 — skipping analysis")
        return _make_null_result(
            m=m, N=N, d=0, pca_dim=pca_dim,
            raw_two_axis_fcr=raw_two_axis_fcr, raw_helix_fcr=raw_helix_fcr,
            group_sizes=group_sizes, reason="d=0"
        )

    logger.debug(f"  PCA: {d} components (from {m} groups), top eigenvalue={eigenvalues[0]:.4f}")

    # Project all activations into PCA space
    acts_mean = raw_dc  # same mean used for centering
    projected = (acts - acts_mean) @ components.T  # (N, d)

    # ------------------------------------------------------------------
    # Step 3: Centroids in PCA space
    # ------------------------------------------------------------------
    pca_centroids, _ = compute_centroids_grouped(projected, labels, unique_vals)
    pca_dc = pca_centroids.mean(axis=0)
    pca_centroids_dc = pca_centroids - pca_dc

    # ------------------------------------------------------------------
    # Step 4: Fourier analysis in PCA space
    # ------------------------------------------------------------------
    fourier_res = fourier_all_coordinates(pca_centroids_dc, unique_vals, period, logger)
    linear_power = compute_linear_power(pca_centroids_dc, v_linear, group_sizes)
    helix_res = compute_helix_fcr(fourier_res, linear_power, v_linear, group_sizes, logger)

    two_axis_fcr = fourier_res["two_axis_fcr"]
    helix_fcr = helix_res["helix_fcr"]

    logger.debug(
        f"  PCA space (d={d}): two_axis_fcr={two_axis_fcr:.4f}, "
        f"helix_fcr={helix_fcr:.4f}, best_freq={fourier_res['two_axis_best_freq']}"
    )

    # ------------------------------------------------------------------
    # Step 5: Permutation null in PCA space
    # ------------------------------------------------------------------
    if n_perms > 0:
        null_res = permutation_null(
            projected, labels, unique_vals, period, v_linear, n_perms, rng, logger
        )
        p_two_axis = compute_pvalues(two_axis_fcr, null_res["null_two_axis_fcr"])
        p_helix = compute_pvalues(helix_fcr, null_res["null_helix_fcr"])
        p_uniform = compute_pvalues(
            fourier_res["uniform_fcr_top1"], null_res["null_uniform_fcr"]
        )
        p_coord = compute_pvalues_array(
            fourier_res["per_coord_fcr_top1"], null_res["null_coord_fcr"]
        )
        p_linear = compute_pvalues_array(linear_power, null_res["null_linear_power"])
        p_value_floor = null_res["p_value_floor"]
        p_saturated = (p_two_axis <= p_value_floor + 1e-12) or (
            p_helix <= p_value_floor + 1e-12
        )
    else:
        p_two_axis = p_helix = p_uniform = None
        p_coord = np.full(d, np.nan)
        p_linear = np.full(d, np.nan)
        p_value_floor = None
        p_saturated = None

    # ------------------------------------------------------------------
    # Step 6: Detection logic (identical to main Phase G)
    # ------------------------------------------------------------------
    if n_perms > 0 and d >= 2:
        coord_a = fourier_res["two_axis_coord_a"]
        coord_b = fourier_res["two_axis_coord_b"]

        circle_detected = (
            p_two_axis < PERM_ALPHA
            and p_coord[coord_a] < COORD_P_THRESHOLD
            and p_coord[coord_b] < COORD_P_THRESHOLD
        )

        helix_best_freq = helix_res["helix_best_freq"]
        helix_best_freq_idx = helix_best_freq - 1
        helix_top2 = fourier_res["per_freq_top2_coords"][helix_best_freq_idx]
        helix_coord_a = helix_top2[0]
        helix_coord_b = helix_top2[1]
        helix_linear_coord = helix_res["helix_linear_coord"]

        helix_detected = (
            p_helix < PERM_ALPHA
            and p_coord[helix_coord_a] < COORD_P_THRESHOLD
            and p_coord[helix_coord_b] < COORD_P_THRESHOLD
            and p_linear[helix_linear_coord] < LINEAR_P_THRESHOLD
        )
    else:
        circle_detected = False
        helix_detected = False

    if helix_detected:
        geometry_detected = "helix"
    elif circle_detected:
        geometry_detected = "circle"
    else:
        geometry_detected = "none"

    logger.info(
        f"  p_two_axis={_fmt(p_two_axis)}, p_helix={_fmt(p_helix)}, "
        f"p_uniform={_fmt(p_uniform)}, p_saturated={p_saturated}"
    )
    logger.info(
        f"  circle_detected={circle_detected}, helix_detected={helix_detected}, "
        f"geometry_detected={geometry_detected}"
    )

    # ------------------------------------------------------------------
    # Build result dict
    # ------------------------------------------------------------------
    result = {
        "N": N,
        "m": m,
        "d": d,
        "pca_dim": pca_dim,
        "period": period,
        "group_sizes": group_sizes.tolist(),
        # PCA-space statistics (primary)
        "two_axis_fcr": two_axis_fcr,
        "two_axis_best_freq": int(fourier_res["two_axis_best_freq"]),
        "two_axis_coord_a": int(fourier_res["two_axis_coord_a"]),
        "two_axis_coord_b": int(fourier_res["two_axis_coord_b"]),
        "helix_fcr": helix_fcr,
        "helix_best_freq": int(helix_res["helix_best_freq"]),
        "helix_linear_coord": int(helix_res["helix_linear_coord"]),
        "helix_linear_power": float(helix_res["helix_linear_power"]),
        "uniform_fcr": float(fourier_res["uniform_fcr_top1"]),
        # P-values
        "p_two_axis": p_two_axis,
        "p_helix": p_helix,
        "p_uniform": p_uniform,
        "p_coord": p_coord.tolist() if isinstance(p_coord, np.ndarray) else p_coord,
        "p_linear": p_linear.tolist() if isinstance(p_linear, np.ndarray) else p_linear,
        "p_value_floor": p_value_floor,
        "p_saturated": p_saturated,
        "n_perms": n_perms,
        # Detection
        "circle_detected": circle_detected,
        "helix_detected": helix_detected,
        "geometry_detected": geometry_detected,
        # Raw 4096-dim statistics (secondary)
        "raw_two_axis_fcr": raw_two_axis_fcr,
        "raw_helix_fcr": raw_helix_fcr,
        # PCA eigenvalues
        "pca_eigenvalues": eigenvalues.tolist(),
        "pca_var_explained": float(eigenvalues.sum()),
    }
    return result


def _fmt(v):
    """Format a p-value for logging."""
    return f"{v:.4f}" if v is not None else "None"


def _make_null_result(m, N, d, pca_dim, raw_two_axis_fcr, raw_helix_fcr,
                      group_sizes, reason):
    """Return a null result dict when analysis cannot be performed."""
    return {
        "N": N, "m": m, "d": d, "pca_dim": pca_dim, "period": DIGIT_PERIOD,
        "group_sizes": group_sizes.tolist() if hasattr(group_sizes, 'tolist') else list(group_sizes),
        "two_axis_fcr": None, "two_axis_best_freq": None,
        "two_axis_coord_a": None, "two_axis_coord_b": None,
        "helix_fcr": None, "helix_best_freq": None,
        "helix_linear_coord": None, "helix_linear_power": None,
        "uniform_fcr": None,
        "p_two_axis": None, "p_helix": None, "p_uniform": None,
        "p_coord": None, "p_linear": None,
        "p_value_floor": None, "p_saturated": None, "n_perms": 0,
        "circle_detected": False, "helix_detected": False,
        "geometry_detected": "none",
        "raw_two_axis_fcr": raw_two_axis_fcr, "raw_helix_fcr": raw_helix_fcr,
        "pca_eigenvalues": None, "pca_var_explained": None,
        "skip_reason": reason,
    }


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------
def run_all_numtok(config, n_perms, pca_dim, pilot, logger):
    """Run Fourier screening on all number-token concept/level/layer cells.

    Returns list of result dicts (one per analysis cell).
    """
    data_root = config["paths"]["data_root"]
    output_root = os.path.join(data_root, "phase_g")
    os.makedirs(os.path.join(output_root, "summary"), exist_ok=True)

    levels = [3] if pilot else LEVELS
    layers = [16] if pilot else NUMTOK_LAYERS
    rng = np.random.default_rng(42)

    all_results = []
    checkpoint_path = os.path.join(output_root, "numtok_checkpoint.pkl")
    t_start = time.time()
    n_total = _count_cells(levels, layers)

    logger.info(f"Number-token Fourier screening: {n_total} analysis cells")
    logger.info(f"  levels={levels}, layers={layers}, n_perms={n_perms}, pca_dim={pca_dim}")

    # Cache coloring DFs
    coloring_dfs = {}
    for level in levels:
        coloring_dfs[level] = load_coloring_df(level, data_root)

    n_done = 0
    for level in levels:
        df = coloring_dfs[level]
        concepts = get_digit_concepts_for_level(level)

        for layer in layers:
            # Load both positions' activations (cached per layer)
            acts_cache = {}
            for pos in ["a", "b"]:
                fpath = os.path.join(
                    data_root, "activations_numtok",
                    f"level{level}_layer_{layer:02d}_pos_{pos}.npy"
                )
                if os.path.exists(fpath):
                    acts_cache[pos] = np.load(fpath).astype(np.float32)
                    logger.debug(
                        f"Loaded {fpath}: shape={acts_cache[pos].shape}"
                    )

            for col_name, pos in concepts:
                if pos not in acts_cache:
                    logger.warning(
                        f"  Skipping {col_name} L{level}/layer{layer}/pos_{pos}: "
                        f"no activation file"
                    )
                    continue

                acts = acts_cache[pos]
                labels = df[col_name].values
                unique_vals = np.sort(df[col_name].unique())
                m = len(unique_vals)

                # v_linear: the digit values themselves (0-9 or 1-9)
                v_linear = unique_vals.astype(float)

                # Check minimum group size
                min_group = min(
                    np.sum(labels == v) for v in unique_vals
                )
                if min_group < MIN_POPULATION:
                    logger.warning(
                        f"  Skipping {col_name} L{level}/layer{layer}/pos_{pos}: "
                        f"min_group={min_group} < {MIN_POPULATION}"
                    )
                    continue

                n_done += 1
                logger.info(
                    f"ANALYZE [{n_done}/{n_total}]: {col_name} / L{level} / "
                    f"layer{layer} / pos_{pos} (P={DIGIT_PERIOD}, m={m}, N={len(labels)})"
                )

                result = analyze_numtok_concept(
                    acts, labels, unique_vals, v_linear,
                    DIGIT_PERIOD, pca_dim, n_perms, rng, logger
                )

                # Add metadata
                result["concept"] = col_name
                result["level"] = level
                result["layer"] = layer
                result["position"] = pos

                all_results.append(result)

                # Save per-concept JSON
                json_dir = os.path.join(
                    output_root, "numtok",
                    f"L{level}", f"layer_{layer:02d}", f"pos_{pos}"
                )
                os.makedirs(json_dir, exist_ok=True)
                json_path = os.path.join(json_dir, f"{col_name}_fourier_results.json")
                with open(json_path, "w") as f:
                    json.dump(_jsonable(result), f, indent=2)

                # Checkpoint
                if len(all_results) % CHECKPOINT_INTERVAL == 0:
                    with open(checkpoint_path, "wb") as f:
                        pickle.dump(all_results, f)
                    elapsed = time.time() - t_start
                    logger.info(
                        f"  Checkpoint: {len(all_results)} results saved "
                        f"({elapsed:.0f}s elapsed)"
                    )

    elapsed = time.time() - t_start
    logger.info(f"All analyses complete: {len(all_results)} results in {elapsed:.1f}s")

    return all_results


def _count_cells(levels, layers):
    """Count total analysis cells for progress reporting."""
    n = 0
    for level in levels:
        concepts = get_digit_concepts_for_level(level)
        n += len(concepts) * len(layers)
    return n


def _jsonable(d):
    """Convert numpy types to JSON-serializable Python types."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.bool_,)):
            out[k] = bool(v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# FDR correction and summary CSV
# ---------------------------------------------------------------------------
def save_summary_csv(all_results, output_root, logger):
    """Save summary CSV with FDR correction."""
    if not all_results:
        logger.warning("No results to save.")
        return

    df = pd.DataFrame(all_results)

    # FDR correction on p_two_axis and p_helix (BH method)
    for col in ["p_two_axis", "p_helix"]:
        valid = df[col].notna()
        if valid.sum() > 0:
            pvals = df.loc[valid, col].values.astype(float)
            qvals = false_discovery_control(pvals, method="bh")
            df.loc[valid, f"{col}_fdr"] = qvals
        else:
            df[f"{col}_fdr"] = np.nan

    # Drop internal array columns that don't belong in CSV
    drop_cols = ["p_coord", "p_linear", "pca_eigenvalues", "group_sizes"]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    csv_path = os.path.join(output_root, "summary", "numtok_fourier_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Summary CSV saved: {csv_path} ({len(df)} rows)")

    # Print summary stats
    n_circle = (df["geometry_detected"] == "circle").sum()
    n_helix = (df["geometry_detected"] == "helix").sum()
    n_none = (df["geometry_detected"] == "none").sum()
    logger.info(f"Detection summary: {n_helix} helix, {n_circle} circle, {n_none} none")

    if "p_two_axis_fdr" in df.columns:
        n_fdr_sig = (df["p_two_axis_fdr"] < FDR_THRESHOLD).sum()
        logger.info(f"FDR-significant (q < {FDR_THRESHOLD}): {n_fdr_sig}/{len(df)}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase G: Number-token Fourier screening"
    )
    parser.add_argument(
        "--config", required=True, help="Path to config.yaml"
    )
    parser.add_argument(
        "--n-perms", type=int, default=DEFAULT_N_PERMS,
        help=f"Number of permutations (default: {DEFAULT_N_PERMS})"
    )
    parser.add_argument(
        "--pca-dim", type=int, default=DEFAULT_PCA_DIM,
        help=f"Max PCA dimensions (default: {DEFAULT_PCA_DIM})"
    )
    parser.add_argument(
        "--pilot", action="store_true",
        help="Pilot mode: L3/layer16 only (~2 min)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    log_dir = os.path.join(config["paths"]["workspace"], "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "phase_g_numtok_fourier.log")
    logger = setup_logging(log_path)

    logger.info("=" * 70)
    logger.info("Phase G: Number-Token Fourier Screening")
    logger.info(f"  config: {args.config}")
    logger.info(f"  n_perms: {args.n_perms}")
    logger.info(f"  pca_dim: {args.pca_dim}")
    logger.info(f"  pilot: {args.pilot}")
    logger.info("=" * 70)

    all_results = run_all_numtok(
        config, args.n_perms, args.pca_dim, args.pilot, logger
    )

    output_root = os.path.join(config["paths"]["data_root"], "phase_g")
    save_summary_csv(all_results, output_root, logger)

    logger.info("Done.")


if __name__ == "__main__":
    main()
