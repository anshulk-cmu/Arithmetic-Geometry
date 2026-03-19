#!/usr/bin/env python3
"""Phase A — Pre-flight Diagnostics: two lightweight checks before Phase C/D.

Analyses:
  1. Activation norm profile — mean/std of L2 norms per level × layer × population.
     Tells you whether late-layer UMAP is norm-dominated and whether to normalize
     before Phase C.
  2. Cross-layer CKA matrices — 9×9 linear CKA per level. Tells you which layers
     are redundant so you don't waste time running Phase C on all 9 layers when
     layers 20/24/28 are essentially identical.

Everything else (eigenspectrum, Spearman sweep, ANOVA, Fourier probe, LDA separation,
subspace overlap, principal angles) belongs in Phase C/D/Fourier where it will be done
properly with the right framing, null models, and connection to the paper narrative.

Usage:
  python phase_a_analysis.py                        # Run both analyses
  python phase_a_analysis.py --config custom.yaml
  python phase_a_analysis.py --only norms           # Just norm profile
  python phase_a_analysis.py --only cka             # Just CKA matrices

All outputs go to /data/.../phase_a/analysis/ (JSON).
All plots go to /home/.../plots/phase_a/analysis/ (PNG).
"""

import argparse
import json
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

LEVELS = [1, 2, 3, 4, 5]
LAYERS = [4, 6, 8, 12, 16, 20, 24, 28, 31]
CKA_SUBSAMPLE = 1000


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def derive_paths(cfg):
    ws = Path(cfg["paths"]["workspace"])
    dr = Path(cfg["paths"]["data_root"])
    return {
        "workspace": ws,
        "data_root": dr,
        "answers_dir": dr / "answers",
        "act_dir": dr / "activations",
        "analysis_out": dr / "phase_a" / "analysis",
        "analysis_plots": ws / "plots" / "phase_a" / "analysis",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(workspace):
    log_dir = workspace / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase_a_analysis")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fh = RotatingFileHandler(log_dir / "phase_a_analysis.log", maxBytes=10_000_000, backupCount=3)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
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


def get_correctness_mask(level, answers_dir):
    """Return boolean array: True for correct answers."""
    with open(answers_dir / f"level_{level}.json") as f:
        data = json.load(f)
    return np.array([r["correct"] for r in data["results"]])


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: ACTIVATION NORM PROFILE
# ═══════════════════════════════════════════════════════════════════════════════

def analysis_norm_profile(paths, logger):
    """Mean/std of L2 norms per level × layer × population (all, correct, wrong)."""
    logger.info("Analysis 1: Activation Norm Profile...")
    results = {}

    for level in tqdm(LEVELS, desc="Norm profile"):
        correct_mask = get_correctness_mask(level, paths["answers_dir"])
        has_wrong = np.sum(~correct_mask) >= 30
        has_correct = np.sum(correct_mask) >= 30

        for layer in LAYERS:
            X = load_activations(level, layer, paths["act_dir"])
            norms = np.linalg.norm(X, axis=1)

            entry = {
                "level": level, "layer": layer,
                "all_mean": float(norms.mean()),
                "all_std": float(norms.std()),
                "all_median": float(np.median(norms)),
            }

            if has_correct:
                nc = norms[correct_mask]
                entry["correct_mean"] = float(nc.mean())
                entry["correct_std"] = float(nc.std())

            if has_wrong:
                nw = norms[~correct_mask]
                entry["wrong_mean"] = float(nw.mean())
                entry["wrong_std"] = float(nw.std())

            if has_correct and has_wrong:
                entry["norm_ratio_correct_over_wrong"] = float(nc.mean() / (nw.mean() + 1e-12))

            results[f"L{level}_layer{layer}"] = entry

    logger.info(f"  Done: {len(results)} entries")
    return results


def plot_norm_profile(results, plot_dir, logger):
    """Norm vs layer, one line per level, with correct/wrong split for L3-L5."""
    plot_dir.mkdir(parents=True, exist_ok=True)

    # All populations
    fig, ax = plt.subplots(figsize=(10, 6))
    for level in LEVELS:
        layers_vals = []
        for layer in LAYERS:
            key = f"L{level}_layer{layer}"
            if key in results:
                layers_vals.append((layer, results[key]["all_mean"], results[key]["all_std"]))
        if layers_vals:
            ls, means, stds = zip(*layers_vals)
            ax.errorbar(ls, means, yerr=stds, fmt="o-", label=f"L{level}", markersize=5, capsize=3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 Norm (mean ± std)")
    ax.set_title("Activation Norm Profile — All Problems")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "norm_profile_all.png", dpi=150)
    plt.close()

    # Correct vs wrong for L3-L5
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for idx, level in enumerate([3, 4, 5]):
        ax = axes[idx]
        for pop, style, color in [("correct", "o-", "tab:green"), ("wrong", "s--", "tab:red")]:
            vals = []
            for layer in LAYERS:
                key = f"L{level}_layer{layer}"
                mean_key = f"{pop}_mean"
                if key in results and mean_key in results[key]:
                    vals.append((layer, results[key][mean_key]))
            if vals:
                ls, ms = zip(*vals)
                ax.plot(ls, ms, style, color=color, label=pop, markersize=5)
        ax.set_xlabel("Layer")
        ax.set_title(f"L{level}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("L2 Norm (mean)")
    fig.suptitle("Activation Norms — Correct vs Wrong", fontsize=12)
    plt.tight_layout()
    plt.savefig(plot_dir / "norm_profile_correct_wrong.png", dpi=150)
    plt.close()

    logger.info("  Plotted norm_profile_all.png and norm_profile_correct_wrong.png")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: CROSS-LAYER CKA
# ═══════════════════════════════════════════════════════════════════════════════

def linear_cka(X, Y):
    """Compute linear CKA between two (n, d) matrices using Gram (n x n) approach.

    CKA = HSIC(K_X, K_Y) / sqrt(HSIC(K_X, K_X) * HSIC(K_Y, K_Y))
    where K = X @ X.T (Gram matrix), HSIC = ||H @ K @ H||_F^2 / n^2, H = I - 11'/n.
    Simplified: HSIC(K_X, K_Y) = tr(K_X_c @ K_Y_c) where K_c = H @ K @ H.
    """
    X_c = X - X.mean(axis=0)
    Y_c = Y - Y.mean(axis=0)
    K_X = X_c @ X_c.T
    K_Y = Y_c @ Y_c.T
    K_X -= K_X.mean(axis=0, keepdims=True)
    K_X -= K_X.mean(axis=1, keepdims=True)
    K_Y -= K_Y.mean(axis=0, keepdims=True)
    K_Y -= K_Y.mean(axis=1, keepdims=True)
    hsic_xy = np.sum(K_X * K_Y)
    hsic_xx = np.sum(K_X * K_X)
    hsic_yy = np.sum(K_Y * K_Y)
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def analysis_cka_matrices(paths, logger):
    """9×9 linear CKA between all layer pairs, per level.

    For large datasets (L5 = 122K), we determine the subsample indices first
    and load only the subsampled rows via np.load with mmap_mode, avoiding
    loading all 9 full activation files (~40 GB for L5) into RAM.
    """
    logger.info("Analysis 2: Cross-Layer CKA Matrices...")
    results = {}
    rng = np.random.RandomState(42)

    for level in tqdm(LEVELS, desc="CKA"):
        # Determine n_full from the first layer file shape (via mmap, no RAM)
        first_path = paths["act_dir"] / f"level{level}_layer{LAYERS[0]}.npy"
        n_full = np.load(first_path, mmap_mode="r").shape[0]

        n_sub = min(CKA_SUBSAMPLE, n_full)
        idx = rng.choice(n_full, n_sub, replace=False) if n_sub < n_full else np.arange(n_full)
        idx_sorted = np.sort(idx)  # sorted for sequential disk access

        # Load only subsampled rows per layer
        layer_data = {}
        for layer in LAYERS:
            X_mmap = np.load(paths["act_dir"] / f"level{level}_layer{layer}.npy", mmap_mode="r")
            layer_data[layer] = np.array(X_mmap[idx_sorted], dtype=np.float64)

        logger.debug(f"  L{level}: loaded {n_sub} subsampled rows from {len(LAYERS)} layers "
                     f"(full dataset: {n_full})")

        cka_matrix = np.zeros((len(LAYERS), len(LAYERS)))
        for i, l1 in enumerate(LAYERS):
            for j, l2 in enumerate(LAYERS):
                if i <= j:
                    cka_val = linear_cka(layer_data[l1], layer_data[l2])
                    cka_matrix[i, j] = cka_val
                    cka_matrix[j, i] = cka_val

        # Identify redundant pairs (CKA > 0.98)
        redundant_pairs = []
        for i in range(len(LAYERS)):
            for j in range(i + 1, len(LAYERS)):
                if cka_matrix[i, j] > 0.98:
                    redundant_pairs.append((LAYERS[i], LAYERS[j], round(float(cka_matrix[i, j]), 4)))

        results[f"L{level}"] = {
            "level": level, "n_subsample": int(n_sub),
            "layers": LAYERS,
            "cka_matrix": cka_matrix.tolist(),
            "redundant_pairs": redundant_pairs,
        }

    logger.info(f"  Done: {len(results)} CKA matrices")
    return results


def plot_cka_matrices(results, plot_dir, logger):
    """One CKA heatmap per level."""
    plot_dir.mkdir(parents=True, exist_ok=True)

    for level in LEVELS:
        key = f"L{level}"
        if key not in results:
            continue
        matrix = np.array(results[key]["cka_matrix"])
        layers = results[key]["layers"]

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer")
        ax.set_title(f"L{level} — Linear CKA (n={results[key]['n_subsample']})")

        for i in range(len(layers)):
            for j in range(len(layers)):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if matrix[i, j] < 0.5 else "black")

        plt.colorbar(im, ax=ax, label="CKA")
        plt.tight_layout()
        plt.savefig(plot_dir / f"cka_L{level}.png", dpi=150)
        plt.close()

    logger.info(f"  Plotted CKA matrices")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

ALL_ANALYSES = ["norms", "cka"]


def main():
    parser = argparse.ArgumentParser(description="Phase A — Pre-flight Diagnostics")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated list of analyses to run (norms, cka)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = derive_paths(cfg)
    logger = setup_logging(paths["workspace"])
    t0 = time.time()

    logger.info("=" * 70)
    logger.info("Phase A — Pre-flight Diagnostics starting")
    logger.info("=" * 70)

    paths["analysis_out"].mkdir(parents=True, exist_ok=True)
    paths["analysis_plots"].mkdir(parents=True, exist_ok=True)

    analyses_to_run = args.only.split(",") if args.only else ALL_ANALYSES

    # ── NORM PROFILE ──────────────────────────────────────────────────────
    if "norms" in analyses_to_run:
        out_path = paths["analysis_out"] / "norm_profile.json"
        if out_path.exists():
            logger.info("Norm profile: loading cached results...")
            with open(out_path) as f:
                norm_results = json.load(f)
        else:
            norm_results = analysis_norm_profile(paths, logger)
            with open(out_path, "w") as f:
                json.dump(norm_results, f, indent=2)
        plot_norm_profile(norm_results, paths["analysis_plots"], logger)

        # Log norm ratio summary for the "is UMAP norm-dominated?" diagnostic
        for level in [3, 4, 5]:
            for layer in [16, 24, 31]:
                key = f"L{level}_layer{layer}"
                entry = norm_results.get(key, {})
                if "correct_mean" in entry and "wrong_mean" in entry:
                    ratio = entry.get("norm_ratio_correct_over_wrong", 0)
                    logger.info(f"  {key}: correct norm {entry['correct_mean']:.1f} "
                                f"vs wrong {entry['wrong_mean']:.1f}, ratio {ratio:.3f}")

    # ── CKA MATRICES ─────────────────────────────────────────────────────
    if "cka" in analyses_to_run:
        out_path = paths["analysis_out"] / "cka_matrices.json"
        if out_path.exists():
            logger.info("CKA matrices: loading cached results...")
            with open(out_path) as f:
                cka_results = json.load(f)
        else:
            cka_results = analysis_cka_matrices(paths, logger)
            with open(out_path, "w") as f:
                json.dump(cka_results, f, indent=2)
        plot_cka_matrices(cka_results, paths["analysis_plots"], logger)

        # Log redundant pairs
        for key, entry in cka_results.items():
            if entry.get("redundant_pairs"):
                logger.info(f"  {key}: {len(entry['redundant_pairs'])} redundant layer pairs (CKA > 0.98)")
                for l1, l2, cka_val in entry["redundant_pairs"]:
                    logger.info(f"    layers {l1} & {l2}: CKA = {cka_val}")

    elapsed = time.time() - t0
    logger.info(f"Phase A diagnostics complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
