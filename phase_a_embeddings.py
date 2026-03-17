#!/usr/bin/env python3
"""Phase A — Embeddings: UMAP/t-SNE embedding, CSV construction, interestingness scoring, selective plotting.

Produces:
  - 117 CSVs with UMAP/t-SNE coordinates + all label variables
  - ~8,190 interestingness scores
  - ~213 selective plots (tiered: mandatory + score-driven + validation)

Usage:
  python phase_a_embeddings.py                                      # Full pipeline
  python phase_a_embeddings.py --skip-plots                         # CSVs + scores only
  python phase_a_embeddings.py --plot L3 16 all a_tens umap_2d      # On-demand single plot
"""

import argparse
import json
import logging
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score
from tqdm import tqdm

# GPU-accelerated UMAP/t-SNE via cuML, with CPU fallback
try:
    from cuml.manifold import UMAP as cuUMAP
    from cuml.manifold import TSNE as cuTSNE
    _GPU_AVAILABLE = True
except ImportError:
    _GPU_AVAILABLE = False

if not _GPU_AVAILABLE:
    from umap import UMAP as cpuUMAP
    from sklearn.manifold import TSNE as cpuTSNE

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

PLACE_NAMES = ["units", "tens", "hundreds", "thousands", "ten_thousands", "hundred_thousands"]
LEVELS = [1, 2, 3, 4, 5]
LAYERS = [4, 6, 8, 12, 16, 20, 24, 28, 31]
PLOT_LEVELS = [3, 5]
PLOT_LAYERS = [4, 16, 31]
MIN_POPULATION = 30  # skip populations smaller than this

# Variables that only exist for wrong answers (NaN for correct)
ERROR_VARS = {"abs_error", "rel_error", "signed_error", "underestimate", "even_pred", "div10_pred", "error_category"}
IDENTITY_COLS = {"problem_idx", "a", "b", "ground_truth", "predicted"}


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
        "phase_a_data": dr / "phase_a",
        "phase_a_plots": ws / "plots" / "phase_a",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(workspace):
    log_dir = workspace / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase_a_emb")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fh = RotatingFileHandler(log_dir / "phase_a_embeddings.log", maxBytes=10_000_000, backupCount=3)
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

def load_labels(level, labels_dir):
    with open(labels_dir / f"level_{level}.json") as f:
        return json.load(f)


def load_answers(level, answers_dir):
    with open(answers_dir / f"level_{level}.json") as f:
        return json.load(f)


def load_activations(level, layer, act_dir):
    path = act_dir / f"level{level}_layer{layer}.npy"
    return np.load(path)


# ═══════════════════════════════════════════════════════════════════════════════
# COLORING DATAFRAME
# ═══════════════════════════════════════════════════════════════════════════════

def build_coloring_df(level, labels_dir, answers_dir):
    """Build DataFrame with one row per problem, all coloring variables."""
    label_data = load_labels(level, labels_dir)
    answer_data = load_answers(level, answers_dir)
    problems = label_data["problems"]
    results = answer_data["results"]
    rows = []

    for idx, (prob, res) in enumerate(zip(problems, results)):
        L = prob["labels"]
        row = {"problem_idx": idx, "a": L["a"], "b": L["b"]}

        # Correctness and predictions
        row["correct"] = res["correct"]
        row["predicted"] = res["predicted"]
        row["ground_truth"] = L["product"]

        # Input digits by place value
        for place, val in L["a_decomposition"].items():
            if place != "num_digits":
                row[f"a_{place}"] = val
        for place, val in L["b_decomposition"].items():
            if place != "num_digits":
                row[f"b_{place}"] = val

        # Partial products
        for key, val in L["partial_products"].items():
            row[f"pp_{key}"] = val

        # Column sums
        for j, val in enumerate(L["column_sums"]):
            row[f"col_sum_{j}"] = val

        # Carries
        carries = L["carries"]
        for j, val in enumerate(carries):
            row[f"carry_{j}"] = val

        # Derived carry variables
        row["n_nonzero_carries"] = sum(1 for c in carries if c > 0)
        row["total_carry_sum"] = sum(carries)
        row["max_carry_value"] = max(carries)

        # Answer digits (MSF)
        for j, val in enumerate(L["answer_digits_msf"]):
            row[f"ans_digit_{j}_msf"] = val
        row["n_answer_digits"] = len(L["answer_digits_msf"])

        # Product
        row["product"] = L["product"]

        # Per-digit correctness — True for correct answers
        gt = L["product"]
        pred = res["predicted"]
        if res["correct"]:
            gt_s = str(gt)
            for pos in range(len(gt_s)):
                row[f"digit_correct_pos{pos}"] = True
        elif pred is not None:
            gt_s, pred_s = str(gt), str(pred)
            if len(gt_s) == len(pred_s):
                for pos in range(len(gt_s)):
                    row[f"digit_correct_pos{pos}"] = gt_s[pos] == pred_s[pos]

        # Error properties (only for wrong answers with valid prediction)
        if not res["correct"] and pred is not None:
            row["abs_error"] = abs(pred - gt)
            row["rel_error"] = abs(pred - gt) / gt if gt != 0 else np.nan
            row["signed_error"] = pred - gt
            row["underestimate"] = pred < gt
            row["even_pred"] = pred % 2 == 0
            row["div10_pred"] = pred % 10 == 0
            if len(str(pred)) != len(str(gt)):
                row["error_category"] = "magnitude_error"
            elif abs(pred - gt) / gt < 0.05:
                row["error_category"] = "close_arithmetic"
            else:
                row["error_category"] = "large_arithmetic"
        elif not res["correct"] and pred is None:
            row["error_category"] = "garbage"

        rows.append(row)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# POPULATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_populations(df, level):
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
# EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════════

def get_embedding_params(n_samples):
    """Return adjusted UMAP/t-SNE parameters based on population size."""
    if n_samples < 100:
        return {"n_neighbors": 15, "perplexity": 15}
    elif n_samples < 500:
        return {"n_neighbors": 20, "perplexity": 20}
    else:
        return {"n_neighbors": 30, "perplexity": 30}


def compute_embedding(X, method, params):
    """Compute embedding. method in {umap_2d, umap_3d, tsne_2d}.

    Uses cuML GPU implementations when available (~10-50x faster),
    falls back to CPU umap-learn / sklearn.
    """
    n = X.shape[0]
    X_f32 = np.ascontiguousarray(X, dtype=np.float32)

    if _GPU_AVAILABLE:
        if method == "umap_2d":
            model = cuUMAP(n_components=2, n_neighbors=params["n_neighbors"],
                           min_dist=0.1, metric="euclidean", random_state=42)
            return np.asarray(model.fit_transform(X_f32))
        elif method == "umap_3d":
            model = cuUMAP(n_components=3, n_neighbors=params["n_neighbors"],
                           min_dist=0.1, metric="euclidean", random_state=42)
            return np.asarray(model.fit_transform(X_f32))
        elif method == "tsne_2d":
            perp = min(params["perplexity"], (n - 1) // 3)
            model = cuTSNE(n_components=2, perplexity=perp,
                           learning_rate=200.0, max_iter=2000, random_state=42)
            return np.asarray(model.fit_transform(X_f32))
        else:
            raise ValueError(f"Unknown method: {method}")
    else:
        if method == "umap_2d":
            model = cpuUMAP(n_components=2, n_neighbors=params["n_neighbors"],
                            min_dist=0.1, metric="euclidean", random_state=42)
            return model.fit_transform(X_f32)
        elif method == "umap_3d":
            model = cpuUMAP(n_components=3, n_neighbors=params["n_neighbors"],
                            min_dist=0.1, metric="euclidean", random_state=42)
            return model.fit_transform(X_f32)
        elif method == "tsne_2d":
            perp = min(params["perplexity"], (n - 1) // 3)
            model = cpuTSNE(n_components=2, perplexity=perp, learning_rate="auto",
                            max_iter=2000, random_state=42, init="pca")
            return model.fit_transform(X_f32)
        else:
            raise ValueError(f"Unknown method: {method}")


def get_or_compute_embedding(X, method, params, save_path):
    """Load from disk if exists, else compute and save."""
    if save_path.exists():
        return np.load(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    emb = compute_embedding(X, method, params)
    np.save(save_path, emb)
    return emb


# ═══════════════════════════════════════════════════════════════════════════════
# CSV CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def build_and_save_csv(df_pop, embeddings, activation_norms, csv_path):
    """Combine coloring df with embeddings and norms, save as CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out = df_pop.reset_index(drop=True).copy()
    out["activation_norm"] = activation_norms

    for method, emb in embeddings.items():
        if method == "umap_2d":
            out["umap_2d_x"] = emb[:, 0]
            out["umap_2d_y"] = emb[:, 1]
        elif method == "umap_3d":
            out["umap_3d_x"] = emb[:, 0]
            out["umap_3d_y"] = emb[:, 1]
            out["umap_3d_z"] = emb[:, 2]
        elif method == "tsne_2d":
            out["tsne_2d_x"] = emb[:, 0]
            out["tsne_2d_y"] = emb[:, 1]

    out.to_csv(csv_path, index=False)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# INTERESTINGNESS SCORING
# ═══════════════════════════════════════════════════════════════════════════════

EMBEDDING_COLS = {"umap_2d_x", "umap_2d_y", "umap_3d_x", "umap_3d_y", "umap_3d_z",
                  "tsne_2d_x", "tsne_2d_y"}
SKIP_SCORING = IDENTITY_COLS | EMBEDDING_COLS | {"activation_norm"}

# Variables where angular correlation should also be computed (digit-valued 0-9)
DIGIT_PREFIXES = ("a_units", "a_tens", "a_hundreds", "b_units", "b_tens", "b_hundreds",
                  "ans_digit_")


def classify_variable(name, values):
    """Return (metric_type, is_digit) for a variable."""
    nunique = values.nunique()
    if name == "correct" or name.startswith("digit_correct_") or name == "underestimate" or name == "even_pred" or name == "div10_pred":
        return "silhouette", False
    if name == "error_category":
        return "silhouette", False
    if nunique <= 20:
        is_digit = any(name.startswith(p) for p in DIGIT_PREFIXES)
        return "silhouette", is_digit
    return "spearman", False


def compute_interestingness_score(values, emb_2d, metric_type):
    """Compute a single interestingness score."""
    valid = values.notna()
    if valid.sum() < 30:
        return np.nan
    v = values[valid].values
    e = emb_2d[valid.values]

    if metric_type == "silhouette":
        labels = pd.Categorical(v).codes
        n_labels = len(set(labels))
        if n_labels < 2 or n_labels >= len(labels):
            return np.nan
        return silhouette_score(e, labels)
    elif metric_type == "spearman":
        r_x = abs(spearmanr(e[:, 0], v, nan_policy="omit").statistic)
        r_y = abs(spearmanr(e[:, 1], v, nan_policy="omit").statistic)
        return max(r_x, r_y)
    return np.nan


def compute_angular_score(values, emb_2d):
    """Angular correlation: Spearman between arctan2 of embedding and digit value."""
    valid = values.notna()
    if valid.sum() < 30:
        return np.nan
    v = values[valid].values
    e = emb_2d[valid.values]
    cx, cy = e[:, 0].mean(), e[:, 1].mean()
    theta = np.arctan2(e[:, 1] - cy, e[:, 0] - cx)
    return abs(spearmanr(theta, v, nan_policy="omit").statistic)


def score_single_csv(csv_path):
    """Score all variables in a CSV for interestingness. Returns list of dicts."""
    df = pd.read_csv(csv_path)
    stem = csv_path.stem  # e.g. L3_layer16_all
    parts = stem.split("_")
    level = int(parts[0][1:])
    layer = int(parts[1].replace("layer", ""))
    pop = parts[2]

    results = []
    # Score against both UMAP 2D and t-SNE 2D
    for method, xcol, ycol in [("umap_2d", "umap_2d_x", "umap_2d_y"),
                                ("tsne_2d", "tsne_2d_x", "tsne_2d_y")]:
        if xcol not in df.columns:
            continue
        emb_2d = df[[xcol, ycol]].values

        for col in df.columns:
            if col in SKIP_SCORING or col in EMBEDDING_COLS:
                continue
            if col == "correct" and pop != "all":
                continue  # constant in correct/wrong pops

            metric_type, is_digit = classify_variable(col, df[col])
            score = compute_interestingness_score(df[col], emb_2d, metric_type)

            results.append({
                "level": level, "layer": layer, "population": pop,
                "method": method, "variable": col,
                "metric_type": metric_type, "score": score,
            })

            # Angular correlation for digit variables
            if is_digit:
                ang_score = compute_angular_score(df[col], emb_2d)
                results.append({
                    "level": level, "layer": layer, "population": pop,
                    "method": method, "variable": col,
                    "metric_type": "angular", "score": ang_score,
                })

    return results


def score_all_csvs(csv_dir, logger):
    """Score all CSVs and return a DataFrame of interestingness scores."""
    csv_files = sorted(csv_dir.glob("*.csv"))
    logger.info(f"Scoring {len(csv_files)} CSVs...")
    all_scores = []
    for csv_path in tqdm(csv_files, desc="Scoring CSVs"):
        all_scores.extend(score_single_csv(csv_path))
    return pd.DataFrame(all_scores)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def get_point_style(n):
    if n >= 3000:
        return 3, 0.5
    elif n >= 1000:
        return 5, 0.6
    elif n >= 200:
        return 10, 0.7
    else:
        return 20, 0.8


def get_cmap(col, values):
    """Choose colormap based on variable type."""
    if col == "correct" or col.startswith("digit_correct_") or col in ("underestimate", "even_pred", "div10_pred"):
        return "coolwarm"
    if col == "error_category":
        return "tab10"
    if col == "signed_error":
        return "coolwarm"
    nunique = values.nunique() if hasattr(values, "nunique") else len(set(values))
    if nunique <= 10:
        return "tab10"
    return "viridis"


def plot_2d_scatter(emb, colors, cmap, title, xlabel, ylabel, colorbar_label, save_path):
    """Generate a single 2D scatter plot."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(emb)
    s, alpha = get_point_style(n)

    fig, ax = plt.subplots(figsize=(10, 8))

    valid = pd.notna(colors)
    c_valid = colors[valid] if hasattr(colors, '__getitem__') else np.array(colors)[valid]
    e_valid = emb[valid]

    # Handle categorical coloring
    if cmap == "tab10" and hasattr(c_valid, "dtype") and not np.issubdtype(c_valid.dtype, np.number):
        cats = sorted(set(c_valid))
        cat_map = {c: i for i, c in enumerate(cats)}
        c_numeric = np.array([cat_map[c] for c in c_valid])
        scatter = ax.scatter(e_valid[:, 0], e_valid[:, 1], c=c_numeric,
                             cmap=cmap, s=s, alpha=alpha, edgecolors="none", rasterized=True)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_ticks(range(len(cats)))
        cbar.set_ticklabels(cats)
        cbar.set_label(colorbar_label)
    else:
        c_arr = np.array(c_valid, dtype=float)
        scatter = ax.scatter(e_valid[:, 0], e_valid[:, 1], c=c_arr,
                             cmap=cmap, s=s, alpha=alpha, edgecolors="none", rasterized=True)
        plt.colorbar(scatter, ax=ax, label=colorbar_label)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_3d_interactive(df, color_col, title, save_path):
    """Generate interactive 3D plotly scatter."""
    import plotly.express as px
    save_path.parent.mkdir(parents=True, exist_ok=True)

    valid = df[color_col].notna()
    df_valid = df[valid].copy()

    fig = px.scatter_3d(df_valid, x="umap_3d_x", y="umap_3d_y", z="umap_3d_z",
                        color=color_col, opacity=0.5, title=title)
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    fig.write_html(save_path, include_plotlyjs="cdn")


def generate_single_plot(csv_path, variable, method, plot_dir):
    """Generate one plot from a CSV. Returns save path."""
    df = pd.read_csv(csv_path)
    stem = csv_path.stem
    parts = stem.split("_")
    level_str, layer_str, pop = parts[0], parts[1], parts[2]

    if variable not in df.columns:
        return None

    title = f"{level_str} {layer_str} | {pop} | {method} | {variable}"
    cmap = get_cmap(variable, df[variable])

    if method == "umap_2d":
        emb = df[["umap_2d_x", "umap_2d_y"]].values
        save_path = plot_dir / level_str / layer_str / pop / "umap_2d" / f"{variable}.png"
        plot_2d_scatter(emb, df[variable].values, cmap, title,
                        "UMAP 1", "UMAP 2", variable, save_path)
        return save_path
    elif method == "tsne_2d":
        emb = df[["tsne_2d_x", "tsne_2d_y"]].values
        save_path = plot_dir / "tsne_validation" / f"{stem}_{variable}.png"
        plot_2d_scatter(emb, df[variable].values, cmap, title,
                        "t-SNE 1", "t-SNE 2", variable, save_path)
        return save_path
    elif method == "umap_3d":
        save_path = plot_dir / "umap_3d" / f"{stem}_{variable}.html"
        plot_3d_interactive(df, variable, title, save_path)
        return save_path
    return None


def generate_mandatory_plots(csv_dir, plot_dir, logger):
    """Tier 1: mandatory UMAP 2D plots for PLOT_LEVELS x PLOT_LAYERS."""
    mandatory_prefixes = {
        "all": ["correct", "n_nonzero_carries", "activation_norm",
                "a_units", "a_tens", "a_hundreds", "b_units", "b_tens", "b_hundreds"],
        "correct": ["a_units", "a_tens", "a_hundreds", "b_units", "b_tens", "b_hundreds"],
        "wrong": ["a_units", "a_tens", "a_hundreds", "b_units", "b_tens", "b_hundreds"],
    }
    count = 0
    for level in PLOT_LEVELS:
        for layer in PLOT_LAYERS:
            for pop in ["all", "correct", "wrong"]:
                csv_path = csv_dir / f"L{level}_layer{layer:02d}_{pop}.csv"
                if not csv_path.exists():
                    continue
                df = pd.read_csv(csv_path)
                for var in mandatory_prefixes[pop]:
                    if var in df.columns and df[var].notna().sum() >= 30:
                        generate_single_plot(csv_path, var, "umap_2d", plot_dir)
                        count += 1
    logger.info(f"Tier 1 mandatory: {count} plots")
    return count


def generate_score_driven_plots(csv_dir, plot_dir, scores_df, logger):
    """Tier 2: top 5 by interestingness score per (level, layer, pop), UMAP 2D only."""
    count = 0
    for level in PLOT_LEVELS:
        for layer in PLOT_LAYERS:
            for pop in ["all", "correct", "wrong"]:
                csv_path = csv_dir / f"L{level}_layer{layer:02d}_{pop}.csv"
                if not csv_path.exists():
                    continue

                mask = ((scores_df["level"] == level) & (scores_df["layer"] == layer) &
                        (scores_df["population"] == pop) & (scores_df["method"] == "umap_2d") &
                        (scores_df["metric_type"] != "angular"))
                sub = scores_df[mask].dropna(subset=["score"]).sort_values("score", ascending=False)

                # Skip variables already in mandatory set
                mandatory_vars = {"correct", "n_nonzero_carries", "activation_norm",
                                  "a_units", "a_tens", "a_hundreds", "b_units", "b_tens", "b_hundreds"}
                seen = set()
                for _, row in sub.iterrows():
                    if row["variable"] in mandatory_vars:
                        continue
                    if row["variable"] in seen:
                        continue
                    seen.add(row["variable"])
                    plot_path = plot_dir / f"L{level}" / f"layer_{layer:02d}" / pop / "umap_2d" / f"{row['variable']}.png"
                    if not plot_path.exists():
                        generate_single_plot(csv_path, row["variable"], "umap_2d", plot_dir)
                        count += 1
                    if len(seen) >= 5:
                        break
    logger.info(f"Tier 2 score-driven: {count} plots")
    return count


def generate_tsne_validation(csv_dir, plot_dir, scores_df, logger, top_n=30):
    """Tier 3: t-SNE 2D for top N findings overall."""
    mask = (scores_df["method"] == "umap_2d") & (scores_df["metric_type"] != "angular")
    top = scores_df[mask].dropna(subset=["score"]).nlargest(top_n, "score")

    count = 0
    for _, row in top.iterrows():
        csv_path = csv_dir / f"L{row['level']}_layer{row['layer']:02d}_{row['population']}.csv"
        if not csv_path.exists():
            continue
        generate_single_plot(csv_path, row["variable"], "tsne_2d", plot_dir)
        count += 1
    logger.info(f"Tier 3 t-SNE validation: {count} plots")
    return count


def generate_3d_exploration(csv_dir, plot_dir, scores_df, logger, top_n=15):
    """Tier 4: UMAP 3D interactive HTML for top N findings."""
    mask = (scores_df["method"] == "umap_2d") & (scores_df["metric_type"] != "angular")
    top = scores_df[mask].dropna(subset=["score"]).nlargest(top_n, "score")

    count = 0
    for _, row in top.iterrows():
        csv_path = csv_dir / f"L{row['level']}_layer{row['layer']:02d}_{row['population']}.csv"
        if not csv_path.exists():
            continue
        generate_single_plot(csv_path, row["variable"], "umap_3d", plot_dir)
        count += 1
    logger.info(f"Tier 4 3D exploration: {count} plots")
    return count


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY HEATMAPS
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_heatmap(heatmap_data, title, save_path, vmin=0, vmax=1, cmap="YlOrRd",
                  cbar_label="Score", diverging=False):
    """Render a single variable × layer heatmap and save."""
    if heatmap_data.empty:
        return
    fig, ax = plt.subplots(figsize=(12, max(6, len(heatmap_data) * 0.3)))
    kwargs = {"aspect": "auto", "cmap": cmap, "vmin": vmin, "vmax": vmax}
    im = ax.imshow(heatmap_data.values, **kwargs)
    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns)
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index, fontsize=7)
    ax.set_xlabel("Layer")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=cbar_label)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_interestingness_heatmaps(scores_df, plot_dir, logger):
    """Per-population heatmaps + correct-vs-wrong difference heatmap per level."""
    heatmap_dir = plot_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    mask = (scores_df["method"] == "umap_2d") & (scores_df["metric_type"] != "angular")
    sub = scores_df[mask].copy()
    n_plots = 0

    for level in LEVELS:
        ldf = sub[sub["level"] == level]
        if ldf.empty:
            continue

        # One heatmap per population
        pop_pivots = {}
        for pop in ["all", "correct", "wrong"]:
            pdf = ldf[ldf["population"] == pop]
            if pdf.empty:
                continue
            pivot = pdf.pivot_table(index="variable", columns="layer", values="score", aggfunc="first")
            if pivot.empty:
                continue
            pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]
            _plot_heatmap(pivot, f"L{level} — UMAP Interestingness ({pop})",
                          heatmap_dir / f"L{level}_{pop}.png")
            pop_pivots[pop] = pivot
            n_plots += 1

        # Correct-vs-wrong difference heatmap
        if "correct" in pop_pivots and "wrong" in pop_pivots:
            shared_vars = pop_pivots["correct"].index.intersection(pop_pivots["wrong"].index)
            shared_layers = pop_pivots["correct"].columns.intersection(pop_pivots["wrong"].columns)
            if len(shared_vars) > 0 and len(shared_layers) > 0:
                diff = (pop_pivots["correct"].loc[shared_vars, shared_layers]
                        - pop_pivots["wrong"].loc[shared_vars, shared_layers])
                diff = diff.loc[diff.abs().max(axis=1).sort_values(ascending=False).index]
                vabs = max(0.3, float(diff.abs().max().max()))
                _plot_heatmap(diff, f"L{level} — Score Difference (correct − wrong)",
                              heatmap_dir / f"L{level}_correct_minus_wrong.png",
                              vmin=-vabs, vmax=vabs, cmap="RdBu_r",
                              cbar_label="Δ score (correct − wrong)")
                n_plots += 1

    logger.info(f"Generated {n_plots} interestingness heatmaps")


def generate_top_findings(scores_df, scores_dir, logger, top_n=50):
    """Save top N findings to markdown."""
    mask = (scores_df["method"] == "umap_2d") & (scores_df["metric_type"] != "angular")
    top = scores_df[mask].dropna(subset=["score"]).nlargest(top_n, "score")

    lines = ["# Top 50 UMAP Interestingness Findings\n"]
    lines.append("| Rank | Level | Layer | Pop | Variable | Metric | Score |")
    lines.append("|------|-------|-------|-----|----------|--------|-------|")
    for i, (_, row) in enumerate(top.iterrows(), 1):
        lines.append(f"| {i} | L{row['level']} | {row['layer']} | {row['population']} | "
                     f"{row['variable']} | {row['metric_type']} | {row['score']:.4f} |")

    out_path = scores_dir / "top_50_findings.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Top {top_n} findings saved to {out_path}")


def generate_correct_wrong_comparison(scores_df, scores_dir, logger):
    """Table of score differences between correct and wrong populations.

    Shows which concepts degrade or improve between populations — one of the
    most important outputs for Paper 1.
    """
    mask = (scores_df["method"] == "umap_2d") & (scores_df["metric_type"] != "angular")
    sub = scores_df[mask].dropna(subset=["score"])

    correct = sub[sub["population"] == "correct"][["level", "layer", "variable", "score"]]
    wrong = sub[sub["population"] == "wrong"][["level", "layer", "variable", "score"]]

    merged = correct.merge(wrong, on=["level", "layer", "variable"], suffixes=("_correct", "_wrong"))
    if merged.empty:
        logger.info("No correct/wrong overlap for comparison table")
        return

    merged["delta"] = merged["score_correct"] - merged["score_wrong"]
    merged["abs_delta"] = merged["delta"].abs()
    merged = merged.sort_values("abs_delta", ascending=False)

    lines = ["# Correct vs Wrong — Score Comparison\n"]
    lines.append("Positive Δ = stronger structure in correct population.\n")
    lines.append("| Level | Layer | Variable | Correct | Wrong | Δ (C−W) |")
    lines.append("|-------|-------|----------|---------|-------|---------|")
    for _, row in merged.head(50).iterrows():
        lines.append(f"| L{row['level']:.0f} | {row['layer']:.0f} | {row['variable']} | "
                     f"{row['score_correct']:.4f} | {row['score_wrong']:.4f} | "
                     f"{row['delta']:+.4f} |")

    out_path = scores_dir / "correct_wrong_comparison.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    # Also save as CSV for downstream use
    merged.to_csv(scores_dir / "correct_wrong_comparison.csv", index=False)
    logger.info(f"Correct/wrong comparison: {len(merged)} variable pairs, "
                f"top |Δ| = {merged['abs_delta'].iloc[0]:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase A — Embeddings pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--plot", nargs=5, metavar=("LEVEL", "LAYER", "POP", "VAR", "METHOD"),
                        help="On-demand: generate a single plot (e.g., L3 16 all a_tens umap_2d)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = derive_paths(cfg)
    logger = setup_logging(paths["workspace"])
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("Phase A — Embeddings pipeline starting")
    logger.info(f"  Embedding backend: {'cuML GPU' if _GPU_AVAILABLE else 'CPU (umap-learn + sklearn)'}")
    logger.info("=" * 70)

    csv_dir = paths["phase_a_data"] / "csvs"
    emb_dir = paths["phase_a_data"] / "embeddings"
    scores_dir = paths["phase_a_data"] / "scores"
    pkl_dir = paths["phase_a_data"] / "coloring_dfs"
    plot_dir = paths["phase_a_plots"]

    # ── ON-DEMAND SINGLE PLOT ──────────────────────────────────────────────
    if args.plot:
        level_str, layer_str, pop, var, method = args.plot
        level = int(level_str.replace("L", ""))
        layer = int(layer_str)
        csv_path = csv_dir / f"L{level}_layer{layer:02d}_{pop}.csv"
        if not csv_path.exists():
            print(f"CSV not found: {csv_path}")
            sys.exit(1)
        out = generate_single_plot(csv_path, var, method, plot_dir)
        if out:
            print(f"Plot saved: {out}")
        else:
            print(f"Could not generate plot (variable '{var}' not in CSV or method '{method}' unavailable)")
        return

    # ── STEP 1: BUILD COLORING DATAFRAMES ──────────────────────────────────
    logger.info("Step 1: Building coloring DataFrames...")
    coloring_dfs = {}
    for level in LEVELS:
        pkl_path = pkl_dir / f"L{level}_coloring.pkl"
        if pkl_path.exists():
            coloring_dfs[level] = pd.read_pickle(pkl_path)
            logger.info(f"  L{level}: loaded from cache ({len(coloring_dfs[level])} rows)")
        else:
            pkl_dir.mkdir(parents=True, exist_ok=True)
            coloring_dfs[level] = build_coloring_df(level, paths["labels_dir"], paths["answers_dir"])
            coloring_dfs[level].to_pickle(pkl_path)
            logger.info(f"  L{level}: built and cached ({len(coloring_dfs[level])} rows, "
                        f"{len(coloring_dfs[level].columns)} cols)")
    logger.info(f"Step 1 done: {sum(len(df) for df in coloring_dfs.values())} total problems")

    # ── STEP 2 & 3: EMBEDDINGS + CSVs ─────────────────────────────────────
    logger.info("Steps 2-3: Computing embeddings and building CSVs...")
    methods = ["umap_2d", "umap_3d", "tsne_2d"]
    total_emb = 0
    total_csv = 0

    for level in LEVELS:
        df = coloring_dfs[level]
        for layer in LAYERS:
            X_full = load_activations(level, layer, paths["act_dir"])
            pops = get_populations(df, level)

            for pop_name, df_pop in pops.items():
                csv_path = csv_dir / f"L{level}_layer{layer:02d}_{pop_name}.csv"

                if csv_path.exists():
                    total_csv += 1
                    total_emb += 3
                    continue

                t_combo = time.time()

                # Slice activations to population
                idx = df_pop.index.values
                X_pop = X_full[idx]
                n = X_pop.shape[0]
                params = get_embedding_params(n)

                # Compute embeddings
                embeddings = {}
                for method in methods:
                    t_emb = time.time()
                    emb_path = emb_dir / f"L{level}" / f"layer_{layer:02d}" / f"{pop_name}_{method}.npy"
                    embeddings[method] = get_or_compute_embedding(X_pop, method, params, emb_path)
                    total_emb += 1
                    logger.debug(f"    {method}: {time.time() - t_emb:.1f}s")

                # Activation norms
                norms = np.linalg.norm(X_pop, axis=1)

                # Build and save CSV
                build_and_save_csv(df_pop, embeddings, norms, csv_path)
                total_csv += 1
                logger.info(f"  L{level} layer {layer:02d} {pop_name}: {n} points, "
                            f"{time.time() - t_combo:.1f}s")

    logger.info(f"Steps 2-3 done: {total_emb} embeddings, {total_csv} CSVs")

    # ── STEP 4: INTERESTINGNESS SCORES ─────────────────────────────────────
    scores_path = scores_dir / "interestingness_scores.csv"
    if scores_path.exists():
        logger.info("Step 4: Loading cached interestingness scores...")
        scores_df = pd.read_csv(scores_path)
    else:
        logger.info("Step 4: Computing interestingness scores...")
        scores_df = score_all_csvs(csv_dir, logger)
        scores_dir.mkdir(parents=True, exist_ok=True)
        scores_df.to_csv(scores_path, index=False)
    logger.info(f"Step 4 done: {len(scores_df)} scores ({scores_df['score'].notna().sum()} valid)")

    # ── STEP 5: SUMMARIES ──────────────────────────────────────────────────
    logger.info("Step 5: Generating summaries...")
    generate_interestingness_heatmaps(scores_df, plot_dir, logger)
    generate_top_findings(scores_df, scores_dir, logger)
    generate_correct_wrong_comparison(scores_df, scores_dir, logger)
    logger.info("Step 5 done")

    # ── STEP 6: PLOTS ─────────────────────────────────────────────────────
    if not args.skip_plots:
        logger.info("Step 6: Generating plots (tiered)...")
        n1 = generate_mandatory_plots(csv_dir, plot_dir, logger)
        n2 = generate_score_driven_plots(csv_dir, plot_dir, scores_df, logger)
        n3 = generate_tsne_validation(csv_dir, plot_dir, scores_df, logger)
        n4 = generate_3d_exploration(csv_dir, plot_dir, scores_df, logger)
        logger.info(f"Step 6 done: {n1 + n2 + n3 + n4} total plots")
    else:
        logger.info("Step 6: Skipped (--skip-plots)")

    elapsed = time.time() - t0
    logger.info(f"Phase A embeddings pipeline complete in {elapsed:.1f}s ({elapsed / 60:.1f}m)")


if __name__ == "__main__":
    main()
