#!/usr/bin/env python3
"""
Arithmetic Geometry Error Analysis

Analyzes wrong answers from the multiplication pipeline to find
systematic patterns: per-digit accuracy, carry correlation,
error structure, and input difficulty.

No GPU required. Reads saved answers + labels, produces 6 plots
and a JSON summary.

Usage:
    python analysis.py
    python analysis.py --config config.yaml
"""

import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from logging.handlers import RotatingFileHandler
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


# ====================================================================
# 1. CONFIG + LOGGING
# ====================================================================

def load_config(config_path=None):
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ws = Path(cfg["paths"]["workspace"])
    dr = Path(cfg["paths"]["data_root"])
    cfg["paths"]["labels_dir"] = str(ws / "labels")
    cfg["paths"]["logs_dir"] = str(ws / "logs")
    cfg["paths"]["plots_dir"] = str(ws / "plots")
    cfg["paths"]["activations_dir"] = str(dr / "activations")
    cfg["paths"]["answers_dir"] = str(dr / "answers")
    return cfg


def setup_logging(cfg):
    logs_dir = Path(cfg["paths"]["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("arith_analysis")
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(ch)

    fh = RotatingFileHandler(
        logs_dir / "analysis.log", maxBytes=10 * 1024 * 1024, backupCount=3
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    logger.addHandler(fh)
    return logger


# ====================================================================
# 2. DATA LOADING
# ====================================================================

def load_answers(cfg, logger):
    ans_dir = Path(cfg["paths"]["answers_dir"])
    answers = {}
    for lvl in range(1, 6):
        path = ans_dir / f"level_{lvl}.json"
        with open(path) as f:
            answers[lvl] = json.load(f)
        n = len(answers[lvl]["results"])
        nc = answers[lvl]["n_correct"]
        acc = answers[lvl]["accuracy"]
        logger.info(f"Level {lvl}: {n} results, accuracy {acc:.1%} ({nc}/{n})")
    return answers


def load_labels(cfg, logger):
    lab_dir = Path(cfg["paths"]["labels_dir"])
    labels = {}
    for lvl in range(1, 6):
        path = lab_dir / f"level_{lvl}.json"
        with open(path) as f:
            data = json.load(f)
        labels[lvl] = [p["labels"] for p in data["problems"]]
        logger.info(f"Level {lvl}: loaded {len(labels[lvl])} label sets")
    return labels


def merge_data(answers, labels, logger):
    merged = {}
    for lvl in range(1, 6):
        results = answers[lvl]["results"]
        labs = labels[lvl]
        assert len(results) == len(labs), f"Level {lvl}: count mismatch"
        level_data = []
        for r, lab in zip(results, labs):
            assert r["a"] == lab["a"] and r["b"] == lab["b"]
            entry = {**r, **lab}
            level_data.append(entry)
        merged[lvl] = level_data
        logger.info(f"Level {lvl}: merged {len(level_data)} entries")
    return merged


# ====================================================================
# 3. ERROR CLASSIFICATION
# ====================================================================

def classify_errors(merged, logger):
    for lvl in sorted(merged):
        counts = Counter()
        for p in merged[lvl]:
            if p["correct"]:
                p["error_category"] = None
                continue

            pred = p["predicted"]
            gt = p["ground_truth"]

            if pred is None:
                p["error_category"] = "garbage"
            elif len(str(pred)) != len(str(gt)):
                p["error_category"] = "magnitude_error"
            elif abs(pred - gt) / gt < 0.05:
                p["error_category"] = "close_arithmetic"
            else:
                p["error_category"] = "large_arithmetic"

            counts[p["error_category"]] += 1

        n_wrong = sum(counts.values())
        parts = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
        logger.info(f"Level {lvl}: {n_wrong} wrong — {parts}")


# ====================================================================
# 4. PER-DIGIT ACCURACY
# ====================================================================

def compute_per_digit_accuracy(merged, logger):
    result = {}
    for lvl in sorted(merged):
        max_gt_digits = max(len(str(p["ground_truth"])) for p in merged[lvl])
        correct_by_pos = [0] * max_gt_digits
        total_by_pos = [0] * max_gt_digits

        n_same_mag = 0
        for p in merged[lvl]:
            pred = p["predicted"]
            gt = p["ground_truth"]
            if pred is None:
                continue
            gt_str = str(gt)
            pred_str = str(pred)
            if len(pred_str) != len(gt_str):
                continue

            n_same_mag += 1
            for i in range(len(gt_str)):
                total_by_pos[i] += 1
                if gt_str[i] == pred_str[i]:
                    correct_by_pos[i] += 1

        acc_by_pos = [
            correct_by_pos[i] / total_by_pos[i] if total_by_pos[i] > 0 else float("nan")
            for i in range(max_gt_digits)
        ]

        result[lvl] = {
            "n_same_magnitude": n_same_mag,
            "n_digits": max_gt_digits,
            "per_position_accuracy": acc_by_pos,
            "per_position_correct": correct_by_pos,
            "per_position_total": total_by_pos,
        }
        formatted = [f"{a:.1%}" for a in acc_by_pos if not np.isnan(a)]
        logger.info(
            f"Level {lvl}: per-digit accuracy (MSF) = [{', '.join(formatted)}] "
            f"(n={n_same_mag} same-magnitude)"
        )

    return result


# ====================================================================
# 5. CARRY CORRELATION
# ====================================================================

def compute_carry_correlation(merged, logger):
    result = {}
    for lvl in sorted(merged):
        by_num_carries = defaultdict(lambda: {"n": 0, "n_correct": 0})
        by_max_carry = defaultdict(lambda: {"n": 0, "n_correct": 0})
        carry_sum_bins = [0, 1, 4, 7, 11, 16, 21, 31, 999]
        by_carry_sum = defaultdict(lambda: {"n": 0, "n_correct": 0})

        for p in merged[lvl]:
            carries = p["carries"]
            num_nz = sum(1 for c in carries if c > 0)
            max_c = max(carries) if carries else 0
            total_c = sum(carries)
            correct = p["correct"]

            by_num_carries[num_nz]["n"] += 1
            by_num_carries[num_nz]["n_correct"] += correct

            by_max_carry[max_c]["n"] += 1
            by_max_carry[max_c]["n_correct"] += correct

            for i in range(len(carry_sum_bins) - 1):
                if carry_sum_bins[i] <= total_c < carry_sum_bins[i + 1]:
                    label = f"{carry_sum_bins[i]}-{carry_sum_bins[i+1]-1}"
                    by_carry_sum[label]["n"] += 1
                    by_carry_sum[label]["n_correct"] += correct
                    break

        for d in [by_num_carries, by_max_carry, by_carry_sum]:
            for k in d:
                d[k]["accuracy"] = (
                    d[k]["n_correct"] / d[k]["n"] if d[k]["n"] > 0 else 0.0
                )

        result[lvl] = {
            "by_num_carries": dict(by_num_carries),
            "by_max_carry": dict(by_max_carry),
            "by_carry_sum": dict(by_carry_sum),
        }

        parts = []
        for k in sorted(by_num_carries):
            d = by_num_carries[k]
            parts.append(f"{k} carries: {d['accuracy']:.1%} (n={d['n']})")
        logger.info(f"Level {lvl}: accuracy by carry count — {', '.join(parts)}")

    return result


# ====================================================================
# 6. ERROR STRUCTURE
# ====================================================================

def compute_error_structure(merged, logger):
    result = {}
    for lvl in sorted(merged):
        wrong = [
            p for p in merged[lvl]
            if not p["correct"] and p["predicted"] is not None
        ]
        if not wrong:
            result[lvl] = None
            continue

        errors = np.array([p["predicted"] - p["ground_truth"] for p in wrong])
        abs_errors = np.abs(errors)
        gt = np.array([p["ground_truth"] for p in wrong], dtype=float)
        rel_errors = abs_errors / gt

        n_total = len(errors)
        n_even = int(np.sum(errors % 2 == 0))
        n_div10 = int(np.sum(errors % 10 == 0))
        n_div20 = int(np.sum(errors % 20 == 0))
        n_div100 = int(np.sum(errors % 100 == 0))

        units_mismatch = 0
        units_complement = 0
        for p in wrong:
            gt_u = p["ground_truth"] % 10
            pr_u = p["predicted"] % 10
            if gt_u != pr_u:
                units_mismatch += 1
                if (gt_u + pr_u) % 10 == 0:
                    units_complement += 1

        n_negative = int(np.sum(errors < 0))
        n_positive = int(np.sum(errors > 0))

        stats = {
            "n_wrong": n_total,
            "mean_error": float(np.mean(errors)),
            "median_error": float(np.median(errors)),
            "std_error": float(np.std(errors)),
            "mean_abs_error": float(np.mean(abs_errors)),
            "median_abs_error": float(np.median(abs_errors)),
            "mean_rel_error": float(np.mean(rel_errors)),
            "median_rel_error": float(np.median(rel_errors)),
            "p95_rel_error": float(np.percentile(rel_errors, 95)),
            "even_frac": n_even / n_total,
            "div10_frac": n_div10 / n_total,
            "div20_frac": n_div20 / n_total,
            "div100_frac": n_div100 / n_total,
            "units_mismatch": units_mismatch,
            "units_complement": units_complement,
            "units_complement_frac": (
                units_complement / units_mismatch if units_mismatch > 0 else 0.0
            ),
            "n_negative": n_negative,
            "n_positive": n_positive,
            "underestimate_frac": n_negative / n_total,
            "errors": errors.tolist(),
            "rel_errors": rel_errors.tolist(),
        }
        result[lvl] = stats

        logger.info(
            f"Level {lvl}: {n_total} arithmetic errors | "
            f"median rel err={stats['median_rel_error']:.3%} | "
            f"even={stats['even_frac']:.1%} | "
            f"10s complement={stats['units_complement_frac']:.1%} | "
            f"underestimate={stats['underestimate_frac']:.1%}"
        )

    return result


# ====================================================================
# 7. INPUT DIFFICULTY
# ====================================================================

def compute_input_difficulty(merged, logger):
    result = {}
    for lvl in sorted(merged):
        by_a_lead = defaultdict(lambda: {"n": 0, "n_correct": 0})
        by_b_lead = defaultdict(lambda: {"n": 0, "n_correct": 0})
        by_ndigits = defaultdict(lambda: {"n": 0, "n_correct": 0})
        by_magnitude = defaultdict(lambda: {"n": 0, "n_correct": 0})

        for p in merged[lvl]:
            c = p["correct"]
            a_lead = int(str(p["a"])[0])
            b_lead = int(str(p["b"])[0])
            gt = p["ground_truth"]
            nd = len(str(gt))

            by_a_lead[a_lead]["n"] += 1
            by_a_lead[a_lead]["n_correct"] += c
            by_b_lead[b_lead]["n"] += 1
            by_b_lead[b_lead]["n_correct"] += c
            by_ndigits[nd]["n"] += 1
            by_ndigits[nd]["n_correct"] += c

            mag = 10 ** (nd - 1)
            bin_idx = (gt // mag) * mag
            label = f"{bin_idx}-{bin_idx + mag - 1}"
            by_magnitude[label]["n"] += 1
            by_magnitude[label]["n_correct"] += c

        for d in [by_a_lead, by_b_lead, by_ndigits, by_magnitude]:
            for k in d:
                d[k]["accuracy"] = (
                    d[k]["n_correct"] / d[k]["n"] if d[k]["n"] > 0 else 0.0
                )

        result[lvl] = {
            "by_a_leading": dict(by_a_lead),
            "by_b_leading": dict(by_b_lead),
            "by_product_ndigits": dict(by_ndigits),
            "by_product_magnitude": dict(by_magnitude),
        }

        parts = []
        for nd in sorted(by_ndigits):
            d = by_ndigits[nd]
            parts.append(f"{nd}-digit: {d['accuracy']:.1%} (n={d['n']})")
        logger.info(f"Level {lvl}: accuracy by product digits — {', '.join(parts)}")

    return result


# ====================================================================
# 8. PLOTTING
# ====================================================================

COLORS = {
    "correct": "#4CAF50",
    "close_arithmetic": "#FFC107",
    "magnitude_error": "#FF9800",
    "large_arithmetic": "#F44336",
    "garbage": "#212121",
}


def plot_per_digit_heatmap(digit_accuracy, cfg, logger):
    plots_dir = Path(cfg["paths"]["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)
    levels = sorted(digit_accuracy)
    max_digits = max(digit_accuracy[l]["n_digits"] for l in levels)

    matrix = np.full((len(levels), max_digits), np.nan)
    for i, lvl in enumerate(levels):
        accs = digit_accuracy[lvl]["per_position_accuracy"]
        for j, a in enumerate(accs):
            matrix[i, j] = a

    fig, ax = plt.subplots(figsize=(10, 4))
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#F5F5F5")
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    for i in range(len(levels)):
        for j in range(max_digits):
            v = matrix[i, j]
            if not np.isnan(v):
                color = "white" if v < 0.4 else "black"
                ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

    ax.set_yticks(range(len(levels)))
    ax.set_yticklabels([f"Level {l}" for l in levels])
    ax.set_xticks(range(max_digits))
    ax.set_xticklabels([f"Pos {j}\n(MSF)" for j in range(max_digits)])
    ax.set_title("Per-Digit Accuracy by Position (MSF, same-magnitude only)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Accuracy")
    fig.tight_layout()
    fig.savefig(plots_dir / "per_digit_accuracy_heatmap.png", dpi=150)
    plt.close(fig)
    logger.info("Saved per_digit_accuracy_heatmap.png")


def plot_error_distributions(error_structure, cfg, logger):
    plots_dir = Path(cfg["paths"]["plots_dir"])
    plot_levels = [l for l in [3, 4, 5] if error_structure.get(l)]

    fig, axes = plt.subplots(1, len(plot_levels), figsize=(5 * len(plot_levels), 4))
    if len(plot_levels) == 1:
        axes = [axes]

    for ax, lvl in zip(axes, plot_levels):
        errors = np.array(error_structure[lvl]["errors"])
        lo, hi = np.percentile(errors, [1, 99])
        clipped = errors[(errors >= lo) & (errors <= hi)]

        ax.hist(clipped, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
        ax.axvline(0, color="red", linewidth=1, linestyle="--")
        ax.set_title(f"Level {lvl} (n={len(errors)})")
        ax.set_xlabel("Error (predicted - ground truth)")
        ax.set_ylabel("Count")
        med = np.median(errors)
        ax.axvline(med, color="orange", linewidth=1, linestyle=":")
        ax.text(0.95, 0.95, f"median={med:.0f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8)

    fig.suptitle("Error Distributions (arithmetic errors only)", fontsize=12)
    fig.tight_layout()
    fig.savefig(plots_dir / "error_distributions.png", dpi=150)
    plt.close(fig)
    logger.info("Saved error_distributions.png")


def plot_carry_accuracy(carry_correlation, cfg, logger):
    plots_dir = Path(cfg["paths"]["plots_dir"])
    fig, ax = plt.subplots(figsize=(8, 5))

    for lvl in sorted(carry_correlation):
        data = carry_correlation[lvl]["by_num_carries"]
        if not data:
            continue
        xs = sorted(data.keys())
        ys = [data[x]["accuracy"] for x in xs]
        ns = [data[x]["n"] for x in xs]
        ax.plot(xs, ys, marker="o", label=f"Level {lvl}")
        for x, y, n in zip(xs, ys, ns):
            if n < 50:
                ax.annotate(f"n={n}", (x, y), fontsize=6, alpha=0.6)

    ax.set_xlabel("Number of Non-Zero Carries")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Carry Count")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "accuracy_vs_carries.png", dpi=150)
    plt.close(fig)
    logger.info("Saved accuracy_vs_carries.png")


def plot_product_magnitude(input_difficulty, cfg, logger):
    plots_dir = Path(cfg["paths"]["plots_dir"])
    fig, ax = plt.subplots(figsize=(10, 5))

    for lvl in [3, 4, 5]:
        data = input_difficulty[lvl]["by_product_ndigits"]
        if not data:
            continue
        xs = sorted(data.keys())
        ys = [data[x]["accuracy"] for x in xs]
        ns = [data[x]["n"] for x in xs]
        ax.bar(
            [f"{x}d\nL{lvl}" for x in xs], ys,
            color=f"C{lvl - 1}", alpha=0.75,
        )
        for i, (x, y, n) in enumerate(zip(xs, ys, ns)):
            ax.text(i + (lvl - 3) * len(xs), y + 0.02, f"{y:.0%}\nn={n}",
                    ha="center", fontsize=7)

    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Product Digit Count")
    ax.set_ylim(0, 1.15)
    fig.tight_layout()
    fig.savefig(plots_dir / "accuracy_vs_magnitude.png", dpi=150)
    plt.close(fig)
    logger.info("Saved accuracy_vs_magnitude.png")


def plot_error_categories(merged, cfg, logger):
    plots_dir = Path(cfg["paths"]["plots_dir"])
    levels = [1, 2, 3, 4, 5]
    categories = ["correct", "close_arithmetic",
                  "magnitude_error", "large_arithmetic", "garbage"]

    counts = {cat: [] for cat in categories}
    for lvl in levels:
        level_counts = Counter()
        for p in merged[lvl]:
            if p["correct"]:
                level_counts["correct"] += 1
            else:
                cat = p.get("error_category", "garbage")
                level_counts[cat] += 1
        for cat in categories:
            counts[cat].append(level_counts.get(cat, 0))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(levels))
    bottom = np.zeros(len(levels))

    for cat in categories:
        vals = np.array(counts[cat], dtype=float)
        ax.bar(x, vals, bottom=bottom, label=cat, color=COLORS.get(cat, "#999"),
               width=0.6)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in levels])
    ax.set_ylabel("Count")
    ax.set_title("Error Category Breakdown by Level")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "error_categories.png", dpi=150)
    plt.close(fig)
    logger.info("Saved error_categories.png")


def plot_digit_accuracy_by_carry(merged, cfg, logger):
    plots_dir = Path(cfg["paths"]["plots_dir"])
    plot_levels = [3, 4, 5]

    fig, axes = plt.subplots(1, len(plot_levels), figsize=(5 * len(plot_levels), 4))
    if len(plot_levels) == 1:
        axes = [axes]

    for ax, lvl in zip(axes, plot_levels):
        max_digits = max(len(str(p["ground_truth"])) for p in merged[lvl])
        carry0_correct = [0] * max_digits
        carry0_total = [0] * max_digits
        carry_pos_correct = [0] * max_digits
        carry_pos_total = [0] * max_digits

        for p in merged[lvl]:
            pred = p["predicted"]
            gt = p["ground_truth"]
            if pred is None or len(str(pred)) != len(str(gt)):
                continue

            gt_str = str(gt)
            pred_str = str(pred)
            carries = p["carries"]
            n_d = len(gt_str)

            for pos_msf in range(n_d):
                pos_lsf = n_d - 1 - pos_msf
                incoming_carry = carries[pos_lsf - 1] if pos_lsf > 0 else 0
                digit_correct = (gt_str[pos_msf] == pred_str[pos_msf])

                if incoming_carry == 0:
                    carry0_total[pos_msf] += 1
                    carry0_correct[pos_msf] += digit_correct
                else:
                    carry_pos_total[pos_msf] += 1
                    carry_pos_correct[pos_msf] += digit_correct

        positions = list(range(max_digits))
        acc_no_carry = [
            carry0_correct[i] / carry0_total[i] if carry0_total[i] > 0 else float("nan")
            for i in positions
        ]
        acc_with_carry = [
            carry_pos_correct[i] / carry_pos_total[i] if carry_pos_total[i] > 0 else float("nan")
            for i in positions
        ]

        ax.plot(positions, acc_no_carry, "o-", label="carry_in = 0", color="#4CAF50")
        ax.plot(positions, acc_with_carry, "s-", label="carry_in > 0", color="#F44336")
        ax.set_xticks(positions)
        ax.set_xticklabels([f"Pos {i}" for i in positions], fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Level {lvl}")
        ax.set_ylabel("Digit Accuracy")
        ax.set_xlabel("MSF Position")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Per-Digit Accuracy Split by Incoming Carry", fontsize=12)
    fig.tight_layout()
    fig.savefig(plots_dir / "digit_accuracy_by_carry.png", dpi=150)
    plt.close(fig)
    logger.info("Saved digit_accuracy_by_carry.png")


# ====================================================================
# 9. SUMMARY OUTPUT
# ====================================================================

def build_summary(answers, merged, digit_accuracy, carry_correlation,
                  error_structure, input_difficulty):
    summary = {"levels": {}}
    for lvl in sorted(merged):
        n = len(merged[lvl])
        n_correct = sum(1 for p in merged[lvl] if p["correct"])

        cat_counts = Counter(
            p.get("error_category") for p in merged[lvl]
            if p.get("error_category") is not None
        )

        lvl_summary = {
            "n_problems": n,
            "accuracy": n_correct / n,
            "n_correct": n_correct,
            "error_categories": dict(cat_counts),
            "per_digit_accuracy": digit_accuracy.get(lvl, {}),
        }

        if error_structure.get(lvl):
            es = {k: v for k, v in error_structure[lvl].items()
                  if k not in ("errors", "rel_errors")}
            lvl_summary["error_structure"] = es

        summary["levels"][lvl] = lvl_summary

    return summary


def save_summary(summary, cfg, logger):
    ans_dir = Path(cfg["paths"]["answers_dir"])
    path = ans_dir / "analysis_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Saved analysis summary to {path}")


def print_report(summary, logger):
    logger.info("=" * 60)
    logger.info("ANALYSIS REPORT: Llama 3.1 8B Multiplication")
    logger.info("=" * 60)

    logger.info("")
    logger.info("ACCURACY:")
    for lvl in sorted(summary["levels"]):
        s = summary["levels"][lvl]
        acc = s["accuracy"]
        n = s["n_problems"]
        nc = s["n_correct"]
        logger.info(f"  Level {lvl}: {acc:.1%} ({nc}/{n})")

    logger.info("")
    logger.info("ERROR CATEGORIES:")
    for lvl in sorted(summary["levels"]):
        s = summary["levels"][lvl]
        cats = s["error_categories"]
        if not cats:
            logger.info(f"  Level {lvl}: no errors")
            continue
        total = sum(cats.values())
        parts = [f"{k}: {v} ({v/total:.0%})" for k, v in sorted(cats.items())]
        logger.info(f"  Level {lvl}: {total} wrong — {', '.join(parts)}")

    logger.info("")
    logger.info("PER-DIGIT ACCURACY (MSF, same-magnitude):")
    for lvl in sorted(summary["levels"]):
        da = summary["levels"][lvl].get("per_digit_accuracy", {})
        accs = da.get("per_position_accuracy", [])
        if accs:
            formatted = [f"Pos{i}={a:.0%}" for i, a in enumerate(accs)]
            logger.info(f"  Level {lvl}: {', '.join(formatted)}")

    logger.info("")
    logger.info("KEY FINDINGS:")
    for lvl in [3, 4, 5]:
        es = summary["levels"][lvl].get("error_structure", {})
        if es:
            logger.info(
                f"  Level {lvl}: "
                f"even={es.get('even_frac', 0):.0%}, "
                f"10s_comp={es.get('units_complement_frac', 0):.0%}, "
                f"underest={es.get('underestimate_frac', 0):.0%}, "
                f"median_rel_err={es.get('median_rel_error', 0):.3%}"
            )

    logger.info("=" * 60)


# ====================================================================
# 10. MAIN ORCHESTRATOR
# ====================================================================

def main(config_path=None):
    t_start = time.time()

    cfg = load_config(config_path)
    logger = setup_logging(cfg)
    logger.info("=" * 60)
    logger.info("Arithmetic Geometry Error Analysis")
    logger.info("=" * 60)

    logger.info("--- Loading data ---")
    answers = load_answers(cfg, logger)
    labels = load_labels(cfg, logger)
    merged = merge_data(answers, labels, logger)

    logger.info("--- Classifying errors ---")
    classify_errors(merged, logger)

    logger.info("--- Per-digit accuracy ---")
    digit_accuracy = compute_per_digit_accuracy(merged, logger)

    logger.info("--- Carry correlation ---")
    carry_correlation = compute_carry_correlation(merged, logger)

    logger.info("--- Error structure ---")
    error_structure = compute_error_structure(merged, logger)

    logger.info("--- Input difficulty ---")
    input_difficulty = compute_input_difficulty(merged, logger)

    logger.info("--- Generating plots ---")
    plot_per_digit_heatmap(digit_accuracy, cfg, logger)
    plot_error_distributions(error_structure, cfg, logger)
    plot_carry_accuracy(carry_correlation, cfg, logger)
    plot_product_magnitude(input_difficulty, cfg, logger)
    plot_error_categories(merged, cfg, logger)
    plot_digit_accuracy_by_carry(merged, cfg, logger)

    logger.info("--- Summary ---")
    summary = build_summary(answers, merged, digit_accuracy,
                            carry_correlation, error_structure, input_difficulty)
    save_summary(summary, cfg, logger)
    print_report(summary, logger)

    logger.info(f"Analysis complete in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arithmetic Geometry Error Analysis")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)
