#!/usr/bin/env python3
"""
Arithmetic Geometry Dataset Generator

Generates multi-level multiplication problems, extracts Llama 3.1 8B
residual stream activations, computes algorithm-agnostic mathematical
labels, and evaluates model accuracy via greedy decoding.

Usage:
    python pipeline.py                    # uses config.yaml in same directory
    python pipeline.py --config path.yaml # custom config path
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from logging.handlers import RotatingFileHandler
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Torch / HF imports deferred to where they're needed so that
# dataset generation + label verification can run without GPU.


# ====================================================================
# 1. CONFIG
# ====================================================================

def load_config(config_path=None):
    """Load config.yaml, validate, and derive paths."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Validate layers against known model depth
    num_layers = cfg["model"]["num_layers"]
    for layer in cfg["model"]["layers"]:
        assert 0 <= layer < num_layers, (
            f"Layer {layer} out of range [0, {num_layers})"
        )

    # Derive concrete paths
    ws = Path(cfg["paths"]["workspace"])
    dr = Path(cfg["paths"]["data_root"])
    cfg["paths"]["labels_dir"] = str(ws / "data")
    cfg["paths"]["logs_dir"] = str(ws / "logs")
    cfg["paths"]["activations_dir"] = str(dr / "activations")
    cfg["paths"]["answers_dir"] = str(dr / "answers")

    # Ensure all directories exist
    for key in ("labels_dir", "logs_dir", "activations_dir", "answers_dir"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)

    return cfg


# ====================================================================
# 2. LOGGING
# ====================================================================

def setup_logging(cfg):
    """Console + rotating file logger."""
    logs_dir = Path(cfg["paths"]["logs_dir"])
    logger = logging.getLogger("arith_geom")
    logger.setLevel(logging.DEBUG)

    fmt_console = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )
    fmt_file = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s"
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt_console)
    logger.addHandler(ch)

    fh = RotatingFileHandler(
        logs_dir / "pipeline.log", maxBytes=10 * 1024 * 1024, backupCount=3
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_file)
    logger.addHandler(fh)

    return logger


# ====================================================================
# 3. PROBLEM GENERATION
# ====================================================================

def generate_problems(cfg, logger):
    """Generate (a, b) pairs for every level. Returns dict[level] -> info."""
    rng = np.random.RandomState(cfg["dataset"]["seed"])
    n_default = cfg["dataset"]["problems_per_level"]
    levels_cfg = cfg["dataset"]["levels"]
    all_problems = {}

    for lvl in sorted(levels_cfg):
        lc = levels_cfg[lvl]
        a_lo, a_hi = lc["a_range"]
        b_lo, b_hi = lc["b_range"]

        if lc.get("unique_only", False):
            pairs = [
                (a, b)
                for a in range(a_lo, a_hi + 1)
                for b in range(b_lo, b_hi + 1)
            ]
            a_vals = [p[0] for p in pairs]
            b_vals = [p[1] for p in pairs]
        else:
            a_vals = rng.randint(a_lo, a_hi + 1, size=n_default).tolist()
            b_vals = rng.randint(b_lo, b_hi + 1, size=n_default).tolist()

        unique_count = len(set(zip(a_vals, b_vals)))

        # Digit-coverage check for levels with random sampling
        if not lc.get("unique_only", False):
            a_leading = Counter(int(str(a)[0]) for a in a_vals)
            b_leading = Counter(int(str(b)[0]) for b in b_vals)
            logger.debug(
                f"Level {lvl} digit coverage: a_leading={dict(a_leading)}, "
                f"b_leading={dict(b_leading)}"
            )

        n = len(a_vals)
        all_problems[lvl] = {
            "a": a_vals,
            "b": b_vals,
            "n_problems": n,
            "unique_problems": unique_count,
        }
        logger.info(
            f"Level {lvl}: {n} problems ({unique_count} unique), "
            f"a in [{a_lo},{a_hi}], b in [{b_lo},{b_hi}]"
        )

    total = sum(v["n_problems"] for v in all_problems.values())
    logger.info(f"Total problems across all levels: {total}")
    return all_problems


# ====================================================================
# 4. LABEL COMPUTATION
# ====================================================================

PLACE_NAMES = [
    "units", "tens", "hundreds", "thousands",
    "ten_thousands", "hundred_thousands",
]


def decompose_digits(n):
    """Decompose integer into {place_name: digit_value, num_digits: int}."""
    s = str(n)
    d = {}
    for i, ch in enumerate(reversed(s)):
        d[PLACE_NAMES[i]] = int(ch)
    d["num_digits"] = len(s)
    return d


def compute_labels(a, b):
    """Compute all algorithm-agnostic labels for a * b."""
    product = a * b

    # Digit decomposition (LSF: index 0 = units)
    a_digits = [int(d) for d in reversed(str(a))]
    b_digits = [int(d) for d in reversed(str(b))]
    n_a, n_b = len(a_digits), len(b_digits)
    n_cols = n_a + n_b - 1  # columns that receive partial products

    # Pairwise partial products
    partial_products = {}
    for i in range(n_a):
        for j in range(n_b):
            partial_products[f"a{i}_x_b{j}"] = a_digits[i] * b_digits[j]

    # Column sums and which products map to each column
    column_sums = [0] * n_cols
    column_products = {c: [] for c in range(n_cols)}
    for i in range(n_a):
        for j in range(n_b):
            col = i + j
            column_sums[col] += a_digits[i] * b_digits[j]
            column_products[col].append(f"a{i}_x_b{j}")

    # Carry propagation (LSF order)
    carries = []
    running_sums = []
    answer_digits_lsf = []
    carry = 0

    for col in range(n_cols):
        rs = column_sums[col] + carry
        running_sums.append(rs)
        answer_digits_lsf.append(rs % 10)
        carry = rs // 10
        carries.append(carry)

    # Remaining carry beyond partial-product columns
    while carry > 0:
        answer_digits_lsf.append(carry % 10)
        carry //= 10

    # Strip leading zeros (shouldn't happen for nonzero products)
    while len(answer_digits_lsf) > 1 and answer_digits_lsf[-1] == 0:
        answer_digits_lsf.pop()

    answer_digits_msf = list(reversed(answer_digits_lsf))

    # Sanity: reconstruct
    reconstructed = sum(d * 10**i for i, d in enumerate(answer_digits_lsf))
    assert reconstructed == product, (
        f"Reconstruction failed: {reconstructed} != {product} for {a}*{b}"
    )

    # Per-digit difficulty (indexed by LSF position)
    digit_difficulty = []
    for pos in range(len(answer_digits_lsf)):
        prods = column_products.get(pos, [])
        n_pp = len(prods)
        digit_difficulty.append({
            "position_lsf": pos,
            "num_partial_products": n_pp,
            "max_column_sum": n_pp * 81,  # each partial product <= 9*9
            "carry_chain_length": pos,
        })

    return {
        "a": a,
        "b": b,
        "product": product,
        "a_digits_lsf": a_digits,
        "b_digits_lsf": b_digits,
        "a_decomposition": decompose_digits(a),
        "b_decomposition": decompose_digits(b),
        "partial_products": partial_products,
        "column_sums": column_sums,
        "column_products": {str(k): v for k, v in column_products.items()},
        "carries": carries,
        "running_sums": running_sums,
        "answer_digits_lsf": answer_digits_lsf,
        "answer_digits_msf": answer_digits_msf,
        "digit_difficulty": digit_difficulty,
    }


def compute_all_labels(all_problems, logger):
    """Compute labels for every problem at every level."""
    all_labels = {}
    for lvl in sorted(all_problems):
        probs = all_problems[lvl]
        labels = [
            compute_labels(a, b)
            for a, b in zip(probs["a"], probs["b"])
        ]
        all_labels[lvl] = labels
        logger.info(f"Level {lvl}: computed {len(labels)} label sets")
    return all_labels


# ====================================================================
# 5. LABEL VERIFICATION
# ====================================================================

def compute_carry_bounds(a_range, b_range):
    """Compute tight per-column max carry from digit ranges.

    Uses the fact that max carry at every column is achieved when all
    digits are 9 (proven: maximising carry_in and column_sum never
    conflict because both are maximised by all-9 inputs).
    """
    n_a = len(str(a_range[1]))  # digits in max a
    n_b = len(str(b_range[1]))  # digits in max b
    n_cols = n_a + n_b - 1

    # Max column sums (all digit positions maxed at 9)
    max_col_sums = [0] * n_cols
    for i in range(n_a):
        for j in range(n_b):
            max_col_sums[i + j] += 81  # 9 * 9

    # Propagate to get max carries
    max_carries = []
    carry = 0
    for col in range(n_cols):
        rs = max_col_sums[col] + carry
        carry = rs // 10
        max_carries.append(carry)

    return max_carries


def verify_labels(all_problems, all_labels, cfg, logger):
    """Assert every label is mathematically consistent and within bounds."""
    levels_cfg = cfg["dataset"]["levels"]

    for lvl in sorted(all_labels):
        lc = levels_cfg[lvl]
        max_carries = compute_carry_bounds(lc["a_range"], lc["b_range"])
        logger.info(f"Level {lvl}: max carry bounds = {max_carries}")

        for idx, lab in enumerate(all_labels[lvl]):
            # Product matches
            assert lab["product"] == lab["a"] * lab["b"], (
                f"L{lvl}[{idx}]: product mismatch"
            )

            # Carries within bounds
            for col, cv in enumerate(lab["carries"]):
                assert cv <= max_carries[col], (
                    f"L{lvl}[{idx}] col {col}: carry {cv} > bound {max_carries[col]}"
                )

            # Running-sum consistency
            carry_in = 0
            for col in range(len(lab["column_sums"])):
                expected = lab["column_sums"][col] + carry_in
                assert lab["running_sums"][col] == expected, (
                    f"L{lvl}[{idx}] col {col}: running_sum "
                    f"{lab['running_sums'][col]} != {expected}"
                )
                carry_in = expected // 10

            # Digit reconstruction
            recon = sum(
                d * 10**i for i, d in enumerate(lab["answer_digits_lsf"])
            )
            assert recon == lab["product"], (
                f"L{lvl}[{idx}]: recon {recon} != product {lab['product']}"
            )

        logger.info(f"Level {lvl}: all {len(all_labels[lvl])} labels verified")


# ====================================================================
# 6. PROMPT FORMATTING
# ====================================================================

def format_prompts(all_problems, cfg):
    """Build prompt strings per level."""
    tmpl = cfg["dataset"]["prompt_template"]
    return {
        lvl: [tmpl.format(a=a, b=b)
              for a, b in zip(p["a"], p["b"])]
        for lvl, p in all_problems.items()
    }


# ====================================================================
# 7. SAVE DATASETS
# ====================================================================

def save_datasets(all_problems, all_labels, all_prompts, cfg, logger):
    """Write per-level JSON to workspace/data/."""
    out_dir = Path(cfg["paths"]["labels_dir"])

    for lvl in sorted(all_problems):
        probs = all_problems[lvl]
        dataset = {
            "level": lvl,
            "n_problems": probs["n_problems"],
            "unique_problems": probs["unique_problems"],
            "level_config": cfg["dataset"]["levels"][lvl],
            "problems": [
                {"index": i, "prompt": all_prompts[lvl][i],
                 "labels": all_labels[lvl][i]}
                for i in range(probs["n_problems"])
            ],
        }
        path = out_dir / f"level_{lvl}.json"
        with open(path, "w") as f:
            json.dump(dataset, f)
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"Level {lvl}: saved {path.name} ({size_mb:.1f} MB)")


# ====================================================================
# 8. TOKENIZER
# ====================================================================

def load_tokenizer(cfg, logger):
    """Load tokenizer (CPU, fast), set pad token."""
    from transformers import AutoTokenizer

    name = cfg["model"]["name"]
    logger.info(f"Loading tokenizer: {name}")
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    logger.info(
        f"Tokenizer ready (vocab={tokenizer.vocab_size}, "
        f"pad=eos={tokenizer.eos_token_id})"
    )
    return tokenizer


def verify_tokenization(tokenizer, all_prompts, logger):
    """Check 20 samples per level for consistent tokenization."""
    n_samples = 20
    ok = True

    for lvl in sorted(all_prompts):
        prompts = all_prompts[lvl]
        samples = prompts[:min(n_samples, len(prompts))]

        lengths = set()
        last_toks = set()
        for p in samples:
            ids = tokenizer.encode(p)
            lengths.add(len(ids))
            last_toks.add(tokenizer.decode([ids[-1]]))

        logger.info(
            f"Level {lvl}: token lengths={lengths}, last tokens={last_toks}"
        )

        if len(lengths) > 1:
            logger.warning(f"Level {lvl}: inconsistent token lengths {lengths}")
            ok = False
        for lt in last_toks:
            if "=" not in lt:
                logger.warning(
                    f"Level {lvl}: last token '{lt}' missing '='"
                )
                ok = False

    if ok:
        logger.info("Tokenization verification: PASS")
    else:
        logger.warning("Tokenization verification: issues found (see above)")
    return ok


# ====================================================================
# 9. MODEL
# ====================================================================

def load_model(cfg, logger):
    """Load model once. Returns (model, device)."""
    import torch
    from transformers import AutoModelForCausalLM

    name = cfg["model"]["name"]
    dtype = getattr(torch, cfg["model"]["dtype"])

    logger.info(f"Loading model: {name} (dtype={cfg['model']['dtype']})")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    logger.info(f"Model loaded in {time.time()-t0:.1f}s on {device}")
    return model, device


# ====================================================================
# 10. ACTIVATION EXTRACTION
# ====================================================================

def make_hook(storage, layer_idx):
    """Factory function — avoids closure-over-loop-variable bug.

    Each returned hook captures its own `layer_idx` value.
    """
    def hook_fn(module, input, output):
        # output[0]: (batch, seq_len, hidden_dim)
        storage[layer_idx] = output[0][:, -1, :].detach().cpu()
    return hook_fn


def extract_activations(model, tokenizer, all_prompts, cfg, logger):
    """Extract residual-stream activations at the '=' token."""
    import torch

    layers = cfg["model"]["layers"]
    batch_size = cfg["generation"]["batch_size"]
    act_dir = Path(cfg["paths"]["activations_dir"])
    hidden_dim = cfg["model"]["hidden_dim"]
    device = next(model.parameters()).device

    for lvl in sorted(all_prompts):
        prompts = all_prompts[lvl]
        n = len(prompts)

        # --- Checkpoint: skip level if all layer files exist + correct shape
        all_exist = True
        for layer in layers:
            fpath = act_dir / f"activations_level{lvl}_layer{layer}.npy"
            if fpath.exists():
                try:
                    arr = np.load(fpath, mmap_mode="r")
                    if arr.shape != (n, hidden_dim):
                        all_exist = False
                        break
                except Exception:
                    all_exist = False
                    break
            else:
                all_exist = False
                break
        if all_exist:
            logger.info(f"Level {lvl}: all activation files present, skipping")
            continue

        # --- Register hooks
        captured = {}
        hooks = []
        for layer in layers:
            h = model.model.layers[layer].register_forward_hook(
                make_hook(captured, layer)
            )
            hooks.append(h)

        # --- Collect activations
        level_acts = {layer: [] for layer in layers}
        logger.info(
            f"Level {lvl}: extracting {n} prompts x {len(layers)} layers"
        )
        t0 = time.time()

        with torch.no_grad():
            for bs in range(0, n, batch_size):
                be = min(bs + batch_size, n)
                batch = prompts[bs:be]
                inputs = tokenizer(batch, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                model(**inputs)

                for layer in layers:
                    level_acts[layer].append(
                        captured[layer].numpy().astype(np.float32)
                    )
                captured.clear()

                done = be
                if (done % (batch_size * 10)) < batch_size or done == n:
                    logger.info(f"  Level {lvl}: {done}/{n}")

        # --- Remove hooks
        for h in hooks:
            h.remove()

        # --- Concatenate + save
        for layer in layers:
            arr = np.concatenate(level_acts[layer], axis=0)
            np.save(act_dir / f"activations_level{lvl}_layer{layer}.npy", arr)
        del level_acts

        elapsed = time.time() - t0
        logger.info(f"Level {lvl}: extraction done in {elapsed:.1f}s")

    logger.info("Activation extraction complete")


# ====================================================================
# 11. POST-EXTRACTION SANITY CHECKS
# ====================================================================

def post_extraction_checks(all_prompts, cfg, logger):
    """Validate saved activations: no NaN/Inf, distinct, norm range."""
    layers = cfg["model"]["layers"]
    act_dir = Path(cfg["paths"]["activations_dir"])
    hidden_dim = cfg["model"]["hidden_dim"]
    passed = True

    for lvl in sorted(all_prompts):
        n = len(all_prompts[lvl])
        for layer in layers:
            fpath = act_dir / f"activations_level{lvl}_layer{layer}.npy"
            arr = np.load(fpath)

            # Shape
            if arr.shape != (n, hidden_dim):
                logger.error(
                    f"L{lvl} layer {layer}: shape {arr.shape} != "
                    f"({n}, {hidden_dim})"
                )
                passed = False
                continue

            # NaN / Inf
            if np.any(np.isnan(arr)):
                logger.error(f"L{lvl} layer {layer}: NaN detected")
                passed = False
            if np.any(np.isinf(arr)):
                logger.error(f"L{lvl} layer {layer}: Inf detected")
                passed = False

            # First two *different* problems must have different activations
            if n >= 2 and all_prompts[lvl][0] != all_prompts[lvl][1]:
                if np.allclose(arr[0], arr[1]):
                    logger.error(
                        f"L{lvl} layer {layer}: first two different prompts "
                        "have identical activations (closure bug?)"
                    )
                    passed = False

            # Norm statistics
            norms = np.linalg.norm(arr, axis=1)
            logger.debug(
                f"L{lvl} layer {layer}: norms "
                f"min={norms.min():.1f} mean={norms.mean():.1f} "
                f"max={norms.max():.1f}"
            )

        logger.info(f"Level {lvl}: sanity checks done")

    if not passed:
        raise RuntimeError("Post-extraction sanity checks FAILED (see logs)")
    logger.info("All post-extraction sanity checks PASSED")


# ====================================================================
# 12. ANSWER GENERATION
# ====================================================================

def parse_number(text):
    """Extract first integer from generated text."""
    digits = []
    for ch in text.strip():
        if ch.isdigit():
            digits.append(ch)
        elif digits:
            break
    return int("".join(digits)) if digits else None


def generate_answers(model, tokenizer, all_prompts, cfg, logger):
    """Greedy-decode every prompt. Returns dict[level] -> list of results."""
    import torch

    batch_size = cfg["generation"]["batch_size"]
    max_new = cfg["generation"]["max_new_tokens"]
    device = next(model.parameters()).device
    all_answers = {}

    for lvl in sorted(all_prompts):
        prompts = all_prompts[lvl]
        n = len(prompts)
        results = []

        logger.info(f"Level {lvl}: generating answers for {n} problems")
        t0 = time.time()

        with torch.no_grad():
            for bs in range(0, n, batch_size):
                be = min(bs + batch_size, n)
                batch = prompts[bs:be]
                inputs = tokenizer(batch, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                input_len = inputs["input_ids"].shape[1]

                outputs = model.generate(
                    **inputs, max_new_tokens=max_new, do_sample=False,
                )

                for i in range(len(batch)):
                    gen_ids = outputs[i][input_len:]
                    raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
                    results.append({
                        "raw_text": raw,
                        "predicted": parse_number(raw),
                    })

                done = be
                if (done % (batch_size * 10)) < batch_size or done == n:
                    logger.info(f"  Level {lvl}: {done}/{n}")

        elapsed = time.time() - t0
        logger.info(f"Level {lvl}: generation done in {elapsed:.1f}s")
        all_answers[lvl] = results

    return all_answers


# ====================================================================
# 13. SAVE ANSWERS
# ====================================================================

def save_answers(all_answers, all_labels, cfg, logger):
    """Write per-level answer JSON with correctness flags."""
    ans_dir = Path(cfg["paths"]["answers_dir"])

    for lvl in sorted(all_answers):
        answers = all_answers[lvl]
        labels = all_labels[lvl]

        n_correct = 0
        rows = []
        for i, (ans, lab) in enumerate(zip(answers, labels)):
            correct = ans["predicted"] == lab["product"]
            n_correct += correct
            rows.append({
                "index": i,
                "a": lab["a"],
                "b": lab["b"],
                "ground_truth": lab["product"],
                "predicted": ans["predicted"],
                "correct": correct,
                "raw_text": ans["raw_text"],
            })

        n_total = len(rows)
        acc = n_correct / n_total if n_total else 0.0
        output = {
            "level": lvl,
            "n_problems": n_total,
            "n_correct": n_correct,
            "accuracy": acc,
            "results": rows,
        }

        path = ans_dir / f"answers_level_{lvl}.json"
        with open(path, "w") as f:
            json.dump(output, f)
        logger.info(
            f"Level {lvl}: accuracy {acc:.1%} ({n_correct}/{n_total})"
        )

    return {
        lvl: sum(1 for a, l in zip(all_answers[lvl], all_labels[lvl])
                 if a["predicted"] == l["product"]) / len(all_answers[lvl])
        for lvl in all_answers
    }


# ====================================================================
# 14. DIAGNOSTIC PLOTS
# ====================================================================

def generate_plots(all_prompts, all_problems, accuracies, cfg, logger):
    """Three validation PNG plots saved to logs/."""
    logs_dir = Path(cfg["paths"]["logs_dir"])
    layers = cfg["model"]["layers"]
    act_dir = Path(cfg["paths"]["activations_dir"])
    levels = sorted(all_prompts)

    # ---- 1. Accuracy bar chart ----
    fig, ax = plt.subplots(figsize=(8, 5))
    accs = [accuracies[lvl] for lvl in levels]
    bars = ax.bar([f"L{l}" for l in levels], accs, color="steelblue")
    for bar, a in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{a:.1%}", ha="center", fontsize=10,
        )
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Difficulty Level")
    fig.tight_layout()
    fig.savefig(logs_dir / "accuracy_by_level.png", dpi=150)
    plt.close(fig)
    logger.info("Saved accuracy_by_level.png")

    # ---- 2. Activation norm profile ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for lvl in levels:
        n = len(all_prompts[lvl])
        means = []
        for layer in layers:
            arr = np.load(
                act_dir / f"activations_level{lvl}_layer{layer}.npy",
                mmap_mode="r",
            )
            means.append(np.linalg.norm(arr, axis=1).mean())
        ax.plot(layers, means, marker="o", label=f"Level {lvl}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean L2 Norm")
    ax.set_title("Activation Norm Profile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(logs_dir / "activation_norm_profile.png", dpi=150)
    plt.close(fig)
    logger.info("Saved activation_norm_profile.png")

    # ---- 3. Digit coverage heatmaps ----
    fig, axes = plt.subplots(1, 5, figsize=(24, 4))
    levels_cfg = cfg["dataset"]["levels"]

    for idx, lvl in enumerate(levels):
        ax = axes[idx]
        probs = all_problems[lvl]
        lc = levels_cfg[lvl]

        if lc.get("unique_only", False):
            a_lo, a_hi = lc["a_range"]
            b_lo, b_hi = lc["b_range"]
            grid = np.zeros((a_hi - a_lo + 1, b_hi - b_lo + 1))
            for a, b in zip(probs["a"], probs["b"]):
                grid[a - a_lo, b - b_lo] += 1
            im = ax.imshow(grid, cmap="Blues", aspect="auto", origin="lower")
            ax.set_xlabel("b")
            ax.set_ylabel("a")
        else:
            a_first = [int(str(a)[0]) for a in probs["a"]]
            b_first = [int(str(b)[0]) for b in probs["b"]]
            grid = np.zeros((9, 9))
            for af, bf in zip(a_first, b_first):
                grid[af - 1, bf - 1] += 1
            im = ax.imshow(grid, cmap="Blues", aspect="auto", origin="lower")
            ax.set_xlabel("b leading digit")
            ax.set_ylabel("a leading digit")
        ax.set_title(f"Level {lvl}")

    fig.suptitle("Digit Coverage", fontsize=13)
    fig.tight_layout()
    fig.savefig(logs_dir / "digit_coverage.png", dpi=150)
    plt.close(fig)
    logger.info("Saved digit_coverage.png")


# ====================================================================
# 15. MAIN ORCHESTRATOR
# ====================================================================

def main(config_path=None):
    t_start = time.time()

    # (1) Config
    cfg = load_config(config_path)

    # (2) Logging
    logger = setup_logging(cfg)
    logger.info("=" * 60)
    logger.info("Arithmetic Geometry Pipeline")
    logger.info("=" * 60)

    # (3) Generate problems
    logger.info("--- Generating problems ---")
    all_problems = generate_problems(cfg, logger)

    # (4) Compute labels
    logger.info("--- Computing labels ---")
    all_labels = compute_all_labels(all_problems, logger)

    # (5) Verify labels
    logger.info("--- Verifying labels ---")
    verify_labels(all_problems, all_labels, cfg, logger)

    # (6) Format prompts
    all_prompts = format_prompts(all_problems, cfg)

    # (7) Save datasets
    logger.info("--- Saving datasets ---")
    save_datasets(all_problems, all_labels, all_prompts, cfg, logger)

    # (8) Tokenizer
    logger.info("--- Loading tokenizer ---")
    tokenizer = load_tokenizer(cfg, logger)
    verify_tokenization(tokenizer, all_prompts, logger)

    # (9) Model (loaded once, reused for extraction + generation)
    logger.info("--- Loading model ---")
    model, device = load_model(cfg, logger)

    # (10) Extract activations
    logger.info("--- Extracting activations ---")
    extract_activations(model, tokenizer, all_prompts, cfg, logger)

    # (11) Sanity checks
    logger.info("--- Post-extraction sanity checks ---")
    post_extraction_checks(all_prompts, cfg, logger)

    # (12) Generate answers
    logger.info("--- Generating answers ---")
    all_answers = generate_answers(model, tokenizer, all_prompts, cfg, logger)

    # (13) Save answers
    logger.info("--- Saving answers ---")
    accuracies = save_answers(all_answers, all_labels, cfg, logger)

    # (14) Diagnostic plots
    logger.info("--- Generating diagnostic plots ---")
    generate_plots(all_prompts, all_problems, accuracies, cfg, logger)

    # (15) Summary
    t_total = time.time() - t_start
    logger.info("=" * 60)
    logger.info(f"Pipeline complete in {t_total/60:.1f} min")

    act_dir = Path(cfg["paths"]["activations_dir"])
    act_bytes = sum(f.stat().st_size for f in act_dir.glob("*.npy"))
    logger.info(f"Activation storage: {act_bytes / 1024**3:.2f} GB")

    ans_dir = Path(cfg["paths"]["answers_dir"])
    ans_bytes = sum(f.stat().st_size for f in ans_dir.glob("*.json"))
    logger.info(f"Answer storage: {ans_bytes / 1024**2:.1f} MB")

    for lvl in sorted(accuracies):
        logger.info(f"  Level {lvl} accuracy: {accuracies[lvl]:.1%}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arithmetic Geometry Pipeline")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)
