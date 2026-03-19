#!/usr/bin/env python3
"""Two-phase L5 dataset generation: screen all 810,000 problems, select balanced subset.

Phase 1 — Enumerate all 810,000 3-digit × 3-digit problems and evaluate correctness
          with greedy decoding. Cache results for re-use.
Phase 2 — Select a carry-balanced subset using hierarchical stratification on carry_0,
          then carry_1 within each carry_0 group.

The output is a JSON file that pipeline.py reads via the `problems_file` config key.

Usage:
  python generate_l5_problems.py --config config.yaml           # Full run (enumerate + evaluate + select)
  python generate_l5_problems.py --config config.yaml --reselect  # Skip evaluation, re-run selection only
"""

import argparse
import json
import logging
import time
from collections import Counter
from logging.handlers import RotatingFileHandler
from math import ceil
from pathlib import Path

import numpy as np
import yaml

# Import reusable functions from pipeline.py
from pipeline import compute_labels, load_config, load_model, load_tokenizer, parse_number


# =============================================================================
# CONSTANTS
# =============================================================================

A_LO, A_HI = 100, 999
B_LO, B_HI = 100, 999
TOTAL_PROBLEMS = (A_HI - A_LO + 1) * (B_HI - B_LO + 1)  # 810,000

# Selection parameters
CORRECT_CAP = 500     # max correct problems to include per carry_0 value
FLOOR_ALL = 500       # min total problems per carry_0 value in selected set
FLOOR_CORRECT = 100   # target: at least this many correct per concept value

BATCH_SIZE = 256      # generation batch size (A6000 48GB has ~32GB headroom)
MAX_NEW_TOKENS = 12


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(cfg):
    log_dir = Path(cfg["paths"]["logs_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "generate_l5_problems.log"

    logger = logging.getLogger("l5_screen")
    logger.setLevel(logging.DEBUG)

    fh = RotatingFileHandler(log_path, maxBytes=50 * 1024 * 1024, backupCount=2)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s"
    ))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# =============================================================================
# STEP 1: ENUMERATE AND LABEL
# =============================================================================

def enumerate_all_l5(logger):
    """Generate all 810,000 (a, b) pairs and compute carry values.

    Returns dict with numpy arrays: a_vals, b_vals, products, carries (per column).
    """
    logger.info(f"Enumerating all {TOTAL_PROBLEMS:,} L5 problems...")
    t0 = time.time()

    a_vals = np.empty(TOTAL_PROBLEMS, dtype=np.int32)
    b_vals = np.empty(TOTAL_PROBLEMS, dtype=np.int32)
    products = np.empty(TOTAL_PROBLEMS, dtype=np.int64)

    # L5 has 5 carry columns (3-digit × 3-digit → up to 6-digit product)
    carries = [np.empty(TOTAL_PROBLEMS, dtype=np.int8) for _ in range(5)]

    idx = 0
    for a in range(A_LO, A_HI + 1):
        for b in range(B_LO, B_HI + 1):
            a_vals[idx] = a
            b_vals[idx] = b
            products[idx] = a * b

            # Compute carries via column sums (same algorithm as pipeline.py)
            a_digits = [a % 10, (a // 10) % 10, (a // 100) % 10]
            b_digits = [b % 10, (b // 10) % 10, (b // 100) % 10]
            n_cols = 5  # 3 + 3 - 1

            col_sums = [0] * n_cols
            for i in range(3):
                for j in range(3):
                    col_sums[i + j] += a_digits[i] * b_digits[j]

            carry = 0
            for col in range(n_cols):
                rs = col_sums[col] + carry
                carry = rs // 10
                carries[col][idx] = carry

            idx += 1

    elapsed = time.time() - t0
    logger.info(f"Enumeration complete: {idx:,} problems in {elapsed:.1f}s")

    return {
        "a_vals": a_vals,
        "b_vals": b_vals,
        "products": products,
        "carries": carries,  # list of 5 arrays
    }


# =============================================================================
# STEP 2: EVALUATE CORRECTNESS
# =============================================================================

def evaluate_correctness(enumeration, cfg, logger):
    """Run greedy decoding on all 810K problems, return boolean correct array."""
    a_vals = enumeration["a_vals"]
    b_vals = enumeration["b_vals"]
    products = enumeration["products"]
    n = len(a_vals)

    logger.info(f"Evaluating {n:,} problems with model...")

    # Load model and tokenizer
    tokenizer = load_tokenizer(cfg, logger)
    model, device = load_model(cfg, logger)

    # Build prompts
    tmpl = cfg["dataset"]["prompt_template"]
    prompts = [tmpl.format(a=int(a_vals[i]), b=int(b_vals[i])) for i in range(n)]

    correct = np.zeros(n, dtype=bool)

    import torch

    t0 = time.time()
    with torch.no_grad():
        for bs in range(0, n, BATCH_SIZE):
            be = min(bs + BATCH_SIZE, n)
            batch = prompts[bs:be]

            inputs = tokenizer(batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]

            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            for i in range(len(batch)):
                gen_ids = outputs[i][input_len:]
                raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
                predicted = parse_number(raw)
                correct[bs + i] = (predicted == int(products[bs + i]))

            done = be
            if (done % (BATCH_SIZE * 50)) < BATCH_SIZE or done == n:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n - done) / rate if rate > 0 else 0
                n_correct_so_far = correct[:done].sum()
                logger.info(
                    f"  {done:>7,}/{n:,} "
                    f"({done/n*100:.1f}%) "
                    f"correct_so_far={n_correct_so_far:,} "
                    f"({n_correct_so_far/done*100:.2f}%) "
                    f"ETA={eta/60:.0f}min"
                )

    elapsed = time.time() - t0
    n_correct = correct.sum()
    logger.info(
        f"Evaluation complete in {elapsed/60:.1f}min: "
        f"{n_correct:,}/{n:,} correct ({n_correct/n*100:.3f}%)"
    )

    # Free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return correct


def save_cache(enumeration, correct, cache_path, logger):
    """Save evaluation results for re-use."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        a_vals=enumeration["a_vals"],
        b_vals=enumeration["b_vals"],
        products=enumeration["products"],
        correct=correct,
        carry_0=enumeration["carries"][0],
        carry_1=enumeration["carries"][1],
        carry_2=enumeration["carries"][2],
        carry_3=enumeration["carries"][3],
        carry_4=enumeration["carries"][4],
    )
    size_mb = cache_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved evaluation cache: {cache_path} ({size_mb:.1f} MB)")


def load_cache(cache_path, logger):
    """Load evaluation cache. Returns (enumeration dict, correct array)."""
    logger.info(f"Loading evaluation cache: {cache_path}")
    data = np.load(cache_path)
    enumeration = {
        "a_vals": data["a_vals"],
        "b_vals": data["b_vals"],
        "products": data["products"],
        "carries": [data[f"carry_{i}"] for i in range(5)],
    }
    correct = data["correct"]
    logger.info(
        f"Loaded {len(correct):,} problems, "
        f"{correct.sum():,} correct ({correct.mean()*100:.3f}%)"
    )
    return enumeration, correct


# =============================================================================
# STEP 3: BALANCED SELECTION
# =============================================================================

def stratified_sample(indices, labels, n_target, rng):
    """Sample n_target indices, stratified by labels.

    For each unique label value, allocate proportionally with a floor of 1.
    If n_target >= len(indices), return all indices.
    """
    if n_target >= len(indices):
        return indices.copy()

    unique_vals = np.unique(labels)
    if len(unique_vals) <= 1:
        return rng.choice(indices, size=n_target, replace=False)

    # Proportional allocation with floor
    val_indices = {v: indices[labels == v] for v in unique_vals}
    val_counts = {v: len(idx) for v, idx in val_indices.items()}
    total = sum(val_counts.values())

    allocation = {}
    remaining = n_target
    for v in unique_vals:
        allocation[v] = max(1, int(n_target * val_counts[v] / total))
        allocation[v] = min(allocation[v], val_counts[v])  # can't exceed available
        remaining -= allocation[v]

    # Distribute remaining quota to largest groups
    for v in sorted(unique_vals, key=lambda x: -val_counts[x]):
        if remaining <= 0:
            break
        extra = min(remaining, val_counts[v] - allocation[v])
        allocation[v] += extra
        remaining -= extra

    selected = []
    for v in unique_vals:
        pool = val_indices[v]
        n = min(allocation[v], len(pool))
        selected.append(rng.choice(pool, size=n, replace=False))

    return np.concatenate(selected)


def select_balanced(enumeration, correct, logger):
    """Select a carry-balanced subset from the screened 810K problems.

    Strategy:
    - For each carry_0 value:
      - If correct count <= CORRECT_CAP: include ALL problems (to get every correct answer)
      - If correct count > CORRECT_CAP: include CORRECT_CAP correct + enough wrong for balance
    - Within each group, stratify on carry_1 when subsampling
    """
    rng = np.random.RandomState(42)
    carry_0 = enumeration["carries"][0]
    carry_1 = enumeration["carries"][1]
    n_total = len(correct)

    logger.info("=" * 60)
    logger.info("SELECTION ALGORITHM")
    logger.info(f"Parameters: CORRECT_CAP={CORRECT_CAP}, FLOOR_ALL={FLOOR_ALL}")
    logger.info("=" * 60)

    all_indices = np.arange(n_total)
    selected_indices = []

    carry_0_vals = sorted(np.unique(carry_0))
    logger.info(f"carry_0 values: {carry_0_vals}")

    for v0 in carry_0_vals:
        mask_v0 = carry_0 == v0
        group_indices = all_indices[mask_v0]
        group_correct = correct[mask_v0]
        group_carry_1 = carry_1[mask_v0]

        n_group = len(group_indices)
        n_group_correct = group_correct.sum()
        group_accuracy = n_group_correct / n_group if n_group > 0 else 0

        correct_indices = group_indices[group_correct]
        wrong_indices = group_indices[~group_correct]

        if n_group_correct <= CORRECT_CAP:
            # Rare group: include ALL problems to get every correct answer
            selected_indices.append(group_indices)
            logger.info(
                f"  carry_0={v0}: INCLUDE ALL {n_group:,} "
                f"(correct={n_group_correct:,}, accuracy={group_accuracy:.3f})"
            )
        else:
            # Abundant group: cap correct, sample wrong for balance
            n_correct_to_include = CORRECT_CAP

            # Select correct problems, stratified on carry_1
            correct_carry_1 = carry_1[correct_indices]
            selected_correct = stratified_sample(
                correct_indices, correct_carry_1, n_correct_to_include, rng
            )

            # Determine how many wrong to include
            # Enough total so that the all-population has FLOOR_ALL, or match the
            # ratio implied by capping correct
            n_total_target = max(
                FLOOR_ALL,
                ceil(n_correct_to_include / group_accuracy) if group_accuracy > 0 else FLOOR_ALL,
            )
            n_wrong_to_include = max(0, n_total_target - len(selected_correct))

            # Select wrong problems, stratified on carry_1
            wrong_carry_1 = carry_1[wrong_indices]
            selected_wrong = stratified_sample(
                wrong_indices, wrong_carry_1, n_wrong_to_include, rng
            )

            selected_indices.append(selected_correct)
            selected_indices.append(selected_wrong)

            n_sel = len(selected_correct) + len(selected_wrong)
            logger.info(
                f"  carry_0={v0}: SELECT {n_sel:,} "
                f"(correct={len(selected_correct):,}/{n_group_correct:,}, "
                f"wrong={len(selected_wrong):,}, "
                f"accuracy={group_accuracy:.3f})"
            )

    # Combine and sort by original index (preserves a deterministic order)
    selected = np.unique(np.concatenate(selected_indices))
    logger.info(f"\nTotal selected: {len(selected):,} problems")
    logger.info(f"Selected correct: {correct[selected].sum():,}")
    logger.info(f"Selected wrong: {(~correct[selected]).sum():,}")

    return selected


# =============================================================================
# STEP 4: VERIFY AND SAVE
# =============================================================================

def print_balance_report(enumeration, correct, selected, logger):
    """Print detailed balance report for all carry variables and input digits."""
    a_vals = enumeration["a_vals"][selected]
    b_vals = enumeration["b_vals"][selected]
    sel_correct = correct[selected]

    logger.info("\n" + "=" * 60)
    logger.info("BALANCE REPORT")
    logger.info("=" * 60)

    report = {}

    # Carry balance
    for ci in range(5):
        carry_name = f"carry_{ci}"
        carry_vals = enumeration["carries"][ci][selected]
        unique_vals = sorted(np.unique(carry_vals))

        logger.info(f"\n{carry_name}:")
        carry_report = {}
        for v in unique_vals:
            mask = carry_vals == v
            total = mask.sum()
            cor = (mask & sel_correct).sum()
            flag = " *** BELOW 100" if cor < FLOOR_CORRECT else ""
            logger.info(f"  value={v:>3d}: total={total:>6,}, correct={cor:>5,}{flag}")
            carry_report[int(v)] = {"total": int(total), "correct": int(cor)}
        report[carry_name] = carry_report

    # Input digit balance
    for name, vals in [
        ("a_units", a_vals % 10),
        ("a_tens", (a_vals // 10) % 10),
        ("a_hundreds", (a_vals // 100) % 10),
        ("b_units", b_vals % 10),
        ("b_tens", (b_vals // 10) % 10),
        ("b_hundreds", (b_vals // 100) % 10),
    ]:
        unique_vals = sorted(np.unique(vals))
        logger.info(f"\n{name}:")
        digit_report = {}
        for v in unique_vals:
            mask = vals == v
            total = mask.sum()
            cor = (mask & sel_correct).sum()
            flag = " *** BELOW 100" if cor < FLOOR_CORRECT else ""
            logger.info(f"  value={v}: total={total:>6,}, correct={cor:>5,}{flag}")
            digit_report[int(v)] = {"total": int(total), "correct": int(cor)}
        report[name] = digit_report

    return report


def identify_hard_ceilings(enumeration, correct, logger):
    """Identify carry values where correct count in ALL 810K < FLOOR_CORRECT."""
    logger.info("\n" + "=" * 60)
    logger.info("HARD CEILINGS (full 810K space)")
    logger.info("=" * 60)

    ceilings = {}
    for ci in range(5):
        carry_name = f"carry_{ci}"
        carry_vals = enumeration["carries"][ci]
        unique_vals = sorted(np.unique(carry_vals))

        for v in unique_vals:
            mask = carry_vals == v
            n_correct = (mask & correct).sum()
            n_total = mask.sum()
            if n_correct < FLOOR_CORRECT:
                key = f"{carry_name}_{v}"
                ceilings[key] = {
                    "total_in_space": int(n_total),
                    "correct_in_space": int(n_correct),
                }
                logger.info(
                    f"  {carry_name}={v}: {n_correct:,} correct out of {n_total:,} "
                    f"(ceiling — cannot reach {FLOOR_CORRECT})"
                )

    if not ceilings:
        logger.info("  No hard ceilings — all carry values have >= 100 correct in 810K")

    return ceilings


def save_selected(enumeration, correct, selected, balance_report, ceilings, output_path, logger):
    """Save the selected problem set for pipeline.py."""
    a_selected = enumeration["a_vals"][selected].tolist()
    b_selected = enumeration["b_vals"][selected].tolist()

    n_selected = len(selected)
    n_correct = int(correct[selected].sum())
    n_total_correct_in_space = int(correct.sum())

    output = {
        "metadata": {
            "total_screened": int(len(correct)),
            "total_correct_in_space": n_total_correct_in_space,
            "accuracy": float(correct.mean()),
            "selection_parameters": {
                "correct_cap": CORRECT_CAP,
                "floor_all": FLOOR_ALL,
                "floor_correct": FLOOR_CORRECT,
            },
            "balance_report": balance_report,
            "hard_ceilings": ceilings,
        },
        "a": a_selected,
        "b": b_selected,
        "n_selected": n_selected,
        "n_correct": n_correct,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"\nSaved selected problems: {output_path} ({size_mb:.1f} MB)")
    logger.info(f"  n_selected={n_selected:,}, n_correct={n_correct:,}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Two-phase L5 dataset generation: screen 810K, select balanced subset"
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--reselect", action="store_true",
        help="Skip model evaluation, load cache, re-run selection only"
    )
    args = parser.parse_args()

    # Config
    cfg = load_config(args.config)
    logger = setup_logging(cfg)

    logger.info("=" * 60)
    logger.info("L5 Two-Phase Dataset Generation")
    logger.info("=" * 60)

    data_root = Path(cfg["paths"]["data_root"])
    screening_dir = data_root / "l5_screening"
    screening_dir.mkdir(parents=True, exist_ok=True)

    cache_path = screening_dir / "l5_evaluation_cache.npz"
    output_path = screening_dir / "l5_selected_problems.json"

    t_start = time.time()

    if args.reselect and cache_path.exists():
        # Re-selection mode: load from cache
        enumeration, correct = load_cache(cache_path, logger)
    else:
        # Full run: enumerate, evaluate, cache
        enumeration = enumerate_all_l5(logger)
        correct = evaluate_correctness(enumeration, cfg, logger)
        save_cache(enumeration, correct, cache_path, logger)

    # Selection
    selected = select_balanced(enumeration, correct, logger)

    # Verification
    balance_report = print_balance_report(enumeration, correct, selected, logger)
    ceilings = identify_hard_ceilings(enumeration, correct, logger)

    # Save
    save_selected(enumeration, correct, selected, balance_report, ceilings, output_path, logger)

    elapsed = time.time() - t_start
    logger.info(f"\nTotal time: {elapsed/60:.1f} min")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
