#!/usr/bin/env python3
"""Build the curated set for Part B (B.2 onward).

Selects 8,000 multiplication problems from L3, L4, L5 with:
- Stratified concept coverage (per-(concept, value) floor of 30 where math allows)
- Difficulty-matched correct/wrong pairs at L4 (1,000) and L5 (1,400)
- Joint-coverage check on (concept_i, concept_j) pairs

Outputs:
  /data/user_data/anshulk/arithmetic-geometry/curated/curated_set_v1.json
  /home/anshulk/arithmetic-geometry/docs/curated_set_coverage_report.md
  /home/anshulk/arithmetic-geometry/logs/build_curated_set.log

Plan: /home/anshulk/.claude/plans/b-1-step-1-deep-comet.md
"""

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pipeline import compute_labels, load_config
from generate_l5_problems import stratified_sample


# =============================================================================
# CONSTANTS
# =============================================================================

LEVELS = [3, 4, 5]
CONCEPT_FLOOR = 30
MATCHING_LOSS_LIMIT = 0.30          # >30% unmatched aborts the build
UNDOCUMENTED_GAP_LIMIT = 10
ROUND_TRIP_SAMPLE = 100             # rows checked in Pass 0 + Pass 4
MATCHED_PAIR_PERMUTATIONS = 1000

# Resolved budget (replaces B.1.2 in docs/next_steps.md)
BUDGET = {
    3: {"correct": 1200, "wrong": 1200},
    4: {"correct": 1000, "wrong": 1800},
    5: {"correct": 1400, "wrong": 1400},
}

# Concept registry — single source of truth: phase_g_fourier.py:56-74
DIGIT_CONCEPTS_BY_LEVEL = {
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
    3: ["carry_0", "carry_1"],
    4: ["carry_0", "carry_1", "carry_2"],
    5: ["carry_0", "carry_1", "carry_2", "carry_3", "carry_4"],
}

# Concepts whose leading-digit position excludes 0
LEADING_DIGIT_CONCEPTS = {"a_hundreds", "b_hundreds", "ans_digit_0_msf"}

# Mathematically excluded (level, concept, value) cells — operand-tens cannot be 0 for L3/L4
# because the operand range is [10, 99] so the tens digit is always ≥ 1.
# (At L5, both operands are in [100, 999] so tens=0 IS achievable, e.g., 105.)
MATHEMATICALLY_EXCLUDED_CELLS = {
    (3, "a_tens", 0),
    (3, "b_tens", 0),
    (4, "b_tens", 0),
}

# Documented hard-ceiling gaps — populated at runtime in pass0 from the global pool.
# Any (level, concept, value) cell whose underlying pool has < CONCEPT_FLOOR examples
# is by definition unreachable; classify as documented rather than failing.
DOCUMENTED_GAPS = {}  # (level, concept, value) -> reason

# Matching axes for Pass 3
MATCHING_AXES = ("magnitude_tier", "carry_count_tier", "answer_length")

# Output paths
DEFAULT_DATA_ROOT = Path("/data/user_data/anshulk/arithmetic-geometry")
DEFAULT_REPO_ROOT = Path("/home/anshulk/arithmetic-geometry")


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(repo_root):
    log_dir = repo_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "build_curated_set.log"

    # Preserve previous log
    if log_path.exists():
        prev = log_path.with_suffix(".log.prev")
        if prev.exists():
            prev.unlink()
        log_path.rename(prev)

    logger = logging.getLogger("build_curated_set")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # idempotent on rerun

    fh = RotatingFileHandler(log_path, maxBytes=20 * 1024 * 1024, backupCount=2)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    ))

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S"
    ))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, log_path


# =============================================================================
# TIER HELPERS
# =============================================================================

def magnitude_tier_of_operand(x):
    """0 if leading digit in {1,2,3}; 1 if {4,5,6}; 2 if {7,8,9}."""
    leading = int(str(x)[0])
    if leading in (1, 2, 3):
        return 0
    if leading in (4, 5, 6):
        return 1
    return 2


def magnitude_tier(a, b):
    """Joint tier in [0, 8]: 3 * tier(a) + tier(b)."""
    return 3 * magnitude_tier_of_operand(a) + magnitude_tier_of_operand(b)


def carry_count_tier(carries):
    """0 (low) if nzc<=1, 1 (medium) if nzc in {2,3}, 2 (high) if nzc>=4."""
    nzc = sum(1 for c in carries if c > 0)
    if nzc <= 1:
        return 0
    if nzc <= 3:
        return 1
    return 2


def answer_length(a, b):
    return len(str(a * b))


def nonzero_carry_count(carries):
    return sum(1 for c in carries if c > 0)


def leading_digit_pair_index(a, b):
    """Encode (leading_a, leading_b) as 9*la + lb, both in 1..9."""
    la = int(str(a)[0])
    lb = int(str(b)[0])
    return 9 * (la - 1) + (lb - 1)


# =============================================================================
# PASS 0 — load pools (labels join answers)
# =============================================================================

def load_pools(cfg, logger):
    logger.info("Pass 0 start: loading labels and answers for L3, L4, L5.")
    labels_dir = Path(cfg["paths"]["labels_dir"])
    answers_dir = Path(cfg["paths"]["answers_dir"])

    pools = {}
    for lvl in LEVELS:
        labels_path = labels_dir / f"level_{lvl}.json"
        answers_path = answers_dir / f"level_{lvl}.json"
        logger.debug(f"Loading {labels_path}")
        with open(labels_path) as f:
            labels_doc = json.load(f)
        labels = labels_doc["problems"]
        logger.debug(f"Loading {answers_path}")
        with open(answers_path) as f:
            answers_doc = json.load(f)
        answers = answers_doc["results"]

        if len(labels) != len(answers):
            raise RuntimeError(
                f"L{lvl}: label/answer length mismatch "
                f"({len(labels)} vs {len(answers)})"
            )

        rows = []
        n_mismatches = 0
        for i, (lab_outer, ans) in enumerate(zip(labels, answers)):
            # On-disk labels are wrapped: each entry has {index, prompt, labels: {...}}.
            lab = lab_outer["labels"]
            if lab["a"] != ans["a"] or lab["b"] != ans["b"]:
                logger.error(
                    f"L{lvl} row {i}: (a,b) mismatch labels=({lab['a']},{lab['b']}) "
                    f"answers=({ans['a']},{ans['b']})"
                )
                n_mismatches += 1
            if lab["product"] != ans["ground_truth"]:
                logger.error(
                    f"L{lvl} row {i}: product/ground_truth mismatch "
                    f"({lab['product']} vs {ans['ground_truth']})"
                )
                n_mismatches += 1
            rows.append({
                "source_index": i,
                "level": lvl,
                "a": lab["a"],
                "b": lab["b"],
                "product": lab["product"],
                "carries": lab["carries"],
                "predicted": ans["predicted"],
                "correct": bool(ans["correct"]),
                "raw_text": ans["raw_text"],
            })
        if n_mismatches:
            raise RuntimeError(
                f"L{lvl}: {n_mismatches} (a,b)/product mismatches between "
                f"labels and answers — aborting."
            )

        df = pd.DataFrame(rows)
        # Dedup on (a, b) within the level — keep first occurrence so every later
        # pass operates on a unique-(a,b) pool. L3 has ~1,900 duplicate (a,b) rows
        # because it was sampled randomly from the 8,100 unique 2-digit×2-digit
        # space; L4 and L5 have negligible duplicates.
        n_pre = len(df)
        df = df.drop_duplicates(subset=["a", "b"], keep="first").reset_index(drop=True)
        n_dropped = n_pre - len(df)
        n_correct = int(df["correct"].sum())
        n_wrong = int((~df["correct"]).sum())
        logger.info(
            f"Pass 0 L{lvl}: loaded {len(df)} unique-(a,b) rows "
            f"({n_correct}c/{n_wrong}w); dropped {n_dropped} duplicate rows."
        )
        pools[lvl] = df

    return pools


def derive_documented_gaps(pools, logger):
    """Populate DOCUMENTED_GAPS based on pool-level scarcity.

    A (level, concept, value) cell is documented if either:
    - The cell is in MATHEMATICALLY_EXCLUDED_CELLS (operand-tens=0 at L3/L4).
    - The full pool at that level has fewer than CONCEPT_FLOOR examples for that
      (concept, value) combination. The curated set cannot exceed the pool.
    """
    for cell in MATHEMATICALLY_EXCLUDED_CELLS:
        DOCUMENTED_GAPS[cell] = "mathematically excluded (operand range starts at 10)"

    for lvl, df in pools.items():
        records = df.to_dict("records")
        for concept in _all_concepts_for_level(lvl):
            counts = _per_value_counts(records, concept)
            for v in _value_range_for_concept(concept, lvl):
                pool_n = counts.get(v, 0)
                if pool_n < CONCEPT_FLOOR:
                    if (lvl, concept, v) in MATHEMATICALLY_EXCLUDED_CELLS:
                        continue  # already documented
                    DOCUMENTED_GAPS[(lvl, concept, v)] = (
                        f"L{lvl} pool has only {pool_n} examples (< floor {CONCEPT_FLOOR})"
                    )
    logger.info(
        f"Documented gap set derived: {len(DOCUMENTED_GAPS)} cells "
        f"(includes {len(MATHEMATICALLY_EXCLUDED_CELLS)} math-excluded)."
    )
    for cell, reason in sorted(DOCUMENTED_GAPS.items()):
        logger.debug(f"DOCUMENTED_GAP {cell}: {reason}")


def enrich_with_tiers_inplace(pools, logger):
    logger.info("Computing magnitude_tier, carry_count_tier, answer_length per row.")
    for lvl, df in pools.items():
        df["magnitude_tier"] = df.apply(
            lambda r: magnitude_tier(r["a"], r["b"]), axis=1
        ).astype(np.int8)
        df["carry_count_tier"] = df["carries"].apply(carry_count_tier).astype(np.int8)
        df["answer_length"] = df.apply(
            lambda r: answer_length(r["a"], r["b"]), axis=1
        ).astype(np.int8)
        df["nonzero_carry_count"] = df["carries"].apply(nonzero_carry_count).astype(np.int8)
        df["leading_digit_pair_index"] = df.apply(
            lambda r: leading_digit_pair_index(r["a"], r["b"]), axis=1
        ).astype(np.int16)
        # Cell key for stratification
        df["cell_key"] = list(zip(df["magnitude_tier"], df["carry_count_tier"], df["answer_length"]))
        n_cells = df["cell_key"].nunique()
        logger.info(
            f"Tier enrichment L{lvl}: rows={len(df)}, distinct cells={n_cells}"
        )
        for tier_name in ("magnitude_tier", "carry_count_tier", "answer_length"):
            hist = dict(Counter(df[tier_name]))
            logger.debug(f"L{lvl} {tier_name} histogram: {hist}")


def round_trip_compute_labels_check(pools, logger, sample_size=ROUND_TRIP_SAMPLE):
    """Verify on-disk labels round-trip through compute_labels()."""
    logger.info(
        f"compute_labels round-trip check: sampling {sample_size} rows per level."
    )
    rng = np.random.RandomState(0)
    labels_dir = Path(load_config()["paths"]["labels_dir"])  # use canonical config

    for lvl, df in pools.items():
        with open(labels_dir / f"level_{lvl}.json") as f:
            disk_labels = json.load(f)["problems"]
        n = len(df)
        idx_sample = rng.choice(n, size=min(sample_size, n), replace=False)
        for idx in idx_sample:
            lab_disk = disk_labels[int(idx)]["labels"]
            lab_recomp = compute_labels(lab_disk["a"], lab_disk["b"])
            for k in ("a", "b", "product", "carries", "answer_digits_msf"):
                if lab_disk[k] != lab_recomp[k]:
                    raise RuntimeError(
                        f"compute_labels round-trip failed at L{lvl} idx={idx} "
                        f"key={k}: disk={lab_disk[k]} recomp={lab_recomp[k]}"
                    )
        logger.debug(f"L{lvl}: {len(idx_sample)} rows round-tripped cleanly.")
    logger.info("compute_labels round-trip check passed.")


# =============================================================================
# PASS 1 — difficulty stratification (L4, L5)
# =============================================================================

def _stratify_population(df_pop, n_target, rng, logger, label):
    """Stratified draw of n_target indices from df_pop using cell_key."""
    if len(df_pop) <= n_target:
        logger.warning(
            f"{label}: requested {n_target} but only {len(df_pop)} available; "
            f"taking all."
        )
        return df_pop["source_index"].to_numpy()

    indices = df_pop["source_index"].to_numpy()
    # Stable cell-key encoding for stratified_sample (it expects 1D label array)
    cell_to_int = {c: i for i, c in enumerate(sorted(set(df_pop["cell_key"])))}
    label_arr = np.array([cell_to_int[c] for c in df_pop["cell_key"]])
    drawn = stratified_sample(indices, label_arr, n_target, rng)

    # Per-cell allocation report (DEBUG)
    cell_counts_pool = Counter(df_pop["cell_key"])
    drawn_set = set(drawn.tolist())
    cell_counts_drawn = Counter(
        df_pop[df_pop["source_index"].isin(drawn_set)]["cell_key"]
    )
    for cell in sorted(cell_to_int):
        pool_n = cell_counts_pool[cell]
        drawn_n = cell_counts_drawn[cell]
        logger.debug(
            f"{label} cell={cell} pool={pool_n} drawn={drawn_n}"
        )
    return drawn


def pass1_difficulty_stratify(pools, rng, logger):
    logger.info("Pass 1 start: stratify L4 (1000c/1800w) and L5 (1400c/1400w) by tier triple.")
    out = {}
    for lvl in (4, 5):
        df = pools[lvl]
        df_correct = df[df["correct"]]
        df_wrong = df[~df["correct"]]
        n_correct = BUDGET[lvl]["correct"]
        n_wrong = BUDGET[lvl]["wrong"]
        idx_c = _stratify_population(df_correct, n_correct, rng, logger,
                                       f"Pass1 L{lvl} correct")
        idx_w = _stratify_population(df_wrong, n_wrong, rng, logger,
                                       f"Pass1 L{lvl} wrong")
        out[lvl] = {
            "correct": list(idx_c),
            "wrong": list(idx_w),
        }
        logger.info(
            f"Pass 1 L{lvl}: drew {len(idx_c)} correct + {len(idx_w)} wrong."
        )
    return out


# =============================================================================
# PASS 2 — concept coverage (L3 entirely, plus L4/L5 fill)
# =============================================================================

def _digit_value_for_concept(row, concept):
    """Extract the value of `concept` from an enriched row."""
    if concept.startswith("a_"):
        if concept == "a_units":
            return int(str(row["a"]).rjust(3, "0")[-1])
        if concept == "a_tens":
            s = str(row["a"]).rjust(3, "0")
            return int(s[-2])
        if concept == "a_hundreds":
            s = str(row["a"]).rjust(3, "0")
            return int(s[-3])
    if concept.startswith("b_"):
        if concept == "b_units":
            return int(str(row["b"]).rjust(3, "0")[-1])
        if concept == "b_tens":
            s = str(row["b"]).rjust(3, "0")
            return int(s[-2])
        if concept == "b_hundreds":
            s = str(row["b"]).rjust(3, "0")
            return int(s[-3])
    if concept.startswith("ans_digit_") and concept.endswith("_msf"):
        # ans_digit_0_msf is the leading digit; index increments rightward
        idx = int(concept.split("_")[2])
        product = row["product"]
        s = str(product)
        if idx >= len(s):
            return None
        return int(s[idx])
    if concept.startswith("carry_"):
        idx = int(concept.split("_")[1])
        carries = row["carries"]
        if idx >= len(carries):
            return None
        return carries[idx]
    raise ValueError(f"Unknown concept: {concept}")


def _value_range_for_concept(concept, level):
    """Iterable of mathematically valid values for (concept, level).

    Excludes math-impossible cells like a_tens=0 / b_tens=0 at L3/L4
    (operand range [10, 99] forces tens >= 1).
    """
    # Operand-tens cells where the operand range is [10, 99] — tens must be >= 1.
    if concept == "a_tens" and level in (3, 4):
        return range(1, 10)
    if concept == "b_tens" and level in (3, 4):
        return range(1, 10)

    if concept.startswith("carry_"):
        # Validated max ranges per next_steps.md A.2 / pipeline.compute_carry_bounds.
        if concept == "carry_0":
            return range(0, 9)
        if concept == "carry_1":
            return range(0, 18) if level == 5 else range(0, 19)  # safe upper bound
        if concept == "carry_2":
            return range(0, 27) if level == 5 else range(0, 19)
        if concept == "carry_3":
            return range(0, 19)
        if concept == "carry_4":
            return range(0, 10)
    if concept in LEADING_DIGIT_CONCEPTS:
        return range(1, 10)
    return range(0, 10)


def _per_value_counts(rows_iter, concept):
    """Count rows by concept value."""
    counts = Counter()
    for row in rows_iter:
        v = _digit_value_for_concept(row, concept)
        if v is None:
            continue
        counts[v] += 1
    return counts


def _all_concepts_for_level(level):
    return DIGIT_CONCEPTS_BY_LEVEL[level] + CARRY_CONCEPTS_BY_LEVEL[level]


def _row_dict(df, idx):
    return df.loc[df["source_index"] == idx].iloc[0].to_dict()


def pass2_concept_coverage(pools, pass1_indices, rng, logger):
    logger.info(
        f"Pass 2 start: concept floor={CONCEPT_FLOOR}, registry has 17 unique concepts."
    )
    out = {3: {"correct": [], "wrong": []}, 4: {"correct": [], "wrong": []},
           5: {"correct": [], "wrong": []}}

    # L3: stratify on cell_key, then concept-fill from any below-floor cells
    df3 = pools[3]
    df3_c = df3[df3["correct"]]
    df3_w = df3[~df3["correct"]]
    out[3]["correct"] = list(_stratify_population(
        df3_c, BUDGET[3]["correct"], rng, logger, "Pass2 L3 correct"
    ))
    out[3]["wrong"] = list(_stratify_population(
        df3_w, BUDGET[3]["wrong"], rng, logger, "Pass2 L3 wrong"
    ))
    logger.info(
        f"Pass 2 L3: drew {len(out[3]['correct'])} correct + "
        f"{len(out[3]['wrong'])} wrong."
    )

    # L3 / L4 / L5: fill below-floor concept cells against the unused pool
    for lvl in LEVELS:
        df = pools[lvl]
        if lvl == 3:
            already = set(out[3]["correct"]) | set(out[3]["wrong"])
        else:
            already = set(pass1_indices[lvl]["correct"]) | set(pass1_indices[lvl]["wrong"])
        # Compute current counts per (concept, value) over the Pass1 selection
        sel = df[df["source_index"].isin(already)]
        counts_by_concept = {}
        for concept in _all_concepts_for_level(lvl):
            counts_by_concept[concept] = _per_value_counts(sel.to_dict("records"), concept)

        # Identify below-floor non-documented cells and how much they need
        gaps = []  # list of (concept, value, need)
        for concept in _all_concepts_for_level(lvl):
            for v in _value_range_for_concept(concept, lvl):
                if (lvl, concept, v) in DOCUMENTED_GAPS:
                    continue
                cur = counts_by_concept[concept][v]
                if cur < CONCEPT_FLOOR:
                    gaps.append((concept, v, CONCEPT_FLOOR - cur))

        if not gaps:
            logger.info(f"Pass 2 L{lvl}: Pass-1 selection already meets all floors.")
            continue

        logger.info(
            f"Pass 2 L{lvl}: {len(gaps)} below-floor cells; greedy-filling."
        )

        # Pool of unused rows
        pool_df = df[~df["source_index"].isin(already)].copy()
        pool_records = pool_df.to_dict("records")

        # Pre-index pool rows by (concept, value)
        index_by_cv = defaultdict(list)
        for rec in pool_records:
            for concept in _all_concepts_for_level(lvl):
                v = _digit_value_for_concept(rec, concept)
                if v is None:
                    continue
                index_by_cv[(concept, v)].append(rec["source_index"])

        # Greedy multi-cell-coverage gain.
        # We use index_by_cv to limit the candidate scan: only rows that touch
        # at least one currently-deficient (concept, value) cell can have positive gain.
        added = []
        gap_remaining = {(c, v): need for c, v, need in gaps}
        used = set()
        # Map source_index -> record for O(1) lookup once we know which idx is best.
        rec_by_src = {rec["source_index"]: rec for rec in pool_records}
        max_rounds = 10000
        round_n = 0
        while gap_remaining and round_n < max_rounds:
            round_n += 1
            # Build the candidate set: union of source_indices that touch any
            # currently-deficient (concept, value) cell, minus rows already used.
            candidate_src = set()
            for key in gap_remaining:
                for src in index_by_cv.get(key, []):
                    if src not in used:
                        candidate_src.add(src)
            if not candidate_src:
                break

            best_idx = None
            best_gain = 0
            best_row = None
            for src in candidate_src:
                rec = rec_by_src[src]
                gain = 0
                for concept in _all_concepts_for_level(lvl):
                    v = _digit_value_for_concept(rec, concept)
                    if v is None:
                        continue
                    if gap_remaining.get((concept, v), 0) > 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_idx = src
                    best_row = rec
            if best_idx is None or best_gain == 0:
                break
            used.add(best_idx)
            added.append(best_idx)
            # Decrement gaps for cells this row touches
            for concept in _all_concepts_for_level(lvl):
                v = _digit_value_for_concept(best_row, concept)
                if v is None:
                    continue
                key = (concept, v)
                if gap_remaining.get(key, 0) > 0:
                    gap_remaining[key] -= 1
                    if gap_remaining[key] == 0:
                        del gap_remaining[key]
            if round_n % 50 == 0:
                logger.debug(
                    f"Pass 2 L{lvl} greedy round {round_n}: gaps={len(gap_remaining)} added={len(added)}"
                )

        if gap_remaining:
            for (concept, v), need in gap_remaining.items():
                logger.warning(
                    f"COVERAGE_GAP_UNDOCUMENTED L{lvl} {concept}={v}: still short {need} after fill."
                )
        # Split added by correct/wrong; for L3 these extend the existing selection,
        # for L4/L5 they form the Pass-2 'extras' bucket combined later in main().
        added_set = set(added)
        added_correct = list(df[df["source_index"].isin(added_set) & df["correct"]]["source_index"])
        added_wrong = list(df[df["source_index"].isin(added_set) & ~df["correct"]]["source_index"])
        if lvl == 3:
            out[3]["correct"] = list(set(out[3]["correct"]) | set(added_correct))
            out[3]["wrong"] = list(set(out[3]["wrong"]) | set(added_wrong))
        else:
            out[lvl]["correct"] = added_correct
            out[lvl]["wrong"] = added_wrong
        logger.info(
            f"Pass 2 L{lvl}: added {len(added)} extras "
            f"({len(added_correct)} correct, {len(added_wrong)} wrong); "
            f"{len(gap_remaining)} gaps remaining."
        )

    return out


# =============================================================================
# PASS 3 — matched-pair construction (L4, L5)
# =============================================================================

def pass3_match_pairs(pools, combined, rng, logger):
    logger.info(
        "Pass 3 start: matching L4 (1000) and L5 (1400) correct->wrong on "
        "(magnitude_tier, carry_count_tier, answer_length); relax carry_count_tier ±1 on miss."
    )
    matches_all = {4: [], 5: []}
    unmatched_dropped = {4: 0, 5: 0}

    for lvl in (4, 5):
        df = pools[lvl]
        target_correct = BUDGET[lvl]["correct"]
        # The full correct universe for matching is Pass1 + Pass2 correct
        c_idx_set = set(combined[lvl]["correct"])
        # Wrong universe: everything we have in Pass1 + Pass2 wrong, plus whatever else
        # is available in df_wrong if we have to refill (we don't refill here — Pass3 only
        # selects from the budgeted wrongs initially, but allows Pass4 replacement loop).
        w_idx_set = set(combined[lvl]["wrong"])

        df_correct = df[df["source_index"].isin(c_idx_set)].copy()
        df_wrong_all = df[~df["correct"]].copy()  # full wrong pool for matching

        # Sort correct by descending difficulty
        df_correct_sorted = df_correct.sort_values(
            by=["magnitude_tier", "carry_count_tier", "answer_length"],
            ascending=[False, False, False]
        )

        used_wrong = set()
        pair_counter = 0
        n_strict = 0
        n_relaxed = 0
        kept_correct = []
        kept_wrong = []

        # Build a fast lookup of wrong rows by tier triple
        wrong_by_tier = defaultdict(list)
        for w in df_wrong_all.to_dict("records"):
            key = (w["magnitude_tier"], w["carry_count_tier"], w["answer_length"])
            wrong_by_tier[key].append(w)

        for _, c in df_correct_sorted.iterrows():
            ckey = (int(c["magnitude_tier"]), int(c["carry_count_tier"]), int(c["answer_length"]))
            # Strict
            cands = [w for w in wrong_by_tier[ckey] if w["source_index"] not in used_wrong]
            relaxed = False
            if not cands:
                # Relax carry_count_tier by ±1
                relaxed_cands = []
                for cct_offset in (-1, 1):
                    rkey = (ckey[0], ckey[1] + cct_offset, ckey[2])
                    relaxed_cands.extend(
                        w for w in wrong_by_tier.get(rkey, [])
                        if w["source_index"] not in used_wrong
                    )
                cands = relaxed_cands
                relaxed = True

            if not cands:
                logger.warning(
                    f"Pass 3 L{lvl}: unmatched correct idx={int(c['source_index'])} "
                    f"tier={ckey} (no candidates strict or relaxed)."
                )
                unmatched_dropped[lvl] += 1
                continue

            # L1 score over leading_digit_pair_index, nonzero_carry_count, answer_length
            def _score(w):
                return (
                    abs(int(w["leading_digit_pair_index"]) - int(c["leading_digit_pair_index"])) * 10
                    + abs(int(w["nonzero_carry_count"]) - int(c["nonzero_carry_count"])) * 3
                    + abs(int(w["answer_length"]) - int(c["answer_length"])) * 1
                )

            # Permute then pick min for deterministic tie-break
            order = rng.permutation(len(cands))
            cands_perm = [cands[i] for i in order]
            winner = min(cands_perm, key=_score)
            used_wrong.add(winner["source_index"])
            pair_id = f"L{lvl}_pair_{pair_counter:04d}"
            matches_all[lvl].append({
                "pair_id": pair_id,
                "correct_idx": int(c["source_index"]),
                "wrong_idx": int(winner["source_index"]),
                "relaxed": relaxed,
            })
            kept_correct.append(int(c["source_index"]))
            kept_wrong.append(int(winner["source_index"]))
            pair_counter += 1
            if relaxed:
                n_relaxed += 1
            else:
                n_strict += 1
            if pair_counter % 200 == 0:
                logger.info(
                    f"Pass 3 L{lvl}: {pair_counter} pairs ({n_strict} strict, {n_relaxed} relaxed)"
                )
            if pair_counter >= target_correct:
                break

        loss_rate = unmatched_dropped[lvl] / max(target_correct, 1)
        logger.info(
            f"Pass 3 L{lvl}: pairs={len(matches_all[lvl])} "
            f"(strict={n_strict}, relaxed={n_relaxed}); "
            f"unmatched_dropped={unmatched_dropped[lvl]} (loss={loss_rate:.1%})."
        )
        if loss_rate > MATCHING_LOSS_LIMIT:
            raise RuntimeError(
                f"MATCHING_LOSS L{lvl}: lost {loss_rate:.1%} of "
                f"{target_correct} correct rows; aborting."
            )

        # Stash kept indices for use in Pass 4 budget reconciliation
        combined[lvl]["matched_correct"] = kept_correct
        combined[lvl]["matched_wrong"] = kept_wrong

    return matches_all


# =============================================================================
# PASS 4 — validate, dedupe, write
# =============================================================================

def pass4_validate_and_trim(pools, pass1, pass2, matches_all, cfg, logger,
                              metadata_extras):
    logger.info("Pass 4 start: assembling final curated set.")

    # For L4/L5, keep matched-pair members and then top up wrongs to budget from
    # the Pass1+Pass2 wrong selection (matched wrongs may be < BUDGET[lvl]["wrong"]).
    selected_indices = {}  # level -> set of source_indices
    matched_pair_id = {}   # (level, source_index) -> pair_id
    matched_relaxed = {}   # same -> bool

    for lvl in (4, 5):
        # Precompute correct-flag map once per level (avoids rebuilding a pandas
        # index on every membership lookup below).
        correct_by_src = dict(zip(
            pools[lvl]["source_index"].astype(int),
            pools[lvl]["correct"].astype(bool),
        ))

        sel = set()
        for m in matches_all[lvl]:
            sel.add(m["correct_idx"])
            sel.add(m["wrong_idx"])
            matched_pair_id[(lvl, m["correct_idx"])] = m["pair_id"]
            matched_pair_id[(lvl, m["wrong_idx"])] = m["pair_id"]
            matched_relaxed[(lvl, m["correct_idx"])] = m["relaxed"]
            matched_relaxed[(lvl, m["wrong_idx"])] = m["relaxed"]

        n_matched_wrong = sum(1 for s in sel if not correct_by_src[s])
        n_matched_correct = len(sel) - n_matched_wrong
        wrong_budget = BUDGET[lvl]["wrong"]
        need_extra_wrong = wrong_budget - n_matched_wrong
        # Take Pass1 (and Pass2) wrongs that aren't already matched
        candidate_wrong = list(set(pass1[lvl]["wrong"]) | set(pass2[lvl]["wrong"]))
        extra_pool = [s for s in candidate_wrong if s not in sel]
        rng = np.random.RandomState(cfg["dataset"]["seed"] + lvl)
        if need_extra_wrong > 0 and extra_pool:
            chosen = rng.choice(extra_pool, size=min(need_extra_wrong, len(extra_pool)),
                                  replace=False)
            sel.update(int(x) for x in chosen)
            logger.info(
                f"Pass 4 L{lvl}: added {len(chosen)} unmatched wrongs to reach wrong budget {wrong_budget}."
            )
        elif need_extra_wrong > 0:
            logger.warning(
                f"Pass 4 L{lvl}: wrong budget short by {need_extra_wrong} (no extra pool)."
            )
        selected_indices[lvl] = sel
        logger.info(
            f"Pass 4 L{lvl}: selected {len(sel)} rows "
            f"({n_matched_correct} matched correct + {n_matched_wrong} matched wrong + extras)."
        )

    # L3: just take Pass2 correct + wrong
    sel3 = set(pass2[3]["correct"]) | set(pass2[3]["wrong"])
    selected_indices[3] = sel3
    logger.info(f"Pass 4 L3: selected {len(sel3)} rows (no matching at L3).")

    # Defensive within-level (a,b) dedup. Pass 0 already dedups the pool by (a, b),
    # so this should drop zero rows in normal operation. We keep the check as a
    # guard against future pool-loading changes — if it ever fires non-zero, that
    # is a bug, not expected behaviour.
    n_duplicates = 0
    final_problems = []
    curated_id = 0
    for lvl in LEVELS:
        df = pools[lvl]
        sub = df[df["source_index"].isin(selected_indices[lvl])]
        seen_ab = set()
        for _, r in sub.iterrows():
            ab = (int(r["a"]), int(r["b"]))
            if ab in seen_ab:
                n_duplicates += 1
                logger.warning(
                    f"Pass 4 unexpected duplicate dropped L{lvl} idx="
                    f"{int(r['source_index'])} (a,b)={ab} — Pass 0 dedup should "
                    f"have caught this; investigate."
                )
                continue
            seen_ab.add(ab)
            src = int(r["source_index"])
            final_problems.append({
                "curated_id": curated_id,
                "level": lvl,
                "source_index": src,
                "a": int(r["a"]),
                "b": int(r["b"]),
                "product": int(r["product"]),
                "predicted": int(r["predicted"]),
                "correct": bool(r["correct"]),
                "raw_text": r["raw_text"],
                "magnitude_tier": int(r["magnitude_tier"]),
                "carry_count_tier": int(r["carry_count_tier"]),
                "answer_length": int(r["answer_length"]),
                "nonzero_carry_count": int(r["nonzero_carry_count"]),
                "matched_pair_id": matched_pair_id.get((lvl, src)),
                "matched_relaxed": matched_relaxed.get((lvl, src), False) if matched_pair_id.get((lvl, src)) else None,
                # 'labels' filled in below from disk to preserve byte-identity
            })
            curated_id += 1
    logger.info(f"Pass 4 duplicates dropped: {n_duplicates}.")

    # Attach 'labels' to each problem (load disk labels once per level)
    for lvl in LEVELS:
        with open(Path(cfg["paths"]["labels_dir"]) / f"level_{lvl}.json") as f:
            disk_labels = json.load(f)["problems"]
        for p in final_problems:
            if p["level"] == lvl:
                p["labels"] = disk_labels[p["source_index"]]["labels"]

    # 100-row deep round-trip check
    rng = np.random.RandomState(0)
    sample = rng.choice(len(final_problems), size=min(ROUND_TRIP_SAMPLE, len(final_problems)),
                          replace=False)
    for i in sample:
        p = final_problems[int(i)]
        recomp = compute_labels(p["a"], p["b"])
        for k in ("a", "b", "product", "carries", "answer_digits_msf"):
            if p["labels"][k] != recomp[k]:
                raise RuntimeError(
                    f"Pass 4 round-trip failed L{p['level']} idx={p['source_index']} "
                    f"a={p['a']} b={p['b']} key={k}"
                )
    logger.info(f"Pass 4 round-trip check passed on {len(sample)} rows.")

    # Activation index check
    act_dir = Path(cfg["paths"]["activations_dir"])
    layer = cfg["model"]["layers"][0]
    for lvl in LEVELS:
        f = act_dir / f"level{lvl}_layer{layer}.npy"
        if not f.exists():
            logger.warning(f"Activation file missing: {f}")
            continue
        n_rows = np.load(f, mmap_mode="r").shape[0]
        for p in final_problems:
            if p["level"] == lvl and not (0 <= p["source_index"] < n_rows):
                raise RuntimeError(
                    f"Activation index out of range: L{lvl} src={p['source_index']} n_rows={n_rows}"
                )
        logger.debug(f"L{lvl} activation index check OK (n_rows={n_rows}).")

    return final_problems, n_duplicates


def pass5_concept_topup(final_problems, pools, cfg, logger, max_rounds=20000):
    """Top up below-floor non-documented (level, concept, value) cells by adding
    rows from the unused pool. Runs after Pass 4's assembly and dedup.

    Added rows are not part of any matched pair (matched_pair_id = None).
    They participate in coverage but not in the L4/L5 paired comparisons.
    """
    logger.info("Pass 5 start: post-assembly concept-coverage top-up.")

    by_level = defaultdict(list)
    for p in final_problems:
        by_level[p["level"]].append(p)
    selected_src = {lvl: {p["source_index"] for p in plist}
                       for lvl, plist in by_level.items()}

    # Load disk labels per level once
    disk_labels_by_lvl = {}
    for lvl in LEVELS:
        with open(Path(cfg["paths"]["labels_dir"]) / f"level_{lvl}.json") as f:
            disk_labels_by_lvl[lvl] = json.load(f)["problems"]

    next_id = max((p["curated_id"] for p in final_problems), default=-1) + 1
    n_added_total = 0

    for lvl in LEVELS:
        rows = [
            {"a": p["a"], "b": p["b"], "product": p["product"],
             "carries": p["labels"]["carries"]}
            for p in by_level[lvl]
        ]
        # Compute per-(concept, value) counts and find non-documented gaps
        gaps = []  # (concept, value, need)
        for concept in _all_concepts_for_level(lvl):
            counts = _per_value_counts(rows, concept)
            for v in _value_range_for_concept(concept, lvl):
                if (lvl, concept, v) in DOCUMENTED_GAPS:
                    continue
                cur = counts.get(v, 0)
                if cur < CONCEPT_FLOOR:
                    gaps.append((concept, v, CONCEPT_FLOOR - cur))
        if not gaps:
            logger.info(f"Pass 5 L{lvl}: no below-floor non-documented cells.")
            continue
        logger.info(
            f"Pass 5 L{lvl}: {len(gaps)} below-floor cells to fill."
        )

        df = pools[lvl]
        pool = df[~df["source_index"].isin(selected_src[lvl])].to_dict("records")
        used = set()
        added = []
        gap_remaining = {(c, v): need for c, v, need in gaps}
        round_n = 0
        while gap_remaining and round_n < max_rounds:
            round_n += 1
            best_idx = None
            best_gain = 0
            best_row = None
            for rec in pool:
                if rec["source_index"] in used:
                    continue
                gain = 0
                for concept, v in gap_remaining:
                    pv = _digit_value_for_concept(rec, concept)
                    if pv == v:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_idx = rec["source_index"]
                    best_row = rec
            if best_idx is None or best_gain == 0:
                break
            used.add(best_idx)
            added.append(best_row)
            for concept in _all_concepts_for_level(lvl):
                v = _digit_value_for_concept(best_row, concept)
                if v is None:
                    continue
                key = (concept, v)
                if gap_remaining.get(key, 0) > 0:
                    gap_remaining[key] -= 1
                    if gap_remaining[key] == 0:
                        del gap_remaining[key]

        logger.info(
            f"Pass 5 L{lvl}: added {len(added)} rows; "
            f"remaining gaps={len(gap_remaining)}."
        )

        # Append to final_problems with full metadata
        for rec in added:
            src = int(rec["source_index"])
            final_problems.append({
                "curated_id": next_id,
                "level": lvl,
                "source_index": src,
                "a": int(rec["a"]),
                "b": int(rec["b"]),
                "product": int(rec["product"]),
                "predicted": int(rec["predicted"]),
                "correct": bool(rec["correct"]),
                "raw_text": rec["raw_text"],
                "magnitude_tier": int(rec["magnitude_tier"]),
                "carry_count_tier": int(rec["carry_count_tier"]),
                "answer_length": int(rec["answer_length"]),
                "nonzero_carry_count": int(rec["nonzero_carry_count"]),
                "matched_pair_id": None,
                "matched_relaxed": None,
                "labels": disk_labels_by_lvl[lvl][src]["labels"],
            })
            next_id += 1
            n_added_total += 1

    logger.info(f"Pass 5 done: added {n_added_total} concept-topup rows total.")
    return n_added_total


# =============================================================================
# COVERAGE TABLES + REPORT
# =============================================================================

def build_coverage_tables(final_problems, logger):
    by_level = defaultdict(list)
    for p in final_problems:
        by_level[p["level"]].append(p)

    digit_coverage = {}
    carry_coverage = {}
    n_below_floor_documented = 0
    n_below_floor_undocumented = 0

    for lvl in sorted(by_level):
        rows = [
            {
                "a": p["a"],
                "b": p["b"],
                "product": p["product"],
                "carries": p["labels"]["carries"],
            }
            for p in by_level[lvl]
        ]
        for concept in DIGIT_CONCEPTS_BY_LEVEL[lvl]:
            counts = _per_value_counts(rows, concept)
            digit_coverage[(lvl, concept)] = dict(counts)
            for v in _value_range_for_concept(concept, lvl):
                if counts[v] < CONCEPT_FLOOR:
                    if (lvl, concept, v) in DOCUMENTED_GAPS:
                        n_below_floor_documented += 1
                    else:
                        n_below_floor_undocumented += 1
                        logger.warning(
                            f"COVERAGE_GAP_UNDOCUMENTED L{lvl} {concept}={v} count={counts[v]}"
                        )
        for concept in CARRY_CONCEPTS_BY_LEVEL[lvl]:
            counts = _per_value_counts(rows, concept)
            carry_coverage[(lvl, concept)] = dict(counts)
            for v in _value_range_for_concept(concept, lvl):
                if counts[v] < CONCEPT_FLOOR:
                    if (lvl, concept, v) in DOCUMENTED_GAPS:
                        n_below_floor_documented += 1
                    else:
                        n_below_floor_undocumented += 1
                        logger.warning(
                            f"COVERAGE_GAP_UNDOCUMENTED L{lvl} {concept}={v} count={counts[v]}"
                        )
    return digit_coverage, carry_coverage, n_below_floor_documented, n_below_floor_undocumented


def matched_pair_diagnostics(final_problems, logger):
    diag = {}
    by_pair = defaultdict(list)
    for p in final_problems:
        if p.get("matched_pair_id"):
            by_pair[p["matched_pair_id"]].append(p)

    for lvl in (4, 5):
        pairs = [pp for pp in by_pair.values() if pp and pp[0]["level"] == lvl]
        n_pairs = len(pairs)
        n_relaxed = sum(1 for pp in pairs if pp[0].get("matched_relaxed"))
        diffs = {"leading_digit_pair_index": [], "nonzero_carry_count": [], "answer_length": []}
        for pp in pairs:
            if len(pp) != 2:
                continue
            c = pp[0] if pp[0]["correct"] else pp[1]
            w = pp[1] if pp[0]["correct"] else pp[0]
            diffs["leading_digit_pair_index"].append(
                leading_digit_pair_index(c["a"], c["b"]) - leading_digit_pair_index(w["a"], w["b"])
            )
            diffs["nonzero_carry_count"].append(c["nonzero_carry_count"] - w["nonzero_carry_count"])
            diffs["answer_length"].append(c["answer_length"] - w["answer_length"])
        # Permutation p-values: under the null that pair labels are exchangeable
        rng = np.random.RandomState(7)
        pvals = {}
        for axis, vals in diffs.items():
            arr = np.array(vals, dtype=float)
            obs = float(np.abs(arr.mean()))
            count = 0
            for _ in range(MATCHED_PAIR_PERMUTATIONS):
                signs = rng.choice([-1, 1], size=arr.size)
                if abs((arr * signs).mean()) >= obs:
                    count += 1
            pvals[axis] = (count + 1) / (MATCHED_PAIR_PERMUTATIONS + 1)
        diag[lvl] = {
            "n_pairs": n_pairs,
            "n_relaxed": n_relaxed,
            "mean_diff": {a: float(np.mean(v)) if v else 0.0 for a, v in diffs.items()},
            "permutation_pvalues": pvals,
        }
        logger.info(
            f"L{lvl} matched pairs: n={n_pairs} relaxed={n_relaxed} "
            f"mean_diffs={diag[lvl]['mean_diff']} pvals={pvals}"
        )
    return diag


def write_coverage_report(out_path, final_problems, digit_cov, carry_cov, mp_diag,
                              metadata, logger):
    lines = []
    lines.append("# Curated set v1 — coverage report")
    lines.append("")
    lines.append(f"- Generated: {metadata['generated_at_utc']}")
    lines.append(f"- Seed: {metadata['seed']}")
    lines.append(f"- Code commit: `{metadata['code_commit_hash']}`")
    lines.append(f"- Config sha256: `{metadata['config_sha256']}`")
    lines.append(f"- Build script sha256: `{metadata['build_script_sha256']}`")
    lines.append(f"- n_problems: {metadata['n_problems']}")
    lines.append(f"- Output: `{metadata['output_path']}`")
    lines.append("")

    lines.append("## Per-level counts")
    lines.append("")
    lines.append("| Level | Correct | Wrong | Total |")
    lines.append("|------:|--------:|------:|------:|")
    for lvl in LEVELS:
        c = sum(1 for p in final_problems if p["level"] == lvl and p["correct"])
        w = sum(1 for p in final_problems if p["level"] == lvl and not p["correct"])
        lines.append(f"| L{lvl} | {c} | {w} | {c+w} |")
    lines.append("")

    lines.append("## Per-position digit coverage")
    lines.append("")
    for (lvl, concept), counts in sorted(digit_cov.items()):
        vrange = list(_value_range_for_concept(concept, lvl))
        below = [v for v in vrange if counts.get(v, 0) < CONCEPT_FLOOR]
        flag = " (gap)" if below else ""
        lines.append(f"### L{lvl} `{concept}`{flag}")
        cells = [f"{v}={counts.get(v, 0)}" for v in vrange]
        lines.append("- " + ", ".join(cells))
        if below:
            lines.append(f"- below-floor values: {below}")
        lines.append("")

    lines.append("## Per-carry coverage")
    lines.append("")
    for (lvl, concept), counts in sorted(carry_cov.items()):
        vrange = list(_value_range_for_concept(concept, lvl))
        below = [v for v in vrange if counts.get(v, 0) < CONCEPT_FLOOR]
        flag = " (gap)" if below else ""
        lines.append(f"### L{lvl} `{concept}`{flag}")
        cells = [f"{v}={counts.get(v, 0)}" for v in vrange]
        lines.append("- " + ", ".join(cells))
        if below:
            documented = [v for v in below if (lvl, concept, v) in DOCUMENTED_GAPS]
            undocumented = [v for v in below if (lvl, concept, v) not in DOCUMENTED_GAPS]
            if documented:
                lines.append(f"- documented gaps: {documented}")
            if undocumented:
                lines.append(f"- UNDOCUMENTED gaps: {undocumented}")
        lines.append("")

    lines.append("## Matched-pair diagnostics")
    lines.append("")
    for lvl, d in sorted(mp_diag.items()):
        lines.append(f"### L{lvl}")
        lines.append(f"- n_pairs: {d['n_pairs']}")
        lines.append(f"- n_relaxed (carry_count_tier ±1): {d['n_relaxed']}")
        lines.append(f"- mean correct−wrong difference per axis: {d['mean_diff']}")
        lines.append(f"- permutation p-values: {d['permutation_pvalues']}")
        lines.append("")

    lines.append("## Documented hard-ceiling gaps")
    lines.append("")
    for (lvl, concept, v), reason in sorted(DOCUMENTED_GAPS.items()):
        lines.append(f"- L{lvl} `{concept}`={v}: {reason}")
    lines.append("")

    lines.append("## Reproducibility manifest")
    lines.append("")
    lines.append(f"- Seed: {metadata['seed']}")
    lines.append(f"- numpy: {metadata['numpy_version']}")
    lines.append(f"- pandas: {metadata['pandas_version']}")
    lines.append(f"- python: {metadata['python_version']}")
    lines.append(f"- Concept registry source: {metadata['concept_registry_source']}")
    lines.append(f"- Duplicate rule: {metadata['duplicate_rule']}")
    lines.append(f"- Matching rule: {metadata['matching_rule']}")
    lines.append("")

    out_path.write_text("\n".join(lines))
    logger.info(f"Wrote coverage report to {out_path}")


# =============================================================================
# ORCHESTRATOR
# =============================================================================

def _sha256_of_path(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def _git_commit_hash(repo_root):
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unversioned"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_DATA_ROOT / "curated" / "curated_set_v1.json"),
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPO_ROOT / "docs" / "curated_set_coverage_report.md"),
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite existing output JSON"
    )
    args = parser.parse_args()

    repo_root = DEFAULT_REPO_ROOT
    logger, log_path = setup_logging(repo_root)
    logger.info("=" * 72)
    logger.info("build_curated_set start")
    logger.info("=" * 72)

    cfg = load_config(args.config)
    seed = int(cfg["dataset"]["seed"])
    rng = np.random.RandomState(seed)
    logger.info(f"Seed: {seed} (np.random.RandomState)")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.force:
        raise SystemExit(
            f"Output {output_path} already exists; use --force to overwrite."
        )

    t0 = time.time()
    pools = load_pools(cfg, logger)
    enrich_with_tiers_inplace(pools, logger)
    round_trip_compute_labels_check(pools, logger)
    derive_documented_gaps(pools, logger)

    pass1 = pass1_difficulty_stratify(pools, rng, logger)
    pass2 = pass2_concept_coverage(pools, pass1, rng, logger)

    # Combine Pass 1 + Pass 2 into the working selection
    combined = {3: pass2[3], 4: {"correct": [], "wrong": []}, 5: {"correct": [], "wrong": []}}
    for lvl in (4, 5):
        combined[lvl]["correct"] = list(set(pass1[lvl]["correct"]) | set(pass2[lvl]["correct"]))
        combined[lvl]["wrong"] = list(set(pass1[lvl]["wrong"]) | set(pass2[lvl]["wrong"]))

    matches_all = pass3_match_pairs(pools, combined, rng, logger)

    config_path = (
        Path(args.config) if args.config else Path(__file__).parent / "config.yaml"
    )
    metadata_extras = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": seed,
        "code_commit_hash": _git_commit_hash(repo_root),
        "config_path": str(config_path),
        "config_sha256": _sha256_of_path(config_path),
        "build_script": "build_curated_set.py",
        "build_script_sha256": _sha256_of_path(__file__),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "python_version": platform.python_version(),
        "concept_registry_source": "phase_g_fourier.py:56-74",
        "duplicate_rule": "exact (Hamming = 0)",
        "matching_rule": (
            "strict on (magnitude_tier, carry_count_tier, answer_length); "
            "relax carry_count_tier by ±1 on miss"
        ),
        "log_path": str(log_path),
        "output_path": str(output_path),
    }

    final_problems, n_duplicates = pass4_validate_and_trim(
        pools, pass1, pass2, matches_all, cfg, logger, metadata_extras
    )

    n_topup = pass5_concept_topup(final_problems, pools, cfg, logger)

    # Coverage tables and report (after top-up)
    digit_cov, carry_cov, n_below_doc, n_below_undoc = build_coverage_tables(
        final_problems, logger
    )
    if n_below_undoc > UNDOCUMENTED_GAP_LIMIT:
        raise RuntimeError(
            f"Undocumented gap count {n_below_undoc} exceeds limit "
            f"{UNDOCUMENTED_GAP_LIMIT}; aborting."
        )

    mp_diag = matched_pair_diagnostics(final_problems, logger)

    # Build run-log summary
    run_log_summary = {
        "pass0_pool_counts": {
            str(lvl): [int(pools[lvl]["correct"].sum()), int((~pools[lvl]["correct"]).sum())]
            for lvl in LEVELS
        },
        "pass1_drawn": {
            str(lvl): len(pass1[lvl]["correct"]) + len(pass1[lvl]["wrong"]) for lvl in (4, 5)
        },
        "pass2_drawn": {
            str(lvl): len(pass2[lvl]["correct"]) + len(pass2[lvl]["wrong"]) for lvl in LEVELS
        },
        "pass3_pairs": {
            str(lvl): {
                "strict": sum(1 for m in matches_all[lvl] if not m["relaxed"]),
                "relaxed": sum(1 for m in matches_all[lvl] if m["relaxed"]),
                "unmatched_dropped": (
                    BUDGET[lvl]["correct"] - len(matches_all[lvl])
                ),
            }
            for lvl in (4, 5)
        },
        "pass4_duplicates_dropped": int(n_duplicates),
        "pass5_topup_added": int(n_topup),
        "pass5_below_floor_documented": int(n_below_doc),
        "pass5_below_floor_undocumented": int(n_below_undoc),
        "runtime_seconds": float(time.time() - t0),
    }

    metadata = {
        "schema_version": "v1",
        **metadata_extras,
        "n_problems": len(final_problems),
        "per_level_counts": {
            str(lvl): {
                "correct": sum(1 for p in final_problems if p["level"] == lvl and p["correct"]),
                "wrong": sum(1 for p in final_problems if p["level"] == lvl and not p["correct"]),
            }
            for lvl in LEVELS
        },
        "n_matched_pairs": {str(lvl): len(matches_all[lvl]) for lvl in (4, 5)},
        "tier_definitions": {
            "magnitude_tier": "small=1-3, medium=4-6, large=7-9; joint=3*tier(a)+tier(b)",
            "carry_count_tier": "low=<=1, medium=2-3, high=>=4 nonzero carries",
            "answer_length": "len(str(a*b))",
        },
        "run_log_summary": run_log_summary,
    }

    output = {
        "metadata": metadata,
        "problems": final_problems,
        "coverage_summary": {
            "digit_coverage": {
                f"L{lvl}/{c}": v for (lvl, c), v in digit_cov.items()
            },
            "carry_coverage": {
                f"L{lvl}/{c}": v for (lvl, c), v in carry_cov.items()
            },
            "matched_pair_diagnostics": {str(k): v for k, v in mp_diag.items()},
            "documented_gaps": {
                f"L{lvl}/{c}={v}": reason
                for (lvl, c, v), reason in DOCUMENTED_GAPS.items()
            },
        },
    }

    output_path.write_text(json.dumps(output, indent=2, default=str))
    out_sha = _sha256_of_path(output_path)
    logger.info(f"Wrote {output_path} sha256={out_sha}")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_coverage_report(report_path, final_problems, digit_cov, carry_cov, mp_diag,
                            metadata, logger)
    rep_sha = _sha256_of_path(report_path)
    logger.info(f"Wrote {report_path} sha256={rep_sha}")

    logger.info(
        f"Build complete: n_problems={len(final_problems)} "
        f"n_matched_pairs={metadata['n_matched_pairs']} "
        f"runtime={time.time()-t0:.1f}s"
    )


if __name__ == "__main__":
    main()
