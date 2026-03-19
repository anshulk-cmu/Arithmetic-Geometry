#!/usr/bin/env python3
"""Phase B — Concept Deconfounding: label-level correlation diagnostics.

Measures how tangled the concept labels are with each other, classifies each
correlation as structural / sampling-induced / residualization-induced, and
decides whether Phase C's single-confound removal (product magnitude only) is
sufficient or whether additional deconfounding is needed.

Operates entirely on labels (the coloring DataFrame + concept registry).
No activations, no GPU.  Runs in under a minute on CPU.

Produces:
  - Correlation matrices (raw + post-product-residualization) per level
  - Classified pair list with action recommendations
  - Deconfounding plan JSON consumed by Phase C
  - Heatmap plots of both matrices
  - Spearman follow-up for top pairs
  - Summary JSON with aggregate statistics

Usage:
  python phase_b_deconfounding.py --config config.yaml          # Full run
  python phase_b_deconfounding.py --config config.yaml --level 3  # Single level
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
import pandas as pd
import yaml
from scipy.stats import spearmanr


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

LEVELS = [1, 2, 3, 4, 5]
POPULATIONS = ["all", "correct", "wrong"]
MIN_POPULATION = 30

# Thresholds
R_REPORT_THRESHOLD = 0.1     # include in classified_pairs output
R_ACTION_THRESHOLD = 0.3     # triggers classification decision
SPEARMAN_TOP_K = 30          # Spearman follow-up for top K pairs by |r_resid|

# Preprocessing constants (must match phase_c_subspaces.py exactly)
PP_N_BINS = 9
PRODUCT_N_BINS = 10
MIN_GROUP_SIZE = 20

# Structural relationship registry: pairs linked by arithmetic, keyed by
# (concept_prefix_a, concept_prefix_b).  Used for classification.
# "same_column" means they share a column index (carry_k <-> col_sum_k).
# "carry_chain" means adjacent carries.
# "column_contributor" means a partial product feeds into a column sum.
# "digit_to_carry0" means an input digit determines carry_0 via column 0.
# "derived_aggregate" means one is an aggregate of the other family.


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
        "coloring_dir": dr / "phase_a" / "coloring_dfs",
        "phase_b_data": dr / "phase_b",
        "correlation_dir": dr / "phase_b" / "correlation_matrices",
        "phase_b_plots": ws / "plots" / "phase_b",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(workspace):
    log_dir = workspace / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase_b")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fh = RotatingFileHandler(log_dir / "phase_b_deconfounding.log",
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
# CONCEPT REGISTRY  (duplicated from phase_c_subspaces.py to avoid import
# issues with matplotlib/scipy on headless nodes)
# ═══════════════════════════════════════════════════════════════════════════════

def bin_partial_product(values, n_bins=PP_N_BINS):
    bins = np.linspace(-0.5, 81.5, n_bins + 1)
    return np.digitize(values, bins) - 1


def bin_product_deciles(values, n_bins=PRODUCT_N_BINS):
    try:
        binned = pd.qcut(pd.Series(values), q=n_bins, labels=False,
                         duplicates="drop")
    except ValueError:
        binned = pd.cut(pd.Series(values), bins=n_bins, labels=False)
    return np.asarray(binned, dtype=np.float64)


def preprocess_concept(values, preprocess_type):
    if preprocess_type is None:
        return np.asarray(values, dtype=np.float64)
    elif preprocess_type == "filter_min_group":
        return np.asarray(values, dtype=np.float64)
    elif preprocess_type == "bin_9":
        return bin_partial_product(
            np.asarray(values, dtype=np.float64)).astype(np.float64)
    elif preprocess_type == "bin_deciles":
        return bin_product_deciles(values).astype(np.float64)
    else:
        raise ValueError(f"Unknown preprocess type: {preprocess_type}")


def filter_concept_values(values, min_size=MIN_GROUP_SIZE):
    """Set rare values to NaN.  Returns None if <2 groups survive."""
    unique_vals, counts = np.unique(values[~np.isnan(values)],
                                    return_counts=True)
    keep_mask = counts >= min_size
    dropped_vals = set(unique_vals[~keep_mask])

    filtered = values.copy()
    for v in dropped_vals:
        filtered[filtered == v] = np.nan

    surviving = np.unique(filtered[~np.isnan(filtered)])
    if len(surviving) < 2:
        return None
    return filtered


def get_concept_registry(level, df, pop_name="all"):
    """Return list of concept dicts for this level/population.

    Each dict: {"name": str, "column": str, "tier": int,
                "preprocess": str|None}

    Mirrors phase_c_subspaces.get_concept_registry exactly.
    """
    cols = set(df.columns)
    concepts = []

    # Tier 1: Input digits
    digit_cols = ["a_units", "a_tens", "a_hundreds",
                  "b_units", "b_tens", "b_hundreds"]
    for col in digit_cols:
        if col in cols:
            concepts.append({"name": col, "column": col,
                             "tier": 1, "preprocess": None})

    # Tier 1: Answer digits
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

    # Tier 3: Per-digit correctness
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
# STRUCTURAL RELATIONSHIP CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_column_index(name):
    """Extract trailing integer from concept name, e.g. carry_2 -> 2."""
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    return None


def _partial_product_columns(pp_name):
    """Return the column index a partial product contributes to.

    pp_a{i}_x_b{j} contributes to column i+j.
    """
    # pp_a0_x_b1 -> i=0, j=1 -> column 1
    parts = pp_name.replace("pp_a", "").replace("_x_b", " ").split()
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return int(parts[0]) + int(parts[1])
    return None


def _get_input_digit_place(name):
    """Return (operand, place_index) for input digits.

    a_units -> ('a', 0), a_tens -> ('a', 1), a_hundreds -> ('a', 2)
    b_units -> ('b', 0), b_tens -> ('b', 1), b_hundreds -> ('b', 2)
    """
    place_map = {"units": 0, "tens": 1, "hundreds": 2}
    for prefix in ("a_", "b_"):
        if name.startswith(prefix):
            place = name[len(prefix):]
            if place in place_map:
                return (prefix[0], place_map[place])
    return None


def _get_leading_place(level):
    """Return the place index of the leading (most significant) digit.

    L1: 1-digit × 1-digit -> units (index 0) is the only/leading digit
    L2: 2-digit × 1-digit -> tens (index 1) for a, units (index 0) for b
    L3: 2-digit × 2-digit -> tens (index 1) for both
    L4: 3-digit × 2-digit -> hundreds (index 2) for a, tens (index 1) for b
    L5: 3-digit × 3-digit -> hundreds (index 2) for both
    """
    if level == 1:
        return {"a": 0, "b": 0}
    elif level == 2:
        return {"a": 1, "b": 0}
    elif level == 3:
        return {"a": 1, "b": 1}
    elif level == 4:
        return {"a": 2, "b": 1}
    else:
        return {"a": 2, "b": 2}


def classify_pair(name_a, name_b, level):
    """Classify the structural relationship between two concepts.

    Returns one of:
      "structural"           - forced by multiplication arithmetic
      "residualization"      - created by product residualization (leading digits)
      "sampling"             - potentially induced by carry-stratified sampling
      "none"                 - no known structural link
    """
    # Sort for canonical order
    a, b = sorted([name_a, name_b])

    # ── Same concept family checks ──────────────────────────────────────

    # carry_k <-> col_sum_k  (deterministic: carry_k = floor((col_sum_k + carry_{k-1}) / 10))
    if a.startswith("carry_") and b.startswith("col_sum_"):
        idx_a = _parse_column_index(a)
        idx_b = _parse_column_index(b)
        if idx_a is not None and idx_b is not None and idx_a == idx_b:
            return "structural"

    # Adjacent carries: carry_k <-> carry_{k+1}  (carry chain)
    if a.startswith("carry_") and b.startswith("carry_"):
        idx_a = _parse_column_index(a)
        idx_b = _parse_column_index(b)
        if idx_a is not None and idx_b is not None:
            return "structural"

    # col_sum_k <-> col_sum_j  (linked through carry chain)
    if a.startswith("col_sum_") and b.startswith("col_sum_"):
        return "structural"

    # carry_k <-> col_sum_j where j != k (linked through carry chain)
    if (a.startswith("carry_") and b.startswith("col_sum_")) or \
       (a.startswith("col_sum_") and b.startswith("carry_")):
        return "structural"

    # Partial product <-> column sum it contributes to
    pp, cs = None, None
    if a.startswith("pp_") and b.startswith("col_sum_"):
        pp, cs = a, b
    elif a.startswith("col_sum_") and b.startswith("pp_"):
        pp, cs = b, a
    if pp is not None and cs is not None:
        pp_col = _partial_product_columns(pp)
        cs_col = _parse_column_index(cs)
        if pp_col is not None and cs_col is not None:
            # pp contributes to col_sum at its column, but also indirectly
            # to later columns through the carry chain
            return "structural"

    # Partial product <-> carry (indirect through column sum -> carry)
    if (a.startswith("carry_") and b.startswith("pp_")) or \
       (a.startswith("pp_") and b.startswith("carry_")):
        return "structural"

    # Input digit <-> partial product it participates in
    # a_units (place 0) participates in pp_a0_x_b*
    # a_tens  (place 1) participates in pp_a1_x_b*
    digit_info_a = _get_input_digit_place(a)
    digit_info_b = _get_input_digit_place(b)
    if digit_info_a is not None and b.startswith("pp_"):
        operand, place = digit_info_a
        if f"pp_{operand[0]}{place}" in b.replace("_x_", f"{place}_x_") or \
           f"a{place}" in b or f"b{place}" in b:
            # More precise check: digit a_i participates in pp_a{i}_x_b{*}
            parts = b.replace("pp_a", "").replace("_x_b", " ").split()
            if len(parts) == 2:
                if (operand == "a" and parts[0] == str(place)) or \
                   (operand == "b" and parts[1] == str(place)):
                    return "structural"
    if digit_info_b is not None and a.startswith("pp_"):
        operand, place = digit_info_b
        parts = a.replace("pp_a", "").replace("_x_b", " ").split()
        if len(parts) == 2:
            if (operand == "a" and parts[0] == str(place)) or \
               (operand == "b" and parts[1] == str(place)):
                return "structural"

    # Input digit <-> carry/col_sum (structural: digits determine column sums
    # which determine carries)
    if digit_info_a is not None and (b.startswith("carry_") or
                                      b.startswith("col_sum_")):
        return "structural"
    if digit_info_b is not None and (a.startswith("carry_") or
                                      a.startswith("col_sum_")):
        return "structural"

    # Input digit <-> partial product it does NOT directly participate in
    # (still structural: e.g. a_units correlates with pp_a0_x_b1 through b_units)
    if (digit_info_a is not None and b.startswith("pp_")) or \
       (digit_info_b is not None and a.startswith("pp_")):
        return "structural"

    # Partial product <-> partial product (all pp's are sub-terms of product,
    # share input digits as factors, and become correlated/anti-correlated
    # after product residualization through the suppression effect)
    if a.startswith("pp_") and b.startswith("pp_"):
        return "structural"

    # Answer digits <-> carries/col_sums (answer digits are determined by carries)
    if (a.startswith("ans_digit_") and
        (b.startswith("carry_") or b.startswith("col_sum_") or
         b.startswith("pp_"))):
        return "structural"
    if (b.startswith("ans_digit_") and
        (a.startswith("carry_") or a.startswith("col_sum_") or
         a.startswith("pp_"))):
        return "structural"

    # Answer digits <-> input digits (structural: product = a × b)
    if (a.startswith("ans_digit_") and digit_info_b is not None) or \
       (b.startswith("ans_digit_") and digit_info_a is not None):
        return "structural"

    # Answer digits <-> other answer digits
    if a.startswith("ans_digit_") and b.startswith("ans_digit_"):
        return "structural"

    # Derived aggregates <-> their components
    aggregates = {"n_nonzero_carries", "total_carry_sum", "max_carry_value",
                  "n_answer_digits"}
    if a in aggregates or b in aggregates:
        other = b if a in aggregates else a
        if (other.startswith("carry_") or other.startswith("col_sum_") or
            other.startswith("ans_digit_") or other.startswith("pp_") or
            _get_input_digit_place(other) is not None or
            other in aggregates):
            return "structural"

    # product_binned <-> everything (product = a × b, correlated with all)
    if a == "product_binned" or b == "product_binned":
        return "structural"

    # correct <-> anything (correctness correlates with difficulty)
    if a == "correct" or b == "correct":
        return "structural"

    # digit_correct_pos* <-> anything
    if a.startswith("digit_correct_") or b.startswith("digit_correct_"):
        return "structural"

    # ── Residualization-induced check ───────────────────────────────────
    # Product residualization creates anti-correlation between leading digits
    # of a and b at the same place value (e.g., a_tens <-> b_tens at L3,
    # a_hundreds <-> b_hundreds at L5).
    # This happens because product ≈ (leading_a × leading_b × 10^k), so
    # removing product constrains leading_a × leading_b ≈ const.
    if digit_info_a is not None and digit_info_b is not None:
        op_a, place_a = digit_info_a
        op_b, place_b = digit_info_b
        if op_a != op_b:
            # Cross-operand digit pair
            leading = _get_leading_place(level)
            if place_a == leading[op_a] and place_b == leading[op_b]:
                return "residualization"

    # ── Sampling-induced check ──────────────────────────────────────────
    # At L5, carry-stratified sampling can induce correlations between
    # input digits that are independent under uniform sampling.
    # Flag cross-operand digit pairs at L5 that aren't leading digits.
    if level == 5 and digit_info_a is not None and digit_info_b is not None:
        op_a, _ = digit_info_a
        op_b, _ = digit_info_b
        if op_a != op_b:
            return "sampling"

    return "none"


# ═══════════════════════════════════════════════════════════════════════════════
# CORE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def build_concept_matrix(df, concepts):
    """Build preprocessed concept value matrix from coloring DataFrame.

    Returns:
        names: list of concept names
        values: list of numpy arrays (one per concept, with NaN for filtered)
        tiers: list of tier ints
    """
    names = []
    values = []
    tiers = []

    for concept in concepts:
        c_name = concept["name"]
        c_col = concept["column"]
        c_pre = concept["preprocess"]

        raw = df[c_col].values
        preprocessed = preprocess_concept(raw, c_pre)

        # Apply filter_min_group where needed
        if c_pre == "filter_min_group":
            filtered = filter_concept_values(preprocessed)
            if filtered is None:
                continue
            preprocessed = filtered

        names.append(c_name)
        values.append(preprocessed)
        tiers.append(concept["tier"])

    return names, values, tiers


def pairwise_pearson(values_list):
    """Compute pairwise Pearson correlation matrix.

    Handles NaN by computing each pair over rows where both are valid.

    Args:
        values_list: list of C numpy arrays, each of length N

    Returns:
        R: (C, C) correlation matrix
        N_valid: (C, C) count of valid pairs used for each correlation
    """
    C = len(values_list)
    R = np.eye(C)
    N_valid = np.zeros((C, C), dtype=np.int64)

    for i in range(C):
        vi = values_list[i]
        for j in range(i + 1, C):
            vj = values_list[j]
            mask = ~(np.isnan(vi) | np.isnan(vj))
            n = mask.sum()
            N_valid[i, j] = n
            N_valid[j, i] = n

            if n < 10:
                R[i, j] = np.nan
                R[j, i] = np.nan
                continue

            vi_m = vi[mask]
            vj_m = vj[mask]
            vi_c = vi_m - vi_m.mean()
            vj_c = vj_m - vj_m.mean()

            denom = np.sqrt((vi_c @ vi_c) * (vj_c @ vj_c))
            if denom < 1e-12:
                R[i, j] = 0.0
                R[j, i] = 0.0
            else:
                r = (vi_c @ vj_c) / denom
                R[i, j] = r
                R[j, i] = r

    np.fill_diagonal(N_valid, np.array([
        (~np.isnan(v)).sum() for v in values_list], dtype=np.int64))

    return R, N_valid


def residualize_product_labels(values_list, product_values):
    """Remove linear effect of product from each concept's label values.

    This is label-level residualization (not activation-level).
    It mirrors Phase C's product residualization but on labels, to produce
    R_resid: the correlation structure that remains after Phase C's product
    removal step.

    Args:
        values_list: list of C numpy arrays
        product_values: (N,) raw product values

    Returns:
        resid_list: list of C residualized numpy arrays
    """
    p = np.asarray(product_values, dtype=np.float64)

    resid_list = []
    for v in values_list:
        v_c = v.copy()
        valid = ~np.isnan(v_c)
        mean_v = np.nanmean(v_c)
        v_c[valid] = v_c[valid] - mean_v

        # OLS on valid entries only: both numerator and denominator must
        # use the same row subset to get a correct beta estimate.
        # Product is never NaN, so the mask is just the concept's validity.
        p_valid = p[valid]
        p_valid_c = p_valid - p_valid.mean()
        p_dot_p = p_valid_c @ p_valid_c

        if p_dot_p < 1e-12:
            resid_list.append(v_c)
            continue

        beta = (v_c[valid] @ p_valid_c) / p_dot_p
        v_r = v_c.copy()
        v_r[valid] = v_c[valid] - beta * p_valid_c
        resid_list.append(v_r)

    return resid_list


def compute_spearman_top_k(values_list, names, R_resid, k=SPEARMAN_TOP_K):
    """Compute Spearman correlation for the top-k pairs by |r_resid|.

    Returns list of dicts with Spearman rho for comparison.
    """
    C = len(names)
    pairs = []
    for i in range(C):
        for j in range(i + 1, C):
            r = R_resid[i, j]
            if not np.isnan(r):
                pairs.append((abs(r), i, j, r))

    pairs.sort(reverse=True)
    results = []

    for _, i, j, r_pearson in pairs[:k]:
        vi = values_list[i]
        vj = values_list[j]
        mask = ~(np.isnan(vi) | np.isnan(vj))
        n = mask.sum()
        if n < 10:
            rho = np.nan
        else:
            rho, _ = spearmanr(vi[mask], vj[mask])

        results.append({
            "concept_a": names[i],
            "concept_b": names[j],
            "r_pearson_resid": float(r_pearson),
            "rho_spearman_resid": float(rho) if not np.isnan(rho) else None,
            "n_valid": int(n),
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION AND DECONFOUNDING PLAN
# ═══════════════════════════════════════════════════════════════════════════════

def classify_all_pairs(names, tiers, R_raw, R_resid, level):
    """Classify all concept pairs and determine actions.

    Returns list of dicts, one per pair with |r_raw| or |r_resid| > threshold.
    """
    C = len(names)
    rows = []

    for i in range(C):
        for j in range(i + 1, C):
            r_raw = R_raw[i, j]
            r_resid = R_resid[i, j]

            if np.isnan(r_raw) and np.isnan(r_resid):
                continue

            max_abs = max(abs(r_raw) if not np.isnan(r_raw) else 0,
                         abs(r_resid) if not np.isnan(r_resid) else 0)

            if max_abs < R_REPORT_THRESHOLD:
                continue

            classification = classify_pair(names[i], names[j], level)

            # Determine action based on classification and magnitude
            abs_resid = abs(r_resid) if not np.isnan(r_resid) else 0
            if abs_resid < R_ACTION_THRESHOLD:
                action = "accept"
            elif classification == "structural":
                action = "accept"
            elif classification == "residualization":
                action = "flag_use_raw"
            elif classification == "sampling":
                action = "deconfound"
            elif classification == "none":
                action = "investigate"
            else:
                action = "investigate"

            rows.append({
                "concept_a": names[i],
                "concept_b": names[j],
                "tier_a": tiers[i],
                "tier_b": tiers[j],
                "r_raw": float(r_raw) if not np.isnan(r_raw) else None,
                "r_resid": float(r_resid) if not np.isnan(r_resid) else None,
                "classification": classification,
                "action": action,
            })

    rows.sort(key=lambda r: -abs(r["r_resid"] or 0))
    return rows


def build_deconfounding_plan(classified_pairs, names):
    """Build per-concept confound lists from classified pairs.

    For each concept, lists additional confounds (beyond product) that should
    be regressed out before computing its subspace in Phase C.

    Only pairs with action="deconfound" contribute.
    Pairs with action="flag_use_raw" are noted separately.
    """
    confounds = {name: [] for name in names}
    use_raw = set()

    for pair in classified_pairs:
        if pair["action"] == "deconfound":
            ca, cb = pair["concept_a"], pair["concept_b"]
            if ca in confounds and cb not in confounds[ca]:
                confounds[ca].append(cb)
            if cb in confounds and ca not in confounds[cb]:
                confounds[cb].append(ca)
        elif pair["action"] == "flag_use_raw":
            use_raw.add(pair["concept_a"])
            use_raw.add(pair["concept_b"])

    # product_binned already uses raw activations in Phase C
    use_raw.discard("product_binned")

    # Remove concepts with empty confound lists
    confounds = {k: sorted(v) for k, v in confounds.items() if v}

    return {
        "confounds": confounds,
        "use_raw_activations": sorted(use_raw),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_correlation_heatmap(R, names, title, save_path, vmin=-1, vmax=1):
    """Plot a correlation matrix as an annotated heatmap."""
    C = len(names)
    figsize = max(8, 0.35 * C)
    fig, ax = plt.subplots(1, 1, figsize=(figsize, figsize))

    im = ax.imshow(R, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                   interpolation="nearest")

    ax.set_xticks(range(C))
    ax.set_xticklabels(names, rotation=90, fontsize=max(4, 9 - C // 10))
    ax.set_yticks(range(C))
    ax.set_yticklabels(names, fontsize=max(4, 9 - C // 10))
    ax.set_title(title, fontsize=12)

    # Annotate cells for small matrices
    if C <= 30:
        for i in range(C):
            for j in range(C):
                val = R[i, j]
                if np.isnan(val):
                    continue
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=max(3, 7 - C // 8), color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_resid_delta(R_raw, R_resid, names, title, save_path):
    """Plot the change in correlation due to product residualization."""
    delta = R_resid - R_raw
    # NaN where either is NaN
    delta[np.isnan(R_raw) | np.isnan(R_resid)] = np.nan

    max_abs = np.nanmax(np.abs(delta))
    if max_abs < 0.01:
        max_abs = 1.0  # avoid degenerate colorbar

    plot_correlation_heatmap(delta, names,
                             title, save_path,
                             vmin=-max_abs, vmax=max_abs)


# ═══════════════════════════════════════════════════════════════════════════════
# PER-LEVEL RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def process_level(level, paths, logger):
    """Run Phase B analysis for one level.  Returns result dict."""
    logger.info(f"Processing L{level}...")
    t0 = time.time()

    df = load_coloring_df(level, paths["coloring_dir"])
    populations = get_populations(df)
    product_values = df["product"].values.astype(np.float64)

    level_results = {
        "level": level,
        "n_problems": len(df),
        "populations": {},
    }

    for pop_name, pop_df in populations.items():
        logger.info(f"  Population '{pop_name}': {len(pop_df)} problems")

        concepts = get_concept_registry(level, pop_df, pop_name)

        # Exclude product_binned from correlation analysis:
        # it uses raw activations in Phase C, so product residualization
        # doesn't apply to it.
        concepts = [c for c in concepts if c["name"] != "product_binned"]

        names, values, tiers = build_concept_matrix(pop_df, concepts)
        C = len(names)
        logger.info(f"    {C} concepts after preprocessing")

        if C < 2:
            logger.warning(f"    Skipping: fewer than 2 viable concepts")
            continue

        pop_product = product_values[pop_df.index.values]

        # Step 1: Raw correlation matrix
        R_raw, N_valid = pairwise_pearson(values)
        logger.info(f"    R_raw computed ({C}x{C})")

        # Step 2: Post-product-residualization correlation matrix
        resid_values = residualize_product_labels(values, pop_product)
        R_resid, _ = pairwise_pearson(resid_values)
        logger.info(f"    R_resid computed")

        # Step 3: Classify pairs
        classified = classify_all_pairs(names, tiers, R_raw, R_resid, level)
        n_above_action = sum(1 for p in classified
                             if abs(p.get("r_resid") or 0) > R_ACTION_THRESHOLD)
        n_structural = sum(1 for p in classified
                           if p["classification"] == "structural"
                           and abs(p.get("r_resid") or 0) > R_ACTION_THRESHOLD)
        n_resid_induced = sum(1 for p in classified
                              if p["classification"] == "residualization"
                              and abs(p.get("r_resid") or 0) > R_ACTION_THRESHOLD)
        n_sampling = sum(1 for p in classified
                         if p["classification"] == "sampling"
                         and abs(p.get("r_resid") or 0) > R_ACTION_THRESHOLD)
        n_unexplained = sum(1 for p in classified
                            if p["classification"] == "none"
                            and abs(p.get("r_resid") or 0) > R_ACTION_THRESHOLD)
        n_deconfound = sum(1 for p in classified
                           if p["action"] == "deconfound")
        n_flag_raw = sum(1 for p in classified
                         if p["action"] == "flag_use_raw")

        logger.info(f"    Pairs |r_resid| > {R_ACTION_THRESHOLD}: {n_above_action} "
                     f"(structural={n_structural}, resid_induced={n_resid_induced}, "
                     f"sampling={n_sampling}, unexplained={n_unexplained})")
        logger.info(f"    Actions: deconfound={n_deconfound}, "
                     f"flag_use_raw={n_flag_raw}")

        # Step 4: Spearman follow-up on top pairs
        spearman_results = compute_spearman_top_k(
            resid_values, names, R_resid)

        # Step 5: Deconfounding plan
        plan = build_deconfounding_plan(classified, names)

        # ── Save outputs ────────────────────────────────────────────────

        corr_dir = paths["correlation_dir"]
        corr_dir.mkdir(parents=True, exist_ok=True)

        # Correlation matrices as CSV
        r_raw_df = pd.DataFrame(R_raw, index=names, columns=names)
        r_raw_df.to_csv(corr_dir / f"L{level}_{pop_name}_raw.csv")

        r_resid_df = pd.DataFrame(R_resid, index=names, columns=names)
        r_resid_df.to_csv(corr_dir / f"L{level}_{pop_name}_residualized.csv")

        # Heatmap plots
        plot_dir = paths["phase_b_plots"]
        plot_correlation_heatmap(
            R_raw, names,
            f"Raw Label Correlation — L{level} {pop_name} (N={len(pop_df)})",
            plot_dir / f"heatmap_L{level}_{pop_name}_raw.png")
        plot_correlation_heatmap(
            R_resid, names,
            f"Post-Product-Residualization — L{level} {pop_name} (N={len(pop_df)})",
            plot_dir / f"heatmap_L{level}_{pop_name}_residualized.png")
        plot_resid_delta(
            R_raw, R_resid, names,
            f"Residualization Delta (R_resid - R_raw) — L{level} {pop_name}",
            plot_dir / f"heatmap_L{level}_{pop_name}_delta.png")

        # Store for summary
        level_results["populations"][pop_name] = {
            "n_problems": len(pop_df),
            "n_concepts": C,
            "concept_names": names,
            "n_pairs_total": C * (C - 1) // 2,
            "n_pairs_above_action_threshold": n_above_action,
            "n_structural": n_structural,
            "n_residualization_induced": n_resid_induced,
            "n_sampling_induced": n_sampling,
            "n_unexplained": n_unexplained,
            "n_deconfound": n_deconfound,
            "n_flag_use_raw": n_flag_raw,
            "classified_pairs": classified,
            "spearman_top_k": spearman_results,
            "deconfounding_plan": plan,
        }

    elapsed = time.time() - t0
    logger.info(f"  L{level} complete ({elapsed:.1f}s)")
    return level_results


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def generate_summary(all_results, paths, logger):
    """Generate summary files from all level results."""
    data_dir = paths["phase_b_data"]
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Classified pairs CSV (all levels, all populations)
    all_pairs = []
    for result in all_results:
        level = result["level"]
        for pop_name, pop_data in result["populations"].items():
            for pair in pop_data["classified_pairs"]:
                row = {"level": level, "population": pop_name}
                row.update(pair)
                all_pairs.append(row)

    if all_pairs:
        pairs_df = pd.DataFrame(all_pairs)
        pairs_df.to_csv(data_dir / "classified_pairs.csv", index=False)
        logger.info(f"Saved classified_pairs.csv ({len(pairs_df)} rows)")

    # 2. Spearman comparison CSV
    all_spearman = []
    for result in all_results:
        level = result["level"]
        for pop_name, pop_data in result["populations"].items():
            for entry in pop_data["spearman_top_k"]:
                row = {"level": level, "population": pop_name}
                row.update(entry)
                all_spearman.append(row)

    if all_spearman:
        sp_df = pd.DataFrame(all_spearman)
        sp_df.to_csv(data_dir / "spearman_comparison.csv", index=False)
        logger.info(f"Saved spearman_comparison.csv ({len(sp_df)} rows)")

    # 3. Deconfounding plan JSON (merged across populations — use "all" population)
    merged_plan = {"per_level": {}}
    for result in all_results:
        level = result["level"]
        pop_data = result["populations"].get("all")
        if pop_data is None:
            continue
        plan = pop_data["deconfounding_plan"]
        merged_plan["per_level"][str(level)] = plan

    # Top-level flags
    # Decision is based on the "all" population, which Phase C primarily uses.
    # Correct/wrong sub-populations are noted in the full output but don't
    # drive the top-level decision.
    any_deconfound = any(
        bool(r["populations"].get("all", {})
             .get("deconfounding_plan", {}).get("confounds"))
        for r in all_results
    )
    any_use_raw = any(
        bool(r["populations"].get("all", {})
             .get("deconfounding_plan", {}).get("use_raw_activations"))
        for r in all_results
    )
    merged_plan["needs_multi_concept_residualization"] = any_deconfound
    merged_plan["needs_raw_activation_override"] = any_use_raw

    with open(data_dir / "deconfounding_plan.json", "w") as f:
        json.dump(merged_plan, f, indent=2)
    logger.info(f"Saved deconfounding_plan.json")

    # 4. Summary JSON
    summary = {"levels": {}}
    for result in all_results:
        level = result["level"]
        level_summary = {
            "n_problems": result["n_problems"],
            "populations": {},
        }
        for pop_name, pop_data in result["populations"].items():
            level_summary["populations"][pop_name] = {
                "n_problems": pop_data["n_problems"],
                "n_concepts": pop_data["n_concepts"],
                "n_pairs_total": pop_data["n_pairs_total"],
                "n_pairs_above_action_threshold": pop_data["n_pairs_above_action_threshold"],
                "breakdown": {
                    "structural": pop_data["n_structural"],
                    "residualization_induced": pop_data["n_residualization_induced"],
                    "sampling_induced": pop_data["n_sampling_induced"],
                    "unexplained": pop_data["n_unexplained"],
                },
                "actions": {
                    "deconfound": pop_data["n_deconfound"],
                    "flag_use_raw": pop_data["n_flag_use_raw"],
                },
            }
        summary["levels"][str(level)] = level_summary

    summary["decision"] = (
        "multi_concept_residualization" if any_deconfound
        else "raw_activation_override_only" if any_use_raw
        else "product_residualization_sufficient"
    )

    with open(data_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary.json")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI & MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase B — Concept Deconfounding: label-level "
                    "correlation diagnostics")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--level", type=int, nargs="*",
                        help="Specific levels to run (default: all)")
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    cfg = load_config(args.config)
    paths = derive_paths(cfg)
    logger = setup_logging(paths["workspace"])

    # Banner
    logger.info("=" * 70)
    logger.info("Phase B — Concept Deconfounding")
    logger.info("=" * 70)

    levels = args.level if args.level else LEVELS
    logger.info(f"Levels: {levels}")
    logger.info(f"Report threshold: |r| > {R_REPORT_THRESHOLD}")
    logger.info(f"Action threshold: |r| > {R_ACTION_THRESHOLD}")
    logger.info(f"Spearman top-k: {SPEARMAN_TOP_K}")

    # Pre-flight: check coloring DFs exist
    missing = []
    for level in levels:
        pkl = paths["coloring_dir"] / f"L{level}_coloring.pkl"
        if not pkl.exists():
            missing.append(str(pkl))
    if missing:
        logger.error(f"Missing {len(missing)} coloring DataFrames:")
        for p in missing:
            logger.error(f"  {p}")
        logger.error("Run Phase A first.")
        return

    # Create output directories
    for key in ["phase_b_data", "correlation_dir", "phase_b_plots"]:
        paths[key].mkdir(parents=True, exist_ok=True)

    # ── Process each level ─────────────────────────────────────────────
    all_results = []
    for level in levels:
        result = process_level(level, paths, logger)
        all_results.append(result)

    # ── Summary ────────────────────────────────────────────────────────
    logger.info("")
    logger.info("Generating summary outputs...")
    generate_summary(all_results, paths, logger)

    # ── Final report ───────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"Phase B complete: {len(levels)} levels, "
                f"{elapsed:.1f}s ({elapsed / 60:.1f} min)")
    logger.info("=" * 70)

    # Print decision
    # Decision is based on the "all" population, which Phase C primarily uses.
    # Correct/wrong sub-populations are noted in the full output but don't
    # drive the top-level decision.
    any_deconfound = any(
        bool(r["populations"].get("all", {})
             .get("deconfounding_plan", {}).get("confounds"))
        for r in all_results
    )
    any_use_raw = any(
        bool(r["populations"].get("all", {})
             .get("deconfounding_plan", {}).get("use_raw_activations"))
        for r in all_results
    )

    if any_deconfound:
        logger.info("DECISION: Multi-concept residualization needed.")
        logger.info("  See deconfounding_plan.json for per-concept confound lists.")
    elif any_use_raw:
        logger.info("DECISION: Product residualization sufficient, but some concepts")
        logger.info("  should use raw activations to avoid residualization artifacts.")
        logger.info("  See deconfounding_plan.json for details.")
    else:
        logger.info("DECISION: Product residualization sufficient. No additional "
                     "deconfounding needed.")

    # Log top pairs per level
    for result in all_results:
        level = result["level"]
        pop_data = result["populations"].get("all")
        if pop_data is None:
            continue
        pairs = pop_data["classified_pairs"]
        above = [p for p in pairs
                 if abs(p.get("r_resid") or 0) > R_ACTION_THRESHOLD
                 and p["action"] != "accept"]
        if above:
            logger.info(f"  L{level} actionable pairs:")
            for p in above[:10]:
                logger.info(f"    {p['concept_a']:25s} vs {p['concept_b']:25s} "
                             f"r_resid={p['r_resid']:+.3f}  "
                             f"class={p['classification']:15s}  "
                             f"action={p['action']}")


if __name__ == "__main__":
    main()
