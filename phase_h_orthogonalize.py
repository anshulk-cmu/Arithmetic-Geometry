#!/usr/bin/env python3
"""
Phase H / B.2: Orthogonalization control for carry helix superposition.

For each carry helix discovered by Phase G, project activations away from the
row space spanned by algebraically-related concept subspaces, then re-run Phase
G's Fourier conjunction test on the raw and orthogonalized projections.
"""

import argparse
import json
import logging
import math
import time
from collections import defaultdict
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import pandas as pd

from phase_g_fourier import (
    CARRY_CONCEPTS_BY_LEVEL,
    FDR_THRESHOLD,
    LAYERS,
    MIN_CARRY_MOD10_VALUES,
    MIN_POPULATION,
    N_PERMUTATIONS,
    POPULATIONS,
    _benjamini_hochberg,
    _get_phase_c_group_labels,
    analyze_one,
    compute_labels_and_linear_values,
    derive_paths as derive_phase_g_paths,
    get_fourier_concepts,
    get_population_mask,
    load_coloring_df,
    load_config,
    load_phase_c_eigenvalues,
    load_phase_c_metadata,
    load_phase_c_projected,
    load_phase_d_merged_basis,
    load_residualized,
    resolve_carry_binned_spec,
)


CARRY_CONCEPTS = {f"carry_{i}" for i in range(5)}
DEFAULT_THRESHOLDS = (0.30, 0.50, 0.20)
DEFAULT_CORRELATE_BASES = ("phase_c", "phase_d_merged")
DEFAULT_BRANCH_GRID = (
    (0.30, "phase_c"),
    (0.50, "phase_c"),
    (0.20, "phase_c"),
    (0.30, "phase_d_merged"),
)
SCALAR_AGGREGATES = {
    "n_nonzero_carries",
    "n_answer_digits",
    "max_carry_value",
    "total_carry_sum",
    "product",
    "product_binned",
    "correct",
}
QR_RTOL = 1e-8
VERDICT_OWN_DROP = 0.30
VERDICT_INHERITED_DROP = 0.50


def setup_logging(workspace):
    """Configure rotating file + console logging for Phase H."""
    log_dir = Path(workspace) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("phase_h")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S"
    )

    fh = RotatingFileHandler(
        log_dir / "phase_h_orthogonalize.log", maxBytes=10_000_000, backupCount=3
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def derive_paths(cfg):
    """Extend Phase G's path registry with B.2 inputs and outputs."""
    paths = derive_phase_g_paths(cfg)
    data_root = Path(cfg["paths"]["data_root"])
    workspace = Path(cfg["paths"]["workspace"])
    paths.update(
        {
            "curated_set": data_root / "curated" / "curated_set_v1.json",
            "phase_b_pairs": data_root / "phase_b" / "classified_pairs.csv",
            "phase_h_output": data_root / "phase_h" / "orthogonalize",
            "phase_h_summary": data_root / "phase_h" / "summary",
            "phase_h_plots": workspace / "plots" / "phase_h",
        }
    )
    return paths


def json_safe(obj):
    """Convert numpy/pandas scalars and arrays into JSON-safe Python objects."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if math.isnan(value) else value
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if pd.isna(obj) if not isinstance(obj, (str, bytes, list, tuple, dict)) else False:
        return None
    return obj


def load_curated_indices(paths):
    """Return {level: set(source_index)} from curated_set_v1.json."""
    with open(paths["curated_set"]) as f:
        curated = json.load(f)
    by_level = defaultdict(set)
    for problem in curated["problems"]:
        by_level[int(problem["level"])].add(int(problem["source_index"]))
    return {level: set(indices) for level, indices in by_level.items()}


def load_phase_b_pairs(paths):
    """Load Phase B classified pair table."""
    df = pd.read_csv(paths["phase_b_pairs"])
    df["level"] = df["level"].astype(int)
    return df


def load_phase_g_targets(paths, include_controls, n_controls, seed, logger):
    """
    Build the B.2 target table from Phase G helices plus optional none controls.
    """
    helices_path = paths["phase_g_summary"] / "phase_g_helices.csv"
    results_path = paths["phase_g_summary"] / "phase_g_results.csv"

    helices = pd.read_csv(helices_path)
    targets = helices[helices["concept"].isin(CARRY_CONCEPTS)].copy()
    targets["target_kind"] = "helix"

    if include_controls:
        all_results = pd.read_csv(results_path)
        controls = all_results[
            all_results["concept"].isin(CARRY_CONCEPTS)
            & (all_results["geometry_detected"] == "none")
        ].copy()
        if n_controls and len(controls) > n_controls:
            group_cols = ["concept", "level", "population"]
            controls = (
                controls.groupby(group_cols, group_keys=False)
                .apply(
                    lambda g: g.sample(
                        n=max(1, round(n_controls * len(g) / len(controls))),
                        random_state=seed,
                    )
                )
                .head(n_controls)
            )
        controls["target_kind"] = "control_none"
        targets = pd.concat([targets, controls], ignore_index=True)

    targets = targets.sort_values(
        ["target_kind", "level", "layer", "population", "concept", "subspace_type", "period_spec"]
    ).reset_index(drop=True)
    logger.info("Loaded B.2 targets: %d rows", len(targets))
    return targets


def derive_correlate_set(target_concept, level, population, threshold, pairs_df):
    """
    Resolve the Phase B structural correlate set for a target concept.

    The set is data-driven: structural Phase B pairs at the same level and
    population with |r_resid| above threshold, excluding scalar aggregates.
    """
    mask = (
        (pairs_df["level"] == int(level))
        & (pairs_df["population"] == population)
        & (pairs_df["classification"] == "structural")
        & (pairs_df["r_resid"].abs() >= float(threshold))
        & (
            (pairs_df["concept_a"] == target_concept)
            | (pairs_df["concept_b"] == target_concept)
        )
    )

    rows = pairs_df[mask].copy()
    correlates = []
    audit_rows = []
    for _, row in rows.iterrows():
        other = row["concept_b"] if row["concept_a"] == target_concept else row["concept_a"]
        if other == target_concept or other in SCALAR_AGGREGATES:
            included = False
            reason = "excluded_scalar_or_self"
        else:
            included = True
            reason = "included"
            correlates.append(str(other))
        audit_rows.append(
            {
                "concept_a": row["concept_a"],
                "concept_b": row["concept_b"],
                "other_concept": other,
                "r_raw": float(row["r_raw"]),
                "r_resid": float(row["r_resid"]),
                "classification": row["classification"],
                "included": included,
                "reason": reason,
            }
        )

    return sorted(set(correlates)), audit_rows


def basis_path(paths, level, layer, population, concept, basis_kind):
    """Return the basis path for Phase C or Phase D merged bases."""
    if basis_kind == "phase_c":
        return (
            paths["phase_c_subspaces"]
            / f"L{level}"
            / f"layer_{layer:02d}"
            / population
            / concept
            / "basis.npy"
        )
    if basis_kind == "phase_d_merged":
        return (
            paths["phase_d_subspaces"]
            / f"L{level}"
            / f"layer_{layer:02d}"
            / population
            / concept
            / "merged_basis.npy"
        )
    raise ValueError(f"Unknown basis kind: {basis_kind}")


def load_basis(paths, level, layer, population, concept, basis_kind, logger):
    """Load one basis matrix whose rows are basis vectors."""
    path = basis_path(paths, level, layer, population, concept, basis_kind)
    if not path.exists():
        logger.debug("Basis missing: %s", path)
        return None
    basis = np.load(path)
    if basis.ndim != 2 or basis.shape[1] != 4096:
        raise ValueError(f"Bad basis shape for {path}: {basis.shape}")
    if basis.shape[0] == 0:
        return None
    return basis.astype(np.float64, copy=False)


def load_correlate_bases(paths, level, layer, population, correlates, basis_kind, logger):
    """Load and stack correlate bases; missing bases are recorded and skipped."""
    bases = []
    loaded = []
    missing = []
    dims = {}
    for concept in correlates:
        basis = load_basis(paths, level, layer, population, concept, basis_kind, logger)
        if basis is None:
            missing.append(concept)
            continue
        bases.append(basis)
        loaded.append(concept)
        dims[concept] = int(basis.shape[0])

    if not bases:
        return None, loaded, missing, dims
    return np.vstack(bases), loaded, missing, dims


def build_orthogonal_complement(correlate_basis, rtol=QR_RTOL):
    """
    Build an orthonormal row-space basis Q for the correlate subspaces.

    The projector is applied as X -> X - (X @ Q) @ Q.T. We do not materialize
    the 4096x4096 projection matrix during real runs.
    """
    if correlate_basis.ndim != 2:
        raise ValueError("correlate_basis must be 2D")
    if correlate_basis.shape[0] == 0:
        raise ValueError("correlate_basis has zero rows")

    q_full, r = np.linalg.qr(correlate_basis.T, mode="reduced")
    diag = np.abs(np.diag(r))
    if len(diag) == 0 or diag.max() == 0:
        rank = 0
        q_rank = q_full[:, :0]
    else:
        rank = int(np.sum(diag > rtol * diag.max()))
        q_rank = q_full[:, :rank]

    if rank == 0:
        residual_norm = 1.0
        nullspace_norm = 1.0
        q_orthonormality_norm = 0.0
    else:
        reconstructed = (correlate_basis @ q_rank) @ q_rank.T
        denom = max(np.linalg.norm(correlate_basis), 1e-12)
        residual_norm = float(np.linalg.norm(reconstructed - correlate_basis) / denom)
        nullspace_norm = residual_norm
        q_orthonormality_norm = float(
            np.linalg.norm(q_rank.T @ q_rank - np.eye(rank), ord="fro")
        )

    return q_rank, {
        "d_correlates_nominal": int(correlate_basis.shape[0]),
        "d_correlates_rank": int(rank),
        "residual_norm": residual_norm,
        "nullspace_norm": nullspace_norm,
        "q_orthonormality_norm": q_orthonormality_norm,
        "idempotent_norm": q_orthonormality_norm,
        "idempotent_norm_note": "Computed as ||Q.T@Q-I||_F; P=I-QQ.T is idempotent when Q is orthonormal.",
        "qr_rtol": float(rtol),
    }


def apply_projector(x, q_rank):
    """Apply P_perp to an activation matrix without forming P_perp."""
    if q_rank.shape[1] == 0:
        return x.copy()
    return x - (x @ q_rank) @ q_rank.T


def project_orthogonalized_from_raw(raw_projected, x_full, q_rank, target_basis):
    """
    Project orthogonalized activations into a target basis.

    If raw_projected = (X - mean) @ B.T, then:
      (P_perp X - mean) @ B.T = raw_projected - (X @ Q) @ (Q.T @ B.T)
    so this exactly preserves Phase C's centering convention without requiring
    Phase C to persist the training mean.
    """
    if q_rank.shape[1] == 0:
        return raw_projected.copy()
    return raw_projected - (x_full @ q_rank) @ (q_rank.T @ target_basis.T)


def get_target_basis(paths, level, layer, population, concept, subspace_type, logger):
    """Load the target basis used by the original Phase G detection."""
    if subspace_type == "phase_c":
        return load_basis(paths, level, layer, population, concept, "phase_c", logger)
    if subspace_type == "phase_d_merged":
        return load_basis(paths, level, layer, population, concept, "phase_d_merged", logger)
    raise ValueError(f"Unsupported target subspace_type: {subspace_type}")


def get_concept_info(level, concept, coloring_df, logger):
    """Fetch Phase G concept registry entry for a concept."""
    concepts = get_fourier_concepts(level, coloring_df, logger)
    for info in concepts:
        if info["name"] == concept:
            return info
    raise ValueError(f"Concept {concept} not found in L{level} Phase G registry")


def resolve_period_spec(row, concept_info, level, layer, population, coloring_df, paths, logger):
    """Reconstruct the exact Phase G period values and labels for a target row."""
    concept = row["concept"]
    period_spec = row["period_spec"]
    period = int(row["period"])
    values = np.array(json.loads(row["values_tested"]), dtype=int)

    spec = None
    for candidate in concept_info["period_specs"]:
        if candidate["spec_name"] == period_spec:
            spec = candidate
            break
    if spec is None:
        raise ValueError(f"period_spec {period_spec} not registered for {concept}")

    pop_mask = get_population_mask(coloring_df, population, logger)
    raw_labels_pop = coloring_df[concept_info["column"]].values[pop_mask]
    valid_mask = (
        ~np.isnan(raw_labels_pop)
        if np.issubdtype(raw_labels_pop.dtype, np.floating)
        else np.ones(len(raw_labels_pop), dtype=bool)
    )
    raw_labels_valid = raw_labels_pop[valid_mask].astype(int)

    phase_c_groups, _ = _get_phase_c_group_labels(
        level, layer, population, concept, paths, logger
    )
    if phase_c_groups is not None:
        labels, unique_vals, v_linear_mapped = compute_labels_and_linear_values(
            raw_labels_valid, phase_c_groups
        )
    else:
        labels = raw_labels_valid
        unique_vals = np.array(sorted(np.unique(labels)), dtype=int)
        v_linear_mapped = unique_vals.astype(float)

    if spec.get("needs_phase_c_groups"):
        resolved = resolve_carry_binned_spec(
            concept_info, phase_c_groups, raw_labels_valid, logger
        )
        if resolved is None:
            raise ValueError(f"Could not resolve carry_binned for {concept}")
        v_linear = resolved["v_linear"]
        keep_mask_labels = np.ones(len(labels), dtype=bool)
        unique_values = unique_vals
    else:
        if period_spec == "carry_mod10" and concept_info["is_carry"]:
            present_values = np.array([v for v in values if v in unique_vals], dtype=int)
            if len(present_values) < MIN_CARRY_MOD10_VALUES:
                raise ValueError(
                    f"Only {len(present_values)} carry_mod10 values present for {concept}"
                )
            values = present_values
        keep_mask_labels = np.isin(labels, values)
        unique_values = values
        v_linear = values.astype(float)

    return {
        "period": period,
        "period_spec": period_spec,
        "values": values,
        "v_linear": v_linear,
        "labels_all_valid": labels,
        "valid_mask_pop": valid_mask,
        "keep_mask_labels": keep_mask_labels,
        "unique_values": unique_values,
    }


def select_curated_population_rows(coloring_df, level, population, curated_indices):
    """Map curated full-data indices into population-relative row positions."""
    pop_mask = get_population_mask(coloring_df, population, logging.getLogger("phase_h"))
    pop_indices_full = np.where(pop_mask)[0]
    curated_mask_pop = np.isin(pop_indices_full, list(curated_indices.get(level, set())))
    return pop_indices_full, curated_mask_pop


def build_projected_matrices(
    row,
    paths,
    coloring_df,
    curated_indices,
    period_ctx,
    q_rank,
    target_basis,
    logger,
):
    """
    Return raw/orthogonalized projected matrices and labels for one target cell.
    """
    level = int(row["level"])
    layer = int(row["layer"])
    population = row["population"]
    concept = row["concept"]
    subspace_type = row["subspace_type"]

    pop_indices_full, curated_mask_pop = select_curated_population_rows(
        coloring_df, level, population, curated_indices
    )
    valid_mask = period_ctx["valid_mask_pop"]
    if len(valid_mask) != len(pop_indices_full):
        raise ValueError(
            f"Population valid mask length mismatch: {len(valid_mask)} vs {len(pop_indices_full)}"
        )

    labels_all = period_ctx["labels_all_valid"]
    keep_labels = period_ctx["keep_mask_labels"]

    base_mask_pop = curated_mask_pop & valid_mask
    row_indices_valid = pop_indices_full[base_mask_pop]
    labels_curated_valid = labels_all[curated_mask_pop[valid_mask]]

    keep_curated = keep_labels[curated_mask_pop[valid_mask]]
    row_indices = row_indices_valid[keep_curated]
    labels = labels_curated_valid[keep_curated]

    if len(row_indices) < MIN_POPULATION:
        raise ValueError(f"Only {len(row_indices)} curated samples after filtering")

    x_full = load_residualized(level, layer, paths, logger)[row_indices].astype(
        np.float64, copy=False
    )

    if subspace_type == "phase_c":
        projected_all = load_phase_c_projected(
            level, layer, population, concept, paths, logger
        )
        if projected_all is None:
            raise FileNotFoundError(f"Missing Phase C projected_all for {concept}")
        raw_valid = projected_all[valid_mask]
        raw_curated_valid = raw_valid[curated_mask_pop[valid_mask]]
        raw_projected = raw_curated_valid[keep_curated].astype(np.float64, copy=False)
        eigenvalues = load_phase_c_eigenvalues(
            level, layer, population, concept, paths, logger
        )
    elif subspace_type == "phase_d_merged":
        raw_projected = x_full @ target_basis.T
        eigenvalues = None
    else:
        raise ValueError(f"Unsupported subspace_type: {subspace_type}")

    orthog_projected = project_orthogonalized_from_raw(
        raw_projected, x_full, q_rank, target_basis
    )

    return raw_projected, orthog_projected, labels, eigenvalues, len(row_indices)


def compact_analysis_result(result):
    """Keep the scalar Phase G fields needed in B.2 outputs."""
    keys = [
        "two_axis_fcr",
        "two_axis_best_freq",
        "two_axis_coord_a",
        "two_axis_coord_b",
        "two_axis_p_value",
        "uniform_fcr_top1",
        "uniform_fcr_p_value",
        "eigenvalue_fcr_top1",
        "fcr_top1_max",
        "fcr_top1_max_coord",
        "fcr_top1_max_freq",
        "dominant_freq_mode",
        "n_sig_coords_at_mode_freq",
        "circle_detected",
        "helix_detected",
        "geometry_detected",
        "multi_freq_pattern",
        "helix_fcr",
        "helix_best_freq",
        "helix_linear_coord",
        "helix_p_value",
        "p_value_floor",
        "p_saturated",
    ]
    out = {key: result.get(key) for key in keys}
    helix_res = result.get("_helix_res", {})
    if helix_res:
        out["helix_linear_power"] = helix_res.get("helix_linear_power")
        out["total_power_helix"] = helix_res.get("total_power_helix")
    return json_safe(out)


def compute_drop(raw, orthog):
    """Compute FCR and helix-power drops."""
    def rel_drop(raw_value, orthog_value):
        raw_value = float(raw_value)
        orthog_value = float(orthog_value)
        if abs(raw_value) < 1e-12:
            return np.nan
        return (raw_value - orthog_value) / raw_value

    raw_power = raw.get("total_power_helix")
    orthog_power = orthog.get("total_power_helix")
    return {
        "two_axis_fcr_abs": float(raw["two_axis_fcr"] - orthog["two_axis_fcr"]),
        "two_axis_fcr_rel": rel_drop(raw["two_axis_fcr"], orthog["two_axis_fcr"]),
        "helix_fcr_abs": float(raw["helix_fcr"] - orthog["helix_fcr"]),
        "helix_fcr_rel": rel_drop(raw["helix_fcr"], orthog["helix_fcr"]),
        "helix_power_abs": (
            float(raw_power - orthog_power)
            if raw_power is not None and orthog_power is not None
            else np.nan
        ),
        "helix_power_rel": (
            rel_drop(raw_power, orthog_power)
            if raw_power is not None and orthog_power is not None
            else np.nan
        ),
    }


def verdict_from_drop(drop, orthog_helix_q_value=None, target_kind="helix"):
    """Apply the pre-registered B.2 verdict rule."""
    if target_kind != "helix":
        return "control"
    rel = drop.get("helix_fcr_rel")
    loses_fdr = (
        orthog_helix_q_value is not None
        and not pd.isna(orthog_helix_q_value)
        and orthog_helix_q_value >= FDR_THRESHOLD
    )
    if loses_fdr or (not pd.isna(rel) and rel > VERDICT_INHERITED_DROP):
        return "inherited"
    if not pd.isna(rel) and rel < VERDICT_OWN_DROP:
        return "own_structure"
    if not pd.isna(rel) and VERDICT_OWN_DROP <= rel <= VERDICT_INHERITED_DROP:
        return "ambiguous"
    return "unclassified"


def process_cell(
    row,
    threshold,
    correlate_basis_kind,
    paths,
    pairs_df,
    curated_indices,
    coloring_cache,
    n_perms,
    rng,
    logger,
):
    """Run one B.2 cell/branch."""
    level = int(row["level"])
    layer = int(row["layer"])
    population = row["population"]
    concept = row["concept"]
    subspace_type = row["subspace_type"]
    period_spec = row["period_spec"]

    logger.info(
        "B.2 cell: %s L%d/layer%d/%s target=%s/%s threshold=%.2f correlates=%s",
        concept,
        level,
        layer,
        population,
        subspace_type,
        period_spec,
        threshold,
        correlate_basis_kind,
    )

    if level not in coloring_cache:
        coloring_cache[level] = load_coloring_df(level, paths, logger)
    coloring_df = coloring_cache[level]
    concept_info = get_concept_info(level, concept, coloring_df, logger)
    period_ctx = resolve_period_spec(
        row, concept_info, level, layer, population, coloring_df, paths, logger
    )

    correlates, correlate_audit = derive_correlate_set(
        concept, level, population, threshold, pairs_df
    )
    stacked_basis, loaded_correlates, missing_correlates, correlate_dims = load_correlate_bases(
        paths, level, layer, population, correlates, correlate_basis_kind, logger
    )
    if stacked_basis is None:
        raise ValueError(f"No correlate bases loaded for {concept} L{level}/{population}")

    q_rank, validation = build_orthogonal_complement(stacked_basis)
    target_basis = get_target_basis(
        paths, level, layer, population, concept, subspace_type, logger
    )
    if target_basis is None:
        raise FileNotFoundError(f"Missing target basis for {concept} {subspace_type}")

    raw_projected, orthog_projected, labels, eigenvalues, n_rows = build_projected_matrices(
        row,
        paths,
        coloring_df,
        curated_indices,
        period_ctx,
        q_rank,
        target_basis,
        logger,
    )

    unique_values = np.asarray(period_ctx["unique_values"], dtype=int)
    v_linear = np.asarray(period_ctx["v_linear"], dtype=float)
    present_mask = np.array([np.sum(labels == v) > 0 for v in unique_values])
    if not present_mask.all():
        missing_values = unique_values[~present_mask].tolist()
        logger.warning(
            "Curated subset missing %d value groups for %s/%s: %s; "
            "keeping period=%d but dropping empty groups from Fourier centroids",
            len(missing_values),
            concept,
            period_spec,
            missing_values,
            period_ctx["period"],
        )
        unique_values = unique_values[present_mask]
        v_linear = v_linear[present_mask]
    if len(unique_values) < 3:
        raise ValueError(
            f"Only {len(unique_values)} non-empty value groups for {concept}/{period_spec}"
        )

    raw_result = analyze_one(
        level,
        layer,
        population,
        concept_info,
        period_ctx["period"],
        period_spec,
        unique_values,
        v_linear,
        subspace_type,
        raw_projected,
        labels,
        unique_values,
        eigenvalues,
        n_perms,
        rng,
        logger,
    )
    orthog_result = analyze_one(
        level,
        layer,
        population,
        concept_info,
        period_ctx["period"],
        period_spec,
        unique_values,
        v_linear,
        subspace_type,
        orthog_projected,
        labels,
        unique_values,
        eigenvalues,
        n_perms,
        rng,
        logger,
    )

    raw = compact_analysis_result(raw_result)
    orthog = compact_analysis_result(orthog_result)
    drop = compute_drop(raw, orthog)

    return {
        "target_kind": row.get("target_kind", "helix"),
        "level": level,
        "layer": layer,
        "population": population,
        "concept": concept,
        "subspace_type": subspace_type,
        "period_spec": period_spec,
        "period": int(period_ctx["period"]),
        "correlate_threshold": float(threshold),
        "correlate_basis": correlate_basis_kind,
        "correlate_set": loaded_correlates,
        "correlate_set_requested": correlates,
        "correlate_set_missing_bases": missing_correlates,
        "correlate_dims": correlate_dims,
        "correlate_audit": correlate_audit,
        "d_correlates_nominal": validation["d_correlates_nominal"],
        "d_correlates_rank": validation["d_correlates_rank"],
        "n_curated_rows": int(n_rows),
        "raw": raw,
        "orthog": orthog,
        "drop": json_safe(drop),
        "verdict": verdict_from_drop(drop, target_kind=row.get("target_kind", "helix")),
        "validation": validation,
    }


def flatten_result(result):
    """Flatten nested result JSON into one summary CSV row."""
    base_keys = [
        "target_kind",
        "level",
        "layer",
        "population",
        "concept",
        "subspace_type",
        "period_spec",
        "period",
        "correlate_threshold",
        "correlate_basis",
        "d_correlates_nominal",
        "d_correlates_rank",
        "n_curated_rows",
        "verdict",
    ]
    row = {key: result.get(key) for key in base_keys}
    row["correlate_set"] = json.dumps(result.get("correlate_set", []))
    row["correlate_set_missing_bases"] = json.dumps(
        result.get("correlate_set_missing_bases", [])
    )
    for prefix in ("raw", "orthog"):
        for key, value in result[prefix].items():
            row[f"{prefix}_{key}"] = value
    for key, value in result["drop"].items():
        row[f"drop_{key}"] = value
    for key, value in result["validation"].items():
        if key.endswith("_note"):
            continue
        row[f"validation_{key}"] = value
    return row


def result_output_path(paths, result):
    """Return per-cell JSON path, including branch keys to avoid overwrites."""
    threshold = f"{result['correlate_threshold']:.2f}".replace(".", "p")
    filename = (
        f"{result['concept']}_{result['subspace_type']}_{result['period_spec']}_"
        f"thr{threshold}_{result['correlate_basis']}.json"
    )
    return (
        paths["phase_h_output"]
        / f"L{result['level']}"
        / f"layer_{result['layer']:02d}"
        / result["population"]
        / result["concept"]
        / filename
    )


def write_cell_json(paths, result):
    """Write one B.2 per-cell JSON artifact."""
    out_path = result_output_path(paths, result)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(json_safe(result), f, indent=2)


def apply_b2_fdr_and_verdicts(results):
    """Apply FDR to B.2 raw/orthog p-values and update verdicts."""
    if not results:
        return results
    df = pd.DataFrame([flatten_result(r) for r in results])
    for prefix in ("raw", "orthog"):
        for test in ("two_axis", "helix"):
            p_col = f"{prefix}_{test}_p_value"
            q_col = f"{prefix}_{test}_q_value"
            if p_col not in df.columns:
                continue
            p_vals = df[p_col].astype(float).fillna(1.0).values
            df[q_col] = _benjamini_hochberg(p_vals)

    for i, result in enumerate(results):
        for prefix in ("raw", "orthog"):
            for test in ("two_axis", "helix"):
                q_col = f"{prefix}_{test}_q_value"
                if q_col in df.columns:
                    result[prefix][f"{test}_q_value"] = float(df.loc[i, q_col])
        result["verdict"] = verdict_from_drop(
            result["drop"],
            orthog_helix_q_value=result["orthog"].get("helix_q_value"),
            target_kind=result.get("target_kind", "helix"),
        )
    return results


def save_summary(paths, results, logger):
    """Write summary CSV and correlate-set audit JSON."""
    paths["phase_h_summary"].mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([flatten_result(r) for r in results])
    summary_path = paths["phase_h_summary"] / "orthogonalization_results.csv"
    df.to_csv(summary_path, index=False)

    correlate_sets = {}
    for result in results:
        key = (
            f"L{result['level']}/layer_{result['layer']:02d}/"
            f"{result['population']}/{result['concept']}/"
            f"{result['subspace_type']}/{result['period_spec']}/"
            f"thr={result['correlate_threshold']:.2f}/"
            f"basis={result['correlate_basis']}"
        )
        correlate_sets[key] = {
            "correlate_set": result["correlate_set"],
            "missing_bases": result["correlate_set_missing_bases"],
            "audit": result["correlate_audit"],
        }
    with open(paths["phase_h_summary"] / "correlate_sets.json", "w") as f:
        json.dump(json_safe(correlate_sets), f, indent=2)

    logger.info("Saved %s (%d rows)", summary_path, len(df))
    return df


def run_all(args, paths, logger):
    """Run B.2 across selected target cells and sensitivity branches."""
    pairs_df = load_phase_b_pairs(paths)
    curated_indices = load_curated_indices(paths)
    targets = load_phase_g_targets(
        paths, args.include_controls, args.n_controls, args.seed, logger
    )

    if args.level:
        targets = targets[targets["level"].isin(args.level)]
    if args.layer:
        targets = targets[targets["layer"].isin(args.layer)]
    if args.population:
        targets = targets[targets["population"].isin(args.population)]
    if args.concept:
        targets = targets[targets["concept"].isin(args.concept)]
    if args.subspace_type:
        targets = targets[targets["subspace_type"].isin(args.subspace_type)]
    if args.period_spec:
        targets = targets[targets["period_spec"].isin(args.period_spec)]
    if args.limit:
        targets = targets.head(args.limit)

    if args.full_grid:
        branch_grid = [
            (threshold, correlate_basis_kind)
            for threshold in args.threshold
            for correlate_basis_kind in args.correlate_basis
        ]
    else:
        threshold_set = set(round(t, 8) for t in args.threshold)
        basis_set = set(args.correlate_basis)
        branch_grid = [
            (threshold, basis_kind)
            for threshold, basis_kind in DEFAULT_BRANCH_GRID
            if round(threshold, 8) in threshold_set and basis_kind in basis_set
        ]
    logger.info(
        "Running B.2: %d targets x %d branches, n_perms=%d",
        len(targets),
        len(branch_grid),
        args.n_perms,
    )
    logger.info("Branch grid: %s", branch_grid)

    rng = np.random.default_rng(args.seed)
    coloring_cache = {}
    results = []
    failures = []
    t0 = time.time()

    for _, target in targets.iterrows():
        for threshold, correlate_basis_kind in branch_grid:
            try:
                result = process_cell(
                    target,
                    threshold,
                    correlate_basis_kind,
                    paths,
                    pairs_df,
                    curated_indices,
                    coloring_cache,
                    args.n_perms,
                    rng,
                    logger,
                )
                results.append(result)
            except Exception as exc:
                logger.exception("B.2 cell failed: %s", exc)
                failures.append(
                    {
                        "target": target.to_dict(),
                        "threshold": threshold,
                        "correlate_basis": correlate_basis_kind,
                        "error": str(exc),
                    }
                )

    results = apply_b2_fdr_and_verdicts(results)
    for result in results:
        write_cell_json(paths, result)
    df = save_summary(paths, results, logger)

    if failures:
        fail_path = paths["phase_h_summary"] / "orthogonalization_failures.json"
        with open(fail_path, "w") as f:
            json.dump(json_safe(failures), f, indent=2)
        logger.warning("Saved %d failures to %s", len(failures), fail_path)

    logger.info(
        "B.2 complete: %d successful rows, %d failures, %.1f minutes",
        len(results),
        len(failures),
        (time.time() - t0) / 60,
    )
    if len(df) > 0:
        logger.info("Verdicts: %s", df["verdict"].value_counts().to_dict())
    return df, failures


def toy_fourier_fcr(projected, labels, values):
    """Fast deterministic FCR helper for toy validation."""
    from phase_g_fourier import compute_centroids_grouped, compute_helix_fcr, compute_linear_power, fourier_all_coordinates

    centroids, group_sizes = compute_centroids_grouped(projected, labels, values)
    centroids -= centroids.mean(axis=0)
    fourier = fourier_all_coordinates(centroids, values, len(values))
    linear = compute_linear_power(centroids, values.astype(float), group_sizes)
    helix = compute_helix_fcr(fourier, linear, values.astype(float), group_sizes)
    return fourier["two_axis_fcr"], helix["helix_fcr"], helix["total_power_helix"]


def make_toy_data(mode, hidden_dim=512, n_per_value=80, noise=0.03, seed=0):
    """Synthetic activations for projector math validation."""
    rng = np.random.default_rng(seed)
    values = np.arange(12)
    labels = np.repeat(values, n_per_value)
    theta = 2 * np.pi * labels / len(values)
    z = (labels - labels.mean()) / labels.std()
    n = len(labels)

    x = noise * rng.standard_normal((n, hidden_dim))
    target_basis = np.eye(3, hidden_dim)

    if mode == "own":
        x[:, 0] += np.cos(theta)
        x[:, 1] += np.sin(theta)
        x[:, 2] += z
        correlate_basis = np.eye(6, hidden_dim, k=10)
    elif mode == "inherited":
        x[:, 0] += np.cos(theta)
        x[:, 1] += np.sin(theta)
        x[:, 2] += z
        correlate_basis = np.eye(3, hidden_dim)
    elif mode == "power_split":
        x[:, 0] += 0.7 * np.cos(theta)
        x[:, 1] += 0.7 * np.sin(theta)
        x[:, 2] += 0.7 * z
        x[:, 3] += 0.7 * np.cos(theta)
        x[:, 4] += 0.7 * np.sin(theta)
        x[:, 5] += 0.7 * z
        target_basis = np.eye(6, hidden_dim)
        correlate_basis = np.eye(3, hidden_dim, k=3)
    elif mode == "null_bleed":
        x[:, 0] += np.cos(theta)
        x[:, 1] += np.sin(theta)
        correlate_basis = np.eye(2, hidden_dim)
    else:
        raise ValueError(mode)

    raw = x @ target_basis.T
    q_rank, validation = build_orthogonal_complement(correlate_basis)
    orthog = project_orthogonalized_from_raw(raw, x, q_rank, target_basis)
    return raw, orthog, labels, values, correlate_basis, q_rank, validation


def run_toy_validation(logger):
    """Run toy checks that validate the B.2 linear algebra."""
    logger.info("=" * 60)
    logger.info("B.2 TOY VALIDATION")
    logger.info("=" * 60)
    passed = True

    raw, orthog, labels, values, _, _, validation = make_toy_data("own", seed=1)
    raw_fcr, raw_hfcr, raw_power = toy_fourier_fcr(raw, labels, values)
    orth_fcr, orth_hfcr, orth_power = toy_fourier_fcr(orthog, labels, values)
    drop = (raw_hfcr - orth_hfcr) / raw_hfcr
    power_drop = (raw_power - orth_power) / raw_power
    ok = abs(drop) < 0.05 and abs(power_drop) < 0.05 and validation["nullspace_norm"] < 1e-10
    logger.info(
        "T1 own helix: raw_hfcr=%.3f orth_hfcr=%.3f drop=%.3f power_drop=%.3f %s",
        raw_hfcr,
        orth_hfcr,
        drop,
        power_drop,
        "PASS" if ok else "FAIL",
    )
    passed &= ok

    raw, orthog, labels, values, _, _, validation = make_toy_data("inherited", seed=2)
    raw_fcr, raw_hfcr, raw_power = toy_fourier_fcr(raw, labels, values)
    orth_fcr, orth_hfcr, orth_power = toy_fourier_fcr(orthog, labels, values)
    drop = (raw_hfcr - orth_hfcr) / raw_hfcr
    power_drop = (raw_power - orth_power) / raw_power
    ok = drop > 0.50 and power_drop > 0.95 and validation["nullspace_norm"] < 1e-10
    logger.info(
        "T2 inherited helix: raw_hfcr=%.3f orth_hfcr=%.3f drop=%.3f power_drop=%.3f %s",
        raw_hfcr,
        orth_hfcr,
        drop,
        power_drop,
        "PASS" if ok else "FAIL",
    )
    passed &= ok

    raw, orthog, labels, values, _, _, validation = make_toy_data("power_split", seed=3)
    raw_fcr, raw_hfcr, raw_power = toy_fourier_fcr(raw, labels, values)
    orth_fcr, orth_hfcr, orth_power = toy_fourier_fcr(orthog, labels, values)
    drop = (raw_hfcr - orth_hfcr) / raw_hfcr
    power_drop = (raw_power - orth_power) / raw_power
    ok = power_drop > 0.35 and power_drop < 0.65 and validation["nullspace_norm"] < 1e-10
    logger.info(
        "T3 split power: raw_hfcr=%.3f orth_hfcr=%.3f fcr_drop=%.3f power_drop=%.3f %s",
        raw_hfcr,
        orth_hfcr,
        drop,
        power_drop,
        "PASS" if ok else "FAIL",
    )
    logger.info(
        "T3 note: FCR is a ratio, so split amplitude can preserve FCR while power drops."
    )
    passed &= ok

    raw, orthog, labels, values, _, _, validation = make_toy_data("null_bleed", seed=4)
    raw_fcr, raw_hfcr, raw_power = toy_fourier_fcr(raw, labels, values)
    orth_fcr, orth_hfcr, orth_power = toy_fourier_fcr(orthog, labels, values)
    drop = (raw_hfcr - orth_hfcr) / raw_hfcr
    power_drop = (raw_power - orth_power) / raw_power
    ok = drop > 0.50 and power_drop > 0.95 and validation["nullspace_norm"] < 1e-10
    logger.info(
        "T4 correlate bleed: raw_hfcr=%.3f orth_hfcr=%.3f drop=%.3f power_drop=%.3f %s",
        raw_hfcr,
        orth_hfcr,
        drop,
        power_drop,
        "PASS" if ok else "FAIL",
    )
    passed &= ok

    rng = np.random.default_rng(5)
    b = rng.standard_normal((20, 128))
    q_rank, validation = build_orthogonal_complement(b)
    x = np.eye(128)
    x_orth = apply_projector(x, q_rank)
    residual = np.linalg.norm(b @ x_orth) / np.linalg.norm(b)
    ok = residual < 1e-10 and validation["q_orthonormality_norm"] < 1e-10
    logger.info(
        "T5 projector null-space: residual=%.3e q_orth=%.3e %s",
        residual,
        validation["q_orthonormality_norm"],
        "PASS" if ok else "FAIL",
    )
    passed &= ok

    logger.info("Toy validation: %s", "PASS" if passed else "FAIL")
    return passed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase H / B.2 orthogonalization control for carry helices"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--toy", action="store_true", help="Run toy validation only")
    parser.add_argument("--level", type=int, nargs="+", default=None)
    parser.add_argument("--layer", type=int, nargs="+", default=None)
    parser.add_argument("--population", nargs="+", choices=POPULATIONS, default=None)
    parser.add_argument("--concept", nargs="+", default=None)
    parser.add_argument("--subspace-type", nargs="+", choices=["phase_c", "phase_d_merged"], default=None)
    parser.add_argument("--period-spec", nargs="+", default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
        help="Correlate |r_resid| thresholds",
    )
    parser.add_argument(
        "--correlate-basis",
        nargs="+",
        choices=["phase_c", "phase_d_merged"],
        default=list(DEFAULT_CORRELATE_BASES),
        help="Basis family used for correlate removal",
    )
    parser.add_argument("--n-perms", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Debug: first N targets only")
    parser.add_argument("--include-controls", action="store_true")
    parser.add_argument("--n-controls", type=int, default=200)
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Run all threshold x correlate-basis combinations instead of the pre-registered four-branch grid",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    cfg = load_config(args.config)
    paths = derive_paths(cfg)
    logger = setup_logging(paths["workspace"])

    logger.info("Phase H / B.2 Orthogonalization started")
    logger.info("Config: %s", args.config)
    logger.info("Paths: workspace=%s data_root=%s", paths["workspace"], paths["data_root"])

    if args.toy:
        ok = run_toy_validation(logger)
        if not ok:
            raise SystemExit(1)
        return

    run_all(args, paths, logger)


if __name__ == "__main__":
    main()
