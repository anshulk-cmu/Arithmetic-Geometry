#!/usr/bin/env python3
"""
Phase G K&T Replication Pilot: Validate Fourier methodology against
Kantamneni & Tegmark (2025) published results.

Go/no-go gate: Verify that single-token integers [0, 360] in Llama 3.1 8B
show periodic structure at periods T in {2, 5, 10}.

Gate criterion: Each of {2, 5, 10} appears in the top-10 peaks at at least
one of layers {0, 1, 4, 8} (not necessarily the same layer).
Candidate periods restricted to T in [2, 30] to exclude low-frequency
linear-trend artifacts.
"""

import sys
import numpy as np
import torch
from pathlib import Path
import json
import argparse
import logging
from logging.handlers import RotatingFileHandler
import time
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

INTEGER_RANGE = (0, 360)  # Match K&T's range
TARGET_PERIODS = {2, 5, 10}
# Cap at T=30 to avoid low-frequency linear-trend artifacts dominating the spectrum
CANDIDATE_PERIODS = list(range(2, 31))
KT_LAYERS = [0, 1, 4, 8]
BATCH_SIZE = 64
GATE_TOP_N = 10  # Each target period must appear in top-N at ANY layer


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════


def setup_logging(workspace):
    """Configure logging for K&T pilot."""
    log_dir = Path(workspace) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("kt_pilot")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S"
    )

    fh = RotatingFileHandler(
        log_dir / "phase_g_kt_pilot.log", maxBytes=10_000_000, backupCount=3
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════


def load_model_and_tokenizer(model_path, cpu_only=False, logger=None):
    """Load Llama 3.1 8B model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    logger.info("Tokenizer loaded: vocab_size=%d", tokenizer.vocab_size)

    device_map = "cpu" if cpu_only else "auto"
    logger.info("Loading model from %s (device_map=%s)", model_path, device_map)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    model.eval()
    logger.info("Model loaded in %.1fs, device=%s", time.time() - t0,
                next(model.parameters()).device)

    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# ACTIVATION EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════


def extract_activations(model, tokenizer, integers, layers, logger):
    """Extract residual stream activations at the number-token position."""
    N = len(integers)
    hidden_dim = model.config.hidden_size
    logger.info("Extracting activations: %d integers, %d layers, hidden_dim=%d",
                N, len(layers), hidden_dim)

    # Verify all integers are single tokens (BOS + number = 2 tokens)
    logger.info("Verifying single-token assumption for all %d integers...", N)
    for x in integers:
        ids = tokenizer(str(x), add_special_tokens=True)["input_ids"]
        assert len(ids) == 2, (
            f"Integer {x} tokenizes to {len(ids)} tokens (expected 2: BOS + number). "
            f"Tokens: {[tokenizer.decode([t]) for t in ids]}"
        )
    logger.info("  All integers verified as single-token")

    activations = {layer: np.zeros((N, hidden_dim), dtype=np.float32) for layer in layers}

    n_batches = (N + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info("Processing %d batches of size %d", n_batches, BATCH_SIZE)

    t0 = time.time()
    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, N)
        batch_ints = integers[start:end]
        batch_strs = [str(x) for x in batch_ints]

        inputs = tokenizer(batch_strs, return_tensors="pt", padding=True,
                           add_special_tokens=True)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        logger.debug("  Batch %d/%d: ints %d-%d, input_ids shape=%s",
                      batch_idx + 1, n_batches, batch_ints[0], batch_ints[-1],
                      input_ids.shape)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states

        # Last position is always the number token under left padding
        seq_len = input_ids.shape[1]
        for layer in layers:
            hs = hidden_states[layer]
            for i in range(len(batch_ints)):
                activations[layer][start + i] = hs[i, -1].float().cpu().numpy()

        if (batch_idx + 1) % max(1, n_batches // 10) == 0:
            elapsed = time.time() - t0
            logger.info("  Batch %d/%d (%.1fs elapsed)", batch_idx + 1, n_batches, elapsed)

    elapsed = time.time() - t0
    logger.info("Extraction complete: %.1fs (%.3fs/integer)", elapsed, elapsed / N)

    return activations


# ═══════════════════════════════════════════════════════════════════════════════
# FOURIER ANALYSIS (K&T-style magnitude spectrum)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_magnitude_spectrum(activations, integers, candidate_periods, logger):
    """Compute Parseval total-power spectrum: sum of squared magnitudes across dims."""
    N, hidden_dim = activations.shape
    logger.info("Computing magnitude spectrum: N=%d, hidden_dim=%d, %d candidate periods",
                N, hidden_dim, len(candidate_periods))

    mean_act = activations.mean(axis=0, keepdims=True)
    centered = activations - mean_act

    magnitude_spectrum = {}
    t0 = time.time()

    for period in candidate_periods:
        angles = 2.0 * np.pi * integers.astype(float) / period
        cos_basis = np.cos(angles)
        sin_basis = np.sin(angles)

        a_coeff = cos_basis @ centered  # (hidden_dim,)
        b_coeff = sin_basis @ centered  # (hidden_dim,)
        total_power = float(np.sum(a_coeff ** 2 + b_coeff ** 2))
        magnitude_spectrum[period] = total_power

    elapsed = time.time() - t0
    logger.debug("  Magnitude spectrum computed in %.2fs", elapsed)

    return magnitude_spectrum


def check_gate_layer(magnitude_spectrum, logger):
    """Check which target periods appear in top-N for a single layer.

    Returns (set of found target periods, details dict).
    """
    sorted_periods = sorted(magnitude_spectrum.keys(),
                            key=lambda t: magnitude_spectrum[t], reverse=True)
    top_n = sorted_periods[:GATE_TOP_N]
    top_n_set = set(top_n)

    found = TARGET_PERIODS & top_n_set

    logger.info("  Top-%d periods by power: %s", GATE_TOP_N, top_n)
    for t in sorted(TARGET_PERIODS):
        rank = sorted_periods.index(t) + 1 if t in sorted_periods else -1
        mag = magnitude_spectrum.get(t, 0)
        in_top = "YES" if t in found else "no"
        logger.info("    T=%d: power=%.1f, rank=%d, in_top_%d=%s",
                     t, mag, rank, GATE_TOP_N, in_top)

    return found, {
        "top_n_periods": top_n,
        "target_found_at_layer": sorted(found),
        "period_powers": {t: magnitude_spectrum[t] for t in sorted(TARGET_PERIODS)},
        "period_ranks": {
            t: sorted_periods.index(t) + 1 if t in sorted_periods else -1
            for t in sorted(TARGET_PERIODS)
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════


def plot_magnitude_spectrum(magnitude_spectrum, layer, output_dir, logger):
    """Plot the magnitude spectrum for one layer."""
    periods = sorted(magnitude_spectrum.keys())
    magnitudes = [magnitude_spectrum[t] for t in periods]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(periods, magnitudes, width=0.8, color="steelblue", edgecolor="none")

    for t in TARGET_PERIODS:
        if t in magnitude_spectrum:
            ax.bar(t, magnitude_spectrum[t], width=0.8, color="red", edgecolor="none")

    ax.set_xlabel("Period T")
    ax.set_ylabel("Total Fourier power (Parseval)")
    ax.set_title(f"K&T Replication — Layer {layer} — Magnitude Spectrum")
    ax.set_xlim(0, max(periods) + 2)

    for t in sorted(TARGET_PERIODS):
        if t in magnitude_spectrum:
            ax.annotate(f"T={t}", (t, magnitude_spectrum[t]),
                        xytext=(5, 10), textcoords="offset points",
                        fontsize=9, color="red", fontweight="bold")

    fname = output_dir / f"kt_magnitude_spectrum_layer{layer}.png"
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.debug("  Saved %s", fname)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Phase G K&T Replication Pilot")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Run on CPU (slower but no GPU needed)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    workspace = Path(cfg["paths"]["workspace"])
    data_root = Path(cfg["paths"]["data_root"])
    model_path = cfg["model"]["name"]
    logger = setup_logging(workspace)

    output_dir = data_root / "phase_g" / "kt_pilot"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = workspace / "plots" / "phase_g" / "kt_pilot"
    plot_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)

    logger.info("=" * 60)
    logger.info("K&T REPLICATION PILOT")
    logger.info("=" * 60)
    logger.info("Integer range: [%d, %d]", *INTEGER_RANGE)
    logger.info("Target periods: %s", TARGET_PERIODS)
    logger.info("Candidate periods: T=%d to T=%d", CANDIDATE_PERIODS[0], CANDIDATE_PERIODS[-1])
    logger.info("Gate: each target in top-%d at any layer", GATE_TOP_N)
    logger.info("Layers: %s", KT_LAYERS)
    logger.info("Model: %s", model_path)

    t_total = time.time()

    integers = np.arange(INTEGER_RANGE[0], INTEGER_RANGE[1] + 1)
    N = len(integers)
    logger.info("N = %d integers", N)

    model, tokenizer = load_model_and_tokenizer(model_path, cpu_only=args.cpu_only,
                                                 logger=logger)

    activations = extract_activations(model, tokenizer, integers.tolist(), KT_LAYERS, logger)

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model and tokenizer freed from memory")

    # Gate: each target period must appear in top-N at at least one layer
    all_layer_results = {}
    target_found_union = set()

    for layer in KT_LAYERS:
        logger.info("--- Layer %d ---", layer)
        acts = activations[layer]
        logger.info("  Activations shape: %s", acts.shape)

        mag_spec = compute_magnitude_spectrum(acts, integers, CANDIDATE_PERIODS, logger)
        found, details = check_gate_layer(mag_spec, logger)
        target_found_union |= found
        all_layer_results[str(layer)] = details

        plot_magnitude_spectrum(mag_spec, layer, plot_dir, logger)

    gate_passed = TARGET_PERIODS.issubset(target_found_union)
    logger.info("Target periods found across all layers: %s", sorted(target_found_union))

    summary = {
        "integer_range": list(INTEGER_RANGE),
        "n_integers": N,
        "target_periods": sorted(TARGET_PERIODS),
        "candidate_periods_range": [CANDIDATE_PERIODS[0], CANDIDATE_PERIODS[-1]],
        "gate_top_n": GATE_TOP_N,
        "layers_tested": KT_LAYERS,
        "gate_passed": gate_passed,
        "target_found_union": sorted(target_found_union),
        "per_layer_results": all_layer_results,
        "elapsed_seconds": time.time() - t_total,
    }

    with open(output_dir / "kt_pilot_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    if gate_passed:
        logger.info("K&T PILOT: GATE PASSED — Fourier code validated")
        logger.info("Proceed with main Phase G experiment.")
    else:
        logger.error("K&T PILOT: GATE FAILED — Fourier code may have a bug")
        logger.error("Do NOT proceed with main experiment. Debug first.")
        logger.error("Missing targets: %s", sorted(TARGET_PERIODS - target_found_union))
    logger.info("Total time: %.1f minutes", (time.time() - t_total) / 60)
    logger.info("Results saved to %s", output_dir)
    logger.info("=" * 60)

    if not gate_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
