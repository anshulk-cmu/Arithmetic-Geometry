#!/usr/bin/env python3
"""
Extract activations at operand-token positions for Phase G number-token probe.

Finds operand positions by decoded token content (not hardcoded index) to
handle variable tokenization lengths across levels.

Saves to: activations_numtok/level{L}_layer_{LL:02d}_pos_{a|b}.npy (float16).
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import argparse
import logging
from logging.handlers import RotatingFileHandler
import time
import yaml


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

LEVELS = [2, 3, 4, 5]
EXTRACTION_LAYERS = [4, 8, 12, 16, 20, 24]  # Skip boundary layers
BATCH_SIZE = 256


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════


def setup_logging(workspace):
    """Configure logging for number-token extraction."""
    log_dir = Path(workspace) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("numtok_extract")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S"
    )

    fh = RotatingFileHandler(
        log_dir / "extract_number_token_acts.log", maxBytes=10_000_000, backupCount=3
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
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════


def load_prompts(level, data_root, logger):
    """Load prompts for a given level from the coloring DataFrame.

    Returns (prompts, operands_a, operands_b, n) where operands_a/b are
    string representations of the operands for token-content matching.
    """
    pkl_path = data_root / "phase_a" / "coloring_dfs" / f"L{level}_coloring.pkl"
    logger.info("Loading coloring DF: %s", pkl_path)
    df = pd.read_pickle(pkl_path)
    n = len(df)
    logger.info("  Loaded %d problems for L%d", n, level)

    a_vals = df["a"].values.astype(int)
    b_vals = df["b"].values.astype(int)
    prompts = [f"{int(a)} * {int(b)} =" for a, b in zip(a_vals, b_vals)]
    operands_a = [str(int(a)) for a in a_vals]
    operands_b = [str(int(b)) for b in b_vals]
    logger.debug("  Sample prompt: '%s'", prompts[0])
    return prompts, operands_a, operands_b, n


def load_model_and_tokenizer(model_path, logger):
    """Load Llama 3.1 8B model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    logger.info("Tokenizer loaded: vocab_size=%d", tokenizer.vocab_size)

    logger.info("Loading model from %s", model_path)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    logger.info("Model loaded in %.1fs", time.time() - t0)

    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════


def find_operand_positions(input_ids_row, tokenizer, operand_a_str, operand_b_str):
    """Find token positions of operands a and b by matching decoded content."""
    tokens = [tokenizer.decode([t]).strip() for t in input_ids_row]
    pos_a = None
    pos_b = None
    for i, t in enumerate(tokens):
        if t == operand_a_str and pos_a is None:
            pos_a = i
        elif t == operand_b_str and pos_a is not None and pos_b is None:
            pos_b = i
    if pos_a is None:
        raise ValueError(f"Operand a '{operand_a_str}' not found in tokens: {tokens}")
    if pos_b is None:
        raise ValueError(f"Operand b '{operand_b_str}' not found in tokens: {tokens}")
    return pos_a, pos_b


def extract_at_positions(model, tokenizer, prompts, operands_a, operands_b,
                         layers, logger):
    """Extract residual stream activations at operand-token positions."""
    N = len(prompts)
    hidden_dim = model.config.hidden_size
    logger.info("Extracting: N=%d prompts, layers=%s, operand positions by content, "
                "hidden_dim=%d", N, layers, hidden_dim)

    activations = {}
    for layer in layers:
        for operand in ("a", "b"):
            activations[(layer, operand)] = np.zeros((N, hidden_dim), dtype=np.float16)

    n_batches = (N + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info("Processing %d batches of size %d", n_batches, BATCH_SIZE)

    t0 = time.time()
    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, N)
        batch_prompts = prompts[start:end]
        batch_a = operands_a[start:end]
        batch_b = operands_b[start:end]

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                           add_special_tokens=True)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        logger.debug("  Batch %d/%d: prompts %d-%d, seq_len=%d",
                      batch_idx + 1, n_batches, start, end - 1,
                      input_ids.shape[1])

        # Find operand positions for each item in the batch
        positions_a = []
        positions_b = []
        for i in range(end - start):
            pos_a, pos_b = find_operand_positions(
                input_ids[i].cpu().tolist(), tokenizer, batch_a[i], batch_b[i]
            )
            positions_a.append(pos_a)
            positions_b.append(pos_b)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states

        for layer in layers:
            hs = hidden_states[layer]  # (batch, seq_len, hidden_dim)
            for i in range(end - start):
                activations[(layer, "a")][start + i] = (
                    hs[i, positions_a[i]].float().cpu().numpy().astype(np.float16)
                )
                activations[(layer, "b")][start + i] = (
                    hs[i, positions_b[i]].float().cpu().numpy().astype(np.float16)
                )

        if (batch_idx + 1) % max(1, n_batches // 10) == 0:
            elapsed = time.time() - t0
            logger.info("  Batch %d/%d (%.1fs elapsed, ETA %.1fs)",
                        batch_idx + 1, n_batches, elapsed,
                        elapsed / (batch_idx + 1) * (n_batches - batch_idx - 1))

    elapsed = time.time() - t0
    logger.info("Extraction complete: %.1fs (%.3fs/prompt)", elapsed, elapsed / N)

    return activations


# ═══════════════════════════════════════════════════════════════════════════════
# SAVING
# ═══════════════════════════════════════════════════════════════════════════════


def save_activations(activations, level, output_dir, logger):
    """Save extracted activations to disk using consistent naming convention."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for (layer, operand), acts in activations.items():
        fname = f"level{level}_layer_{layer:02d}_pos_{operand}.npy"
        fpath = output_dir / fname
        np.save(fpath, acts)
        logger.info("  Saved %s: shape=%s, dtype=%s (%.1f MB)",
                     fname, acts.shape, acts.dtype,
                     acts.nbytes / 1e6)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def verify_tokenization(tokenizer, prompts, operands_a, operands_b, level, logger):
    """Verify all prompts at this level tokenize consistently and operands are findable."""
    logger.info("  Verifying tokenization for all %d prompts at L%d...", len(prompts), level)
    all_lengths = []
    for p in prompts:
        ids = tokenizer(p, add_special_tokens=True)["input_ids"]
        all_lengths.append(len(ids))
    unique_lengths = set(all_lengths)
    logger.info("  Level %d tokenization lengths: %s (counts: %s)",
                level, unique_lengths,
                {l: all_lengths.count(l) for l in sorted(unique_lengths)})

    # Verify operand-finding works on a sample (full verification happens during extraction)
    n_check = min(100, len(prompts))
    rng = np.random.default_rng(seed=42)
    check_idx = rng.choice(len(prompts), n_check, replace=False)
    for idx in check_idx:
        ids = tokenizer(prompts[idx], add_special_tokens=True)["input_ids"]
        find_operand_positions(ids, tokenizer, operands_a[idx], operands_b[idx])
    logger.info("  Operand-finding verified on %d random samples", n_check)


def main():
    parser = argparse.ArgumentParser(
        description="Extract activations at operand-token positions"
    )
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    parser.add_argument("--level", type=int, nargs="+", default=None,
                        help="Specific levels to extract (default: all)")
    parser.add_argument("--layer", type=int, nargs="+", default=None,
                        help="Specific layers to extract (default: EXTRACTION_LAYERS)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    workspace = Path(cfg["paths"]["workspace"])
    data_root = Path(cfg["paths"]["data_root"])
    model_path = cfg["model"]["name"]

    logger = setup_logging(workspace)
    logger.info("=" * 60)
    logger.info("NUMBER-TOKEN ACTIVATION EXTRACTION")
    logger.info("=" * 60)

    output_base = data_root / "activations_numtok"
    levels = args.level if args.level else LEVELS
    layers = args.layer if args.layer else EXTRACTION_LAYERS

    logger.info("Levels: %s", levels)
    logger.info("Layers: %s", layers)
    logger.info("Operand positions: found by token content (a, b)")
    logger.info("Output: %s", output_base)

    # Load model once
    model, tokenizer = load_model_and_tokenizer(model_path, logger)

    t_total = time.time()

    for level in levels:
        logger.info("--- Level %d ---", level)

        # Resume check: skip if all output files already exist
        expected_files = [
            output_base / f"level{level}_layer_{layer:02d}_pos_{op}.npy"
            for layer in layers for op in ("a", "b")
        ]
        if all(f.exists() for f in expected_files):
            logger.info("  Level %d already complete, skipping", level)
            continue

        prompts, operands_a, operands_b, n = load_prompts(level, data_root, logger)

        # Verify tokenization for this level
        verify_tokenization(tokenizer, prompts, operands_a, operands_b, level, logger)

        # Extract
        activations = extract_at_positions(
            model, tokenizer, prompts, operands_a, operands_b, layers, logger
        )

        # Save
        save_activations(activations, level, output_base, logger)

        logger.info("  Level %d complete: %d prompts, %d activation files saved",
                     level, n, len(activations))

    # Free model and tokenizer
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total_elapsed = time.time() - t_total
    logger.info("=" * 60)
    logger.info("EXTRACTION COMPLETE: %.1f minutes (%.1f hours)",
                total_elapsed / 60, total_elapsed / 3600)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
