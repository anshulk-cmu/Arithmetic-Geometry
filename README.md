# Arithmetic Geometry

Mechanistic interpretability study of how Llama 3.1 8B internally represents multiplication. We extract residual stream activations across 9 layers for 16,064 multiplication problems at 5 difficulty levels, paired with algorithm-agnostic mathematical labels (partial products, column sums, carries, running sums).

## Research Question

When a transformer gets multiplication wrong, where does the internal math break down? We study how activation geometry degrades across difficulty levels (1×1 through 3×3 digit multiplication) and between correct vs. wrong answers.

## Dataset

| Level | Type | Problems | Expected Accuracy |
|-------|------|----------|-------------------|
| 1 | 1-digit × 1-digit | 64 (all unique pairs, 2-9) | ~90%+ |
| 2 | 2-digit × 1-digit | 4,000 | ~40-70% |
| 3 | 2-digit × 2-digit | 4,000 | ~20-40% |
| 4 | 3-digit × 2-digit | 4,000 | ~5-15% |
| 5 | 3-digit × 3-digit | 4,000 | ~2% |

**Total: 16,064 problems**

### Labels (algorithm-agnostic)

For every problem `a × b`, we compute:
- Input digit decomposition by place value
- All pairwise partial products (e.g., for 2×2: 4 products)
- Column sums grouped by `i+j=k`
- Carries and running sums (LSF propagation)
- Answer digits in both MSF and LSF order
- Per-digit difficulty annotation (partial product count, max column sum, carry chain length)

These are mathematical facts about the product, not steps of any algorithm.

### Carry Bounds (verified)

| Level | Max Carries by Column |
|-------|----------------------|
| 1 (1×1) | [8] |
| 2 (2×1) | [8, 8] |
| 3 (2×2) | [8, 17, 9] |
| 4 (3×2) | [8, 17, 17, 9] |
| 5 (3×3) | [8, 17, 26, 18, 9] |

All bounds are tight (achieved by all-9 inputs, e.g., 999 × 999).

## Model

**Llama 3.1 8B base** (`meta-llama/Meta-Llama-3.1-8B`), loaded locally in bfloat16.

- 32 transformer layers, hidden dim 4096
- Activations extracted at layers: [4, 6, 8, 12, 16, 20, 24, 28, 31]
- Residual stream at the `=` token (last prompt position)
- Prompt format: `"{a} * {b} ="` (MSF, standard written English)
- Greedy decoding (`do_sample=False`, `max_new_tokens=12`)

## Project Structure

```
/home/anshulk/arithmetic-geometry/     (code + lightweight data)
├── pipeline.py          # full pipeline: generation → extraction → evaluation
├── config.yaml          # all parameters, paths, model config
├── run.sh               # SLURM job script (A6000 GPU)
├── data/                # per-level JSON labels (16 MB total)
│   ├── level_1.json
│   ├── level_2.json
│   ├── level_3.json
│   ├── level_4.json
│   └── level_5.json
└── logs/                # pipeline logs + diagnostic plots
    ├── pipeline.log
    ├── accuracy_by_level.png
    ├── activation_norm_profile.png
    └── digit_coverage.png

/data/user_data/anshulk/arithmetic-geometry/   (heavy files, not in repo)
├── model/               # Llama 3.1 8B weights (~15 GB)
├── activations/         # .npy files (~2.9 GB total)
│   └── activations_level{N}_layer{L}.npy   (45 files)
└── answers/             # per-level accuracy results
    └── answers_level_{N}.json
```

## Setup

```bash
# Create conda environment
conda create -n geometry python=3.11 -y
conda activate geometry
pip install torch numpy matplotlib pyyaml transformers accelerate huggingface_hub

# Set HuggingFace token (for model download)
conda env config vars set HF_TOKEN=<your_token> -n geometry
conda deactivate && conda activate geometry

# Download model locally
python -c "
from huggingface_hub import snapshot_download
snapshot_download('meta-llama/Meta-Llama-3.1-8B',
                  local_dir='/data/user_data/anshulk/arithmetic-geometry/model')
"
```

## Usage

```bash
# Run on SLURM (A6000 GPU)
conda activate geometry
sbatch run.sh

# Or run directly (if GPU available)
python pipeline.py
```

The pipeline executes in order:
1. Generate 16,064 problems with full mathematical labels
2. Verify all labels (carry bounds, running sum consistency, product reconstruction)
3. Save labeled datasets to `data/`
4. Load tokenizer → verify tokenization (all levels: 6 tokens, last = `" ="`)
5. Load model once (bfloat16, ~16 GB VRAM)
6. Extract activations at 9 layers with checkpoint/resume
7. Post-extraction sanity checks (no NaN/Inf, distinct activations, norm range)
8. Greedy decode all problems, compute per-level accuracy
9. Save answers with correctness flags
10. Generate 3 diagnostic plots

## Key Design Decisions

- **MSF (most-significant-first)** output matches the model's training distribution. The first predicted digit is the hardest (depends on all carries). Per-digit difficulty is annotated.
- **Level 1 has only 64 unique problems** (2-9 × 2-9). Forward pass is deterministic, so repeating prompts gives identical activations. Effective sample size is 64.
- **Model loads once**, shared across pilot, extraction, and answer generation.
- **Checkpoint/resume**: extraction skips level-layer combos where the `.npy` file already exists with correct shape.
- **`make_hook()` factory function** avoids the Python closure-over-loop-variable bug in hook registration.
- **`pad_token = eos_token`** required for Llama's tokenizer (no default pad token).

## Downstream Analysis (planned)

The dataset feeds into:
- **Phase A**: UMAP/t-SNE visualization colored by every label variable
- **Phase C**: rSVD for concept subspace identification
- **Phase D**: LDA for carry-value discriminative directions
- **Fourier screening**: periodicity in centroid sequences across levels
- **Correct vs. wrong geometric comparison**: the core analysis
