# Arithmetic Geometry

Mechanistic interpretability study of how Llama 3.1 8B internally represents multiplication. We extract residual stream activations across 9 layers for 16,064 multiplication problems at 5 difficulty levels, paired with algorithm-agnostic mathematical labels (partial products, column sums, carries, running sums), then analyze error patterns to understand where the internal math breaks down.

## Research Question

When a transformer gets multiplication wrong, where does the internal math break down? We study how activation geometry degrades across difficulty levels (1x1 through 3x3 digit multiplication) and between correct vs. wrong answers.

## Dataset

| Level | Type | Problems |
|-------|------|----------|
| 1 | 1-digit x 1-digit | 64 (all unique pairs, 2-9) |
| 2 | 2-digit x 1-digit | 4,000 |
| 3 | 2-digit x 2-digit | 4,000 |
| 4 | 3-digit x 2-digit | 4,000 |
| 5 | 3-digit x 3-digit | 4,000 |

**Total: 16,064 problems**

### Labels (algorithm-agnostic)

For every problem `a x b`, we compute:
- Input digit decomposition by place value
- All pairwise partial products (e.g., for 2x2: 4 products)
- Column sums grouped by `i+j=k`
- Carries and running sums (LSF propagation)
- Answer digits in both MSF and LSF order
- Per-digit difficulty annotation (partial product count, max column sum, carry chain length)

These are mathematical facts about the product, not steps of any algorithm.

### Carry Bounds (verified)

| Level | Max Carries by Column |
|-------|----------------------|
| 1 (1x1) | [8] |
| 2 (2x1) | [8, 8] |
| 3 (2x2) | [8, 17, 9] |
| 4 (3x2) | [8, 17, 17, 9] |
| 5 (3x3) | [8, 17, 26, 18, 9] |

## Model

**Llama 3.1 8B base** (`meta-llama/Meta-Llama-3.1-8B`), loaded locally in bfloat16.

- 32 transformer layers, hidden dim 4096
- Activations extracted at layers: [4, 6, 8, 12, 16, 20, 24, 28, 31]
- Residual stream at the `=` token (last prompt position)
- Prompt format: `"{a} * {b} ="` (MSF, standard written English)
- Greedy decoding (`do_sample=False`, `max_new_tokens=12`)

## Project Structure

```
/home/anshulk/arithmetic-geometry/       (workspace, in git)
├── pipeline.py            # Stage 1: generate, extract, evaluate
├── analysis.py            # Stage 2: error pattern analysis (no GPU)
├── config.yaml            # all parameters, paths, model config
├── run.sh                 # SLURM job script — runs both stages
├── .gitignore
├── README.md
├── labels/                # per-level problem labels + analysis summary
│   ├── level_{1-5}.json
│   └── analysis_summary.json
├── plots/                 # all diagnostic plots (9 total)
│   ├── accuracy_by_level.png           # pipeline
│   ├── activation_norm_profile.png     # pipeline
│   ├── digit_coverage.png             # pipeline
│   ├── per_digit_accuracy_heatmap.png  # analysis
│   ├── error_distributions.png         # analysis
│   ├── accuracy_vs_carries.png         # analysis
│   ├── accuracy_vs_magnitude.png       # analysis
│   ├── error_categories.png            # analysis
│   └── digit_accuracy_by_carry.png     # analysis
└── logs/                  # pipeline + analysis logs, SLURM output
    ├── pipeline.log
    ├── analysis.log
    └── slurm-*.out/err

/data/user_data/anshulk/arithmetic-geometry/  (heavy files, not in git)
├── model/                 # Llama 3.1 8B weights (~15 GB)
├── activations/           # .npy files (~2.2 GB, 45 files)
│   └── level{N}_layer{L}.npy
└── answers/               # per-level accuracy + analysis summary
    ├── level_{1-5}.json
    └── analysis_summary.json
```

## Setup

```bash
conda create -n geometry python=3.11 -y
conda activate geometry
pip install torch numpy matplotlib pyyaml transformers accelerate huggingface_hub

# Download model locally
python -c "
from huggingface_hub import snapshot_download
snapshot_download('meta-llama/Meta-Llama-3.1-8B',
                  local_dir='/data/user_data/anshulk/arithmetic-geometry/model')
"
```

## Usage

```bash
# Full end-to-end on SLURM (A6000 GPU)
conda activate geometry
sbatch run.sh

# Or run stages separately
python pipeline.py          # Stage 1: requires GPU
python analysis.py          # Stage 2: CPU only
```

### Stage 1: Pipeline (`pipeline.py`)

1. Generate 16,064 problems with full mathematical labels
2. Verify all labels (carry bounds, running sum consistency, product reconstruction)
3. Save labeled datasets to `labels/`
4. Load tokenizer, verify tokenization (all levels: 6 tokens, last = `" ="`)
5. Load model once (bfloat16, ~16 GB VRAM)
6. Extract activations at 9 layers with checkpoint/resume
7. Post-extraction sanity checks (no NaN/Inf, distinct activations, norm range)
8. Greedy decode all problems, compute per-level accuracy
9. Save answers with correctness flags
10. Generate 3 diagnostic plots

### Stage 2: Analysis (`analysis.py`)

1. Load answers and labels, merge into enriched dataset
2. Classify errors: close arithmetic, magnitude error, large arithmetic
3. Per-digit accuracy (MSF) — which digit positions fail?
4. Carry correlation — accuracy vs number of carries
5. Error structure — even/odd bias, divisibility, 10's complement, underestimation
6. Input difficulty — accuracy vs leading digits, product magnitude
7. Generate 6 analysis plots
8. Save JSON summary + text report

## Key Design Decisions

- **MSF (most-significant-first)** output matches the model's training distribution
- **Level 1 has only 64 unique problems** (2-9 x 2-9), effective sample size is 64
- **Model loads once**, shared across extraction and answer generation
- **Checkpoint/resume**: extraction skips level-layer combos where `.npy` already exists
- **`make_hook()` factory function** avoids the Python closure-over-loop-variable bug
- **`pad_token = eos_token`** required for Llama's tokenizer
- **Comma-aware parser** handles model outputs like `"3,894"` correctly

## Downstream Analysis (planned)

- **Phase A**: UMAP/t-SNE visualization colored by every label variable
- **Phase C**: rSVD for concept subspace identification
- **Phase D**: LDA for carry-value discriminative directions
- **Fourier screening**: periodicity in centroid sequences across levels
- **Correct vs. wrong geometric comparison**: the core analysis
