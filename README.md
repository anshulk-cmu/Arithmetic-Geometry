# Arithmetic Geometry

Mechanistic interpretability study of how Llama 3.1 8B internally represents multiplication. We extract residual stream activations across 9 layers for 16,064 multiplication problems at 5 difficulty levels, pair them with algorithm-agnostic mathematical labels, then analyze where the internal math breaks down for wrong answers.

## Research Question

When a transformer gets multiplication wrong, where does the internal math break down? Bai et al. (2025) found Fourier-basis digit representations in tiny 2-layer models. Gurnee et al. (2025) found helical manifolds in Claude 3.5 Haiku. Nobody has systematically tested whether these structures exist in a large pre-trained model doing arithmetic. We fill that gap using Llama 3.1 8B with a difficulty gradient from trivial (1x1 digit, 100% accuracy) to near-impossible (3x3 digit, 6% accuracy).

## Dataset

| Level | Type | Problems | Correct | Wrong | Accuracy |
|-------|------|----------|---------|-------|----------|
| 1 | 1x1 digit | 64 | 64 | 0 | 100.0% |
| 2 | 2x1 digit | 4,000 | 3,977 | 23 | 99.4% |
| 3 | 2x2 digit | 4,000 | 2,638 | 1,362 | 66.0% |
| 4 | 3x2 digit | 4,000 | 1,147 | 2,853 | 28.7% |
| 5 | 3x3 digit | 4,000 | 239 | 3,761 | 6.0% |

**16,064 problems total. 8,065 correct, 7,999 wrong.** Levels 3-5 are the effective range for correct/wrong geometric comparison.

## Model

- **Llama 3.1 8B base** (not instruct) — base model avoids RLHF confounds
- 32 transformer layers, 4096-dim hidden state, loaded in bfloat16 (~16 GB VRAM)
- Activations extracted at layers **[4, 6, 8, 12, 16, 20, 24, 28, 31]** spanning early/mid/late
- Residual stream captured at the `" ="` token (last prompt position, attends to full input)
- Prompt format: `"{a} * {b} ="` — MSF to match pre-training distribution
- Greedy decoding (`do_sample=False`, deterministic)

## Labels

Every problem gets algorithm-agnostic mathematical labels: all pairwise partial products, column sums grouped by output position (`i+j=k`), carries and running sums via LSF propagation, answer digits in both MSF and LSF order, and per-digit difficulty annotations (partial product count, max column sum, carry chain length). These are mathematical facts about the product, not steps of any specific algorithm. The model may use long multiplication, Fourier-space computation, or something entirely alien — the labels test whether it represents these quantities regardless of method. All 16,064 label sets are verified against tight per-column carry bounds.

## Project Structure

```
arithmetic-geometry/                     (workspace, in git)
├── pipeline.py                          # generate problems, extract activations, evaluate
├── analysis.py                          # error pattern analysis (CPU only, no GPU)
├── config.yaml                          # all parameters, paths, model config
├── run.sh                               # SLURM job script (runs both stages)
├── labels/                              # per-level labels + analysis summary
│   ├── level_{1-5}.json
│   └── analysis_summary.json
├── plots/                               # 9 diagnostic plots (3 pipeline + 6 analysis)
├── logs/                                # pipeline.log, analysis.log, slurm output
└── docs/
    └── datageneration_stage_analysis.md # complete data generation reference

/data/user_data/anshulk/arithmetic-geometry/  (heavy files, not in git)
├── model/                               # Llama 3.1 8B weights (~30 GB)
├── activations/                         # 45 .npy files, 2.21 GB total
│   └── level{N}_layer{L}.npy           # shape (n_problems, 4096), float32
└── answers/                             # per-level predictions + correctness
    └── level_{1-5}.json
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
# Full end-to-end on SLURM (A6000 GPU, ~5 minutes)
sbatch run.sh

# Or run stages separately
python pipeline.py          # Stage 1: generate, extract, evaluate (requires GPU)
python analysis.py          # Stage 2: error analysis + plots (CPU only)
```

**Stage 1** (`pipeline.py`): Generates problems, computes and verifies labels, loads the model once, extracts activations at 9 layers with checkpoint/resume, greedy-decodes answers, saves everything. Supports resume if interrupted.

**Stage 2** (`analysis.py`): Loads saved answers and labels, classifies errors, computes per-digit accuracy, carry correlation, error structure (even bias, 10's complement, underestimation), input difficulty, generates 6 analysis plots, and saves a JSON summary.

## Documentation

- **[Data Generation Stage Analysis](docs/datageneration_stage_analysis.md)** — Complete reference for this stage: every design decision, full math walkthroughs with real examples, tokenization patterns, the hook mechanism, carry bound proofs, error classification logic, and all results with numbers verified against logs.

## Key Findings (Data Generation Stage)

- **Accuracy gradient works**: monotonic drop from 100% (1x1) to 6% (3x3)
- **U-shaped per-digit accuracy**: first and last digits are easy (84-99%), middle digits collapse (16% at Level 5 position 3) — confirms Bai et al.'s carry-chain bottleneck in a production-scale model
- **Carries dominate difficulty**: Level 5 with 0 carries = 56.5% accuracy; with 5 carries = 2.5%
- **Errors are structured, not random**: median relative error is 0.24% at Level 5; 86-95% of errors are close arithmetic (same digit count, <5% off)
- **Even error bias**: 92-95% of errors preserve the ground truth's parity
- **Zero garbage output**: the model always produces a parseable number

## Downstream Analysis (planned)

- **Phase A**: UMAP/t-SNE visualization colored by every label variable
- **Phase C**: rSVD for concept subspace identification
- **Phase D**: LDA for carry-value discriminative directions
- **Fourier screening**: periodicity in centroid sequences across levels
- **Correct vs. wrong geometric comparison**: the core analysis
