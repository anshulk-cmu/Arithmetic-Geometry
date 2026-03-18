# Arithmetic Geometry

The Linear Representation Hypothesis (LRH) claims that concepts in LLMs are encoded as linear directions in activation space. We show that LRH is necessary but insufficient for understanding compositional reasoning. Linear subspace methods accurately capture *atomic* concepts — individual digits, carries, partial products — but fundamentally miss how these concepts *compose*. The actual computational structure lives on nonlinear manifolds *within* those linear subspaces: a 9-dimensional digit subspace may contain a 1-dimensional circle encoding digit identity via Fourier features; computation operates as geometric transformations (rotations, translations) on these manifolds; and failure occurs when these transformations break. A linear probe finds the 9D room but completely misses that the data sits on a 1D circle inside it.

We use multi-digit multiplication in Llama 3.1 8B as a controlled testbed. The computational graph is fully known (partial products, column sums, carries, answer assembly), difficulty scales from trivial to near-impossible, and errors are structured — so we can study exactly where and how composition breaks down, and build the nonlinear geometric tools to characterize it.

## Research Question

The LRH finds the ingredients. But does it find the recipe? Multiplication requires composing multiple intermediate steps — carry propagation across columns — and it is precisely this compositional structure that breaks in the model's representations. Linear probes can detect failure (correct vs. wrong) but cannot explain it. We ask: what is the geometric structure *within* concept subspaces, how do these structures interact during composition, and what changes geometrically when computation fails?

Bai et al. (2025) found Fourier-basis digit representations in tiny 2-layer models. Gurnee et al. (2025) found helical manifolds in Claude 3.5 Haiku. Qian et al. (2024) showed carry propagation is the bottleneck. Nobody has systematically tested whether these structures exist in a large pre-trained model, mapped the full computational graph geometrically, or characterized what changes in the manifold geometry between correct and wrong answers. We fill that gap using Llama 3.1 8B with a difficulty gradient from trivial (1x1 digit, 100% accuracy) to near-impossible (3x3 digit, 6% accuracy).

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

## Pipeline Status

| Stage | Status | Script | Output |
|-------|--------|--------|--------|
| Data Generation | Complete | `pipeline.py`, `analysis.py` | 45 activation files (2.21 GB), labels, error analysis |
| Phase A: Visual Reconnaissance | Complete | `phase_a_embeddings.py`, `phase_a_analysis.py` | 351 UMAP/t-SNE embeddings, interestingness scores, priority list |
| Phase C: Concept Subspaces | Complete | `phase_c_subspaces.py` | 2,835 subspaces, significance tests, cross-layer alignment |
| Phase D: LDA for Carries | Planned | — | — |
| Fourier Screening | Planned | — | — |

## Key Findings

### Data Generation

- **Accuracy gradient works**: monotonic drop from 100% (1x1) to 6% (3x3)
- **U-shaped per-digit accuracy**: first and last digits are easy (84-99%), middle digits collapse (16% at L5 position 3) — confirms Bai et al.'s carry-chain bottleneck in a production model
- **Carries dominate difficulty**: L5 with 0 carries = 56.5% accuracy; with 5 carries = 2.5%
- **Errors are structured, not random**: median relative error 0.24% at L5; 86-95% close arithmetic
- **Even error bias**: 92-95% of errors preserve the ground truth's parity

### Phase A: Visual Reconnaissance

- **Layer 16 is the information peak**: most label variables create visible clusters here
- **Product magnitude is the dominant axis**: must be controlled in downstream analysis
- **Correct/wrong answers separate clearly at L3-L5**: maximum divergence score |Δ| = 0.695
- **CKA reveals layer groups**: layers 20-24-28 are globally similar (CKA > 0.98), but concept-specific subspaces can still rotate between them

### Phase C: LRH Works for Atomic Concepts

- **2,835 subspaces identified, 2,513 significant (88.6%)** by permutation null (1,000 shuffles, α=0.01)
- **Input digits use maximum rank**: a_units (10 values) → 9D subspace, a_tens (9 values) → 8D, all with CV > 0.99. Every ingredient of multiplication is linearly recoverable
- **Column sums are strongly represented**: col_sum_0 through col_sum_4 all significant (CV 0.97-0.99), confirming the model tracks intermediate computation. Linear methods find the ingredients
- **Carries, partial products, product magnitude** — all have significant linear subspaces. The LRH is validated for every atomic concept we tested

### Phase C: LRH Fails for Composition

- **Middle answer digits have no linear subspace**: ans_digit_1_msf (the thousands digit, requiring carry propagation to compute) has dim_perm=0 at every layer and level. The model has all the ingredients (digits, carries, column sums) in clean linear subspaces, but the composed result is not linearly encoded. This is direct evidence that composition is nonlinear
- **Correct answers compress, wrong answers don't**: at L5, correct subspaces collapse to 1-3D while wrong maintain 8-9D. The model uses a geometrically efficient (likely nonlinear) code when it succeeds — a code that linear subspace methods see as *lower*-dimensional, not *higher*. The correct computation lives on a compact manifold inside the subspace
- **Correct/wrong divergence is subtle (8-40° principal angles)**: the same subspace is used for both, but the *shape within* differs. The critical difference is in curvature, clustering, and manifold topology — precisely what linear methods cannot characterize
- **Subspace dimensionality is flat across layers**: carries hold the same linear rank from layer 4 to layer 31, yet the model is clearly computing with them. The computation transforms nonlinearly between layers while projecting to the same linear shadow

## Project Structure

```
arithmetic-geometry/                       (workspace, in git)
├── pipeline.py                            # generate problems, extract activations, evaluate
├── analysis.py                            # error pattern analysis (CPU only)
├── phase_a_embeddings.py                  # UMAP/t-SNE embeddings + interestingness scoring
├── phase_a_analysis.py                    # Phase A summary analysis
├── phase_c_subspaces.py                   # concept subspace identification via cond. covariance + SVD
├── config.yaml                            # all parameters, paths, model config
├── run.sh                                 # SLURM: data generation (GPU)
├── run_phase_a.sh                         # SLURM: Phase A (GPU for embeddings)
├── run_phase_c.sh                         # SLURM: Phase C (CPU only, 12 cores)
├── labels/                                # per-level labels + analysis summary
│   ├── level_{1-5}.json
│   └── analysis_summary.json
├── plots/                                 # all generated plots
│   ├── *.png                              # 9 data generation diagnostic plots
│   ├── phase_a/                           # 243 UMAP/t-SNE embedding plots + heatmaps
│   └── phase_c/                           # 955 plots (eigenvalue spectra, heatmaps, trajectories)
├── logs/                                  # execution logs and SLURM output
└── docs/
    ├── datageneration_stage_analysis.md   # complete data generation reference
    ├── phase_a_analysis.md                # complete Phase A reference
    └── phase_c_analysis.md                # complete Phase C reference

/data/user_data/anshulk/arithmetic-geometry/  (heavy files, not in git)
├── model/                                 # Llama 3.1 8B weights (~30 GB)
├── activations/                           # 45 .npy files, 2.21 GB total
│   └── level{N}_layer{L}.npy             # shape (n_problems, 4096), float32
├── answers/                               # per-level predictions + correctness
│   └── level_{1-5}.json
├── phase_a/                               # Phase A outputs
│   ├── coloring_dfs/                      # 5 .pkl coloring DataFrames
│   ├── embeddings/                        # 117 CSVs with 2D coordinates
│   └── interestingness/                   # scoring results
└── phase_c/                               # Phase C outputs (~5 GB)
    ├── residualized/                      # 45 product-residualized activation files
    ├── subspaces/                         # 2,835 concept subspaces (basis, eigenvalues, null)
    │   └── L{N}/layer_{LL}/{pop}/{concept}/
    ├── projections/                       # projected activations for downstream
    └── summary/                           # master CSVs + significance tables
```

## Setup

```bash
conda create -n geometry python=3.11 -y
conda activate geometry
pip install torch numpy scipy pandas scikit-learn matplotlib seaborn pyyaml \
            transformers accelerate huggingface_hub tqdm joblib umap-learn

# Download model locally
python -c "
from huggingface_hub import snapshot_download
snapshot_download('meta-llama/Meta-Llama-3.1-8B',
                  local_dir='/data/user_data/anshulk/arithmetic-geometry/model')
"
```

## Usage

```bash
# Data generation (A6000 GPU, ~5 minutes)
sbatch run.sh

# Phase A: visual reconnaissance (A6000 GPU, ~8 minutes)
sbatch run_phase_a.sh

# Phase C: concept subspaces (CPU only, 12 cores, ~58 minutes)
sbatch run_phase_c.sh

# Or run Phase C with pilot mode first (L3/layer16 only, ~2 minutes)
python phase_c_subspaces.py --config config.yaml --pilot
```

## Documentation

- **[Data Generation Stage Analysis](docs/datageneration_stage_analysis.md)** — Every design decision, full math walkthroughs, tokenization patterns, carry bound proofs, error classification, all results verified against logs
- **[Phase A Analysis](docs/phase_a_analysis.md)** — UMAP/t-SNE embedding methodology, interestingness scoring, CKA matrices, activation norm profiles, priority list for downstream phases
- **[Phase C Analysis](docs/phase_c_analysis.md)** — Conditional covariance + SVD methodology, permutation null validation, eigenvalue spectra, cross-layer alignment, correct/wrong divergence, all 2,835 subspaces documented

## What's Next: Beyond Linear Subspaces

Phase C established the linear baseline — every atomic concept has a clean subspace, but composition is missing. The next stages look *inside* those subspaces for the nonlinear structure that encodes the actual computation.

- **Fourier screening**: do digit centroids sit on circles inside their 9D subspaces? If a_tens values 0-9 trace a periodic curve, that's a Fourier encoding — the same structure Bai et al. found in toy models, now tested in a production LLM. Circles enable rotation-based arithmetic; lines don't
- **GPLVM / GP metric tensors**: full nonlinear manifold characterization with uncertainty. Discovers intrinsic dimensionality and shape without assuming circles or helices. For correct vs. wrong: does the correct population trace a clean 1D curve while the wrong population scatters? The GP gives uncertainty bars, which matter for rare carry values
- **Phase D — LDA for carries**: discriminative linear analysis to catch low-variance carry directions Phase C may have missed. Last linear method — completing the linear toolkit before going fully nonlinear
- **Manifold interaction**: how do digit manifolds and carry manifolds compose? If carry propagation is a geometric operation (rotation on the carry manifold conditional on the digit manifold), that explains both how computation works and how it fails
- **Causal validation**: ablation and patching along discovered manifolds. Steer activations along the digit circle and verify the output rotates accordingly. This is the difference between "the model stores X here" and "the model uses X here"
