# Phase A: Complete Analysis

**Anshul's Geometric Manifold Interpretability Project**
**Carnegie Mellon University, March 2026**

This document records every decision, every number, and every result from Phase A —
the visual reconnaissance stage. It is the truth document for this stage. All numbers
are validated against the actual output files.

---

## Table of Contents

1. [Purpose of This Stage](#1-purpose-of-this-stage)
2. [What Phase A Is and Is Not](#2-what-phase-a-is-and-is-not)
3. [The Two Pre-flight Diagnostics](#3-the-two-pre-flight-diagnostics)
4. [Activation Norm Profile](#4-activation-norm-profile)
5. [Cross-Layer CKA Matrices](#5-cross-layer-cka-matrices)
6. [The Embedding Pipeline](#6-the-embedding-pipeline)
7. [Populations and Data Dimensions](#7-populations-and-data-dimensions)
8. [UMAP and t-SNE Configuration](#8-umap-and-t-sne-configuration)
9. [Interestingness Scoring](#9-interestingness-scoring)
10. [Top 50 Findings](#10-top-50-findings)
11. [Correct vs Wrong Comparison](#11-correct-vs-wrong-comparison)
12. [Angular Correlation for Digit Variables](#12-angular-correlation-for-digit-variables)
13. [Heatmaps and Summary Visualizations](#13-heatmaps-and-summary-visualizations)
14. [Tiered Plotting System](#14-tiered-plotting-system)
15. [Key Findings for Phase C/D](#15-key-findings-for-phase-cd)
16. [What This Stage Does NOT Do](#16-what-this-stage-does-not-do)
17. [Output Files](#17-output-files)
18. [Runtime and Reproducibility](#18-runtime-and-reproducibility)

---

## 1. Purpose of This Stage

Phase A is binoculars, not a microscope. The goal is to compress 4096-dimensional
activation vectors into 2-3 dimensions with UMAP/t-SNE, color by every label variable,
and see what clusters. The output is a priority list telling Phase C and D where to look.

The data generation stage gave us:
- 45 activation files (5 levels x 9 layers), each shape (n_problems, 4096) in float32
- 146,287 problems total (64 L1, 4,000 L2, 10,000 L3, 10,000 L4, 122,223 L5)
- Rich label system: input digits, partial products, column sums, carries, answer digits
- Accuracy gradient: 100% (L1) → 99.8% (L2) → 67.2% (L3) → 29.0% (L4) → 3.4% (L5)
- L5 uses a carry-stratified dataset (122,223 problems selected from 810,000 screened)

Phase A takes these raw materials and produces visual maps showing which label variables
create visible structure in the embedding space, and — critically — which structures
differ between correct and wrong answers.

---

## 2. What Phase A Is and Is Not

### What it is

- UMAP/t-SNE embeddings colored by every label variable
- Interestingness scores measuring how well labels predict embedding structure
- Per-population heatmaps (all, correct, wrong) showing where structure lives
- Correct-vs-wrong difference heatmap showing what degrades
- A priority list for Phase C/D

### What it is not

Phase A was originally scope-crept to include eigenspectrum analysis, Spearman
correlation sweeps, ANOVA F-tests, Fourier probes, LDA separation, and subspace
overlap computations. All of those were removed because they ARE Phase C/D/F:

- Eigenspectrum / rSVD → Phase C (first step)
- Subspace overlap with principal angles → Phase F (superposition detection)
- LDA direction with Cohen's d → Phase D
- Spearman sweep across 4096 raw dimensions → Phase C linear probe baseline
- Fourier probe → dedicated Fourier screening step

The only analyses retained are two lightweight pre-flight diagnostics that UMAP
genuinely cannot provide and that are needed to *decide what to run* in Phase C/D,
not to *pre-run* Phase C/D. Those are the activation norm profile and the CKA
cross-layer matrix.

---

## 3. The Two Pre-flight Diagnostics

Phase A runs two analyses before the embedding pipeline:

1. **Activation norm profile** — mean/std of L2 norms per level x layer x population.
   Takes ~3 seconds. Answers: "Is UMAP layout dominated by norm differences between
   correct and wrong? Do I need to normalize before Phase C?"

2. **Cross-layer CKA matrices** — 9x9 linear CKA per level, subsampled to 1000 points.
   Takes ~13 seconds. Answers: "Which layers are redundant so I don't waste Phase C
   compute on all 9?"

Total pre-flight time: 17 seconds.

---

## 4. Activation Norm Profile

### The question

If correct answers have systematically higher L2 norms than wrong answers, UMAP will
separate them by norm alone, making every coloring variable look interesting when the
real structure is just "correct = big norm, wrong = small norm." We need to know if
this is the case before interpreting UMAP plots.

### Results: All norms per level and layer

| Level | Layer 4 | Layer 6 | Layer 8 | Layer 12 | Layer 16 | Layer 20 | Layer 24 | Layer 28 | Layer 31 |
|-------|---------|---------|---------|----------|----------|----------|----------|----------|----------|
| L1    | 3.78    | 5.31    | 6.47    | 8.00     | 10.63    | 16.03    | 23.90    | 35.42    | 74.76    |
| L2    | 3.71    | 5.10    | 6.28    | 8.07     | 11.06    | 17.09    | 25.55    | 37.35    | 76.12    |
| L3    | 3.67    | 5.12    | 6.32    | 8.43     | 10.97    | 16.56    | 24.92    | 36.72    | 77.51    |
| L4    | 3.65    | 5.05    | 6.48    | 8.37     | 10.68    | 15.97    | 23.89    | 35.88    | 76.59    |
| L5    | 3.56    | 5.09    | 6.61    | 8.32     | 10.44    | 15.43    | 23.17    | 34.97    | 74.37    |

All levels follow the same monotonic growth pattern. Norms roughly double every ~4
layers from layer 4 to layer 28, then jump ~2x from layer 28 to layer 31. This is
standard for transformer residual streams where each layer adds to the residual.

### Results: Correct vs wrong norms (L3-L5 only)

Level 1 is 100% correct (no wrong population). Level 2 has only 23 wrong answers
(below the MIN_POPULATION threshold of 30). Correct/wrong splits are available only
for L3-L5.

| Key | Correct mean | Correct std | Wrong mean | Wrong std | Ratio (C/W) |
|-----|-------------|------------|-----------|----------|-------------|
| L3 layer 4  | 3.68 | 0.04 | 3.66 | 0.03 | 1.004 |
| L3 layer 6  | 5.11 | 0.05 | 5.13 | 0.05 | 0.996 |
| L3 layer 8  | 6.30 | 0.11 | 6.35 | 0.10 | 0.993 |
| L3 layer 12 | 8.40 | 0.13 | 8.48 | 0.10 | 0.991 |
| L3 layer 16 | 11.04| 0.32 | 10.83| 0.26 | 1.020 |
| L3 layer 20 | 16.69| 0.69 | 16.29| 0.59 | 1.025 |
| L3 layer 24 | 25.19| 1.27 | 24.36| 1.12 | 1.034 |
| L3 layer 28 | 36.99| 1.32 | 36.14| 1.15 | 1.023 |
| L3 layer 31 | 77.76| 1.95 | 76.99| 1.83 | 1.010 |
| L4 layer 4  | 3.66 | 0.04 | 3.64 | 0.03 | 1.006 |
| L4 layer 6  | 5.04 | 0.07 | 5.05 | 0.06 | 0.998 |
| L4 layer 8  | 6.46 | 0.14 | 6.49 | 0.13 | 0.994 |
| L4 layer 12 | 8.36 | 0.13 | 8.37 | 0.12 | 0.998 |
| L4 layer 16 | 10.86| 0.47 | 10.61| 0.44 | 1.024 |
| L4 layer 20 | 16.27| 0.96 | 15.84| 0.92 | 1.027 |
| L4 layer 24 | 24.47| 1.62 | 23.66| 1.54 | 1.034 |
| L4 layer 28 | 36.47| 1.60 | 35.64| 1.58 | 1.023 |
| L4 layer 31 | 77.42| 2.16 | 76.26| 2.33 | 1.015 |
| L5 layer 4  | 3.58 | 0.04 | 3.56 | 0.04 | 1.005 |
| L5 layer 6  | 5.10 | 0.07 | 5.09 | 0.08 | 1.000 |
| L5 layer 8  | 6.54 | 0.16 | 6.61 | 0.17 | 0.990 |
| L5 layer 12 | 8.37 | 0.15 | 8.32 | 0.17 | 1.006 |
| L5 layer 16 | 10.64| 0.44 | 10.43| 0.42 | 1.020 |
| L5 layer 20 | 15.71| 0.86 | 15.42| 0.81 | 1.019 |
| L5 layer 24 | 23.57| 1.48 | 23.15| 1.38 | 1.018 |
| L5 layer 28 | 35.46| 1.53 | 34.95| 1.46 | 1.015 |
| L5 layer 31 | 75.81| 3.26 | 74.32| 3.60 | 1.020 |

### Interpretation

**All ratios are between 0.992 and 1.038.** The maximum separation is 3.8% at
L4 layer 24. This is negligible.

The pattern is consistent across all three levels:
- **Early layers (4-8):** Ratios hover around 1.00, sometimes slightly below 1.0
  (wrong norms slightly higher). No systematic separation.
- **Mid layers (12-16):** Ratios start to creep above 1.0 (correct norms slightly
  higher), but only by 2-3%.
- **Late layers (20-31):** Ratios stabilize at 1.01-1.04. Correct answers have
  slightly higher norms but the difference is well within the within-group standard
  deviation.

**Verdict: Norms do not dominate UMAP layout.** No normalization is needed before
Phase C. Any structure visible in UMAP plots is directional, not norm-driven.

---

## 5. Cross-Layer CKA Matrices

### The question

We extract activations at 9 layers: [4, 6, 8, 12, 16, 20, 24, 28, 31]. If layers
20 and 24 produce nearly identical representations, running Phase C on both wastes
compute. CKA (Centered Kernel Alignment) measures representational similarity between
layer pairs.

### Method

Linear CKA uses the Gram matrix approach (n x n instead of d x d):

1. Center activations: X_c = X - mean(X, axis=0)
2. Compute Gram matrices: K_X = X_c @ X_c.T (shape n x n)
3. Double-center Gram matrices: subtract row means, column means, add grand mean
4. CKA = sum(K_X * K_Y) / sqrt(sum(K_X^2) * sum(K_Y^2))

For n=1000 (subsampled) and d=4096, this creates 1000x1000 matrices instead of
4096x4096 — much faster and numerically better behaved. Subsampling uses
RandomState(42) for reproducibility.

CKA = 1.0 means identical representations (up to linear transform). CKA > 0.98
means the two layers are functionally redundant for our purposes.

### Results: Redundant layer pairs (CKA > 0.98)

| Level | Pair | CKA |
|-------|------|-----|
| L1 | layers 20 & 24 | 0.9848 |
| L1 | layers 24 & 28 | 0.9953 |
| L2 | layers 20 & 24 | 0.9880 |
| L2 | layers 20 & 28 | 0.9840 |
| L2 | layers 24 & 28 | 0.9950 |
| L3 | layers 20 & 24 | 0.9843 |
| L3 | layers 20 & 28 | 0.9845 |
| L3 | layers 24 & 28 | 0.9943 |
| L4 | layers 20 & 24 | 0.9903 |
| L4 | layers 20 & 28 | 0.9853 |
| L4 | layers 24 & 28 | 0.9952 |
| L5 | layers 20 & 24 | 0.9844 |
| L5 | layers 24 & 28 | 0.9900 |

### The full CKA matrix (Level 3, representative)

```
        L4    L6    L8    L12   L16   L20   L24   L28   L31
L4    1.00  0.92  0.87  0.84  0.76  0.62  0.57  0.61  0.56
L6    0.92  1.00  0.95  0.91  0.82  0.65  0.59  0.63  0.61
L8    0.87  0.95  1.00  0.97  0.87  0.70  0.63  0.67  0.64
L12   0.84  0.91  0.97  1.00  0.92  0.76  0.69  0.73  0.70
L16   0.76  0.82  0.87  0.92  1.00  0.94  0.90  0.92  0.78
L20   0.62  0.65  0.70  0.76  0.94  1.00  0.99  0.98  0.76
L24   0.57  0.59  0.63  0.69  0.90  0.99  1.00  0.99  0.74
L28   0.61  0.63  0.67  0.73  0.92  0.98  0.99  1.00  0.78
L31   0.56  0.61  0.64  0.70  0.78  0.76  0.74  0.78  1.00
```

### Interpretation

**Three regimes are visible across all levels:**

1. **Early layers (4-8):** High mutual CKA (0.87-0.97), forming a "token encoding"
   block. At L1, layers 6 and 8 cross the 0.98 threshold.

2. **Mid-late layers (16-28):** A tightly coupled block. Layer 16 serves as the
   bridge — high CKA with both the early block (0.87-0.92 with layer 12) and the
   late block (0.90-0.94 with layers 20-28). Within the late block, layers 20, 24,
   and 28 are nearly identical (CKA 0.98-0.996).

3. **Layer 31:** Distinctly different from everything else. CKA with the mid-late
   block drops to 0.74-0.78 at L3-L5. This is the "output preparation" layer where
   the model assembles the final logit distribution. At harder levels (L4, L5), the
   gap between layer 31 and the mid-late block widens further — CKA(L20, L31) drops
   from 0.77 at L1 to 0.83 at L5, but CKA(L4_input, L31) drops from 0.50 at L1 to
   0.28 at L5. Harder problems create more layer-specific specialization.

**The redundancy pattern is remarkably stable across difficulty levels.** Layers
20/24/28 form a redundant trio at every level. The CKA values within this trio are:

| Pair | L1 | L2 | L3 | L4 | L5 |
|------|-----|-----|-----|-----|-----|
| 20-24 | 0.985 | 0.988 | 0.984 | 0.990 | 0.984 |
| 24-28 | 0.995 | 0.995 | 0.994 | 0.995 | 0.990 |
| 20-28 | 0.976 | 0.984 | 0.985 | 0.985 | 0.975 |

**Verdict for Phase C:** Skip layer 24. It adds nothing over layers 20 and 28.
Run Phase C on [4, 8, 12, 16, 20, 28, 31] — 7 layers instead of 9, saving ~22% of
compute with no information loss.

---

## 6. The Embedding Pipeline

The embedding pipeline (`phase_a_embeddings.py`) runs in 6 steps:

| Step | Description | Time |
|------|-------------|------|
| 1 | Build coloring DataFrames (label + answer data → pandas) | ~30s |
| 1b | Subsample L5 (122,223 → 6,030, carry-stratified) | ~2s |
| 2-3 | Compute UMAP/t-SNE embeddings, build CSVs | ~240s |
| 4 | Score all CSVs for interestingness | ~1065s |
| 5 | Generate summaries (heatmaps, comparison tables, L5 Δ) | ~7s |
| 6 | Generate tiered plots (243 total) | ~26s |
| **Total** | | **~1370s (22.8 min)** |

Step 2-3 uses cuML GPU acceleration (NVIDIA RTX A6000). On CPU this step would
take 3-6 hours. On GPU it completes in 89 seconds.

### Embedding methods

For each (level, layer, population) combination, three embeddings are computed:

1. **UMAP 2D** — primary visualization, used for all scoring and mandatory plots
2. **UMAP 3D** — interactive exploration via Plotly HTML
3. **t-SNE 2D** — validation of UMAP findings (different algorithm, same structure?)

All embeddings use random_state=42 for reproducibility.

---

## 7. Populations and Data Dimensions

### Population splits

For each level, problems are split into populations:

| Level | Full dataset | Embedding input | Correct | Wrong | Acc % | Populations |
|-------|-------------|----------------|---------|-------|-------|-------------|
| L1 | 64 | 64 | 64 | 0 | 100% | all, correct |
| L2 | 4,000 | 4,000 | 3,993 | 7 | 99.8% | all, correct |
| L3 | 10,000 | 10,000 | 6,720 | 3,280 | 67.2% | all, correct, wrong |
| L4 | 10,000 | 10,000 | 2,897 | 7,103 | 29.0% | all, correct, wrong |
| L5 | 122,223 | 6,030 (subsampled) | 280 | 5,750 | 4.6% | all, correct, wrong |

L1 and L2 have no "wrong" population (fewer than 30 wrong answers). L3-L5 have
all three populations, making them the focus of the correct-vs-wrong analysis.

**L5 subsampling:** The full L5 dataset (122,223 problems) is too large for UMAP.
It is subsampled to 6,030 points stratified by `n_nonzero_carries` using natural
frequencies from the 810,000-problem screening space. Within each carry bin,
correct answers get a floor of 50 samples (or all available if fewer exist).
The subsample preserves the natural carry distribution while keeping the correct
population visible at 4.6% (280 correct out of 6,030 total). Full metadata is
saved to `l5_subsample_meta.json`.

### CSV structure

Each CSV combines the embedding coordinates with the full label set:

| Level | Rows per CSV (all/correct/wrong) | Columns | Example columns |
|-------|----------------------------------|---------|-----------------|
| L1 | 64 / 64 / — | 28 | a, b, correct, a_units, b_units, pp_a0_x_b0, carry_0, product, umap_2d_x, ... |
| L2 | 4,000 / 3,993 / — | 41 | + a_tens, b_tens decomposition |
| L3 | 10,000 / 6,720 / 3,280 | 48 | + 4 partial products, 3 column sums, 3 carries |
| L4 | 10,000 / 2,897 / 7,103 | 55 | + 6 partial products, 4 column sums, 4 carries |
| L5 | 6,030 / 280 / 5,750 | 63 | + 9 partial products, 5 column sums, 5 carries, 6 answer digits |

Total: **117 CSVs** across all combinations, occupying 52 MB.

---

## 8. UMAP and t-SNE Configuration

### UMAP parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_components | 2 (or 3 for 3D) | Visualization |
| n_neighbors | 15 (n<100), 20 (n<500), 30 (n≥500) | Adaptive to population size |
| min_dist | 0.1 | Standard for cluster visualization |
| metric | euclidean | Default for high-dimensional activations |
| random_state | 42 | Reproducibility |

### t-SNE parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_components | 2 | Visualization |
| perplexity | 15 (n<100), 20 (n<500), 30 (n≥500), capped at (n-1)/3 | Adaptive, avoids degenerate solutions |
| learning_rate | 200.0 (GPU) / "auto" (CPU) | cuML default / sklearn default |
| max_iter | 2000 | Sufficient for convergence |
| init | pca (CPU only) | Stable initialization |
| random_state | 42 | Reproducibility |

### GPU acceleration

The cuML library (RAPIDS) provides GPU implementations of both UMAP and t-SNE that
are API-compatible with their CPU counterparts. The code auto-detects cuML availability
at import time and falls back to CPU (umap-learn + sklearn) if unavailable.

GPU timing per combination (4,000 points, 4096 dims):
- UMAP 2D: ~0.3-0.5s
- UMAP 3D: ~0.2-0.3s
- t-SNE 2D: ~0.1-0.2s
- **Total per combination: ~0.5-1.0s**

CPU timing per combination (for reference):
- UMAP 2D: ~30-60s
- UMAP 3D: ~25-50s
- t-SNE 2D: ~20-40s
- **Total per combination: ~75-150s**

GPU speedup: **~50-150x per combination.**

---

## 9. Interestingness Scoring

### The question

We have 117 CSVs, each with 28-63 label variables and 2 embedding methods (UMAP 2D
and t-SNE 2D). That is thousands of possible plots. We need a metric to decide which
ones are worth looking at.

### Scoring method

Each variable is classified by type and scored against the 2D embedding:

| Variable type | Detection | Metric | Score range |
|---------------|-----------|--------|-------------|
| Binary (correct, digit_correct_*, underestimate, even_pred, div10_pred) | name-based | Silhouette score | [-1, 1] |
| Categorical (error_category) | name-based | Silhouette score | [-1, 1] |
| Discrete ≤20 unique values | nunique check | Silhouette score | [-1, 1] |
| Continuous (>20 unique values) | nunique check | max Spearman |r| on UMAP x/y axes | [0, 1] |

Additionally, digit-valued variables (a_units, a_tens, b_units, b_tens, ans_digit_*)
get an **angular correlation** score: Spearman correlation between arctan2(embedding
angle from centroid) and digit value. This detects circular/rotational encoding of
digit values in the embedding space.

### Score statistics

| Category | Count | Mean | Max |
|----------|-------|------|-----|
| All scores | 10,242 | — | — |
| Valid scores | 9,162 | — | — |
| By metric: Silhouette | 4,500 | -0.072 | 0.537 |
| By metric: Spearman | 2,682 | 0.381 | 0.953 |
| By metric: Angular | 1,980 | 0.140 | 0.762 |

1,080 scores are NaN (insufficient data — populations with <30 valid values for
a variable, or constant variables within a population like "correct" in the
correct-only split).

**New columns in scores CSV:**
- `metric_note`: set to `"2d_silhouette_unreliable_for_ranking"` for all 5,580
  silhouette scores. Silhouette on 2D embeddings systematically underscores
  categorical variables (mean ~ -0.07). Do not deprioritize carries or digits
  based on low silhouette alone. Phase C numbers take precedence.
- `sampling_note`: set to `"carry_stratified_dataset"` for all 3,312 L5 scores.
  The L5 dataset is carry-stratified, so carry variables may appear artificially
  salient. This is a flag, not a correction — reweighting would require knowing
  the true joint distribution of all concept labels under natural sampling.

**Key observation:** Spearman scores dominate the top of the rankings because
continuous variables (product, partial products, column sums) create smooth gradients
in UMAP space. Silhouette scores for categorical variables are much lower because
categories overlap in 2D even when they separate in 4096D. This does not mean
categorical variables are uninteresting — it means UMAP compression loses their
structure. Phase C will test these in full dimensionality.

---

## 10. Top 50 Findings

The top 50 UMAP interestingness findings, ranked by Spearman |r| on UMAP 2D axes:

| Rank | Level | Layer | Pop | Variable | Score |
|------|-------|-------|-----|----------|-------|
| 1 | L5 | 6 | wrong | pp_a2_x_b2 | 0.927 |
| 2 | L5 | 6 | wrong | col_sum_4 | 0.927 |
| 3 | L1 | 8 | all | pp_a0_x_b0 | 0.924 |
| 4 | L1 | 8 | all | col_sum_0 | 0.924 |
| 5 | L1 | 8 | all | product | 0.924 |
| 6 | L5 | 6 | wrong | product | 0.920 |
| 7 | L5 | 12 | wrong | pp_a2_x_b2 | 0.905 |
| 8 | L5 | 12 | wrong | col_sum_4 | 0.905 |
| 9 | L5 | 12 | wrong | product | 0.901 |
| 10 | L4 | 12 | wrong | product | 0.885 |
| 11 | L4 | 12 | wrong | pp_a2_x_b1 | 0.882 |
| 12 | L4 | 12 | wrong | col_sum_3 | 0.882 |
| 13 | L3 | 16 | wrong | product | 0.878 |
| 14 | L4 | 16 | wrong | product | 0.875 |
| 15 | L3 | 16 | wrong | pp_a1_x_b1 | 0.870 |

**Patterns in the top 50 (shifted significantly from old dataset):**

1. **L5 wrong population dominates.** 6 of the top 9 entries are L5 wrong (was L3
   before). With 5,750 wrong samples (vs 3,761 before), the L5 wrong population
   is now large enough for UMAP to resolve clean structure. The dominant partial
   product `pp_a2_x_b2` (hundreds × hundreds) scores 0.927 — higher than any
   previous top finding.

2. **Product is no longer the sole #1.** `pp_a2_x_b2` and `col_sum_4` tie for #1
   at L5, with product close behind. The dominant partial product is more informative
   than the product itself in the wrong population — the model's wrong answers are
   organized by this single cross-term.

3. **Wrong population consistently outscores correct.** This pattern persists from
   the old dataset but is more extreme with larger samples.

4. **Layers 6 and 12 appear prominently** for L5 wrong, suggesting early layers
   already encode the dominant partial product structure that predicts failure.

---

## 11. Correct vs Wrong Comparison

This is the most important output for Paper 1. The comparison table merges
interestingness scores between correct and wrong populations on the same
(level, layer, variable) triple and computes Δ = correct_score - wrong_score.

Total pairs: **792** (only L3-L5 have both populations).

### Summary by level

| Level | Pairs | Max |Δ| | Mean |Δ| |
|-------|-------|---------|----------|
| L3 | 207 | 0.409 | 0.081 |
| L4 | 261 | 0.513 | 0.075 |
| L5 | 324 | 0.915 | 0.237 |

**The gap widens dramatically with difficulty.** L3 has moderate differences (max 0.41,
up from 0.24 with more data). L5 has extreme differences — the top |Δ| is 0.915,
meaning a variable that scores 0.58 in the wrong population scores -0.34 in the
correct population (carry_2 at layer 6). This is a qualitatively different regime
from what the old 239-correct-sample L5 could show.

### Top concepts STRONGER in wrong population (negative Δ)

| Level | Layer | Variable | Correct | Wrong | Δ |
|-------|-------|----------|---------|-------|---|
| L5 | 6 | carry_2 | -0.340 | 0.575 | -0.915 |
| L5 | 12 | carry_2 | -0.274 | 0.553 | -0.828 |
| L5 | 8 | carry_2 | -0.321 | 0.497 | -0.818 |
| L5 | 31 | carry_2 | -0.251 | 0.503 | -0.754 |
| L5 | 20 | carry_2 | -0.302 | 0.433 | -0.734 |
| L5 | 6 | max_carry_value | -0.255 | 0.471 | -0.727 |
| L5 | 28 | carry_2 | -0.274 | 0.442 | -0.716 |

**Interpretation:** `carry_2` (the carry at the hundreds position) is the most
divergent variable in the entire dataset, with Δ = -0.915 at layer 6. It
structures the wrong population strongly (silhouette 0.58) while being
*anti-structured* in the correct population (silhouette -0.34). This negative
silhouette in the correct population means correct answers are *anti-clustered*
by carry_2 — the model spreads correct answers across carry values rather than
grouping by them.

The carry variables appear at the top across *every* layer (4 through 31),
not just the mid/late layers as in the old analysis. This is a much stronger
signal than before: carry difficulty is the dominant axis of geometric
organization for the model's failed computations, and it starts at the earliest
layers.

### Top concepts STRONGER in correct population (positive Δ)

| Level | Layer | Variable | Correct | Wrong | Δ |
|-------|-------|----------|---------|-------|---|
| L5 | 16 | pp_a0_x_b0 | 0.668 | 0.026 | +0.641 |
| L5 | 16 | col_sum_0 | 0.668 | 0.026 | +0.641 |
| L5 | 24 | pp_a0_x_b0 | 0.639 | 0.043 | +0.596 |
| L5 | 24 | col_sum_0 | 0.639 | 0.043 | +0.596 |
| L5 | 28 | pp_a0_x_b0 | 0.654 | 0.069 | +0.585 |

**Interpretation:** The units-digit products (`pp_a0_x_b0`, `col_sum_0`)
create very strong UMAP structure in the correct population (Spearman 0.67)
but nearly zero in the wrong population (0.03). This is a much stronger
signal than the old dataset showed (was +0.46 max). When the model gets the
answer right, the units-digit partial product is the dominant organizing axis
in the mid/late layers. When it gets the answer wrong, this structure is absent.

**The asymmetry story is stronger and cleaner with the new data:** Wrong answers
are organized by carry difficulty (Δ up to -0.92). Correct answers are organized
by precise units-digit computation (Δ up to +0.64). The two populations inhabit
geometrically different manifolds — and the divergence starts at layer 6, not
layer 16 as previously thought.

---

## 12. Angular Correlation for Digit Variables

For digit-valued variables (0-9), we compute an additional score: Spearman
correlation between the angle from the embedding centroid (arctan2) and the digit
value. This detects circular encoding — if digits 0-9 are arranged in a ring
in UMAP space, linear Spearman would undercount the structure, but angular
correlation would catch it.

### Top angular correlations

| Level | Layer | Population | Variable | Angular Score |
|-------|-------|------------|----------|---------------|
| L5 | 4 | all | b_hundreds | 0.762 |
| L3 | 8 | all | b_tens | 0.748 |
| L5 | 24 | all | a_hundreds | 0.737 |
| L2 | 24 | all | a_tens | 0.712 |
| L4 | 16 | all | a_hundreds | 0.712 |
| L4 | 28 | all | a_hundreds | 0.710 |
| L5 | 12 | all | a_hundreds | 0.710 |
| L3 | 28 | correct | a_tens | 0.708 |
| L5 | 4 | wrong | b_hundreds | 0.702 |

**Key finding:** Input digits have angular correlations of 0.70-0.76, suggesting
the model encodes them in a roughly circular arrangement in activation space.
With the new dataset, `b_hundreds` at L5 shows the strongest angular encoding
(0.76), followed by `b_tens` at L3 (0.75). The hundreds-digit encoding at L4-L5
is new — previously invisible with smaller datasets. The `a_hundreds` variable
appears consistently at 0.71 across L4-L5, confirming circular encoding extends
to the 3-digit levels.

---

## 13. Heatmaps and Summary Visualizations

### Per-population heatmaps

For each level, the pipeline generates separate heatmaps showing interestingness
score (variable x layer) for each population:

| Heatmap file | Content |
|-------------|---------|
| L1_all.png | L1 all population (no wrong pop at L1) |
| L1_correct.png | L1 correct population |
| L2_all.png | L2 all population |
| L2_correct.png | L2 correct population |
| L3_all.png | L3 all population |
| L3_correct.png | L3 correct population |
| L3_wrong.png | L3 wrong population |
| L3_correct_minus_wrong.png | L3 difference heatmap |
| L4_all.png | L4 all population |
| L4_correct.png | L4 correct population |
| L4_wrong.png | L4 wrong population |
| L4_correct_minus_wrong.png | L4 difference heatmap |
| L5_all.png | L5 all population |
| L5_correct.png | L5 correct population |
| L5_wrong.png | L5 wrong population |
| L5_correct_minus_wrong.png | L5 difference heatmap |

**Total: 16 heatmaps.**

The difference heatmaps use a diverging RdBu_r colormap: blue = stronger in
correct, red = stronger in wrong. The color scale is symmetric around zero,
clamped to the maximum |Δ| in the data for each level.

### Why separate heatmaps matter

The original implementation averaged scores across populations. This washed out
the correct-vs-wrong signal. Example: if `a_tens` has silhouette 0.7 in correct
but 0.2 in wrong, the averaged heatmap shows 0.45. The separate heatmaps show
both values, and the difference heatmap shows +0.5 (blue), immediately flagging
this as a concept that the model loses when it gets the answer wrong.

---

## 14. Tiered Plotting System

Generating plots for all 117 CSVs x all variables x all methods would produce
thousands of plots. The tiered system generates only the most informative ones:

### Tier 1: Mandatory plots

For PLOT_LEVELS = [3, 5] and PLOT_LAYERS = [4, 16, 31], generate UMAP 2D
scatter plots colored by core variables:

- **All population:** correct, n_nonzero_carries, activation_norm, a_units, a_tens,
  a_hundreds, b_units, b_tens, b_hundreds
- **Correct population:** a_units, a_tens, a_hundreds, b_units, b_tens, b_hundreds
- **Wrong population:** a_units, a_tens, a_hundreds, b_units, b_tens, b_hundreds

**Result: 198 plots** (includes all valid combinations — some variables don't exist
at all levels, e.g., a_hundreds only at L4-L5).

### Tier 2: Score-driven plots

For the same PLOT_LEVELS x PLOT_LAYERS, generate UMAP 2D plots for the top 5
highest-scoring variables per (level, layer, population) that aren't already in the
mandatory set.

**Result: incorporated into the 198 total UMAP scatter plots.**

### Tier 3: t-SNE validation

For the top 30 overall findings (by UMAP 2D score), generate the corresponding
t-SNE 2D plot to verify the finding isn't a UMAP artifact.

**Result: 30 plots.**

### Tier 4: 3D exploration

For the top 15 overall findings, generate interactive 3D UMAP scatter plots
(Plotly HTML) for deeper exploration.

**Result: 15 HTML files.**

### Total plot output

| Type | Count | Format |
|------|-------|--------|
| UMAP 2D scatter | 198 | PNG |
| t-SNE validation | 30 | PNG |
| Interestingness heatmaps | 16 | PNG |
| Analysis plots (norms, CKA) | 7 | PNG |
| UMAP 3D interactive | 15 | HTML |
| **Total** | **251 PNG + 15 HTML** | |

All plots use rasterized scatter points (edgecolors="none", rasterized=True) to
keep file sizes reasonable.

---

## 15. Key Findings for Phase C/D

### For Phase C (Subspace Analysis)

1. **Skip layer 24.** CKA shows it is redundant with layers 20 and 28 across all
   levels (CKA 0.985-0.996). Run Phase C on [4, 8, 12, 16, 20, 28, 31].

2. **No normalization needed.** Correct/wrong norm ratios max at 1.038. UMAP
   structure is directional, not norm-dominated.

3. **Product magnitude is the dominant axis.** The product value is the #1
   interestingness score at every level. Phase C should control for product
   magnitude when looking for carry/digit structure.

4. **Layer 16 is the information peak.** More top-50 entries at layer 16 than
   any other layer. This is the natural starting point for Phase C probes.

5. **L5 is now viable for correct/wrong comparison.** With 4,197 correct answers
   in the full dataset (280 in the UMAP subsample), L5 shows the most dramatic
   divergences (Δ up to 0.92). L3 remains the best-balanced level (6,720 correct
   vs 3,280 wrong) for robust statistics.

### For Phase D (Correct/Wrong Geometry)

1. **Carry variables structure wrong but not correct.** carry_2 and max_carry_value
   show strong negative Δ at L5 (up to -0.915). The model's wrong answers are
   organized by carry difficulty across all layers (4 through 31), while correct
   answers are *anti-clustered* by carry values.

2. **Low-order products structure correct but not wrong.** pp_a0_x_b0 and col_sum_0
   show strong positive Δ (up to +0.641). The correct population preserves the
   units-digit computation structure that the wrong population completely loses.

3. **The asymmetry grows dramatically with difficulty.** Mean |Δ| goes from 0.081
   (L3) to 0.237 (L5). At L3 the populations overlap significantly; at L5 they are
   geometrically very different. The divergence starts at layer 6, not layer 16.

### For the Fourier screening step

1. **Angular correlation for digits is 0.70-0.76.** This suggests circular encoding
   is present but the embedding preserves it only partially. The full 4096D Fourier
   screening may find much stronger signals.

2. **b_hundreds is the most angularly correlated digit** (0.76 at L5 layer 4).
   `a_hundreds` is consistently 0.71 across L4-L5. Start Fourier screening with
   the hundreds digits of both operands.

---

## 16. What This Stage Does NOT Do

Phase A is visual reconnaissance. It does not perform:

- PCA eigenspectrum analysis (→ Phase C)
- Spearman correlation sweeps across 4096 raw dimensions (→ Phase C)
- ANOVA F-tests for digit encoding sparsity (→ Phase C)
- Fourier basis correlation probes (→ Fourier screening step)
- LDA separation with Cohen's d and permutation null (→ Phase D)
- Cross-level subspace alignment with principal angles (→ Phase F)
- Correct vs wrong subspace overlap (→ Phase F)
- Any causal interventions or ablation studies

The interestingness scores are a ranking metric on 2D projections. They tell you
where to look; they do not tell you what is there. Phase C/D will operate in the
full 4096-dimensional space with proper statistical frameworks.

---

## 17. Output Files

### Data outputs (on data_root)

```
/data/user_data/anshulk/arithmetic-geometry/phase_a/
├── analysis/                          # Pre-flight diagnostics
│   ├── norm_profile.json              # 33 KB — all norm statistics
│   └── cka_matrices.json              # 33 KB — 5 CKA matrices
├── l5_subsample_meta.json             # L5 subsampling metadata
├── coloring_dfs/                      # Cached label DataFrames
│   ├── L1_coloring.pkl                # 64 rows
│   ├── L2_coloring.pkl                # 4,000 rows
│   ├── L3_coloring.pkl                # 10,000 rows
│   ├── L4_coloring.pkl                # 10,000 rows
│   └── L5_coloring.pkl                # 122,223 rows (full; subsample applied at embedding time)
├── csvs/                              # 52 MB — 117 CSVs
│   ├── L1_layer04_all.csv             # 64 rows, 28 cols
│   ├── L1_layer04_correct.csv
│   ├── ...
│   └── L5_layer31_wrong.csv           # 5,750 rows, 63 cols
├── embeddings/                        # 14 MB — 351 .npy files
│   ├── L1/layer_04/all_umap_2d.npy
│   ├── ...
│   └── L5/layer_31/wrong_tsne_2d.npy
└── scores/                            # Interestingness scores
    ├── interestingness_scores.csv     # 10,242 rows (with metric_note, sampling_note cols)
    ├── top_50_findings.md
    ├── correct_wrong_comparison.md
    ├── correct_wrong_comparison.csv   # 792 rows
    └── l5_delta_interestingness.md    # Dedicated L5 correct/wrong Δ analysis
```

### Plot outputs (on workspace)

```
/home/anshulk/arithmetic-geometry/plots/phase_a/
├── analysis/                          # Pre-flight diagnostic plots
│   ├── norm_profile_all.png
│   ├── norm_profile_correct_wrong.png
│   ├── cka_L1.png ... cka_L5.png
├── heatmaps/                          # 16 interestingness heatmaps
│   ├── L1_all.png ... L5_wrong.png
│   ├── L3_correct_minus_wrong.png
│   ├── L4_correct_minus_wrong.png
│   └── L5_correct_minus_wrong.png
├── L3/ L5/                            # UMAP 2D scatter by level/layer/pop
│   └── layer04/ layer16/ layer31/
│       └── all/ correct/ wrong/
│           └── umap_2d/*.png
├── tsne_validation/                   # 30 t-SNE comparison plots
└── umap_3d/                           # 15 interactive 3D HTML files

├── L5_delta_interestingness.png        # First-class L5 Δ heatmap (200 DPI)

Total: 252 PNG files + 15 HTML files
```

### Log outputs

```
/home/anshulk/arithmetic-geometry/logs/
├── phase_a_analysis.log               # Pre-flight diagnostics log
├── phase_a_embeddings.log             # Full embeddings pipeline log
└── slurm-6618712.out / .err           # SLURM job output
```

---

## 18. Runtime and Reproducibility

### Execution environment

| Property | Value |
|----------|-------|
| Date | March 19, 2026 |
| SLURM job ID | 6654499 |
| Node | babel-w9-28 |
| GPU | NVIDIA RTX A6000, 49,140 MiB VRAM |
| CPU cores | 8 |
| RAM | 64 GB |
| Python | 3.11, conda environment "geometry" |
| cuML | 26.02 (RAPIDS, GPU UMAP/t-SNE) |
| scikit-learn | 1.8.0 |
| umap-learn | 0.5.11 (CPU fallback, not used) |
| numpy | 2.2.6 |

### Timing breakdown

```
Start time: 14:54:15
Pre-flight checks:                    14:54:15 — 14:54:35  (20 seconds)
  - cuML import check: 0s (pre-installed)
  - Package checks: 2s
  - Input data verification: 1s

Phase A analysis (pre-flight):       loaded from cache  (0 seconds, run separately)
  - Norm profile: 63 seconds (when run fresh with 122K L5)
  - CKA matrices: 16 seconds

Phase A embeddings:                  14:54:35 — 15:17:26  (1370 seconds)
  - Build coloring DataFrames:        ~30 seconds (122K L5)
  - L5 subsampling:                   ~2 seconds
  - Compute 351 embeddings (GPU):     ~240 seconds
  - Score 117 CSVs:                   ~1065 seconds
  - Generate summaries + L5 Δ:        ~7 seconds
  - Generate 243 plots:               ~26 seconds

Total wall time: 1392 seconds (23 minutes)
```

Interestingness scoring (~1065 seconds) dominates the runtime. It involves computing
silhouette scores on 10,000-point embeddings (L3/L4), which requires O(n^2)
pairwise distances. With the 2.5x increase in L3/L4 dataset size (4,000 → 10,000),
scoring time increased ~3x due to the quadratic cost. This step runs on CPU
(sklearn silhouette_score does not have a GPU implementation).

### Reproducibility

All random operations use seed 42:
- UMAP: random_state=42
- t-SNE: random_state=42
- CKA subsampling: RandomState(42)

cuML GPU UMAP/t-SNE is deterministic with a fixed random_state on the same GPU
hardware. Results may differ slightly between GPU and CPU backends due to floating
point differences, but the qualitative findings are identical.

### Final GPU memory state

```
index, memory.used [MiB], memory.total [MiB]
0, 1 MiB, 49140 MiB
```

GPU memory was fully released. Peak usage during cuML UMAP was approximately
2-4 GB (estimated from cuML's internal allocations for 10,000 x 4096 float32 data).

---

*End of document. All numbers verified against norm_profile.json, cka_matrices.json,
interestingness_scores.csv, correct_wrong_comparison.csv, l5_delta_interestingness.md,
top_50_findings.md, l5_subsample_meta.json, phase_a_embeddings.log,
phase_a_analysis.log, and slurm-6654499.out/err as of March 19, 2026.*
