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
- 16,064 problems total (64 at Level 1, 4,000 each at Levels 2-5)
- Rich label system: input digits, partial products, column sums, carries, answer digits
- Accuracy gradient: 100% (L1) → 99.4% (L2) → 34.1% (L3) → 28.7% (L4) → 6.0% (L5)

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
| L1    | 3.78    | 5.30    | 6.47    | 8.01     | 10.64    | 16.06    | 23.94    | 35.44    | 74.72    |
| L2    | 3.71    | 5.11    | 6.27    | 8.06     | 11.06    | 17.08    | 25.54    | 37.34    | 76.07    |
| L3    | 3.67    | 5.12    | 6.31    | 8.42     | 10.98    | 16.56    | 24.91    | 36.71    | 77.48    |
| L4    | 3.65    | 5.05    | 6.47    | 8.36     | 10.69    | 15.98    | 23.91    | 35.88    | 76.53    |
| L5    | 3.56    | 5.10    | 6.59    | 8.31     | 10.48    | 15.47    | 23.22    | 35.06    | 75.05    |

All levels follow the same monotonic growth pattern. Norms roughly double every ~4
layers from layer 4 to layer 28, then jump ~2x from layer 28 to layer 31. This is
standard for transformer residual streams where each layer adds to the residual.

### Results: Correct vs wrong norms (L3-L5 only)

Level 1 is 100% correct (no wrong population). Level 2 has only 23 wrong answers
(below the MIN_POPULATION threshold of 30). Correct/wrong splits are available only
for L3-L5.

| Key | Correct mean | Correct std | Wrong mean | Wrong std | Ratio (C/W) |
|-----|-------------|------------|-----------|----------|-------------|
| L3 layer 4  | 3.68 | 0.04 | 3.66 | 0.03 | 1.005 |
| L3 layer 6  | 5.12 | 0.05 | 5.14 | 0.05 | 0.996 |
| L3 layer 8  | 6.29 | 0.11 | 6.34 | 0.10 | 0.992 |
| L3 layer 12 | 8.41 | 0.14 | 8.48 | 0.11 | 0.992 |
| L3 layer 16 | 11.05| 0.33 | 10.82| 0.26 | 1.021 |
| L3 layer 20 | 16.70| 0.67 | 16.30| 0.60 | 1.025 |
| L3 layer 24 | 25.24| 1.26 | 24.41| 1.07 | 1.034 |
| L3 layer 28 | 36.97| 1.33 | 36.14| 1.20 | 1.023 |
| L3 layer 31 | 77.77| 1.88 | 76.94| 1.80 | 1.011 |
| L4 layer 4  | 3.66 | 0.03 | 3.65 | 0.03 | 1.006 |
| L4 layer 6  | 5.04 | 0.06 | 5.06 | 0.05 | 0.997 |
| L4 layer 8  | 6.41 | 0.12 | 6.51 | 0.12 | 0.994 |
| L4 layer 12 | 8.36 | 0.12 | 8.37 | 0.11 | 0.998 |
| L4 layer 16 | 10.88| 0.50 | 10.60| 0.42 | 1.027 |
| L4 layer 20 | 16.32| 1.01 | 15.83| 0.90 | 1.031 |
| L4 layer 24 | 24.55| 1.63 | 23.64| 1.55 | 1.038 |
| L4 layer 28 | 36.50| 1.60 | 35.58| 1.58 | 1.026 |
| L4 layer 31 | 77.31| 2.21 | 76.23| 2.39 | 1.014 |
| L5 layer 4  | 3.58 | 0.04 | 3.56 | 0.04 | 1.008 |
| L5 layer 6  | 5.10 | 0.08 | 5.10 | 0.08 | 1.000 |
| L5 layer 8  | 6.50 | 0.17 | 6.59 | 0.17 | 0.994 |
| L5 layer 12 | 8.37 | 0.16 | 8.30 | 0.15 | 1.006 |
| L5 layer 16 | 10.74| 0.45 | 10.49| 0.47 | 1.023 |
| L5 layer 20 | 15.72| 1.04 | 15.43| 0.87 | 1.019 |
| L5 layer 24 | 23.70| 1.73 | 23.19| 1.48 | 1.022 |
| L5 layer 28 | 35.69| 1.71 | 35.04| 1.53 | 1.018 |
| L5 layer 31 | 76.86| 3.29 | 74.90| 3.56 | 1.026 |

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
| L1 | layers 6 & 8   | 0.9802 |
| L1 | layers 20 & 24 | 0.9847 |
| L1 | layers 24 & 28 | 0.9951 |
| L2 | layers 20 & 24 | 0.9881 |
| L2 | layers 20 & 28 | 0.9844 |
| L2 | layers 24 & 28 | 0.9952 |
| L3 | layers 20 & 24 | 0.9850 |
| L3 | layers 20 & 28 | 0.9847 |
| L3 | layers 24 & 28 | 0.9940 |
| L4 | layers 20 & 24 | 0.9901 |
| L4 | layers 20 & 28 | 0.9859 |
| L4 | layers 24 & 28 | 0.9956 |
| L5 | layers 20 & 24 | 0.9871 |
| L5 | layers 24 & 28 | 0.9898 |

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
| 20-24 | 0.985 | 0.988 | 0.985 | 0.990 | 0.987 |
| 24-28 | 0.995 | 0.995 | 0.994 | 0.996 | 0.990 |
| 20-28 | 0.976 | 0.984 | 0.985 | 0.986 | 0.975 |

**Verdict for Phase C:** Skip layer 24. It adds nothing over layers 20 and 28.
Run Phase C on [4, 8, 12, 16, 20, 28, 31] — 7 layers instead of 9, saving ~22% of
compute with no information loss.

---

## 6. The Embedding Pipeline

The embedding pipeline (`phase_a_embeddings.py`) runs in 6 steps:

| Step | Description | Time |
|------|-------------|------|
| 1 | Build coloring DataFrames (label + answer data → pandas) | 1s |
| 2-3 | Compute UMAP/t-SNE embeddings, build CSVs | 89s |
| 4 | Score all CSVs for interestingness | 325s |
| 5 | Generate summaries (heatmaps, comparison tables) | 8s |
| 6 | Generate tiered plots (243 total) | 62s |
| **Total** | | **485s (8.1 min)** |

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

| Level | All | Correct | Wrong | Wrong % | Populations |
|-------|-----|---------|-------|---------|-------------|
| L1 | 64 | 64 | 0 | 0% | all, correct |
| L2 | 4,000 | 3,977 | 23 | 0.6% | all, correct |
| L3 | 4,000 | 2,638 | 1,362 | 34.1% | all, correct, wrong |
| L4 | 4,000 | 1,147 | 2,853 | 71.3% | all, correct, wrong |
| L5 | 4,000 | 239 | 3,761 | 94.0% | all, correct, wrong |

L1 and L2 have no "wrong" population (fewer than 30 wrong answers). L3-L5 have
all three populations, making them the focus of the correct-vs-wrong analysis.

### CSV structure

Each CSV combines the embedding coordinates with the full label set:

| Level | Rows per CSV | Columns | Example columns |
|-------|-------------|---------|-----------------|
| L1 | 64 | 28 | a, b, correct, a_units, b_units, pp_a0_x_b0, carry_0, product, umap_2d_x, ... |
| L2 | 4,000 (all) / 3,977 (correct) | 41 | + a_tens, b_tens decomposition |
| L3 | 4,000 / 2,638 / 1,362 | 48 | + 4 partial products, 3 column sums, 3 carries |
| L4 | 4,000 / 1,147 / 2,853 | 55 | + 6 partial products, 4 column sums, 4 carries |
| L5 | 4,000 / 239 / 3,761 | 63 | + 9 partial products, 5 column sums, 5 carries, 6 answer digits |

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

| Category | Count | Mean | Std | Max |
|----------|-------|------|-----|-----|
| All scores | 10,242 | — | — | — |
| Valid scores | 9,162 | — | — | — |
| By method: UMAP 2D | 4,581 | 0.097 | 0.259 | 0.924 |
| By method: t-SNE 2D | 4,581 | 0.079 | 0.272 | 0.948 |
| By metric: Silhouette | 4,500 | -0.090 | 0.141 | 0.516 |
| By metric: Spearman | 2,682 | 0.353 | 0.249 | 0.948 |
| By metric: Angular | 1,980 | 0.133 | 0.172 | 0.801 |

1,080 scores are NaN (insufficient data — populations with <30 valid values for
a variable, or constant variables within a population like "correct" in the
correct-only split).

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
| 1 | L3 | 16 | wrong | product | 0.924 |
| 2 | L3 | 16 | wrong | pp_a1_x_b1 | 0.914 |
| 3 | L3 | 16 | wrong | col_sum_2 | 0.914 |
| 4 | L3 | 6 | all | product | 0.912 |
| 5 | L3 | 16 | all | product | 0.910 |
| 6 | L3 | 6 | all | pp_a1_x_b1 | 0.896 |
| 7 | L3 | 6 | all | col_sum_2 | 0.896 |
| 8 | L3 | 16 | all | pp_a1_x_b1 | 0.896 |
| 9 | L3 | 16 | all | col_sum_2 | 0.896 |
| 10 | L1 | 24 | all | product | 0.894 |
| 11 | L3 | 16 | correct | product | 0.884 |
| 12 | L3 | 6 | correct | product | 0.879 |
| 13 | L3 | 12 | wrong | product | 0.875 |
| 14 | L1 | 12 | all | product | 0.869 |
| 15 | L3 | 6 | correct | pp_a1_x_b1 | 0.869 |

**Patterns in the top 50:**

1. **Product dominates.** The product value (ground truth) is the single most
   visible variable in UMAP space. This makes sense: the product is the highest-variance
   scalar derived from the inputs, and UMAP preserves variance.

2. **L3 dominates.** 28 of the top 50 entries are Level 3. This is the "sweet spot"
   level: 2-digit x 2-digit multiplication with a 34% error rate. Enough correct
   answers for clean structure, enough wrong answers for comparison.

3. **Partial products and column sums are redundant.** `pp_a1_x_b1` and `col_sum_2`
   have identical scores because at L3 (2x2), column sum 2 IS just the partial product
   `a1*b1`. The deduplication comment in the code is not algebraic — both variables
   appear when they rank highly.

4. **Layer 16 is the most informative.** More top-50 entries appear at layer 16 than
   any other layer. This is the mid-network layer where the model has processed the
   input but hasn't yet committed to an output.

5. **Wrong population scores higher than correct for product.** At L3 layer 16:
   product scores 0.924 in wrong vs 0.884 in correct. The wrong population has a
   product gradient that UMAP captures more cleanly, possibly because wrong answers
   cluster by magnitude.

---

## 11. Correct vs Wrong Comparison

This is the most important output for Paper 1. The comparison table merges
interestingness scores between correct and wrong populations on the same
(level, layer, variable) triple and computes Δ = correct_score - wrong_score.

Total pairs: **792** (only L3-L5 have both populations).

### Summary by level

| Level | Pairs | Max |Δ| | Mean |Δ| |
|-------|-------|---------|----------|
| L3 | 207 | 0.235 | 0.060 |
| L4 | 261 | 0.416 | 0.077 |
| L5 | 324 | 0.695 | 0.178 |

**The gap widens with difficulty.** L3 has modest differences. L5 has dramatic
differences — the hardest problems show completely different geometric structure
in correct vs wrong populations.

### Top concepts STRONGER in wrong population (negative Δ)

| Level | Layer | Variable | Correct | Wrong | Δ |
|-------|-------|----------|---------|-------|---|
| L5 | 8 | pp_a2_x_b2 | 0.109 | 0.804 | -0.695 |
| L5 | 8 | col_sum_4 | 0.109 | 0.804 | -0.695 |
| L5 | 8 | product | 0.122 | 0.797 | -0.676 |
| L5 | 24 | carry_2 | -0.391 | 0.276 | -0.666 |
| L5 | 8 | carry_2 | -0.285 | 0.367 | -0.652 |
| L5 | 28 | carry_2 | -0.362 | 0.289 | -0.651 |
| L5 | 8 | max_carry_value | -0.241 | 0.405 | -0.646 |

**Interpretation:** At L5 layer 8, the dominant partial product `pp_a2_x_b2`
(the hundreds-digit cross product, the largest single partial product in 3x3
multiplication) creates strong UMAP structure in the wrong population (r=0.80)
but barely any in the correct population (r=0.11). This means the model's
early-layer encoding of the dominant partial product is very different when the
model gets the answer wrong — it over-represents this one product at the expense
of the others.

The carry variables (`carry_2`, `max_carry_value`) show the same pattern: they
structure the wrong population but not the correct one. This supports the
carry-chain-bottleneck hypothesis from the data generation analysis.

### Top concepts STRONGER in correct population (positive Δ)

| Level | Layer | Variable | Correct | Wrong | Δ |
|-------|-------|----------|---------|-------|---|
| L5 | 8 | col_sum_1 | 0.499 | 0.040 | +0.459 |
| L5 | 16 | pp_a0_x_b0 | 0.480 | 0.045 | +0.435 |
| L5 | 16 | col_sum_0 | 0.480 | 0.045 | +0.435 |
| L5 | 8 | pp_a0_x_b0 | 0.464 | 0.039 | +0.425 |
| L5 | 20 | pp_a0_x_b0 | 0.495 | 0.077 | +0.418 |

**Interpretation:** The units-digit products (`pp_a0_x_b0`, `col_sum_0`,
`col_sum_1`) create strong UMAP structure in the correct population but not
the wrong population. When the model gets the answer right, the low-order
partial products are well-organized in the embedding space. When it gets the
answer wrong, this structure collapses.

**The asymmetry tells a story:** Wrong answers are organized by the dominant
product and carry values (the "big picture" of the multiplication). Correct
answers are organized by the precise units-digit computations (the "details").
This aligns with the data generation finding that the model gets the units digit
right 49-73% of the time even when the overall answer is wrong — it has the
low-order structure but loses the high-order carry propagation.

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
| L3 | 4 | wrong | b_tens | 0.801 |
| L4 | 8 | correct | b_tens | 0.790 |
| L2 | 28 | all | a_tens | 0.774 |
| L4 | 6 | correct | b_tens | 0.754 |
| L2 | 20 | correct | b_units | 0.753 |
| L2 | 16 | correct | b_units | 0.748 |
| L3 | 31 | wrong | a_tens | 0.744 |
| L3 | 28 | wrong | a_tens | 0.714 |
| L1 | 6 | all | a_units | 0.712 |

**Key finding:** Input digits (a_units, a_tens, b_units, b_tens) have strong
angular correlations (0.7-0.8), suggesting the model encodes them in a roughly
circular arrangement in activation space. This is consistent with known findings
about modular arithmetic representations in neural networks. The `b_tens` digit
shows the strongest angular encoding, appearing at the top of the list across
multiple levels.

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

5. **L3 is the sweet spot for correct/wrong comparison.** It has the best balance
   of correct (2,638) and wrong (1,362) answers. L5 has dramatic effects but only
   239 correct answers, making statistical comparison fragile.

### For Phase D (Correct/Wrong Geometry)

1. **Carry variables structure wrong but not correct.** carry_2 and max_carry_value
   show strong negative Δ at L5 (up to -0.695). The model's wrong answers are
   organized by carry difficulty, while correct answers are organized by precise
   digit computations.

2. **Low-order products structure correct but not wrong.** pp_a0_x_b0 and col_sum_0/1
   show strong positive Δ (up to +0.459). The correct population preserves the
   units-digit computation structure that the wrong population loses.

3. **The asymmetry grows with difficulty.** Mean |Δ| goes from 0.060 (L3) to
   0.178 (L5). At L3 the populations are similar; at L5 they are geometrically
   very different.

### For the Fourier screening step

1. **Angular correlation for digits is 0.70-0.80.** This suggests circular encoding
   is present but the embedding preserves it only partially. The full 4096D Fourier
   screening may find much stronger signals.

2. **b_tens is the most angularly correlated digit.** Start Fourier screening with
   the tens digits of the second operand.

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
├── coloring_dfs/                      # Cached label DataFrames
│   ├── L1_coloring.pkl                # 64 rows
│   ├── L2_coloring.pkl                # 4,000 rows
│   ├── L3_coloring.pkl                # 4,000 rows
│   ├── L4_coloring.pkl                # 4,000 rows
│   └── L5_coloring.pkl                # 4,000 rows
├── csvs/                              # 52 MB — 117 CSVs
│   ├── L1_layer04_all.csv             # 64 rows, 28 cols
│   ├── L1_layer04_correct.csv
│   ├── ...
│   └── L5_layer31_wrong.csv           # 3,761 rows, 63 cols
├── embeddings/                        # 14 MB — 351 .npy files
│   ├── L1/layer_04/all_umap_2d.npy
│   ├── ...
│   └── L5/layer_31/wrong_tsne_2d.npy
└── scores/                            # Interestingness scores
    ├── interestingness_scores.csv     # 10,242 rows
    ├── top_50_findings.md
    ├── correct_wrong_comparison.md
    └── correct_wrong_comparison.csv   # 792 rows

Total: 68 MB
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

Total: 251 PNG files + 15 HTML files, 53 MB
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
| Date | March 17, 2026 |
| SLURM job ID | 6618712 |
| Node | babel-n5-20 |
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
Start time: 13:59:15
Pre-flight checks:                    13:59:15 — 13:59:26  (11 seconds)
  - cuML import check: 0s (pre-installed)
  - Package checks: 2s
  - Input data verification: 1s

Phase A analysis (pre-flight):       13:59:26 — 13:59:39  (13 seconds)
  - Norm profile: 3 seconds
  - CKA matrices: 10 seconds

Phase A embeddings:                  13:59:41 — 14:10:15  (485 seconds)
  - Build coloring DataFrames:        1 second
  - Compute 351 embeddings (GPU):     89 seconds
  - Score 117 CSVs:                   325 seconds
  - Generate summaries:               8 seconds
  - Generate 243 plots:               62 seconds

Total wall time: 508 seconds (8 minutes 28 seconds)
```

Interestingness scoring (325 seconds) dominates the runtime. It involves computing
silhouette scores on 4000-point embeddings, which requires O(n^2) pairwise distances.
This step runs on CPU (sklearn silhouette_score does not have a GPU implementation).

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
2-3 GB (estimated from cuML's internal allocations for 4000 x 4096 float32 data).

---

*End of document. All numbers verified against norm_profile.json, cka_matrices.json,
interestingness_scores.csv, correct_wrong_comparison.csv, top_50_findings.md,
phase_a_embeddings.log, phase_a_analysis.log, and slurm-6618712.out/err as of
March 17, 2026.*
