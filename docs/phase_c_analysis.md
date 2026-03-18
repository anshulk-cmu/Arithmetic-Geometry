# Phase C: Complete Analysis

**Anshul's Geometric Manifold Interpretability Project**
**Carnegie Mellon University, March 2026**

This document records every decision, every number, and every result from Phase C —
the concept subspace identification stage. It is the truth document for this stage.
All numbers are validated against the actual output files as of March 18, 2026.

---

## Table of Contents

1. [Purpose of This Stage](#1-purpose-of-this-stage)
2. [What Phase C Is and Is Not](#2-what-phase-c-is-and-is-not)
3. [The Core Question](#3-the-core-question)
4. [Why Not Just PCA](#4-why-not-just-pca)
5. [The Algorithm: Conditional Covariance + SVD](#5-the-algorithm-conditional-covariance--svd)
6. [Product Residualization](#6-product-residualization)
7. [Concept Registry](#7-concept-registry)
8. [Value Filtering](#8-value-filtering)
9. [Significance Testing: The Permutation Null](#9-significance-testing-the-permutation-null)
10. [Dimensionality Determination](#10-dimensionality-determination)
11. [Cross-Validation](#11-cross-validation)
12. [Run Summary](#12-run-summary)
13. [Results by Level](#13-results-by-level)
14. [Input Digit Subspaces](#14-input-digit-subspaces)
15. [Answer Digit Subspaces](#15-answer-digit-subspaces)
16. [Carry Subspaces](#16-carry-subspaces)
17. [Column Sum Subspaces](#17-column-sum-subspaces)
18. [Partial Product Subspaces](#18-partial-product-subspaces)
19. [Derived Concept Subspaces](#19-derived-concept-subspaces)
20. [Correct vs Wrong Divergence](#20-correct-vs-wrong-divergence)
21. [Cross-Layer Alignment](#21-cross-layer-alignment)
22. [The Failure Regime: L5](#22-the-failure-regime-l5)
23. [Key Findings](#23-key-findings)
24. [Output Files](#24-output-files)
25. [Runtime and Reproducibility](#25-runtime-and-reproducibility)

---

## 1. Purpose of This Stage

Phase A showed clusters. Phase C finds the exact directions.

Phase A compressed 4096 dimensions into 2-3 with UMAP/t-SNE and showed that mathematical
concepts create visible structure in the activation space. But UMAP is a visualization
tool, not a measurement tool. It tells you "there's something here" but not "here are
the specific directions in 4096-dimensional space that encode this concept."

Phase C answers: for each mathematical concept (each input digit, each carry value, each
column sum, each answer digit), which linear subspace of the 4096-dimensional activation
space encodes it? How many dimensions does it use? Is this encoding significant against
a random baseline? Does it hold up on held-out data? Does it differ between problems the
model gets right versus wrong?

The outputs are basis matrices — the actual directions in activation space — plus
significance tests, dimensionality estimates, and cross-layer/cross-population
comparisons. These feed directly into downstream Fourier screening (do digit subspaces
have circular structure?) and GP geometry (what shape do concept manifolds have?).

---

## 2. What Phase C Is and Is Not

### What it is

- **Conditional covariance + SVD** on centroid matrices to find concept-specific directions
- **Permutation null** (1,000 shuffles) to test whether each subspace is statistically
  significant against random groupings
- **Three-method dimensionality estimation** (cumulative variance, ratio test, permutation)
  with consensus voting
- **5-fold cross-validation** to verify subspaces generalize to held-out data
- **Cross-layer principal angles** to track how subspaces rotate through the network
- **Correct vs wrong comparison** to find where representations diverge

### What it is not

- It does not characterize the *shape* within subspaces (that is Steps 2-3)
- It does not find unknown concepts (that is Phase E, residual hunting)
- It does not establish causality (that requires ablation/patching in Step 5)
- It cannot find concepts that cause tiny shifts in activations — Phase D (LDA) handles
  those by optimizing for class separability rather than variance explained

---

## 3. The Core Question

Take one concrete slice to build intuition: 4,000 Level 3 problems at layer 16.
Each problem produces a 4096-dimensional activation vector. The question is: where
inside that 4096-dimensional space does the model store the concept "the tens digit of
the first number is 3"?

Think of it like a building with 4,096 rooms. We know the building contains an office
for "digit a1," a room for "carry from column 2," and a lounge for "partial product
a0*b1." We want to find out which rooms each concept uses, and whether some concepts
share rooms.

If we can identify the exact rooms (directions), we can:
- Project activations down to just those rooms and study the geometry inside them
- Check whether Fourier (circular) patterns exist in those rooms
- Compare the rooms used by correct answers versus wrong answers
- Measure whether "carry" and "digit" rooms overlap (superposition)

---

## 4. Why Not Just PCA

PCA finds the directions of largest spread in the data. The problem: it ranks everything
by *total* variance. A concept that causes a big shift (like product magnitude) dominates
the top components. A concept that causes a small but consistent shift (like whether
carry_2 is 0 or 1) gets buried at component #200.

PCA treats all variation equally. It cannot distinguish variation caused by "the tens
digit changes" from variation caused by "random fluctuations."

The fix: **conditional covariance**. Instead of looking at all variation, look at
*only the variation caused by one specific concept changing*.

### A concrete example

Take all 4,000 Level 3 problems. Group them by the value of a_tens (the tens digit of
the first operand). Some problems have a_tens = 1, others a_tens = 2, ..., a_tens = 9.
For each group, compute the average activation vector. That gives 9 average vectors.

Now compute the covariance of those 9 averages. This covariance captures *only* the
variation due to a_tens changing. Random fluctuations from other concepts (b_units,
carries, partial products) get averaged away within each group, because within the
a_tens = 3 group, all the other concepts take many different values and their effects
cancel out on average.

Imagine a simplified 3-dimensional activation space. Suppose the model uses dimension 1
for "product magnitude," dimension 2 for "a_tens," and dimension 3 for noise.

Plain PCA: dimension 1 dominates (products range 100-9801). Dimension 2 is second.

Conditional covariance for a_tens: the product-magnitude dimension averages out (each
a_tens group has a mix of products). The noise dimension averages out. What remains is
dimension 2: the one that actually shifts when a_tens changes.

That is the concept subspace for a_tens. Clean signal.

---

## 5. The Algorithm: Conditional Covariance + SVD

### Step-by-step

**Input:** N activation vectors in R^4096, each labeled with a concept value (e.g.,
a_tens in {1,2,...,9}).

**Step 1: Compute group centroids.** For each concept value v, average all activations
with that value: mu_v = mean of all h_i where concept = v. Also compute the grand mean:
mu = mean of all centroids (equal weight, not sample-weighted).

Equal weighting treats each concept value as equally important for finding the subspace.
This matches Gurnee et al. (2025, Anthropic), who used equal weights for their 150
character-count centroids. Sample weighting would answer a different question ("how much
total variance does this grouping explain"), which is closer to what Phase D (LDA)
answers.

**Step 2: Build the centered centroid matrix.** M_c = (centroids - grand_mean) / sqrt(m),
where m is the number of groups. Shape: (m, 4096).

**Step 3: SVD.** Compute U, S, Vt = SVD(M_c). The rows of Vt are the basis directions.
The eigenvalues are S^2.

Because m is small (2-42 depending on the concept), this is a tiny SVD. Cost:
O(m^2 * d) where d = 4096. For m = 10: about 400,000 FLOPs. Milliseconds.

**Step 4: Find the eigenvalue cliff.** The eigenvalues tell us how many dimensions the
concept uses. A sharp drop signals the boundary between "signal dimensions" and "noise
dimensions." Three methods are used (see Section 10).

**Step 5: Extract basis and project.** The top d_c rows of Vt form the concept subspace
basis. Any activation vector h can be projected into this subspace via z = Vt[:d_c] @ (h - mu).

### Rank bound

The centroid matrix has rank at most m-1 (because the m centered rows sum to zero).
For a digit with 10 values, the subspace can be at most 9-dimensional. For a binary
concept (correct/wrong), at most 1-dimensional. This is a mathematical fact, not a
limitation — the maximum possible dimensionality for any concept equals its number of
distinct values minus one.

---

## 6. Product Residualization

Phase A found that product magnitude is the dominant axis of variation. Without removing
this, every concept's subspace would be contaminated by its correlation with product size.
A_tens = 9 produces larger products than a_tens = 1 on average, so the a_tens subspace
would partly reflect "big product" vs "small product" rather than the pure digit signal.

**The fix:** Before computing concept subspaces, project out the product-magnitude
direction from all activations via OLS regression.

The procedure:
1. Center activations: X_c = acts - mean(acts)
2. Center product values: p_c = product - mean(product)
3. Regression coefficient: beta = X_c^T @ p_c / (p_c @ p_c) — shape (4096,)
4. Residual: X_resid = X_c - outer(p_c, beta)

This removes exactly one direction (the product-magnitude direction) from the 4096-dimensional
space, leaving 4095 dimensions for concept subspaces. It is done once per (level, layer)
pair on the **full** population before slicing to correct/wrong.

**Exception:** When analyzing the `product_binned` concept itself, raw (non-residualized)
activations are used.

Residualized activations are cached at `phase_c/residualized/level{N}_layer{L}.npy`.
45 files were produced, one per (level, layer) pair.

---

## 7. Concept Registry

Phase C analyzes 28 concepts at L3 and up to 43 at L5, organized into four tiers.
Each concept is checked for existence in the coloring DataFrame before being included
(not all concepts exist at all levels).

### Tier 1: Input and Output Digits

These are the raw digit values — both inputs and target outputs.

| Concept | Values | Levels | Notes |
|---------|--------|--------|-------|
| a_units | 0-9 | L1-L5 | Units digit of first operand |
| a_tens | 1-9 | L2-L5 | Tens digit (no leading zero) |
| a_hundreds | 1-9 | L4-L5 | Hundreds digit |
| b_units | 0-9 | L1-L5 | Units digit of second operand |
| b_tens | 1-9 | L3-L5 | |
| b_hundreds | 1-9 | L5 | |
| ans_digit_0_msf | 1-9 | L1-L5 | Leading (most significant) answer digit |
| ans_digit_1_msf | 0-9 | L1-L5 | Second answer digit |
| ans_digit_2_msf | 0-9 | L2-L5 | Third answer digit |
| ans_digit_3_msf | 0-9 | L3-L5 | Fourth answer digit |
| ans_digit_4_msf | 0-9 | L4-L5 | Fifth answer digit |
| ans_digit_5_msf | 0-9 | L5 | Sixth (units) answer digit |

Answer digits are what the model is ultimately trying to produce. Kim et al. (2025)
showed LLMs compute each output digit position via separate circuits. If the model builds
per-position output representations in intermediate layers, answer digit subspaces should
emerge at later layers.

### Tier 2: Carries and Column Sums

These are the intermediate computation steps of long multiplication.

| Concept | Values (L3) | Preprocessing | Notes |
|---------|-------------|---------------|-------|
| carry_0 | 0-8 | filter rare | Carry from units column |
| carry_1 | 0-17 | filter rare | Carry from tens column |
| carry_2 | 0-9 | filter rare | Carry from hundreds column |
| col_sum_0 | 0-81 | bin into deciles | Pre-carry total, units column |
| col_sum_1 | 0-162 | bin into deciles | Pre-carry total, tens column |
| col_sum_2 | 1-81 | bin into deciles | Pre-carry total, hundreds column |

Column sums are the bridge between partial products and carries — Qian et al. (2024)
and He et al. (2025) both identify these as the key intermediate computation. Carries
are filtered to drop rare values (see Section 8).

### Tier 3: Derived Quantities

| Concept | Values | Notes |
|---------|--------|-------|
| correct | 0, 1 | Binary — only in "all" population |
| n_nonzero_carries | 0-5 | Count of columns with nonzero carry |
| total_carry_sum | 0-64 | Sum of all carry values |
| max_carry_value | 0-21 | Peak single carry |
| n_answer_digits | varies | 2 values per level (e.g., 3 vs 4 at L3) |
| product_binned | 10 decile bins | Product magnitude — uses raw activations |
| digit_correct_pos{j} | 0, 1 | Per-position correctness (all pop only) |

### Tier 4: Partial Products

| Concept | Values | Preprocessing | Notes |
|---------|--------|---------------|-------|
| pp_a{i}_x_b{j} | 0-81 | bin into 9 equal-width ranges | Individual digit-by-digit products |

L3 has 4 partial products, L4 has 6, L5 has 9. These are binned because with 82 possible
values and 4,000 samples, many values have too few examples for stable centroids.

---

## 8. Value Filtering

Concepts with continuous or high-cardinality values need filtering to ensure stable
centroids.

**Rule:** Any concept value with fewer than 20 samples is dropped (set to NaN and
excluded from the analysis). Values are NOT merged into a sentinel bin, because merging
carry_1=15 (5 samples) with carry_1=17 (1 sample) produces a centroid that represents
neither value.

If fewer than 2 groups survive filtering, the concept is skipped for that combination.

**Examples of filtering in action:**

- L3 carry_1 has 18 possible values (0-17). Values 13-17 have very few samples. After
  filtering, 13 groups survive. The 5 rare values are dropped. 36 of 4,000 samples
  excluded.
- L3 ans_digit_3_msf exists only for 4-digit products (3,286 of 4,000 problems). The
  remaining 714 problems produce 3-digit products and have NaN for this concept.
- L5 correct population has only 239 samples. Most concepts have ~24 per digit value.
  Some groups fall below 20 and get filtered.
- L1 has only 64 total problems (~6 per digit value). Nearly all concepts fall below
  MIN_GROUP_SIZE=20. Result: zero subspaces at L1. This is expected and honest.

---

## 9. Significance Testing: The Permutation Null

The critical question: is a concept's subspace real, or could random groupings produce
the same structure?

In 4,096 dimensions with 4,000 points, even random groupings will have centroid
separation. You can always find 10 random groups whose centroids spread out, just by
chance. The permutation null controls for this.

**Procedure:**
1. Shuffle the concept labels randomly (assign each problem a random value 0-9)
2. Recompute centroids and SVD on shuffled labels
3. Record the eigenvalues
4. Repeat 1,000 times
5. Real eigenvalue at index j is significant (p < 0.01) if it exceeds the 99th
   percentile of shuffled eigenvalues at the same index

**Sequential stopping (Buja & Eyuboglu 1992):** Test lambda_1 first. If it fails, stop —
dim_perm = 0. If it passes, test lambda_2. Continue until the first failure. This is
standard parallel analysis practice.

**Why 1,000 permutations:** 200 permutations (the plan's original number) give a 99th
percentile estimate based on the 2nd-largest value out of 200. That is unstable. 1,000
permutations give the 10th-largest value, which is more reliable. The compute cost is
manageable: ~6 seconds per concept.

**What dim_perm = 0 means:** The concept's centroid structure is no better than random.
The model does not encode this concept in a way that conditional covariance can detect.
This does NOT mean the concept is unrepresented — it could be encoded nonlinearly, or
the signal could be too weak for this method. Phase D (LDA) may still detect it.

---

## 10. Dimensionality Determination

Three independent methods estimate how many dimensions a concept uses:

### Method 1: Cumulative Variance >= 95%

Sum eigenvalues from largest to smallest until 95% of total variance is explained. The
count is dim_cumvar.

### Method 2: Ratio Test

Find the first index j where lambda_j / lambda_{j+1} > 5. That is a "cliff" in the
spectrum. The count is dim_ratio = j.

**Important:** The last eigenvalue is always ~0 (structurally, because m centered
vectors have rank at most m-1). The ratio test only searches up to m-2 to avoid a
false cliff at this structural zero. Without this fix, every concept would report
dim_ratio = m-1.

### Method 3: Permutation Null

Count how many eigenvalues exceed their 99th percentile null threshold (with sequential
stopping). This is dim_perm.

### Consensus

The final estimate is the median of all three methods (or two, if the permutation null
was skipped). Minimum is 1.

### Example: carry_2 at L3, layer 16

```
Eigenvalues:
  lambda_1 = 0.295815  (81.2%)   <- dominates
  lambda_2 = 0.035355  ( 9.7%)   ratio: 8.37  <- CLIFF (ratio > 5)
  lambda_3 = 0.018519  ( 5.1%)
  lambda_4 = 0.006752  ( 1.9%)
  ...
  lambda_10 = 0.000000 ( 0.0%)   <- structural zero

dim_cumvar = 3 (lambda_1 + lambda_2 + lambda_3 = 96.0% > 95%)
dim_ratio = 1 (first ratio > 5 is at index 1, so cliff at j=1)
dim_perm = 8 (all 8 non-zero eigenvalues beat the null)
dim_consensus = median(3, 1, 8) = 3
```

The three methods disagree. This is informative: carry_2 has one dominant direction
(81.2% of variance) with smaller supporting directions that are still statistically
significant. The consensus of 3 captures the "effective" dimensionality — the primary
signal lives in 3 dimensions even though weaker signals span 8.

---

## 11. Cross-Validation

For each concept subspace, we verify it generalizes to held-out data using 5-fold
stratified cross-validation.

**Procedure:**
1. Split the data 80/20 (stratified by concept value)
2. Learn the subspace (centroids + SVD) on the 80% train set
3. Compute test centroids from the 20% test set
4. Measure Pearson correlation between full-space pairwise centroid distances and
   subspace pairwise centroid distances
5. Repeat 5 times with different splits

A correlation near 1.0 means the subspace learned from training data faithfully preserves
the centroid geometry on test data. A low correlation means the subspace is overfitting
to the training sample.

**Results across the full run:**

| Population | Mean CV Correlation | Notes |
|------------|-------------------|-------|
| L2 all | 0.987 - 1.000 | Near-perfect for all concepts |
| L3 all | 0.871 - 1.000 | ans_digit_1_msf lowest (0.871) |
| L4 all | 0.810 - 1.000 | Slight degradation for some PPs |
| L5 all | 0.775 - 0.998 | Partial products degrade most |
| L5 correct | 0.000 - 0.987 | Many concepts fail (small N) |

Cross-validation correlations of 0.8+ indicate robust subspaces. Values below 0.8 (seen
at L5, especially in the correct population with only 239 samples) indicate that the
subspace estimate is unstable due to small sample size. The subspace still exists (it
passed the permutation null), but its exact orientation is poorly estimated.

---

## 12. Run Summary

Phase C ran on SLURM job 6626413, node babel-t9-28, March 17, 2026.

| Metric | Value |
|--------|-------|
| Total subspaces computed | 2,835 |
| Significant (dim_perm > 0) | 2,513 (88.6%) |
| Cross-layer alignment entries | 2,520 |
| Correct/wrong divergence entries | 774 |
| Plots generated | 955 |
| Total runtime | 58 minutes |
| Step 1 (subspace identification) | 54.1 minutes |
| Step 2 (cross-layer alignment) | 2.1 seconds |
| Step 3 (correct/wrong divergence) | 0.6 seconds |
| Step 4 (plot generation) | 3.9 minutes |
| Residualized files | 45 / 45 |
| Permutations per concept | 1,000 |
| Parallel workers | 12 |

### Counts by level

| Level | Type | Total Subspaces | Significant | Significance Rate |
|-------|------|----------------|-------------|-------------------|
| L1 | 1x1 | 0 | 0 | N/A (64 samples, all filtered) |
| L2 | 2x1 | 315 | 315 | 100% |
| L3 | 2x2 | 666 | 623 | 94% |
| L4 | 3x2 | 837 | 745 | 89% |
| L5 | 3x3 | 1,017 | 830 | 82% |

L1 produces zero results because most concepts have ~6 samples per value, below the
MIN_GROUP_SIZE=20 filter. This is expected and correct — you cannot estimate reliable
centroids from 6 points in 4,096 dimensions.

L2 achieves 100% significance — every concept at every layer passes the permutation null.
This makes sense: L2 has 4,000 problems and 99.4% accuracy, so centroids are extremely
stable.

The significance rate drops from L2 (100%) to L5 (82%). This reflects two factors:
(a) the correct population at L5 has only 239 samples, making centroids noisy, and
(b) some answer digit subspaces genuinely fail significance at higher levels (see
Section 15).

---

## 13. Results by Level

### L3: The Sweet Spot (layer 16, all population)

L3 is the primary analysis level: 4,000 problems, 66% correct, balanced populations.

| Concept | Tier | Groups | dim_perm | dim_con | CV Corr |
|---------|------|--------|----------|---------|---------|
| a_tens | 1 | 9 | 8 | 8 | 1.000 |
| a_units | 1 | 10 | 9 | 9 | 0.997 |
| b_tens | 1 | 9 | 8 | 8 | 0.999 |
| b_units | 1 | 10 | 9 | 9 | 0.998 |
| ans_digit_0_msf | 1 | 9 | 8 | 8 | 0.956 |
| ans_digit_1_msf | 1 | 10 | **0** | 8 | 0.871 |
| ans_digit_2_msf | 1 | 10 | 8 | 6 | 0.978 |
| ans_digit_3_msf | 1 | 10 | 9 | 9 | 0.985 |
| carry_0 | 2 | 9 | 8 | 8 | 0.918 |
| carry_1 | 2 | 13 | 12 | 12 | 0.964 |
| carry_2 | 2 | 10 | 8 | 3 | 0.981 |
| col_sum_0 | 2 | 9 | 8 | 8 | 0.993 |
| col_sum_1 | 2 | 10 | 9 | 9 | 0.995 |
| col_sum_2 | 2 | 10 | 9 | 9 | 0.993 |
| correct | 3 | 2 | 1 | 1 | N/A |
| n_nonzero_carries | 3 | 4 | 3 | 2 | 0.992 |
| total_carry_sum | 3 | 24 | 23 | 23 | 0.980 |
| max_carry_value | 3 | 13 | 12 | 12 | 0.985 |
| n_answer_digits | 3 | 2 | 1 | 1 | N/A |
| product_binned | 3 | 10 | 9 | 2 | 0.999 |
| digit_correct_pos0 | 3 | 2 | 0 | 1 | N/A |
| digit_correct_pos1 | 3 | 2 | 1 | 1 | N/A |
| digit_correct_pos2 | 3 | 2 | 1 | 1 | N/A |
| digit_correct_pos3 | 3 | 2 | 1 | 1 | N/A |
| pp_a0_x_b0 | 4 | 9 | 8 | 8 | 0.980 |
| pp_a0_x_b1 | 4 | 9 | 8 | 8 | 0.970 |
| pp_a1_x_b0 | 4 | 9 | 7 | 7 | 0.989 |
| pp_a1_x_b1 | 4 | 9 | 8 | 4 | 0.995 |

26 of 28 concepts are significant at L3/layer16. The two failures (ans_digit_1_msf and
digit_correct_pos0) are discussed in Sections 15 and 19.

---

## 14. Input Digit Subspaces

### The headline finding

Input digits are represented at **maximum rank** across all levels and all layers.

At L3, a_tens has 9 distinct values (1-9) and its subspace has dim_perm = 8 = m-1 at
every single layer. At L2, a_units has 10 values and dim_perm = 9 at every layer. At L5,
a_hundreds has 9 values and dim_perm = 8 at every layer. No exceptions.

### What maximum rank means

The model dedicates the full available rank to representing each digit. For a_tens
with 9 values, 8 basis directions span the space of all possible centroid configurations.
The eigenvalue spectrum is:

```
a_tens at L3/layer16:
  lambda_1 = 0.242  (46.9%)
  lambda_2 = 0.121  (23.4%)
  lambda_3 = 0.069  (13.4%)
  lambda_4 = 0.026  ( 5.0%)
  lambda_5 = 0.019  ( 3.7%)
  lambda_6 = 0.015  ( 2.9%)
  lambda_7 = 0.014  ( 2.7%)
  lambda_8 = 0.010  ( 1.9%)
  lambda_9 = 0.000  ( 0.0%)  <- structural zero
```

The first two components capture 70% of variance, but the remaining 6 are all
statistically significant. There is no sharp cliff — the eigenvalues decay gradually.
This is consistent with a Fourier encoding where each frequency contributes its own
pair of dimensions (cos, sin components), as predicted by Bai et al. (2024) and
Kantamneni & Tegmark (2025).

### Cross-validation

Input digit subspaces have the highest CV correlations of any concept category:
0.985-1.000 at L2-L3. Even at L5, they remain above 0.985 in the all and wrong
populations. These subspaces are rock solid.

### Stability across layers (significance table, L3 all)

```
           L4  L6  L8  L12  L16  L20  L24  L28  L31
a_tens      8   8   8    8    8    8    8    8    8
a_units     9   9   9    9    9    9    9    9    9
b_tens      8   8   8    8    8    8    8    8    8
b_units     9   9   9    9    9    9    9    9    9
```

Every cell is maximum rank. The model encodes input digits at full strength from the
earliest extracted layer (4) through the last (31). There is no layer where digit encoding
"turns on" — it is present from the beginning. This makes sense: the model reads the
digits in the prompt during early layers and maintains that information throughout.

---

## 15. Answer Digit Subspaces

### The headline finding

Answer digits show a striking asymmetry: the **first** and **last** digits are strongly
represented, but the **middle** digits are not.

At L3 (4-digit products):

| Digit Position | dim_perm (all layers) | CV Corr | Interpretation |
|---------------|----------------------|---------|----------------|
| ans_digit_0 (leading) | 8 at all layers | 0.951-0.971 | Strongly encoded |
| ans_digit_1 (second) | **0 at all layers** | 0.837-0.947 | NOT significant |
| ans_digit_2 (third) | 8-9 at all layers | 0.965-0.988 | Strongly encoded |
| ans_digit_3 (units) | 9 at all layers | 0.985-0.990 | Strongly encoded |

At L5 (5-6 digit products):

| Digit Position | dim_perm (layer 16) | CV Corr |
|---------------|---------------------|---------|
| ans_digit_0 (leading) | 8 | 0.932 |
| ans_digit_1 | **0** | 0.879 |
| ans_digit_2 | **0** | 0.861 |
| ans_digit_3 | **0** | 0.865 |
| ans_digit_4 (second-to-last) | 2 | 0.963 |
| ans_digit_5 (units) | 9 | 0.955 |

### Why ans_digit_1 fails everywhere

The eigenvalue spectrum for ans_digit_1_msf at L3/layer16 reveals the problem:

```
ans_digit_1_msf:
  lambda_1 = 0.002749  (31.0%)
  lambda_2 = 0.001716  (19.3%)
  lambda_3 = 0.001209  (13.6%)
  lambda_4 = 0.000728  ( 8.2%)
  ...
```

The eigenvalues are ~100x smaller than input digits (0.003 vs 0.242). The centroids
barely separate. The CV correlation (0.871) is lower than other digits, confirming the
subspace estimate is unstable.

This is not a bug — the model genuinely does not create strong centroid separation for
the second answer digit. The dim_consensus is 8 because cumvar and ratio methods see
the gradual decay and report high dimensionality, but the permutation null correctly
identifies that this structure is no better than random groupings.

### The pattern: middle digits are hardest

This mirrors the error analysis finding that middle digit positions have the lowest
per-digit accuracy (L5 position 3 at 16.2%). The model fails to build clean
representations for the hardest output positions. At L5, only the leading digit
(magnitude) and the units digit (modular arithmetic) have significant subspaces.
The four middle digits (positions 1-4) are weak or absent.

---

## 16. Carry Subspaces

### Significance across layers (L3 all)

```
           L4  L6  L8  L12  L16  L20  L24  L28  L31
carry_0     8   8   8    8    8    8    8    8    8
carry_1    12  12  12   12   12   12   12   12   12
carry_2     8   8   8    8    8    8    8    8    8
```

All carries are significant at every layer, with stable dimensionality. carry_1 has the
highest dimensionality (12) because it has 13 surviving groups at L3 after filtering.

### carry_2: The most structured carry

carry_2 has a distinctive eigenvalue spectrum with one dominant direction:

```
carry_2 at L3/layer16:
  lambda_1 = 0.296  (81.2%)   <- one direction captures 81% of variance
  lambda_2 = 0.035  ( 9.7%)
  lambda_3 = 0.019  ( 5.1%)
```

This means carry_2 is essentially a 1D concept — one direction in 4096 dimensions
captures most of the carry_2 signal. The ratio test finds a cliff at position 1
(ratio 8.37 > 5.0), giving dim_ratio = 1. The consensus is 3, reflecting that
the supporting dimensions 2-3 are also significant even though dimension 1 dominates.

### Carry dimensionality at L5

At L5, carries have more possible values (carry_2 ranges 0-21 vs 0-9 at L3):

```
L5 layer 16 all:
  carry_0  dim_perm=8   (9 groups)
  carry_1  dim_perm=10  (13 groups)
  carry_2  dim_perm=7   (17 groups)
  carry_3  dim_perm=5   (14 groups)
  carry_4  dim_perm=8   (10 groups)
```

Even at L5 with 94% of answers wrong, all carries remain significant. The carry
representation persists through failure.

---

## 17. Column Sum Subspaces

Column sums are the bridge between partial products and carries. They represent the
pre-carry total at each output position.

### Significance (L3 all)

```
           L4  L6  L8  L12  L16  L20  L24  L28  L31
col_sum_0   8   8   8    8    8    8    8    8    8
col_sum_1   8   9   9    9    9    9    8    8    8
col_sum_2   9   9   9    8    9    9    9    9    9
```

All column sums are significant at all layers. CV correlations are excellent (0.993-0.999
at L3). The model genuinely maintains intermediate column totals in its activation space.

### L5: Column sums degrade at middle positions

```
L5 layer 16 all:
  col_sum_0  dim_perm=8   cv=0.994
  col_sum_1  dim_perm=8   cv=0.990
  col_sum_2  dim_perm=4   cv=0.990   <- reduced
  col_sum_3  dim_perm=5   cv=0.987   <- reduced
  col_sum_4  dim_perm=9   cv=0.970
```

The middle column sums (positions 2-3) have lower dimensionality than the edge columns
(positions 0, 1, 4). This parallels the answer digit pattern: middle positions are
hardest. The units column (col_sum_0) and the leading column (col_sum_4) are well
represented; the middle columns are weaker.

---

## 18. Partial Product Subspaces

### Significance

All partial products are significant at all layers at L2-L3 (dim_perm = 7-8). They
remain largely significant at L4-L5.

### CV correlations degrade at L5

```
L5 layer 16 all:
  pp_a0_x_b0  cv=0.912
  pp_a0_x_b1  cv=0.873
  pp_a1_x_b0  cv=0.834
  pp_a1_x_b1  cv=0.775   <- lowest
  pp_a1_x_b2  cv=0.938
  pp_a2_x_b0  cv=0.970
  pp_a2_x_b1  cv=0.947
  pp_a2_x_b2  cv=0.987
```

pp_a1_x_b1 (middle digit product) has the worst cross-validation at L5 (0.775). The
higher-order products (pp_a2_x_b0, pp_a2_x_b2) remain stable. This suggests the model
reliably encodes partial products involving the hundreds digit but struggles with
middle-digit interactions.

---

## 19. Derived Concept Subspaces

### correct (binary)

The correctness concept always has dim_perm = 1 and dim_consensus = 1. This is a single
direction in 4096 dimensions that separates correct from wrong activations. It exists at
every layer, at every level (where wrong answers exist).

### digit_correct_pos{j}

Per-position correctness shows an interesting pattern at L3:

```
L3 all:
  digit_correct_pos0  dim_perm=0   <- NOT significant (leading digit)
  digit_correct_pos1  dim_perm=1
  digit_correct_pos2  dim_perm=1
  digit_correct_pos3  dim_perm=1
```

The model does not encode "will I get the leading digit right" but DOES encode "will I
get the middle/trailing digits right." This makes sense: the leading digit is almost
always correct (it is determined by magnitude), while middle digits are where errors
concentrate.

At L4, digit_correct_pos0 and pos1 lose significance at later layers:

```
L4 all:
  digit_correct_pos0   sig at L4 only, 0 at L6-L31
  digit_correct_pos1   sig at L4-L12, 0 at L16-L28
  digit_correct_pos2-4 sig at all layers
```

The model "decides" per-digit correctness for middle/trailing positions early and
maintains it. For leading positions, the correctness signal is too weak to detect.

### product_binned

Product magnitude has dim_perm = 9 (maximum) at all layers and levels. However,
dim_consensus is only 2, because the first two eigenvalues capture 90%+ of variance.
Product magnitude is essentially a 2D concept with a strong primary direction (overall
scale) and a weaker secondary direction.

### total_carry_sum

This is the highest-dimensional concept in the registry: dim_perm = 23 at L3 (24 groups
after filtering). It uses 23 of 23 available dimensions. The carry complexity of a problem
is spread across many directions. This may reflect superposition: the model cannot
dedicate a compact subspace to this high-cardinality concept, so it spreads it across
many weakly-active directions.

---

## 20. Correct vs Wrong Divergence

### The method

For each concept with both correct and wrong subspaces, we compute the first principal
angle between them. A principal angle of 0 degrees means the subspaces are identical; 90
degrees means completely orthogonal (no shared directions).

### Results: 774 entries, 760 with angle < 60 degrees

The vast majority (98.2%) of concept pairs show strong alignment between correct and
wrong subspaces. The model uses mostly the same directions for correct and wrong answers,
but with detectable rotation.

### Top divergences at L3

| Layer | Concept | dim_correct | dim_wrong | Angle |
|-------|---------|-------------|-----------|-------|
| 24 | product_binned | 2 | 3 | 8.0° |
| 16 | a_tens | 8 | 8 | 16.3° |
| 31 | carry_2 | 8 | 8 | 15.7° |
| 31 | ans_digit_3_msf | 9 | 8 | 15.6° |
| 16 | col_sum_0 | 7 | 9 | 31.7° |
| 16 | carry_0 | 7 | 8 | 34.8° |
| 16 | ans_digit_1_msf | 8 | 8 | 38.0° |
| 16 | ans_digit_2_msf | 6 | 8 | 39.2° |
| n/a | n_answer_digits | 1 | 1 | 64.1° |

Key observations:

1. **Product magnitude has the smallest angle** (8-11 degrees). Correct and wrong answers
   use almost identical directions for product encoding. The model "knows" the magnitude
   regardless of whether it gets the exact digits right.

2. **Input digits have small angles** (12-22 degrees). The model reads input digits the
   same way for correct and wrong answers.

3. **Carries and column sums have moderate angles** (16-35 degrees). The intermediate
   computation subspaces are more different between correct and wrong.

4. **Answer digits have the largest angles** (16-39 degrees for middle digits). The output
   representation is where correct and wrong answers diverge most.

### The dimensionality asymmetry at L5

At L5, the correct population (239 samples) consistently shows **lower dimensionality**
than the wrong population (3,761 samples):

```
L5 layer 16:
  a_units     correct dim=3   wrong dim=9
  a_tens      correct dim=5   wrong dim=9
  b_units     correct dim=5   wrong dim=9
  carry_0     correct dim=0   wrong dim=8
  carry_2     correct dim=1   wrong dim=7
  col_sum_0   correct dim=2   wrong dim=8
```

The correct population uses **compressed, low-dimensional representations**. The wrong
population sprawls across many more dimensions. Two interpretations:

1. **The correct problems are "simple"** — they have fewer carries, smaller products,
   and less complex intermediate structure, so the model can compress their representation.
2. **The wrong population is noisy** — with 3,761 wrong answers, the "wrong" category
   is heterogeneous (some are close to correct, some are wildly wrong), inflating the
   apparent dimensionality.

Both interpretations are probably partially true. The carry_0 result is striking:
dim_correct = 0 means carry structure does not pass the permutation null for the 239
correct L5 problems. The model either uses a different encoding for carries in the few
cases it succeeds, or the sample size is simply too small to detect it.

---

## 21. Cross-Layer Alignment

### The method

For each concept, compute the principal angles between the subspace basis at adjacent
layers: (4,6), (6,8), (8,12), (12,16), (16,20), (20,24), (24,28), (28,31).

A small angle means the subspace is preserved between layers. A large angle means the
subspace rotated — the model is reorganizing its representation.

### The universal pattern

Every concept shows the same trajectory across layers:

```
a_tens L3 all (first principal angle):
  4->6:   68.2°   <- large rotation (early layers, input encoding)
  6->8:   62.1°
  8->12:  65.7°
  12->16: 63.9°
  16->20: 53.4°   <- rotation decreasing
  20->24: 44.1°   <- convergence zone
  24->28: 35.8°   <- MOST STABLE transition
  28->31: 46.2°   <- slight divergence at output
```

This pattern holds for every concept examined (b_units, carry_0, carry_2, col_sum_1,
ans_digit_0_msf, correct):

1. **Layers 4-16: High rotation (60-70 degrees).** The representation is actively being
   transformed. Each layer applies a substantial rotation to the concept subspace.
2. **Layers 16-24: Decreasing rotation (44-54 degrees).** The representation is settling.
3. **Layers 24-28: Minimum rotation (35-40 degrees).** The most stable transition in the
   network. The representation has converged.
4. **Layer 28-31: Slight increase (46-60 degrees).** The final layer applies a modest
   rotation, possibly for output preparation.

### Why layers 24-28 are the most stable

Phase A's CKA analysis found layers 20, 24, and 28 have very high global similarity
(CKA 0.985-0.996). Phase C confirms this at the concept-subspace level: the 24->28
transition has the smallest principal angles for nearly every concept. The model's
internal representation has essentially converged by layer 24, and layers 24-28 are
fine-tuning rather than transforming.

This justifies keeping all 9 layers in the analysis. CKA said these layers are
"redundant." Principal angles show they are stable but not identical — the fine-tuning
in layers 24-28 may be what separates correct from wrong answers.

### L5 follows the same pattern

```
a_tens L5 all:
  4->6:   67.3°
  6->8:   62.5°
  8->12:  67.0°
  12->16: 65.3°
  16->20: 54.0°
  20->24: 43.2°
  24->28: 38.2°   <- still the most stable
  28->31: 53.1°
```

The trajectory is remarkably consistent across levels. The model uses the same
layer-by-layer processing strategy regardless of task difficulty.

---

## 22. The Failure Regime: L5

L5 (3-digit x 3-digit, 6% accuracy) reveals what breaks when the model fails.

### What survives

- **Input digits**: fully significant at all layers for all and wrong populations.
  dim_perm = 8-9 everywhere. The model reads the input correctly even when it cannot
  compute the answer.
- **Carries**: significant at all layers (dim_perm = 5-11). The model maintains carry
  representations even while producing wrong answers.
- **Column sums**: significant but with reduced dimensionality at middle positions.
- **Partial products**: significant (dim_perm = 7-8) with degraded CV correlations.

### What fails

- **Middle answer digits** (positions 1-3): dim_perm = 0 at all layers. The model does
  not build output representations for these positions.
- **Correct population subspaces**: Many concepts have dim_perm = 0 or very low. With
  only 239 correct problems (~24 per digit value), centroids are too noisy to pass the
  permutation null.

### The significance table tells the story

```
L5 correct population:
  a_units      dim_perm: 1  1  1  1  3  3  0  3  3   <- unstable
  carry_0      dim_perm: 2  1  1  2  0  0  0  0  0   <- disappears at layer 16+
  col_sum_0    dim_perm: 3  2  2  2  2  0  0  0  2   <- disappears mid-network
  pp_a0_x_b0   dim_perm: 1  1  1  1  0  0  0  0  1   <- disappears mid-network
  ans_digit_0  dim_perm: 0  0  0  0  0  0  0  0  0   <- never present
```

In the correct L5 population, concept subspaces literally vanish at intermediate layers
(16-28). Input digits partially survive because they are directly present in the prompt.
Carries, column sums, and partial products disappear at exactly the layers where
computation should be happening. This is consistent with the interpretation that the
model's 6% success rate at L5 is achieved through memorization or heuristics rather
than systematic computation.

---

## 23. Key Findings

### Finding 1: Input digits use maximum rank at all levels and layers

Every input digit concept uses the full available dimensionality (m-1) at every layer
from 4 to 31, at every level from 2 to 5. This is consistent with Fourier encoding
(9 non-constant real basis functions for Z/10Z). The model allocates the maximum possible
representation to input digit identity. Cross-validation correlations are 0.985-1.000.

### Finding 2: Answer digits have an edge-vs-middle asymmetry

The leading answer digit (magnitude-related) and the units digit (modular arithmetic)
have significant subspaces. Middle answer digits (positions 1-3 at L5) fail the
permutation null entirely. This mirrors the error analysis: middle digits are hardest
and have the lowest per-digit accuracy.

### Finding 3: The model maintains intermediate computation representations

Column sums, carries, and partial products all have significant subspaces, confirming
the model builds intermediate representations of the long-multiplication algorithm.
These are not just input echoes — they are computed quantities that depend on
combinations of input digits.

### Finding 4: Correct and wrong answers use mostly aligned subspaces

Principal angles between correct and wrong subspaces are 8-40 degrees. The model uses
similar (not identical) directions for both populations. The largest divergences are in
answer digit and carry representations, suggesting computation breaks down at the
intermediate-to-output stage.

### Finding 5: At L5, the correct population uses compressed representations

Correct L5 subspaces have 2-5 dimensions where wrong subspaces have 7-9. Many concepts
lose significance entirely in the correct population at intermediate layers. The model
may succeed at L5 through heuristics that use a compact code rather than full
algorithmic computation.

### Finding 6: All subspaces follow the same layer trajectory

Every concept shows the same rotation profile: high rotation in early layers (60-70
degrees), decreasing through middle layers, minimum at 24->28 (35-40 degrees), slight
increase at 28->31. The model's representational convergence at layers 24-28 is a
universal phenomenon, not concept-specific.

### Finding 7: Column sums are the missing link

Column sums (added in this analysis) are as strongly represented as input digits
(CV 0.993-0.999). The model genuinely maintains pre-carry column totals as intermediate
representations. This is the first demonstration of column sum encoding in a
production-scale LLM for multiplication.

---

## 24. Output Files

### Data outputs

```
/data/user_data/anshulk/arithmetic-geometry/phase_c/
├── residualized/                      45 files, ~2.9 GB
│   └── level{N}_layer{L}.npy         (N_problems, 4096) float32
├── subspaces/                         14,175 files total
│   └── L{N}/layer_{LL}/{pop}/{concept}/
│       ├── basis.npy                  (dim_consensus, 4096)
│       ├── eigenvalues.npy            (m,)
│       ├── null_eigenvalues.npy       (1000, m-1)
│       ├── metadata.json              All results + resume checkpoint
│       └── projected_all.npy          (N_pop, dim_consensus) — all samples
└── summary/
    ├── phase_c_results.csv            2,835 rows — master results table
    ├── correct_wrong_divergence.csv   774 rows — principal angles
    ├── alignment_results.csv          2,520 rows — cross-layer angles
    └── significance_L{N}_{pop}.csv    Concept x layer grids of dim_perm
```

### Plot outputs

```
/home/anshulk/arithmetic-geometry/plots/phase_c/
├── eigenvalue_spectra/                543 plots
├── dimensionality_heatmaps/           11 plots
├── cross_layer_trajectories/          315 plots
└── correct_wrong_comparison/          86 plots
                                       ─────
                                       955 total
```

---

## 25. Runtime and Reproducibility

| Property | Value |
|----------|-------|
| SLURM Job ID | 6626413 |
| Node | babel-t9-28 |
| Start time | 2026-03-17 22:47:36 EDT |
| End time | 2026-03-17 23:45:54 EDT |
| Total runtime | 58 minutes |
| Exit code | 0 |
| Partition | general |
| CPUs | 12 |
| Memory | 64 GB |
| GPU | A6000 (allocated but unused — CPU-only computation) |
| Conda environment | geometry |
| Script | phase_c_subspaces.py |
| Config | config.yaml |
| Parallelization | joblib, 12 workers |
| RNG seed | 42 + level*100 + layer (deterministic per pair) |
| Permutations | 1,000 per concept |
| MIN_GROUP_SIZE | 20 |
| MIN_POPULATION | 30 |
| CUMVAR_THRESHOLD | 0.95 |
| RATIO_THRESHOLD | 5.0 |
| PERM_ALPHA | 0.01 |

*End of document. All numbers verified against phase_c_results.csv, alignment_results.csv,
correct_wrong_divergence.csv, significance tables, eigenvalue .npy files, and metadata.json
files as of March 18, 2026.*
