# Phase C: Complete Analysis

**Anshul's Geometric Manifold Interpretability Project**
**Carnegie Mellon University, March 2026**

This document records every decision, every number, and every result from Phase C —
the concept subspace identification stage. It is the truth document for this stage.
All numbers are validated against the actual output files as of March 20, 2026.

This is the second run of Phase C. The first run (March 17, SLURM 6626413) used the
old data (4K samples at L3/L4, 239 correct at L5). This run uses the new data (10K
at L3/L4, 122K at L5 with 4,197 correct). The algorithm is the same. The data changed.
The results are substantially different at L5.

---

## Table of Contents

1. [Purpose of This Stage](#1-purpose-of-this-stage)
2. [What Phase C Is and Is Not](#2-what-phase-c-is-and-is-not)
3. [The Core Question](#3-the-core-question)
4. [Why Not Just PCA](#4-why-not-just-pca)
5. [The Algorithm: Conditional Covariance + SVD](#5-the-algorithm-conditional-covariance--svd)
6. [Product Residualization](#6-product-residualization)
7. [Concept Registry](#7-concept-registry)
8. [Value Filtering and L5 Carry Binning](#8-value-filtering-and-l5-carry-binning)
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
23. [What Changed From the First Run](#23-what-changed-from-the-first-run)
24. [Key Findings](#24-key-findings)
25. [Output Files](#25-output-files)
26. [Runtime and Reproducibility](#26-runtime-and-reproducibility)

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

Take one concrete slice to build intuition: 10,000 Level 3 problems at layer 16.
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

Take all 10,000 Level 3 problems. Group them by the value of a_tens (the tens digit of
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

Because m is small (2-68 depending on the concept), this is a tiny SVD. Cost:
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
activations are used. This is the one concept that deliberately uses unresidulized
activations, because residualizing product from the product concept would be circular.

**Phase B's validation:** Phase B (the label-level deconfounding stage, run March 19)
confirmed that product residualization is sufficient. After removing product from labels,
the only strong remaining correlations (|r| > 0.3) are structurally forced (carry chains,
column sums, partial products) or residualization-induced (leading digit suppression).
No sampling-induced correlations require additional deconfounding. Phase B's
deconfounding_plan.json specifies `raw_activation_override_only` — no code changes
needed for Phase C's residualization. Phase C does not read this file; the validation
is documented, not programmatic.

Residualized activations are cached at `phase_c/residualized/level{N}_layer{L}.npy`.
45 files were produced, one per (level, layer) pair. Total: 21 GB.

**GPU acceleration:** Residualization uses CuPy when available. The regression coefficient
beta is computed via GPU matmul, and the outer-product subtraction is done on-device. At
L5 with 122,223 × 4096 activations, this saves several seconds per layer.

---

## 7. Concept Registry

Phase C analyzes 28 concepts at L3 and up to 43 at L5, organized into four tiers.
Each concept is checked for existence in the coloring DataFrame before being included
(not all concepts exist at all levels). The coloring DataFrame has more columns than
the registry uses — L3 has 40 columns but only 28 are in the registry. The extra 12
are identity columns (problem_idx, a, b, predicted, ground_truth) and error-only
variables (abs_error, rel_error, signed_error, underestimate, even_pred, div10_pred,
error_category) that are not mathematical concepts.

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
are filtered to drop rare values at L3/L4 and binned at L5 (see Section 8).

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
values and 10,000 samples, many values have too few examples for stable centroids.

---

## 8. Value Filtering and L5 Carry Binning

### The general rule

Concepts with continuous or high-cardinality values need filtering to ensure stable
centroids.

Any concept value with fewer than 20 samples is dropped (set to NaN and excluded
from the analysis). If fewer than 2 groups survive filtering, the concept is skipped
for that combination.

### L5 carry binning: the new addition

At L5, carries have wider ranges than at L3/L4. carry_1 ranges 0-17, carry_2 ranges
0-21. The tail values (carry_1 = 13, 14, 15, 16, 17) each have very few samples.
The old approach (drop rare values) would lose those problems entirely. The new
approach merges tail values into a single group, preserving the data.

The binning thresholds come from the data generation analysis:

| Carry | Individual values | Merged group | Total groups |
|-------|-------------------|--------------|--------------|
| carry_0 | 0-8 | none needed | 9 |
| carry_1 | 0-11 | >= 12 | 13 |
| carry_2 | 0-12 | >= 13 | 14 |
| carry_3 | 0-8 | >= 9 | 10 |
| carry_4 | 0-4 | >= 5 | 6 |

The binning happens before filtering. After binning, every merged group has at least
199 samples in the all population, well above MIN_GROUP_SIZE=20. Zero values are dropped
that the binning was designed to save.

This only applies to L5. L3 and L4 carries use the original filter-rare approach.

### Verification from the actual run

The filter_meta in the results confirms the binning worked:

```
carry_0: 9 groups, 0 dropped (no binning needed)
carry_1: 13 groups, 0 dropped (values 0-11 individual, >=12 merged)
carry_2: 14 groups, 0 dropped (values 0-12 individual, >=13 merged)
carry_3: 10 groups, 0 dropped (values 0-8 individual, >=9 merged)
carry_4:  6 groups, 0 dropped (values 0-4 individual, >=5 merged)
```

### Examples of filtering in action

- L3 carry_1 has 16 possible values (0-15 after the 10K sample increase). Values 15
  and above have very few samples. After filtering, 15 groups survive.
- L3 ans_digit_3_msf exists only for 4-digit products. Problems producing 3-digit
  products have NaN for this concept.
- L1 has only 64 total problems (~6 per digit value). Nearly all concepts fall below
  MIN_GROUP_SIZE=20. Result: zero subspaces at L1. This is expected and honest.

### Note on Phase B alignment

Phase B (run March 19) analyzed label correlations using unbinned L5 carries, because
Phase B ran before the binning was added. The impact is negligible: Phase B's conclusion
(no deconfounding needed beyond product) doesn't depend on whether tail carry values
are binned or dropped, because those tail values contribute very little to the overall
correlation structure.

---

## 9. Significance Testing: The Permutation Null

The critical question: is a concept's subspace real, or could random groupings produce
the same structure?

In 4,096 dimensions with 10,000 points, even random groupings will have centroid
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
permutations give the 10th-largest value, which is more reliable.

**GPU acceleration:** The permutation null is 99% of the compute cost. At L5 with
122,223 samples, each permutation does a (42 × 122,223) @ (122,223 × 4096) matrix
multiply. One thousand of those per concept. On 12 CPU cores the original run took 58
minutes for all levels. With GPU (CuPy on an A6000), this run took 201 minutes total —
longer because L5 is 30x larger than before, but the GPU kept it manageable. Without
the GPU, the estimated CPU time would have been 4-6 hours.

The GPU path batches permutations: build one_hot matrices directly on GPU (avoiding
PCIe transfer), do batched matmuls, loop SVDs on GPU. Memory per batch at L5: ~4 GB,
well within the A6000's 48 GB.

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
  lambda_1 = 0.291331  (83.6%)   <- dominates
  lambda_2 = 0.033383  ( 9.6%)   ratio: 8.72  <- CLIFF (ratio > 5)
  lambda_3 = 0.013117  ( 3.8%)
  lambda_4 = 0.004464  ( 1.3%)
  lambda_5 = 0.003145  ( 0.9%)
  ...
  lambda_10 = 0.000000 ( 0.0%)   <- structural zero

dim_cumvar = 3 (lambda_1 + lambda_2 + lambda_3 = 97.0% > 95%)
dim_ratio = 1 (first ratio > 5 is at index 1, so cliff at j=1)
dim_perm = 8 (all 8 non-zero eigenvalues beat the null)
dim_consensus = median(3, 1, 8) = 3
```

The three methods disagree. This is informative: carry_2 has one dominant direction
(83.6% of variance) with smaller supporting directions that are still statistically
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
| L2 all | 0.936 - 1.000 | Near-perfect for all concepts |
| L3 all | 0.909 - 1.000 | ans_digit_1_msf lowest (0.909) |
| L4 all | 0.937 - 1.000 | All concepts robust |
| L5 all | 0.838 - 1.000 | ans_digit_2,3 lowest (~0.84-0.89) |
| L5 correct | 0.859 - 0.998 | pp_a2_x_b2 lowest (0.859) |

Cross-validation correlations of 0.8+ indicate robust subspaces. The 10K→122K scale-up
dramatically improved L5 correct population stability: the old run had many CV < 0.5
with only 239 correct samples. Now with 4,197 correct samples, the worst case is
pp_a2_x_b2 at 0.859 — still robust.

---

## 12. Run Summary

Phase C ran on SLURM job 6659197, node babel-s9-16, March 19-20, 2026.

| Metric | Value |
|--------|-------|
| Total subspaces computed | 2,844 |
| Significant (dim_perm > 0) | 2,750 (96.7%) |
| Cross-layer alignment entries | 2,528 |
| Correct/wrong divergence entries | 792 |
| Plots generated | 961 |
| Total runtime | 201 minutes (3h 21m) |
| Step 1 (subspace identification) | ~197 minutes |
| Step 4 (plot generation) | 163.8 seconds |
| Residualized files | 45 / 45 |
| Permutations per concept | 1,000 |
| GPU | A6000, used via CuPy for permutation null + residualization |
| Parallel workers | 1 (GPU handles parallelism, n_jobs=1 to avoid contention) |

### Counts by level

| Level | Type | Total Subspaces | Significant | Significance Rate |
|-------|------|----------------|-------------|-------------------|
| L1 | 1x1 | 0 | 0 | N/A (64 samples, all filtered) |
| L2 | 2x1 | 306 | 306 | 100% |
| L3 | 2x2 | 666 | 650 | 97.6% |
| L4 | 3x2 | 837 | 798 | 95.3% |
| L5 | 3x3 | 1,035 | 996 | 96.2% |

L1 produces zero results because most concepts have ~6 samples per value, below the
MIN_GROUP_SIZE=20 filter. This is expected and correct — you cannot estimate reliable
centroids from 6 points in 4,096 dimensions.

L2 achieves 100% significance — every concept at every layer passes the permutation null.
This makes sense: L2 has 4,000 problems and 99.4% accuracy, so centroids are extremely
stable.

### Comparison with the first run

| Metric | First run (Mar 17) | This run (Mar 19) |
|--------|-------------------|-------------------|
| Total subspaces | 2,835 | 2,844 |
| Significant | 2,513 (88.6%) | 2,750 (96.7%) |
| L5 significance | 830/1,017 (82%) | 996/1,035 (96.2%) |
| L5 correct n_samples | 239 | 4,197 |
| Runtime | 58 min (12 CPU) | 201 min (1 GPU) |
| Total disk | ~2.9 GB | 26 GB |

The significance rate jumped from 88.6% to 96.7%, driven almost entirely by L5. With
17x more correct samples, concepts that previously failed the permutation null now
pass. The runtime is longer because L5 has 30x more total samples (122K vs 4K), but
GPU acceleration kept it manageable.

---

## 13. Results by Level

### L3: The Sweet Spot (layer 16, all population)

L3 is the primary analysis level: 10,000 problems, 66% correct, balanced populations.

| Concept | Tier | Groups | dim_perm | dim_con | CV Corr |
|---------|------|--------|----------|---------|---------|
| a_tens | 1 | 9 | 8 | 8 | 1.000 |
| a_units | 1 | 10 | 9 | 9 | 0.999 |
| b_tens | 1 | 9 | 8 | 8 | 1.000 |
| b_units | 1 | 10 | 9 | 9 | 1.000 |
| ans_digit_0_msf | 1 | 9 | 8 | 8 | 0.978 |
| ans_digit_1_msf | 1 | 10 | 3 | 8 | 0.909 |
| ans_digit_2_msf | 1 | 10 | 9 | 5 | 0.989 |
| ans_digit_3_msf | 1 | 10 | 9 | 9 | 0.995 |
| carry_0 | 2 | 9 | 8 | 8 | 0.962 |
| carry_1 | 2 | 15 | 14 | 14 | 0.987 |
| carry_2 | 2 | 10 | 8 | 3 | 0.997 |
| col_sum_0 | 2 | 9 | 8 | 8 | 0.998 |
| col_sum_1 | 2 | 10 | 9 | 9 | 0.999 |
| col_sum_2 | 2 | 10 | 9 | 9 | 0.999 |
| correct | 3 | 2 | 1 | 1 | N/A |
| n_nonzero_carries | 3 | 4 | 3 | 2 | 0.998 |
| total_carry_sum | 3 | 28 | 27 | 27 | 0.994 |
| max_carry_value | 3 | 15 | 14 | 14 | 0.980 |
| n_answer_digits | 3 | 2 | 1 | 1 | N/A |
| product_binned | 3 | 10 | 9 | 2 | 0.999 |
| digit_correct_pos0 | 3 | 2 | 1 | 1 | N/A |
| digit_correct_pos1 | 3 | 2 | 1 | 1 | N/A |
| digit_correct_pos2 | 3 | 2 | 1 | 1 | N/A |
| digit_correct_pos3 | 3 | 2 | 1 | 1 | N/A |
| pp_a0_x_b0 | 4 | 9 | 8 | 8 | 0.975 |
| pp_a0_x_b1 | 4 | 9 | 8 | 6 | 0.996 |
| pp_a1_x_b0 | 4 | 9 | 8 | 8 | 0.997 |
| pp_a1_x_b1 | 4 | 9 | 8 | 3 | 0.997 |

28 of 28 concepts are significant at L3/layer16. The first run had 26/28 (ans_digit_1
and digit_correct_pos0 failed). The scale-up from 4,000 to 10,000 made the difference:
ans_digit_1_msf now achieves dim_perm = 3 (up from 0), and all digit_correct_pos
concepts are significant.

### Key changes from the first run at L3

**ans_digit_1_msf:** The second answer digit went from dim_perm = 0 at ALL layers to a
mixed pattern: [0, 0, 9, 3, 3, 3, 1, 1, 0]. It is now significant at layers 8-28 but
still fails at the edges (layers 4, 6, 31). This is a real but weak signal that the
first run's smaller dataset couldn't detect. The eigenvalues are still ~100x smaller
than input digits, confirming this is a faint encoding.

**carry_1:** Groups increased from 13 to 15 (more rare carry values now pass the
MIN_GROUP_SIZE=20 filter with 10K samples). Dimensionality increased accordingly.

**total_carry_sum:** Groups increased from 24 to 28.

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
  lambda_1 = 0.248  (47.7%)
  lambda_2 = 0.122  (23.5%)
  lambda_3 = 0.067  (12.8%)
  lambda_4 = 0.027  ( 5.1%)
  lambda_5 = 0.019  ( 3.6%)
  lambda_6 = 0.014  ( 2.8%)
  lambda_7 = 0.014  ( 2.6%)
  lambda_8 = 0.010  ( 1.8%)
  lambda_9 = 0.000  ( 0.0%)  <- structural zero
```

The first two components capture 71% of variance, but the remaining 6 are all
statistically significant. There is no sharp cliff — the eigenvalues decay gradually.
This is consistent with a Fourier encoding where each frequency contributes its own
pair of dimensions (cos, sin components), as predicted by Bai et al. (2024) and
Kantamneni & Tegmark (2025).

### Cross-validation

Input digit subspaces have the highest CV correlations of any concept category:
0.996-1.000 at L2-L3. Even at L5 correct, they remain 0.985-0.995. These subspaces
are rock solid.

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

### L5 correct: the big change

In the first run with 239 correct L5 samples, input digit subspaces were fragile:
a_units had dim_perm fluctuating between 0 and 3 across layers. Now with 4,197
correct samples:

```
L5 correct:
           L4  L6  L8  L12  L16  L20  L24  L28  L31
a_units     9   9   9    9    9    9    9    9    9
a_tens      9   9   9    9    9    9    9    9    9
a_hundreds  8   8   8    8    8    8    8    8    8
```

Maximum rank at every layer. The "vanishing subspaces" from the first run were entirely
a sample-size artifact. The model encodes input digits identically in correct and wrong
populations at L5.

---

## 15. Answer Digit Subspaces

### The headline finding

Answer digits show a striking asymmetry: the **edge** digits are strongly represented,
but the **middle** digits are weak or absent. This pattern holds across all levels and
was already present in the first run. The scale-up sharpened the picture.

### L3 answer digits across layers (all population)

```
                    L4  L6  L8  L12  L16  L20  L24  L28  L31
ans_digit_0 (lead)   8   8   8    8    8    8    8    8    8
ans_digit_1 (2nd)    0   0   9    3    3    3    1    1    0
ans_digit_2 (3rd)    9   9   9    9    9    9    9    9    9
ans_digit_3 (units)  9   9   9    9    9    9    9    9    9
```

ans_digit_0 (leading digit) and ans_digit_3 (units digit) are fully significant at
every layer. ans_digit_2 is also fully significant. ans_digit_1 (second digit) is the
weak one — it passes at layers 8-28 but fails at the edges. With 10K samples it is
now partially detected; with 4K it was invisible.

### L4 answer digits (5 positions, all population)

```
                    L4  L6  L8  L12  L16  L20  L24  L28  L31
ans_digit_0 (lead)   8   8   8    8    8    8    8    8    8
ans_digit_1 (2nd)    0   0   0    1    1    0    1    1    1
ans_digit_2 (3rd)    3   3   2    2    2    2    0    2    1
ans_digit_3 (4th)    5   9   9    9    9    9    9    9    9
ans_digit_4 (units)  9   9   9    9    9    9    9    9    9
```

The same edge-vs-middle pattern. Positions 0 and 4 (edges) are fully significant.
Positions 1 and 2 (middle) are weak. Position 3 (second-from-last) is strong.

### L5 answer digits (6 positions)

**All population:**

```
                    L4  L6  L8  L12  L16  L20  L24  L28  L31
ans_digit_0 (lead)   8   8   8    8    8    8    8    8    8
ans_digit_1 (2nd)    6   3   4    7    7    7    7    7    9
ans_digit_2 (3rd)    1   1   2    1    1    0    0    0    1
ans_digit_3 (4th)    2   1   1    1    0    0    0    0    0
ans_digit_4 (5th)    9   9   9    9    9    9    9    9    9
ans_digit_5 (units)  9   9   9    9    9    9    9    9    9
```

**Correct population:**

```
                    L4  L6  L8  L12  L16  L20  L24  L28  L31
ans_digit_0 (lead)   4   5   8    8    8    8    8    8    8
ans_digit_1 (2nd)    0   0   0    0    0    0    0    0    0
ans_digit_2 (3rd)    0   0   0    0    0    0    0    0    0
ans_digit_3 (4th)    7   8   9    9    8    2    2    2    2
ans_digit_4 (5th)    9   9   9    9    9    9    9    9    9
ans_digit_5 (units)  7   7   7    7    7    7    7    7    7
```

This is the most important table in Phase C. For problems the model gets right:
- ans_digit_0 (leading): significant from layer 8 onward. Gets stronger as you go
  deeper. The model builds a magnitude representation early, then refines it.
- ans_digit_1 and ans_digit_2 (middle): **dim_perm = 0 at every single layer**. Even
  with 4,197 samples, the model does not build detectable centroid separation for these
  positions. This is a confirmed null result, not a sample-size artifact.
- ans_digit_3 (fourth): significant at early layers (dim_perm 7-9 at L4-L12), then
  drops sharply to dim_perm = 2 at layers 20-31. The model starts building this
  representation but loses it in later layers.
- ans_digit_4 and ans_digit_5 (trailing): fully significant at all layers. The units
  digit is strongly encoded throughout.

### Why ans_digit_1 fails even with 4,197 samples

The eigenvalue spectrum tells the story:

```
ans_digit_1_msf at L3/layer16:
  lambda_1 = 0.002179  (37.7%)
  lambda_2 = 0.001159  (20.0%)
  lambda_3 = 0.000757  (13.1%)
  lambda_4 = 0.000414  ( 7.2%)
  ...
```

The eigenvalues are ~100x smaller than input digits (0.002 vs 0.248). The centroids
barely separate. The model genuinely does not create strong centroid separation for the
second answer digit. This is not a power issue — it is a signal issue.

### The pattern: edges easy, middle hard

| Level | Easy positions | Hard positions |
|-------|---------------|----------------|
| L3 (4 digits) | 0 (lead), 2, 3 (units) | 1 (second) |
| L4 (5 digits) | 0 (lead), 3, 4 (units) | 1, 2 (middle) |
| L5 (6 digits) | 0 (lead), 4, 5 (units) | 1, 2, 3 (middle) |

This mirrors the error analysis: middle digit positions have the lowest per-digit
accuracy. The model fails to build clean representations for the hardest output
positions. The leading digit (determined by magnitude) and the trailing digits
(determined by modular arithmetic of the lowest-order columns) are easier. The middle
digits require propagating carries through multiple columns — the hardest part of
long multiplication.

---

## 16. Carry Subspaces

### Significance across layers (L3 all)

```
           L4  L6  L8  L12  L16  L20  L24  L28  L31
carry_0     8   8   8    8    8    8    8    8    8
carry_1    14  14  14   14   14   14   14   14   14
carry_2     8   8   8    8    8    8    8    8    8
```

All carries are significant at every layer, with perfectly stable dimensionality. carry_1
has the highest dimensionality (14) because it has 15 surviving groups at L3 after
filtering (the 10K scale-up allows more rare values to pass MIN_GROUP_SIZE=20).

### carry_2: The most structured carry

carry_2 has a distinctive eigenvalue spectrum with one dominant direction:

```
carry_2 at L3/layer16:
  lambda_1 = 0.291  (83.6%)   <- one direction captures 84% of variance
  lambda_2 = 0.033  ( 9.6%)
  lambda_3 = 0.013  ( 3.8%)
```

This means carry_2 is essentially a 1D concept — one direction in 4096 dimensions
captures most of the carry_2 signal. The ratio test finds a cliff at position 1
(ratio 8.72 > 5.0), giving dim_ratio = 1. The consensus is 3, reflecting that
the supporting dimensions 2-3 are also significant even though dimension 1 dominates.

### Carry dimensionality at L5 (all population)

```
           L4  L6  L8  L12  L16  L20  L24  L28  L31
carry_0     8   8   8    8    8    8    8    8    8
carry_1    12  12  12   12   12   12   12   12   12
carry_2     9   9   9    9   10   10   10   10   10
carry_3     9   9   9    9    9    9    9    9    9
carry_4     5   5   5    5    5    5    5    5    5
```

With 122K samples, every carry at every layer achieves maximum or near-maximum rank.
carry_4 has dim_perm = 5 out of 5 possible (6 groups - 1). The carry representation
is perfectly stable across the full network depth.

### L5 correct: carries survive

```
L5 correct:
           L4  L6  L8  L12  L16  L20  L24  L28  L31
carry_0     8   8   8    8    8    8    8    8    8
carry_1    12  10  11   12   12   12   12   12   12
carry_2     8   7   7    8    8    8    8    8   10
carry_3     7   9   9    9    7    9    9    9    9
carry_4     5   5   5    5    5    5    5    5    5
```

This is a dramatic reversal from the first run, where carry_0 had dim_perm = 0 at
layers 16+ with only 239 correct samples. Now every carry is significant at every
layer. The model maintains full carry representations for the problems it gets right.
The "vanishing carry subspaces" were a sample-size artifact, not a property of the model.

---

## 17. Column Sum Subspaces

Column sums are the bridge between partial products and carries. They represent the
pre-carry total at each output position.

### Significance (L3 all)

```
           L4  L6  L8  L12  L16  L20  L24  L28  L31
col_sum_0   8   8   8    8    8    8    8    8    8
col_sum_1   9   9   9    9    9    9    9    9    9
col_sum_2   9   9   9    9    9    9    9    9    9
```

All column sums are significant at all layers with maximum or near-maximum rank. CV
correlations are excellent (0.998-0.999 at L3). The model genuinely maintains intermediate
column totals in its activation space.

### L5 all: column sums remain strong

```
           L4  L6  L8  L12  L16  L20  L24  L28  L31
col_sum_0   9   9   9    9    9    9    9    9    9
col_sum_1   9   9   9    9    9    9    9    9    9
col_sum_2   8   8   9    8    8    8    8    8    8
col_sum_3   9   9   9    9    9    9    9    9    9
col_sum_4   9   9   9    9    9    9    9    9    9
```

The first run showed col_sum_2 and col_sum_3 with "reduced dimensionality" (dim_perm 4-5).
With 122K samples, those intermediate column sums now achieve dim_perm 8-9. The
"degradation at middle positions" was noise from the smaller dataset, not a real pattern.

### L5 correct: column sums hold up

```
           L4  L6  L8  L12  L16  L20  L24  L28  L31
col_sum_0   9   9   9    9    9    9    9    9    9
col_sum_1   9   9   9    8    8    8    8    8    8
col_sum_2   5   5   5    5    4    4    7    4    4
col_sum_3   9   9   9    9    7    9    9    9    9
col_sum_4   9   9   9    9    9    9    9    9    9
```

col_sum_2 (the middle column) does show genuinely reduced dimensionality in the correct
population (dim_perm 4-7 vs 8-9 in the all population). This is the one real "middle
position degradation" that survives the scale-up. The edge columns (0, 1, 4) are at
full or near-full rank. This is consistent with the answer digit pattern: the middle of
the carry chain is where computation is hardest.

---

## 18. Partial Product Subspaces

### Significance

All partial products are significant at all layers at all levels (L2-L5), with dim_perm
7-8 in both all and correct populations.

### L5 all: perfect

```
L5 layer 16 all:
  pp_a0_x_b0  dp=8  cv=1.000
  pp_a0_x_b1  dp=8  cv=1.000
  pp_a0_x_b2  dp=8  cv=1.000
  pp_a1_x_b0  dp=8  cv=1.000
  pp_a1_x_b1  dp=8  cv=1.000
  pp_a1_x_b2  dp=8  cv=1.000
  pp_a2_x_b0  dp=8  cv=1.000
  pp_a2_x_b1  dp=8  cv=1.000
  pp_a2_x_b2  dp=8  cv=0.998
```

With 122K samples, every partial product achieves near-perfect CV. The first run showed
pp_a1_x_b1 with CV = 0.775 — that was noise from the smaller dataset.

### L5 correct: CV degradation at higher-order products

```
L5 layer 16 correct:
  pp_a0_x_b0  dp=8  cv=0.998
  pp_a0_x_b1  dp=8  cv=0.965
  pp_a0_x_b2  dp=8  cv=0.952
  pp_a1_x_b0  dp=8  cv=0.971
  pp_a1_x_b1  dp=8  cv=0.966
  pp_a1_x_b2  dp=7  cv=0.912
  pp_a2_x_b0  dp=8  cv=0.903
  pp_a2_x_b1  dp=7  cv=0.924
  pp_a2_x_b2  dp=7  cv=0.859   <- lowest
```

pp_a2_x_b2 (hundreds × hundreds) has the lowest CV at 0.859. The products involving
the hundreds digit have the lowest cross-validation stability. This makes sense: the
hundreds digits have the most skewed distribution in the correct population
(a_hundreds=1 has 979 samples, a_hundreds=9 has 184). The centroids for rare
hundreds-digit values are noisier.

But note: dim_perm = 7 for pp_a2_x_b2. The subspace is still significant. The CV
degradation reflects centroid orientation noise, not absence of signal.

---

## 19. Derived Concept Subspaces

### correct (binary)

The correctness concept always has dim_perm = 1 and dim_consensus = 1. This is a single
direction in 4096 dimensions that separates correct from wrong activations. It exists at
every layer, at every level (where wrong answers exist).

### digit_correct_pos{j}

Per-position correctness shows a new pattern with the 10K scale-up at L3:

```
L3 all:
                       L4  L6  L8  L12  L16  L20  L24  L28  L31
digit_correct_pos0      1   1   1    1    1    0    0    0    0
digit_correct_pos1      1   1   1    1    1    1    1    1    1
digit_correct_pos2      1   1   1    1    1    1    1    1    1
digit_correct_pos3      1   1   1    1    1    1    1    1    1
```

digit_correct_pos0 (leading digit correctness) is now significant at early layers
(4-16) but loses significance at later layers (20-31). In the first run with 4K
samples, it was dim_perm = 0 everywhere. The 10K scale-up reveals that the model
does encode leading-digit correctness weakly at early layers, but this signal fades
as the network progresses. The model "decides" leading-digit correctness early, then
discards the information.

Middle/trailing digit correctness (pos1-3) is significant at all layers, consistent with
the first run. The model maintains per-digit correctness signals throughout the network
for the harder positions.

### product_binned

Product magnitude has dim_perm = 9 (maximum) at all layers and levels. However,
dim_consensus is only 2, because the first two eigenvalues capture 90%+ of variance.
Product magnitude is essentially a 2D concept with a strong primary direction (overall
scale) and a weaker secondary direction.

### total_carry_sum

This is the highest-dimensional concept in the registry: dim_perm = 27 at L3 (28 groups
after filtering, up from 24 in the first run). It uses 27 of 27 available dimensions.
At L5, total_carry_sum has 68 groups (the most of any concept) and dim_perm = 33. The
carry complexity of a problem is spread across many directions.

---

## 20. Correct vs Wrong Divergence

### The method

For each concept with both correct and wrong subspaces, we compute the first principal
angle between them. A principal angle of 0 degrees means the subspaces are identical; 90
degrees means completely orthogonal (no shared directions).

### Results: 792 entries, 787 with angle < 60 degrees

The vast majority (99.4%) of concept pairs show strong alignment between correct and
wrong subspaces. The model uses mostly the same directions for correct and wrong answers,
but with detectable rotation.

### Top divergences at L3

| Layer | Concept | dim_correct | dim_wrong | Angle |
|-------|---------|-------------|-----------|-------|
| 24 | product_binned | 2 | 3 | 9.0° |
| 28 | product_binned | 2 | 3 | 9.7° |
| 4 | b_tens | 8 | 8 | 10.2° |
| 24 | total_carry_sum | 23 | 25 | 10.9° |
| 16 | product_binned | 2 | 3 | 11.6° |
| ... | | | | |
| 16 | n_answer_digits | 1 | 1 | 58.0° |
| 6 | n_answer_digits | 1 | 1 | 61.6° |
| 12 | n_answer_digits | 1 | 1 | 63.7° |
| 4 | n_answer_digits | 1 | 1 | 65.2° |
| 8 | n_answer_digits | 1 | 1 | 66.3° |

Key observations:

1. **Product magnitude has the smallest angle** (9-12 degrees). Correct and wrong answers
   use almost identical directions for product encoding. The model "knows" the magnitude
   regardless of whether it gets the exact digits right.

2. **Input digits have small angles** (10-20 degrees). The model reads input digits the
   same way for correct and wrong answers.

3. **n_answer_digits has the largest angles** (52-66 degrees). This is a binary concept
   (product has 3 vs 4 digits at L3). The single direction separating these two groups
   rotates substantially between correct and wrong populations.

4. **n_nonzero_carries** also has large angles (53-55 degrees at some layers). The
   carry-complexity direction differs meaningfully between correct and wrong answers.

### L5 correct vs wrong at layer 16

| Concept | dim_correct | dim_wrong | Angle |
|---------|-------------|-----------|-------|
| a_units | 9 | 9 | 10.8° |
| a_tens | 9 | 9 | 12.2° |
| a_hundreds | 8 | 8 | 12.8° |
| b_units | 6 | 9 | 14.2° |
| b_tens | 9 | 9 | 14.6° |
| b_hundreds | 8 | 8 | 17.6° |
| carry_0 | 4 | 8 | 18.0° |
| carry_2 | 3 | 4 | 13.0° |
| ans_digit_1_msf | 8 | 7 | 47.6° |
| ans_digit_3_msf | 6 | 8 | 40.3° |
| ans_digit_2_msf | 8 | 8 | 38.3° |
| n_nonzero_carries | 3 | 3 | 33.3° |
| col_sum_1 | 3 | 2 | 30.6° |

The gradient from input (10-18°) through intermediate (13-31°) to output (38-48°) is
clear. The model's representation diverges most at the answer-generation stage, not at
the input-encoding stage. This is where the computation fails for wrong answers.

### The dimensionality story at L5: first run vs rerun

First run (239 correct):
```
a_units     correct dim=3   wrong dim=9
carry_0     correct dim=0   wrong dim=8
col_sum_0   correct dim=2   wrong dim=8
```

This run (4,197 correct):
```
a_units     correct dim=9   wrong dim=9
carry_0     correct dim=4   wrong dim=8
col_sum_0   correct dim=4   wrong dim=8
```

The "compressed correct representations" from the first run were largely a sample-size
artifact. With 4,197 samples, correct dimensionalities are much closer to wrong
dimensionalities. But they are not identical: carries and column sums still show
lower dimensionality in the correct population (dim_correct 3-4 vs dim_wrong 2-8).
This may reflect genuine compression — the model uses a more efficient encoding for
problems it solves correctly — or it may reflect the skewed digit distribution in the
correct population creating less uniform centroids.

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
  4->6:   68.6°   <- large rotation (early layers, input encoding)
  6->8:   62.6°
  8->12:  65.6°
  12->16: 63.9°
  16->20: 53.5°   <- rotation decreasing
  20->24: 44.4°   <- convergence zone
  24->28: 35.9°   <- MOST STABLE transition
  28->31: 46.3°   <- slight divergence at output
```

This pattern holds for every concept examined:

1. **Layers 4-16: High rotation (60-69 degrees).** The representation is actively being
   transformed. Each layer applies a substantial rotation to the concept subspace.
2. **Layers 16-24: Decreasing rotation (44-54 degrees).** The representation is settling.
3. **Layers 24-28: Minimum rotation (36-40 degrees).** The most stable transition in the
   network. The representation has converged.
4. **Layer 28-31: Slight increase (46-53 degrees).** The final layer applies a modest
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
  4->6:   67.2°
  6->8:   62.4°
  8->12:  66.5°
  12->16: 65.6°
  16->20: 54.6°
  20->24: 43.7°
  24->28: 38.4°   <- still the most stable
  28->31: 53.2°
```

The trajectory is remarkably consistent across levels. The model uses the same
layer-by-layer processing strategy regardless of task difficulty.

---

## 22. The Failure Regime: L5

L5 (3-digit x 3-digit, 3.4% accuracy) reveals what breaks when the model fails.
The rerun with 17x more correct samples resolves the ambiguity from the first run:
we can now distinguish "absent because the sample was too small" from "absent because
the model genuinely doesn't encode it."

### What survives (confirmed with 4,197 correct samples)

- **Input digits**: fully significant at all layers for all populations.
  dim_perm = 8-9 everywhere. The model reads the input correctly even when it cannot
  compute the answer. This was uncertain in the first run; now it is confirmed.
- **Carries**: significant at all layers for all populations. dim_perm = 5-12. The
  carry representation persists through failure. The first run showed carries
  vanishing in the correct population — that was a sample-size artifact.
- **Column sums**: significant with full or near-full rank at all layers. Edge columns
  (0, 1, 4) are fully significant in the correct population. Middle column (col_sum_2)
  shows genuine dimensionality reduction.
- **Partial products**: significant (dim_perm 7-8) with strong CV even in the correct
  population (0.859-0.998).

### What fails (confirmed null results)

- **Middle answer digits** (positions 1-2 in correct): dim_perm = 0 at ALL nine layers.
  This is the key confirmed finding. With 4,197 samples (~300-900 per digit value
  depending on the digit), the model does not build detectable centroid separation
  for the second and third answer digits of problems it gets right. This is not a
  power issue.
- **Middle answer digits** (positions 2-3 in all/wrong): dim_perm = 0-2 at most layers.
  Even across 122K problems, the model barely represents these positions.
- **ans_digit_3** in correct population: significant at early layers (dim_perm 7-9 at
  L4-L12) but drops to dim_perm = 2 at layers 20-31. The model starts building this
  representation but cannot maintain it through the later layers.

### The significance table tells the story

```
L5 correct population (4,197 samples):
                    L4  L6  L8  L12  L16  L20  L24  L28  L31
a_units              9   9   9    9    9    9    9    9    9  <- full
carry_0              8   8   8    8    8    8    8    8    8  <- full
col_sum_0            9   9   9    9    9    9    9    9    9  <- full
pp_a0_x_b0           8   8   8    8    8    8    8    8    8  <- full
ans_digit_0 (lead)   4   5   8    8    8    8    8    8    8  <- builds up
ans_digit_1 (2nd)    0   0   0    0    0    0    0    0    0  <- ABSENT
ans_digit_2 (3rd)    0   0   0    0    0    0    0    0    0  <- ABSENT
ans_digit_3 (4th)    7   8   9    9    8    2    2    2    2  <- fades
ans_digit_4 (5th)    9   9   9    9    9    9    9    9    9  <- full
ans_digit_5 (units)  7   7   7    7    7    7    7    7    7  <- full
```

Compare with the first run (239 correct samples):
```
a_units              1   1   1    1    3    3    0    3    3  <- unstable
carry_0              2   1   1    2    0    0    0    0    0  <- vanished
ans_digit_0 (lead)   0   0   0    0    0    0    0    0    0  <- absent
```

The difference is stark. What looked like a model that could barely represent anything
in the correct population was actually a model with full representations — we just
couldn't detect them with 239 samples.

### The real finding at L5

The model maintains full input digit, carry, column sum, and partial product
representations for the problems it gets right. The computation pipeline is intact.
What breaks is the *output stage*: the model cannot convert its correct intermediate
representations into correct middle answer digits. The leading digit and the trailing
digits are fine (they depend on magnitude and simple modular arithmetic). The middle
digits — which require propagating carries through the full column chain — are where
the 96.6% failure rate manifests.

This is the key evidence for the project thesis: the model has the intermediate
representations (LRH is satisfied) but cannot compose them correctly for the hardest
output positions. Linear representation is necessary but insufficient.

---

## 23. What Changed From the First Run

### Changes that were real (confirmed by scale-up)

1. **Middle answer digits are genuinely absent** at L5 correct. dim_perm = 0 for
   positions 1-2 with 4,197 samples confirms the first run's finding.
2. **Edge answer digits are genuinely present.** Leading and units digits pass the
   permutation null with high dimensionality.
3. **Cross-layer alignment follows the same trajectory.** The convergence pattern
   (high rotation early, minimum at 24-28) is identical between runs.

### Changes that were artifacts (overturned by scale-up)

1. **"Vanishing carry subspaces" in L5 correct.** The first run showed carry_0 with
   dim_perm = 0 at layers 16+. Now carry_0 has dim_perm = 8 at all layers. The model
   does encode carries for correct problems — we just couldn't see it with 239 samples.
2. **"Compressed correct representations."** The first run showed correct subspaces
   using 2-5 dimensions where wrong used 7-9. With more data, correct dims are much
   closer to wrong dims (typically within 2-3 of each other).
3. **"Column sum degradation at middle positions" at L5.** With 122K samples, col_sum_2
   and col_sum_3 achieve dim_perm 8-9 in the all population. The "degradation" was
   noise. (Exception: col_sum_2 in the correct population does show genuine reduction.)
4. **"Partial product CV degradation" at L5.** The first run showed pp_a1_x_b1 with
   CV = 0.775. Now it's 1.000 in the all population. The CV instability was entirely
   from small sample sizes.
5. **"ans_digit_1_msf fails everywhere at L3."** With 10K samples, it now shows
   dim_perm = 3-9 at layers 8-28. The signal was there but too weak for 4K samples.

### The lesson

With small samples, Phase C systematically underestimates dimensionality. The permutation
null is conservative: it requires each eigenvalue to beat the 99th percentile of random
shuffles. With N = 239, the random shuffles themselves produce noisy eigenvalues, making
the null threshold high relative to the signal. Increasing N reduces both the signal noise
(centroids become more stable) and the null noise (shuffled eigenvalues concentrate), making
genuine signals detectable.

The practical implication: any dim_perm = 0 finding at L5 correct in the first run should
be treated as "undetectable at that sample size," not "absent." The rerurn resolved this
by increasing the correct population 17x.

---

## 24. Key Findings

### Finding 1: Input digits use maximum rank at all levels and layers

Every input digit concept uses the full available dimensionality (m-1) at every layer
from 4 to 31, at every level from 2 to 5, in all three populations. This is confirmed
for the L5 correct population with 4,197 samples. Cross-validation correlations are
0.985-1.000. The model allocates the maximum possible representation to input digit
identity, consistent with Fourier encoding (9 non-constant real basis functions for
Z/10Z).

### Finding 2: Answer digits have an edge-vs-middle asymmetry

The leading answer digit (magnitude-related) and the trailing digits (modular arithmetic)
have significant subspaces. Middle answer digits fail the permutation null entirely in
the L5 correct population — confirmed with 4,197 samples, not a power issue. The
asymmetry scales with problem length: L3 has 1 weak position, L4 has 2, L5 has 2-3.

### Finding 3: The model maintains full intermediate computation representations

Column sums, carries, and partial products all have significant subspaces at maximum
or near-maximum rank, even in the L5 correct population. The "vanishing subspaces" from
the first run were sample-size artifacts. The model builds and maintains a complete
intermediate representation of the long-multiplication algorithm for problems it gets
right.

### Finding 4: The computation fails at the output stage

Combining Findings 1-3: the model has input digits (full rank), intermediate
computations (full rank), but cannot produce middle output digits. The bottleneck is
not representation — it's composition. The model has the pieces but cannot assemble them
for the hardest positions. This supports the thesis that linear representation is
necessary but insufficient.

### Finding 5: Correct and wrong answers use mostly aligned subspaces

Principal angles between correct and wrong subspaces are 9-48 degrees. The largest
divergences are in answer digit representations (38-48 degrees), while input digits
show near-identical directions (10-18 degrees). The computation failure manifests in
the output encoding, not the input encoding.

### Finding 6: All subspaces follow the same layer trajectory

Every concept shows the same rotation profile: high rotation in early layers (60-69
degrees), decreasing through middle layers, minimum at 24->28 (36-40 degrees), slight
increase at 28->31. This is invariant across concepts, levels, and populations.

### Finding 7: Column sums are as strongly represented as input digits

Column sums (the pre-carry intermediate totals) achieve CV correlations of 0.998-1.000
at L3. They are significant at maximum rank across all layers. This is the first
demonstration of column sum encoding in a production-scale LLM for multiplication,
confirming the computational pathway proposed by Qian et al. (2024).

### Finding 8: The scale-up resolved every ambiguity

The first run left open whether L5 correct findings were real nulls or power failures.
The 17x increase in correct samples (239 → 4,197) answered definitively: input digits,
carries, column sums, and partial products are all fully represented. Only middle answer
digits are genuinely absent. This changes the narrative from "the model can barely
represent anything at L5" to "the model represents everything except the hardest outputs."

---

## 25. Output Files

### Data outputs

```
/data/user_data/anshulk/arithmetic-geometry/phase_c/
├── residualized/                      45 files, 21 GB
│   └── level{N}_layer{L}.npy         (N_problems, 4096) float32
├── subspaces/                         ~14,000 files total, 5.5 GB
│   └── L{N}/layer_{LL}/{pop}/{concept}/
│       ├── basis.npy                  (dim_consensus, 4096)
│       ├── eigenvalues.npy            (m,)
│       ├── null_eigenvalues.npy       (1000, m-1)
│       ├── metadata.json              All results + resume checkpoint
│       └── projected_all.npy          (N_pop, dim_consensus) — all samples
└── summary/
    ├── phase_c_results.csv            2,844 rows — master results table
    ├── correct_wrong_divergence.csv   792 rows — principal angles
    └── alignment_results.csv          2,528 rows — cross-layer angles
```

Total disk: 26 GB (21 GB residualized cache, 5.5 GB subspaces).

### Plot outputs

```
/home/anshulk/arithmetic-geometry/plots/phase_c/
├── eigenvalue_spectra/                546 plots
├── dimensionality_heatmaps/           11 plots
├── cross_layer_trajectories/          316 plots
└── correct_wrong_comparison/          88 plots
                                       ─────
                                       961 total
```

---

## 26. Runtime and Reproducibility

| Property | Value |
|----------|-------|
| SLURM Job ID | 6659197 |
| Node | babel-s9-16 |
| Start time | 2026-03-19 20:38:20 EDT |
| End time | 2026-03-19 23:59:51 EDT |
| Total runtime | 201 minutes (3h 21m) |
| Exit code | 0 |
| Partition | general |
| CPUs | 12 |
| Memory | 64 GB |
| GPU | A6000 (used for permutation null + residualization via CuPy) |
| Conda environment | geometry |
| Script | phase_c_subspaces.py |
| Config | config.yaml |
| Parallelization | n_jobs=1, GPU handles parallelism |
| RNG seed | 42 + level*100 + layer (deterministic per pair) |
| Permutations | 1,000 per concept |
| MIN_GROUP_SIZE | 20 |
| MIN_POPULATION | 30 |
| CUMVAR_THRESHOLD | 0.95 |
| RATIO_THRESHOLD | 5.0 |
| PERM_ALPHA | 0.01 |

### Previous run for reference

| Property | Value |
|----------|-------|
| SLURM Job ID | 6626413 |
| Node | babel-t9-28 |
| Date | 2026-03-17 |
| Runtime | 58 minutes |
| GPU | A6000 (allocated but unused — CPU-only computation) |
| Workers | 12 (joblib) |
| Data | 4K at L3/L4, 4K at L5 (239 correct) |

*End of document. All numbers verified against phase_c_results.csv, alignment_results.csv,
correct_wrong_divergence.csv, and metadata.json files as of March 20, 2026.*
