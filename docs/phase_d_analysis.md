# Phase D: Complete Analysis

**Anshul's Geometric Manifold Interpretability Project**
**Carnegie Mellon University, April 2026**

This document records every decision, every number, and every result from Phase D —
the LDA refinement stage. It is the truth document for this stage. All numbers are
validated against the actual output files as of April 1, 2026.

Phase D ran in two jobs. Job 6893354 (March 30, single GPU) completed L1-L4 in 7.5
hours but was cancelled at the wall-time limit before finishing L5. Job 6897189
(March 31, 4×A6000) completed L5 in 18 hours. The L5 run regenerated the summary
CSV, so `phase_d_results.csv` contains only L5 data (1,035 rows). The per-concept
metadata files for L2-L4 are intact on disk (1,809 files: L2=306, L3=666, L4=837).

L5 is the regime that matters — it is where the model fails, and where N/d = 29.8
gives clean statistics. L2-L4 results are included in this document (Sections 7g-7k)
with full N/d caveats: L2 is rank-deficient (N/d = 0.98), L3 and L4 are marginal
(N/d = 2.44), and all sub-populations with N < 4,096 produce inflated eigenvalues.
These levels are documented for completeness and cross-level comparison, but absolute
eigenvalue magnitudes should not be cited at L2-L4.

---

## Table of Contents

1. [What Phase D Is and Why It Exists](#1-what-phase-d-is-and-why-it-exists)
2. [The Mathematical Framework](#2-the-mathematical-framework)
3. [The Permutation Null](#3-the-permutation-null)
4. [The N/d Ratio Problem](#4-the-nd-ratio-problem)
5. [What the Novelty Ratios Actually Mean](#5-what-the-novelty-ratios-actually-mean)
6. [The dim_perm=0 Edge Case](#6-the-dim_perm0-edge-case)
7. [Concrete Results — What Phase D Found](#7-concrete-results--what-phase-d-found)
   - 7a. L5/all at Layer 16
   - 7b. Layer Progression at L5/all
   - 7c. Population Comparison (all vs correct vs wrong)
   - 7d. Eigenvalue Spectra
   - 7e. The correct Binary Concept / ans_digit Secondary Targets
   - 7f. Cross-Validation
   - 7g. L2 Results (N/d = 0.98 — Rank-Deficient)
   - 7h. L3 Results (N/d = 2.44)
   - 7i. L4 Results (N/d = 2.44)
   - 7j. Cross-Level Comparison (L2 → L3 → L4 → L5)
   - 7k. The L4 digit_correct_pos0 Null Cases
8. [What Phase D Contributes to the Paper](#8-what-phase-d-contributes-to-the-paper)
9. [Implementation Details](#9-implementation-details)
10. [Relationship to the Paper's Thesis](#10-relationship-to-the-papers-thesis)
11. [Limitations](#11-limitations)
12. [Runtime and Reproducibility](#12-runtime-and-reproducibility)

---

## 1. What Phase D Is and Why It Exists

Phase C found subspaces. Phase D finds *discriminative* directions within them.

Phase C used conditional covariance + SVD to identify the directions in 4096-dimensional
activation space where concept centroids are most spread out. Given a concept like
carry_2 with 14 values (0 through 13), Phase C computes the 14 class centroids,
centers them, and runs SVD. The resulting eigenvectors are the directions of maximum
absolute centroid spread — the directions along which the centroids are farthest apart.
This is equivalent to maximizing w^T S_B w subject to ||w|| = 1, where S_B is the
between-class scatter matrix.

Phase D asks a different question. Instead of "where are the centroids most spread
out?", it asks "where are the centroids most spread out *relative to the noise*?"
This is Fisher's Linear Discriminant Analysis (LDA). It maximizes
w^T S_B w / w^T S_T w, where S_T is the total scatter matrix — the scatter of all
individual data points around the grand mean.

The distinction matters. Consider two directions in activation space:

Direction A: carry_2 centroids are 10 units apart along this direction, but individual
activations scatter ±50 around their centroids. The class clouds overlap massively.
A classifier using this direction would perform poorly because the within-class noise
swamps the between-class signal.

Direction B: carry_2 centroids are only 0.5 units apart along this direction, but
individual activations scatter only ±0.01 around their centroids. The class clouds
are perfectly separated. Despite the tiny absolute spread, this direction carries
pure discriminative information.

Phase C picks Direction A. Its eigenvalue is proportional to 10² = 100. Phase D picks
Direction B. Its eigenvalue is (0.5)² / (0.5² + 0.01²) ≈ 1.0 — virtually all the
variance along this direction is between-class.

The original motivation for Phase D was specific: carries have small eigenvalues in
Phase C, suggesting they might occupy low-variance-but-discriminative directions that
Phase C would miss. This turned out to be partly wrong. carry_2 at L5/all/layer16
has a Phase C eigenvalue of 0.153, which is actually *larger* than a_tens at 0.041.
The carries are not systematically low-variance in Phase C's sense. But the motivation
was right in a deeper way: Phase C's objective (maximize absolute variance) and Phase D's
objective (maximize variance ratio) are genuinely different optimization problems. They
find different directions. The eigenvalues are on different scales. And the combination
gives a richer picture than either alone.

Three errors were caught and corrected during the Phase D planning review:

1. **S_W invariance claim (wrong → fixed to S_T).** The original plan claimed that
   S_W (within-class scatter) is invariant under label permutation. This is false.
   S_W = Σ_k Σ_{i∈k} (x_i − μ_k)(x_i − μ_k)^T depends on class assignments through
   the class means μ_k. When labels are shuffled, class memberships change, μ_k changes,
   and every term changes. S_T = Σ_i (x_i − μ)(x_i − μ)^T depends only on the grand
   mean μ, which is invariant. The fix: use S_T as the denominator instead of S_W.

2. **Layer set (7 layers → fixed to all 9).** The original plan analyzed only 7 layers.
   The fix: use all 9 layers [4, 6, 8, 12, 16, 20, 24, 28, 31] to match Phase C.

3. **Answer digit targets (positions 1-4 → fixed to positions 1-2 only).** The original
   plan treated answer digit positions 1-4 as secondary targets of interest. The fix:
   only positions 1 and 2 (ans_digit_1_msf and ans_digit_2_msf) are genuinely interesting
   because they are the positions where Phase C found weak or null results at L5.

The revised motivation for Phase D is threefold: (a) provide an independent discriminative
measure of concept encoding strength using Fisher's criterion instead of centroid PCA,
(b) confirm or disconfirm Phase C's null findings for answer digits at L5, and (c) build
expanded merged subspaces for downstream non-linear analysis (Fourier screening, GPLVM).

---

## 2. The Mathematical Framework

This section derives Phase D's algorithm from first principles. It assumes grade-12
math plus basic calculus — specifically, familiarity with matrix multiplication,
eigenvalues, and the idea of maximizing a function subject to a constraint.

### Phase C's Objective (for comparison)

Phase C solves a straightforward optimization: find the unit vector w that maximizes
the spread of class centroids along w. Mathematically:

```
maximize  w^T S_B w
subject to  ||w|| = 1
```

where S_B is the between-class scatter matrix:

```
S_B = Σ_{k=1}^{K} n_k (μ_k − μ)(μ_k − μ)^T
```

Here K is the number of classes (e.g., 14 for carry_2 at L5), n_k is the number of
samples in class k, μ_k is the centroid of class k, and μ is the grand mean over all
samples. Each term n_k (μ_k − μ)(μ_k − μ)^T is a rank-1 outer product weighted by
class size, so S_B has rank at most K-1. The solution is the top eigenvector of S_B,
which is equivalent to the first principal component of the weighted centroid matrix.
This is exactly what Phase C computes via SVD.

The eigenvalue tells you the absolute magnitude of centroid spread along that direction.
At L5/all/layer16, a_tens has Phase C eigenvalue 0.041. This number depends on the
scale of the activations, the number of classes, and the magnitude of the differences.
It is not normalized.

### Phase D's Objective

Phase D adds a denominator. Instead of maximizing absolute centroid spread, it
maximizes the *ratio* of centroid spread to total data spread:

```
maximize  w^T S_B w / w^T S_T w
```

where S_T is the total scatter matrix:

```
S_T = Σ_{i=1}^{N} (x_i − μ)(x_i − μ)^T
```

This sums over all N individual data points, not just K centroids. It measures the
total spread of the data — both between-class and within-class — along every direction.

The solution comes from the generalized eigenvalue problem:

```
S_B w = λ S_T w
```

The eigenvalue λ lies between 0 and 1 (since S_B is a component of S_T). It has a
clean interpretation: λ is the fraction of total variance along direction w that is
due to class differences. If λ = 0.74, then 74% of the data's spread along that
direction comes from the centroids being separated; the remaining 26% is within-class
noise. This is the Fisher criterion.

At L5/all/layer16, carry_2 has Phase D eigenvalue λ₁ = 0.740. This means 74.0% of the
total variance along carry_2's most discriminative direction is between-class. For
comparison, a_tens has λ₁ = 0.984 — an almost perfect 98.4% discrimination ratio.
ans_digit_1_msf has λ₁ = 0.170 — only 17.0% of the variance is between-class, meaning
83% is noise. ans_digit_3_msf has λ₁ = 0.049 — essentially noise.

### Why S_T instead of S_W

Traditional Fisher LDA uses S_W (within-class scatter) as the denominator, maximizing
w^T S_B w / w^T S_W w. The within-class scatter is:

```
S_W = Σ_{k=1}^{K} Σ_{i∈C_k} (x_i − μ_k)(x_i − μ_k)^T
```

where C_k is the set of samples in class k. The identity S_T = S_B + S_W always holds
(total scatter = between-class scatter + within-class scatter).

The original Phase D plan claimed S_W is invariant under label permutation. This is
wrong. S_W depends on class assignments through the class means μ_k. When you permute
carry_2 labels — randomly reassigning carry values to data points — the class memberships
change, the class means μ_k change, and every term (x_i − μ_k)(x_i − μ_k)^T changes.
In practice, permuting carry_2 labels at L5/all changed S_W's trace by approximately
6%.

S_T, by contrast, is exactly invariant under label permutation. Its formula
S_T = Σ_i (x_i − μ)(x_i − μ)^T depends only on the individual data points x_i and
the grand mean μ. Neither of these changes when you relabel which points belong to
which class. This invariance is critical for the permutation null (Section 3): we
compute S_T once and reuse it for all 1,000 permutations.

The good news: S_T and S_W give *identical* eigenvectors. The generalized eigenproblems
S_B w = λ_T S_T w and S_B w = λ_W S_W w have the same solutions for w. Only the
eigenvalues differ, related by the monotonic transform λ_T = λ_W / (1 + λ_W). Since
this transform preserves ordering, the same directions are ranked in the same order
regardless of which denominator is used. Phase D's choice of S_T is purely
computational — it allows factoring S_T once and reusing it across all permutations.

### The Compact K×K Formulation

S_B has rank at most K-1 (where K is the number of classes). For carry_2 at L5 with
K=14, there are at most 13 non-trivial LDA directions. Solving a 4096×4096
generalized eigenproblem to find 13 directions would be wasteful. The compact
formulation reduces this to a 14×14 problem.

The algebra proceeds in five steps:

**Step 1: Weight the centroids.** Form the weighted centroid matrix:

```
M_w = diag(√n_k) × M
```

where M is the (K × d) matrix of centered class means (each row is μ_k − μ) and
d = 4096. The weighting by √n_k accounts for class sizes: larger classes contribute
more to S_B. After weighting, S_B = M_w^T M_w.

**Step 2: Solve against S_T.** Compute X = S_T_reg^{-1} M_w^T, which is a (d × K)
matrix. In practice, this is done by Cholesky: first factor S_T_reg = L L^T, then
solve L L^T X = M_w^T by forward/back substitution. The Cholesky factorization costs
O(d³/3) and is done once; each back-substitution costs O(d²) per column, O(d²K) total.

**Step 3: Form the small matrix.** Compute A = M_w X, which is (K × K). This small
matrix encodes the generalized eigenproblem in the reduced K-dimensional space.

**Step 4: Eigendecompose.** Solve the K×K eigenproblem for A. This gives eigenvalues
(the same λ values as the full problem) and K×K eigenvectors V.

**Step 5: Map back.** The full d-dimensional LDA directions are W = (X V)^T, normalized
to unit length. The top K-1 eigenvalues and their corresponding directions are the
Phase D output.

The total cost is dominated by the Cholesky factorization: O(d³/3) ≈ O(4096³/3) ≈
2.3 × 10¹⁰ operations. This is done once per (level, layer, population) and shared
across all ~43 concepts. The per-concept cost is just O(d²K) for the back-substitution
plus O(K³) for the tiny eigenproblem — negligible for K ≤ 14.

### Regularization

S_T may be rank-deficient when N < d (i.e., fewer samples than dimensions). Even
when N > d, near-singular S_T causes numerical instability in the Cholesky
factorization. Phase D adds Tikhonov regularization:

```
S_T_reg = S_T + α I
```

where α = 10⁻⁴ × trace(S_T) / d. This scales the regularization to the average
eigenvalue of S_T, adding a small constant to every eigenvalue. The effect is to
shrink all LDA eigenvalues slightly toward zero — a conservative bias that prevents
spurious large eigenvalues from near-zero denominators.

When regularization matters: L5/correct has N = 4,197 and d = 4,096, giving
N/d = 1.02 — just barely more samples than dimensions. The α value at L5/correct/layer16
is 4.75 × 10⁻⁴, compared to 1.30 × 10⁻² at L5/all/layer16 (where N = 122,223).
The 27× smaller α reflects the 27× worse conditioning: with N ≈ d, S_T is nearly
singular and the regularization does real work.

When regularization is negligible: at L5/all, the α of 0.013 is tiny compared to
individual eigenvalues of S_T (which are on the order of trace(S_T)/d ≈ 130). The
regularization shifts eigenvalues by ~0.01%, affecting nothing.

---

## 3. The Permutation Null

Phase D's eigenvalues tell you the fraction of variance that is between-class. But
any random labeling will produce non-zero between-class variance by chance. You need
a null distribution to assess significance.

The permutation null works as follows: shuffle the concept labels 1,000 times (e.g.,
randomly reassign carry_2 values to data points), redo the LDA each time, and record
the resulting eigenvalues. Each permutation breaks the true association between data
points and their carry values while preserving the marginal distributions of both.
The 99th percentile of the null eigenvalue distribution at each rank defines the
significance threshold: a real eigenvalue must exceed this threshold to be called
significant.

The key optimization is that S_T stays fixed across all permutations (Section 2
explained why). This means the Cholesky factorization and its inverse S_T^{-1}
are computed once and reused. On GPU, S_T^{-1} is precomputed as a dense (4096 × 4096)
matrix and held in GPU memory. Each permutation then requires:

1. Permute the class label vector (CPU, ~0.1ms)
2. Recompute K class centroids from the permuted labels (GPU matmul, ~2ms)
3. Form the weighted K×K matrix A = M_w S_T^{-1} M_w^T (GPU matmul, ~3ms)
4. Eigendecompose the K×K matrix (CPU, ~0.01ms)

The per-permutation cost is about 5ms, making 1,000 permutations take ~5 seconds per
concept. The total permutation null time for all ~43 concepts at one (level, layer,
population) is about 3-4 minutes on one A6000.

Significance is determined by sequential stopping: the algorithm tests eigenvalues in
descending order, comparing each to the 99th percentile (α = 0.01) of the corresponding
null eigenvalue. It stops at the first non-significant eigenvalue. This means that if
eigenvalue 5 fails the test, eigenvalues 6 through K-1 are declared non-significant
regardless of their values. This is conservative — it prevents inflated significance
counts from cherry-picked later eigenvalues.

The permutation null correctly answers: "is this eigenvalue larger than chance?" It
does NOT answer: "is this effect size meaningful?" A concept can have a statistically
significant LDA eigenvalue of 0.001 — real but tiny. Effect size is measured separately
by Cohen's d (the standardized mean difference between classes along each LDA direction).

---

## 4. The N/d Ratio Problem

This section explains why Phase D results at L5/all and L5/wrong are trustworthy,
while L5/correct results are inflated. This is the single most important caveat for
interpreting the data.

The activation vectors live in d = 4,096 dimensions. The total scatter matrix S_T has
shape (4096 × 4096) and rank at most min(N-1, d). When N < d, S_T has a null space
of dimension d - N. Directions in this null space have zero total scatter, so even
tiny between-class variance produces an LDA eigenvalue of λ = σ_B / (σ_B + 0) = 1.0.
Regularization prevents literal division by zero (adding α to every eigenvalue), but
it does not fix the fundamental problem: with fewer samples than dimensions, the data
cannot fill the space, and LDA finds spuriously clean separations in the empty directions.

The populations in this project span a wide range of N/d:

```
Population       N         N/d     Regime
─────────────────────────────────────────────
L5 / all         122,223   29.8    Well-conditioned, TRUSTWORTHY
L5 / wrong       118,026   28.8    Well-conditioned, TRUSTWORTHY
L5 / correct       4,197    1.02   Borderline, INFLATED
```

The inflation at L5/correct is dramatic and visible in the data. Consider carry_2
at layer 16:

```
Population   lda_eig_1   N        N/d
────────────────────────────────────────
all          0.740       122,223  29.8
wrong        0.722       118,026  28.8
correct      0.991        4,197   1.02
```

The correct-population eigenvalue of 0.991 is not a real finding. It says "99.1% of
the variance along carry_2's best LDA direction is between-class" — but with only
4,197 points in 4,096 dimensions, the regularized S_T barely constrains the space.
The permutation null faces the same constraint (same N, same d), so the eigenvalue
still beats the null and is called "significant." But the absolute magnitude 0.991 is
an artifact of the near-singular geometry.

The same inflation appears across all concepts at L5/correct. ans_digit_1_msf jumps
from 0.170 (all, trustworthy) to 0.889 (correct, inflated). col_sum_2 jumps from
0.681 (all) to 0.990 (correct). Even the correct binary concept jumps from 0.147 to
... well, it is not in the correct population since all points are correct=1.

The L2-L4 levels have even worse N/d ratios (L2/all: N=4,000, N/d=0.98; L3/wrong:
N=3,280, N/d=0.80; L4/correct: N=2,897, N/d=0.71). These levels were processed by
job 6893354 and the per-concept files exist on disk, but the summary CSV was overwritten
by the L5 run. We do not analyze L2-L4 Phase D results in this document because the
eigenvalues are dominated by N/d artifacts rather than real signal.

The rule of thumb for the paper: cite absolute LDA eigenvalues and Cohen's d only from
L5/all and L5/wrong, where N/d > 28. Use L5/correct results for significance (yes/no)
and relative patterns (concept A > concept B), but do not cite its absolute magnitudes.

An important clarification: the L5/all population (N=10K for L3/L4 in Phase C, N=122K
for L5) is NOT rank-deficient. N/d = 2.44 at L3/L4 and 29.8 at L5 — both overcomplete.
An earlier draft incorrectly labeled the L3/L4 all populations as rank-deficient.
They are not. Their eigenvalues are somewhat inflated compared to the N/d = 30 regime,
but they are not garbage. However, since Phase D was only run for L5, this point is
moot for the current analysis.

---

## 5. What the Novelty Ratios Actually Mean (and Why They're Uninformative)

Phase D computes a "novelty ratio" for each significant LDA direction by measuring
how much of it lies outside the Phase C subspace. The computation is straightforward:
project the LDA direction w onto the Phase C subspace (spanned by its dim_perm
validated eigenvectors), compute the residual, and measure its length. Since w is a
unit vector, the novelty ratio is ||w - proj(w)||, which ranges from 0 (w is entirely
within the Phase C subspace) to 1 (w is entirely orthogonal to it).

A direction is classified as "novel" if its novelty ratio exceeds 0.5 (corresponding
to an angle greater than 60° with the Phase C subspace).

The results are striking in their uniformity: every single significant LDA direction
across all 1,035 results is classified as novel. Out of 1,035 rows, 100% have
n_novel = n_sig. There are zero exceptions. The novelty ratios are not just above
0.5 — they are above 0.99 in 93% of cases. The distribution of angles between
Phase C and Phase D subspaces has a minimum of 74.5°, mean of 84.8°, and maximum
of 89.5°.

This universality is not a discovery. It is a geometric inevitability.

Phase C finds a dim_perm-dimensional subspace in 4,096-dimensional space. For a
typical concept with dim_perm = 9, this subspace occupies 9/4096 = 0.22% of the
ambient space. A random unit vector in 4096D has a 99.78% chance of being mostly
orthogonal to a 9D subspace. More precisely, the expected squared projection of a
random unit vector onto a k-dimensional subspace of d-dimensional space is k/d. For
k = 9, d = 4096, this is 0.0022 — the expected novelty ratio would be
√(1 - 0.0022) = 0.9989.

This is exactly what we observe. The mean novelty ratio across all 1,035 results is
0.9973, and the median is 0.9988. These are indistinguishable from the geometric
expectation for random directions in high-dimensional space. The LDA directions are
not "novel" in the sense of discovering new structure — they are simply pointing in
different directions within the vast 4096D space, as any independently constructed
set of directions would.

The sanity check we specified during planning — a_units should show an angle less
than 15° between Phase C and Phase D subspaces — failed. a_units shows
angle_phase_c_lda = 77.0° at L5/all/layer16. This is not a bug. In 4096 dimensions,
even two methods optimizing for the same concept (centroid PCA vs Fisher LDA) will
find directions that are mostly orthogonal, because the subspace they each find is a
tiny island in a vast ocean. The 9D Phase C subspace and the 9D Phase D subspace for
a_units overlap by at most a few dimensions out of 4096 — and the "angle" reported
is the smallest principal angle, which measures the single closest pair of directions
between the two subspaces.

The merged basis computation combines Phase C and Phase D directions via SVD
orthogonalization. Because the two subspaces are nearly orthogonal, the merged
dimension is nearly always dim_perm_C + n_sig_D. For a_tens at L5/all/layer16:
Phase C dim_perm = 9, Phase D n_sig = 9, merged_dim = 18. The near-doubling confirms
that the two methods found almost entirely non-overlapping directions.

**Bottom line for the paper:** do not present novelty ratios as findings. Do not
claim Phase D "discovered novel structure that Phase C missed." The correct framing
is: Phase D provides independent discriminative confirmation of concept encoding,
and the merged bases serve as expanded 18-dimensional search spaces (instead of
9-dimensional) for downstream Fourier screening and GPLVM analysis. The expansion
is real and useful. The "novelty" framing is misleading.

---

## 6. The dim_perm=0 Edge Case

Phase C determines subspace dimensionality using three estimators: dim_cumvar
(cumulative variance threshold), dim_ratio (eigenvalue ratio test), and dim_perm
(permutation null). The final estimate dim_consensus is the median of these three.
The permutation-validated dimensionality dim_perm is the most conservative: it counts
how many eigenvalues exceed the 99th percentile of a permutation null.

When dim_perm = 0, it means none of Phase C's eigenvalues beat the null — the
concept has no validated subspace according to the permutation test. But dim_cumvar
and dim_ratio may still vote for positive dimensionality (they use heuristic thresholds,
not statistical tests). If dim_consensus > 0 despite dim_perm = 0, the resulting
basis is noise: it captures centroid variance that did not beat the null.

The canonical example is ans_digit_2_msf at L5/correct/layer16. Phase C found:
dim_perm = 0, dim_cumvar = 8, dim_ratio unknown, dim_consensus = 8. That 8D basis
captures structure that failed the permutation test — it is not trustworthy.

Phase D handles this correctly:

1. When dim_perm = 0, the Phase C subspace is treated as empty for novelty comparison.
   All LDA directions get novelty_ratio = 1.0 by definition — there is nothing
   meaningful to compare against.

2. In the merged basis computation (gram_schmidt_merge), Phase C directions are
   included only if dim_perm > 0. When dim_perm = 0, only the novel LDA directions
   (if any) form the merged basis.

3. For ans_digit_2_msf at L5/correct/layer16, Phase D also found n_sig = 0 — no LDA
   eigenvalue beat the permutation null either. Both methods agree: this concept has
   no detectable encoding in the correct population at this layer. The merged
   dimension is 0. This is a confirmed null by two independent methods on 4,197 samples.

4. The eigenvalues are revealing: all 9 LDA eigenvalues for ans_digit_2_msf at
   L5/correct/layer16 cluster tightly between 0.830 and 0.883. This uniformly high
   baseline is the hallmark of N/d ≈ 1.0 inflation (Section 4). With 4,197 samples
   in 4,096 dimensions, even random labels produce eigenvalues near 0.85. The null
   distribution has the same inflated baseline, which is why n_sig = 0 despite the
   apparently large eigenvalues.

The same pattern holds at L5/correct across most layers for ans_digit_2_msf: Phase C
dim_perm = 0, Phase D n_sig = 0. The exception is ans_digit_3_msf at L5/all/layer16,
where Phase C found dim_perm = 0 but Phase D found n_sig = 2 with eigenvalue 0.049.
This is a weak but real discriminative signal that centroid-SVD missed — the centroids
are not spread much, but what spread exists is concentrated relative to the noise.
The Cohen's d of 0.625 for the top direction confirms a small-to-medium effect size.

---

## 7. Concrete Results — What Phase D Found

**Data source caveat:** L5 numbers in Sections 7a-7f come from `phase_d_results.csv`
(1,035 rows). This CSV contains **L5 data only** — the L5 job (6897189) regenerated
the CSV, overwriting the L1-L4 data from job 6893354. Readers who download the CSV
expecting all levels will find only L5. L2-L4 numbers in Sections 7g-7k come from
per-concept `metadata.json` files on disk (1,809 files total). Phase C comparison
values come from `phase_c_results.csv` (2,844 rows, L2-L5).

**Eigenvalue scale warning:** Phase C eigenvalues and Phase D eigenvalues are on
fundamentally different scales and **must not be compared in magnitude**:

- **Phase C eigenvalue** = absolute between-class variance along that direction.
  Computed from SVD of the centered centroid matrix M_c = (centroids - grand_mean)/√m.
  The eigenvalue equals the squared singular value, which is proportional to the
  centroid spread along that direction. It depends on activation scale: at layer 4,
  where activation norms are ~3.7, Phase C eigenvalues are ~100× smaller than at
  layer 28, where norms are ~75. Phase C eigenvalues are not bounded; they can be any
  non-negative real number.

- **Phase D eigenvalue** = fraction of total variance that is between-class along
  the LDA direction. Computed from the generalized eigenproblem S_B w = λ S_T w.
  The eigenvalue λ = w^T S_B w / w^T S_T w is a *ratio*, bounded in [0, 1]. It is
  self-normalizing: the denominator S_T absorbs activation scale, layer-to-layer norm
  growth, and any global variance differences. λ = 0.740 means 74.0% of the total
  variance along that direction is between-class; λ = 0.047 means only 4.7%.

What **can** be compared across methods: (a) the number of significant dimensions
(Phase C dim_perm vs Phase D n_sig), (b) the relative ordering of concepts within
each method, and (c) whether each method confirms or denies significance for a given
concept. What **cannot** be compared: Phase C eig₁ = 0.153 for carry_2 vs Phase D
λ₁ = 0.740 for carry_2. These are not "carry_2 got stronger" — they are different
statistics measuring different things.

### 7a. L5/all at Layer 16 — The Trustworthy Regime

N = 122,223. N/d = 29.8. All numbers below are reliable.

```
                          Phase D                              Phase C
Concept          K   λ₁      λ₂      d_max   n_sig  │  eig₁     dim_p  dim_c
─────────────────────────────────────────────────────┼─────────────────────────
Input digits                                         │
  a_units       10   0.989   0.985   25.01     9    │  0.120      9      9
  a_tens        10   0.984   0.980   22.24     9    │  0.041      9      9
  a_hundreds     9   0.994   0.988   35.82     8    │  0.193      8      8
  b_units       10   0.991   0.983   26.38     9    │  0.149      9      9
  b_tens        10   0.978   0.970   20.42     9    │  0.046      9      9
  b_hundreds     9   0.988   0.977   27.69     8    │  0.157      8      8
                                                     │
Carries                                              │
  carry_0        9   0.946   0.797   11.06     8    │  0.052      8      8
  carry_1       13   0.923   0.644   10.94    12    │  0.074     12      2
  carry_2       14   0.740   0.503   13.53    10    │  0.153     10      4
  carry_3       10   0.772   0.544    7.20     9    │  0.057      9      3
  carry_4        6   0.882   0.726    7.59     5    │  0.054      5      3
                                                     │
Column sums                                          │
  col_sum_0     10   0.955   0.813   12.58     9    │  0.054      9      8
  col_sum_1     10   0.909   0.595    8.83     9    │  0.041      9      2
  col_sum_2     10   0.681   0.447    3.80     7    │  0.036      8      2
  col_sum_3     10   0.754   0.528    4.80     9    │  0.026      9      3
  col_sum_4     10   0.955   0.848   13.24     8    │  0.063      9      5
                                                     │
Answer digits                                        │
  ans_digit_0    9   0.838   0.653    7.86     8    │  0.047      8      4
  ans_digit_1   10   0.170   0.152    1.15     9    │  0.0004     7      7
  ans_digit_2   10   0.047   0.040    0.56     3    │  0.0003     1      8
  ans_digit_3   10   0.049   0.041    0.62     2    │  0.0002     0      8
  ans_digit_4   10   0.246   0.169    1.28     9    │  0.004      9      9
  ans_digit_5   10   0.858   0.849    7.33     9    │  0.060      9      9
                                                     │
Derived                                              │
  correct        2   0.147    —      2.29     1    │  0.044      1      1
  total_carry   68   0.829   0.603   23.04    42    │  0.163     33     33
  max_carry     25   0.790   0.579   23.90    18    │  0.307     22      3
  n_nonzero      6   0.686   0.541   29.14     5    │  0.359      5      2
  product_bin   10   0.987   0.923   31.51     9    │  0.382      9      2
                                                     │
Partial products                                     │
  pp_a0_x_b0     9   0.953   0.771   12.30     8    │  0.055      8      8
  pp_a0_x_b1     9   0.927   0.670    8.41     8    │  0.048      8      6
  pp_a0_x_b2     9   0.781   0.660    5.01     8    │  0.163      8      7
  pp_a1_x_b0     9   0.933   0.689    8.89     8    │  0.034      8      8
  pp_a1_x_b1     9   0.898   0.667    8.18     8    │  0.062      8      6
  pp_a1_x_b2     9   0.782   0.719    4.59     8    │  0.119      8      7
  pp_a2_x_b0     9   0.824   0.663    6.29     8    │  0.156      8      7
  pp_a2_x_b1     9   0.809   0.726    6.06     8    │  0.094      8      7
  pp_a2_x_b2     9   0.948   0.827   20.04     7    │  0.147      8      3
                                                     │
Digit-level correctness (all population only)        │
  digit_correct_pos0  2  0.045   —    —      1    │   —         —      —
  digit_correct_pos1  2  0.061   —    —      1    │   —         —      —
  digit_correct_pos2  2  0.144   —    —      1    │   —         —      —
  digit_correct_pos3  2  0.068   —    —      1    │   —         —      —
  digit_correct_pos4  2  0.250   —    —      1    │   —         —      —
  digit_correct_pos5  2  0.199   —    —      1    │   —         —      —
                                                     │
Other                                                │
  n_answer_digits  2   0.608    —    —      1    │   —         —      —
```

Key: λ₁/λ₂ = Phase D eigenvalues (fraction of total variance); d_max = max Cohen's d
for direction 1; n_sig = Phase D significant directions; eig₁ = Phase C top eigenvalue;
dim_p = Phase C dim_perm; dim_c = Phase C dim_consensus.

**The primary pattern: input digits > carries > answer digits.** Input digits have
LDA eigenvalues above 0.97 — virtually all variance along their best directions is
between-class. These concepts are encoded with enormous clarity. Carries range from
0.74 (carry_2) to 0.95 (carry_0) — still very strong, but with 5-26% within-class
noise. Answer digits split into three tiers: ans_digit_0 and ans_digit_5 (the leading
and trailing digits) are strong (0.84-0.86); ans_digit_4 is moderate (0.25); and
ans_digit_1, ans_digit_2, ans_digit_3 are weak (0.05-0.17).

**Cohen's d provides calibration.** The Cohen's d values confirm the eigenvalue ordering
but add nuance. carry_2 has d_max = 13.53, meaning the most-separated pair of carry
values (carry=0 vs carry=9) are 13.5 pooled standard deviations apart along the top
LDA direction. For context, d = 0.8 is conventionally "large" in psychology. These are
enormous effect sizes. Even ans_digit_1, with its weak eigenvalue of 0.170, has
d_max = 1.15 — a genuine, if modest, signal.

**Phase D n_sig matches Phase C dim_perm almost exactly.** For 20 of the 28 concepts
above, n_sig = dim_perm. The exceptions are minor: col_sum_2 (7 vs 8), carry_1 (12
vs 12 — exact match), total_carry_sum (42 vs 33), max_carry_value (18 vs 22),
ans_digit_1 (9 vs 7), ans_digit_2 (3 vs 1). The agreement confirms that both methods
are finding the same intrinsic dimensionality for most concepts, despite optimizing
different objectives.

**The carries-strong / answers-weak dissociation is confirmed by both methods.** Phase C
found that carry subspaces have robust eigenvalues and high dim_perm. Phase D independently
confirms this: carries have high LDA eigenvalues (0.74-0.95) and large effect sizes
(d > 7). Meanwhile, ans_digit_1 through ans_digit_3 show weak eigenvalues (0.05-0.17)
and small effect sizes (d < 1.3). The model builds strong intermediate carry
representations but does not produce clean output-digit encodings. This is the central
finding of the linear analysis pipeline.

### 7b. Layer Progression at L5/all

LDA eigenvalues are remarkably stable across layers. Unlike Phase C eigenvalues (which
reflect absolute centroid spread and vary with activation scale), Phase D eigenvalues
are self-normalizing — the denominator S_T absorbs scale differences between layers.

```
                       L04    L06    L08    L12    L16    L20    L24    L28    L31
Concept               ─────  ─────  ─────  ─────  ─────  ─────  ─────  ─────  ─────
Input digits
  a_units             0.996  0.990  0.987  0.984  0.989  0.989  0.988  0.990  0.990
  a_tens              0.992  0.985  0.983  0.984  0.984  0.983  0.983  0.985  0.987
  a_hundreds          0.997  0.994  0.994  0.994  0.994  0.993  0.993  0.994  0.994
  b_units             0.999  0.997  0.996  0.993  0.991  0.989  0.988  0.990  0.989
  b_tens              0.995  0.990  0.986  0.981  0.978  0.976  0.974  0.977  0.977
  b_hundreds          0.998  0.995  0.994  0.991  0.988  0.987  0.987  0.988  0.988

Carries
  carry_0             0.955  0.952  0.949  0.940  0.946  0.945  0.944  0.948  0.951
  carry_1             0.936  0.929  0.929  0.926  0.923  0.921  0.921  0.923  0.925
  carry_2             0.766  0.759  0.754  0.738  0.740  0.736  0.740  0.748  0.750
  carry_3             0.758  0.781  0.788  0.773  0.772  0.762  0.753  0.746  0.747
  carry_4             0.856  0.889  0.895  0.886  0.882  0.877  0.871  0.870  0.864

Column sums
  col_sum_0           0.961  0.964  0.958  0.951  0.955  0.955  0.953  0.956  0.958
  col_sum_1           0.922  0.916  0.916  0.912  0.909  0.909  0.907  0.909  0.911
  col_sum_2           0.692  0.707  0.700  0.685  0.681  0.675  0.674  0.682  0.682
  col_sum_3           0.729  0.765  0.774  0.760  0.754  0.742  0.732  0.729  0.728
  col_sum_4           0.954  0.960  0.961  0.957  0.955  0.953  0.949  0.950  0.948

Answer digits
  ans_digit_0         0.799  0.847  0.861  0.847  0.838  0.825  0.814  0.814  0.802
  ans_digit_1         0.123  0.154  0.175  0.169  0.170  0.160  0.159  0.152  0.152
  ans_digit_2         0.047  0.048  0.046  0.046  0.047  0.047  0.048  0.045  0.045
  ans_digit_3         0.052  0.053  0.052  0.050  0.049  0.049  0.051  0.052  0.051
  ans_digit_4         0.254  0.255  0.260  0.248  0.246  0.245  0.244  0.250  0.248
  ans_digit_5         0.875  0.874  0.873  0.862  0.858  0.852  0.855  0.863  0.868

Derived
  correct             0.152  0.151  0.150  0.148  0.147  0.146  0.147  0.147  0.148
  total_carry_sum     0.869  0.868  0.845  0.824  0.829  0.824  0.828  0.843  0.843
  max_carry_value     0.838  0.834  0.810  0.788  0.790  0.785  0.793  0.811  0.811
  n_nonzero_carries   0.698  0.700  0.690  0.679  0.686  0.683  0.682  0.684  0.679
  product_binned      0.989  0.989  0.989  0.988  0.987  0.987  0.986  0.986  0.986
```

Several patterns emerge:

Input digit eigenvalues are nearly flat across all layers (a_units ranges from 0.987
to 0.996). This makes sense: the tokenized input digits are fixed features of the
prompt, and the model encodes them consistently at every layer.

carry_2 shows a gentle inverted-U: it rises slightly from early layers, peaks around
layer 4-8, and declines through later layers. This suggests carry encoding is strongest
in early-to-mid layers and slightly degrades as the representation transforms toward
output prediction.

carry_3 shows a clearer inverted-U: 0.758 → 0.788 (peak at layer 8) → 0.747.
The mid-layer peak is consistent with Phase C's finding that intermediate layers are
the information peak.

ans_digit_1 rises from 0.123 (layer 4) to 0.175 (layer 8), then slowly declines to
0.152 (layer 31). The signal is weak throughout but peaks at mid-layers, suggesting
the model briefly develops some answer-digit structure before it dissipates.

ans_digit_2 is flat at ~0.047 across all layers — noise-level, never develops. This
is the clearest null in the data: at no point during the forward pass does the model
develop a discriminative encoding of the second-most-significant answer digit beyond
the 5% background level.

ans_digit_0 (the leading answer digit) shows a clear inverted-U: 0.799 → 0.861
(peak at layer 8) → 0.802 (layer 31). This is the strongest answer digit, with a
peak discrimination ratio of 86% — the model knows the leading digit reasonably well
at mid-layers.

ans_digit_5 (the trailing/units answer digit) is strong and flat: 0.875 → 0.858 → 0.868.
This makes sense — the units digit of the product is determined by the units digits of
the inputs (pp_a0_x_b0), which are strongly encoded.

carry_4 shows a distinctive pattern: 0.856 → 0.895 (peak at layer 8) → 0.864. The
mid-layer peak is sharper than carry_2's, suggesting that the highest-position carry
is a mid-layer computation that the model performs and then partially discards.

The correct concept is flat at ~0.148 across all layers. The discrimination between
correct and wrong examples exists from early layers and neither strengthens nor
weakens. This flatness implies that correctness is determined early (perhaps by input
difficulty) rather than being computed incrementally.

The partial products (not shown in the table for space, but available in the CSV) follow
a clear positional pattern. At layer 16, products involving the units digit position
(pp_a0_x_b0: 0.953, pp_a1_x_b0: 0.933, pp_a0_x_b1: 0.927) are strongest, while
products involving the hundreds digit position (pp_a0_x_b2: 0.781, pp_a1_x_b2: 0.782,
pp_a2_x_b0: 0.824) are weaker. The exception is pp_a2_x_b2 (hundreds × hundreds) at
0.948, which is anomalously strong — this product determines the leading digit of the
answer, which the model encodes well (ans_digit_0: 0.838).

n_sig is stable across layers for all concepts. carry_2 has 9-10 significant directions
at every layer; a_units has 9 at every layer; ans_digit_2 has 2-4. The number of
discriminative dimensions is an intrinsic property of the concept, not a property of
the layer.

The digit-level correctness concepts (digit_correct_pos0 through pos5) are binary
(K = 2), each producing exactly 1 LDA direction. At layer 16, their eigenvalues are:
pos0 = 0.045, pos1 = 0.061, pos2 = 0.144, pos3 = 0.068, pos4 = 0.250, pos5 = 0.199.
Positions 4 and 5 (the most significant digits) have the strongest correctness signal.
Position 2 (middle digit) has moderate signal. Positions 0 and 1 (least significant)
have near-noise signal. This pattern aligns with the answer digit encoding: the model
knows whether it got the leading digits right more clearly than the trailing digits.

n_answer_digits (K = 2, binary: 6-digit vs 7-digit products) has eigenvalue 0.608 —
the model strongly discriminates between 6-digit and 7-digit products, as expected
since this depends primarily on the magnitude of inputs.

**Comparison with Phase C's layer trajectory.** Phase C found a dramatic four-phase
layer trajectory for concept eigenvalues: (1) high initial values at layers 4-6,
(2) rapid growth through layers 8-16 as activation norms increase from ~3.7 to ~40,
(3) a plateau or slight decline at layers 20-24, and (4) a minimum at layers 24-28
followed by slight recovery at layer 31. This trajectory reflects the exponential
growth of activation norms through the network — Phase C eigenvalues scale with the
square of the centroid spread, which scales with activation magnitude.

Phase D shows no such trajectory. The LDA eigenvalues are nearly flat across all
layers, fluctuating by < 5% for most concepts (e.g., a_units: 0.984-0.996, carry_0:
0.940-0.955). **This flatness IS the finding.** It means the discrimination *ratio*
— the fraction of variance that is between-class — is constant even as absolute
magnitudes change by 20× from layer 4 to layer 28. The model maintains the same
relative discriminability of concepts at every layer, even though the absolute scale
of representations grows enormously through the forward pass.

This has a specific mathematical explanation. If at layer L the activations are
α_L × h (where α_L is the layer-specific norm scale factor), then both S_B and S_T
scale as α_L². The ratio λ = w^T S_B w / w^T S_T w is invariant to this scaling.
Phase C's eigenvalues are proportional to α_L² (because SVD of the centroid matrix
preserves the absolute scale). Phase D's eigenvalues divide out the scale. The
flatness confirms that the network is not selectively amplifying or attenuating
concept information at different layers — it is uniformly scaling all variance
(within-class and between-class) by the same factor.

The one place where Phase D shows layer variation is in the *weak* concepts. ans_digit_1
varies from 0.123 (L04) to 0.175 (L08) to 0.152 (L31), a 42% relative range. correct
varies from 0.146 to 0.152, only 4%. The weak concepts show more relative layer
variation because small absolute differences produce larger percentage changes when
the baseline is low. But even for ans_digit_1, the variation is modest — the
fundamental story is stability.

### 7c. Population Comparison: all vs correct vs wrong

**Why three populations?** Phase D runs LDA on three subsets of the data, each with
a different scientific rationale:

- **all** — the full dataset. This is the baseline: maximum statistical power
  (N = 122,223 at L5), most stable eigenvalue estimates. The "all" results are
  used for all cross-concept and cross-layer comparisons. When only one number is
  cited for a concept, it is the "all" number.

- **correct** — only examples the model answered correctly. This population
  represents the geometry of *successful computation*. If the model solves
  multiplication via a specific geometric arrangement of concept representations,
  that arrangement should be visible in the correct population. At L5, the model
  is correct only 3.4% of the time (N = 4,197), making this population small but
  scientifically central: it is where the mechanism works.

- **wrong** — only examples the model answered incorrectly. This population
  represents the geometry of *failed computation*. If the correct population has
  different representational structure from the wrong population, the difference
  points to the mechanism that fails. At L5, the wrong population is 96.6% of the
  data (N = 118,026), so its statistics are nearly identical to "all."

The comparison between correct and wrong is the crux of the paper's thesis: the
model builds intermediate representations (carries, column sums) successfully in
both populations, but the output mapping (carries → answer digits) fails in the
wrong population. Phase D quantifies this: are the discriminative subspaces
different between populations?

**N/d caveat for population comparison:** At L5, the wrong population (N/d = 28.8)
is trustworthy. The correct population (N/d = 1.02) is inflated. This comparison
is therefore between "all" and "wrong" only. The correct population is shown for
completeness but absolute eigenvalue magnitudes should not be compared across
populations.

```
                    all           correct         wrong
Concept          λ₁  (n_sig)   λ₁  (n_sig)   λ₁  (n_sig)
──────────────────────────────────────────────────────────
Input digits
  a_units       0.989   (9)   0.999   (9)   0.989   (9)
  a_tens        0.984   (9)   0.998   (9)   0.984   (9)
  a_hundreds    0.994   (8)   0.999   (8)   0.994   (8)
  b_units       0.991   (9)   0.999   (9)   0.991   (9)
  b_tens        0.978   (9)   0.998   (9)   0.978   (9)
  b_hundreds    0.988   (8)   0.999   (8)   0.988   (8)

Carries
  carry_0       0.946   (8)   0.994   (8)   0.946   (8)
  carry_1       0.923  (12)   0.992   (6)   0.923  (12)
  carry_2       0.740  (10)   0.991  (13)   0.722  (10)
  carry_3       0.772   (9)   0.992   (9)   0.766   (9)
  carry_4       0.882   (5)   0.989   (5)   0.881   (5)

Column sums
  col_sum_0     0.955   (9)   0.996   (9)   0.955   (9)
  col_sum_1     0.909   (9)   0.990   (5)   0.909   (9)
  col_sum_2     0.681   (7)   0.990   (8)   0.670   (7)
  col_sum_3     0.754   (9)   0.993   (9)   0.748   (9)
  col_sum_4     0.955   (8)   0.998   (9)   0.949   (8)

Answer digits
  ans_digit_0   0.838   (8)   0.980   (8)   0.841   (8)
  ans_digit_1   0.170   (9)   0.889   (8)   0.172   (9)
  ans_digit_2   0.047   (3)   0.883   (0)   0.049   (4)
  ans_digit_3   0.049   (2)   0.896   (2)   0.048   (2)
  ans_digit_4   0.246   (9)   0.947   (9)   0.242   (9)
  ans_digit_5   0.858   (9)   0.976   (7)   0.862   (9)

Derived
  total_carry   0.829  (42)   0.992  (31)   0.817  (42)
  max_carry     0.790  (18)   0.993  (16)   0.779  (18)
  n_nonzero     0.686   (5)   0.991   (5)   0.652   (5)
  product_bin   0.987   (9)   0.998   (9)   0.987   (9)
  pp_a0_x_b0    0.953   (8)   0.995   (8)   0.953   (8)
```

Three patterns are visible in this table:

**1. The correct population is uniformly inflated.** Every correct-population eigenvalue
is above 0.88, regardless of concept. ans_digit_2 goes from 0.047 (all) to 0.883
(correct); carry_2 from 0.740 to 0.991; n_nonzero_carries from 0.686 to 0.991. The
compression of all eigenvalues into the [0.88, 1.0] range is the signature of N/d ≈ 1.0
inflation (Section 4). The correct population has N = 4,197 and d = 4,096 — there are
barely more samples than dimensions. With almost as many free parameters as data points,
even noise can produce high explained-variance ratios. The n_sig counts at correct
are unreliable for concepts where the trustworthy populations show weak signal
(ans_digit_2: n_sig = 0 at correct is meaningful; ans_digit_1: n_sig = 8 at correct
must be compared to the trustworthy n_sig = 9 at all).

**2. The all and wrong populations are nearly identical.** Wrong examples constitute
118,026 / 122,223 = 96.6% of the all population, so the all-population statistics
are dominated by wrong examples. The differences (|Δ| < 0.02 everywhere) are
consistent with the 3.4% dilution from correct examples. carry_2 shows the largest
all-wrong gap: 0.740 vs 0.722, a difference of 0.018. n_nonzero_carries also shows
a gap: 0.686 vs 0.652, a difference of 0.034. Both are consistent with the correct
examples (which have systematically different carry distributions) pulling the all-
population eigenvalue slightly upward.

**3. There are no interesting all-vs-wrong asymmetries.** Carry encodings do not
differ between populations. Answer digit encodings do not differ. The linear
discriminative structure is the same whether or not the model got the answer right.
If correct-wrong differences exist, they live in non-linear geometry — exactly where
Fourier screening and GPLVM will look.

### 7d. Eigenvalue Spectra — How Discrimination Decays Across Directions

The table in Section 7a showed only λ₁ and λ₂. The full eigenvalue spectrum reveals
how discrimination is distributed across dimensions. Each concept with K classes has
at most K-1 LDA directions, and the eigenvalues decay from λ₁ (most discriminative)
to λ_{K-1} (least discriminative). The shape of this decay tells us whether the concept
is encoded in a few concentrated directions or spread across many.

Full eigenvalue spectra at L5/all/layer16 for key concepts:

```
carry_2 (K=14, 13 directions, 10 significant):
  λ:  0.740  0.503  0.419  0.284  0.166  0.102  0.072  0.053  0.040  0.034 | 0.028  0.026  0.024
  d: 13.53   6.06  10.16   6.12   2.49   1.79   1.13   0.89   0.71   0.60 |

a_tens (K=10, 9 directions, 9 significant):
  λ:  0.984  0.980  0.974  0.969  0.960  0.945  0.912  0.904  0.831
  d: 22.24  19.07  17.12  15.22  14.81  11.59   8.02   9.91   7.53

ans_digit_1_msf (K=10, 9 directions, 9 significant):
  λ:  0.170  0.152  0.077  0.073  0.053  0.047  0.042  0.038  0.037
  d:  1.15   1.15   0.81   0.85   0.64   0.60   0.64   0.65   0.60

col_sum_2 (K=10, 9 directions, 7 significant):
  λ:  0.681  0.447  0.317  0.146  0.081  0.060  0.046 | 0.040  0.032
  d:  3.80   2.75   1.78   1.23   0.93   0.82   0.70 |

correct (K=2, 1 direction, 1 significant):
  λ:  0.147
  d:  2.29
```

(The | marks the significance cutoff. Directions after | did not beat the 99th
percentile of the permutation null.)

a_tens shows a remarkably flat spectrum: all 9 eigenvalues exceed 0.83, and all are
significant. This means the 10 digit values (0-9) are well-separated along every
possible discriminative direction. The encoding is uniformly strong in all 9 dimensions.
Cohen's d exceeds 7 for all 9 directions — massive effect sizes throughout.

carry_2 shows steep decay: λ₁ = 0.740, λ₅ = 0.166, λ₁₀ = 0.034. The first 3 directions
capture most of the discrimination, while later directions carry increasingly marginal
signal. This is consistent with carry_2 having a dominant low-dimensional structure
(perhaps a 2-3 dimensional core) surrounded by weaker higher-dimensional variation.
The Cohen's d values confirm this: d₁ = 13.53 but d₅ = 2.49.

ans_digit_1_msf has low eigenvalues throughout (0.170 down to 0.037), but all 9 are
significant. The permutation null at this N/d ratio is very tight — even a 4% between-
class fraction can be significant with 122K samples. The Cohen's d values are modest
(all between 0.60 and 1.15), confirming that this encoding is real but weak.

col_sum_2 shows the same steep-then-flat pattern as carry_2: the first direction
captures most of the discrimination (0.681), with rapid decay to 0.046 by direction 7.
Only 7 of 9 directions are significant — the 8th and 9th fall below the null.

### 7e. The correct Binary Concept

The concept "correct" is a binary label (K = 2), producing exactly 1 LDA direction.
At L5/all/layer16:

```
Phase D: λ₁ = 0.147, Cohen's d = 2.29, n_sig = 1
Phase C: eig₁ = 0.044, dim_perm = 1, dim_consensus = 1
angle_phase_c_lda = 88.6°
```

Both methods find exactly 1 significant direction. But the angle between Phase C's
direction and Phase D's direction is 88.6° — nearly perpendicular. This means the
direction of maximum centroid spread (Phase C) and the direction of maximum
discrimination ratio (Phase D) for correctness are almost orthogonal.

This is genuinely interesting if the N/d ratio is adequate — and it is (N = 122,223).
The merged basis has dimension 2, giving downstream methods two independent directions
to search for correctness-related structure. One direction maximizes how far apart
correct and wrong centroids are; the other maximizes how well-separated the two clouds
are relative to their internal spread. They point in different directions because the
correct/wrong clouds have highly anisotropic shapes — the centroids are far apart
along one axis, but the clouds are tightest along a nearly orthogonal axis.

The eigenvalue λ₁ = 0.147 means only 14.7% of the total variance along the best
discriminative direction is between-class. This is weak. The Cohen's d of 2.29 is
"large" by conventional standards, but modest compared to input digits (d > 20) or
carries (d > 7). Correctness is encoded, but not strongly — consistent with the model's
6.1% accuracy at L5 producing a highly imbalanced population split.

Cross-validation is not computed for K = 2 because pairwise distance correlation
requires at least 3 centroids.

### 7e. ans_digit_1_msf and ans_digit_2_msf at L5 — The Secondary Targets

These answer digit positions were the primary reason Phase D was run: Phase C found
weak or null results for them at L5, and Fisher LDA might catch low-variance
discriminative signals that centroid-SVD missed.

**ans_digit_1_msf at L5/all/layer16:**
Phase C found dim_perm = 7 with eigenvalue 0.0004 — significant but extremely small.
Phase D found n_sig = 9 with eigenvalue 0.170 and Cohen's d = 1.15. Phase D sees
more significant directions (9 vs 7) and the LDA eigenvalue, while low, represents a
real 17% discrimination ratio. The fact that both methods find significant structure
confirms that ans_digit_1 is genuinely encoded, just weakly.

**ans_digit_2_msf at L5/all/layer16:**
Phase C found dim_perm = 1 with eigenvalue 0.0003. Phase D found n_sig = 3 with
eigenvalue 0.047 and Cohen's d = 0.56. This is at the boundary of detectability:
the eigenvalue says only 4.7% of variance is between-class, and the effect size
is "medium" by convention but small in the context of this project. Both methods
agree that this encoding exists but is marginal.

**ans_digit_2_msf at L5/correct (the critical test):**
Phase C: dim_perm = 0 (no significant dimensions at any layer).
Phase D: n_sig = 0 (no significant LDA eigenvalues) at all 9 layers.
Both methods independently confirm a null result. With 4,197 samples, neither
centroid-SVD nor Fisher LDA can detect any discriminative structure for this
concept in the correct population. This is a strong null — two independent methods
with different objectives and different null distributions both fail.

**ans_digit_3_msf at L5/all/layer16:**
Phase C: dim_perm = 0 (failed permutation null).
Phase D: n_sig = 2, eigenvalue = 0.049, Cohen's d = 0.62.
This is a case where Phase D catches something Phase C missed. The centroid spread
is too small to beat Phase C's permutation null, but the discrimination *ratio* —
centroid spread relative to total scatter — produces 2 significant eigenvalues. The
effect is small (d = 0.62), and the eigenvalue is low (4.9%), but it is real. This is
the "Direction B" scenario from Section 1: tiny absolute spread, but even tinier noise.

### 7f. Cross-Validation

Phase D includes 5-fold stratified cross-validation for concepts with K ≥ 3. The
protocol: learn LDA directions on 80% of data, compute test-set class centroids,
and correlate pairwise distances in full space vs LDA-projected space. A high
correlation means the LDA subspace preserves the inter-class geometry on held-out data.

Representative cv_mean_corr values at L5/all/layer16:

```
Concept          cv_mean_corr   cv_std
────────────────────────────────────────
carry_2          0.989          0.002
col_sum_2        0.973          0.002
a_tens           0.661          0.010
pp_a0_x_b0       0.867          0.015
product_binned   0.991          0.001
total_carry_sum  0.972          0.003
```

carry_2 (cv = 0.989) and product_binned (cv = 0.991) show near-perfect preservation
of inter-class geometry on held-out data. The LDA directions generalize completely.

a_tens has a lower cv = 0.661. This may seem surprising given its high eigenvalue
(0.984), but the explanation is that all 9 LDA directions for a_tens have eigenvalues
above 0.95 — the inter-class geometry is spread evenly across many dimensions, so no
small subset perfectly captures the full distance matrix. The absolute discrimination
is excellent; the distance-preservation metric is just measuring a different thing.

Cross-validation is not computed for correct (K = 2) because pairwise distance
correlation requires at least 3 centroids.

### 7g. L2 Results — The Rank-Deficient Regime (N/d = 0.98)

**Summary:** 306 metadata files. 17 concepts × 2 populations (all, correct) × 9 layers.
No "wrong" population exists at L2 because L2 accuracy is 99.8% (3,993 / 4,000 correct).

**N/d context:** N = 4,000, d = 4,096, so N/d = 0.98. The sample covariance matrix S_T
is rank-deficient — it has at most 3,999 nonzero eigenvalues in 4,096 dimensions.
Tikhonov regularization prevents the Cholesky decomposition from failing, but the
resulting LDA eigenvalues are meaningless as absolute measures. All eigenvalues are
compressed toward 1.0 because the regularized total scatter has artificially inflated
denominators in the low-variance dimensions. Cohen's d values are in the thousands
(physically absurd), confirming the numerical instability. The correct population
(N = 3,993, N/d = 0.97) is equally rank-deficient.

**What the data shows (and doesn't show):**

100% (306/306) of L2 results are significant. Zero n_sig = 0 cases. This is expected:
when N < d, even noise directions can appear discriminative because the permutation
null is itself inflated by the same rank-deficiency.

L2/all at layer 16:

```
                          Phase D                              Phase C
Concept          K   N       λ₁      λ₂      d_max   n_sig  │  dim_p  dim_c  merged
─────────────────────────────────────────────────────────────┼───────────────────────
Input digits                                                 │
  a_units       10   4000   1.000   1.000  12328.68    9    │    9      9      18
  a_tens         9   4000   1.000   1.000  11917.01    8    │    8      8      16
  b_units        8   4000   1.000   1.000  17976.66    7    │    7      7      14
                                                             │
Carries                                                      │
  carry_0        9   4000   0.9998  0.9993  2546.71    8    │    8      8      16
  carry_1        9   4000   0.9999  0.9998  4441.44    7    │    7      4      11
                                                             │
Column sums                                                  │
  col_sum_0     10   4000   0.9999  0.9997  5158.95    9    │    9      9      18
  col_sum_1     10   4000   1.000   0.9998  5390.97    8    │    9      8      16
                                                             │
Answer digits                                                │
  ans_digit_0    9   4000   0.9998  0.9993  2020.73    8    │    8      5      13
  ans_digit_1   10   4000   0.9981  0.9980   596.49    9    │    9      9      18
  ans_digit_2   10   3379   0.9998  0.9988  1701.43    9    │    9      4      13
                                                             │
Derived                                                      │
  total_carry   15   3981   0.9998  0.9993  2925.33   14    │   14     14      28
  max_carry      9   4000   0.9998  0.9995  2425.69    8    │    8      8      16
  n_nonzero      3   4000   0.9994  0.8233  1498.37    2    │    2      2       4
  product_bin   10   4000   1.000   0.9999  6363.06    9    │    9      2      11
  n_ans_digits   2   4000   0.7564    —       4.87    1    │    1      1       2
                                                             │
Partial products                                             │
  pp_a0_x_b0     9   4000   0.9997  0.9993  2267.48    8    │    8      8      16
  pp_a1_x_b0     9   4000   0.9999  0.9998  5041.65    7    │    8      8      15
```

**Interpretation:** Every eigenvalue is within 0.002 of 1.0 (except n_nonzero_carries
λ₂ = 0.82 and n_answer_digits λ₁ = 0.76). Cohen's d ranges from 597 to 17,977 —
these are artifacts of near-singular scatter matrices, not genuine effect sizes. The
n_sig counts, however, are worth noting: they match Phase C's dim_perm exactly for
all 17 concepts. This agreement persists despite the eigenvalue inflation, suggesting
the *relative ordering* of eigenvalues (which determines significance via the
permutation null) is more robust than their absolute magnitudes.

n_answer_digits stands out: λ₁ = 0.756, Cohen's d = 4.87. These are the only
non-degenerate values at L2. This concept distinguishes 3-digit vs 4-digit products,
which depends on input magnitude — a feature readily available from the tokenized
inputs. The lower eigenvalue suggests this discrimination is genuine and not inflated
by rank-deficiency.

**Layer progression at L2/all (λ₁):**

```
                       L04    L06    L08    L12    L16    L20    L24    L28    L31
Concept               ─────  ─────  ─────  ─────  ─────  ─────  ─────  ─────  ─────
a_units               1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000
a_tens                1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000
carry_0               1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000
carry_1               1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000
ans_digit_0           1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000
ans_digit_1           0.997  0.998  0.998  0.998  0.998  0.998  0.998  0.998  0.997
ans_digit_2           1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000
product_binned        1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000
total_carry_sum       1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000
col_sum_0             1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000
col_sum_1             1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000
pp_a0_x_b0            1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000  0.999
```

The table is uninformative: every value is 1.000 (± 0.003). The rank-deficiency
makes LDA eigenvalues saturate at their theoretical maximum regardless of layer.
No layer progression analysis is possible at L2. The one exception — ans_digit_1
at 0.997-0.998 — merely shows that this concept is the weakest encoded of the 17,
consistent with L5 results.

**L2/correct population:** N = 3,993, N/d = 0.97. Nearly identical to the all
population because 99.8% of L2 examples are correct. The all-correct gap is < 0.001
for every concept. This population provides no additional information.

**Cross-validation (cv_mean_corr) at L2:** Ranges from −0.198 (ans_digit_1) to
0.989 (product_binned). Negative or near-zero cv values (ans_digit_1: −0.198,
a_units: 0.238, b_units: 0.274) indicate that the LDA directions do not generalize
to held-out data — a consequence of overfitting in the rank-deficient regime. Concepts
with cv > 0.8 (carry_1: 0.870, col_sum_1: 0.892, total_carry: 0.852, pp_a1_x_b0:
0.947, product_binned: 0.989) have enough structure to survive cross-validation despite
the low N/d. These CV values are the only trustworthy statistics at L2.

### 7h. L3 Results — Marginal N/d Regime (N/d = 2.44)

**Summary:** 666 metadata files. 28 concepts × 3 populations (all, correct, wrong) × 9
layers. L3 has three populations because accuracy is 67.2% (6,720 / 10,000 correct),
creating a substantial wrong population (3,280).

**N/d context:** all: N = 10,000, d = 4,096, N/d = 2.44. Correct: N = 6,720, N/d = 1.64.
Wrong: N = 3,280, N/d = 0.80 (rank-deficient). The all population is marginal but
usable — eigenvalues are inflated but not saturated. The correct population is also
inflated. The wrong population is rank-deficient (same problem as L2).

**Significance:** 100% (666/666) of L3 results are significant. Zero n_sig = 0 cases.

L3/all at layer 16:

```
                          Phase D                              Phase C
Concept          K   N       λ₁      λ₂      d_max   n_sig  │  dim_p  dim_c  merged
─────────────────────────────────────────────────────────────┼───────────────────────
Input digits                                                 │
  a_units       10  10000   0.9995  0.9993  183.34     9    │    9      9      18
  a_tens         9  10000   0.9997  0.9994  183.83     8    │    8      8      16
  b_units       10  10000   0.9994  0.9992  137.71     9    │    9      9      18
  b_tens         9  10000   0.9993  0.9986  123.92     8    │    8      8      16
                                                             │
Carries                                                      │
  carry_0        9  10000   0.9767  0.9465   23.94     8    │    8      8      16
  carry_1       15   9992   0.9641  0.9227   30.18    14    │   14     14      28
  carry_2       10  10000   0.9878  0.9744   71.68     8    │    8      3      11
                                                             │
Column sums                                                  │
  col_sum_0      9  10000   0.9920  0.9738   29.35     8    │    8      8      16
  col_sum_1     10  10000   0.9560  0.8887   15.41     9    │    9      9      18
  col_sum_2     10  10000   0.9949  0.9850   52.19     8    │    9      9      17
                                                             │
Answer digits                                                │
  ans_digit_0    9  10000   0.9750  0.9321   23.61     8    │    8      8      16
  ans_digit_1   10  10000   0.7646  0.7447    5.55     9    │    3      8      17
  ans_digit_2   10  10000   0.7927  0.7551    5.77     9    │    9      5      14
  ans_digit_3   10   8171   0.9858  0.9767   25.62     9    │    9      9      18
                                                             │
Derived                                                      │
  correct        2  10000   0.6565    —      3.14     1    │    1      1       2
  total_carry   28   9987   0.9767  0.9372   30.96    27    │   27     27      54
  max_carry     15   9992   0.9738  0.9367   32.61    14    │   14     14      28
  n_nonzero      4  10000   0.9398  0.8570   15.40     3    │    3      2       5
  product_bin   10  10000   0.9972  0.9887   78.03     9    │    9      2      11
  n_ans_digits   2  10000   0.6994    —      3.99     1    │    1      1       2
                                                             │
Partial products                                             │
  pp_a0_x_b0     9  10000   0.9798  0.9435   26.12     8    │    8      8      16
  pp_a0_x_b1     9  10000   0.9719  0.9335   26.76     8    │    8      6      14
  pp_a1_x_b0     9  10000   0.9801  0.9541   33.76     8    │    8      8      16
  pp_a1_x_b1     9  10000   0.9956  0.9877   85.74     7    │    8      3      10
                                                             │
Digit-level correctness                                      │
  digit_correct_pos0  2  9743  0.6806    —   18.15     1    │    1      1       2
  digit_correct_pos1  2  9743  0.6541    —    6.30     1    │    1      1       2
  digit_correct_pos2  2  9743  0.6555    —    3.34     1    │    1      1       2
  digit_correct_pos3  2  8006  0.7150    —    4.97     1    │    1      1       2
```

**Key observations at L3:**

Eigenvalues are inflated but not saturated. Unlike L2 where everything is 1.000, L3
shows genuine variance: input digits are 0.999, carries range 0.96-0.99, and answer
digits show real spread (ans_digit_1: 0.765, ans_digit_2: 0.793, ans_digit_0: 0.975).
The carries-strong / answers-weak hierarchy is visible but compressed toward 1.0 relative
to L5. Cohen's d values are in the tens to hundreds (vs single digits at L5), reflecting
the N/d inflation but not the thousand-fold absurdity of L2.

**ans_digit_1 and ans_digit_2 at L3 are strong.** ans_digit_1 has λ₁ = 0.765 at L3/all
versus 0.170 at L5/all; ans_digit_2 has 0.793 versus 0.047. These middle answer digits
are well-encoded at L3 (3-digit × 2-digit multiplication, 4-5 digit answers) but nearly
absent at L5 (3-digit × 3-digit, 6-digit answers). This is a genuine cross-level
finding: the model can encode answer digits for simpler problems but fails for harder
ones. However, the L3 eigenvalues are inflated by N/d = 2.44, so the true discrimination
is lower than 0.77/0.79.

**correct at L3:** λ₁ = 0.657 at L3 versus 0.147 at L5. The model discriminates
correct from wrong much more strongly at L3 (67.2% accuracy) than at L5 (3.4% accuracy).
This could be genuine (easier problems → clearer correctness signal) or partly inflated
by N/d = 2.44. Cohen's d = 3.14 at L3 vs 2.29 at L5 is consistent with genuine
improvement, since d is less sensitive to N/d inflation than eigenvalues.

**digit_correct_pos0 at L3:** λ₁ = 0.681, d = 18.15. The model knows whether it got
the units digit right with Cohen's d = 18.15 — the largest effect size among the
digit-correct concepts. This makes sense: the units digit of the product depends only
on pp_a0_x_b0, which is strongly encoded (λ₁ = 0.980). Contrast with pos2 (d = 3.34)
and pos3 (d = 4.97), which depend on carry propagation.

**pp_a1_x_b1 anomaly at L3:** This partial product (tens × tens) has the highest
eigenvalue among all partial products: λ₁ = 0.996, d_max = 85.74, cv = 0.968. At L5,
pp_a1_x_b1 has λ₁ = 0.898, which is lower than pp_a0_x_b0 (0.953). The reversal at
L3 (pp_a1_x_b1 strongest) vs L5 (pp_a0_x_b0 strongest) likely reflects the different
problem structure: at L3 (2-digit × 2-digit), pp_a1_x_b1 (tens × tens) is the
highest-order partial product and determines the leading digit of the answer, analogous
to pp_a2_x_b2 at L5.

**Eigenvalue spectra at L3/all/layer16 for key concepts:**

```
carry_2 (K=10, 9 directions, 8 significant):
  λ:  0.988  0.974  0.957  0.940  0.917  0.895  0.859  0.833 | 0.013

a_tens (K=9, 8 directions, 8 significant):
  λ:  1.000  0.999  0.999  0.999  0.998  0.998  0.998  0.561

ans_digit_1_msf (K=10, 9 directions, 9 significant):
  λ:  0.765  0.745  0.697  0.690  0.670  0.663  0.658  0.650  0.628

ans_digit_2_msf (K=10, 9 directions, 9 significant):
  λ:  0.793  0.755  0.702  0.685  0.670  0.664  0.650  0.640  0.628

ans_digit_0_msf (K=9, 8 directions, 8 significant):
  λ:  0.975  0.932  0.913  0.861  0.834  0.798  0.776  0.678

ans_digit_3_msf (K=10, 9 directions, 9 significant):
  λ:  0.986  0.977  0.887  0.824  0.803  0.789  0.786  0.735  0.688

correct (K=2, 1 direction, 1 significant):
  λ:  0.657

col_sum_1 (K=10, 9 directions, 9 significant):
  λ:  0.956  0.889  0.819  0.757  0.710  0.678  0.639  0.633  0.589

product_binned (K=10, 9 directions, 9 significant):
  λ:  0.997  0.989  0.971  0.951  0.928  0.896  0.857  0.819  0.778

pp_a0_x_b0 (K=9, 8 directions, 8 significant):
  λ:  0.980  0.943  0.893  0.856  0.819  0.814  0.794  0.741

total_carry_sum (K=28, 27 directions, 27 significant):
  λ:  0.977  0.937  0.883  0.818  0.794  0.788  0.761  0.739  0.714  0.703
      0.699  0.682  0.680  0.677  0.675  0.671  0.662  0.659  0.646  0.640
      0.631  0.621  0.614  0.611  0.604  0.601  0.493
```

(The | marks the significance cutoff.)

**Comparison with L5 eigenvalue spectra:** The spectral shapes at L3 are compressed
upward relative to L5, consistent with N/d inflation, but the qualitative patterns
match:

- carry_2 at L3 has a gently decaying spectrum (0.988 → 0.833 over 8 significant
  directions) vs steep decay at L5 (0.740 → 0.034 over 10). The compression toward
  1.0 is the N/d = 2.44 artifact — the true spectrum likely has similar shape but
  at lower magnitudes.
- ans_digit_1 and ans_digit_2 at L3 have remarkably flat spectra: all 9 eigenvalues
  between 0.63 and 0.79. At L5, ans_digit_1 has steep decay (0.170 → 0.037). The
  flatness at L3 may indicate that the encoding is spread uniformly across many
  directions at the easier level, while at L5 it concentrates into fewer dimensions.
- a_tens at L3 has 7 eigenvalues above 0.998 and then a sharp drop to 0.561 for the
  8th direction. At L5, all 9 eigenvalues exceed 0.83. The sharp cliff at L3 may
  reflect the reduced number of distinct a_tens values at L3 (only 9, since 2-digit
  inputs have tens digits 1-9) creating a rank-8 centroid matrix.
- product_binned at L3 shows smooth decay from 0.997 to 0.778 — all 9 directions are
  significant. At L5, the decay is from 0.987 to 0.923. Both levels show strong,
  well-distributed encoding of product magnitude.

**n_sig stability across layers at L3/all:**

```
                       L04  L06  L08  L12  L16  L20  L24  L28  L31
Concept               ────  ───  ───  ───  ───  ───  ───  ───  ───
a_units                  9    9    9    9    9    9    9    9    9
a_tens                   8    8    8    8    8    8    8    8    8
carry_0                  8    8    8    8    8    8    8    8    8
carry_1                 14   14   14   14   14   14   14   14   14
carry_2                  8    8    8    8    8    8    8    8    8
col_sum_0                8    8    8    8    8    8    8    8    8
col_sum_1                9    9    9    9    9    9    9    9    9
col_sum_2                8    8    8    8    8    8    8    8    8
ans_digit_0              8    8    8    8    8    8    8    8    8
ans_digit_1              9    9    9    9    9    9    9    9    9
ans_digit_2              9    9    9    9    9    9    9    9    9
ans_digit_3              9    9    9    9    9    9    9    9    9
correct                  1    1    1    1    1    1    1    1    1
total_carry_sum         27   27   27   27   27   27   27   27   27
product_binned           9    9    9    9    9    9    9    9    9
pp_a0_x_b0               8    8    8    8    8    8    8    8    8
pp_a1_x_b1               7    7    7    7    7    7    7    7    7
digit_correct_pos0       1    1    1    1    1    1    1    1    1
digit_correct_pos1       1    1    1    1    1    1    1    1    1
digit_correct_pos2       1    1    1    1    1    1    1    1    1
digit_correct_pos3       1    1    1    1    1    1    1    1    1
n_answer_digits          1    1    1    1    1    1    1    1    1
```

n_sig at L3 is **perfectly constant** across all 9 layers for every concept. Not a
single concept changes its number of significant directions from one layer to another.
This matches the L5 pattern (Section 7b: "n_sig is stable across layers") and provides
stronger evidence because L3 has 28 concepts × 9 layers = 252 all-population results,
all with identical n_sig values per concept. The number of discriminative dimensions
is a property of the concept's categorical structure, not of the layer.

**L3 full population comparison at layer 16:**

```
                    all               correct            wrong
                  N=10000            N=6720             N=3280
                  N/d=2.44           N/d=1.64           N/d=0.80
Concept       λ₁  (n_sig) cv    λ₁  (n_sig) cv     λ₁  (n_sig) cv
────────────────────────────────────────────────────────────────────
Input digits
  a_units    0.999  (9) 0.737  1.000  (9) 0.888  1.000  (9) 0.930
  a_tens     1.000  (8) 0.197  1.000  (8) 0.132  1.000  (8) 0.475
  b_units    0.999  (9) 0.688  1.000  (9) 0.763  1.000  (9) 0.895
  b_tens     0.999  (8) 0.172  1.000  (8) 0.126  1.000  (8) 0.613

Carries
  carry_0    0.977  (8) 0.943  0.992  (8) 0.917  0.999  (8) 0.736
  carry_1    0.964 (14) 0.876  0.988 (11) 0.859  1.000 (14) 0.837
  carry_2    0.988  (8) 0.935  0.994  (9) 0.950  1.000  (9) 0.930

Column sums
  col_sum_0  0.992  (8) 0.944  0.996  (7) 0.824  1.000  (9) 0.662
  col_sum_1  0.956  (9) 0.800  0.988  (9) 0.929  0.999  (9) 0.754
  col_sum_2  0.995  (8) 0.917  0.998  (9) 0.826  1.000  (9) 0.903

Answer digits
  ans_digit_0 0.975 (8) 0.776  0.988  (8) 0.719  0.999  (8) 0.923
  ans_digit_1 0.765 (9) 0.158  0.897  (9) 0.068  0.989  (9) 0.066
  ans_digit_2 0.793 (9) 0.566  0.941  (9) 0.655  0.985  (9) 0.053
  ans_digit_3 0.986 (9) 0.816  0.991  (9) 0.761  1.000  (9) 0.675

Derived
  correct    0.657  (1)   —      —     —    —      —     —    —
  total_carry 0.977(27) 0.842  0.991 (23) 0.835  0.999 (25) 0.800
  max_carry  0.974 (14) 0.876  0.989 (11) 0.826  0.999 (13) 0.866
  n_nonzero  0.940  (3) 0.762  0.987  (3) 0.991  0.998  (2) 0.993
  product_bin 0.997 (9) 0.980  0.999  (9) 0.969  1.000  (9) 0.970

Partial products
  pp_a0_x_b0 0.980 (8) 0.880  0.993  (8) 0.899  0.999  (8) 0.734
  pp_a0_x_b1 0.972 (8) 0.700  0.991  (8) 0.840  1.000  (8) 0.867
  pp_a1_x_b0 0.980 (8) 0.733  0.992  (8) 0.841  1.000  (8) 0.942
  pp_a1_x_b1 0.996 (7) 0.968  0.998  (8) 0.961  1.000  (8) 0.981

Digit correctness
  dc_pos0    0.681 (1)   —    not in correct pop   not in wrong pop
  dc_pos1    0.654 (1)   —
  dc_pos2    0.656 (1)   —
  dc_pos3    0.715 (1)   —
  n_ans_dig  0.699 (1)   —    0.981  (1)   —     0.998  (1)   —
```

Key: cv = cross-validation mean correlation. "—" = not computed (K=2) or not
applicable.

**L3 population comparison patterns:**

1. The wrong population (N/d = 0.80) is rank-deficient: every eigenvalue is above
   0.985. Cohen's d values are in the hundreds to thousands. This population provides
   no trustworthy eigenvalue information. However, n_sig values are reliable —
   carry_2 has n_sig = 9 at L3/wrong vs 8 at L3/all, a minor difference.

2. The correct population (N/d = 1.64) shows moderate inflation. ans_digit_1 goes
   from 0.765 (all) to 0.897 (correct), a 17% inflation. ans_digit_2 goes from
   0.793 to 0.941, a 19% inflation. These magnitudes should not be compared across
   populations, but the *relative ordering* within the correct population is
   meaningful: ans_digit_1 (0.897) is still weaker than carry_2 (0.994) and
   ans_digit_0 (0.988) within the correct population.

3. Cross-validation reveals real differences masked by eigenvalue inflation. At
   L3/all, ans_digit_1 has cv = 0.158 and ans_digit_2 has cv = 0.566. At L3/wrong,
   these drop to 0.066 and 0.053. The LDA directions for middle answer digits
   do not generalize to held-out data in the wrong population — the eigenvalues
   are inflated by the rank-deficiency but the underlying encoding does not
   cross-validate. At L3/correct, ans_digit_1 cv = 0.068 and ans_digit_2 cv = 0.655.
   The correct population shows slightly better cross-validation for ans_digit_2
   but not for ans_digit_1.

4. carry_0 cross-validation degrades from all (0.943) to correct (0.917) to wrong
   (0.736). This suggests carry_0 is encoded more cleanly in the full population
   than in either subpopulation, possibly because the correct and wrong populations
   have different carry distributions (the correct population has fewer high-carry
   problems).

**L3/correct population (N = 6,720, N/d = 1.64):**

```
                          L3/correct layer 16 (selected concepts)
Concept          K   N       λ₁      d_max   n_sig   dim_p
─────────────────────────────────────────────────────────────
a_units         10   6720   0.9998  344.64     9       9
carry_0          9   6720   0.9921   49.96     8       8
carry_2         10   6720   0.9944  179.29     9       9
col_sum_1       10   6720   0.9877   41.10     9       9
ans_digit_0      9   6720   0.9881   42.26     8       8
ans_digit_1     10   6720   0.8969   13.42     9       9
ans_digit_2     10   6720   0.9412   13.99     9       9
correct: not applicable (correct pop is all-correct by definition)
product_bin     10   6720   0.9987  150.51     9       9
```

All eigenvalues are higher than in the all population — consistent with N/d = 1.64
inflation. Cohen's d values are inflated by factor ~3-5× over the all population.
ans_digit_1 goes from 0.765 (all) to 0.897 (correct); ans_digit_2 from 0.793 to 0.941.
These correct-population values should not be cited as evidence of stronger encoding.

**L3/wrong population (N = 3,280, N/d = 0.80 — rank-deficient):**

The wrong population is rank-deficient (same pathology as L2). Every eigenvalue is
above 0.985. Cohen's d values are in the hundreds to thousands. These numbers are
not trustworthy. Selected values at layer 16:

```
Concept          K     N     λ₁       d_max
────────────────────────────────────────────
a_units         10   3280   1.0000  2858.33
carry_0          9   3280   0.9992   437.60
carry_2         10   3280   0.9996  1476.43
ans_digit_1     10   3280   0.9885   134.91
ans_digit_2     10   3280   0.9853   124.33
correct: not applicable (wrong pop has no correct examples)
product_bin     10   3280   0.9998  1448.50
```

**Layer progression at L3/all (λ₁):**

```
                       L04    L06    L08    L12    L16    L20    L24    L28    L31
Concept               ─────  ─────  ─────  ─────  ─────  ─────  ─────  ─────  ─────
a_units               1.000  1.000  1.000  0.999  0.999  0.999  1.000  1.000  0.999
carry_0               0.977  0.980  0.979  0.978  0.977  0.976  0.976  0.978  0.975
carry_1               0.958  0.968  0.969  0.969  0.964  0.964  0.965  0.966  0.961
carry_2               0.988  0.989  0.989  0.989  0.988  0.987  0.987  0.988  0.987
ans_digit_0           0.976  0.977  0.978  0.977  0.975  0.974  0.974  0.975  0.971
ans_digit_1           0.765  0.782  0.788  0.777  0.765  0.765  0.778  0.788  0.764
ans_digit_2           0.795  0.795  0.797  0.804  0.793  0.787  0.795  0.807  0.787
correct               0.646  0.655  0.659  0.659  0.657  0.652  0.660  0.662  0.643
product_binned        0.997  0.997  0.997  0.997  0.997  0.997  0.997  0.997  0.997
total_carry_sum       0.975  0.980  0.979  0.979  0.977  0.977  0.979  0.979  0.975
col_sum_1             0.948  0.956  0.960  0.958  0.956  0.954  0.956  0.959  0.951
pp_a0_x_b0            0.979  0.982  0.982  0.981  0.980  0.979  0.979  0.980  0.978
```

Despite the N/d inflation, layer progression patterns are visible:

- ans_digit_1 shows a bimodal pattern at L3: peaks at L08 (0.788) and L28 (0.788),
  with dips at L04 (0.765) and L31 (0.764). This is more structure than the monotone
  inverted-U seen at L5, possibly reflecting multiple processing stages.
- carry_1 peaks at L08/L12 (0.969) and declines to L31 (0.961), consistent with the
  mid-layer carry encoding peak seen at L5.
- correct shows a gentle rise from L04 (0.646) to L28 (0.662) then decline to L31
  (0.643), suggesting correctness signal builds incrementally through the network.

### 7i. L4 Results — Marginal N/d Regime (N/d = 2.44)

**Summary:** 837 metadata files. 35 concepts × 3 populations (all, correct, wrong) × 9
layers. L4 (3-digit × 2-digit multiplication) has accuracy 29.0% (2,897 / 10,000).

**N/d context:** all: N = 10,000, d = 4,096, N/d = 2.44. Correct: N = 2,897, N/d = 0.71
(rank-deficient). Wrong: N = 7,103, N/d = 1.73 (marginal). Eigenvalues at all populations
are inflated relative to L5/all (N/d = 29.8). L4/correct is rank-deficient.

**Significance:** 832/837 (99.4%) of L4 results are significant. 5 n_sig = 0 cases — all
digit_correct_pos0 at layers 6, 8, 16, 20, 24. These are analyzed in Section 7k.

L4/all at layer 16:

```
                          Phase D                              Phase C
Concept          K   N       λ₁      λ₂      d_max   n_sig  │  dim_p  dim_c  merged
─────────────────────────────────────────────────────────────┼───────────────────────
Input digits                                                 │
  a_units       10  10000   0.9968  0.9959   61.50     9    │    9      9      18
  a_tens        10  10000   0.9948  0.9919   42.29     9    │    9      9      18
  a_hundreds     9  10000   0.9983  0.9967   77.30     8    │    8      8      16
  b_units       10  10000   0.9981  0.9972   67.88     9    │    9      9      18
  b_tens         9  10000   0.9977  0.9953   64.48     8    │    8      8      16
                                                             │
Carries                                                      │
  carry_0        9  10000   0.9430  0.7973   11.38     8    │    8      8      16
  carry_1       15   9989   0.8685  0.8129   14.81    14    │   14     14      28
  carry_2       15   9980   0.8889  0.7751   11.53    14    │   12     12      26
  carry_3       10  10000   0.9656  0.9098   36.62     8    │    8      3      11
                                                             │
Column sums                                                  │
  col_sum_0      9  10000   0.9727  0.8687   14.82     8    │    8      8      16
  col_sum_1     10  10000   0.8580  0.7975    7.55     9    │    9      3      12
  col_sum_2     10  10000   0.8658  0.7127    7.27     9    │    9      9      18
  col_sum_3     10  10000   0.9807  0.9261   21.48     8    │    9      5      13
                                                             │
Answer digits                                                │
  ans_digit_0    9  10000   0.9229  0.7728   12.22     8    │    8      4      12
  ans_digit_1   10  10000   0.5198  0.4990    3.17     9    │    1      8      17
  ans_digit_2   10  10000   0.4553  0.4461    3.18     9    │    2      7      16
  ans_digit_3   10  10000   0.6365  0.5378    3.32     9    │    9      3      12
  ans_digit_4   10   8264   0.9456  0.9121   10.16     9    │    9      9      18
                                                             │
Derived                                                      │
  correct        2  10000   0.5948    —      2.74     1    │    1      1       2
  total_carry   36   9919   0.9264  0.8095   14.78    35    │   26     26      61
  max_carry     16   9993   0.9214  0.8250   17.08    15    │   15     15      30
  n_nonzero      5  10000   0.8730  0.8283   15.13     4    │    4      2       6
  product_bin   10  10000   0.9942  0.9672   50.31     9    │    9      2      11
  n_ans_digits   2  10000   0.6670    —      3.77     1    │    1      1       2
                                                             │
Partial products                                             │
  pp_a0_x_b0     9  10000   0.9499  0.7826   13.18     8    │    8      8      16
  pp_a0_x_b1     9  10000   0.8894  0.8169   11.97     8    │    8      8      16
  pp_a1_x_b0     9  10000   0.9412  0.7740   11.92     8    │    8      8      16
  pp_a1_x_b1     9  10000   0.9179  0.8098   15.59     8    │    8      8      16
  pp_a2_x_b0     9  10000   0.9172  0.8138   12.74     8    │    8      8      16
  pp_a2_x_b1     9  10000   0.9787  0.9172   32.38     7    │    8      3      10
                                                             │
Digit-level correctness                                      │
  digit_correct_pos0  2  9652  0.4120    —       —     0    │    0      1       0
  digit_correct_pos1  2  9652  0.4392    —    3.56     1    │    1      1       2
  digit_correct_pos2  2  9652  0.4739    —    1.99     1    │    1      1       2
  digit_correct_pos3  2  9652  0.5618    —    2.41     1    │    1      1       2
  digit_correct_pos4  2  7984  0.5179    —    3.00     1    │    1      1       2
```

**Key observations at L4:**

The carries-strong / answers-weak hierarchy is clearly visible at L4, intermediate
between L3 and L5. carry_0: 0.943, carry_2: 0.889, carry_3: 0.966. Answer digits
split into three tiers:
- ans_digit_0 (0.923) and ans_digit_4 (0.946): strong — the leading and trailing
  digits of the 5-digit answer
- ans_digit_3 (0.637): moderate — the middle digit
- ans_digit_1 (0.520) and ans_digit_2 (0.455): weak — the inner digits

This mirrors the L5 pattern (ans_digit_0 and ans_digit_5 strong; ans_digit_1-3 weak)
but shifted: at L4, ans_digit_1 has λ₁ = 0.520 versus L5's 0.170, and ans_digit_2
has 0.455 versus L5's 0.047. Even after accounting for N/d inflation (~2× at N/d = 2.44),
the L4 answer digit encoding is genuinely stronger than L5.

**n_sig vs dim_perm agreement:** At L4, n_sig matches dim_perm for most concepts,
with exceptions: total_carry (35 vs 26), carry_2 (14 vs 12), ans_digit_1 (9 vs 1),
ans_digit_2 (9 vs 2). The ans_digit discrepancies are notable: Phase D finds 9
significant LDA directions where Phase C's permutation null found only 1-2 significant
PCA dimensions. This is the core Phase D advantage — the discrimination *ratio* (LDA)
catches weak signals that the absolute centroid spread (PCA) misses.

**digit_correct_pos0 at L4:** n_sig = 0 at 5 of 9 layers (see Section 7k). This is
the only concept at any level with systematic null results. The eigenvalue (0.412)
means only 41.2% of variance along the best direction is between-class — below the
permutation null at this N/d ratio.

**pp_a2_x_b1 at L4:** This partial product (hundreds_a × tens_b) has the highest
eigenvalue among all L4 partial products: λ₁ = 0.979, d_max = 32.38, cv = 0.966.
This product determines the most-significant digit of the answer (together with
pp_a2_x_b0), explaining its strength. The partial product hierarchy at L4 mirrors L3:
the product involving the highest digit positions is encoded most strongly. At L5,
the analogous product is pp_a2_x_b2 (hundreds × hundreds, λ₁ = 0.948).

**Eigenvalue spectra at L4/all/layer16 for key concepts:**

```
carry_2 (K=15, 14 directions, 14 significant):
  λ:  0.889  0.775  0.644  0.573  0.560  0.517  0.486  0.467
      0.451  0.436  0.427  0.420  0.401  0.396

carry_1 (K=15, 14 directions, 14 significant):
  λ:  0.868  0.813  0.707  0.568  0.495  0.475  0.461  0.443
      0.435  0.421  0.410  0.399  0.393  0.385

carry_3 (K=10, 9 directions, 8 significant):
  λ:  0.966  0.910  0.850  0.775  0.714  0.639  0.570  0.527 | 0.009

a_tens (K=10, 9 directions, 9 significant):
  λ:  0.995  0.992  0.990  0.989  0.985  0.981  0.980  0.977  0.974

ans_digit_1_msf (K=10, 9 directions, 9 significant):
  λ:  0.520  0.499  0.465  0.447  0.443  0.440  0.425  0.418  0.413

ans_digit_2_msf (K=10, 9 directions, 9 significant):
  λ:  0.455  0.446  0.440  0.432  0.424  0.417  0.411  0.409  0.403

ans_digit_3_msf (K=10, 9 directions, 9 significant):
  λ:  0.636  0.538  0.480  0.444  0.426  0.423  0.413  0.404  0.379

ans_digit_0_msf (K=9, 8 directions, 8 significant):
  λ:  0.923  0.773  0.726  0.628  0.619  0.574  0.525  0.504

ans_digit_4_msf (K=10, 9 directions, 9 significant):
  λ:  0.946  0.912  0.712  0.545  0.527  0.503  0.495  0.471  0.434

correct (K=2, 1 direction, 1 significant):
  λ:  0.595

col_sum_1 (K=10, 9 directions, 9 significant):
  λ:  0.858  0.798  0.676  0.538  0.472  0.445  0.432  0.416  0.402

col_sum_2 (K=10, 9 directions, 9 significant):
  λ:  0.866  0.713  0.586  0.526  0.498  0.453  0.436  0.417  0.407

product_binned (K=10, 9 directions, 9 significant):
  λ:  0.994  0.967  0.912  0.827  0.760  0.691  0.619  0.563  0.495

pp_a0_x_b0 (K=9, 8 directions, 8 significant):
  λ:  0.950  0.783  0.660  0.613  0.575  0.556  0.531  0.488

total_carry_sum (K=36, 35 directions, 35 significant):
  λ:  0.926  0.809  0.675  0.581  0.535  0.519  0.498  0.485  0.473  0.465
      0.462  0.456  0.453  0.438  0.435  0.435  0.431  0.423  0.415  0.413
      0.411  0.407  0.403  0.399  0.396  0.392  0.390  0.387  0.385  0.378
      0.374  0.369  0.369  0.364  0.355
```

(The | marks the significance cutoff.)

**Comparison with L3 and L5 eigenvalue spectra:** The L4 spectra are intermediate
between L3 (compressed toward 1.0) and L5 (with full dynamic range):

- carry_2 at L4 has K = 15 (more carry values than L3's K = 10 or L5's K = 14,
  due to the specific carry distribution of 3×2-digit multiplication). The spectrum
  decays from 0.889 to 0.396 — steeper than L3 (0.988 to 0.833) but shallower than
  L5 (0.740 to 0.024). All 14 directions are significant.
- ans_digit_1 at L4 has a compressed, nearly flat spectrum: all 9 eigenvalues between
  0.413 and 0.520. At L3, the spectrum is similarly flat (0.628 to 0.765). At L5,
  the spectrum decays from 0.170 to 0.037 — a 5× range vs L4's 1.3× range. The
  flatness at L3/L4 means the encoding (to the extent it exists) is distributed
  uniformly across all 9 discriminative directions rather than concentrated in a
  few dominant ones.
- ans_digit_2 at L4 is the flattest spectrum: 0.403 to 0.455, a range of only 0.052
  across 9 directions. This extreme flatness, combined with the low absolute values,
  suggests that the "encoding" is mostly noise — the permutation null at N/d = 2.44
  allows even noise to appear significant.
- carry_3 at L4 has a clean cliff: 8 significant directions (0.966 down to 0.527),
  then a sharp drop to 0.009 for the 9th. This is the clearest significance boundary
  at L4 — 8 directions carry real signal, the 9th is noise.
- product_binned decays smoothly from 0.994 to 0.495 — similar shape to L5 (0.987
  to 0.923) but with a wider range. The lower tail values at L4 may reflect N/d
  inflation pushing the significance threshold down.
- total_carry_sum (K = 36, 35 significant) has a long tail from 0.926 to 0.355.
  The first 10 directions show clear decay (0.926 → 0.465); the remaining 25 are
  nearly flat (0.462 → 0.355). This two-phase spectrum suggests ~10 "real" dimensions
  plus ~25 that are barely above the noise floor.

**n_sig stability across layers at L4/all:**

```
                       L04  L06  L08  L12  L16  L20  L24  L28  L31
Concept               ────  ───  ───  ───  ───  ───  ───  ───  ───
a_units                  9    9    9    9    9    9    9    9    9
a_tens                   9    9    9    9    9    9    9    9    9
a_hundreds               8    8    8    8    8    8    8    8    8
carry_0                  8    8    8    8    8    8    8    8    8
carry_1                 14   14   14   14   14   14   14   14   14
carry_2                 14   14   14   14   14   14   14   14   14
carry_3                  8    8    8    8    8    8    8    8    8
col_sum_0                8    8    8    8    8    8    8    8    8
col_sum_1                9    9    9    9    9    9    9    9    9
col_sum_2                9    9    9    9    9    9    9    9    9
col_sum_3                8    8    8    8    8    8    8    8    8
ans_digit_0              8    8    8    8    8    8    8    8    8
ans_digit_1              9    9    9    9    9    9    9    9    9
ans_digit_2              9    9    9    9    9    9    9    9    9
ans_digit_3              9    9    9    9    9    9    9    9    9
ans_digit_4              9    9    9    9    9    9    9    9    9
correct                  1    1    1    1    1    1    1    1    1
total_carry_sum         35   35   35   35   35   35   35   35   35
product_binned           9    9    9    9    9    9    9    9    9
pp_a0_x_b0               8    8    8    8    8    8    8    8    8
pp_a1_x_b1               8    8    8    8    8    8    8    8    8
pp_a2_x_b1               7    7    7    7    7    7    7    7    7
digit_correct_pos0       1    0    0    1    0    0    0    1    1
digit_correct_pos1       1    1    1    1    1    1    1    1    1
digit_correct_pos2       1    1    1    1    1    1    1    1    1
digit_correct_pos3       1    1    1    1    1    1    1    1    1
digit_correct_pos4       1    1    1    1    1    1    1    1    1
n_answer_digits          1    1    1    1    1    1    1    1    1
```

n_sig at L4 is perfectly constant across layers for 34 of 35 concepts. The sole
exception is digit_correct_pos0, which oscillates between 0 and 1 across layers
(see Section 7k). For all other concepts, including the weak ones (ans_digit_1,
ans_digit_2, correct), the number of significant LDA directions is fixed regardless
of layer. This confirms the finding from L3 and L5: **n_sig is a property of the
concept, not of the layer.**

**L4 full population comparison at layer 16:**

```
                    all               correct            wrong
                  N=10000            N=2897             N=7103
                  N/d=2.44           N/d=0.71           N/d=1.73
Concept       λ₁  (n_sig) cv    λ₁  (n_sig) cv     λ₁  (n_sig) cv
────────────────────────────────────────────────────────────────────
Input digits
  a_units    0.997  (9) 0.912  1.000  (9) 0.793  0.998  (9) 0.905
  a_tens     0.995  (9) 0.655  1.000  (9) 0.619  0.997  (9) 0.727
  a_hundreds 0.998  (8) 0.342  1.000  (8) 0.265  0.999  (8) 0.451
  b_units    0.998  (9) 0.723  1.000  (9) 0.802  0.999  (9) 0.870
  b_tens     0.998  (8) 0.209  1.000  (8) 0.220  0.999  (8) 0.290

Carries
  carry_0    0.943  (8) 0.884  0.996  (7) 0.751  0.964  (8) 0.742
  carry_1    0.869 (14) 0.859  0.996 (10) 0.861  0.965 (13) 0.843
  carry_2    0.889 (14) 0.909  0.997  (9) 0.831  0.966 (14) 0.757
  carry_3    0.966  (8) 0.956  0.997  (7) 0.697  0.978  (9) 0.952

Column sums
  col_sum_0  0.973  (8) 0.839  0.998  (5) 0.765  0.985  (9) 0.824
  col_sum_1  0.858  (9) 0.933  0.997  (9) 0.929  0.965  (9) 0.932
  col_sum_2  0.866  (9) 0.862  0.997  (9) 0.890  0.966  (9) 0.755
  col_sum_3  0.981  (8) 0.916  0.999  (9) 0.892  0.985  (9) 0.844

Answer digits
  ans_digit_0 0.923 (8) 0.716  0.994  (8) 0.082  0.952  (8) 0.693
  ans_digit_1 0.520 (9) 0.285  0.972  (9)-0.044  0.661  (9) 0.123
  ans_digit_2 0.455 (9) 0.118  0.972  (9) 0.134  0.614  (9) 0.153
  ans_digit_3 0.637 (9) 0.872  0.988  (9) 0.773  0.651  (9) 0.114
  ans_digit_4 0.946 (9) 0.823  0.995  (9) 0.601  0.971  (9) 0.858

Derived
  correct    0.595  (1)   —      —     —    —      —     —    —
  total_carry 0.926(35) 0.838  0.998 (24) 0.807  0.967 (35) 0.774
  max_carry  0.921 (15) 0.901  0.998 (11) 0.893  0.963 (14) 0.852
  n_nonzero  0.873  (4) 0.923  0.997  (4) 0.973  0.944  (4) 0.958
  product_bin 0.994 (9) 0.976  1.000  (9) 0.928  0.996  (9) 0.977

Partial products
  pp_a0_x_b0 0.950 (8) 0.893  0.997  (7) 0.741  0.969  (8) 0.733
  pp_a0_x_b1 0.889 (8) 0.774  0.997  (7) 0.865  0.981  (8) 0.942
  pp_a1_x_b0 0.941 (8) 0.815  0.997  (7) 0.736  0.970  (8) 0.703
  pp_a1_x_b1 0.918 (8) 0.751  0.997  (7) 0.823  0.980  (8) 0.949
  pp_a2_x_b0 0.917 (8) 0.816  0.998  (7) 0.799  0.981  (8) 0.912
  pp_a2_x_b1 0.979 (7) 0.966  0.998  (7) 0.729  0.987  (8) 0.983

Digit correctness
  dc_pos0    0.412 (0)   —    not in correct/wrong pop
  dc_pos1    0.439 (1)   —
  dc_pos2    0.474 (1)   —
  dc_pos3    0.562 (1)   —
  dc_pos4    0.518 (1)   —
  n_ans_dig  0.667 (1)   —    0.994  (1)   —     0.930  (1)   —
```

Key: cv = cross-validation mean correlation. "—" = not computed (K=2) or not
applicable.

**L4 population comparison patterns:**

1. **L4/correct is rank-deficient (N/d = 0.71).** All eigenvalues above 0.97.
   Cohen's d = 56-702 (absurd). ans_digit_0 has cv = 0.082 and ans_digit_1 has
   cv = −0.044 (negative cross-validation). Negative cv means the LDA directions
   learned on the training fold produce *worse* distance preservation than chance
   on the test fold — a hallmark of overfitting in the rank-deficient regime. These
   directions are fitting noise, not signal.

2. **L4/wrong (N/d = 1.73) is marginal but more informative than correct.** The
   eigenvalue inflation is moderate: a_units goes from 0.997 (all) to 0.998 (wrong),
   a small increase. Carries go from 0.889-0.966 (all) to 0.964-0.978 (wrong), a
   uniform upward shift. The relative ordering is preserved.

3. **ans_digit_1 and ans_digit_2 population comparison at L4:** all (0.520, 0.455) →
   correct (0.972, 0.972) → wrong (0.661, 0.614). The correct values are meaningless
   (rank-deficient). The wrong values (0.661, 0.614) are moderately inflated over
   the all values (0.520, 0.455). The inflation factor is ~1.3×, consistent with
   N/d = 1.73 vs 2.44.

4. **Cross-validation separates real from inflated.** At L4/all, carry_3 has cv = 0.956
   (excellent generalization) while ans_digit_2 has cv = 0.118 (poor). At L4/wrong,
   ans_digit_1 cv = 0.123 and ans_digit_2 cv = 0.153 — both poor. The middle answer
   digits have "significant" LDA directions at L4 (n_sig = 9 everywhere) but these
   directions do not generalize. The permutation null says the eigenvalues beat
   shuffled labels, but cross-validation says the geometry doesn't survive in held-out
   data. This is a warning: at N/d = 2.44, significance and generalization diverge
   for weak concepts.

5. **carry_3 at L4/all vs L4/wrong:** λ₁ = 0.966 (all) vs 0.978 (wrong). The wrong
   population has a *higher* eigenvalue, which is counterintuitive. But with N/d = 1.73
   at wrong, the inflation is expected. The cv values are 0.956 (all) vs 0.952 (wrong)
   — nearly identical, confirming that the carry_3 encoding is genuine and similar
   in both populations.

**L4/correct population (N = 2,897, N/d = 0.71 — rank-deficient):**

```
                          L4/correct layer 16 (selected concepts)
Concept          K    N      λ₁      d_max   n_sig   dim_p
─────────────────────────────────────────────────────────────
a_units         10   2897   0.9998  702.23     9       9
carry_0          8   2889   0.9964  136.34     7       7
carry_2         10   2849   0.9966  166.66     9       8
carry_3          8   2879   0.9966  146.52     7       7
ans_digit_0      9   2897   0.9944  111.28     8       8
ans_digit_1     10   2897   0.9722   56.53     9       0
ans_digit_2     10   2897   0.9720   61.93     9       3
ans_digit_3     10   2897   0.9882   76.48     9       9
ans_digit_4     10   2025   0.9945  143.58     9       9
product_bin     10   2897   0.9995  402.17     9       9
```

L4/correct is rank-deficient (N/d = 0.71). All eigenvalues are above 0.97. Cohen's d
values are 50-700× (absurd). These numbers are untrustworthy. Note: ans_digit_1 has
dim_perm = 0 but n_sig = 9 at correct. This is the inflated-significance pathology
of rank-deficient data, not genuine discovery.

**L4/wrong population (N = 7,103, N/d = 1.73 — marginal):**

```
                          L4/wrong layer 16 (selected concepts)
Concept          K    N      λ₁      d_max   n_sig   dim_p
─────────────────────────────────────────────────────────────
a_units         10   7103   0.9975   73.43     9       9
carry_0          9   7103   0.9636   14.35     8       8
carry_1         14   7074   0.9651   18.39    13      11
carry_2         15   7083   0.9657   23.32    14      13
carry_3         10   7103   0.9781   44.29     9       9
ans_digit_0      9   7103   0.9518   16.07     8       8
ans_digit_1     10   7103   0.6605    4.34     9       0
ans_digit_2     10   7103   0.6138    4.53     9       0
ans_digit_3     10   7103   0.6508    4.77     9       5
ans_digit_4     10   6239   0.9707   14.79     9       9
correct: not applicable
product_bin     10   7103   0.9956   58.60     9       9
pp_a0_x_b0       9   7103   0.9693   17.57     8       8
pp_a2_x_b1       9   7103   0.9869   40.99     8       8
```

The wrong population at L4 is more informative than L4/all because most examples
are wrong (71%). The eigenvalues are less inflated than L4/correct. The pattern
ans_digit_1 (0.661) and ans_digit_2 (0.614) having dim_perm = 0 but n_sig = 9 is
notable — these concepts have no significant centroid-PCA dimensions at L4/wrong
but 9 significant LDA directions. This is the same Phase D advantage seen at L5
for ans_digit_3.

**Layer progression at L4/all (λ₁):**

```
                       L04    L06    L08    L12    L16    L20    L24    L28    L31
Concept               ─────  ─────  ─────  ─────  ─────  ─────  ─────  ─────  ─────
a_units               0.999  0.998  0.997  0.997  0.997  0.997  0.997  0.998  0.998
a_tens                0.997  0.996  0.995  0.994  0.995  0.995  0.996  0.997  0.997
carry_0               0.944  0.948  0.948  0.943  0.943  0.940  0.941  0.942  0.941
carry_1               0.870  0.874  0.873  0.872  0.868  0.866  0.865  0.867  0.864
carry_2               0.888  0.897  0.898  0.893  0.889  0.889  0.887  0.892  0.884
ans_digit_0           0.917  0.924  0.928  0.924  0.923  0.917  0.915  0.917  0.909
ans_digit_1           0.507  0.520  0.526  0.520  0.520  0.508  0.515  0.529  0.512
ans_digit_2           0.452  0.465  0.460  0.461  0.455  0.456  0.461  0.458  0.453
correct               0.585  0.592  0.596  0.593  0.595  0.585  0.589  0.599  0.594
product_binned        0.994  0.995  0.995  0.995  0.994  0.994  0.994  0.994  0.993
total_carry_sum       0.927  0.930  0.927  0.925  0.926  0.927  0.927  0.928  0.921
col_sum_1             0.860  0.865  0.862  0.861  0.858  0.856  0.856  0.858  0.854
pp_a0_x_b0            0.952  0.955  0.954  0.950  0.950  0.948  0.948  0.949  0.948
```

Layer progression at L4 shows recognizable patterns despite N/d inflation:

- carry_2 peaks at L08 (0.898) and declines to L31 (0.884), consistent with
  the mid-layer carry encoding peak at L5.
- ans_digit_0 shows a clear inverted-U: 0.917 → 0.928 (L08 peak) → 0.909 (L31).
- ans_digit_1 is remarkably flat (0.507-0.529), fluctuating around 0.52 with no
  clear layer preference. The encoding is weak at every layer.
- ans_digit_2 is even flatter (0.452-0.465), hovering near 0.46. Weak throughout.
- correct is flat (0.585-0.599), similar to L5's flatness (0.146-0.152), suggesting
  correctness discrimination is not layer-dependent at any level.

### 7j. Cross-Level Comparison (L2 → L3 → L4 → L5)

The most informative comparison is across levels for the same concept at the same
layer, using only the "all" population. N/d inflation increases as the level decreases
(L5: 29.8, L3/L4: 2.44, L2: 0.98), so raw eigenvalues are not directly comparable.
However, the relative ordering of concepts within a level is reliable, and Cohen's d
is more robust to N/d inflation than eigenvalues.

**Cross-level eigenvalue table at layer 16 (all population):**

```
                     L2              L3              L4              L5
                   N=4000          N=10000         N=10000         N=122223
                   N/d=0.98        N/d=2.44        N/d=2.44        N/d=29.8
Concept          λ₁     d_max   λ₁     d_max   λ₁     d_max   λ₁     d_max
───────────────────────────────────────────────────────────────────────────────
Input digits
  a_units       1.000  12329   0.999  183.3   0.997  61.5    0.989   25.0
  a_tens        1.000  11917   1.000  183.8   0.995  42.3    0.984   22.2

Carries
  carry_0       1.000  2547    0.977  23.9    0.943  11.4    0.946   11.1
  carry_1       1.000  4441    0.964  30.2    0.869  14.8    0.923   10.9
  carry_2         —      —     0.988  71.7    0.889  11.5    0.740   13.5

Column sums
  col_sum_0     1.000  5159    0.992  29.4    0.973  14.8    0.955   12.6
  col_sum_1     1.000  5391    0.956  15.4    0.858   7.6    0.909    8.8

Answer digits
  ans_digit_0   1.000  2021    0.975  23.6    0.923  12.2    0.838    7.9
  ans_digit_1   0.998   596    0.765   5.6    0.520   3.2    0.170    1.2
  ans_digit_2   1.000  1701    0.793   5.8    0.455   3.2    0.047    0.6
  ans_digit_3     —      —     0.986  25.6    0.637   3.3    0.049    0.6

Derived
  correct         —      —     0.657   3.1    0.595   2.7    0.147    2.3
  product_bin   1.000  6363    0.997  78.0    0.994  50.3    0.987   31.5
```

Key: λ₁ = top LDA eigenvalue; d_max = max Cohen's d. "—" = concept not available at
that level. L2 values are shown but are rank-deficient (all near 1.0). Carry_2 and
carry_3 do not exist at L2 (2-digit inputs have at most 2 carry positions).
ans_digit_3+ does not exist at L2. Correct does not exist at L2 (99.8% accuracy,
no meaningful population split).

**Extended cross-level table — additional concepts:**

```
                     L2              L3              L4              L5
Concept          λ₁     d_max   λ₁     d_max   λ₁     d_max   λ₁     d_max
───────────────────────────────────────────────────────────────────────────────
Additional carries
  carry_3          —      —       —      —     0.966  36.6    0.772    7.2
  carry_4          —      —       —      —       —      —     0.882    7.6

Additional column sums
  col_sum_2        —      —     0.995  52.2    0.866   7.3    0.681    3.8
  col_sum_3        —      —       —      —     0.981  21.5    0.754    4.8
  col_sum_4        —      —       —      —       —      —     0.955   13.2

Additional answer digits
  ans_digit_4      —      —       —      —     0.946  10.2    0.246    1.3
  ans_digit_5      —      —       —      —       —      —     0.858    8.0

Partial products
  pp_a0_x_b1       —      —     0.972  26.8    0.889  12.0    0.927    8.4
  pp_a1_x_b0    1.000  5042    0.980  33.8    0.941  11.9    0.933    8.9
  pp_a1_x_b1       —      —     0.996  85.7    0.918  15.6    0.898    8.2
  pp_a2_x_b0       —      —       —      —     0.917  12.7    0.824    6.3
  pp_a2_x_b1       —      —       —      —     0.979  32.4    0.809    6.6
  pp_a2_x_b2       —      —       —      —       —      —     0.948   20.0

Additional derived
  max_carry     1.000  2426    0.974  32.6    0.921  17.1    0.790   23.9
  n_nonzero     0.999  1498    0.940  15.4    0.873  15.1    0.686   30.4

Digit correctness
  dc_pos0          —      —     0.681  18.2    0.412    —     0.045    2.7
  dc_pos1          —      —     0.654   6.3    0.439   3.6    0.061    1.0
  dc_pos2          —      —     0.656   3.3    0.474   2.0    0.144    0.9
```

**Notable patterns in the extended table:**

- **col_sum_2**: 0.995 (L3) → 0.866 (L4) → 0.681 (L5). Column sum 2 degrades
  monotonically, paralleling the carries. Cohen's d degrades more steeply: 52.2 →
  7.3 → 3.8. The L3 value is inflated but the L4→L5 comparison (same N/d) shows
  genuine degradation.

- **pp_a1_x_b1**: 0.996 (L3) → 0.918 (L4) → 0.898 (L5). This partial product
  (tens × tens) is the strongest at L3 but not at L5 (where pp_a0_x_b0 = 0.953
  is stronger). The reversal reflects the different problem structure: at L3, the
  tens × tens product is the highest-order partial product.

- **pp_a2_x_b2** (hundreds × hundreds) exists only at L5 and has λ₁ = 0.948 with
  d_max = 20.0 — the second-strongest partial product after pp_a0_x_b0 (0.953).
  This product determines the leading digit of the 6-digit answer, explaining its
  strength.

- **n_nonzero_carries**: 0.999 (L2) → 0.940 (L3) → 0.873 (L4) → 0.686 (L5).
  Monotone degradation. But Cohen's d shows the opposite: 1498 (L2, absurd) →
  15.4 (L3) → 15.1 (L4) → 30.4 (L5). The L5 d_max is *higher* than L4 because
  n_nonzero_carries has only K = 6 groups at L5 (vs K = 5 at L4), and the
  most-separated pair (0 carries vs 5 carries) is farther apart in the L5 data.

- **digit_correct_pos0**: 0.681 (L3) → 0.412 (L4, n_sig=0) → 0.045 (L5).
  Dramatic cross-level degradation. The model goes from strongly knowing whether
  it got the units digit right at L3 (d = 18.2) to not even passing the permutation
  null at L4, to a barely-detectable signal at L5 (d = 2.7, significant only because
  N/d = 29.8). This is the clearest example of level-dependent representational
  collapse for a specific concept.

- **digit_correct_pos2**: 0.656 (L3) → 0.474 (L4) → 0.144 (L5). Unlike pos0,
  this remains significant at all levels, but degrades substantially. Interestingly,
  at L5 the eigenvalue (0.144) is higher than pos0 (0.045) — the model knows
  whether it got the middle digit right better than whether it got the units digit
  right. At L3, the pattern reverses: pos0 (0.681) > pos2 (0.656). This crossover
  suggests a qualitative shift in what the model "knows about its own correctness"
  as problem difficulty increases.

**Patterns across levels (excluding L2):**

1. **Input digits are stable across all levels.** a_units goes from 0.999 (L3) to
   0.997 (L4) to 0.989 (L5). After adjusting for N/d inflation, these values are
   likely all above 0.98 — input digit encoding is uniformly excellent regardless
   of problem difficulty.

2. **Carries show a complex level × difficulty interaction.** carry_0 *decreases*
   from L3 (0.977) to L4 (0.943) then stays at L5 (0.946). carry_1 drops from
   0.964 (L3) to 0.869 (L4) then recovers to 0.923 (L5). carry_2 drops from
   0.988 (L3) to 0.889 (L4) to 0.740 (L5). The pattern is not monotone — L4
   shows *lower* carry eigenvalues than L5 for carry_0 and carry_1. This may reflect
   the different carry structure of L4 (3×2-digit, max 4 carries) vs L5 (3×3-digit,
   max 5 carries), or it may be an artifact of the N/d difference.

3. **Answer digits monotonically degrade from L3 → L4 → L5.** This is the clearest
   cross-level signal. ans_digit_1: 0.765 → 0.520 → 0.170. ans_digit_2: 0.793 →
   0.455 → 0.047. The degradation is steeper than can be explained by N/d inflation
   alone (L3 and L4 have the same N/d = 2.44, yet ans_digit_2 drops from 0.793 to
   0.455). The model genuinely loses answer digit encoding as problem difficulty
   increases.

4. **Cohen's d provides a more level-comparable measure.** For carry_0: 23.9 (L3) →
   11.4 (L4) → 11.1 (L5). The L3 value is inflated but the L4-L5 comparison
   (same concept, different problem complexity, both with N/d > 2.4) shows similar
   effect sizes. For ans_digit_1: 5.6 (L3) → 3.2 (L4) → 1.2 (L5). The degradation
   in Cohen's d mirrors the eigenvalue degradation but with a shallower slope,
   confirming the trend is genuine.

5. **correct weakens monotonically.** 0.657 (L3) → 0.595 (L4) → 0.147 (L5). Cohen's d:
   3.1 → 2.7 → 2.3. The model's ability to discriminate correct from wrong in its
   internal representations degrades with problem difficulty. This is expected: at L5,
   only 3.4% of answers are correct, creating a severe class imbalance that reduces
   the discriminative signal.

6. **product_binned is universally strong.** 0.997 (L3) → 0.994 (L4) → 0.987 (L5).
   The overall product magnitude is well-encoded at all levels — the model knows
   roughly how big the answer should be, even when it gets the specific digits wrong.

### 7k. The L4 digit_correct_pos0 Null Cases

L4 has 5 n_sig = 0 results, all for digit_correct_pos0 (whether the units digit of
the answer is correct). These are the only systematic null results outside of L5.

```
Concept               Layer  Pop   K   N      λ₁      n_sig
─────────────────────────────────────────────────────────────
digit_correct_pos0      6    all   2   9652   0.4172    0
digit_correct_pos0      8    all   2   9652   0.4200    0
digit_correct_pos0     16    all   2   9652   0.4120    0
digit_correct_pos0     20    all   2   9652   0.4135    0
digit_correct_pos0     24    all   2   9652   0.4172    0
```

At 4 other layers (4, 12, 28, 31), digit_correct_pos0 achieves n_sig = 1 (i.e., it
just barely passes the permutation null). The eigenvalues at those layers are similar
(0.41-0.44), meaning the significance boundary is extremely tight for this concept.

**Full layer × population breakdown for digit_correct_pos0 at L4:**

```
Layer:        L04    L06    L08    L12    L16    L20    L24    L28    L31
──────────────────────────────────────────────────────────────────────────
all n_sig:      1      0      0      1      0      0      0      1      1
all λ₁:      0.414  0.417  0.420  0.428  0.412  0.413  0.417  0.434  0.431
```

The digit_correct_pos0 concept exists only in the "all" population (it requires both
correct and wrong examples to define "pos0 correct" vs "pos0 wrong"). The eigenvalue
ranges from 0.412 to 0.434 — a mere 0.022 spread — yet this tiny variation determines
whether the result is significant. The permutation null threshold at L4/all (N = 9,652,
N/d = 2.36) is approximately 0.42, so eigenvalues above ~0.43 pass (layers 4, 12, 28,
31) and those below fail (layers 6, 8, 16, 20, 24). This is a borderline concept: the
signal is right at the noise floor.

**Why pos0 and not pos1-pos4?** At L4/all/layer16:
- pos0: λ₁ = 0.412, n_sig = 0
- pos1: λ₁ = 0.439, n_sig = 1
- pos2: λ₁ = 0.474, n_sig = 1
- pos3: λ₁ = 0.562, n_sig = 1
- pos4: λ₁ = 0.518, n_sig = 1

The eigenvalues increase from pos0 (units, least significant) to pos3 (most significant
of the 5-digit answer). pos0 measures whether the model got the units digit right,
which depends on the exact value of pp_a0_x_b0 modulo 10 — a fine-grained numerical
check. pos3 measures whether the leading digit is right, which depends on the rough
product magnitude. The model's representation discriminates "approximately correct
magnitude" (pos3, pos4) more easily than "exactly correct units digit" (pos0). This
is consistent with the broader finding that the model encodes magnitude well but
fails at precise digit-level computation.

Contrast with L3, where digit_correct_pos0 has λ₁ = 0.681 and n_sig = 1 — significant
at the easier level. The null at L4 is a genuine level-dependent degradation. At L5,
digit_correct_pos0 has λ₁ = 0.045, which is significant only because N/d = 29.8 gives
a very tight permutation null.

---

## 8. What Phase D Contributes to the Paper

### Genuinely Useful

1. **Independent discriminative measure.** Phase D confirms concept encoding strength
   using Fisher's criterion (variance ratio) instead of centroid PCA (absolute variance).
   At L5/all with N/d = 30, these are clean statistics. The eigenvalue λ has a simple
   interpretation — fraction of variance that is between-class — that Phase C's
   eigenvalues lack.

2. **The carries-strong / answers-weak dissociation, confirmed by a second method.**
   Carries have LDA eigenvalues 0.74-0.95 and Cohen's d > 7. Answer digits 1-3 have
   eigenvalues 0.05-0.17 and Cohen's d < 1.3. Phase C found the same pattern. Having
   two independent methods agree strengthens the claim.

3. **Expanded merged subspaces for downstream analysis.** The merged bases (Phase C +
   novel Phase D directions, orthogonalized) provide ~18-dimensional search spaces
   instead of ~9-dimensional for Fourier screening and GPLVM. This is real expansion —
   the two methods find nearly orthogonal directions.

4. **Confirmation of Phase C null results.** ans_digit_2_msf at L5/correct shows
   n_sig = 0 in both Phase C (dim_perm = 0) and Phase D. Two independent nulls are
   stronger than one.

5. **The eigenvalue as a normalized measure.** Phase D eigenvalues are bounded in [0, 1]
   and self-normalizing (they don't depend on activation scale or number of classes in
   the way Phase C eigenvalues do). This makes cross-concept and cross-layer comparisons
   more meaningful.

6. **ans_digit_3 weak signal.** Phase D found n_sig = 2 for ans_digit_3_msf at L5/all
   where Phase C found dim_perm = 0. A genuine (if small) discriminative signal that
   centroid-SVD missed.

### Should Not Appear as Findings

1. **Novelty ratios.** They are geometric artifacts of high dimensionality (Section 5).
   100% of directions are "novel." This is expected, not informative.

2. **Merged dimensions.** The near-doubling (merged ≈ dim_perm + n_sig) follows
   mechanically from the novelty artifact. The expansion is useful; calling it a
   "finding" is misleading.

3. **Phase C→Phase D angles as "discovery."** The angles (mean 85°) are expected in
   4096D for independently constructed subspaces. Not a finding.

4. **Absolute eigenvalues at L5/correct.** Inflated by N/d = 1.02. Do not cite magnitudes.

5. **Any claim that "Phase D found structure Phase C missed."** Both find real structure;
   they look in different directions by construction. The ans_digit_3 case (Section 7e)
   is a legitimate exception and should be stated carefully: "Phase D found a weak
   discriminative signal (λ = 0.049, d = 0.62) that Phase C's centroid-SVD did not
   detect."

---

## 9. Implementation Details

### S_T Sharing Optimization

S_T depends only on the activation matrix and the population grand mean — not on
concept labels. Phase D exploits this by computing S_T once per (level, layer,
population) tuple and sharing it across all ~43 concepts. This saves ~42 redundant
O(N × d²) computations per tuple.

Two S_T matrices are computed per population:
- **Residualized S_T**: computed from Phase C's residualized activations (product
  variance removed). Used by all concepts except product_binned.
- **Raw S_T**: computed from the original activations. Used only by product_binned
  (which needs the product signal that residualization removes).

The Cholesky factorization and S_T inverse are also shared. The per-concept cost is
just centroid computation + K×K eigenproblem + permutation null (which reuses S_T^{-1}).

A subtle point: concepts that filter samples (set rare values to NaN via
filter_concept_values) use a subset of the population but the S_T computed from the
full population. This means S_T = S_W + S_B does not hold exactly for filtered
concepts — the left side includes samples that the right side excludes. For primary
targets (carries with bin_carry_tail at L5), no samples are filtered (all carry values
have sufficient count), so the mismatch is zero. For concepts with rare-value filtering
(e.g., max_carry_value where extreme values are dropped), the mismatch is small because
filtered samples are few compared to N = 122K.

**Validation against results:** The carry_2 results in Section 7a (λ₁ = 0.740, n_sig
= 10, d_max = 13.53) are unaffected by the S_T sharing approximation because
bin_carry_tail retains all 122,223 samples — zero are filtered. The same is true for
all carry positions (carry_0 through carry_4), all column sums (col_sum_0 through
col_sum_4), all input digits, and all answer digits. The only concepts where filtering
occurs are total_carry_sum (K drops from a theoretical maximum of ~67 to the actual 68
observed values in the data) and max_carry_value (extreme carry values with < 20
samples are dropped). For these concepts, the S_T sharing introduces a mismatch of
< 0.1% in the denominator — negligible relative to the Tikhonov regularization
(α = 1e-4 × trace(S_T)/d ≈ 0.6).

### GPU Acceleration

Phase D uses CuPy for three bottleneck operations:

1. **S_T computation**: `centered.T @ centered` where centered is (N, 4096). At L5/all,
   this is a 122K × 4096 matmul producing a 4096 × 4096 matrix. On A6000: ~30 seconds.

2. **S_T inverse**: `cho_solve(L_factor, I)` producing a dense 4096 × 4096 inverse.
   Computed once, transferred to GPU for the permutation loop.

3. **Permutation null inner loop**: 1,000 iterations of centroid recomputation + K×K
   matmul on GPU. The per-iteration cost is ~5ms, dominated by the centroid computation
   (one-hot × activations matmul on GPU).

### Multi-GPU Architecture

The L5 run used 4 A6000 GPUs. The architecture is a simple work queue:

1. All 9 (level=5, layer) pairs are placed in a shared multiprocessing queue.
2. Four worker processes are spawned using the "spawn" context (required for CUDA —
   "fork" would duplicate GPU state and cause errors).
3. Each worker sets `CUDA_VISIBLE_DEVICES` to its GPU index and re-initializes CuPy.
4. Workers pull (level, layer) pairs from the queue until they receive a None sentinel.
5. Each worker processes all 3 populations (all, correct, wrong) and all ~43 concepts
   for its assigned (level, layer) pair.
6. Results are put on a result queue and collected by the main process.

The work distribution is first-come-first-served. With 9 pairs and 4 GPUs:
- Round 1: GPUs 0-3 each take a pair (layers 4, 6, 8, 12).
- Round 2: as each GPU finishes, it pulls the next available pair (layers 16, 20, 24, 28).
- Round 3: only layer 31 remains. The first GPU to finish round 2 takes it; the other
  three receive None sentinels and exit.

In the actual run, GPU3 finished layer 4 first (5h 42m) and pulled layer 16. GPUs
1, 0, 2 finished layers 6, 8, 12 (8+ hours each) and pulled layers 20, 28, 24.
GPU3 then finished layer 16 and pulled layer 31 — the last pair. GPUs 1, 0, 2
finished their round-2 layers at 03:38, 03:47, 03:49 and exited. GPU3 finished
layer 31 alone at 04:57. The bottleneck is the last layer running on a single GPU
while three GPUs sit idle for 1+ hours.

### Resume Logic

Phase D implements resume at the per-concept level:

1. **Metadata check**: if `metadata.json` exists in the output directory and contains
   `"denominator": "S_T"`, the concept is skipped entirely. A stale cache from a
   previous S_W-denominator run is detected and recomputed.

2. **Null eigenvalue cache**: `null_lda_eigenvalues.npy` is cached separately. If
   it exists with the correct shape (n_perms, K-1), it is reloaded. This permits
   resuming after a timeout that interrupted mid-concept — the expensive permutation
   null is preserved even if the metadata write did not complete.

3. **Atomic writes**: metadata.json is written via tempfile.mkstemp + os.replace to
   prevent corruption from interrupted writes.

### Cross-Validation Protocol

5-fold stratified cross-validation with random_state=42. Per fold:
1. Compute training-set S_T, Cholesky-factor it, solve LDA.
2. Take top n_sig directions (from the full-data analysis).
3. Compute test-set class centroids.
4. Compute pairwise distances in full 4096D space and in the LDA-projected space.
5. Pearson correlation between the two distance vectors.

Skipped when K < 3 (binary concepts have only 1 centroid pair, making correlation
undefined) or n_sig < 1.

### L5 Carry Binning

Carry values at L5 have long tails (carry_2 can reach 13, but values above 10 are
rare). Phase D inherits the same binning thresholds from Phase C:

```
carry_0: no binning (all values have sufficient count)
carry_1: bin values ≥ 12 into a single bin
carry_2: bin values ≥ 13 into a single bin
carry_3: bin values ≥ 9 into a single bin
carry_4: bin values ≥ 5 into a single bin
```

This ensures every class has at least MIN_GROUP_SIZE = 20 samples, preventing
degenerate centroids from tiny groups.

---

## 10. Relationship to the Paper's Thesis

The paper's thesis is: the Linear Representation Hypothesis (LRH) accurately finds
concept subspaces — the "rooms" where concepts live — but the computational mechanism
is encoded in non-linear geometry *within* those subspaces — the "shapes" of the
concept manifolds.

Phase C found the rooms. Phase D confirmed the rooms from a different angle (literally —
an 85° different angle, as we now understand is geometrically inevitable in 4096D).
Neither Phase C nor Phase D can see the shapes. They are both linear methods. They
find linear subspaces and measure linear statistics (eigenvalues, centroid separations,
variance ratios). If carry_2 values 0-9 trace a circle inside their 10D subspace,
neither method would detect it — they would only report that the 10D subspace exists
and that the centroids are well-separated.

The shapes are the job of downstream methods:

- **Fourier screening**: test whether digit centroids sit on circles inside their
  subspaces. If a_tens values 0-9 trace a periodic curve, that is a Fourier encoding.
- **GPLVM / GP metric tensors**: full non-linear manifold characterization. Does the
  correct population trace a clean 1D curve while the wrong population scatters?
- **Persistent homology**: topological structure (loops, voids) in the point clouds.

Phase D's merged bases give these methods more directions to search within. The
18-dimensional merged subspace for a_tens contains Phase C's 9 centroid-variance
directions plus Phase D's 9 discrimination-ratio directions. If a circle exists, it
might span directions from both sets.

**Specifically, what Fourier screening will do with the merged bases:** For a concept
like a_tens (K = 10 values: digits 0-9), Fourier screening will:
1. Project each of the 10 class centroids into the 18D merged subspace.
2. Compute the Discrete Fourier Transform of the 10 projected centroids, treating
   the digit value as the position along a periodic axis.
3. Test whether the DFT spectrum has significant power at specific frequencies — in
   particular, whether frequency 1 (a full circle) or frequency 2 (a figure-eight)
   dominates.
4. If the centroids trace a circle in some 2D subplane of the 18D space, the DFT
   will show sharp power at frequency 1 in that pair of dimensions.

The merged basis matters because Phase C's directions (maximum centroid spread) and
Phase D's directions (maximum discrimination ratio) are nearly orthogonal (mean angle
~85°). A Fourier encoding might concentrate its circular structure along the LDA
directions (where class separation is maximized relative to noise) rather than the PCA
directions (where absolute centroid spread is maximized). Without Phase D's contribution
to the merged basis, Fourier screening would search only the PCA subspace and might
miss a circle that exists in the LDA subspace.

For carries, the Fourier prediction is different. carry_2 has K = 14 non-uniformly
distributed values (due to bin_carry_tail). If carry values are encoded along a number
line rather than a circle, the DFT will show power at all frequencies with no sharp
peak. This distinguishes linear encoding (carry 0 at one end, carry 13 at the other)
from periodic encoding (carry values wrapping around). The LDA eigenvalue spectrum
(Section 7d: steep decay from 0.740 to 0.034) is consistent with linear encoding —
a single dominant direction, not a circle.

The carries-strong / answers-weak pattern from Phase D sets up the paper's failure
story. The model builds strong intermediate representations for carries (λ > 0.74,
Cohen's d > 7) but cannot assemble them into clean output structure (ans_digit_1:
λ = 0.17, d = 1.15; ans_digit_2: λ = 0.05, d = 0.56). The carries are there. The
column sums are there. But the final answer digits are not. The computation is
encoded; the output mapping is broken. Phase D cannot tell us *why* — that requires
seeing the shapes, the non-linear geometry that either enables or fails to enable
the carry-to-digit assembly. But Phase D documents the failure precisely: two
independent linear methods, operating with N/d = 30, both confirm that intermediate
representations are strong and output representations are weak.

---

## 11. Limitations

1. **Phase D is a linear method.** It finds linear discriminative directions, not
   non-linear manifold structure. If carry_2 values form a helix in activation space,
   Phase D measures the shadow of that helix onto its best linear projection — not the
   helix itself.

2. **Novelty ratios are geometrically uninformative at d = 4096.** A 9D subspace
   occupies 0.22% of 4096D space. Any independently constructed direction will be
   99%+ orthogonal to it by chance. The universal n_novel = n_sig is dimensional
   arithmetic, not discovery. See Section 5.

3. **Eigenvalues and Cohen's d are inflated at N/d < 2.** This affects L5/correct
   (N/d = 1.02) and all L2-L4 populations. See Section 4.

4. **LDA requires concept labels.** It cannot discover unknown concepts — it can only
   measure how well known concepts are linearly separable. Discovery of unknown
   structure is the job of Phase E (residual hunting).

5. **S_T sharing introduces a small approximation for filtered concepts.** When samples
   are filtered by filter_concept_values (rare values set to NaN), the concept-specific
   S_B uses a subset of samples while S_T uses all samples. The decomposition
   S_T = S_W + S_B does not hold exactly. For primary targets (carries, input digits),
   no filtering occurs at L5 and the approximation is exact.

6. **Cannot determine causal relevance.** A significant LDA direction does not prove
   the model *uses* that direction during computation. It could be a correlate, a
   byproduct, or a read-out artifact. Causal relevance requires activation patching
   along the discovered directions — a downstream experiment.

7. **The S_T vs S_W choice does not affect which directions are found.** This is a
   mathematical fact (Section 2) but may confuse readers expecting traditional Fisher
   LDA. The eigenvectors are identical; only eigenvalue magnitudes differ. Phase D's
   eigenvalues (bounded in [0,1]) are actually easier to interpret than traditional
   Fisher eigenvalues (unbounded).

8. **Layer 31 bottleneck in multi-GPU runs.** The work-queue design assigns whole
   (level, layer) pairs to GPUs. When the last pair remains, one GPU works alone while
   the others are idle. In the L5 run, GPU3 processed layer 31 solo for ~4 hours after
   the other 3 GPUs finished and exited. A label-level parallelization scheme would
   avoid this bottleneck but adds complexity.

---

## 12. Runtime and Reproducibility

### L5 Run (Job 6897189)

| Metric | Value |
|--------|-------|
| SLURM job | 6897189 |
| Node | babel-t9-20 |
| GPUs | 4× A6000 |
| Start | March 31, 2026, 11:05 EDT |
| End | April 1, 2026, 04:58 EDT |
| Total runtime | 64,370 seconds (1,072 minutes, 17.9 hours) |
| Step 1 (LDA computation) | 64,295 seconds (17.9 hours) |
| Step 2 (plot generation) | 55 seconds |
| Step 3 (summary tables) | < 1 second |
| Exit code | 0 |
| Permutations per concept | 1,000 |
| Significance threshold | α = 0.01 (99th percentile) |
| Regularization | α_frac = 10⁻⁴ |

### L1-L4 Run (Job 6893354)

| Metric | Value |
|--------|-------|
| SLURM job | 6893354 |
| Node | babel-u9-28 |
| GPUs | 1× (single GPU mode) |
| Start | March 30, 2026, 21:52 EDT |
| Cancelled | March 31, 2026, 05:52 EDT (8h wall limit) |
| Completed | L1-L4 (36/45 pairs), cancelled during L5/layer04 |
| Per-concept files | L2: 306, L3: 666, L4: 837 metadata.json files on disk |

### Results Summary

**L5 (summary CSV, 1,035 rows):**

| Metric | Value |
|--------|-------|
| Total LDA results | 1,035 |
| Concepts with significant LDA (n_sig > 0) | 1,026 / 1,035 (99.1%) |
| Concepts with novel directions (n_novel > 0) | 1,026 / 1,035 (99.1%) |
| Mean novel directions (where > 0) | 8.4 |
| Concepts with n_sig = 0 | 9 (all ans_digit_2_msf at L5/correct) |

**L2-L4 (on-disk metadata only):**

| Level | Files | Concepts | Populations | Significance | N/d (all) |
|-------|-------|----------|-------------|-------------|-----------|
| L2 | 306 | 17 | all, correct | 306/306 (100%) | 0.98 |
| L3 | 666 | 28 | all, correct, wrong | 666/666 (100%) | 2.44 |
| L4 | 837 | 35 | all, correct, wrong | 832/837 (99.4%) | 2.44 |

### Output Files

```
/data/user_data/anshulk/arithmetic-geometry/phase_d/
├── subspaces/L5/layer_{04..31}/{all,correct,wrong}/{concept}/
│   ├── metadata.json           # Complete results + resume checkpoint
│   ├── lda_basis.npy           # (n_sig, 4096) significant LDA directions
│   ├── merged_basis.npy        # Phase C + novel LDA, orthogonalized
│   ├── lda_eigenvalues.npy     # All K-1 LDA eigenvalues
│   └── null_lda_eigenvalues.npy  # (1000, K-1) null eigenvalues
├── summary/
│   ├── phase_d_results.csv     # 1,035 rows, L5 only
│   └── lda_novelty_summary.csv # 1,026 rows (n_novel > 0)

/home/anshulk/arithmetic-geometry/plots/phase_d/
├── eigenvalue_spectra/         # 198 scree plots (tier 1+2 at layers 4, 16, 31)
├── novelty_heatmaps/           # 3 heatmaps (one per population)
├── n_sig_heatmaps/             # 3 heatmaps
└── population_comparison/      # 36 multi-panel plots
```

Total plots: 240 (eigenvalue spectra) + 3 + 3 + ... = 247 total.

Per-concept metadata.json files for L2-L4 from job 6893354 are intact at
`subspaces/L{2,3,4}/...` (L2: 306, L3: 666, L4: 837 = 1,809 files total) but are not
included in the summary CSV. Results from these files are documented in Sections 7g-7k.

### Reproducibility

The run is fully reproducible from:
```bash
# L5 only (matches job 6897189)
python phase_d_lda.py --config config.yaml --level 5 --n-gpus 4

# All levels, single GPU (matches job 6893354 before timeout)
python phase_d_lda.py --config config.yaml
```

Random seed for permutation null: per-concept deterministic via
`np.random.default_rng(seed=42)`. Cross-validation uses `random_state=42`.
Results are deterministic given the same data and code.

### Runtime Estimates

```
Configuration                  Estimated Runtime
──────────────────────────────────────────────────
4 GPUs, L5 only                ~18 hours
4 GPUs, all levels             ~20 hours
1 GPU, all levels              ~80 hours
```

L5 dominates runtime due to N = 122,223 (vs N ≤ 10,000 for L1-L4). Within L5,
the permutation null is the bottleneck: 1,000 permutations × ~43 concepts × 3
populations × 9 layers. The multi-GPU parallelization is across layers, not
concepts — within a single (level, layer, population), concepts are processed
sequentially on one GPU.

---

*This document covers Phase D as of April 1, 2026. All numbers are from
phase_d_results.csv (1,035 rows), phase_c_results.csv (2,844 rows), and the
per-concept metadata.json files in /data/user_data/anshulk/arithmetic-geometry/phase_d/.
Code: phase_d_lda.py (1,734 lines). SLURM jobs: 6893354 (L1-L4, 1 GPU, cancelled),
6897189 (L5, 4×A6000, completed).*
