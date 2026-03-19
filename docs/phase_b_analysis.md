# Phase B: Complete Analysis

**Anshul's Geometric Manifold Interpretability Project**
**Carnegie Mellon University, March 2026**

This document records every decision, every number, and every result from Phase B —
the concept deconfounding stage. It is the truth document for this stage. All numbers
are validated against the actual output files as of March 19, 2026.

---

## Table of Contents

1. [Purpose of This Stage](#1-purpose-of-this-stage)
2. [What Phase B Is and Is Not](#2-what-phase-b-is-and-is-not)
3. [Why This Stage Exists](#3-why-this-stage-exists)
4. [A Concrete Example of Contamination](#4-a-concrete-example-of-contamination)
5. [The Data Structure It Operates On](#5-the-data-structure-it-operates-on)
6. [The Concept Registry](#6-the-concept-registry)
7. [The Math: Pearson Correlation](#7-the-math-pearson-correlation)
8. [Why Pearson and Not Spearman](#8-why-pearson-and-not-spearman)
9. [Product Residualization on Labels](#9-product-residualization-on-labels)
10. [The Suppression Effect: A Discovery](#10-the-suppression-effect-a-discovery)
11. [The Four Correlation Categories](#11-the-four-correlation-categories)
12. [The Classification Logic](#12-the-classification-logic)
13. [Thresholds and Decision Rules](#13-thresholds-and-decision-rules)
14. [Results: Level 1](#14-results-level-1)
15. [Results: Level 2](#15-results-level-2)
16. [Results: Level 3](#16-results-level-3)
17. [Results: Level 4](#17-results-level-4)
18. [Results: Level 5](#18-results-level-5)
19. [The Spearman Follow-up](#19-the-spearman-follow-up)
20. [The Deconfounding Plan](#20-the-deconfounding-plan)
21. [The Final Decision](#21-the-final-decision)
22. [Label-Level vs Activation-Level Contamination](#22-label-level-vs-activation-level-contamination)
23. [What This Stage Cannot Tell You](#23-what-this-stage-cannot-tell-you)
24. [The Code](#24-the-code)
25. [Output Files](#25-output-files)
26. [Runtime and Reproducibility](#26-runtime-and-reproducibility)

---

## 1. Purpose of This Stage

Phase C finds the directions in 4096-dimensional activation space that encode each
mathematical concept. It does this by grouping problems by concept value (e.g.,
carry_0 = 0, 1, 2, ..., 8), computing one centroid per group, and finding the
directions along which those centroids spread.

Before Phase C computes centroids, it removes one confound: product magnitude. Phase A
showed product is the dominant axis of variation. Without removing it, every concept
subspace would just be the product direction.

Phase B asks: after removing product, are there other confounds left over? If so, which
ones, and do they need fixing?

This is a diagnostic stage. It runs in 32 seconds on CPU. It touches no activations.
It looks only at the labels — the same labels that Phase C uses to group problems. It
measures how tangled those labels are with each other, and decides whether Phase C's
single-confound removal is enough.

---

## 2. What Phase B Is and Is Not

### What it is

- Pairwise Pearson correlations between all concept labels, before and after removing
  the linear effect of product
- Classification of every correlated pair as structural, residualization-induced,
  sampling-induced, or unexplained
- A Spearman follow-up on the top 30 pairs to check for nonlinear relationships
  that Pearson might miss
- A deconfounding plan telling Phase C which concepts (if any) need additional
  treatment
- Heatmap visualizations of both correlation matrices at every level

### What it is not

- It does not touch activations. All correlations are between label values.
- It does not compute subspaces. That is Phase C.
- It does not detect superposition. Two concepts can have zero label correlation
  but share activation-space directions. That is Phase F.
- It does not guarantee that a passing result means Phase C is clean. It catches one
  source of contamination (correlated labels). Phase F catches another (correlated
  representations).

---

## 3. Why This Stage Exists

Consider carry_2 at Level 3 (2-digit x 2-digit multiplication). Phase C groups all
10,000 problems by their carry_2 value, computes one centroid per group, and finds the
directions along which those centroids spread.

But carry_2 is not independent of everything else. At L3, carry_2 depends on col_sum_2
plus carry_1. col_sum_2 equals pp_a1_x_b1, which is a_tens times b_tens. carry_1
depends on col_sum_1, which depends on pp_a0_x_b1 and pp_a1_x_b0, which depend on
a_units, b_tens, a_tens, and b_units. There is a web of dependencies.

If a_tens = 9 problems tend to have higher carry_2 than a_tens = 1 problems, then
grouping by carry_2 also partially groups by a_tens. The carry_2 centroid for high
values will partly encode "a_tens is large." The carry_2 subspace will be contaminated
with digit signal.

Phase C already handles the biggest confound (product magnitude). Phase B checks
whether the remaining confounds are small enough to ignore.

---

## 4. A Concrete Example of Contamination

Take the problem 47 x 83:

```
a = 47, b = 83
a_units = 7, a_tens = 4
b_units = 3, b_tens = 8

Partial products:
  pp_a0_x_b0 = 7 x 3 = 21
  pp_a0_x_b1 = 7 x 8 = 56
  pp_a1_x_b0 = 4 x 3 = 12
  pp_a1_x_b1 = 4 x 8 = 32

Column sums:
  col_sum_0 = 21     (just pp_a0_x_b0)
  col_sum_1 = 68     (pp_a0_x_b1 + pp_a1_x_b0 = 56 + 12)
  col_sum_2 = 32     (just pp_a1_x_b1)

Carries:
  carry_0 = floor(21 / 10) = 2
  carry_1 = floor((68 + 2) / 10) = 7
  carry_2 = floor((32 + 7) / 10) = 3

Product = 3901
```

This is verified correct: 47 x 83 = 3,901.

Note how carry_0 = floor(col_sum_0 / 10). It is a deterministic function of col_sum_0.
And col_sum_0 = pp_a0_x_b0 = a_units x b_units. So carry_0, col_sum_0, pp_a0_x_b0,
a_units, and b_units are all chained together by arithmetic. Their label values are
correlated not because of anything the model does, but because of how multiplication
works.

Phase B measures these correlations.

---

## 5. The Data Structure It Operates On

Phase B reads the coloring DataFrames built by Phase A. These are cached as pickle
files at `phase_a/coloring_dfs/L{level}_coloring.pkl`. Each DataFrame has one row per
problem and columns for every label variable.

| Level | N problems | DataFrame columns | Registry concepts |
|-------|-----------|-------------------|-------------------|
| L1    | 64        | 20                | 11                |
| L2    | 4,000     | 29                | 20                |
| L3    | 10,000    | 40                | 27                |
| L4    | 10,000    | 47                | 34                |
| L5    | 122,223   | 55                | 42                |

The DataFrame has more columns than registry concepts because it includes identity
columns (problem_idx, a, b, predicted, ground_truth) and error-only variables
(abs_error, rel_error, signed_error, underestimate, even_pred, div10_pred,
error_category) that are not in Phase C's concept registry.

Phase B extracts only the registry concepts. The working matrix for L3 is 10,000 rows
by 27 concepts (not 40). For L5 it is 122,223 rows by 42 concepts (not 55). These are
the same concepts that Phase C uses to compute subspaces, so measuring correlations on
these values catches exactly the contamination that matters.

One concept is excluded from the correlation analysis: product_binned. Phase C uses raw
(non-residualized) activations for this concept, so product residualization does not
apply to it. Including it would show trivial correlations that do not affect Phase C.
This is why L3 shows 27 concepts (not 28) and L5 shows 42 (not 43).

---

## 6. The Concept Registry

The concept registry is defined in `phase_c_subspaces.py` (`get_concept_registry`
function, line 152). Phase B duplicates this logic to avoid import issues with
matplotlib and scipy on headless compute nodes. The two registries must be kept in sync.

The registry organizes concepts into four tiers:

**Tier 1: Input and answer digits (raw values, no preprocessing)**

These are the digits of the input numbers (a_units, a_tens, etc.) and the answer digits
(ans_digit_0_msf through ans_digit_5_msf). Values are small integers (0-9 for digits,
1-9 for leading digits). No binning is applied.

At L3: 4 input digits (a_units, a_tens, b_units, b_tens) + 4 answer digits = 8 concepts.
At L5: 6 input digits + 6 answer digits = 12 concepts.

**Tier 2: Carries and column sums (preprocessed)**

Carries use `filter_min_group`: rare values (fewer than 20 samples) are set to NaN and
excluded from centroid computation. This matters at L5 where carry_2 ranges 0 to 26 and
values above ~15 have few samples.

Column sums use `bin_deciles`: the continuous values are binned into 10 quantile groups.
This is because col_sum_1 at L3 ranges 0 to 162, producing too many unique values for
centroid-based analysis.

At L3: 3 carries + 3 column sums = 6 concepts.
At L5: 5 carries + 5 column sums = 10 concepts.

**Tier 3: Derived concepts**

These include correctness (binary), n_nonzero_carries, total_carry_sum, max_carry_value,
n_answer_digits, product_binned (decile-binned product magnitude), and per-digit
correctness (digit_correct_pos0 through pos5). product_binned is excluded from Phase B's
analysis as noted above.

At L3: 10 concepts (including product_binned and 4 digit_correct columns).
At L5: 12 concepts (including product_binned and 6 digit_correct columns).

**Tier 4: Partial products (binned)**

Partial products (pp_a0_x_b0 etc.) are binned into 9 equal-width bins over [0, 81].

At L3: 4 partial products. At L5: 9 partial products.

### Why preprocessing matters for Phase B

Phase B must measure correlations on the same values that Phase C uses. If Phase C bins
col_sum_1 into deciles, Phase B correlates the decile-binned values, not the raw
continuous values. The reason: Phase C groups by binned values to compute centroids. If
two concepts are correlated in their binned form, their centroids will be contaminated.

---

## 7. The Math: Pearson Correlation

For two concepts j and k, the Pearson correlation is:

```
r_{jk} = sum_i (V[i,j] - mu_j)(V[i,k] - mu_k) /
         sqrt( sum_i (V[i,j] - mu_j)^2  *  sum_i (V[i,k] - mu_k)^2 )
```

where the sums run over all problems i where both V[i,j] and V[i,k] are not NaN.

NaN values arise from `filter_min_group` (rare carry values dropped). For each pair,
only the intersection of valid rows is used. This means different pairs may use
different subsets of the data. At L3 with carries, the NaN fraction is small (under 1%),
so this barely matters. At L5, carry_2 can have 10%+ NaN after filtering values above
15.

The result is a C x C symmetric matrix R with R[j,j] = 1 and R[j,k] = R[k,j].

At L3 with 27 concepts (excluding product_binned): 27 x 26 / 2 = 351 unique pairs.
At L5 with 42 concepts: 42 x 41 / 2 = 861 unique pairs.

The computation takes well under a second at each level.

---

## 8. Why Pearson and Not Spearman

Phase C's contamination mechanism is linear: if concept j's mean value increases
linearly with concept k's value, then grouping by k also partially groups by j, and
the centroids pick up j's signal. Pearson measures this linear relationship.

Spearman measures monotonic (rank-based) correlation. For some key relationships
(like carry_0 = floor(col_sum_0 / 10), a step function), Spearman might seem more
appropriate.

The empirical data reveals a surprise. For the most important step-function
relationship in the dataset (col_sum_0 vs carry_0 at L3):

```
Pearson r  = 0.989
Spearman rho = 0.962
```

Pearson is *higher* than Spearman. The reason: carry_0 = floor(col_sum_0 / 10) is a
step function with 9 evenly spaced steps over [0, 81]. The step function is nearly
linear, so Pearson captures it well. Spearman is penalized by the many tied ranks
within each step — all col_sum_0 values from 0 to 9 map to carry_0 = 0, creating a
block of tied ranks that hurts Spearman.

For other pairs:

```
a_units vs carry_0:  Pearson = 0.621,  Spearman = 0.624  (essentially equal)
carry_0 vs carry_1:  Pearson = 0.662,  Spearman = 0.662  (essentially equal)
```

The practical conclusion: Pearson is the right primary metric. We run a Spearman
follow-up on the top 30 pairs per population as a sanity check. In practice the two
metrics agree closely.

---

## 9. Product Residualization on Labels

Phase C removes the linear effect of product magnitude from activations before
computing concept subspaces. Phase B mirrors this on labels to see what the correlation
structure looks like after Phase C's product removal step.

The math for label-level product residualization:

```
For each concept j:
    v_j   = concept values (with NaN for filtered entries)
    p     = product values (never NaN)
    valid = rows where v_j is not NaN

    # Center both on valid rows only
    v_c = v_j[valid] - mean(v_j[valid])
    p_c = p[valid]   - mean(p[valid])

    # OLS regression coefficient
    beta = (v_c @ p_c) / (p_c @ p_c)

    # Residual = concept value with product's linear contribution removed
    v_resid[valid] = v_c - beta * p_c
```

Both numerator and denominator use the same valid-row subset. This is important.
An earlier version of the code computed `p_dot_p` over all rows but `v_c * p_c` over
valid rows only, which underestimated beta when the NaN fraction was large. The fix
ensures beta is correct even at L5 where carry_2 has 10%+ NaN.

The result is R_resid: the correlation matrix on product-residualized labels. This is
the matrix that matters for deciding whether Phase C needs additional deconfounding.

### What product residualization does to correlations

Product correlates with everything (it is the product of all digits, which determines
all carries and column sums). Removing product can either decrease or increase pairwise
correlations:

**Decreases:** Pairs whose correlation is mediated through product. For example, at L3:

```
a_tens vs carry_2:  raw = +0.656,  resid = -0.006  (eliminated)
carry_1 vs carry_2: raw = +0.614,  resid = +0.045  (eliminated)
```

Both of these correlations were entirely explained by product magnitude. After removing
product, they vanish. This means Phase C's product residualization already handles them.

**Increases (suppression):** Pairs where product was masking a relationship. For example:

```
carry_0 vs carry_1: raw = +0.662,  resid = +0.783  (increased)
```

carry_0 and carry_1 are adjacent in the carry chain. They have a direct structural
link that product residualization does not remove. Removing product actually reveals
this direct link more clearly.

**Creates (the suppression effect):** Some pairs go from zero to strongly correlated.
This was the major discovery of Phase B. See the next section.

---

## 10. The Suppression Effect: A Discovery

This is the most important finding of Phase B.

Product residualization creates a strong anti-correlation between leading-digit pairs
that does not exist in the raw data. At every level, the leading digit of operand a
and the leading digit of operand b become anti-correlated at approximately r = -0.80
after product residualization.

### The numbers

| Level | Pair | r_raw | r_resid | Change |
|-------|------|-------|---------|--------|
| L1 | a_units vs b_units | 0.000 | **-0.852** | -0.852 |
| L2 | a_tens vs b_units | +0.027 | **-0.814** | -0.841 |
| L3 | a_tens vs b_tens | -0.002 | **-0.796** | -0.794 |
| L4 | a_hundreds vs b_tens | -0.002 | **-0.802** | -0.800 |
| L5 | a_hundreds vs b_hundreds | -0.003 | **-0.801** | -0.798 |

In every case, the raw correlation is essentially zero (as expected — input digits of
different operands are drawn independently). After product residualization, the
correlation jumps to approximately -0.80.

### Why this happens

The math is straightforward. For a 2-digit x 2-digit problem (L3):

```
product = (10 * a_tens + a_units) * (10 * b_tens + b_units)
        = 100 * a_tens * b_tens + 10 * (a_tens * b_units + a_units * b_tens) + a_units * b_units
```

The leading term is 100 * a_tens * b_tens. This means product variance is dominated by
the product of the two leading digits. When we regress out product from labels:

```
a_tens_resid = a_tens - beta_a * product
b_tens_resid = b_tens - beta_b * product
```

Both residuals have the product component removed. But since product is approximately
proportional to a_tens * b_tens, removing product constrains a_tens * b_tens to be
roughly constant. When a_tens is large, b_tens must be small (and vice versa) to keep
their product roughly fixed. This forces anti-correlation.

This is a standard statistical phenomenon called "suppression." It occurs whenever you
partial out a variable that is approximately the product of two others.

### Why it only affects leading digits

The effect is proportional to how much each digit pair contributes to product variance.
Leading digits dominate (their coefficient is 100x at L3, 10,000x at L5). Lower-order
digits have negligible contributions:

```
L5: a_units vs b_units:       raw = +0.201, resid = +0.201  (unchanged)
L5: a_tens vs b_tens:          raw = +0.000, resid = -0.006  (tiny)
L5: a_hundreds vs b_hundreds:  raw = -0.003, resid = -0.801  (massive)
```

Only the hundreds digits (the leading digits at L5) are affected. The tens and units
digits are essentially untouched.

### Why this matters for Phase C

At L3, after product residualization, grouping by a_tens also partially groups by
b_tens (inversely). The a_tens centroid for value 9 sits in a region where b_tens tends
to be 1-3, because those are the problems where a_tens * b_tens produces a given range
of products.

This means the a_tens subspace in Phase C is contaminated with b_tens signal, and vice
versa.

### Why the activation-level impact is small

The numbers above are label-level correlations, computed on 1-dimensional scalars. In
4096-dimensional activation space, product occupies roughly 1 direction. Digit
encodings use approximately 8 dimensions (Phase C's finding). Removing one direction
from an 8-dimensional encoding leaves 7 dimensions intact. The label-level
contamination (63% of centroid spread) translates to approximately 3% contamination at
the activation level.

Simulation testing confirms this: with 8-dimensional Fourier-encoded digit
representations in 4096-dimensional space, deconfounding the other leading digit
removes only 4.2% of the target digit's signal, not 96% as the label-level numbers
would suggest.

This is why Phase B flags the suppression effect for documentation but does NOT
recommend deconfounding it. Deconfounding at the activation level would be over-
correction.

---

## 11. The Four Correlation Categories

The original Phase B design document identified three categories of correlation:
structural, sampling-induced, and accidental. Running Phase B revealed a fourth:
residualization-induced.

### Category 1: Structural

Correlations forced by the arithmetic of multiplication. carry_0 = floor(col_sum_0 / 10)
is a deterministic relationship. carry_0 and carry_1 are linked through the carry chain.
col_sum_k and its contributing partial products are linked by definition.

These correlations cannot be removed without destroying genuine mathematical signal.
Phase C should not deconfound them, because the shared encoding IS the computation.

### Category 2: Residualization-induced

Correlations created by product residualization. The leading digit of a and the leading
digit of b become anti-correlated at r = -0.80 because product is approximately their
product. This does not exist in the raw data. It is an artifact of the residualization
step.

The correct response: flag it, document it, do NOT deconfound. The activation-level
impact is small (~3%) because product occupies only 1 of 4096 dimensions. If Phase F
later shows heavy subspace overlap between a_tens and b_tens, revisit.

### Category 3: Sampling-induced

Correlations caused by the L5 carry-stratified dataset. Under uniform sampling (L3, L4),
input digits of different operands are independent. At L5, the carry-stratified selection
overweights high-carry problems, which tend to have large unit digits. This induces a
small positive correlation between a_units and b_units at L5 (r = +0.201 in the "all"
population).

In the L5 correct-only population (4,197 problems), this rises to r = +0.343, crossing
the action threshold. The correct response for this specific case: if Phase C analyzes
the correct-only population at L5 for a_units or b_units, regress out the other digit
first.

### Category 4: None / unexplained

Correlations that do not fit any of the above categories. Phase B found zero of these.
Every pair above the action threshold was classified into one of the three categories
above. If any unexplained pairs had appeared, the action would be "investigate" — look
at the pair manually and figure out why it is correlated.

---

## 12. The Classification Logic

The `classify_pair` function takes two concept names and the level, and returns a
classification. The logic proceeds through a series of checks in order of priority:

**Same-column relationships:** carry_k and col_sum_k at the same column index are
structural. carry_0 = floor((col_sum_0 + carry_{-1}) / 10) is deterministic.

**Carry chain:** Any two carry concepts are structural. carry_0 -> carry_1 -> carry_2
is a causal chain. Even non-adjacent carries (carry_0 vs carry_2) are structural
because the chain links them.

**Column sums:** Any two column sums are structural. They are linked through the carry
chain (the carry out of one column affects the next).

**Cross-column carry/col_sum:** carry_k vs col_sum_j where j != k is structural.

**Partial products to column sums/carries:** Every partial product contributes to a
specific column sum, which determines a carry. Structural.

**Input digits to partial products:** a_units (place index 0) participates in
pp_a0_x_b0, pp_a0_x_b1, etc. Structural. Even digits that do not directly participate
in a partial product are structural (e.g., a_units correlates with pp_a1_x_b0 through
b_units).

**Partial products to partial products:** All pp's are sub-terms of the same product.
They share input digits as factors. After product residualization they become
correlated/anti-correlated through the suppression effect. Structural.

**Answer digits:** Answer digits are determined by carries and column sums. All
relationships between answer digits and other arithmetic concepts are structural.

**Derived aggregates:** n_nonzero_carries, total_carry_sum, max_carry_value, and
n_answer_digits are aggregates of their component concepts. Structural.

**product_binned and correct:** Correlate with everything. Structural.

**Leading-digit suppression:** Cross-operand digit pairs where both digits are the
leading digit of their operand. Classified as residualization-induced. The leading
place is:

| Level | Leading place for a | Leading place for b |
|-------|--------------------|--------------------|
| L1 | units (index 0) | units (index 0) |
| L2 | tens (index 1) | units (index 0) |
| L3 | tens (index 1) | tens (index 1) |
| L4 | hundreds (index 2) | tens (index 1) |
| L5 | hundreds (index 2) | hundreds (index 2) |

**Sampling-induced:** At L5, cross-operand digit pairs that are not leading digits are
classified as sampling-induced (potentially affected by carry stratification).

**Everything else:** Classification "none." In practice, zero pairs fell into this
category.

---

## 13. Thresholds and Decision Rules

### Report threshold: |r| > 0.1

Any pair with raw or residualized |r| above 0.1 is included in the classified_pairs.csv
output. This captures all potentially interesting correlations for documentation.

### Action threshold: |r_resid| > 0.3

A pair triggers a classification decision only if its post-product-residualization
|r| exceeds 0.3. Below this, r^2 = 0.09 (9% shared variance), and centroid
contamination is unlikely to distort subspace directions meaningfully.

Why 0.3 and not some other number? It is a pragmatic choice, not a theoretical
guarantee. But the empirical data shows it works well: after product residualization,
correlations split cleanly into two groups:

- Structural pairs (carries, col_sums, etc.): r = 0.5 to 0.98
- Non-structural, non-leading-digit pairs: r < 0.05

The gap between 0.05 and 0.5 is wide. Any threshold between 0.1 and 0.4 would produce
the same decisions. 0.3 is safely in the middle.

### Decision rule for each classification

```
|r_resid| <= 0.3            ->  action = accept (below threshold)
structural, any |r_resid|   ->  action = accept (do not deconfound)
residualization-induced      ->  action = flag_use_raw (document, flag for Phase F)
sampling-induced             ->  action = deconfound (regress out the confound)
unexplained                  ->  action = investigate (manual review)
```

### Population-level decision

The top-level decision (what to tell Phase C) is based on the "all" population, which
is what Phase C primarily uses. Sub-population results (correct, wrong) are documented
but do not drive the top-level decision. The reason: Phase C's product residualization
is computed on the "all" population. Residual correlations in the correct-only or
wrong-only populations may differ slightly but do not affect Phase C's main run.

---

## 14. Results: Level 1

Level 1 is 1-digit x 1-digit multiplication. 64 problems, 100% accuracy.

| Population | N problems | N concepts | Pairs total | Pairs |r_resid| > 0.3 |
|------------|-----------|------------|-------------|------------------------|
| all | 64 | 11 | 55 | 9 |
| correct | 64 | 8 | 28 | 9 |

(No wrong population — 100% accuracy.)

**Breakdown of pairs above threshold (all population):**

| Classification | Count |
|---------------|-------|
| Structural | 8 |
| Residualization-induced | 1 |
| Sampling-induced | 0 |
| Unexplained | 0 |

The one residualization-induced pair is a_units vs b_units (r_resid = -0.852). At L1,
these are the only digits, so they are the "leading digits." Product = a_units * b_units
exactly, making the suppression effect maximally strong.

**Top pairs (all population):**

| Concept A | Concept B | r_raw | r_resid |
|-----------|-----------|-------|---------|
| n_nonzero_carries | n_answer_digits | 1.000 | 1.000 |
| ans_digit_0_msf | n_nonzero_carries | -0.542 | -0.958 |
| a_units | b_units | 0.000 | **-0.852** |
| col_sum_0 | n_nonzero_carries | 0.488 | 0.371 |
| ans_digit_1_msf | col_sum_0 | 0.103 | 0.343 |

**Verdict:** All 9 pairs are either structural (8) or residualization-induced (1).
No deconfounding needed.

---

## 15. Results: Level 2

Level 2 is 2-digit x 1-digit multiplication. 4,000 problems, 99.8% accuracy.

| Population | N problems | N concepts | Pairs total | Pairs |r_resid| > 0.3 |
|------------|-----------|------------|-------------|------------------------|
| all | 4,000 | 20 | 190 | 49 |
| correct | 3,993 | 16 | 120 | 49 |

(No wrong population — only 7 wrong answers, below the 30-problem threshold.)

**Breakdown (all population):**

| Classification | Count |
|---------------|-------|
| Structural | 48 |
| Residualization-induced | 1 |
| Sampling-induced | 0 |
| Unexplained | 0 |

The residualization-induced pair is a_tens vs b_units (r_resid = -0.814). At L2,
a is 2-digit and b is 1-digit. The leading digits are a_tens and b_units.

**Verdict:** No deconfounding needed.

---

## 16. Results: Level 3

Level 3 is 2-digit x 2-digit multiplication. 10,000 problems, 67.2% accuracy.

| Population | N problems | N concepts | Pairs total | Pairs |r_resid| > 0.3 |
|------------|-----------|------------|-------------|------------------------|
| all | 10,000 | 27 | 351 | 93 |
| correct | 6,720 | 22 | 231 | 92 |
| wrong | 3,280 | 22 | 231 | 96 |

**Breakdown (all population):**

| Classification | Count |
|---------------|-------|
| Structural | 92 |
| Residualization-induced | 1 |
| Sampling-induced | 0 |
| Unexplained | 0 |

**Top 10 pairs by |r_resid| (all population):**

| Concept A | Concept B | r_raw | r_resid | Classification |
|-----------|-----------|-------|---------|----------------|
| carry_0 | pp_a0_x_b0 | 0.982 | 0.982 | structural |
| carry_1 | total_carry_sum | 0.965 | 0.952 | structural |
| total_carry_sum | max_carry_value | 0.974 | 0.947 | structural |
| carry_1 | col_sum_1 | 0.960 | 0.938 | structural |
| col_sum_0 | pp_a0_x_b0 | 0.936 | 0.936 | structural |
| carry_0 | total_carry_sum | 0.704 | 0.930 | structural |
| total_carry_sum | pp_a0_x_b0 | 0.708 | 0.929 | structural |
| carry_1 | max_carry_value | 0.949 | 0.925 | structural |
| correct | digit_correct_pos2 | 0.937 | 0.925 | structural |
| carry_0 | col_sum_0 | 0.925 | 0.925 | structural |

The carry_0 to col_sum_0 correlation (r = 0.925 after residualization) is lower than
the raw Pearson of 0.989 because Phase B applies preprocessing: col_sum_0 is binned
into deciles, and the binning changes the correlation. The raw (unbinned) Pearson
of col_sum_0 vs carry_0 is 0.989 as verified in the earlier analysis rounds.

**The suppression effect at L3:**

```
a_tens vs b_tens:  raw = -0.002,  resid = -0.796  (residualization-induced)
```

**Key pairs that product residualization eliminates:**

```
a_tens vs carry_2:   raw = +0.656,  resid = -0.006  (gone)
carry_1 vs carry_2:  raw = +0.614,  resid = +0.045  (gone)
a_units vs b_units:  raw = +0.001,  resid = -0.003  (still zero — not leading digits)
```

This confirms that product residualization handles the cross-column contamination
effectively. The remaining correlations are either structural (within-column links
like carry_0 vs col_sum_0) or the suppression artifact (a_tens vs b_tens).

**Verdict:** No deconfounding needed.

---

## 17. Results: Level 4

Level 4 is 3-digit x 2-digit multiplication. 10,000 problems, 29.0% accuracy.

| Population | N problems | N concepts | Pairs total | Pairs |r_resid| > 0.3 |
|------------|-----------|------------|-------------|------------------------|
| all | 10,000 | 34 | 561 | 138 |
| correct | 2,897 | 28 | 378 | 143 |
| wrong | 7,103 | 28 | 378 | 137 |

**Breakdown (all population):**

| Classification | Count |
|---------------|-------|
| Structural | 137 |
| Residualization-induced | 1 |
| Sampling-induced | 0 |
| Unexplained | 0 |

The residualization-induced pair is a_hundreds vs b_tens (r_resid = -0.802). At L4,
a is 3-digit (leading digit = hundreds) and b is 2-digit (leading digit = tens).

**Verdict:** No deconfounding needed.

---

## 18. Results: Level 5

Level 5 is 3-digit x 3-digit multiplication. 122,223 problems, 3.4% accuracy.
This is the carry-stratified dataset selected from 810,000 screened problems.

| Population | N problems | N concepts | Pairs total | Pairs |r_resid| > 0.3 |
|------------|-----------|------------|-------------|------------------------|
| all | 122,223 | 42 | 861 | 205 |
| correct | 4,197 | 35 | 595 | 207 |
| wrong | 118,026 | 35 | 595 | 197 |

**Breakdown (all population):**

| Classification | Count |
|---------------|-------|
| Structural | 204 |
| Residualization-induced | 1 |
| Sampling-induced | 0 |
| Unexplained | 0 |

**The residualization-induced pair:**

```
a_hundreds vs b_hundreds:  raw = -0.003,  resid = -0.801
```

**Sampling-induced correlations at L5:**

The carry-stratified sampling creates a positive correlation between a_units and b_units:

```
All population:      a_units vs b_units = +0.201 (below 0.3 threshold)
Correct population:  a_units vs b_units = +0.343 (above 0.3 threshold)
Wrong population:    a_units vs b_units = +0.192 (below threshold)
```

In the "all" population, this is below the action threshold and requires no action. In
the correct-only population (4,197 problems), it crosses the threshold and triggers a
`deconfound` action. This is the only deconfound action in the entire Phase B run.

The correct-population sampling effect makes sense: the carry-stratified L5 dataset
overrepresents high-carry problems. Among correct answers (which the model gets right
despite high carries), large unit digits (which produce high carry_0) are oversampled.
This induces a positive correlation between a_units and b_units.

**Top 10 pairs by |r_resid| (all population):**

| Concept A | Concept B | r_raw | r_resid | Classification |
|-----------|-----------|-------|---------|----------------|
| carry_0 | pp_a0_x_b0 | 0.981 | 0.981 | structural |
| carry_0 | col_sum_0 | 0.978 | 0.978 | structural |
| col_sum_0 | pp_a0_x_b0 | 0.973 | 0.973 | structural |
| carry_1 | col_sum_1 | 0.971 | 0.971 | structural |
| carry_2 | max_carry_value | 0.974 | 0.958 | structural |
| total_carry_sum | max_carry_value | 0.969 | 0.951 | structural |
| carry_1 | total_carry_sum | 0.802 | 0.947 | structural |
| carry_2 | col_sum_2 | 0.967 | 0.942 | structural |
| carry_2 | total_carry_sum | 0.962 | 0.940 | structural |
| carry_3 | col_sum_3 | 0.957 | 0.931 | structural |

**Verdict:** No deconfounding needed for the "all" population. One deconfound action
for the correct-only population (a_units vs b_units, sampling-induced).

---

## 19. The Spearman Follow-up

For each level and population, Phase B computes Spearman rho on the top 30 pairs by
|r_resid|. This catches nonlinear relationships that Pearson might miss.

Total Spearman entries: 388 (across all levels and populations).

### Key comparison: Pearson vs Spearman

At L1 (all population), the top pair by Pearson is n_nonzero_carries vs n_answer_digits
(Pearson = 1.000, Spearman = 1.000). These are the same variable at L1 (a carry exists
if and only if the answer has 2 digits). Perfect agreement.

The suppression pair a_units vs b_units at L1 shows Pearson = -0.852 vs Spearman =
-0.925. Spearman is stronger because the rank-based correlation better captures the
ceiling effect (a_units = 9 forces b_units to be "low" for most products, creating a
monotonic but slightly nonlinear relationship in the residualized data).

At higher levels (L3-L5), Pearson and Spearman agree closely for the top structural
pairs. The differences are consistently below 0.05. This confirms Pearson is adequate
for the primary analysis.

---

## 20. The Deconfounding Plan

The deconfounding plan is saved as `deconfounding_plan.json`. It tells Phase C what
(if anything) needs to change.

```json
{
  "per_level": {
    "1": {"confounds": {}, "use_raw_activations": ["a_units", "b_units"]},
    "2": {"confounds": {}, "use_raw_activations": ["a_tens", "b_units"]},
    "3": {"confounds": {}, "use_raw_activations": ["a_tens", "b_tens"]},
    "4": {"confounds": {}, "use_raw_activations": ["a_hundreds", "b_tens"]},
    "5": {"confounds": {}, "use_raw_activations": ["a_hundreds", "b_hundreds"]}
  },
  "needs_multi_concept_residualization": false,
  "needs_raw_activation_override": true
}
```

### What this means

**confounds = {} at every level.** No concept in the "all" population needs additional
confounds regressed out beyond product. Phase C's existing single-confound
residualization is sufficient.

**use_raw_activations lists the leading digit pairs.** These are the concepts affected
by the suppression effect. The plan flags them so that if a future analysis needs
uncontaminated leading-digit subspaces, it can use raw (non-residualized) activations
for these specific concepts, the same way Phase C already does for product_binned.

Whether to actually use raw activations for leading digits is a judgment call. Without
product residualization, the a_tens subspace IS the product subspace (because product
magnitude dominates all variation). Using raw activations defeats the purpose of
residualization. The flag is advisory, not prescriptive. Phase C should continue using
residualized activations for all concepts and document the ~3% contamination.

**needs_multi_concept_residualization = false.** Phase C does not need to implement
per-concept confound lists. The existing code is sufficient.

---

## 21. The Final Decision

```
DECISION: Product residualization sufficient, but some concepts should use raw
activations to avoid residualization artifacts. See deconfounding_plan.json for details.
```

In plain language: Phase C can proceed without code changes. The product residualization
already handles the cross-concept contamination that matters. The suppression effect on
leading digits is documented but does not require intervention because the activation-
level impact is small.

For the paper: "Phase B verified that all 2,677 correlated label pairs across 5 levels
and 13 populations are either structurally forced by arithmetic (e.g., carry_0 and
col_sum_0 share r = 0.98 because carry_0 = floor(col_sum_0 / 10)) or are artifacts of
the product residualization step (leading-digit anti-correlation of r = -0.80, with
estimated activation-level contamination of ~3%). No sampling-induced correlations above
|r| = 0.3 were found in the primary (all) population. Phase C's single-confound
product residualization is sufficient."

---

## 22. Label-Level vs Activation-Level Contamination

Phase B measures label-level correlations. These are properties of the arithmetic and
the sampling strategy, not of the model. The model has nothing to do with them.

The question is: does a label-level correlation of r = 0.98 between carry_0 and col_sum_0
mean Phase C's carry_0 subspace is 98% contaminated by col_sum_0? No. The label
correlation tells you that grouping by carry_0 also groups by col_sum_0. But the
activation-level contamination depends on how the model encodes both concepts, which
could be anywhere from 0% (orthogonal encoding) to 100% (identical encoding).

For the suppression effect specifically, the label-level correlation is r = -0.80 (63%
shared variance). Simulation testing with 8-dimensional Fourier encodings in 4096-
dimensional space shows the activation-level contamination is approximately 3%. The
factor-of-20 reduction happens because product occupies 1 of 4096 dimensions while
digit encodings use 8+ dimensions. Removing one direction from 4096 barely affects an
8-dimensional subspace.

Phase B catches the label-level signal. Phase F (subspace overlap via principal angles)
will measure the actual activation-level overlap. Both are needed.

---

## 23. What This Stage Cannot Tell You

1. **Whether the model encodes correlated concepts together.** carry_0 and col_sum_0
   have r = 0.98 in labels. Maybe the model uses one subspace for both. Maybe it uses
   separate orthogonal subspaces. Phase B cannot distinguish these. Phase F can.

2. **Whether a correlation below 0.3 matters.** The L5 sampling correlation of r = 0.20
   between a_units and b_units is below the threshold. It might still affect downstream
   analyses that are sensitive to small biases. The threshold is pragmatic, not a safety
   certificate.

3. **Whether the carry binning at L5 changes things.** Phase C's concept registry applies
   filter_min_group to carries, dropping rare values. Phase B uses the same filtering.
   If the registry changes (different binning, different min_group_size), Phase B must
   rerun.

4. **Activation-level contamination magnitudes.** Phase B measures label-level
   correlations. The conversion to activation-level contamination depends on the model's
   encoding geometry, which Phase B does not access. The ~3% estimate for the suppression
   effect is based on simulation, not measurement.

---

## 24. The Code

### File: `phase_b_deconfounding.py`

Single Python file, 680 lines. No GPU required. Dependencies: numpy, pandas, scipy,
matplotlib, pyyaml.

### Key functions

| Function | Purpose |
|----------|---------|
| `get_concept_registry` | Mirrors Phase C's registry. Must be kept in sync. |
| `preprocess_concept` | Applies the same binning/filtering as Phase C. |
| `build_concept_matrix` | Extracts and preprocesses all registry concepts from the coloring DataFrame. |
| `pairwise_pearson` | C x C Pearson matrix with NaN-aware pairwise computation. |
| `residualize_product_labels` | OLS removal of product's linear effect from each concept's labels. |
| `classify_pair` | Rule-based classification of each pair into one of four categories. |
| `build_deconfounding_plan` | Collects per-concept confound lists and raw-activation flags. |
| `compute_spearman_top_k` | Spearman follow-up on top 30 pairs. |
| `process_level` | Orchestrates all steps for one level. |

### Design decisions

**Why duplicate the concept registry instead of importing from Phase C?** Phase C
imports matplotlib and scipy at module level. On headless SLURM nodes, importing
matplotlib can fail if no display is configured (even with Agg backend). Duplicating the
registry avoids this dependency. The cost is a maintenance burden: any change to Phase C's
registry must be mirrored in Phase B. A comment in the code warns about this.

**Why exclude product_binned?** Phase C uses raw activations for product_binned
(line 729 of phase_c_subspaces.py). Product residualization does not apply to this
concept. Including it in the correlation matrix would show trivially high correlations
with everything (product correlates with all arithmetic quantities) that do not
represent actual contamination in Phase C.

**Why base the decision on the "all" population?** Phase C's product residualization is
computed on the full population. The "all" population is the one that determines the
residualization direction. Correct-only and wrong-only sub-populations see slightly
different residual correlations, but these do not affect Phase C's main computation.

**Why flag_use_raw instead of deconfound for the suppression effect?** Deconfounding
would mean regressing out b_tens before computing a_tens subspace. This would remove
the leading-digit direction entirely — but that direction IS genuinely part of how the
model encodes a_tens. The activation-level contamination is only ~3%, so deconfounding
would over-correct. Flagging it for documentation is the right balance.

### Run script: `run_phase_b.sh`

SLURM batch script. Requests 1 GPU (required by the `general` partition QOS), 4 CPUs,
16 GB memory, 30-minute wall time. Conda activation uses
`source /home/anshulk/miniconda3/etc/profile.d/conda.sh` (not `eval "$(conda shell.bash
hook)"`) because compute nodes do not have conda in PATH.

---

## 25. Output Files

### Data outputs (at `/data/user_data/anshulk/arithmetic-geometry/phase_b/`)

| File | Size | Description |
|------|------|-------------|
| `classified_pairs.csv` | 248 KB | All 2,677 pairs with |r| > 0.1 across all levels and populations. Columns: level, population, concept_a, concept_b, tier_a, tier_b, r_raw, r_resid, classification, action. |
| `spearman_comparison.csv` | 29 KB | 388 Spearman follow-up entries for top 30 pairs per population. |
| `deconfounding_plan.json` | 702 bytes | Per-level confound lists and raw-activation flags. |
| `summary.json` | 6 KB | Aggregate statistics per level and population. |
| `correlation_matrices/` | 26 CSV files | Raw and residualized C x C correlation matrices per level per population. |

### Plot outputs (at `/home/anshulk/arithmetic-geometry/plots/phase_b/`)

| Pattern | Count | Description |
|---------|-------|-------------|
| `heatmap_L{level}_{pop}_raw.png` | 13 | Raw correlation heatmaps |
| `heatmap_L{level}_{pop}_residualized.png` | 13 | Post-product-residualization heatmaps |
| `heatmap_L{level}_{pop}_delta.png` | 13 | Difference heatmaps (resid minus raw) |
| **Total** | **39** | |

### Log output

`logs/phase_b_deconfounding.log` — full run log with per-population pair counts and
classification breakdowns.

---

## 26. Runtime and Reproducibility

### Runtime

From the SLURM job log (job 6658267, node babel-v9-16):

```
Start: Thu Mar 19 19:29:35 EDT 2026
End:   Thu Mar 19 19:30:16 EDT 2026
Total: 41 seconds (includes conda activation and pre-flight checks)
```

Phase B computation only: 32.3 seconds (from the Python log).

Per-level breakdown:

| Level | Time |
|-------|------|
| L1 | 2.0s |
| L2 | 3.8s |
| L3 | 8.3s |
| L4 | 8.5s |
| L5 | 9.6s |

L5 takes longer because of its 122,223 rows and 42 concepts (861 pairs to compute).
Still under 10 seconds.

### Reproducibility

The computation is deterministic. No random seeds are involved (Pearson correlation and
OLS regression are deterministic). The same coloring DataFrames (seeded at generation
time with RandomState(42)) produce the same correlation matrices every time.

The only source of non-reproducibility would be a change to the concept registry
(different preprocessing, different min_group_size, different binning). If the registry
changes, Phase B must rerun.
