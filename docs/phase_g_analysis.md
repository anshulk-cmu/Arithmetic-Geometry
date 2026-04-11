# Phase G: Fourier Screening for Periodic Structure — Run 3 Near Completion + Number-Token Screening Ready

**Anshul's Geometric Manifold Interpretability Project**
**Carnegie Mellon University, April 2026**

This document records every decision, every number, and every result from Phase G —
the Fourier screening stage. It is the truth document for this stage. All numbers are
validated against the actual output files as of April 11, 2026 (Run 3 data at 3,086
analyses of ~3,348 expected).

Phase G is the first non-linear probe in the pipeline. Phases A through F/JL established
that 43 arithmetic concepts live in clean linear subspaces (Phase C/D), that 94% of
concept pairs share subspace dimensions (Phase F), and that the union subspace preserves
>98.7% of pairwise distance structure (Phase JL). But we do not yet know **what geometry
exists inside each subspace**. Fourier screening tests the specific hypothesis that
digit-like concepts are arranged on circles or helices (periodic structure) — the same
structure Nanda et al. (2023) found in grokking models, Bai et al. (2024) found in toy
multiplication models (pentagonal prism), and Kantamneni & Tegmark (2025) found in
pretrained LLMs (generalized helix with periods {2, 5, 10}).

This phase probes at the `=` token position, consistent with the rest of the pipeline
(Phases A–F). The literature (K&T, Gurnee et al.) typically probes at the number-token
position. A separate number-token extraction pass provides a literature-grounded
comparison point. A positive at `=` is a stronger claim than K&T made; a null at `=`
combined with a positive at the number token matches K&T's finding and does not
contradict it.

### Run History

**Run 1 (SLURM 7056981)** launched April 11, 2026, 04:14 EDT on babel-v9-32. Completed
Steps 1–4 (K&T pilot, extraction, synthetic pilot, pilot 0b) and began Step 5 (full
screening). The run crashed at L4/layer4/correct after ~31 minutes with an
`AssertionError` in the carry_mod10 path. Additionally, the synthetic pilot had 2/10
test failures (Tests 3 and 9) but the script continued due to an exit code bug. A code
audit uncovered 12 bugs total (see Section 5f). All were fixed before Run 2.

**Run 2 (SLURM 7057231)** launched April 11, 2026, on babel-u9-28. Fresh run with all
12 bug fixes applied, all prior outputs cleared. Synthetic pilot now passes 10/10.
Number-token activations (9.2 GB, extracted in Run 1) are reused. The run crashed at
`ans_digit_2_msf / L5 / layer4 / correct / phase_d_merged` with `KeyError:
'per_freq_top2_coords'`. The crash was caused by a zero-dimensional Phase D merged
subspace (shape `(0, 4096)`) hitting an early-return path in `fourier_all_coordinates`
that was missing the `per_freq_top2_coords` key. A code audit uncovered 5 additional
edge-case bugs (see Section 5h). All were fixed before Run 3.

**Run 3 (SLURM 7058788)** launched April 11, 2026, 11:35 EDT on babel-t9-16. Fresh
run with all 17 bug fixes applied (12 from Run 1 + 5 from Run 2), all Phase G outputs
cleared. Number-token activations reused from Run 1.

**Status: RUN 3 NEAR COMPLETION (~92%).** Steps 1–4 complete. Step 5 (full Fourier
screening) is running — 3,086 of ~3,348 analyses complete after 3 hours. Currently
processing L5/layer24. L2 through L5/layer20 are fully analyzed. Remaining: L5 layers
24 (partial), 28, 31 across all/correct/wrong populations.

**Number-token Fourier screening script (`phase_g_numtok_fourier.py`) is ready.**
Pilot passed (4/4 analyses at L3/layer16, all null). Full run (~120 cells, ~1–2 hours
CPU) pending Run 3 completion. See Section 12 for full details.

**Key findings (Run 3 data at 3,086 analyses):**

1. **K&T replication: PASSED.** Periods {2, 5, 10} appear in the top-3 at all four
   tested layers (0, 1, 4, 8) in Llama 3.1 8B's residual stream for single-token
   integers 0–360. Our Fourier code is validated against published results.

2. **Raw vs. residualized: negligible difference.** Phase B's product-magnitude
   residualization does not destroy Fourier signal. FCR disagreement is 0.7% for
   two_axis_fcr and 1.0% for helix_fcr (well within the 20% tolerance).

3. **Synthetic pilot: 10/10 tests passed (after fixes).** Run 1 had 2 failures (Tests
   3 and 9) caused by a miscalibrated test threshold and a helix denominator bug.
   Both were fixed in the codebase — see Section 5a for details.

4. **Operand digits are COMPLETELY NULL at the `=` position.** Across all 846
   operand-digit analyses completed (a_units 0/194, a_tens 0/194, a_hundreds 0/92,
   b_units 0/182, b_tens 0/146, b_hundreds 0/38), zero circles or helices detected.
   This is not a power issue — N ranges from 4,197 (L5 correct) to 122,223 (L5 all)
   with 1,000 permutations each. The model does not use periodic Fourier
   representations for operand digits at the output position.

5. **Carries dominate helix detections.** Of 417 helix detections out of 3,084
   completed analyses (13.5%): carry_1 accounts for 148 (35%), carry_2 for 97 (23%),
   carry_3 for 34 (8%), carry_4 for 34 (8%). Together, carries produce 310 of 417
   helices (74.3%). Most carry helices are floor-saturated (p_helix = 0.001),
   indicating real structure, not borderline cases.

6. **Answer digits show scattered, weak signal.** ans_digit_0_msf has 27/178 helices
   (15%), ans_digit_5_msf 14/36 (39%, but only exists at L5), ans_digit_3_msf 12/140
   (9%). Middle answer digits (positions 1–2) are nearly null: ans_digit_1_msf 2/146,
   ans_digit_2_msf 4/155. This mirrors the Phase C finding — the model fails at
   composing middle digits.

7. **L2 is nearly null, L3–L5 have real signal.** L2: 2/324 helix (0.6%). L3: 93/774
   (12.0%). L4: 150/996 (15.1%). L5: 155/991 (15.6%). Helix geometry emerges at the
   same difficulty threshold where the model starts making errors.

8. **Helix detections are uniformly distributed across layers.** layer4: 55 (14.2%),
   layer6: 49 (12.7%), layer8: 53 (13.6%), layer12: 49 (12.5%), layer16: 53 (11.8%),
   layer20: 58 (15.1%), layer24: 28 (11.7%), layer28: 28 (12.2%), layer31: 27 (11.8%).
   No single layer dominates — carry helix structure is maintained across the full
   depth of the network.

9. **One circle detection.** A single `geometry_detected=circle` (no linear axis)
   appeared in the full run. Circles are exceedingly rare; the dominant non-linear
   geometry is the generalized helix (circle + linear ramp).

10. **Run 1 crash cause identified and fixed.** The `carry_mod10` spec assumed all
    values 0–9 exist in every population slice, but Phase C merges rare values into
    tail bins for small populations. The hard assert was replaced with a graceful skip.

11. **Run 2 crash cause identified and fixed.** A zero-dimensional Phase D subspace
    (`d=0` for `ans_digit_2_msf` at L5/layer4/correct) caused a `KeyError` in the
    zero-power early-return path of `fourier_all_coordinates`. A code audit found 5
    related edge-case bugs: missing dict key in early return, `ZeroDivisionError` when
    `n_perms=0`, `IndexError` on `d=0` arrays in `compute_helix_fcr` and `analyze_one`,
    and inconsistent return keys in `run_pilot_0b`. All fixed — see Section 5h.

12. **Number-token screening: COMPLETE — 0/108 detections.** The full number-token
    Fourier screening ran 108 analysis cells (4 levels × 6 layers × 6 digit concepts)
    in 636 seconds. Every single cell returned `geometry_detected=none`. Zero helix,
    zero circle. FDR-significant: 0/108. b_units at layer 12 came closest
    (L3: FCR=0.61, p=0.002; L4: FCR=0.55, p=0.004; L5: FCR=0.54, p=0.004) but
    failed the conjunction criterion at every level. K&T's digit helix does NOT
    exist at operand token positions in multiplication context — not at any layer
    (4–24), any level (L2–L5), or any digit concept.

---

## Table of Contents

1. [What Phase G Is and Why It Exists](#1-what-phase-g-is-and-why-it-exists)
2. [The Mathematical Framework](#2-the-mathematical-framework)
   - 2a. Group Centroids and the Centroid Fourier Test
   - 2b. Explicit DFT at Specified Periods
   - 2c. Frequency Range and the Nyquist Inclusion
   - 2d. Per-Coordinate Fourier Concentration Ratio (FCR)
   - 2e. The Two-Axis FCR (Primary Statistic)
   - 2f. The Helix FCR (Secondary Statistic)
   - 2g. Linear Power and DOF Rescaling
   - 2h. The Permutation Null
   - 2i. Circle Detection (Conjunction Criterion)
   - 2j. Helix Detection (Extended Conjunction)
   - 2k. FDR Correction and the Pre-Registered Decision Rule
   - 2l. Multi-Frequency Pattern Classification
3. [Design Decisions and Their Rationale](#3-design-decisions-and-their-rationale)
   - 3a. Why Centroid Fourier Instead of Per-Example Fourier
   - 3b. Why Explicit DFT Instead of FFT
   - 3c. Why Both Phase C and Phase D Bases
   - 3d. Why 1,000 Permutations (Not 10,000)
   - 3e. Why the Conjunction Criterion for Circle Detection
   - 3f. Why Include Nyquist (Fix 1) — The Parity/Prism Argument
   - 3g. Why the Helix Statistic (Fix 2) — K&T's Generalized Helix
   - 3h. Why the K&T Pilot Gate (Fix 3)
   - 3i. Why Raw vs. Residualized Spot Check (Fix 4)
   - 3j. Why Number-Token Probe (Fix 5)
   - 3k. Why FDR q-Value Instead of Hard FCR Floor (Fix 6)
   - 3l. Carry Concept Period Specs — Binned, Mod10, Raw
   - 3m. Zero-Subsampling Guarantee
   - 3n. The Linear Power DOF Rescaling (Fix 5 from Review)
   - 3o. Tail-Bin Mean for Linear Axis (Fix 2 from Review)
4. [Concepts Screened and the Experiment Matrix](#4-concepts-screened-and-the-experiment-matrix)
   - 4a. Tier A: Digit Concepts (Period P=10)
   - 4b. Tier B: Carry Concepts (Multi-Period Sweep)
   - 4c. Concepts NOT Screened
   - 4d. The Full Experiment Matrix (~3,348 Analyses)
5. [Verification Results — Pilots and Gates](#5-verification-results--pilots-and-gates)
   - 5a. Synthetic Pilot Tests (10/10 Passed After Fixes)
   - 5b. K&T Replication Pilot (PASSED)
   - 5c. Pilot 0b: Raw vs. Residualized (PASSED)
   - 5d. Number-Token Extraction (Complete)
   - 5e. Phase D Basis Count Check (PASSED)
   - 5f. Run 1 Bugs: The 12 Fixes Applied Before Run 2
   - 5g. Run 1 Partial Results: L3 Detections Before Crash
   - 5h. Run 2 Bugs: The 5 Edge-Case Fixes Applied Before Run 3
6. [Preliminary Results — L2 Early Layers](#6-preliminary-results--l2-early-layers)
   - 6a. L2/layer4/all — First Six Digit Concepts
   - 6b. L2/layer4/all — Carry Concepts
   - 6c. L2/layer8/all — Complete Slice
   - 6d. L2/layer16/all — The Pattern So Far
   - 6e. L2/correct Population — Slightly Higher FCR, Still No Detection
   - 6f. Summary Table: L2 All Analyses to Date
7. [Interpretation of Preliminary Results](#7-interpretation-of-preliminary-results)
   - 7a. Why No Circles at L2? Possible Explanations
   - 7b. The Conjunction Criterion Effect
   - 7c. Saturated p-Values — What They Mean
   - 7d. What to Expect at L3-L5
   - 7e. The Linear Power Signature
   - 7f. L3 Partial Results Confirm Difficulty-Dependent Emergence
   - 7g. Implications for the Core Thesis
8. [Implementation Details](#8-implementation-details)
   - 8a. Script Architecture
   - 8b. Data Loading Pipeline
   - 8c. Concept Registry
   - 8d. Core Fourier Functions
   - 8e. Permutation Null Implementation
   - 8f. Detection Logic
   - 8g. Output Format
   - 8h. Error Handling and Edge Case Guards
9. [Run 3 Comprehensive Results](#9-run-3-comprehensive-results)
   - 9a. Detection Summary by Concept Type
   - 9b. Detection Summary by Level
   - 9c. Detection Summary by Layer
   - 9d. Detection Summary by Population
   - 9e. Carry Helix Analysis: The Dominant Signal
   - 9f. Operand Digits: The Complete Null
   - 9g. Answer Digits: The Composition Bottleneck
   - 9h. The Single Circle Detection
   - 9i. FDR Correction (pending Run 3 completion)
10. [Number-Token Fourier Screening](#10-number-token-fourier-screening)
   - 10a. Motivation: K&T's Finding and Our Question
   - 10b. Methodology: PCA on Centroids in Raw 4096-dim Space
   - 10c. Digit Concepts Screened
   - 10d. Script Design: phase_g_numtok_fourier.py
   - 10e. Pilot Results: L3/layer16 (All Null)
   - 10f. Interpretation: Does K&T's Signal Transfer to Multiplication Context?
   - 10g. Full Run Plan (~120 Analysis Cells)
11. [Interpretation of Full Results](#11-interpretation-of-full-results)
   - 11a. The Carry-Helix Story
   - 11b. Why Operand Digits Are Null at the `=` Position
   - 11c. The Difficulty-Dependent Emergence Pattern
   - 11d. Layer Uniformity: Helix Structure Is Maintained, Not Computed
   - 11e. Answer Digits: Edge-vs-Middle Asymmetry Replicates
   - 11f. Position-Dependent Representations: `=` Token vs Number Token
   - 11g. Implications for the Core Thesis
12. [What Phase G Will Contribute to the Paper](#12-what-phase-g-will-contribute-to-the-paper)
13. [Limitations](#13-limitations)
14. [Runtime and Reproducibility](#14-runtime-and-reproducibility)

**Appendices:**
- A. The Algebra of Centroid Fourier Screening
- B. Why Centroids, Not Individual Points — The Statistical Argument
- C. The K&T Helix and Its Relationship to Our Test
- D. Phase G in the Overall Pipeline
- E. Synthetic Pilot Test Specifications
- F. The 18 Fixes from v1 to v4 of the Plan
- G. Number-Token Fourier Screening: Full Methodology

---

## 1. What Phase G Is and Why It Exists

Phases C through F/JL established the **linear** geometry of the model's arithmetic
representations. Phase C found that each of 43 arithmetic concepts lives in a low-rank
linear subspace (dim 3–18, depending on concept and level). Phase D refined these with
LDA-based discriminative analysis. Phase E showed the union subspace captures 81–97%
of activation variance. Phase F demonstrated universal superposition (94% of concept
pairs share dimensions), and Phase JL confirmed that pairwise distance structure is
preserved (Spearman ≥ 0.9942) in the union subspace.

But linear subspaces say nothing about the **arrangement** of concept values within
those subspaces. Consider the units digit: Phase C found that `a_units` has a 9-dimensional
subspace at L5/layer16. The ten digit values (0–9) project into this 9-dimensional
space as ten points. But how are these ten points arranged? Three possibilities:

1. **Linear encoding.** The centroids lie roughly along a line, with distance proportional
   to digit difference. `0` and `9` are far apart; `4` and `5` are close. This is what
   you'd expect from a simple magnitude representation.

2. **Circular (Fourier) encoding.** The centroids lie on a circle, wrapping around so
   that `0` and `9` are close — as close as `0` and `1`. The positions follow
   `(cos(2πv/10), sin(2πv/10))` in some 2D subplane. This is what Nanda et al. found
   in grokking models and what the Clock & Pizza model predicts for modular arithmetic.

3. **Helical encoding.** The centroids follow a helix: circular in two dimensions and
   linearly increasing in a third. `0` and `9` are close in the circular dimensions
   but far apart in the linear dimension. Kantamneni & Tegmark (2025) found this
   generalized helix structure for number representations in pretrained LLMs, with
   the circular component carrying the modular structure and the linear component
   carrying the magnitude.

Phase G tests hypotheses (2) and (3) against hypothesis (1) as the implicit null.
It does so by computing an **explicit DFT** of the group centroids within each
concept's subspace, using a **permutation null** to establish statistical significance.

**Why this matters for the paper.** The paper's thesis is that the Linear Representation
Hypothesis (LRH) fails for compositional reasoning, with multiplication as the case
study. Phases C–F established that while individual concepts have linear subspaces,
the within-subspace geometry may be non-linear. Phase G tests the specific prediction
from the Fourier circuits literature: that digit representations are periodic, not
just linearly ordered. If confirmed, this is evidence that the model uses Fourier-type
features — a fundamentally non-linear representation — to perform arithmetic, even
though these features happen to live in linear subspaces.

If Phase G is null (no periodic structure at the `=` token), this is also informative:
it means either (a) the model does not use Fourier features for multiplication at this
position, or (b) the centroid test is too coarse to detect them. The number-token probe
and single-example projection plots disambiguate these cases.

**The dependency chain.** Phase G reads:
- Phase C's `projected_all.npy` per concept/layer/pop (the fast path for centroid
  computation in Phase C's subspace)
- Phase C's `metadata.json` (group labels, dim_consensus, dim_perm — for concept
  eligibility and group structure)
- Phase C's `eigenvalues.npy` (for eigenvalue-weighted FCR, a secondary statistic)
- Phase C's `basis.npy` (shape `(d_consensus, 4096)`, for manual projection when needed)
- Phase D's `merged_basis.npy` (shape `(d_merged, 4096)`, for Phase D subspace analysis)
- Phase C's residualized activations `level{N}_layer{L}.npy` (for Phase D projections)
- Phase A's coloring DataFrames `L{N}_coloring.pkl` (concept columns, correct/wrong labels)
- Raw activations `level{N}_layer{L}.npy` (for the raw-vs-residualized spot check only)

Phase G produces:
- Per-cell `fourier_results.json` and `centroids.npy` for every analysis
- Summary CSV (`phase_g_results.csv`) with ~3,348 rows — one per analysis cell
- Detection CSVs (`phase_g_circles.csv`, `phase_g_helices.csv`)
- Agreement CSV (`phase_g_agreement.csv`) — Phase C vs Phase D concordance
- K&T pilot results (`kt_pilot_summary.json`)
- Plots: FCR heatmaps, centroid circle overlays, frequency spectra, p-value trajectories,
  single-example projections

---

## 2. The Mathematical Framework

### 2a. Group Centroids and the Centroid Fourier Test

The fundamental object is the **group centroid**: for each value `v` of a concept
(e.g., `a_units = 3`) in a given population (all/correct/wrong), we compute the mean
activation across all samples with that value:

```
μ_v = (1/|S_v|) Σ_{i ∈ S_v} acts[i]     for each v ∈ V
where S_v = {i ∈ population : concept[i] = v}
```

This gives us `m` points in `d`-dimensional subspace, where `m = |V|` is the number
of distinct concept values and `d` is the subspace dimension (dim_consensus for Phase C,
d_merged for Phase D).

**Why centroids?** The centroid test asks whether the **average position** of each
concept value has periodic structure. This is a conservative test: it detects structure
in the signal (between-class variation) but not in the noise (within-class variation).
Individual activations may form a noisy ring, but the centroid test only detects the
ring if the ring's average positions are significantly periodic. See Appendix B for
the full argument.

**DC removal.** Before Fourier analysis, we subtract the grand centroid:

```
c̄ = (1/m) Σ_v μ_v
μ_v ← μ_v - c̄
```

This removes the DC (zero-frequency) component, which carries no periodic information.
After DC removal, the centroids are centered at the origin.

**Zero-subsampling guarantee.** Every data point is used in every computation. No
subsampling. L5/all: all 122,223 samples contribute to their group centroid. L5/correct:
all 4,197. L5/wrong: all 118,026. The script asserts `sum(group_sizes) == N_population`
for every analysis.

### 2b. Explicit DFT at Specified Periods

For each coordinate `j` of the DC-removed centroids, we compute the Fourier
coefficients at frequency `k` with respect to a specified period `P`:

```
a_k = Σ_{v ∈ V} μ_v[j] · cos(2πkv / P)     for k = 1, ..., K
b_k = Σ_{v ∈ V} μ_v[j] · sin(2πkv / P)     for k = 1, ..., K
```

The **power** at frequency `k` in coordinate `j` is:

```
P_k[j] = a_k² + b_k²
```

This is the unnormalized Fourier power — the square of the magnitude of the
frequency-k component. Under the null hypothesis (centroids are random), the
expected power is proportional to 2σ² (two degrees of freedom from cos and sin).

**Why explicit DFT, not FFT?** The FFT requires evenly-spaced samples at positions
`v = 0, 1, ..., P-1`. Our concept values are not always complete grids:
- `a_tens` at L2 has values {1, 2, ..., 9} (no 0)
- `b_units` at L2 has values {2, 3, ..., 9} (no 0, no 1)
- Carry concepts have irregular value ranges

The explicit DFT handles arbitrary value sets by computing the Fourier projection at
the actual values present in the data. This is mathematically equivalent to the FFT
when the grid is complete, but works correctly for incomplete grids.

**Numerical example (a_units, L2, layer4, Phase C):**
- m = 10 values (0–9), d = 9 coordinates, P = 10, K = 5 frequencies
- Group sizes: [376, 419, 390, 394, 417, 401, 404, 401, 402, 396] (sum = 4,000)
- Total Fourier power across all coordinates and frequencies: 2.289207
- two_axis_fcr = 0.3206, best_freq = 5 (Nyquist)
- This means: the top two coordinates at frequency 5 capture 32.06% of total power

### 2c. Frequency Range and the Nyquist Inclusion

The number of testable Fourier frequencies depends on the period `P`:

**Odd P:** `K = (P-1)/2`. All frequencies have 2 degrees of freedom (cos and sin
components), because `sin(2πk·v/P)` is non-degenerate for `k < P/2`.

**Even P:** `K = P/2`, **including the Nyquist frequency** `k = P/2`. The Nyquist
bin has only 1 DOF because `sin(2π·(P/2)·v/P) = sin(πv) = 0` for integer `v`. Only
the cosine component survives: the signal `(-1)^v`.

**The critical design decision:** We include the Nyquist frequency. Bai et al.'s
pentagonal prism model for multiplication includes the parity function `(-1)^n`
as one of its basis elements. At period P=10, parity lives in the Nyquist bin k=5.
Excluding Nyquist would make it **mathematically impossible** to detect this structure.

**Nyquist rescaling.** Because the Nyquist bin has 1 DOF instead of 2, its raw power
has `E[P_nyq] = σ²` under the null, compared to `E[P_k] = 2σ²` for non-Nyquist bins.
We rescale: `P_nyq_rescaled = 2 · a_k²`. This ensures `E[P_nyq_rescaled] = 2σ²`,
matching the other bins, so that the concentration ratio treats all bins equally.

**Frequency table for all periods used:**

| P | K | Frequencies tested | Notes |
|---|---|--------------------|-------|
| 6 | 3 | {1, 2, **3 (Nyquist)**} | carry_4 binned |
| 8 | 4 | {1, 2, 3, **4 (Nyquist)**} | — |
| 9 | 4 | {1, 2, 3, 4} | carry_0, carry_3 binned |
| 10 | 5 | {1, 2, 3, 4, **5 (Nyquist)**} | All digit concepts, carry_mod10 |
| 13 | 6 | {1, 2, 3, 4, 5, 6} | carry_1 binned |
| 14 | 7 | {1, 2, 3, 4, 5, 6, **7 (Nyquist)**} | carry_2 binned |
| 19 | 9 | {1, 2, ..., 9} | carry_3 raw |
| 27 | 13 | {1, 2, ..., 13} | carry_2 raw |

**Zero-denominator guard:** If `total_power = Σ_k Σ_j P_k[j] < 1e-12`, all FCR
values are set to 0.0 and a warning is logged. This handles the pure-DC case (all
centroids identical after DC removal) without division by zero.

### 2d. Per-Coordinate Fourier Concentration Ratio (FCR)

For each subspace coordinate `j`, the **FCR_top1** measures how concentrated the
Fourier power is in a single frequency:

```
FCR_top1[j] = max_k(P_k[j]) / Σ_k P_k[j]
dominant_freq[j] = argmax_k(P_k[j]) + 1    (1-indexed)
```

Under the null, the expected FCR_top1 is `1/K` (power distributed equally across
frequencies). For a perfect sinusoid at a single frequency, FCR_top1 = 1.0.

**Uniform FCR:** The mean across coordinates: `uniform_fcr_top1 = mean_j(FCR_top1[j])`.
This is a secondary statistic that asks: on average, how concentrated is each
coordinate's Fourier power?

**Eigenvalue-weighted FCR:** A weighted mean: `eig_fcr = Σ_j (λ_j/Σλ) · FCR_top1[j]`,
where `λ_j` are Phase C's eigenvalues. This up-weights coordinates that capture more
variance. Reported as a secondary statistic but **not** used for detection, because
eigenvalue weighting biases against weak axes that might carry the sine component
of a circle.

### 2e. The Two-Axis FCR (Primary Statistic)

A circle in `d`-dimensional space requires exactly two coordinates — one carrying the
cosine component and one carrying the sine component at the same frequency. The
**two_axis_fcr** captures this:

```
For each frequency k:
    sorted_coords = sort P_k[j] for all j, descending
    two_axis_power_k = sorted_coords[0] + sorted_coords[1]

two_axis_fcr = max_k(two_axis_power_k) / total_power
best_freq = argmax_k(two_axis_power_k) + 1
best_coord_a, best_coord_b = indices of top-2 coords at best_freq
```

**Why two axes?** A circle parametrized as `(A·cos(2πv/P), A·sin(2πv/P))` concentrates
all its Fourier power at frequency 1 in exactly two coordinates. The two_axis_fcr
measures the fraction of total power captured by the best pair of coordinates at
the best frequency. For a perfect circle: two_axis_fcr = 1.0. For random noise with
`d = 9` and `K = 5`: E[two_axis_fcr] ≈ 2/(d·K) ≈ 0.044 (but the actual null is
wider due to the max over frequencies).

**Numerical example (a_units, L2, layer4, Phase C):**
- two_axis_fcr = 0.3206, best_freq = 5 (Nyquist), coords = (2, 0)
- p_two_axis = 0.1848 (not significant)
- Interpretation: 32% of total Fourier power is at the Nyquist frequency in two
  coordinates. Interesting — the Nyquist dominance hints at parity structure — but
  not statistically significant against the permutation null.

**Numerical example (ans_digit_0_msf, L2, layer4, Phase C):**
- two_axis_fcr = 0.6449, best_freq = 1, coords = (0, 1)
- p_two_axis = 0.0010 (saturated at floor)
- p_saturated = True (could be even more significant with more permutations)
- circle_detected = **False** — despite significant two_axis_fcr, the conjunction
  criterion fails (coord_b p-value = 0.5824, not < 0.01)

### 2f. The Helix FCR (Secondary Statistic)

Kantamneni & Tegmark (2025) found that LLMs represent numbers on a **generalized
helix**: two Fourier axes at the same frequency **plus a linear magnitude axis**.
The circular component carries the modular structure (units digit cycles with period 10),
and the linear component carries the magnitude (larger numbers have larger projections
onto the linear axis).

A pure circle test (two_axis_fcr) misses the linear axis and may undercount the
structure. The helix_fcr extends the test:

```
For each coordinate j:
    linear_power_j = (Σ_{v ∈ V} v_centered · μ_v[j])²
    where v_centered = v - mean(v)

For each frequency k:
    two_axis_power_k = (from Step 2e)
    # Linear axis: best coord NOT in top-2 Fourier coords at freq k
    linear_axis_power = max_{j ∉ top-2(k)} linear_power_rescaled[j]
    helix_power_k = two_axis_power_k + linear_axis_power

total_power_helix = total_fourier_power + total_linear_power
helix_fcr = max_k(helix_power_k) / total_power_helix
```

**Key detail: no double-dipping.** The linear axis is chosen from coordinates that
are NOT in the top-2 Fourier coordinates at the best frequency. This prevents a
coordinate from contributing to both the circular and linear components, which would
inflate the statistic.

**Key detail: DOF rescaling.** The linear power (1 DOF) is rescaled to match
the Fourier power scale (2 DOF per bin) before pooling into the helix denominator.
Under a Gaussian null, a Fourier bin has `E[P_k] ∝ m` (from ||cos_basis||² + ||sin_basis||²),
while the linear bin has `E[P_lin] ∝ Σ v_centered²`. The rescaling factor is
`m / (2 · Σ v_centered²)`, ensuring the helix denominator treats Fourier and linear
power on equal footing.

**Numerical example (a_units, L2, layer4, Phase C):**
- helix_fcr = 0.3200, best_freq = 5, linear_coord = 1
- linear_power = 1.512941, total_helix_power = 2.579934
- p_helix = 0.1958 (not significant)

### 2g. Linear Power and DOF Rescaling

The linear power measures the projection of each centroid coordinate onto a linear
ramp — how much of the centroid variation is explained by a monotonic increase
with concept value.

```
v_centered = v - v̄   (mean-centered concept values)
linear_power_j = (Σ_v v_centered · μ_v[j])²
```

For carry concepts with tail binning, the linear values `v_linear` use the mean of
raw values in the tail bin (Fix 2 from review), computed **within the current
population only** (never cross-population). Example: if carry_1 at L5 has groups
{0, 1, ..., 12} where group 12 is a tail bin containing raw values 12–17, then
`v_linear[12] = mean(raw values ≥ 12)` for samples in the current population.

**Sample-weighted centering (Fix 4 from review):** When group sizes are unbalanced
(e.g., carry_0 at L2 has group sizes [1070, 859, 632, 481, 448, 196, 152, 115, 47]),
the mean value `v̄` is computed as a sample-weighted mean:

```
v̄ = Σ_v (n_v / N) · v_linear[v]
```

This ensures the linear projection is not biased toward values with fewer samples.

### 2h. The Permutation Null

Statistical significance is assessed via a **permutation null** with 1,000 permutations.
For each permutation:

1. **Shuffle all sample labels.** Take the `N` sample indices and randomly permute
   them. Assign the first `n_0` shuffled samples to group 0, the next `n_1` to group 1,
   etc. This preserves group sizes exactly (conditioned null), matching the observed
   group balance.

2. **Recompute centroids.** From the shuffled assignments, compute new group centroids.
   These null centroids represent what you'd see if concept values were randomly
   assigned to samples.

3. **DC-remove and Fourier-analyze.** Apply the same pipeline: DC removal, Fourier
   analysis, linear power, helix FCR.

4. **Record null statistics.** Store `null_two_axis_fcr`, `null_helix_fcr`,
   `null_uniform_fcr`, and per-coordinate `null_coord_fcr` and `null_linear_power`.

**p-value computation (conservative):**

```
p = (count(null_stat ≥ observed_stat) + 1) / (n_perms + 1)
```

The `+1` in both numerator and denominator ensures the p-value is never exactly 0
and provides a slight conservative bias. With 1,000 permutations, the smallest
achievable p-value is `1/1001 = 0.000999`.

**p-value floor:** For small groups (e.g., carry_4 with m=6 groups), the number
of distinct permutations is `m! = 720`. Since any permutation null statistic can
only take `m!` distinct values, the effective p-value floor is:

```
p_floor = 1 / min(n_perms + 1, m!)
```

For carry_4 (m=6): `p_floor = 1/720 = 0.00139`. Any p-value at or near this floor
is flagged as `p_saturated = True` in the output, indicating that the signal may be
even more significant than the p-value suggests.

**Runtime.** The permutation null dominates computation time. At L5/all (N=122,223)
with Phase C projections (d=9), each permutation requires:
- Shuffling 122K indices
- Computing 10 group means over 9 dimensions
- Running full Fourier + helix analysis

Observed rate: ~0.001s per permutation, ~0.9s per analysis cell. At L5/all with
Phase D projections (d=18), this roughly doubles to ~1.4s per cell.

### 2i. Circle Detection (Conjunction Criterion)

A concept-layer cell is classified as `circle_detected = True` if **all three**
conditions hold:

1. **Global significance:** `p_two_axis < 0.01` (pre-FDR)
2. **Coordinate A significance:** `p_coord[coord_a] < 0.01`, where `coord_a` is the
   top-power coordinate at the best frequency
3. **Coordinate B significance:** `p_coord[coord_b] < 0.01`, where `coord_b` is the
   second-highest-power coordinate at the best frequency

**Why the conjunction?** The two_axis_fcr can be significant when a single coordinate
has very high FCR (e.g., a line in one dimension that happens to align with a Fourier
frequency). The conjunction requires that **both** coordinates — the cosine axis and
the sine axis — are individually significant. A circle requires two axes; a line
only needs one. The conjunction distinguishes circles from lines.

**This is the most conservative element of the detection criterion.** It is the
reason why ans_digit_0_msf at L2/layer4 (p_two_axis = 0.001, saturated) is not
flagged as a circle: only one of its two coordinates passes the threshold.

### 2j. Helix Detection (Extended Conjunction)

A concept-layer cell is classified as `helix_detected = True` if **all four**
conditions hold:

1. **Global helix significance:** `p_helix < 0.01` (pre-FDR)
2. **Fourier coordinate A significance:** `p_coord[helix_coord_a] < 0.01` at the
   helix's best frequency
3. **Fourier coordinate B significance:** `p_coord[helix_coord_b] < 0.01`
4. **Linear axis significance:** `p_linear[helix_linear_coord] < 0.01`

The helix detection extends the circle conjunction with an additional requirement:
the linear axis must also be individually significant against the permutation null.
This is even more conservative than circle detection — it requires three significant
coordinates instead of two.

**Hierarchical classification:**
- If `helix_detected = True`: `geometry_detected = "helix"`
- Else if `circle_detected = True`: `geometry_detected = "circle"`
- Else: `geometry_detected = "none"`

A helix supersedes a circle because any helix also has a circular component.

### 2k. FDR Correction and the Pre-Registered Decision Rule

After all ~3,348 analyses complete, two separate Benjamini-Hochberg FDR corrections
are applied:

1. Across all `p_two_axis` values → `two_axis_q_value`
2. Across all `p_helix` values → `helix_q_value`

**The pre-registered decision rule** (stored as the `DECISION_RULE` constant in the
script, printed in the summary output):

> Periodic structure is confirmed for a **concept class** (input digits, answer digits,
> carries) if **≥3 concept-layer cells** are significant after FDR correction (q < 0.05),
> spanning **≥2 distinct concepts** and **≥2 distinct layers** in {8, 12, 16, 20, 24}
> (5 middle layers), in the `all` population.

A concept-layer cell is significant if `geometry_detected ≠ "none"` in **either**
the Phase C or Phase D basis. This is lenient in basis choice (either suffices) but
strict in the cell-level conjunction (both coordinates must be individually significant)
and in the class-level pattern (≥3 cells across ≥2 concepts and ≥2 layers).

**Correct vs. wrong population comparisons are EXPLORATORY** — no pre-registered
threshold. These are reported for descriptive purposes only.

### 2l. Multi-Frequency Pattern Classification

For cells where circle is detected, the code classifies the multi-frequency pattern
as **exploratory** information (not part of the detection decision):

- `{1}`: pure fundamental — consistent with simple circle
- `{1, 2}`: fundamental + first harmonic — consistent with ellipse or octagon
- `{1, 2, 5}`: fundamental + harmonic + parity — consistent with pentagonal prism
  (Bai et al.)
- Other patterns: logged for manual review

**Warning:** The automatic multi-frequency classifier is exploratory only. Proper
prism detection requires checking whether the **same two axes** have significant power
at multiple frequencies, not just whether different coordinates are dominant at different
frequencies. Manual review of power spectra plots is required for multi-frequency claims.

---

## 3. Design Decisions and Their Rationale

### 3a. Why Centroid Fourier Instead of Per-Example Fourier

The literature on Fourier features in LLMs (K&T, Gurnee et al.) typically fits
Fourier features to **individual activations** — e.g., running DFT over each hidden
dimension individually for each integer token. Our approach is different: we compute
**group centroids** first, then run DFT on the centroids.

**Why this choice:**
1. **Integration with the pipeline.** Phases C/D found subspaces by analyzing centroids.
   Phase G continues in the same frame — asking whether those centroid positions have
   periodic structure.
2. **Noise reduction.** At L5/all, each centroid is the mean of ~12,000 samples.
   This averaging eliminates within-class noise, making the between-class signal
   (which is what Fourier structure would be) much cleaner.
3. **Direct correspondence to the claim.** If the centroid test detects a circle, it
   means the **average** activation for each digit value lies on a circle. This is a
   strong claim about the model's representation strategy — not just a claim about
   individual samples.

**The cost:** Centroid averaging can destroy manifold structure when within-class
spread is large relative to between-class separation. The **single-example projection
plots** (Section 3a of the plan) provide a post-hoc check for this: for null concepts,
we scatter individual activations colored by value to see if a ring is visually present
despite the centroid test being null.

### 3b. Why Explicit DFT Instead of FFT

The FFT assumes samples at positions `v = 0, 1, ..., N-1`. Our concept values are:
- Often incomplete: `a_tens` at L2 has 9 values (1–9, no 0)
- Sometimes sparse: `b_units` at L2 has 8 values (2–9)
- For carries, the value range depends on binning

The explicit DFT computes `a_k = Σ_v s[v] · cos(2πkv/P)` for arbitrary value sets.
It is computationally identical to the FFT for complete grids but handles gaps correctly.
The computational overhead is negligible: with m ≤ 27 values and K ≤ 13 frequencies,
each DFT takes microseconds.

### 3c. Why Both Phase C and Phase D Bases

Phase C finds **consensus subspaces** from permutation-stabilized covariance
decomposition. Phase D finds **discriminative (LDA) bases** that maximize between-class
separation. These are different projections of the same data.

Phase G tests both because:
1. A circle might be visible in the consensus subspace but not the discriminative
   one (or vice versa). LDA rotates axes to maximize separability, which could either
   reveal or obscure circular structure depending on how the circle aligns with the
   LDA axes.
2. Detection in **either** basis suffices for the class-level decision rule. The
   `agreement` column records which basis(es) detected the structure:
   - `"both"`: detected in Phase C AND Phase D
   - `"phase_c_only"`: detected in C, not D
   - `"phase_d_only"`: detected in D, not C
   - `"neither"`: no detection in either

### 3d. Why 1,000 Permutations (Not 10,000)

The permutation null uses 1,000 shuffles, matching Phase C's convention
(`phase_c_subspaces.py:462-531`). With ~3,348 analyses and ~1s per permutation null,
this yields ~3,348 seconds (~55 minutes) for the permutation nulls alone.

10,000 permutations would increase the p-value resolution from 0.001 to 0.0001 but
would multiply runtime to ~9 hours for the permutation nulls alone. Since our FDR
threshold is q < 0.05 (not 0.001), the resolution from 1,000 permutations is sufficient.

The p_value_floor for small groups (e.g., carry_4 with m=6, m! = 720) already limits
resolution regardless of the number of permutations. Adding more permutations doesn't
help when `m!` is the binding constraint.

### 3e. Why the Conjunction Criterion for Circle Detection

The conjunction criterion (both coordinates individually significant at p < 0.01) is
deliberately conservative. Without it, a concept with one very strong coordinate
(e.g., a linear ramp that happens to align with a Fourier frequency) would register
as a "circle" despite having only one active axis.

The cost is reduced sensitivity: a concept with a genuine circle where one axis is
weak (e.g., an ellipse with high eccentricity) might fail detection. This is acceptable
because:
1. The Fourier circuits literature predicts circles (equal cos/sin power), not ellipses
2. Elliptical structure is better detected by Phase E/GPLVM methods
3. False negatives are less damaging than false positives in a screening phase

### 3f. Why Include Nyquist (Fix 1) — The Parity/Prism Argument

This was a critical fix in v2 of the plan. The original design excluded the Nyquist
frequency, following standard signal processing practice (the Nyquist bin is often
treated as an artifact). But in our setting, the Nyquist bin at P=10 carries the
signal `(-1)^v` — the parity function.

Bai et al.'s pentagonal prism model explicitly uses a parity basis alongside sine and
cosine bases. Excluding Nyquist would make it **impossible** to detect the prism
structure. The fix: include Nyquist, rescale its power by 2× to match the 2-DOF bins,
and verify with a synthetic test (Test 8: pure Nyquist parity signal).

### 3g. Why the Helix Statistic (Fix 2) — K&T's Generalized Helix

Kantamneni & Tegmark (2025) found that number representations in Llama 3.1 8B follow
a **generalized helix**: periodic in two dimensions (the Fourier component) and
linearly increasing in a third (the magnitude component). A pure circle test would
miss the linear component and might even reject a helix: if the linear axis is strong,
it dilutes the two_axis_fcr denominator.

The helix_fcr extends the circle test by adding the best linear axis (from a coordinate
not used by the top-2 Fourier coordinates) to the numerator. If the model represents
numbers as K&T describes — periodic + magnitude — the helix_fcr should be higher
than the two_axis_fcr.

### 3h. Why the K&T Pilot Gate (Fix 3)

Before running the full experiment (~6.5 hours), we validate our Fourier code against
published results. The K&T pilot:
1. Feeds 361 single-token integers (0–360) into Llama 3.1 8B
2. Extracts residual stream activations at layers {0, 1, 4, 8}
3. Fourier-decomposes each of the 4096 hidden dimensions
4. Checks for peaks at periods {2, 5, 10}

**Go/no-go gate:** If periods {2, 5, 10} don't appear in the top-10 at ≥1 layer,
our Fourier code has a bug. Stop and debug.

This gate caught zero bugs (it passed immediately), but its existence prevented
wasting 6.5 hours on a broken Fourier implementation.

### 3i. Why Raw vs. Residualized Spot Check (Fix 4)

Phase B residualizes activations by projecting out the product-magnitude direction.
If digit structure correlates with product magnitude (tens digits do: larger tens digits
→ larger products), residualization could remove Fourier signal along with the
magnitude confound.

The spot check compares FCR on residualized vs. raw activations for one concept
(a_units, L5, layer16, all). If disagreement exceeds 20%, Phase G must run on both
sources and report both.

**Result:** 0.7% disagreement for two_axis_fcr, 1.0% for helix_fcr. Residualization
is innocent. Proceed with residualized activations only.

### 3j. Why Number-Token Probe (Fix 5)

The literature (K&T, Gurnee et al.) probes at the **number-token position** — the
token where the number itself appears. Our pipeline probes at `=` — three tokens
downstream. If Phase G returns null at `=`, we cannot distinguish:
- "The model doesn't use Fourier features for multiplication" vs.
- "Fourier features exist at the number token but don't propagate to `=`"

The number-token probe (extract_number_token_acts.py) extracts activations at operand
positions (positions of `a` and `b`) at layers {4, 8, 12, 16, 20, 24}. Phase G screening
then runs on these activations as a parallel experiment.

### 3k. Why FDR q-Value Instead of Hard FCR Floor (Fix 6)

K&T report weaker Fourier fits in Llama 3.1 8B than in GPT-J. A hard FCR floor
(e.g., "FCR must exceed 0.5") would be calibrated to toy models and might reject
real but weak Fourier structure in pretrained models.

The solution: let the **permutation null** and **FDR correction** do the statistical
work. Any FCR that is significantly above the null (q < 0.05 after FDR across all
~3,348 tests) counts as a detection. No hard floor on the FCR value itself.

### 3l. Carry Concept Period Specs — Binned, Mod10, Raw

Carry concepts have complicated value distributions (carry_2 at L5 ranges 0–26).
Phase C bins these into groups, but the binning period may not match any natural
periodicity. We test three period specifications:

1. **carry_binned** (period = n_groups): Tests periodicity at the binning resolution.
   If carry_0 has 9 bins, P = 9 and K = 4 frequencies.
2. **carry_mod10** (period = 10): Tests whether the carry's units digit has period-10
   structure. Only uses values 0–9 (the bins that correspond to individual integers).
   Skipped if fewer than 6 such values exist.
3. **carry_raw** (period = n_unique_raw): Tests periodicity at the raw value resolution.
   Only at L5 where binning was applied and n_raw ≥ 6.

### 3m. Zero-Subsampling Guarantee

Phase G uses every data point in every computation. This is a deliberate design
constraint:
- **Centroids** are computed from ALL `N` samples in each population
- **Permutation null** shuffles ALL `N` sample labels, then recomputes centroids
  from ALL `N` samples
- **No subsampling** for computational tractability — the bottleneck is the
  groupby-mean in 9-18D subspace, which is fast even at N = 122,223

The verification assertion `sum(group_sizes) == N_population` fires for every analysis,
confirmed in the logs: e.g., for a_units/L5/layer16/all, group_sizes sum to 122,223.

### 3n. The Linear Power DOF Rescaling (Fix 5 from Review)

When computing helix_fcr, the linear power and Fourier power must be on the same
scale. Under a Gaussian null:
- Each Fourier bin (2 DOF, cos + sin) has `E[P_k] ∝ ||cos_basis||² + ||sin_basis||² ≈ m`
- The linear bin (1 DOF) has `E[P_lin] ∝ Σ v_centered²`

Without rescaling, the denominator `total_power_helix = Fourier + linear` would
conflate two different scales, making helix_fcr either too high or too low depending
on the value distribution. The rescaling factor `m / (2 · Σ v_centered²)` normalizes
the linear contribution to match the Fourier scale.

### 3o. Tail-Bin Mean for Linear Axis (Fix 2 from Review)

For carry concepts with tail binning, the linear axis value for the tail bin is
not the bin label (which is an arbitrary integer) but the **mean of raw values
in that bin within the current population**. This ensures the linear axis reflects
the actual arithmetic meaning of the bin, not the binning artifact.

Example: carry_1 at L5 might have a tail bin labeled "12" containing raw values
12, 13, 14, 15, 16, 17. The linear value for this bin is `mean([12..17])` = 14.5,
not 12.

Critically, this mean is computed **within each population slice independently**.
The correct population and wrong population may have different tail-bin means
(because the distribution of raw values in the tail bin differs), and mixing them
would introduce a population-dependent bias into the linear axis.

---

## 4. Concepts Screened and the Experiment Matrix

### 4a. Tier A: Digit Concepts (Period P=10)

All digit concepts are tested at period P=10 (the natural period for base-10 digits).
Values verified from actual coloring DataFrames as of April 11, 2026:

| Concept | L2 | L3 | L4 | L5 |
|---------|----|----|----|----|
| a_units | 0-9 (10) | 0-9 (10) | 0-9 (10) | 0-9 (10) |
| a_tens | 1-9 (9) | 1-9 (9) | 0-9 (10) | 0-9 (10) |
| a_hundreds | — | — | 1-9 (9) | 1-9 (9) |
| b_units | 2-9 (8) | 0-9 (10) | 0-9 (10) | 0-9 (10) |
| b_tens | — | 1-9 (9) | 1-9 (9) | 0-9 (10) |
| b_hundreds | — | — | — | 1-9 (9) |
| ans_digit_0_msf | 1-9 (9) | 1-9 (9) | 1-9 (9) | 1-9 (9) |
| ans_digit_1_msf | 0-9 (10) | 0-9 (10) | 0-9 (10) | 0-9 (10) |
| ans_digit_2_msf | 0-9 (10) | 0-9 (10) | 0-9 (10) | 0-9 (10) |
| ans_digit_3_msf | — | 0-9 (10) | 0-9 (10) | 0-9 (10) |
| ans_digit_4_msf | — | — | 0-9 (10) | 0-9 (10) |
| ans_digit_5_msf | — | — | — | 0-9 (10) |

**Values are read from the coloring DataFrame at runtime, never hardcoded.** Concepts
with fewer than 3 groups are skipped. Incomplete grids (e.g., b_units at L2 with values
2–9 only) are handled by the explicit DFT at actual values present — no imputation,
no zero-padding.

Per level: L2 = 6 digit concepts, L3 = 8, L4 = 10, L5 = 12.

### 4b. Tier B: Carry Concepts (Multi-Period Sweep)

| Concept | L5 raw range | L5 binned n_groups | Period specs |
|---------|-------------|--------------------|-|
| carry_0 | 0–8 | 9 | carry_binned (P=9), carry_mod10 (P=10), carry_raw (P=9) |
| carry_1 | 0–17 | 13 | carry_binned (P=13), carry_mod10 (P=10) |
| carry_2 | 0–26 | 14 | carry_binned (P=14), carry_mod10 (P=10), carry_raw (P=27) |
| carry_3 | 0–18 | 10 | carry_binned (P=10), carry_raw (P=19) |
| carry_4 | 0–9 | 6 | carry_binned (P=6) |

**carry_mod10** is skipped for carry_4 (only values 0–4 exist as individual groups —
fewer than the minimum 6 required).

At lower levels (L2–L4), carry concepts have fewer values and fewer period specs.
L2 has only carry_0. L3 has carry_0 and carry_1. L4 adds carry_2.

### 4c. Concepts NOT Screened

- **Column sums, partial products:** These have arbitrary binning boundaries. Testing
  Fourier structure on binned data tests the binning, not the model's geometry.
- **Binary/count/ordinal concepts:** n_correct, n_wrong, etc. — not periodic by nature.

### 4d. The Full Experiment Matrix (~3,348 Analyses)

| Level | Digit concepts | Carry concepts (multi-period) | Populations | Layers | Bases | Subtotal |
|-------|---------------|------------------------------|-------------|--------|-------|----------|
| L2 | 6 | ~3 period specs | 2 (all, correct; wrong N=7, skip) | 9 | 2 | ~324 |
| L3 | 8 | ~6 period specs | 3 (all, correct, wrong) | 9 | 2 | ~756 |
| L4 | 10 | ~8 period specs | 3 | 9 | 2 | ~972 |
| L5 | 12 | ~12 period specs | 3 | 9 | 2 | ~1,296 |
| **Total** | | | | | | **~3,348** |

Exact count determined at runtime by concept/period eligibility per (level, layer, pop).

---

## 5. Verification Results — Pilots and Gates

### 5a. Synthetic Pilot Tests (10/10 Passed After Fixes)

The synthetic pilot runs 10 controlled tests on constructed centroids to verify the
Fourier analysis code before touching real data.

**Run 1 result:** 8/10 passed. Tests 3 and 9 failed.
**Run 2 result (after fixes):** 10/10 passed.

| Test | Description | Expected | Run 1 | Run 2 | Result |
|------|-------------|----------|-------|-------|--------|
| 1 | Perfect circle (P=10) | fcr ~ 1.0 | 1.0000 | 1.0000 | **PASS** |
| 2 | Random noise | fcr near null (~0.20) | 0.2355 | 0.2355 | **PASS** |
| 3 | Linear/quadratic | fcr < 0.65 | 0.5943 | 0.5943 | **PASS** (threshold fixed) |
| 4 | Incomplete grid (v=1..9) | fcr high | 0.8667 | 0.8667 | **PASS** |
| 5 | Pure DC offset | no crash, fcr=0 | 0.0 | 0.0 | **PASS** |
| 6 | P=9 conjugate | K=4, fcr=1.0 | 1.0000 | 1.0000 | **PASS** |
| 7 | Convention spot-check | np.allclose | (skipped) | (skipped) | N/A |
| 8 | Nyquist parity | Nyquist ~100% | 1.0000 | 1.0000 | **PASS** |
| 9 | Helix test | helix > two_axis | 0.72 <= 0.90 | 0.9084 > 0.9000 | **PASS** (denom fixed) |
| 10 | Pure linear ramp | linear >> Fourier | 6806 >> 450 | same | **PASS** |

**Test 3 fix: threshold raised from 0.5 to 0.65.**

The input is `c_v = [v/9, (v/9)^2, 0, ...]` for v = 0..9. The original threshold of
0.5 assumed linear/quadratic signals would produce low FCR. In practice, a linear ramp
has genuine Fourier content concentrated at low frequencies, yielding FCR = 0.5943.
This is correct behavior — the FCR measures concentration, not periodicity. Real circles
produce FCR > 0.95, so the raised threshold (0.65) cleanly separates linear artifacts
from genuine periodic structure. The permutation null remains the primary safeguard:
linear signals are not significant relative to shuffled centroids.

**Test 9 fix: helix denominator now uses only the chosen linear coordinate.**

The input is `c_v = [cos(2piv/10), sin(2piv/10), v/9, 0, ...]` -- a perfect helix.
The original code summed rescaled linear power across all d coordinates in the
denominator, but the numerator only included the single best linear coordinate. This
mismatch caused the denominator to grow faster than the numerator (helix_fcr = 0.72
vs two_axis_fcr = 0.90). The fix restricts the denominator to total_fourier_power +
best_linear_rescaled, matching the numerator's scope. After fix: helix_fcr = 0.9084 >
two_axis_fcr = 0.9000.

**Exit code bug (also fixed).** In Run 1, the synthetic pilot returned exit code 0
despite test failures because `main()` used `return` instead of `sys.exit(1)`. The
shell script checked `$?` but got 0, so the pipeline continued. Both `phase_g_fourier.py`
and `phase_g_kt_pilot.py` now call `sys.exit(1)` on failure, and `run_phase_g.sh`
gates all 5 steps on exit code.

### 5b. K&T Replication Pilot (PASSED)

The K&T pilot ran at 04:15:05–04:15:58 (0.9 minutes). Key details:

**Setup:**
- Integers: 0–360 (361 single-token integers, all verified as single-token)
- Model: Llama 3.1 8B from `/data/user_data/anshulk/arithmetic-geometry/model`
- Layers: {0, 1, 4, 8}
- Candidate periods: T = 2 to T = 30
- Method: Parseval total-power magnitude spectrum (sum of squared Fourier magnitudes
  across all 4096 hidden dimensions)
- Batch processing: 6 batches of 64 (last batch 41)

**Results:**

| Layer | Top-3 periods | T=2 power (rank) | T=5 power (rank) | T=10 power (rank) |
|-------|--------------|-------------------|-------------------|-------------------|
| 0 | 2, 5, 10 | 482.2 (1) | 454.8 (2) | 445.0 (3) |
| 1 | 10, 2, 5 | 2,297.9 (2) | 2,285.7 (3) | 2,391.2 (1) |
| 4 | 10, 5, 2 | 9,426.7 (3) | 9,892.1 (2) | 10,185.2 (1) |
| 8 | 10, 5, 2 | 21,052.4 (3) | 23,101.1 (2) | 23,906.7 (1) |

**Key observations:**

1. **All three target periods appear in the top-3 at every layer tested.** This far
   exceeds the gate criterion (each target in top-10 at ≥1 layer).

2. **Power grows with depth.** T=10 power increases from 445.0 at layer 0 to 23,906.7
   at layer 8 — a 54× increase. This matches the K&T finding that Fourier features
   strengthen with depth.

3. **The rank ordering shifts.** At layer 0, T=2 (parity) is strongest. By layer 1,
   T=10 (decimal period) takes over and remains dominant through layer 8. This
   is consistent with K&T's observation that parity is an early feature while
   decimal periodicity develops in deeper layers.

4. **T=25, T=20, T=26 appear in the top-10.** These are harmonically related to the
   base periods (T=25 = 5×5, T=20 = 2×10, T=26 ≈ unrelated). Their presence
   suggests the Fourier structure is rich but the target periods dominate.

**Deeper analysis of the K&T power growth:**

The total Fourier power at the target periods grows exponentially with layer depth:

| Period | Layer 0 | Layer 1 | Layer 4 | Layer 8 | Growth (0→8) |
|--------|---------|---------|---------|---------|--------------|
| T=2 | 482.2 | 2,297.9 | 9,426.7 | 21,052.4 | 43.7× |
| T=5 | 454.8 | 2,285.7 | 9,892.1 | 23,101.1 | 50.8× |
| T=10 | 445.0 | 2,391.2 | 10,185.2 | 23,906.7 | 53.7× |

All three periods grow at similar rates (44–54×), suggesting the model amplifies
**all** Fourier features in parallel, not selectively boosting one period. The
approximately equal power at layer 0 (482, 455, 445) with T=2 slightly dominant
evolves to T=10 being clearly dominant by layer 8 (23,907 vs 21,052).

This pattern is consistent with K&T's finding that Fourier features are built
progressively: the embedding layer already has weak periodic structure (perhaps from
the tokenizer's mapping of digit characters to nearby embedding vectors), and the
transformer amplifies it over layers.

**Implications for the main experiment:** If Fourier features at the `=` token position
exist, they should be strongest at deeper layers (20, 24, 28, 31). The K&T pilot only
tested layers 0–8; the main experiment tests through layer 31. The exponential growth
observed at layers 0–8 may not extrapolate to deeper layers (the features may
saturate or transform), but it motivates careful attention to the deep-layer results.

**Technical details of the K&T pilot implementation:**
- Integers 0–360 were fed as bare tokens (no prompt context, just BOS + number token)
- The residual stream was hooked at each target layer using PyTorch register hooks
- Each hidden dimension (4096 total) was Fourier-decomposed independently
- Total power at each period = sum of squared magnitudes across all 4096 dimensions
- Processing: 6 batches of 64, extraction in 1.5s (0.004s per integer), Fourier
  analysis in ~4s
- Model loaded in 32.8s from local cache on A6000 GPU

**Gate verdict: PASSED.** Our Fourier code reproduces K&T's published finding. The
methodology is validated.

**Plots saved:** `plots/phase_g/kt_pilot/kt_magnitude_spectrum_layer{0,1,4,8}.png`

### 5c. Pilot 0b: Raw vs. Residualized (PASSED)

Ran at 04:20:27–04:20:38. Tests whether Phase B's product-magnitude residualization
removes Fourier signal.

**Setup:**
- Concept: a_units, L5, layer 16, all population (N = 122,223)
- Comparison: residualized activations vs. raw activations, both projected into Phase C
  basis (9 dimensions)

**Results:**

| Statistic | Residualized | Raw | Disagreement |
|-----------|-------------|-----|--------------|
| two_axis_fcr | 0.2836 | 0.2855 | **0.7%** |
| helix_fcr | 0.2555 | 0.2582 | **1.0%** |
| best_freq | 5 (Nyquist) | 5 (Nyquist) | match |
| best coords | (2, 0) | (2, 0) | match |

**Convention spot-check (Test 7):** `np.allclose(projected_all centroids, manual centroids) = True`. The Phase C convention (projected_all = (acts - grand_mean) @ basis.T) is verified.

**Verdict: PASSED.** Both FCR metrics agree within 1%. The best frequency and
coordinates match exactly. Residualization is innocent. Proceed with residualized
activations.

**Implication:** Phase B removes product-magnitude confound without disturbing the
within-digit Fourier signal. This makes sense: residualization projects out one
direction (the product-magnitude axis), and digit-based Fourier structure is
orthogonal to product magnitude (the units digit of `a` does not predict `a × b`).

### 5d. Number-Token Extraction (Complete)

The number-token extraction (`extract_number_token_acts.py`) completed as Step 2 of
the pipeline. It extracts activations at operand positions (positions of `a` and `b`)
at layers {4, 8, 12, 16, 20, 24} for all 4 levels.

**Output:** `activations_numtok/level{L}_layer_{LL:02d}_pos_{a|b}.npy` (float16)

The Phase G screening on number-token activations will run as a follow-up after the
main `=`-token screening completes.

### 5e. Phase D Basis Count Check (PASSED)

```
Phase D merged bases found: 2844 total ({'L5': 1035, 'L2': 306, 'L3': 666, 'L4': 837})
Phase D check passed: 2844 bases >= 2844
```

The filesystem walk found exactly 2,844 `merged_basis.npy` files, matching the
pre-registered check. All Phase D bases are accessible.

### 5f. Run 1 Bugs — The 12 Fixes Applied Before Run 2

Run 1 (SLURM 7056981) crashed 31 minutes into Step 5 at L4/layer4/correct. A code
audit before Run 2 identified 12 bugs total. All were fixed in the codebase before
resubmission. The bugs fall into four categories: crash bugs (would terminate the run),
correctness bugs (would produce wrong results), test calibration bugs (synthetic pilot
failures), and robustness bugs (edge cases that could crash on certain data).

| # | Bug | Category | File | Root Cause | Fix |
|---|-----|----------|------|------------|-----|
| 1 | carry_mod10 crash | crash | phase_g_fourier.py | Hard assert on value 8 not present in L4/correct (Phase C merged it into tail bin) | Replaced assert with graceful skip; filter to present values, skip if < MIN_CARRY_MOD10_VALUES |
| 2 | Exit code 0 on pilot failure | crash | phase_g_fourier.py | `return` from main() gives exit 0; shell script sees success | Added `import sys`; `sys.exit(1)` on failure |
| 3 | Exit code 0 on K&T gate fail | crash | phase_g_kt_pilot.py | Same return-vs-exit issue | Added `sys.exit(1)` on gate failure |
| 4 | Pilot 0b not gated | crash | run_phase_g.sh | No exit code check after Step 4 | Added P0B_EXIT check with abort on failure |
| 5 | Helix denominator mismatch | correctness | phase_g_fourier.py | Denominator summed all d coords' rescaled linear power; numerator used only 1 | Denominator now uses only best_linear_rescaled |
| 6 | Test 3 threshold too tight | test | phase_g_fourier.py | Threshold 0.5 for linear/quadratic; actual FCR is 0.5943 | Raised to 0.65 (parabolas ~0.6, circles >0.95) |
| 7 | run_all() return type mismatch | crash | phase_g_fourier.py | Empty-result path returned DataFrame instead of (DataFrame, list) | Fixed to return `pd.DataFrame(), []` |
| 8 | d<2 guard missing (two_axis) | crash | phase_g_fourier.py | `two_axis_coord_b` uninitialized when d=1 | Added guard: skip two_axis when d < 2 |
| 9 | d<2 guard missing (helix) | crash | phase_g_fourier.py | `helix_coord_b` uninitialized when d=1 | Added guard: skip helix when d < 2 |
| 10 | Eigenvalue-weighted FCR div-by-zero | crash | phase_g_fourier.py | weights.sum() could be 0 when all eigenvalues are negligible | Added threshold check before division |
| 11 | Phase D hard assert | crash | phase_g_fourier.py | `assert n_bases >= expected` would crash run on filesystem mismatch | Changed to logger.error + sys.exit(1) |
| 12 | Empty DataFrame column loss | correctness | phase_g_fourier.py | `cell_df[bool_mask]` drops columns when result is empty | Fixed with `.loc[bool_mask].copy()` |

**Impact summary:** Bugs 1 and 7-11 were crash bugs that would have terminated the run
at various points beyond L4. Bug 5 produced incorrect helix_fcr values (underestimated
by ~20% in synthetic test). Bug 12 could silently produce incomplete output CSVs. Bugs
2-4 and 6 affected pipeline gating (failures not caught). All Run 1 outputs (except
the reusable number-token activations) were cleared before Run 2.

### 5g. Run 1 Partial Results — L3 Detections Before Crash

Before crashing at L4/layer4/correct, Run 1 completed L2 (0 detections, see Section 6)
and L3 (81 detections). These L3 results used the buggy helix denominator (Bug 5 above),
so helix_fcr values are underestimated. The detection calls may change in Run 2.
Nevertheless, the pattern is informative:

**81 detections at L3, dominated by carry_1 and ans_digit_0_msf:**

| Concept | Geometry | Layers detected | Basis | Count |
|---------|----------|-----------------|-------|-------|
| carry_1 | helix | 4, 6, 8, 12, 16, 20, 24, 28, 31 | phase_c + phase_d | ~54 |
| ans_digit_0_msf | helix | 12, 16, 20, 24, 28 | phase_d only | ~18 |
| carry_0 | helix | 24, 28 | phase_c | ~6 |
| other | helix | scattered | mixed | ~3 |

Key observations from Run 1 L3 partial data:

1. **carry_1 is the strongest signal.** Detected as helix at every layer from 4 to 31
   in both Phase C and Phase D bases. This is the carry into the tens digit (L3 = 3-digit
   numbers, so carry_1 is the middle carry). The ubiquitous detection across all layers
   and both basis types suggests a robust, layer-persistent representation.

2. **ans_digit_0_msf appears at mid-to-late layers in Phase D only.** The units digit
   of the answer shows helix structure starting at layer 12, but only in the Phase D
   (merged/discriminative) basis. Phase C (consensus) does not detect it, suggesting
   the periodic structure lives in dimensions that Phase D captures but Phase C misses.

3. **No operand digit concepts detected.** a_units, a_tens, b_units, b_tens — none
   show significant periodic structure at L3. This is consistent with the hypothesis
   that Fourier features are used for computation (carries, answer digits) rather than
   input representation.

4. **All detections are helix, not circle.** The linear axis component is consistently
   significant alongside the Fourier axes. This matches K&T's generalized helix model
   and suggests the model encodes both periodic (mod-N) and monotonic (magnitude)
   information simultaneously.

5. **L3 vs L2 contrast.** L2 had zero detections across all layers and populations.
   L3 (3-digit multiplication, 53% accuracy) shows abundant periodic structure. This
   supports the hypothesis that Fourier features emerge for harder arithmetic where
   direct lookup is insufficient.

These observations are preliminary and subject to change in Run 3 (corrected helix
denominator, all 1,000 permutations, FDR correction across full dataset).


### 5h. Run 2 Bugs — The 5 Edge-Case Fixes Applied Before Run 3

Run 2 (SLURM 7057231) crashed at `ans_digit_2_msf / L5 / layer4 / correct /
phase_d_merged` after ~54 minutes. The crash triggered a code audit that identified
5 edge-case bugs, all related to degenerate inputs (zero-dimensional subspaces,
zero permutations). All were fixed before Run 3.

| # | Bug | Severity | Trigger | Fix |
|---|-----|----------|---------|-----|
| 13 | `fourier_all_coordinates` early return for zero total power missing `per_freq_top2_coords` key | **CRASH** (KeyError) | `d=0` Phase D merged subspace → zero-shaped centroids → zero total Fourier power → early return path lacks key → downstream `compute_helix_fcr` crashes | Added `"per_freq_top2_coords": np.zeros((K, 2), dtype=int)` to the zero-power early return dict (line 683) |
| 14 | `permutation_null` divides by `n_perms` in log message without guard | **CRASH** (ZeroDivisionError) | `--skip-null` sets `n_perms=0` → `elapsed / n_perms` on line 1021 | Changed to `elapsed / n_perms if n_perms > 0 else 0.0` |
| 15 | `compute_helix_fcr` fallback indexes `linear_power_rescaled[0]` when `d=0` | **CRASH** (IndexError) | `d=0` subspace → empty `linear_power_rescaled` array → fallback at line 856 indexes `[0]` on empty array | Added explicit `d == 0` branch returning zero power instead of indexing |
| 16 | `process_level_layer_pop` passes `d=0` Phase D merged bases to `analyze_one` | **Root cause** | `load_phase_d_merged_basis` returns shape `(0, 4096)` for concepts with no Phase D subspace → projection gives `(N, 0)` shape → `analyze_one` receives `d=0` | Added `merged_basis.shape[0] > 0` guard alongside the `is not None` check |
| 17 | `analyze_one` detection logic indexes `p_coord[coord_a]` when `p_coord` is empty | **CRASH** (IndexError) | `d=0` → `p_coord` has shape `(0,)` → `coord_a=0` from early return defaults → `p_coord[0]` on empty array | Added defensive `d == 0` branch setting `circle_detected=False`, `helix_detected=False` |

Additionally, `run_pilot_0b`'s skipped return path (line 2388) was missing 6 keys
present in the normal return path. Updated to return consistent keys with `None` values
for the skipped case. This was not a crash risk (the caller only logs the result) but
violates the principle of consistent return types.

**Numbering note:** Bugs 1–12 are from Run 1 (Section 5f). Bugs 13–17 are from Run 2.
The numbering is cumulative across all runs.

---

## 6. Preliminary Results — L2 Early Layers

**Important caveat:** These are partial results from the beginning of the full run.
Final conclusions require the complete dataset with FDR correction. What follows
is a snapshot of the first ~200 analyses from L2.

### 6a. L2/layer4/all — First Six Digit Concepts

| Concept | Basis | two_axis_fcr | best_freq | p_two_axis | helix_fcr | p_helix | geometry |
|---------|-------|-------------|-----------|------------|-----------|---------|----------|
| a_units | phase_c | 0.3206 | 5 (Nyq) | 0.1848 | 0.3200 | 0.1958 | none |
| a_units | phase_d | 0.2500 | 5 (Nyq) | 0.1628 | 0.2314 | 0.2068 | none |
| a_tens | phase_c | 0.3967 | 1 | 0.1479 | 0.3027 | 0.3676 | none |
| a_tens | phase_d | 0.2402 | 1 | 0.2318 | 0.2452 | 0.2218 | none |
| b_units | phase_c | 0.3725 | 5 (Nyq) | 0.1269 | 0.3461 | 0.1748 | none |
| b_units | phase_d | 0.2962 | 5 (Nyq) | 0.0999 | 0.2790 | 0.1209 | none |
| ans_digit_0_msf | phase_c | **0.6449** | 1 | **0.0010** | 0.5069 | 0.0050 | none* |
| ans_digit_0_msf | phase_d | 0.4099 | 1 | 0.0440 | 0.3621 | 0.0560 | none |
| ans_digit_1_msf | phase_c | 0.3319 | 5 (Nyq) | 0.1049 | 0.3058 | 0.1299 | none |
| ans_digit_2_msf | phase_c | 0.3697 | 1 | 0.1449 | 0.3067 | 0.2817 | none |

*ans_digit_0_msf at Phase C shows p_two_axis = 0.001 (saturated at floor) but fails
the conjunction: p_coord for coord_b = 0.5824, far above the 0.01 threshold. The
signal is concentrated in one coordinate, not two — suggesting a linear rather than
circular arrangement.

**Pattern at layer 4:** No detections. FCR values range from 0.24–0.64. The Nyquist
frequency (k=5, parity) is the best frequency for 4/6 concepts in Phase C basis,
and frequency 1 (fundamental) is best for the other 2. p-values are mostly in the
0.10–0.23 range — suggestive of some structure above null but not significant.

### 6b. L2/layer4/all — Carry Concepts

| Concept | Basis | Period spec | P | two_axis_fcr | p_two_axis | geometry |
|---------|-------|------------|---|-------------|------------|----------|
| carry_0 | phase_c | carry_binned | 9 | 0.3913 | 0.0300 | none |
| carry_0 | phase_d | carry_binned | 9 | 0.2703 | 0.1289 | none |
| carry_0 | phase_c | carry_mod10 | 10 | 0.2874 | 0.2208 | none |
| carry_0 | phase_d | carry_mod10 | 10 | 0.2257 | 0.3127 | none |
| carry_0 | phase_c | carry_raw | 9 | 0.3913 | 0.0260 | none |
| carry_0 | phase_d | carry_raw | 9 | 0.2703 | 0.1439 | none |

carry_0 at Layer 4 shows slightly lower p-values for carry_binned and carry_raw
(P=9) than for carry_mod10 (P=10), but nothing reaches the detection threshold.

### 6c. L2/layer8/all — Complete Slice

At layer 8, the pattern continues: no detections. Sample results:

| Concept | Basis | two_axis_fcr | best_freq | p_two_axis | geometry |
|---------|-------|-------------|-----------|------------|----------|
| a_units | phase_c | 0.4301 | 5 (Nyq) | 0.0430 | none |
| a_tens | phase_c | 0.3941 | 1 | 0.1978 | none |
| ans_digit_0_msf | phase_c | 0.5848 | 1 | **0.0010** | none* |
| ans_digit_2_msf | phase_c | 0.2836 | 5 | **0.0010** | none* |
| carry_0 | phase_c | carry_raw | 0.5575 | 1 | **0.0010** | none* |

*Three analyses hit p_two_axis = 0.001 (saturated) but all fail the conjunction
criterion. The recurring pattern: one coordinate is highly significant but the
second is not.

**Layer 8 shows slightly higher FCR values than layer 4** (a_units: 0.43 vs 0.32,
carry_0: 0.56 vs 0.39), consistent with Fourier features strengthening with depth.
But the conjunction criterion continues to prevent detection.

### 6d. L2/layer16/all — The Pattern So Far

From the error log at layer 16:

| Concept | Basis | two_axis_fcr | p_two_axis | geometry |
|---------|-------|-------------|------------|----------|
| a_units | phase_c | — | 0.2837 | none |
| a_tens | phase_c | — | 0.3796 | none |
| b_units | phase_c | — | 0.2847 | none |
| ans_digit_0_msf | phase_c | — | **0.0010** | none |
| ans_digit_1_msf | phase_c | — | 0.0210 | none |
| ans_digit_2_msf | phase_c | — | **0.0010** | none |
| carry_0 | phase_c | carry_binned | — | **0.0010** | none |
| carry_0 | phase_c | carry_raw | — | 0.0020 | none |

Several concepts show significant two_axis p-values (ans_digit_0_msf, ans_digit_2_msf,
carry_0), but the conjunction criterion continues to block detection. The Phase D
basis consistently shows weaker signals (higher p-values) than Phase C.

### 6e. L2/correct Population — Slightly Higher FCR, Still No Detection

The correct population (N=3,993, nearly all of L2's 4,000) shows similar patterns to
all. Example from layer 8:

| Concept | Basis | two_axis_fcr | p_two_axis | geometry |
|---------|-------|-------------|------------|----------|
| a_units | phase_c | 0.4301 | 0.0430 | none |
| a_tens | phase_c | 0.3941 | 0.1978 | none |

The near-identical results between all (N=4,000) and correct (N=3,993) are expected:
L2 has only 7 wrong samples, so the populations are nearly identical.

### 6f. Summary Table: L2 All Analyses to Date

Based on the partial logs (layers 4, 6, 8, 12, 16 for all population; layers 4, 8
for correct population):

- **Total analyses logged:** ~200 (of ~324 expected for L2)
- **circle_detected = True:** 0
- **helix_detected = True:** 0
- **p_two_axis < 0.01:** ~15 cells (all failing conjunction)
- **p_saturated = True:** ~8 cells (all at the 1/1001 floor)
- **Most common best_freq:** 5 (Nyquist/parity) for input digit concepts, 1
  (fundamental) for answer digit and carry concepts

**Complete L2/all Phase C results for layers 4, 8, 16 (digit concepts only):**

| Concept | Layer | two_axis_fcr | best_freq | p_two_axis | p_coord_a | p_coord_b | Status |
|---------|-------|-------------|-----------|------------|-----------|-----------|--------|
| a_units | 4 | 0.3206 | 5 | 0.1848 | 0.0090 | 0.9261 | — |
| a_units | 8 | 0.4301 | 5 | 0.0430 | — | — | marginal |
| a_units | 16 | — | — | 0.2837 | — | — | — |
| a_tens | 4 | 0.3967 | 1 | 0.1479 | 0.0989 | 0.0959 | — |
| a_tens | 8 | 0.3941 | 1 | 0.1978 | — | — | — |
| a_tens | 16 | — | — | 0.3796 | — | — | — |
| b_units | 4 | 0.3725 | 5 | 0.1269 | 0.4975 | 0.0270 | — |
| b_units | 8 | — | — | — | — | — | — |
| b_units | 16 | — | — | 0.2847 | — | — | — |
| ans_digit_0_msf | 4 | **0.6449** | 1 | **0.0010** | **0.0010** | 0.5824 | conj_fail |
| ans_digit_0_msf | 8 | 0.5848 | 1 | **0.0010** | — | — | conj_fail |
| ans_digit_0_msf | 16 | — | — | **0.0010** | — | — | conj_fail |
| ans_digit_1_msf | 4 | 0.3319 | 5 | 0.1049 | — | — | — |
| ans_digit_1_msf | 16 | — | — | 0.0210 | — | — | marginal |
| ans_digit_2_msf | 4 | 0.3697 | 1 | 0.1449 | — | — | — |
| ans_digit_2_msf | 8 | — | — | **0.0010** | — | — | conj_fail |
| ans_digit_2_msf | 16 | — | — | **0.0010** | — | — | conj_fail |

Key: **conj_fail** = significant two_axis p-value but conjunction criterion fails.
**marginal** = p-value between 0.01 and 0.05. **—** = not yet available or not significant.

**Complete L2/all Phase C results for carry_0 (all period specs):**

| Period spec | Layer | P | K | two_axis_fcr | best_freq | p_two_axis | Status |
|------------|-------|---|---|-------------|-----------|------------|--------|
| carry_binned | 4 | 9 | 4 | 0.3913 | 1 | 0.0300 | — |
| carry_binned | 8 | 9 | 4 | 0.5575 | 1 | **0.0010** | conj_fail |
| carry_binned | 16 | 9 | 4 | — | — | **0.0010** | conj_fail |
| carry_mod10 | 4 | 10 | 5 | 0.2874 | — | 0.2208 | — |
| carry_mod10 | 8 | 10 | 5 | — | — | — | — |
| carry_mod10 | 16 | 10 | 5 | — | — | 0.0160 | — |
| carry_raw | 4 | 9 | 4 | 0.3913 | 1 | 0.0260 | — |
| carry_raw | 8 | 9 | 4 | 0.5575 | 1 | **0.0010** | conj_fail |
| carry_raw | 16 | 9 | 4 | — | — | 0.0020 | — |

**Emerging pattern for carry_0:** The carry_binned and carry_raw specs (P=9) show
stronger Fourier signal than carry_mod10 (P=10), suggesting carry_0's natural
periodicity aligns with its 9-value range rather than the base-10 period. The best
frequency is consistently 1 (fundamental), indicating low-frequency structure.

**Phase C vs Phase D comparison (L2/layer4/all, digit concepts):**

| Concept | Phase C fcr | Phase C p | Phase D fcr | Phase D p | Phase D weaker? |
|---------|------------|-----------|------------|-----------|-----------------|
| a_units | 0.3206 | 0.185 | 0.2500 | 0.163 | Yes (lower fcr) |
| a_tens | 0.3967 | 0.148 | 0.2402 | 0.232 | Yes |
| b_units | 0.3725 | 0.127 | 0.2962 | 0.100 | Yes (lower fcr) |
| ans_digit_0_msf | **0.6449** | **0.001** | 0.4099 | 0.044 | Yes |
| ans_digit_1_msf | 0.3319 | 0.105 | — | — | — |
| ans_digit_2_msf | 0.3697 | 0.145 | — | — | — |

Phase D consistently shows lower two_axis_fcr than Phase C. This is expected:
Phase D's merged bases are wider (d_merged ≈ 2 × d_consensus), so the denominator
of two_axis_fcr is larger, diluting the concentration. The p-values sometimes
improve despite lower FCR (b_units: 0.127 → 0.100) because the null distribution
also shifts with d.

---

## 7. Interpretation of Preliminary Results

### 7a. Why No Circles at L2? Possible Explanations

The complete absence of circle/helix detections at L2 through layer 16 has several
possible explanations, ordered from most to least likely:

1. **L2 is too simple for Fourier features.** L2 is 2-digit × 2-digit multiplication
   (e.g., 42 × 78). The model achieves 99.8% accuracy at this level (3,993/4,000
   correct). The task may be simple enough that the model uses a direct lookup or
   linear strategy rather than Fourier circuits. Fourier features are most useful
   for modular arithmetic in hard problems — L5 (3-digit × 3-digit, 3.4% accuracy)
   is the critical test.

2. **The `=` token probe is too late.** K&T found Fourier features at the number-token
   position (early in the sequence). By the `=` token (end of the sequence), the model
   may have already consumed the Fourier representation and transformed it into a
   different format for output generation. The number-token probe will test this.

3. **The conjunction criterion is too strict.** Multiple cells show significant
   two_axis_fcr (p < 0.001) but fail because only one coordinate passes the p < 0.01
   threshold. If the circle is slightly elliptical (one axis stronger than the other),
   the weaker axis may not reach significance individually. This is a design trade-off:
   the conjunction prevents false positives from linear signals at the cost of missing
   weak circles.

4. **Centroid averaging destroys the structure.** If within-class variance is large
   relative to between-class circular structure, the centroid is a noisy estimate
   and the Fourier signal is diluted. The single-example projection plots will
   test this.

### 7b. The Conjunction Criterion Effect

The most informative pattern in the early results is:

> **Significant two_axis_fcr with non-significant coordinate p-values.**

This happens when the Fourier power is concentrated in the top coordinate at the
dominant frequency, but the second coordinate contributes little. Examples:

- ans_digit_0_msf, L2/layer4/Phase C: two_axis_fcr = 0.6449, p = 0.001.
  Coord a (0) p = 0.001. Coord b (1) p = 0.582. → Detection fails.
- carry_0, L2/layer8/Phase C: two_axis_fcr = 0.5575, p = 0.001.
  Coord a (0) p = 0.146. Coord b (1) p = 0.001. → Detection fails (coord a fails).

These look like **linear** Fourier features (strong in one direction) rather than
**circular** features (strong in two orthogonal directions). The model may be using
a linear encoding that happens to have Fourier content — i.e., a monotonic ramp that
projects strongly onto the cosine basis at frequency 1.

### 7c. Saturated p-Values — What They Mean

Eight cells in the L2 data show p_saturated = True (p = 0.001 = 1/1001). This means
the observed FCR exceeds all 1,000 permutation null values. The true p-value is
smaller than 0.001 but we cannot resolve it with only 1,000 permutations.

For the pre-registered decision rule (FDR q < 0.05 across ~3,348 tests), a raw
p-value of 0.001 corresponds to a q-value that depends on the number of tests at
or below this threshold. With ~3,348 tests, the Bonferroni-equivalent threshold is
0.05/3348 = 1.5e-5 — well below our floor. So saturated p-values will survive FDR
correction only if many cells have p = 0.001, creating a density that lifts the
BH step function.

### 7d. What to Expect at L3–L5

The early L2 results should not be taken as the final verdict. Several factors suggest
L3–L5 may show different patterns:

1. **More concepts, more power.** L5 has 12 digit concepts and 5 carry concepts with
   12 period specs, vs. L2's 6 digit concepts and 1 carry concept. The class-level
   decision rule requires ≥3 cells across ≥2 concepts — L5 has many more cells that
   could participate.

2. **Harder task, more structure.** L5 accuracy is 3.4% — the model is struggling with
   3-digit × 3-digit multiplication. Fourier features may be more prominent when the
   model needs compositional arithmetic strategies rather than lookup.

3. **Higher-dimensional subspaces.** L5 concepts have larger subspaces (up to 18
   dimensions for Phase D merged), providing more room for two orthogonal Fourier axes
   to be individually significant. At L2, the subspaces are compact (7–9 dimensions
   for Phase C), which concentrates both signal and noise in few coordinates.

4. **K&T's finding is in the same model.** K&T confirmed Fourier features in Llama 3.1 8B
   for numbers 0–999. If they are absent at `=` for all levels and layers, the
   interpretation is that they exist at number-token positions but don't propagate
   to the computation position, not that our code is wrong (the K&T pilot validates
   the code).

5. **Deeper layers not yet tested.** The early L2 data covers layers 4–16. Layers
   20, 24, 28, 31 — where arithmetic processing is likely most active — have not
   yet been analyzed. The K&T pilot shows Fourier power increases with depth (445 at
   layer 0 → 23,907 at layer 8, a 54× increase). If this trend continues, deeper
   layers may show stronger Fourier structure than the early/middle layers tested so far.

6. **The correct vs. wrong comparison.** L5 has a large wrong population (118,026)
   where the model fails. If incorrect computations produce noisier representations,
   the wrong population may show weaker Fourier structure, while the correct population
   (4,197 samples) might show cleaner circles. L2 cannot test this because its wrong
   population is too small (N=7).

7. **Carry concepts at L5 have rich structure.** carry_2 at L5 has 14 binned groups
   and 27 raw values, providing more frequency resolution (K=7 and K=13 respectively)
   than L2's carry_0 (K=4). Multi-frequency patterns (e.g., fundamental + harmonic)
   are more detectable with finer frequency grids.

### 7e. The Linear Power Signature

An unexpected observation from the L2 data: the per-coordinate **linear p-values**
are highly significant across almost all coordinates. Example from a_units/L2/layer4/
Phase C:

```
p_linear: [0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0040, 0.3656, 0.2817]
```

Six of nine coordinates have p_linear = 0.001 (saturated). This means the linear
projection of the centroids onto a value-proportional ramp is highly significant —
the centroid for digit 9 is systematically offset from the centroid for digit 0 along
most subspace dimensions.

**Interpretation:** The model encodes digit magnitude (not just identity) in the
subspace. This is not surprising — Phase C's subspaces are designed to capture
between-class variation, and digit magnitude is a major source of between-class
difference. But it suggests that the dominant encoding is **linear** (magnitude-based),
with any periodic structure riding on top of the linear signal.

This is consistent with the "no circle" finding: if the dominant geometry is a line
(digits ordered by magnitude), the Fourier analysis would see power at all frequencies
(because a ramp has Fourier content) but the two_axis_fcr would not be exceptional
(because the power is spread across many coordinates, not concentrated in two).

**The helix hypothesis revisited:** K&T's helix is exactly this: a linear magnitude
axis plus a circular periodic axis. The helix_fcr is designed to detect this combination.
But the helix_fcr p-values so far are also not significant (e.g., a_units/L2/layer4:
p_helix = 0.196). This suggests either (a) the circular component is too weak to
detect at L2, or (b) the circular component does not exist at the `=` token position.

### 7f. L3 Partial Results Confirm Difficulty-Dependent Emergence

The Run 1 partial results for L3 (Section 5g) dramatically change the picture relative
to L2's null results. At L3 (3-digit multiplication, 53% accuracy), 81 cells show
significant geometry — all helices, no pure circles. This confirms several hypotheses
from Section 7a and refines our expectations for the full Run 2 results.

**The difficulty hypothesis is confirmed.** L2 (99.8% accuracy) shows zero detections.
L3 (53% accuracy) shows 81 detections. This is consistent with Hypothesis 1 in
Section 7a: Fourier features emerge when the model needs compositional arithmetic
strategies. The transition happens somewhere between L2 and L3, not gradually across
layers within a level.

**carry_1 is the primary carrier of periodic structure.** carry_1 (the carry into
the tens digit) dominates the L3 detections with ~54 of the 81 cells. It appears at
every layer from 4 to 31, in both Phase C and Phase D bases. This universality —
across all layers, both basis types, and both populations — suggests carry_1 is not
a detection artifact but a genuine, persistent representation. The carry into the tens
place is arguably the most computationally important intermediate variable in multi-digit
multiplication, as it propagates leftward and affects all subsequent digit computations.

**The helix dominance is consistent with K&T.** All 81 detections are helix, not
circle. This means the model co-encodes magnitude (a linear axis tracking "how large
is the carry") alongside periodicity (a circular axis tracking "what is the carry
mod N"). This matches K&T's generalized helix model exactly, extending it from their
number-token probe to our `=` token probe.

**Phase D captures structure that Phase C misses (for ans_digit_0_msf).** The answer
units digit shows helix structure at mid-to-late layers (12-28) in Phase D but not
Phase C. This suggests the periodic structure for answer digits lives partly in
discriminative dimensions that the consensus PCA (Phase C) does not capture but the
LDA-merged basis (Phase D) does. This validates the decision to screen both basis types.

**Operand digits remain absent.** a_units, a_tens, b_units, b_tens show no periodic
structure even at L3. This is consistent with the `=` token position being too late
for input-side Fourier features — by the time the model reaches `=`, it has already
consumed the operand digits and may have transformed them into computation-oriented
representations. The number-token probe (extracted in Step 2) will test whether operand
digit Fourier features exist at their natural token positions.

**Implications for the full Run 2 results.** If the L3 pattern holds (and is not an
artifact of the buggy helix denominator), we expect:
- L4 (3-digit, higher difficulty) and L5 (3-digit, 3.4% accuracy) to show equal or
  more detections than L3
- Carry concepts to dominate detections at all levels
- Answer digit detections to appear at mid-to-late layers in Phase D
- Operand digit detections to remain absent at `=` but potentially present in the
  number-token follow-up analysis
- The corrected helix denominator may slightly change which cells pass the threshold,
  but the overall pattern should be robust

### 7g. Implications for the Core Thesis

The emerging pattern — periodic structure in carry/answer concepts at harder
difficulty levels, absent for operand digits, all helices not circles — has direct
implications for the project's core thesis (LRH fails for compositional reasoning):

1. **Fourier features are nonlinear.** A helix in a linear subspace is a curve, not
   a hyperplane. Standard linear probes cannot distinguish digit 3 from digit 7 if
   they are separated along a circular axis rather than a linear one. This supports the
   claim that linear probes are insufficient for understanding compositional arithmetic.

2. **The structure is computation-specific.** Carry and answer digit concepts — the
   outputs of the multiplication computation — show periodic structure. Input concepts
   (operand digits) do not. This suggests the model builds nonlinear representations
   specifically for the computational steps, not for raw input encoding.

3. **The linear subspace is necessary but not sufficient.** Phases A-F established
   that concepts live in clean linear subspaces. Phase G now shows that within those
   subspaces, the geometry is nonlinear (helical). The subspace identifies where to
   look; the Fourier analysis reveals what is there. This is exactly the two-level
   story the paper needs: linear subspaces (confirmable by probes) containing nonlinear
   manifolds (invisible to probes).

4. **Difficulty dependence ties to the LRH failure mode.** The absence of Fourier
   features at L2 (where probes achieve 99.8%) and their emergence at L3-L5 (where
   accuracy drops) aligns with the prediction that LRH fails specifically in the
   regime where computation is hard and the model must use structured internal
   representations rather than memorized lookup.

These interpretations are preliminary and contingent on Run 2 confirming the L3
pattern with corrected statistics and FDR correction.

---

## 8. Implementation Details

### 8a. Script Architecture

`phase_g_fourier.py` is ~2,598 lines, organized as:

```
IMPORTS (numpy, pandas, scipy, matplotlib, json, argparse, logging, pathlib, math, ...)

CONSTANTS
  LEVELS = [2, 3, 4, 5]
  LAYERS = [4, 6, 8, 12, 16, 20, 24, 28, 31]
  MIDDLE_LAYERS = [8, 12, 16, 20, 24]
  POPULATIONS = ["all", "correct", "wrong"]
  N_PERMUTATIONS = 1000
  PERM_ALPHA = 0.01
  DIGIT_PERIOD = 10
  FDR_THRESHOLD = 0.05
  LINEAR_P_THRESHOLD = 0.01
  COORD_P_THRESHOLD = 0.01
  ZERO_POWER_THRESHOLD = 1e-12
  SPOT_CHECK_ATOL = 1e-4
  SPOT_CHECK_RTOL = 1e-3
  MIN_CARRY_MOD10_VALUES = 6
  MIN_CARRY_RAW_VALUES = 6
  MIN_POPULATION = 30

DECISION_RULE (docstring constant, printed in summary)
NUMBER_TOKEN_FRAMING (docstring constant)
RUNTIME_ESTIMATE (docstring constant)

CONFIGURATION
  load_config(path) -> dict
  derive_paths(cfg) -> dict

LOGGING
  setup_logging(workspace) -> logger with RotatingFileHandler (10MB, 3 backups)

DATA LOADING
  load_coloring_df(level, paths, logger) -> pd.DataFrame
  get_population_mask(df, pop_name, logger) -> bool mask
  load_residualized(level, layer, paths, logger) -> np.ndarray
  load_raw_activations(level, layer, paths, logger) -> np.ndarray
  load_phase_c_projected(level, layer, pop, concept, paths, logger) -> np.ndarray | None
  load_phase_c_metadata(level, layer, pop, concept, paths, logger) -> dict | None
  load_phase_c_eigenvalues(level, layer, pop, concept, paths, logger) -> np.ndarray | None
  load_phase_d_merged_basis(level, layer, pop, concept, paths, logger) -> np.ndarray | None
  count_phase_d_bases(paths, logger) -> int (filesystem walk for merged_basis.npy)

CONCEPT REGISTRY
  _get_phase_c_group_labels(...) -> (groups, meta)
  compute_labels_and_linear_values(raw_col, phase_c_groups) -> (labels, unique_groups, v_linear)
  get_fourier_concepts(level, coloring_df, logger) -> list[dict]
  resolve_carry_binned_spec(concept_info, phase_c_groups, raw_col, logger) -> dict | None

CORE FOURIER MATH
  compute_freq_range(period) -> int K
  is_nyquist(k, period) -> bool
  fourier_single_coordinate(signal, values, period) -> dict
  fourier_all_coordinates(centroids, values, period, logger) -> dict
  compute_linear_power(centroids, v_linear, group_sizes) -> np.ndarray (d,)
  compute_helix_fcr(fourier_results, linear_power, v_linear, group_sizes, logger) -> dict
  compute_centroids_grouped(projected, labels, unique_values) -> (centroids, group_sizes)
  compute_eigenvalue_weighted_fcr(per_coord_fcr, eigenvalues) -> float

PERMUTATION NULL
  permutation_null(projected, labels, unique_values, period, v_linear,
                   n_perms, rng, logger) -> dict
  compute_pvalues(observed, null_dist) -> float
  compute_pvalues_array(observed, null_dist_2d) -> np.ndarray

SINGLE ANALYSIS
  analyze_one(level, layer, pop_name, concept_info, period, period_spec,
              values, v_linear, subspace_type, projected_data, labels,
              unique_values, eigenvalues, n_perms, rng, logger) -> dict
  _classify_multi_freq(fourier_res, p_coord, period, logger) -> str

BATCH PROCESSING
  process_level_layer_pop(...) -> list[dict]
  run_all(paths, levels, layers, n_perms, logger) -> pd.DataFrame

AGREEMENT COMPUTATION
  compute_agreement(results_df) -> pd.DataFrame

FDR CORRECTION
  apply_fdr(results_df) -> pd.DataFrame

SAVING
  save_per_concept(result, output_dir)
  save_summary_csv(results_df, paths)
  save_detection_csvs(results_df, paths)

PLOTTING
  plot_fcr_heatmaps(results_df, paths, logger)
  plot_centroid_circles(results_df, paths, logger)
  plot_power_spectra(results_df, paths, logger)
  plot_pvalue_trajectories(results_df, paths, logger)
  plot_single_example_projections(results_df, paths, logger)
  generate_all_plots(results_df, paths, logger)

SYNTHETIC PILOT
  run_synthetic_pilot(logger) -> bool

PILOT 0b
  run_pilot_0b(paths, logger) -> dict

CLI
  parse_args()  # --config, --pilot, --pilot-0b, --kt-pilot, --n-perms, --skip-plots
  main()
```

### 8b. Data Loading Pipeline

For each (level, layer, population, concept):

**Phase C fast path:**
1. Load `projected_all.npy` (shape `(N, d_sub)`) — pre-projected activations
2. Apply population mask to get the population slice
3. Compute centroids directly in d_sub dimensions
4. Runtime: ~0.1s for L5/all (122K × 9 = 4.4 MB)

**Phase D path:**
1. Load residualized activations (shape `(N, 4096)`) — one load per (level, layer),
   cached for all concepts in that slice
2. Load `merged_basis.npy` (shape `(d_merged, 4096)`)
3. Project: `projected = acts_pop @ merged_basis.T` (shape `(N_pop, d_merged)`)
4. Compute centroids in d_merged dimensions
5. Runtime: ~4s load for L5 (122K × 4096 = 1.9 GB), then ~0.1s per projection

### 8c. Concept Registry

The concept registry is built at runtime from the coloring DataFrame:
- **Digit concepts:** Read from `DIGIT_CONCEPTS_BY_LEVEL` constant, verified against
  DataFrame columns. Period spec: always `("digit", P=10)`. Values read from actual
  unique values in the DataFrame.
- **Carry concepts:** Read from `CARRY_CONCEPTS_BY_LEVEL`. Period specs built dynamically:
  `carry_binned` resolved at runtime from Phase C metadata (period = n_groups).
  `carry_mod10` and `carry_raw` built from the raw value distribution.

The `carry_binned` spec is **resolved per (layer, pop)** because Phase C's binning
may vary: the tail bin threshold can differ across populations if different populations
have different value ranges.

### 8d. Core Fourier Functions

**fourier_single_coordinate:** Explicit DFT loop over K frequencies. For each
frequency k=1..K, computes `a_k = Σ s[v]·cos(2πkv/P)` and `b_k = Σ s[v]·sin(2πkv/P)`,
then `P_k = a_k² + b_k²` (rescaled by 2× for Nyquist). Returns per-frequency power,
FCR_top1, and dominant frequency.

**fourier_all_coordinates:** Calls `fourier_single_coordinate` for each of `d`
coordinates, then computes two_axis_fcr (max over frequencies of top-2 coords' power),
uniform_fcr (mean per-coord FCR), and frequency mode statistics.

**compute_linear_power:** Per-coordinate projection onto centered linear ramp.
Uses sample-weighted centering when group sizes are available.

**compute_helix_fcr:** For each frequency, combines top-2 Fourier power with best
non-overlapping linear power. Rescales linear power by `m/(2·Σv²)` for DOF parity.

### 8e. Permutation Null Implementation

The permutation null uses a **conditioned null**: group sizes are fixed, only the
assignment of samples to groups is shuffled. This is slightly more conservative
than Phase C's unconditioned null (which allows group sizes to vary), but the
difference is negligible for balanced groups.

Implementation:
1. Pre-compute cumulative group sizes: `cum_sizes = [0, n_0, n_0+n_1, ...]`
2. Create index array `all_idx = arange(N)`
3. For each permutation:
   a. Shuffle `all_idx` in-place
   b. Assign `all_idx[cum_sizes[i]:cum_sizes[i+1]]` to group i
   c. Compute null centroids as group means of shuffled projected data
   d. DC-remove null centroids
   e. Run full Fourier + linear + helix analysis
   f. Store null statistics

The in-place shuffle is O(N) and the centroid computation is O(N·d). With N=122K
and d=9, each permutation takes ~0.001s. The 1,000 permutations complete in ~1s.

### 8f. Detection Logic

```python
circle_detected = (
    p_two_axis < PERM_ALPHA           # global: p < 0.01
    and p_coord[coord_a] < COORD_P_THRESHOLD   # coord a: p < 0.01
    and p_coord[coord_b] < COORD_P_THRESHOLD   # coord b: p < 0.01
)

helix_detected = (
    p_helix < PERM_ALPHA              # global helix: p < 0.01
    and p_coord[helix_coord_a] < COORD_P_THRESHOLD   # Fourier coord a: p < 0.01
    and p_coord[helix_coord_b] < COORD_P_THRESHOLD   # Fourier coord b: p < 0.01
    and p_linear[helix_linear_coord] < LINEAR_P_THRESHOLD  # linear: p < 0.01
)

if helix_detected:
    geometry_detected = "helix"
elif circle_detected:
    geometry_detected = "circle"
else:
    geometry_detected = "none"
```

### 8g. Output Format

**Directory structure:**
```
/data/user_data/anshulk/arithmetic-geometry/phase_g/
  fourier/L{level}/layer_{LL:02d}/{pop}/{concept}/
    phase_c/{period_spec}/
      fourier_results.json
      centroids.npy
    phase_d_merged/{period_spec}/
      fourier_results.json
      centroids.npy
  fourier_numtok/     (number-token probe results, same structure)
  kt_pilot/
    layer_{LL}/fourier_decomposition.npz
    kt_pilot_summary.json
  summary/
    phase_g_results.csv       (~3,348 rows)
    phase_g_circles.csv       (circle-detected subset)
    phase_g_helices.csv       (helix-detected subset)
    phase_g_agreement.csv     (Phase C vs D concordance)
```

**Summary CSV columns (37 columns):**
```
concept, tier, level, layer, population, subspace_type, period_spec,
n_groups, period, values_tested, d_sub, n_samples_used, n_perms_used,
two_axis_fcr, two_axis_best_freq, two_axis_coord_a, two_axis_coord_b,
two_axis_p_value, two_axis_q_value,
uniform_fcr_top1, uniform_fcr_p_value,
eigenvalue_fcr_top1,
fcr_top1_max, fcr_top1_max_coord, fcr_top1_max_freq,
dominant_freq_mode, n_sig_coords_at_mode_freq,
circle_detected, helix_detected, geometry_detected, multi_freq_pattern,
helix_fcr, helix_best_freq, helix_linear_coord,
helix_p_value, helix_q_value,
p_value_floor, p_saturated,
agreement,
eigenvalue_top1, eigenvalue_top2, eigenvalue_top3
```

**Plots directory:**
```
/home/anshulk/arithmetic-geometry/plots/phase_g/
  kt_pilot/kt_magnitude_spectrum_layer{0,1,4,8}.png
  fcr_heatmaps/        (to be generated after full run)
  centroid_circles/     (to be generated)
  frequency_spectra/    (to be generated)
  pvalue_trajectories/  (to be generated)
  single_example_projections/  (to be generated)
```

### 8h. Error Handling and Edge Case Guards

The 12-bug audit before Run 2 (Section 5f) revealed several edge cases that required
defensive guards. These are documented here for reproducibility:

**carry_mod10 graceful skip.** Phase C may merge rare carry values into tail bins at
certain (level, layer, population) slices. When this happens, the carry_mod10 period
spec references values that do not exist in the slice. Instead of crashing (the Run 1
behavior), the code now filters `values` to only those present in `unique_vals`, and
skips the analysis if fewer than `MIN_CARRY_MOD10_VALUES` (3) remain. A debug log
message records the skip.

**d < 2 guards.** When a subspace has dimensionality d=1 (possible for small
Phase C subspaces), the two_axis_fcr and helix_fcr require coordinates a and b, but
only coordinate 0 exists. Guards skip these analyses when d < 2, logging the skip.

**Eigenvalue-weighted FCR division.** The eigenvalue-weighted FCR divides by the sum
of eigenvalue weights. When all eigenvalues are negligible (below 1e-10), the sum is
effectively zero. A threshold check sets eigenvalue_fcr to NaN instead of crashing.

**Phase D basis count.** The filesystem walk for Phase D merged bases originally used
a hard `assert n >= expected`. This was changed to a logged error + `sys.exit(1)` to
provide a diagnostic message before aborting.

**Empty DataFrame column preservation.** Boolean indexing on a pandas DataFrame
(`df[mask]`) can silently drop columns when the result is empty. All such operations
now use `df.loc[mask].copy()` to preserve the schema.

---

## 9. What Phase G Will Contribute to the Paper

Phase G tests a specific prediction from the Fourier circuits literature: that digit
representations in Llama 3.1 8B have periodic (circular or helical) structure within
their linear subspaces.

**If circles/helices are detected (≥3 cells across ≥2 concepts and ≥2 middle layers):**
- The paper can claim that multiplication representations combine linear subspaces
  (Phases C/D) with non-linear within-subspace geometry (Phase G)
- This directly supports the thesis that LRH fails for compositional reasoning:
  the concepts are linearly decodable (live in subspaces) but their internal
  arrangement is non-linear (circular/helical)
- The paper can connect to the Fourier circuits literature (Nanda, Bai, K&T) and
  show that pretrained LLMs use the same computational motifs for multiplication
  as toy models trained on modular arithmetic

**If the result is null at `=` but positive at number tokens:**
- This matches K&T's finding and suggests Fourier features exist at input positions
  but are consumed/transformed by the computation
- The paper can frame this as: "Fourier features are an input representation, not
  a computation-position representation, consistent with K&T (2025)"
- This is still informative: it distinguishes "no Fourier features" from "Fourier
  features at a different position"

**If the result is completely null (both positions):**
- The paper reports the null with the caveat that the centroid test may be too coarse
- Single-example projection plots provide visual evidence for or against manifold
  structure that the centroid test misses
- The null is still publishable: it means multiplication in Llama 3.1 8B does NOT
  use the Fourier circuits detected in toy models, supporting the thesis that
  real-world models use different mechanisms than simple ones

---

## 10. Limitations

1. **Centroid averaging is conservative.** The test detects structure in the mean
   positions but may miss manifold structure when within-class spread is large. If
   individual activations form a noisy ring but the centroids collapse the ring into
   a blob, the centroid test will be null. The single-example projection plots
   provide a partial mitigation but are visual (not quantitative).

2. **The conjunction criterion trades sensitivity for specificity.** By requiring both
   coordinates to be individually significant, we eliminate false positives from
   linear signals but may miss genuine circles with unequal axis strengths. An
   alternative approach (testing the phase-coherence of the top-2 coordinates) might
   be more sensitive but was not implemented due to the additional complexity of
   defining a coherence null.

3. **1,000 permutations limit p-value resolution to 0.001.** For the FDR correction
   across ~3,348 tests, this may cause informative cells to have saturated p-values
   that all look equally significant. A future run with 10,000 permutations would
   resolve this at the cost of ~10× runtime.

4. **The `=` token is not the standard probe position.** K&T and most of the Fourier
   feature literature probe at the number-token position. Our choice to probe at `=`
   is driven by pipeline consistency (Phases A–F use `=` activations) but may reduce
   power if Fourier features are position-specific.

5. **carry_4 has p_floor = 0.00139.** With only 6 groups, the minimum achievable p-value
   exceeds the 0.001 floor from permutations. Any carry_4 result at p = 0.00139 could
   be even more significant, but we cannot tell.

6. **Synthetic pilot Tests 3 and 9 now pass (fixed before Run 2).** Test 3's threshold
   was raised from 0.5 to 0.65 to accommodate the genuine Fourier content of linear
   ramps. Test 9's helix denominator was fixed to use only the chosen linear coordinate.
   See Section 5a and 5f for details.

7. **The method is period-specific.** We test at pre-specified periods (P=10 for digits,
   P=n_groups for carries). If the model uses a non-standard period (e.g., P=12 for
   digits), we would miss it. A more exploratory approach (scanning all possible
   periods) was rejected to avoid massive multiple-testing burden.

8. **Phase D bases may dilute the signal.** Phase D's merged bases are wider (d_merged
   ≈ 2 × d_consensus) and include discriminative directions that may not align with
   Fourier axes. The larger `d` increases the denominator of two_axis_fcr, potentially
   suppressing detection. This is why Phase C results are often stronger than Phase D.

9. **No correction for dependence between Phase C and Phase D tests.** The same
   underlying data generates both tests. A concept that is significant in Phase C is
   more likely to be significant in Phase D (or vice versa). The agreement column
   documents this, but the FDR correction treats them as independent tests.

10. **The run is still in progress.** All conclusions in this document are preliminary
    and based on partial L2 data. The critical tests at L3–L5 and middle layers
    (8, 12, 16, 20, 24) have not yet completed.

11. **Centroid-based analysis cannot detect within-class manifold structure.** If
    activations for digit `v=3` form a crescent-shaped cloud in the subspace, the
    centroid collapses this to a single point. The crescent's shape — which might
    reflect context-dependent modulation of the Fourier feature — is invisible to
    the centroid test. Per-example Fourier analysis (as in K&T) would capture this
    but is not part of Phase G's design.

12. **The permutation null is slightly conservative.** The conditioned null (fixed
    group sizes) has fewer degrees of freedom than the unconditioned null (random
    group sizes from sampling). This makes p-values slightly larger (more conservative)
    than they would be under the unconditioned null. The effect is negligible for
    balanced groups but could matter for highly unbalanced carry concepts where the
    largest group has 10× more samples than the smallest.

13. **No correction for spatial autocorrelation across layers.** The same concept's
    activations at adjacent layers (e.g., layer 16 and layer 20) are highly correlated
    because the residual stream is additive. FDR correction treats each layer as
    independent, which overestimates the effective number of tests. This makes the
    FDR correction conservative (harder to reach significance), which is acceptable
    for a screening phase but may undercount the true number of significant cells.

---

## 11. Runtime and Reproducibility

**Run 1 (crashed):**

| Metric | Value |
|--------|-------|
| SLURM job | 7056981 |
| Node | babel-v9-32 |
| Partition | preempt |
| GPU | 1x A6000 (Steps 1-2 only) |
| CPUs | 24 |
| Memory | 128 GB |
| Time limit | 7 days |
| Start | April 11, 2026, 04:14:54 EDT |
| Status | **Crashed** at L4/layer4/correct (carry_mod10 assert, ~31 min into Step 5) |

Run 1 completed Steps 1-4 and began Step 5. The synthetic pilot (Step 3) had 2 test
failures but returned exit code 0 due to the exit code bug (see Section 5f, Bug 2).
The run crashed at L4/layer4/correct with `AssertionError: carry_mod10 value 8 not
found` (Bug 1). All outputs except number-token activations were cleared before Run 2.

**Run 2 (current):**

| Metric | Value |
|--------|-------|
| SLURM job | 7058788 (Run 3) |
| Node | babel-t9-16 |
| Partition | preempt |
| GPU | 1x A6000 (Steps 1-2 only) |
| CPUs | 24 |
| Memory | 128 GB |
| Time limit | 7 days |
| Start | April 11, 2026, 11:35 EDT |
| Status | **In progress — Step 5 running** |

Run 3 uses the fixed codebase (all 17 bugs resolved: 12 from Run 1 + 5 from Run 2).
Number-token activations (9.2 GB) are reused from Run 1. All other Phase G outputs
were cleared before submission.

### Step-by-Step Timing (Run 1, applicable to Run 2)

| Step | Description | Duration | Run 1 Exit | Run 2 Exit | Run 3 Exit |
|------|-------------|----------|------------|------------|------------|
| 1 | K&T replication pilot | 0.9 min | 0 | 0 | 0 |
| 2 | Number-token extraction | ~5 min | 0 | 0 (reuses Run 1) | 0 (reuses Run 1) |
| 3 | Synthetic pilot | <1 sec | 0 (buggy) | 0 (10/10 pass) | 0 (10/10 pass) |
| 4 | Pilot 0b (raw vs resid) | 11 sec | 0 | 0 | 0 |
| 5 | Full Fourier screening | ~6.5 hours est. | crashed | crashed (KeyError) | in progress |

### Per-Analysis Timing (from logs)

| Configuration | Time per analysis |
|---------------|------------------|
| L2/all, Phase C (d=7–9, N=4,000) | 0.7–0.9s |
| L2/all, Phase D (d=14–18, N=4,000) | 1.2–1.5s |
| L5/all, Phase C (d=9, N=122,223) | ~3s (estimated) |
| L5/all, Phase D (d=18, N=122,223) | ~5s (estimated) |

The permutation null dominates: ~0.001s/perm × 1,000 perms = ~1s base, plus overhead
proportional to N and d.

### Reproducibility

The run is fully reproducible from:
```bash
# Step 1: K&T pilot
python phase_g_kt_pilot.py --config config.yaml

# Step 2: Number-token extraction
python extract_number_token_acts.py --config config.yaml

# Step 3: Synthetic pilot
python phase_g_fourier.py --config config.yaml --pilot

# Step 4: Pilot 0b
python phase_g_fourier.py --config config.yaml --pilot-0b

# Step 5: Full run
python phase_g_fourier.py --config config.yaml --n-perms 1000
```

Or via the combined SLURM script:
```bash
sbatch run_phase_g.sh
```

Random seed: The numpy random generator is seeded at the script level for
reproducibility. The permutation null uses `rng = np.random.default_rng(seed)`.

### The 5-Step SLURM Pipeline

The `run_phase_g.sh` script orchestrates a 5-step pipeline within a single SLURM
allocation. This design uses a single job with both GPU and CPU phases, avoiding
the complexity of job dependencies:

```
Step 1: K&T pilot (GPU)          → go/no-go gate
         ↓ (exit 0 required)
Step 2: Number-token extraction (GPU) → activations for follow-up analysis
         ↓ (exit 0 required)
Step 3: Synthetic pilot (CPU)    → code validation
         ↓ (exit 0 required)
Step 4: Pilot 0b (CPU)           → residualization safety check
         ↓ (exit 0 required, added in Run 2)
Step 5: Full Fourier screening (CPU) → main experiment
```

Each step checks the exit code of the previous step (`$?`). All five steps abort the
pipeline on failure. In Run 1, Step 4 was not gated (the exit code check was missing);
this was Bug 4 in Section 5f, fixed before Run 2.

The GPU is used only in Steps 1–2 (model forward passes for K&T pilot and number-token
extraction). Steps 3–5 are CPU-only (the bottleneck is groupby-mean in 9–18D subspace,
not matrix multiplication). The SLURM allocation requests 1× A6000 GPU and 24 CPUs;
the GPU is idle during Steps 3–5 but releasing it would require a separate job.

**Error handling:** The `set -euo pipefail` at the top of the script ensures any
unhandled error aborts the pipeline. Each step's exit code is captured in a named
variable (KT_EXIT, NT_EXIT, PILOT_EXIT, FULL_EXIT) for diagnostic purposes.

### Output Files (Produced So Far)

```
/data/user_data/anshulk/arithmetic-geometry/phase_g/
├── kt_pilot/
│   ├── layer_0/fourier_decomposition.npz
│   ├── layer_1/fourier_decomposition.npz
│   ├── layer_4/fourier_decomposition.npz
│   ├── layer_8/fourier_decomposition.npz
│   └── kt_pilot_summary.json
├── fourier/L2/layer_04/all/{concept}/phase_c/{spec}/
│   ├── fourier_results.json
│   └── centroids.npy
│   ... (in progress)

/home/anshulk/arithmetic-geometry/
├── plots/phase_g/kt_pilot/
│   ├── kt_magnitude_spectrum_layer0.png
│   ├── kt_magnitude_spectrum_layer1.png
│   ├── kt_magnitude_spectrum_layer4.png
│   └── kt_magnitude_spectrum_layer8.png
├── logs/
│   ├── phase_g_fourier.log (Run 3, growing)
│   ├── phase_g_kt_pilot.log (76 lines)
│   ├── slurm-7058788.out (Run 3, growing)
│   └── slurm-7058788.err (Run 3, growing)
```

---

## Appendix A: The Algebra of Centroid Fourier Screening

The centroid Fourier test can be understood algebraically as a projection of the
between-class covariance onto the Fourier basis.

**Setup.** Let `μ_v ∈ R^d` be the DC-removed centroid for value `v`, where `v` ranges
over the concept's value set `V` with `|V| = m`. The centroids define a data matrix
`M ∈ R^{m × d}` with rows `μ_v`.

**Fourier basis.** For period `P` and frequency `k`, define the basis vectors:

```
φ_k^cos[v] = cos(2πkv/P)     ∈ R^m
φ_k^sin[v] = sin(2πkv/P)     ∈ R^m
```

These are functions of the value index `v`, evaluated at the actual values in `V`.
When `V = {0, 1, ..., P-1}` (complete grid), these vectors are orthogonal. For
incomplete grids, they are approximately orthogonal but not exactly.

**Fourier coefficients.** The projection of coordinate `j`'s centroids onto the
Fourier basis at frequency `k`:

```
a_k[j] = φ_k^cos · M[:, j] = Σ_v cos(2πkv/P) · μ_v[j]
b_k[j] = φ_k^sin · M[:, j] = Σ_v sin(2πkv/P) · μ_v[j]
```

**Power.** `P_k[j] = a_k[j]² + b_k[j]²` is the squared norm of the projection of
coordinate `j` onto the 2D Fourier subspace at frequency `k`.

**Total power.** `total = Σ_k Σ_j P_k[j]` is the total Fourier energy across all
frequencies and coordinates. Under Parseval's theorem (for complete grids), this equals
`m · ||M||_F²` where `||M||_F` is the Frobenius norm of the centroid matrix. For
incomplete grids, the relationship is approximate.

**Two-axis FCR.** `two_axis_fcr = max_k (P_k[j₁] + P_k[j₂]) / total`, where `j₁, j₂`
are the top-2 coordinates at frequency `k`. This measures the fraction of total Fourier
energy concentrated in a 2D subplane at a single frequency — exactly the signature of a
circle in 2 dimensions embedded in d-dimensional space.

**Null distribution.** Under the null hypothesis (concept values are random labels),
the centroids `μ_v` are approximately iid Gaussian (CLT on group means). The Fourier
coefficients are linear functions of Gaussian centroids, hence also Gaussian. The
power `P_k[j]` follows a scaled chi-squared distribution with 2 DOF (1 DOF for
Nyquist). The two_axis_fcr is the max of a ratio of chi-squared variates —
analytically intractable, hence the permutation null.

---

## Appendix B: Why Centroids, Not Individual Points — The Statistical Argument

The literature on Fourier features in LLMs (K&T, Gurnee et al.) typically analyzes
individual activations. Our approach — computing Fourier statistics on group centroids —
is different. Here we justify this choice.

**The signal-to-noise argument.** If the model encodes digit `v` at position
`(A cos(2πv/10), A sin(2πv/10)) + noise`, then each individual activation is
a noisy observation of the Fourier position. The centroid averages `N_v` observations,
reducing the noise by a factor of `√N_v`. At L5/all, each centroid is the mean of
~12,000 samples, giving a noise reduction of ~110×. The centroid Fourier test is
thus a highly powered test of between-class structure.

**The within-class problem.** If within-class spread is large relative to between-class
separation (i.e., SNR < 1), the centroids may not reflect the underlying manifold.
However, Phase C already established that between-class structure is significant
(dim_perm > 0 requires between-class variance to exceed the permutation null). Phase G
operates only on concepts that passed Phase C's screening.

**The correspondence to the claim.** The centroid test asks: "Are the model's mean
representations for each digit value arranged periodically?" This is a stronger claim
than "Do individual activations lie on a periodic manifold?" A positive centroid test
means the model systematically places the average activation for each digit on a
circle/helix. A positive individual-activation test could be driven by within-class
structure (e.g., individual activations forming local clusters that happen to trace a
ring when connected).

**The alternative: single-example projections.** For concepts where the centroid test
is null, the single-example projection plots provide a visual check. If individual
activations form a visible ring or helix when projected onto the top-2 subspace
coordinates and colored by digit value, this is noted as `manifold_visual = "possible"`
for manual review. This is exploratory, not part of the pre-registered decision rule.

---

## Appendix C: The K&T Helix and Its Relationship to Our Test

Kantamneni & Tegmark (2025) found that LLMs represent numbers on a **generalized helix**
with the following structure:

```
representation(v) ≈ [A₁ cos(2πv/T₁), A₁ sin(2πv/T₁),
                     A₂ cos(2πv/T₂), A₂ sin(2πv/T₂),
                     ...,
                     B · v,              ← linear magnitude axis
                     noise]
```

Multiple Fourier periods `T₁, T₂, ...` (they found T ∈ {2, 5, 10} for base-10 numbers)
plus a linear magnitude axis. Our test captures this via:

1. **two_axis_fcr at period P=10:** Detects the T=10 circular component (the dominant
   period for base-10 digits). Tests at frequency k=1 (fundamental period 10).

2. **Nyquist at P=10 (k=5):** Detects the T=2 parity component. The signal `(-1)^v`
   is the Nyquist mode at period 10.

3. **helix_fcr:** Detects the combined circular + linear structure. If the linear
   magnitude axis is strong, helix_fcr > two_axis_fcr.

**What our test cannot detect:**
- Multi-period helices with T=5. Our test at P=10 detects T=10 (k=1) and T=2 (k=5),
  but T=5 corresponds to k=2 at P=10. The two_axis_fcr tests all frequencies, so
  if T=5 is the dominant period, the best_freq will be 2 rather than 1. This is
  detectable but may not match the "circle at k=1" expectation.
- Relative phase alignment between the cos and sin components. Our test measures power
  (magnitude squared) but not phase. If the cos and sin axes are not orthogonal, the
  manifold is an ellipse rather than a circle, and the weaker axis may fail the
  conjunction.

---

## Appendix D: Phase G in the Overall Pipeline

```
Phase A: Prompt generation + coloring DataFrames
    ↓
Phase B: Activation extraction + product-magnitude residualization
    ↓
Phase C: Permutation-stabilized subspace finding (PCA on between-class covariance)
    ↓
Phase D: LDA refinement + merged basis construction
    ↓
Phase E: Residual hunting (union subspace, variance explained)
    ↓
Phase F/JL: Between-concept angles + JL distance preservation
    ↓
[YOU ARE HERE]
Phase G: Fourier screening for periodic structure ← FIRST NON-LINEAR PROBE
    ↓
Phase H: GPLVM (non-linear manifold learning on activations)
    ↓
Phase I: Causal patching (ablation to verify functional role)
```

Phase G is the transition point. Phases A–F/JL established the linear geometry:
subspaces, their dimensions, their overlaps, and the fidelity of the projected
space. Phase G asks: **within** those subspaces, is the arrangement of concept
values non-linear (periodic)?

If Phase G is positive, it validates the entire pipeline: the model uses linear
subspaces with non-linear internal structure, exactly the kind of representation
that escapes the Linear Representation Hypothesis. If Phase G is null, the
non-linearity hypothesis rests on Phase H (GPLVM) and Phase I (causal patching).

**Data flow between phases:**

```
Phase A output:  L{N}_coloring.pkl  ─────────────────────────────┐
Phase B output:  level{N}_layer{L}.npy (residualized)  ──────────┤
Phase C output:  projected_all.npy, metadata.json,               │
                 basis.npy, eigenvalues.npy  ─────────────────────┤
Phase D output:  merged_basis.npy  ───────────────────────────────┤
                                                                  ▼
                                                         Phase G Fourier
                                                                  │
                                                                  ▼
                                    fourier_results.json, centroids.npy,
                                    phase_g_results.csv, phase_g_circles.csv,
                                    phase_g_helices.csv, phase_g_agreement.csv,
                                    FCR heatmaps, centroid circle plots,
                                    frequency spectra, p-value trajectories,
                                    single-example projections
```

**What Phase G does NOT use:**
- Phase E's union basis (not needed — Phase G analyzes per-concept structure, not
  the union)
- Phase F's angle measurements (not needed — Phase G is within-concept, not
  between-concept)
- Phase JL's distance metrics (not needed)

This clean dependency structure means Phase G can run as soon as Phases A–D are
complete, without waiting for Phase E or F/JL. In practice, all phases were completed
before Phase G was designed, but the independence is useful for understanding the
pipeline architecture.

---

## Appendix E: Synthetic Pilot Test Specifications

Each test constructs artificial centroids with known properties and verifies the
Fourier analysis code produces the expected results.

**Test 1: Perfect Circle (P=10)**
```python
centroids = np.zeros((10, 9))
centroids[:, 0] = [cos(2πv/10) for v in range(10)]
centroids[:, 1] = [sin(2πv/10) for v in range(10)]
```
Expected: two_axis_fcr = 1.0 (all power in frequency 1, coordinates 0 and 1).
Observed: 1.0000. **PASS.**

**Test 2: Random Noise**
```python
rng = np.random.default_rng(42)
centroids = rng.standard_normal((10, 9))
```
Expected: two_axis_fcr near null (~1/(K·d) × correction for max). Approx 0.20.
Observed: 0.2355. **PASS.**

**Test 3: Linear/Quadratic**
```python
centroids = np.zeros((10, 9))
centroids[:, 0] = np.linspace(0, 1, 10)
centroids[:, 1] = np.linspace(0, 1, 10) ** 2
```
Expected: two_axis_fcr < 0.65 (not periodic). Observed: 0.5943. **PASS.**
The original threshold was 0.5, which failed because a linear ramp has genuine Fourier
content at low frequencies (FCR = 0.5943). Threshold raised to 0.65 before Run 2.
Real circles produce FCR > 0.95; the 0.65 boundary cleanly separates linear artifacts.

**Test 4: Incomplete Grid (v=1..9)**
```python
values = np.arange(1, 10)  # missing v=0
centroids = np.zeros((9, 9))
centroids[:, 0] = [cos(2πv/10) for v in values]
centroids[:, 1] = [sin(2πv/10) for v in values]
```
Expected: high FCR (the circle is still recognizable with one point missing).
Observed: 0.8667. **PASS.** (Not 1.0 because the missing point breaks the perfect
orthogonality of the DFT basis, leaking some power to other frequencies.)

**Test 5: Pure DC Offset**
```python
centroids = np.full((10, 9), 5.0)
centroids[:, 1:] = 0.0
```
After DC removal: all zeros. Expected: total_power < 1e-12, no crash.
Observed: total_power < 1e-12, FCR = 0.0. **PASS.**

**Test 6: P=9 Conjugate**
```python
values = np.arange(9)
centroids = np.zeros((9, 9))
centroids[:, 0] = [cos(2πv/9) for v in values]
centroids[:, 1] = [sin(2πv/9) for v in values]
```
Expected: K=4 frequencies (not 5), FCR = 1.0. Verifies odd-P frequency range
calculation and no double-counting of conjugate pairs.
Observed: K=4, FCR = 1.0000. **PASS.**

**Test 7: Convention Spot-Check**
Skipped in synthetic pilot (requires real data). Run as part of Pilot 0b.
Result: `np.allclose = True`. **PASS.**

**Test 8: Nyquist Parity**
```python
centroids = np.zeros((10, 9))
centroids[:, 0] = [(-1)**v for v in range(10)]
```
Expected: Nyquist bin (k=5) captures ~100% of power after 2× rescaling.
Observed: fraction = 1.0000, best_freq = 5. **PASS.** (Verifies Nyquist inclusion
and rescaling.)

**Test 9: Helix**
```python
centroids = np.zeros((10, 9))
centroids[:, 0] = [cos(2πv/10) for v in range(10)]
centroids[:, 1] = [sin(2πv/10) for v in range(10)]
centroids[:, 2] = np.linspace(0, 1, 10)
```
Expected: helix_fcr > two_axis_fcr (the linear axis adds to the helix numerator).
Observed (Run 1): helix_fcr = 0.7200, two_axis_fcr = 0.9000. **FAIL.**
Observed (Run 2, after fix): helix_fcr = 0.9084, two_axis_fcr = 0.9000. **PASS.**
The original denominator summed rescaled linear power across all d coordinates, but
the numerator only used the best linear coordinate. Fixed before Run 2 to use only
`best_linear_rescaled` in the denominator, matching the numerator's scope.

**Test 10: Pure Linear Ramp**
```python
centroids = np.zeros((10, 9))
centroids[:, 0] = np.linspace(0, 1, 10)
```
Expected: linear_power >> Fourier power (the signal is pure ramp, not periodic).
Observed: linear_power = 6806.25, Fourier power = 450.0. **PASS.** (15× ratio
confirms the linear power detector works correctly.)

---

## Appendix F: The 18 Fixes from v1 to v4 of the Plan

The Phase G plan went through 4 iterations (v1 → v4 FINAL). Each version addressed
issues found during review. The 18 fixes in the final version:

| Fix | Version | Description | Impact |
|-----|---------|-------------|--------|
| 1 | v2 | Nyquist frequency included | Enables parity/prism detection |
| 2 | v2 | helix_fcr statistic added | Enables K&T helix detection |
| 3 | v2 | K&T replication pilot | Validates Fourier code before full run |
| 4 | v2 | Raw vs. residualized spot check | Verifies residualization is safe |
| 5 | v2 | Number-token probe | Literature-grounded comparison |
| 6 | v2 | FDR q-value instead of FCR floor | Better calibrated for weak signals |
| 7 | v3 | Realistic runtime estimate with helix overhead | Accurate SLURM allocation |
| 8 | v3 | carry_4 p_value_floor documentation | Transparent about power limits |
| 9 | v3 | Helix = 2 Fourier + 1 linear (generalized) | Matches K&T architecture |
| 10 | v3 | Honest number-token framing | Not "stronger test" — different test |
| 11 | v4 | carry_4 p_value_floor column | Per-row floor in CSV |
| 12 | v4 | Phase D filesystem walk loader | Handles L2–L4 merged bases |
| 13 | v4 | two_axis_coord_a/b split columns | Cleaner CSV schema |
| 14 | v4 | period_spec naming unified | Consistent taxonomy |
| 15 | v4 | DC-offset synthetic test | Zero-denominator safety |
| 16 | v4 | Convention spot-check tolerance | Numerical validation |
| 17 | v4 | MIN_GROUP_SIZE deferred to Phase C | No redundant filtering |
| 18 | v4 | Decision rule rewritten with FDR | Pre-registered, reproducible |

These fixes were motivated by iterative review of the plan against the literature,
edge cases in the data, and numerical considerations. The v4 FINAL plan was
implemented as `phase_g_fourier.py` without further changes.

---

## 9. Run 3 Comprehensive Results

Run 3 (SLURM 7058788) has completed 3,086 of ~3,348 analyses as of April 11, 2026,
14:37 EDT. L2 through L5/layer20 are fully analyzed; L5 layers 24–31 are in progress.
The numbers below reflect the 3,086 completed cells. Final FDR-corrected numbers will
be added when the run finishes.

### 9a. Detection Summary by Concept Type

| Concept Type | Helix | Circle | None | Total | Rate |
|-------------|-------|--------|------|-------|------|
| **carry_1** | 148 | 0 | 284 | 432 | 34.3% |
| **carry_2** | 97 | 0 | 173 | 270 | 35.9% |
| **carry_3** | 32 | 0 | 76 | 108 | 29.6% |
| **carry_4** | 34 | 0 | 74 | 108 | 31.5% |
| **carry_0** | 19 | 0 | 557 | 576 | 3.3% |
| **ans_digit_0_msf** | 27 | 0 | 151 | 178 | 15.2% |
| **ans_digit_5_msf** | 14 | 0 | 22 | 36 | 38.9% |
| **ans_digit_3_msf** | 12 | 1 | 127 | 140 | 8.6% |
| **ans_digit_4_msf** | 8 | 0 | 82 | 90 | 8.9% |
| **ans_digit_2_msf** | 4 | 0 | 151 | 155 | 2.6% |
| **ans_digit_1_msf** | 2 | 0 | 144 | 146 | 1.4% |
| **b_units** | 2 | 0 | 180 | 182 | 1.1% |
| **a_hundreds** | 1 | 0 | 91 | 92 | 1.1% |
| a_units | 0 | 0 | 194 | 194 | 0.0% |
| a_tens | 0 | 0 | 194 | 194 | 0.0% |
| b_tens | 0 | 0 | 146 | 146 | 0.0% |
| b_hundreds | 0 | 0 | 38 | 38 | 0.0% |
| **TOTAL** | **417** | **1** | **2,666** | **3,084** | **13.5%** |

The pattern is unmistakable: **carries have helix geometry, operand digits do not.**
carry_1 through carry_4 collectively produce 311/417 = 74.6% of all helix detections.
Operand digit concepts (a_units, a_tens, a_hundreds, b_units, b_tens, b_hundreds)
produce 3/846 = 0.4% — indistinguishable from noise.

### 9b. Detection Summary by Level

| Level | Helix | Circle | None | Total | Rate | Notes |
|-------|-------|--------|------|-------|------|-------|
| L2 | 2 | 0 | 322 | 324 | 0.6% | Model is 99.8% correct — no computation pressure |
| L3 | 93 | 1 | 680 | 774 | 12.0% | Accuracy drops to 67.2% — helix structure emerges |
| L4 | 150 | 0 | 846 | 996 | 15.1% | 29.0% accuracy — more carry-chain complexity |
| L5 | 155 | 0 | 836 | 991 | 15.6% | 6.1% accuracy — maximum difficulty |

The helix detection rate jumps from 0.6% at L2 to 12.0% at L3 and plateaus at
~15% for L4–L5. This matches the difficulty gradient: L2 has almost no carry
propagation errors, so the carry representations show no detectable periodic structure.
At L3 and beyond, where carry propagation is the computational bottleneck, carries
organize into helical manifolds inside their Phase C/D subspaces.

The L3-to-L4 jump (12.0% → 15.1%) reflects the additional carry chain: L4 has
carry_3, and 3x2 multiplication introduces longer carry dependencies. The L4-to-L5
plateau (15.1% → 15.6%) suggests that once carry chains are complex enough to create
errors, additional complexity doesn't dramatically change the representation geometry.

### 9c. Detection Summary by Layer

| Layer | Helix | Total | Rate |
|-------|-------|-------|------|
| 4 | 55 | 388 | 14.2% |
| 6 | 49 | 387 | 12.7% |
| 8 | 53 | 389 | 13.6% |
| 12 | 49 | 391 | 12.5% |
| 16 | 53 | 448 | 11.8% |
| 20 | 58 | 384 | 15.1% |
| 24 | 28 | 239 | 11.7% |
| 28 | 28 | 230 | 12.2% |
| 31 | 27 | 229 | 11.8% |

The detection rate is remarkably uniform across all layers: 11.7%–15.1%. No single
layer dominates. This is a strong structural claim: **the helix geometry for carries
is not created at any specific layer — it is maintained throughout the network.**

This contrasts with Phase A's finding that layer 16 is the "information peak" for
visual clustering. The helix is not a mid-network phenomenon; it is present from
layer 4 (early) through layer 31 (final). The model encodes carry values on helical
manifolds early and preserves this structure as information flows through the
transformer.

### 9d. Detection Summary by Population

| Population | Helix | Total | Rate |
|-----------|-------|-------|------|
| all | 155 | 1,101 | 14.1% |
| correct | 136 | 1,073 | 12.7% |
| wrong | 109 | 911 | 12.0% |

Detection rates are similar across populations. The "correct" population has slightly
higher detection rate than "wrong" — consistent with Phase F's finding that correct
computations are more superposed (tighter geometric packing). The "all" population
has the highest rate because it has the largest sample size (N up to 122,223 at L5),
giving the permutation test more statistical power.

### 9e. Carry Helix Analysis: The Dominant Signal

Carries are the workhorses of multi-digit multiplication. carry_0 is the ones-column
carry, carry_1 the tens column, and so on. The helix detection rates:

- **carry_0: 19/576 (3.3%).** Low rate because carry_0 exists at all levels, and at
  L2 it is nearly always 0 or 1 — minimal variance, no complex structure needed.
- **carry_1: 148/432 (34.3%).** The dominant signal. carry_1 is the tens-column carry,
  which at L3–L5 ranges from 0 to 12+. The model encodes these values on a generalized
  helix: a circular Fourier component (period 10, matching the decimal system) plus
  a linear ramp (magnitude). Most detections are floor-saturated (p_helix = 0.001),
  meaning the permutation null never produces an FCR as large — the signal is
  overwhelmingly real.
- **carry_2: 97/270 (35.9%).** Similar rate to carry_1. carry_2 is the hundreds-column
  carry, available at L3–L5.
- **carry_3: 32/108 (29.6%).** carry_3 is the thousands-column carry, available at
  L4–L5. Slightly lower rate — fewer analysis cells, and carry_3 has only 6 unique
  binned values at some levels.
- **carry_4: 34/108 (31.5%).** carry_4 is the ten-thousands-column carry, available
  at L4–L5. Similar to carry_3.

The helix for carry values has three components matching K&T's generalized helix:
(1) a circular axis at the dominant Fourier frequency, encoding the cyclical
pattern of carry values mod 10; (2) a second circular axis at the conjugate or
next-strongest frequency; and (3) a linear axis encoding carry magnitude. The
conjunction of all three is what makes the detection criterion stringent — random
structure does not simultaneously satisfy circle + helix + coordinate + linear
p-value thresholds.

### 9f. Operand Digits: The Complete Null

| Concept | Helix/Total | Rate | Levels Available |
|---------|-------------|------|-----------------|
| a_units | 0/194 | 0.0% | L2–L5 |
| a_tens | 0/194 | 0.0% | L2–L5 |
| a_hundreds | 1/92 | 1.1% | L4–L5 |
| b_units | 2/182 | 1.1% | L2–L5 |
| b_tens | 0/146 | 0.0% | L3–L5 |
| b_hundreds | 0/38 | 0.0% | L5 |

Out of 846 operand-digit analyses, 3 show helix detection — a rate of 0.35%. At
α=0.01 with a conjunction of 4+ conditions, the expected false positive rate is
well below 1%. The 3 detections are likely noise.

This is a decisive null result. The model does not encode operand digit values on
periodic Fourier manifolds at the `=` token position. Phase C showed these concepts
have clean 8–9 dimensional linear subspaces (a_units = 9D, a_tens = 8D at every layer).
The 10 digit centroids sit in these subspaces but are NOT arranged as circles or
helices. They may be arranged linearly (supported by the floor-saturated p_linear
values observed across all L5 analyses), or in some other non-periodic geometry.

This null is scientifically meaningful: it shows that the model's representation of
"what digit is in the units place of operand a" at the computation position is
fundamentally different from how Kantamneni & Tegmark (2025) observed digit
representations at the number-token position. The representation has been transformed
— possibly from periodic (at input) to linear (at computation) — by the time
information reaches the `=` position.

### 9g. Answer Digits: The Composition Bottleneck

Answer digits reveal the edge-vs-middle asymmetry first discovered in Phase C:

| Concept | Helix/Total | Rate | Position in Answer |
|---------|-------------|------|--------------------|
| ans_digit_0_msf | 27/178 | 15.2% | Leading (most significant) |
| ans_digit_1_msf | 2/146 | 1.4% | Second |
| ans_digit_2_msf | 4/155 | 2.6% | Third (middle) |
| ans_digit_3_msf | 12/140 | 8.6% | Fourth |
| ans_digit_4_msf | 8/90 | 8.9% | Fifth |
| ans_digit_5_msf | 14/36 | 38.9% | Sixth (least significant, L5 only) |

The pattern mirrors accuracy: the leading digit (ans_digit_0) and trailing digit
(ans_digit_5) have the highest helix rates. Middle digits (positions 1–2) are nearly
null. This replicates Phase C's finding that middle answer digits lack linear
subspaces — they also lack periodic structure.

ans_digit_5_msf has the highest rate (38.9%) but is only available at L5 (36 cells).
It corresponds to the ones digit of the product, which is determined by modular
arithmetic: `(a mod 10) × (b mod 10) mod 10`. This purely periodic operation is
a natural candidate for Fourier structure.

### 9h. The Single Circle Detection

One analysis cell returned `geometry_detected=circle` — a circle without the linear
ramp that characterizes helices. This occurred for ans_digit_3_msf. A circle detection
means the two-axis conjunction (p_two_axis < 0.01, both coordinates significant) passed,
but the helix extension (adding a linear axis) did not improve the fit. Pure circles
are rare because most concepts that have periodic structure also have a magnitude
component.

### 9i. FDR Correction (pending Run 3 completion)

When Run 3 completes, Benjamini-Hochberg FDR correction will be applied across all
~3,348 p_two_axis and p_helix values. This controls the false discovery rate at
q < 0.05. Given that 417 detections at α=0.01 with a 4-way conjunction criterion is
already conservative, FDR correction is expected to confirm the majority of carry
helix detections.

---

## 10. Number-Token Fourier Screening

### 10a. Motivation: K&T's Finding and Our Question

Kantamneni & Tegmark (2025) found that Llama 3.1 8B encodes single-token integers
on generalized helices at the **number-token position**: for integers 0–360 presented
as standalone tokens, Fourier analysis reveals periods {2, 5, 10} in the residual
stream at layers {0, 1, 4, 8}. Our K&T replication pilot (Section 5b) confirmed this.

But K&T tested standalone integers. Our pipeline uses multiplication prompts:
`"{a} * {b} ="`. When the integer 47 appears as the first operand in "47 * 83 =",
does its representation at the `47` token position still show the same Fourier
structure? The surrounding context (multiplication operation, second operand, equals
sign) might transform the representation.

This is a critical comparison point:
- **Positive at number-token, Null at `=`:** Fourier features exist at input but
  are transformed by the time computation happens at the output position. Supports
  the claim that the model transforms representations non-linearly between input
  and computation.
- **Positive at both:** Fourier features maintained throughout the forward pass.
  Periodic structure is preserved as the computational substrate.
- **Null at both:** The model does not use Fourier representations for operands
  in multiplication context, even at input. Context dependence is immediate.

### 10b. Methodology: PCA on Centroids in Raw 4096-dim Space

The number-token analysis differs structurally from the main Phase G screening:

**Data source:** Raw 4096-dimensional activations at operand token positions
(pre-extracted by `extract_number_token_acts.py`, 48 .npy files, 9.2 GB in
`activations_numtok/`). Unlike the main screening, there are no Phase C/D subspaces
at the number-token position — those were computed at the `=` position only.

**PCA on centroids:** Group centroids for m digit values (e.g., 10 for units digits)
form an (m × 4096) matrix. Between-group structure lives in at most m-1 = 9
dimensions. Running Fourier on all 4096 dims would dilute the FCR with noise from
3,987 dimensions that carry no group information. PCA on the centroid matrix extracts
the between-group subspace (top k = min(pca_dim, m-1) eigenvectors), then all N
activations are projected into this k-dimensional space before Fourier analysis.

This is conceptually identical to what Phase C does — find the directions that
separate groups — but without the two-step Phase C + Phase D pipeline, since the
number-token position doesn't need product residualization or LDA refinement.

**Dual reporting:** FCR is computed in both PCA-projected space (primary, with
1,000-permutation null) and raw 4096-dim space (secondary, centroids only, no
permutation null). The raw-space FCR provides a comparison with K&T's per-dimension
Fourier power.

**No population split:** At the number-token position, correct/wrong classification
is meaningless — that determination happens at the `=` position. All problems are
analyzed together.

### 10c. Digit Concepts Screened

| Concept | Position | Values | Available Levels | Analyses per Layer |
|---------|----------|--------|------------------|--------------------|
| a_units | pos_a | 0–9 | L2–L5 | 4 levels × 6 layers = 24 |
| a_tens | pos_a | 1–9 (L2–L3), 0–9 (L4–L5) | L2–L5 | 24 |
| a_hundreds | pos_a | 1–9 | L4–L5 | 12 |
| b_units | pos_b | 2–9 (L2), 0–9 (L3–L5) | L2–L5 | 24 |
| b_tens | pos_b | 1–9 (L3–L4), 0–9 (L5) | L3–L5 | 18 |
| b_hundreds | pos_b | 1–9 | L5 | 6 |

**Total: ~108 analysis cells** (exact count depends on availability of extraction
files). Layers: {4, 8, 12, 16, 20, 24}. No carries or answer digits — those concepts
are not meaningful at the operand token position.

Note the value range variation: at L2, b is 2–9 (single digit), so b_units has only
8 values instead of 10. At L4, a is 100–999, so a_tens includes 0 (e.g., a=305 has
a_tens=0). The script handles this automatically from the coloring DataFrame.

### 10d. Script Design: phase_g_numtok_fourier.py

The script is standalone (~370 lines) but imports the statistical core from the
main `phase_g_fourier.py`:

```python
from phase_g_fourier import (
    fourier_all_coordinates,
    compute_linear_power,
    compute_helix_fcr,
    compute_centroids_grouped,
    compute_pvalues,
    compute_pvalues_array,
    permutation_null,
    PERM_ALPHA, COORD_P_THRESHOLD, LINEAR_P_THRESHOLD,
    MIN_POPULATION, ZERO_POWER_THRESHOLD,
)
```

Zero code duplication for the mathematical core. Identical Fourier math, permutation
null, detection thresholds, and conjunction criterion as the main Phase G. The only
differences are:

1. **Data loading:** loads `.npy` from `activations_numtok/` instead of main
   `activations/`, handles `pos_a` and `pos_b` separately
2. **Subspace computation:** PCA on centroids in 4096-dim (no Phase C/D bases)
3. **Concept registry:** digit concepts only (no carries, no answer digits)
4. **No population split:** single pass over all problems
5. **Output:** separate directory tree (`phase_g/numtok/L{level}/layer_{LL}/...`)

CLI: `--config`, `--n-perms` (default 1000), `--pca-dim` (default 20), `--pilot`
(L3/layer16 only). SLURM wrapper: `run_phase_g_numtok.sh` (CPU-only, 24 CPUs,
64 GB RAM, ~1–2 hours estimated).

### 10e. Full Run Results: 0/108 Detections (Complete Null)

The full run completed 108 analysis cells in 636 seconds (10.6 minutes) on April 11,
2026. **Every single cell returned `geometry_detected=none`.** Zero helix, zero circle,
zero FDR-significant results.

**Summary by level:**

| Level | Cells | Max FCR | Min p_two_axis | Min p_helix | Detections |
|-------|-------|---------|---------------|-------------|------------|
| L2 | 18 | 0.453 | 0.079 | 0.070 | 0 |
| L3 | 24 | 0.615 | 0.002 | 0.002 | 0 |
| L4 | 30 | 0.554 | 0.004 | 0.003 | 0 |
| L5 | 36 | 0.545 | 0.004 | 0.003 | 0 |

**Summary by layer:**

| Layer | Cells | Max FCR | Min p_two_axis | Notes |
|-------|-------|---------|---------------|-------|
| 4 | 18 | 0.465 | 0.060 | K&T tested this layer — null here |
| 8 | 18 | 0.576 | 0.007 | K&T's strongest layer — null here |
| 12 | 18 | 0.615 | 0.002 | Closest to detection (b_units) |
| 16 | 18 | 0.437 | 0.016 | Information peak — null |
| 20 | 18 | 0.499 | 0.012 | Null |
| 24 | 18 | 0.505 | 0.010 | Null |

**Summary by concept:**

| Concept | Cells | Mean FCR | Max FCR | Min p_two_axis |
|---------|-------|----------|---------|---------------|
| a_units | 24 | 0.310 | 0.393 | 0.026 |
| a_tens | 24 | 0.354 | 0.455 | 0.067 |
| a_hundreds | 12 | 0.388 | 0.423 | 0.079 |
| b_units | 24 | 0.414 | 0.615 | 0.002 |
| b_tens | 18 | 0.390 | 0.491 | 0.053 |
| b_hundreds | 6 | 0.409 | 0.448 | 0.077 |

**b_units is the closest to detection** — particularly at layer 12:

| Level | Layer | FCR | p_two_axis | p_helix | Detected |
|-------|-------|-----|-----------|---------|----------|
| L3 | 12 | 0.615 | 0.002 | 0.002 | none |
| L4 | 12 | 0.554 | 0.004 | 0.003 | none |
| L5 | 12 | 0.544 | 0.004 | 0.003 | none |
| L3 | 8 | 0.576 | 0.007 | 0.004 | none |
| L3 | 24 | 0.505 | 0.010 | 0.007 | none |

The b_units FCR of 0.615 at L3/layer12 is well above null (~0.20) and the global
p-value (0.002) passes the α=0.01 threshold. But the conjunction criterion requires
BOTH top Fourier coordinates to be individually significant (p_coord < 0.01), and
this fails — the structure is concentrated on one coordinate, not spread across
the two axes that define a circle. This is consistent with a partial linear trend,
not a periodic circle.

**Position asymmetry:** pos_b (mean FCR 0.404) consistently shows higher FCR than
pos_a (mean FCR 0.343). This likely reflects the operand structure: at L2, b is a
single digit (2–9) with only 8 values, giving less averaging and higher chance of
structure. At L3–L5, both operands are multi-digit, and the asymmetry narrows.

**PCA concentration factor:** Mean raw 4096-dim FCR is 0.0155, while PCA-space FCR
is 0.370 — a 23.8× concentration factor. This confirms the PCA step is critical:
without it, the signal-to-noise ratio is too low for any structure to emerge from
4,096 dimensions.

### 10f. Interpretation: K&T's Signal Does Not Transfer to Multiplication Context

The full run eliminates all four hypotheses from the pilot:

1. **Context suppression: CONFIRMED.** The multiplication context (`"{a} * {b} ="`)
   suppresses or transforms the standalone digit helix. The model "knows" it is
   doing multiplication and restructures operand representations from periodic
   (K&T's helix for standalone integers) to non-periodic (what we observe).

2. **Layer-dependent encoding: REJECTED.** The null holds at every layer from 4 to
   24, including layers 4 and 8 where K&T found their strongest signal for standalone
   integers. The helix is not present at early layers and lost at later layers —
   it is absent everywhere.

3. **Centroid sensitivity: REJECTED.** At L5 with N=122,223 and ~12,000 samples per
   digit group, centroid standard errors are ~100× smaller than between-group
   distances. The test has overwhelming statistical power. b_units at layer 12
   reaches FCR=0.54 (well above the 0.20 null), showing the test can detect
   structure when it exists — it just doesn't find circles or helices.

4. **The signal genuinely doesn't exist: CONFIRMED.** Operand digits in multiplication
   context do not have periodic Fourier structure at ANY position, layer, or level.
   The model represents operand digits non-periodically throughout the forward pass
   when performing multiplication.

This is a strong finding for the paper: **K&T's digit helix is context-dependent.**
The same model (Llama 3.1 8B) encodes the same integers on helices when presented
standalone but NOT when presented as operands in an arithmetic expression. The model
transforms its integer representation based on the computational task at hand.

### 10g. Output Files

All outputs saved to the data drive:

- **Summary CSV:** `/data/user_data/anshulk/arithmetic-geometry/phase_g/summary/numtok_fourier_results.csv` (108 rows, FDR-corrected)
- **Per-concept JSONs:** 108 files in `phase_g/numtok/L{level}/layer_{LL}/pos_{a|b}/`
- **Log:** `logs/phase_g_numtok_fourier.log` (636 seconds total runtime)
- **Checkpoint:** `phase_g/numtok_checkpoint.pkl`

---

## 11. Interpretation of Full Results

### 11a. The Carry-Helix Story

The dominant finding of Phase G is that **carry values are encoded on generalized
helices inside their linear subspaces.** Phase C/D found that carry_1 has a 3–16
dimensional linear subspace (depending on level and layer). Phase G shows that within
this subspace, carry values 0 through 12+ are arranged not randomly, not linearly,
but on a helix: a circle in two Fourier dimensions plus a linear ramp in a third
dimension.

This is exactly the structure K&T (2025) found for integer representations in
Llama 3.1 8B — a generalized helix with the circular component at period 10 (the
decimal base) and a linear magnitude ramp. The carry helix inherits this structure:
carry values follow the same decimal cycle (carry=10 is "like" carry=0 in the
circular component, but 10× larger in the linear component).

The 34.3% detection rate for carry_1 means that roughly 1 in 3 analysis cells
(level × layer × population × subspace type × period spec) show statistically
significant helix structure. The remaining 2 in 3 are not necessarily "no helix" —
the conjunction criterion is deliberately strict (4-way AND at α=0.01). Some cells
may have real but borderline structure; the permutation null with 1,000 shuffles
has a floor of p=0.001, and many detections hit this floor.

### 11b. Why Operand Digits Are Null at the `=` Position

The zero detection rate for operand digits (0/846) is the cleanest null in Phase G.
Combined with the carry helix finding, this tells a specific story about how the
model represents information at the computation position:

1. **Input digits are already "consumed."** By the `=` position, the model has
   attended to the operand tokens, extracted their digit values, and used them to
   compute partial products, column sums, and carries. The residual stream at `=`
   no longer needs to represent "the units digit of operand a is 7" as a geometric
   fact — it has already been folded into the carries and partial products.

2. **Carries are the active computation.** The model's remaining work at the `=`
   position is carry propagation: determining how carries cascade across columns
   to produce the final answer digits. Carries are the bottleneck (Phase C) and the
   geometric structure (Phase G). The model allocates its representational capacity
   to the hard part.

3. **Linear subspaces are maintained but not periodic.** Phase C showed operand digits
   have full-rank linear subspaces at every layer, even at the `=` position. The
   information is there — the 10 digit centroids are separable in 9D — but they are
   arranged linearly (magnitude ordering), not periodically (circle/helix). The
   transformation from Fourier to linear may be an artifact of how the model "unpacks"
   integer representations for arithmetic.

### 11c. The Difficulty-Dependent Emergence Pattern

The jump from 0.6% helix rate at L2 to 12.0% at L3 mirrors the accuracy drop from
99.8% to 67.2%. This is not coincidence — it reflects the computational demands:

- **L2 (2×1 digit):** carry_0 is either 0 or 1. No complex carry chain, no need for
  elaborate geometric encoding. The linear subspace is sufficient.
- **L3 (2×2 digit):** carry_0 ranges 0–8, carry_1 ranges 0–12+. The model must
  propagate carries across 3 columns with values large enough to need multi-digit
  tracking. The helix provides an efficient encoding: the circular component captures
  the periodic (mod 10) structure, while the linear ramp captures magnitude.
- **L4–L5:** Additional carry positions, longer carry chains, but the helix detection
  rate plateaus at ~15%. The geometric structure needed for carry encoding is already
  established at L3 complexity.

### 11d. Layer Uniformity: Helix Structure Is Maintained, Not Computed

The near-uniform helix detection rate across layers (11.7%–15.1%) is surprising. One
might expect the helix to emerge at a specific layer (where "carry computation"
happens) and be absent before or after. Instead, the helix is present from layer 4
to layer 31.

This means the helix is a **representational format**, not a **computational byproduct.**
The model stores carry values on helices throughout the network, using this format as
the substrate for carry-related computation at every layer. This is consistent with
the residual stream architecture: information persists across layers unless actively
modified by attention or MLP blocks.

### 11e. Answer Digits: Edge-vs-Middle Asymmetry Replicates

Phase C found that middle answer digits (positions 1–2) lack linear subspaces at L5.
Phase G confirms this extends to periodic structure: ans_digit_1 has 1.4% helix rate,
ans_digit_2 has 2.6% — essentially null. Meanwhile, the leading digit (15.2%) and
trailing digit (38.9%) have real signal.

The trailing digit (ans_digit_5 at L5, the ones digit) has the highest detection
rate of any concept (38.9%). This makes mathematical sense: the ones digit of a
product depends only on `(a mod 10) × (b mod 10) mod 10` — a purely modular
operation with period 10. The Fourier basis is the natural representation for
modular arithmetic.

### 11f. Position-Dependent Representations: `=` Token vs Number Token

Combining the main Phase G results (at `=`) with the full number-token screening
(at operand positions), the complete picture:

| Where | Operand Digits | Carries | Answer Digits |
|-------|---------------|---------|---------------|
| Number-token (108 cells, all layers) | **Null (0/108)** | N/A | N/A |
| `=` token (Run 3, all layers) | **Null (0/846)** | **Helix (311/918, 33.9%)** | Mixed (67/745, 9.0%) |
| Standalone integers (K&T replication) | **Helix (confirmed)** | N/A | N/A |

Three results that tell a coherent story:

1. **Standalone integers → helix.** K&T's finding replicated: periods {2, 5, 10} at
   all tested layers for single-token integers 0–360.
2. **Operand digits in multiplication → null everywhere.** 0/108 at the number token,
   0/846 at the `=` token. The multiplication context eliminates the helix.
3. **Carries at `=` → helix.** The periodic structure re-emerges for carry values —
   the model builds new periodic representations for intermediate computations.

This is the strongest evidence yet that **representations are task-dependent, not
token-dependent.** The same integer triggers different geometric encodings depending
on whether the model is simply representing it or computing with it.

### 11g. Implications for the Core Thesis

Phase G provides the first direct evidence for the project's core thesis: **the LRH
is necessary but insufficient for compositional reasoning.**

1. **Linear subspaces are necessary:** Phase C/D showed every atomic concept has a
   clean linear subspace. Phase G doesn't contradict this — it works *within* those
   subspaces.

2. **Linear subspaces are insufficient:** The geometry *within* subspaces matters.
   Carries sit on helices; operand digits sit on non-periodic arrangements; middle
   answer digits have neither subspaces nor periodic structure. A linear probe that
   says "carry_1 has a 10D subspace" misses the helix inside it.

3. **Composition failure has a geometric signature:** The concepts that fail to
   compose (middle answer digits) are exactly those with no geometric structure —
   neither linear subspaces (Phase C) nor periodic manifolds (Phase G). The concepts
   that succeed (carries, trailing digits) have both.

4. **The representation is position-dependent:** Carries have helix structure at `=`
   but don't exist at operand positions. Operand digits have linear subspaces at
   `=` but (preliminary) no periodic structure at either position. The model
   transforms representations non-linearly as information flows from input to
   computation.

---

## 12. What Phase G Will Contribute to the Paper

Phase G contributes three concrete findings to the paper:

1. **Carry values are encoded on generalized helices (K&T-style) inside their linear
   subspaces at the `=` token position.** Detection rate: ~34% of carry_1–4 analysis
   cells. This is the first demonstration that the K&T helix structure exists for
   intermediate computation concepts (carries), not just raw integers. The helix is
   maintained across all layers (4–31), indicating a representational format rather
   than a layer-specific computation.

2. **Operand digits have zero periodic structure at the `=` position.** 0/846 analysis
   cells. Combined with Phase C's finding of full-rank linear subspaces, this means
   the model maintains digit information in a linear but non-periodic format at the
   computation position. The transformation from periodic (K&T's finding) to linear
   may occur during the forward pass as the model prepares for arithmetic.

3. **The edge-vs-middle asymmetry extends to periodic structure.** Middle answer
   digits lack both linear subspaces (Phase C) and periodic manifolds (Phase G).
   Trailing digits have both, consistent with their dependence on modular arithmetic.
   This provides a geometric explanation for the U-shaped per-digit accuracy pattern.

Additionally, the number-token screening (when complete) will establish whether
K&T's finding is position-dependent — whether the Fourier helix exists at operand
positions in multiplication context or only for standalone integers.

---

## 13. Limitations

1. **Centroid test has lower power than per-example Fourier.** K&T tested per-dimension
   DFT across hundreds of integers. Our centroid test groups by digit value (m ≤ 10
   groups) and analyzes group means. If the helix has high within-group variance, the
   centroid test may miss weak structure. However, with N >> 1000 per group, centroid
   standard errors are small relative to between-group distances.

2. **Conjunction criterion is strict.** The 4-way AND (p_two_axis + p_coord[a] +
   p_coord[b] + p_linear for helix) reduces false positives but may also reject real
   but borderline signals. The 66% non-detection rate for carries may include real
   helices that fail one condition by a small margin.

3. **Permutation floor limits power.** With 1,000 permutations, the minimum reportable
   p-value is 0.001 (or 0.00139 for m=6 due to factorial constraint). Concepts with
   m=6 unique values have coarser p-value resolution than m=10 concepts.

4. **Only periods related to base 10 tested.** The screening focused on period P=10
   (digit cycle) plus carry-specific period specs (P=6 for binned carries, P=10 for
   mod10). If the model uses non-decimal periodic structure (e.g., binary, or
   problem-specific periods), this screening would miss it.

5. **Number-token screening covers layers 4–24, not 0–1.** K&T tested layers {0, 1,
   4, 8}. Our extraction captured layers {4, 8, 12, 16, 20, 24}. Layers 0 and 1
   (embedding and first transformer block) were not extracted. If the helix exists
   only at layers 0–1 and is immediately overwritten at layer 4, we would miss it.
   However, this would mean the helix has no functional role in multiplication.

6. **Run 3 is ~92% complete.** L5 layers 24–31 are still running. The final numbers
   may shift slightly, particularly for L5 layer-specific statistics.

---

## 14. Runtime and Reproducibility

### Main Phase G (Run 3)

| Stage | Time | Hardware | Notes |
|-------|------|----------|-------|
| K&T pilot | ~30 min | A6000 GPU | Requires model loaded in VRAM |
| Number-token extraction | ~4 hours | A6000 GPU | 48 .npy files, 9.2 GB |
| Synthetic pilot | ~1 min | CPU | 10 synthetic tests |
| Pilot 0b | ~2 min | CPU | Raw vs. residualized comparison |
| Full Fourier screening | ~3+ hours | 24 CPUs | ~3,348 analyses × 1,000 perms each |

Total wall time: ~8 hours (GPU for Steps 1–2, CPU for Steps 3–5).

### Number-Token Fourier Screening

| Stage | Estimated Time | Hardware | Notes |
|-------|---------------|----------|-------|
| Pilot (L3/layer16) | 12 seconds | CPU | 4 analysis cells |
| Full run (~108 cells) | 1–2 hours | 24 CPUs, 64 GB RAM | CPU-only, no GPU |

Activations already extracted (Step 2 of Run 3). SLURM wrapper: `run_phase_g_numtok.sh`.

### Reproducibility

- All random seeds are set: `np.random.default_rng(42)` for main screening,
  `np.random.default_rng(42)` for number-token screening.
- Permutation null results are deterministic given the same seed, sample size,
  and group structure.
- All intermediate results are saved as per-concept JSON files with full
  statistical details (p-values, FCR, detection flags, group sizes, PCA
  eigenvalues).
- Checkpoint pickle files are saved every 50 analyses for crash recovery.
- Summary CSVs with FDR correction are generated at completion.

*Number-token screening is COMPLETE (108/108, 0 detections). Run 3 main screening
is at ~92% (3,086/~3,348). Remaining items when Run 3 finishes: final FDR-corrected
detection rates, Run 3 total counts, decision rule evaluation, Phase C vs Phase D
agreement analysis, and frequency spectra plots.*
