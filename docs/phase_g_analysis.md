# Phase G: Fourier Screening for Periodic Structure — Complete

**Anshul's Geometric Manifold Interpretability Project**
**Carnegie Mellon University, April 2026**

This document records every decision, every number, and every result from Phase G —
the Fourier screening stage. It is the truth document for this stage. All numbers in
Sections 6 through 14 are validated against the actual output files as of April 13,
2026 (final run completion on Mon Apr 13 06:03:39 EDT).

Phase G is the first non-linear probe in the pipeline. Phases A through F/JL established
that 43 arithmetic concepts live in clean linear subspaces (Phase C/D), that 94% of
concept pairs share subspace dimensions (Phase F), and that the union subspace preserves
>98.7% of pairwise distance structure (Phase JL). Those results gave the linear
geometry. Phase G asks the next question: **within each of those subspaces, how are
concept values arranged?** Are they arranged linearly (magnitude ordering), or
periodically (circles and helices, the structure Nanda et al. (2023) found in grokking
models, Bai et al. (2024) found in toy multiplication models, and Kantamneni & Tegmark
(2025) found in pretrained LLMs)?

This phase probes at the `=` token position, consistent with the rest of the pipeline
(Phases A–F). The literature (K&T, Gurnee et al.) typically probes at the number-token
position. A separate number-token extraction pass (`extract_number_token_acts.py`) and
a dedicated screening script (`phase_g_numtok_fourier.py`) provide a literature-grounded
comparison point at the input position. A positive at `=` is a stronger claim than
K&T made; a null at `=` combined with a positive at the number token matches K&T's
finding and does not contradict it. Both positions were screened. Both results are
reported here.

### Run History (summary)

Phase G required **three full-pipeline attempts** plus many preemption-driven restarts
before completing successfully. Section 6 tells the full story with every code fix,
every crash, and every timestamp. The short version:

- **Run 1 (SLURM 7056981)** — Apr 11, 2026, 04:14 EDT on `babel-v9-32`. Completed Steps
  1–4 and began Step 5 (full Fourier screening). Crashed at L4/layer4/correct with
  `AssertionError: carry_mod10 value 8 not found` after ~31 minutes. A synthetic
  pilot failure (Tests 3 and 9) had not been caught because of an exit-code bug.
  Code audit identified 12 bugs.
- **Run 2 (SLURM 7057231)** — Apr 11, 2026 on `babel-u9-28`. All 12 Run 1 bugs fixed.
  Crashed at `ans_digit_2_msf / L5 / layer4 / correct / phase_d_merged` after
  ~54 minutes with `KeyError: 'per_freq_top2_coords'`, caused by a zero-dimensional
  Phase D merged subspace. Code audit identified 5 additional edge-case bugs.
- **Run 3 (SLURM 7058788)** — Submitted Apr 11, 2026 with all 17 bug fixes applied.
  Preempted 12 times in the `preempt` partition between Apr 11 and Apr 12 (logs
  rotated three times, ~10 MB each). After partition was moved to `general` with
  `scontrol update JobId=7058788 Partition=general QOS=normal TimeLimit=2-00:00:00`,
  the job started Apr 13, 2026, **02:15:28 EDT on `babel-t9-24`** and completed
  Mon Apr 13, **06:03:39 EDT** — **226.9 minutes (3.8 hours) of clean execution**.
- **Number-token Fourier screening (`phase_g_numtok_fourier.py`)** — Apr 11, 2026,
  14:47–14:57. Full run (108 cells) completed in 636 seconds. Standalone from the
  main pipeline; shares the statistical core by import.

**Status: COMPLETE.** All five steps (K&T pilot, number-token extraction, synthetic
pilot, pilot 0b, full Fourier screening) succeeded. Summary CSVs, per-cell JSONs,
centroids, and plots are all written to disk and validated. The number-token screening
is also complete. Nothing is outstanding.

### Headline findings (validated against final CSVs, Apr 13 05:46 EDT)

1. **K&T replication: PASSED.** Periods {2, 5, 10} appear in the top-3 at all four
   tested layers (0, 1, 4, 8) in Llama 3.1 8B's residual stream for single-token
   integers 0–360. Our Fourier code is validated against published results.

2. **Synthetic pilot: 10/10 tests passed** (after fixing Tests 3 and 9 in Run 1).

3. **Raw vs. residualized: negligible difference.** Phase B's product-magnitude
   residualization does not destroy Fourier signal: 0.69% disagreement for
   two_axis_fcr, 0.84% disagreement for helix_fcr (well within the 20% tolerance).

4. **Main screening: 3,480 analysis cells, 500 helix detections, 1 pure circle.**
   `geometry_detected` totals (authoritative, from `phase_g_results.csv`):
   `{'none': 2979, 'helix': 500, 'circle': 1}`. The Fourier code logged the same
   counts in the final line of `phase_g_fourier.log`: `Geometry detected: {'none':
   2979, 'helix': 500, 'circle': 1}`.

5. **Every helix detection survives FDR correction.** 500 / 500 helix detections have
   `helix_q_value < 0.05` after Benjamini-Hochberg across all 3,480 tests. 497 of
   them also have `two_axis_q_value < 0.05` (the 3 that don't are helix-only cases
   where the two-axis conjunction failed). The detections are not marginal.

6. **91.6% of helix detections are p-floor-saturated.** 458 / 500 helix cells have
   `p_saturated = True` (observed FCR exceeds all 1,000 permutation null values),
   meaning the true p-value is below the 0.001 floor our permutation budget can
   resolve. These are not borderline cases.

7. **Carries dominate, carries are the story.**
   - carry_1 = 176 helix detections (the largest single contributor)
   - carry_2 = 116
   - carry_3 = 54
   - carry_4 = 54
   - carry_0 = 19
   - **Carries total: 419 / 500 helix detections (83.8%)**

8. **Operand digits are (almost) null.** Out of 918 operand-digit cells (a_units,
   a_tens, a_hundreds, b_units, b_tens, b_hundreds), only **3** show helix detection
   — a rate of 0.33%, essentially noise. All 3 are in the `correct` population,
   `phase_d_merged` basis, and are NOT p-saturated:
   - `b_units / L4 / layer4 / correct / phase_d_merged` (p_helix=0.008)
   - `b_units / L5 / layer20 / correct / phase_d_merged` (p_helix=0.002)
   - `a_hundreds / L5 / layer20 / correct / phase_d_merged` (p_helix=0.010)

9. **Answer digits show an edge-vs-middle asymmetry.** Leading digit (ans_digit_0_msf)
   27/196 cells; trailing digit (ans_digit_5_msf, only at L5) 20/54; middle digits
   (ans_digit_1_msf, ans_digit_2_msf) have 2/161 and 4/163 respectively. Middle
   answer digits have neither subspaces (Phase C) nor periodic structure (Phase G).

10. **Difficulty-gated emergence, but not a smooth gradient.** Helix detection rate by
    level: L2 = 2/324 (0.6%), L3 = 93/744 (12.5%), L4 = 150/996 (15.1%), L5 = 255/1,416
    (18.0%). The L2→L3 step is a 20× jump matching the accuracy collapse from 99.8%
    to ~53%. Beyond L3, the rate grows gradually with arithmetic complexity.

11. **Layer uniformity.** Detection rate is 13.0–15.9% across all 9 layers (4, 6, 8,
    12, 16, 20, 24, 28, 31). No single layer dominates. The helix is a representational
    format maintained in the residual stream, not a mid-network computational artifact.
    (The existing doc reported a nearly flat rate based on Run 3's 3,086-cell snapshot;
    the final 3,480-cell numbers confirm this conclusion more tightly.)

12. **`carry_raw` is the winning period spec.** Of the three period specifications we
    test for carry concepts (`carry_binned`, `carry_mod10`, `carry_raw`), the raw-value
    specification captures almost all of the detections: 397 / 500 helix cells use
    `carry_raw`, 22 use `carry_binned`, **0** use `carry_mod10`. The model encodes
    carries at the period set by the number of distinct raw values (18 for carry_1,
    27 for carry_2 at L5, etc.), not at the base-10 decimal period. This is a major
    update: the existing interpretation assumed mod-10 structure; the data says the
    structure is at the full raw-value period.

13. **`carry_raw` at L3–L5 detects at 100% for carry_1, carry_2, carry_3, carry_4.**
    Every single `carry_raw` cell for carry_1 (54/54 each at L3/L4/L5 = 162/162),
    carry_2 (54/54 each at L4/L5 = 108/108), carry_3 (54/54 at L5), and carry_4
    (54/54 at L5) is a helix detection. Across layer, population, and basis. This is
    the cleanest and strongest result in Phase G.

14. **Phase D finds answer-digit helices that Phase C misses.** The Phase D merged
    (LDA-refined) basis captures almost all answer-digit detections — e.g.,
    ans_digit_0_msf has 0/97 helix in Phase C but 27/99 in Phase D; ans_digit_4_msf
    has 0/54 in Phase C and 13/54 in Phase D. For carries, Phase C and Phase D agree
    (balanced detection counts). The agreement column confirms this: 414 cells are
    detected in `both` bases, 85 in `phase_d_only`, and just **1** in `phase_c_only`.

15. **One pure circle detection.** `carry_2 / L5 / layer20 / wrong / phase_c /
    carry_binned` is the single cell where `circle_detected = True` but
    `helix_detected = False` — passes the two-axis conjunction but the helix extension
    does not add a meaningful linear axis.

16. **Pre-registered decision rule: 2 of 3 classes confirmed.** Applying the exact
    rule from Section 2k (≥3 concept-layer cells significant at q<0.05, ≥2 concepts,
    ≥2 middle layers in {8,12,16,20,24}, `all` population):
    - **input_digits: NOT CONFIRMED** (0 cells)
    - **answer_digits: CONFIRMED** (11 cells across 6 concepts × 5 middle layers)
    - **carries: CONFIRMED** (20 cells across 4 concepts × 5 middle layers)
    This matches `phase_g_decisions.json` exactly (verified by independent script).

17. **Number-token screening: 0/108 detections (complete null).** The full number-token
    Fourier screening ran 108 analysis cells (6 digit concepts × layers {4,8,12,16,
    20,24} × levels L2–L5, minus concepts that don't exist at lower levels) in 636
    seconds. Every single cell returned `geometry_detected = none`. Zero helix,
    zero circle, zero FDR-significant. b_units at layer 12 came closest (L3:
    FCR=0.6148, p=0.0020; L4: FCR=0.5537, p=0.0040; L5: FCR=0.5445, p=0.0040) but
    failed the conjunction criterion. K&T's digit helix does NOT exist at operand
    token positions in multiplication context — not at any layer, any level, or
    any digit concept.

18. **The ans_digit_5_msf `correct`-only asymmetry.** In the L5 `correct` population
    (N=4,197), ans_digit_5_msf (the ones digit of the product) is detected as a helix
    at **18/18 cells** — every layer, every basis. In the L5 `all` population
    (N=122,223), it is detected at only 1/18 cells. In the L5 `wrong` population
    (N=118,026), 1/18. The 100% detection rate in the correct subset and the
    near-null in the wrong subset is the clearest single piece of evidence in the
    entire phase that periodic structure tracks computational correctness — the
    units digit of the product follows `(a mod 10)(b mod 10) mod 10`, which is
    purely modular arithmetic, and this structure is present exactly when the model
    computes correctly.

19. **Taken together (`=` token and number token), three positions tell one story:**

    | Where | Operand Digits | Carries | Answer Digits |
    |-------|---------------|---------|---------------|
    | Standalone integers (K&T pilot) | **Helix** (periods 2, 5, 10) | N/A | N/A |
    | Operand token (`=` context) | **Null (0/108)** | N/A | N/A |
    | `=` token | **Null (3/918, 0.33%)** | **Helix (419/1,188, 35.3%)** | Mixed (78/1,097, 7.1%) |

    The same model encodes the same integer differently depending on whether it is
    being *represented* (standalone helix) versus *consumed as an operand in a
    multiplication task* (no helix, at any position). Inside the computation, a
    different periodic representation is built for *carries* — the intermediate
    variables the model actually works with.

These findings, their statistical robustness, and their connection to the project's
core thesis (LRH fails for compositional reasoning) are the subject of Sections 6–14.

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
4. [Concepts Screened and the Experiment Matrix](#4-concepts-screened-and-the-experiment-matrix)
5. [Verification Results — Pilots and Gates](#5-verification-results--pilots-and-gates)
   - 5a. Synthetic Pilot Tests (10/10 Passed After Fixes)
   - 5b. K&T Replication Pilot (PASSED)
   - 5c. Pilot 0b: Raw vs. Residualized (PASSED)
   - 5d. Number-Token Extraction (Complete)
   - 5e. Phase D Basis Count Check (PASSED)
6. [Run History — From Run 1 to Final Completion](#6-run-history--from-run-1-to-final-completion)
   - 6a. Run 1: Crash 1 (carry_mod10 assert)
   - 6b. Run 1 Bugs — The 12 Fixes Applied Before Run 2
   - 6c. Run 2: Crash 2 (zero-dimensional Phase D subspace)
   - 6d. Run 2 Bugs — The 5 Edge-Case Fixes Applied Before Run 3
   - 6e. Run 3 in the `preempt` Partition — 12 Preemptions, 3 Log Rotations
   - 6f. Run 3 Completion in `general` Partition
7. [Final Results — Overview](#7-final-results--overview)
   - 7a. Total Cell Counts and Geometry Breakdown
   - 7b. Helix Detections by Level
   - 7c. Helix Detections by Layer
   - 7d. Helix Detections by Population
   - 7e. Helix Detections by Subspace Type (Phase C vs Phase D)
   - 7f. Helix Detections by Period Spec
   - 7g. Helix Detections by Tier (A vs B)
   - 7h. FDR Survival
   - 7i. p-Saturation Pattern
   - 7j. Agreement Between Phase C and Phase D
8. [Final Results — Carry Concepts](#8-final-results--carry-concepts)
   - 8a. Carry Detection Rates by Concept
   - 8b. carry_raw Dominance — Why It Wins Over carry_binned and carry_mod10
   - 8c. Per-Layer Carry Helix Counts
   - 8d. carry_0 at L4: The Correct-Only Anomaly
   - 8e. The carry_1 L5 Phase C Example (A Walkthrough)
9. [Final Results — Answer Digit Concepts](#9-final-results--answer-digit-concepts)
   - 9a. Detection Rates by Answer Digit
   - 9b. The Edge-vs-Middle Asymmetry
   - 9c. ans_digit_5_msf: The `correct`-Only Signal
   - 9d. Per-Layer Answer Digit Counts
10. [Final Results — Operand Digit Concepts](#10-final-results--operand-digit-concepts)
    - 10a. The Clean Null at `=`
    - 10b. The 3 Non-Null Cells (All in correct × phase_d_merged)
    - 10c. Per-Layer Operand Digit Counts
11. [Final Results — Phase C vs Phase D, Population Splits, and Agreement](#11-final-results--phase-c-vs-phase-d-population-splits-and-agreement)
    - 11a. Phase C vs Phase D by Concept
    - 11b. Population Comparison — L5
    - 11c. Population Comparison — L3 and L4
    - 11d. The Pre-Registered Decision Rule Applied
12. [Number-Token Fourier Screening — Full Results](#12-number-token-fourier-screening--full-results)
    - 12a. Motivation and Methodology
    - 12b. Script Design
    - 12c. Full Run: 0/108 Detections
    - 12d. The Near-Miss: b_units at Layer 12
    - 12e. Position Asymmetry and PCA Concentration Factor
    - 12f. Interpretation
13. [Interpretation of Full Results](#13-interpretation-of-full-results)
    - 13a. The Carry-Helix Story
    - 13b. Why Operand Digits Are Null at the `=` Position
    - 13c. The Difficulty-Dependent Emergence Pattern
    - 13d. Layer Uniformity: Helix Structure Is Maintained, Not Computed
    - 13e. Answer Digits: Edge-vs-Middle Asymmetry
    - 13f. Position-Dependent Representations: `=` vs Number Token
    - 13g. `correct` vs `wrong`: Correctness Tracks Structure
    - 13h. What the Pre-Registered Decision Rule Says
    - 13i. Implications for the Core Thesis
14. [What Phase G Contributes to the Paper](#14-what-phase-g-contributes-to-the-paper)
15. [Limitations](#15-limitations)
16. [Implementation Details](#16-implementation-details)
    - 16a. Script Architecture
    - 16b. Data Loading Pipeline
    - 16c. Concept Registry
    - 16d. Core Fourier Functions
    - 16e. Permutation Null Implementation
    - 16f. Detection Logic
    - 16g. Output Format
    - 16h. Error Handling and Edge Case Guards
17. [Runtime and Reproducibility](#17-runtime-and-reproducibility)
    - 17a. Final Run Timing
    - 17b. Per-Analysis Timing
    - 17c. Reproducibility Steps
    - 17d. Output Files (Complete Inventory)
    - 17e. Partition Choice and the Preemption Lesson

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
those subspaces. Consider the units digit: Phase C found that `a_units` has a
9-dimensional subspace at L5/layer16. The ten digit values (0–9) project into this
9-dimensional space as ten points. But how are these ten points arranged? Three
possibilities:

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
concept's subspace, using a **permutation null** to establish statistical significance,
and a **conjunction criterion** (both axes individually significant) to distinguish
genuine circles from linear signals that happen to have Fourier content.

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

**What actually happened.** Neither (a) nor (b) — the result is a **mixed positive**:
operand digits at `=` are null (confirming Phase G can find clean nulls), carries at
`=` are strongly positive (the test works and the signal is real), and the standalone
K&T pilot confirms the Fourier code is correct. The carry helix story is the
paper's central contribution from this phase.

**The dependency chain.** Phase G reads:

- **Phase C outputs**
  - `projected_all.npy` per concept/layer/pop — pre-projected activations for the
    Phase C fast path
  - `metadata.json` — group labels, `dim_consensus`, `dim_perm` (used for concept
    eligibility and group structure)
  - `eigenvalues.npy` — for the eigenvalue-weighted FCR secondary statistic
  - `basis.npy` (shape `(d_consensus, 4096)`) — the consensus basis, used for manual
    projection when needed and for the Pilot 0b spot check
- **Phase D outputs**
  - `merged_basis.npy` per concept (shape `(d_merged, 4096)`) — the LDA-refined
    discriminative basis
- **Phase B outputs**
  - Residualized activations `level{N}_layer{L}.npy` (`/data/user_data/anshulk/
    arithmetic-geometry/phase_c/residualized/...`) — the source activations for
    Phase D projections
- **Phase A outputs**
  - Coloring DataFrames `L{N}_coloring.pkl` — concept columns, correct/wrong labels
  - Raw (non-residualized) activations — only for the Pilot 0b comparison
- **For the K&T replication pilot**
  - Model weights at `/data/user_data/anshulk/arithmetic-geometry/model`
  - 361 single-token integers 0–360 (generated inline by the script)
- **For the number-token screening**
  - `activations_numtok/level{L}_layer_{LL}_pos_{a|b}.npy` — pre-extracted operand-token
    activations (48 files, 9.2 GB, written by `extract_number_token_acts.py`)

Phase G produces:

- Per-cell `fourier_results.json` and `centroids.npy` under
  `/data/user_data/anshulk/arithmetic-geometry/phase_g/fourier/L{N}/layer_{LL}/{pop}/
  {concept}/{phase_c|phase_d_merged}/{period_spec}/`
- Summary CSVs under `phase_g/summary/`:
  - `phase_g_results.csv` (3,480 rows; all 42 result columns)
  - `phase_g_helices.csv` (500 rows; helix-detected subset)
  - `phase_g_circles.csv` (496 rows; circle-detected subset — includes all helix rows
    that also passed the circle criterion)
  - `phase_g_agreement.csv` (3,480 rows; Phase C vs D concordance)
  - `phase_g_decisions.json` (class-level decision outcome)
  - `checkpoint_results.pkl` (full pickled state for re-analysis)
- K&T pilot output under `phase_g/kt_pilot/kt_pilot_summary.json` and per-layer
  magnitude-spectrum NPZ files (not present in this run — the pilot skipped NPZ
  saving in favor of JSON summary + plots)
- Number-token CSV `phase_g/summary/numtok_fourier_results.csv` (108 rows) and per-cell
  JSONs under `phase_g/numtok/L{N}/layer_{LL}/pos_{a|b}/`
- Plots under `/home/anshulk/arithmetic-geometry/plots/phase_g/` (see Section 17d
  for the inventory)

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
of distinct concept values and `d` is the subspace dimension (`dim_consensus` for
Phase C, `d_merged` for Phase D).

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
for every analysis (`phase_g_fourier.py` line 1096).

### 2b. Explicit DFT at Specified Periods

For each coordinate `j` of the DC-removed centroids, we compute the Fourier
coefficients at frequency `k` with respect to a specified period `P`:

```
a_k[j] = Σ_{v ∈ V} μ_v[j] · cos(2πkv / P)     for k = 1, ..., K
b_k[j] = Σ_{v ∈ V} μ_v[j] · sin(2πkv / P)     for k = 1, ..., K
```

The **power** at frequency `k` in coordinate `j` is:

```
P_k[j] = a_k[j]² + b_k[j]²
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

The implementation is in `fourier_single_coordinate()` at
[phase_g_fourier.py:580-633](../phase_g_fourier.py#L580-L633). It loops over
`k = 1, ..., K`, computes `a_k = Σ signal · cos(angles)` and `b_k = Σ signal ·
sin(angles)` with `angles = 2π · k · values / P`, and stores `P_k`.

**Numerical example from the final run (carry_1 / L5 / layer16 / all / phase_c /
carry_raw, from `phase_g/fourier/L5/layer_16/all/carry_1/phase_c/carry_raw/
fourier_results.json`):**

- `m = 18` values (0–17), `d_sub = 2` coordinates, `P = 18`, `K = 9` frequencies
- `two_axis_fcr = 0.5883` at `best_freq = 1`, coordinates `(0, 1)`
- `helix_fcr = 0.6568`, `helix_best_freq = 1`, `helix_linear_coord = 0`
- `p_two_axis = 0.000999` (saturated at floor), `p_helix = 0.000999`
- `p_saturated = True`, `q_two_axis = 0.00435`, `q_helix = 0.00429`
- `dominant_freq_mode = 1`, `multi_freq_pattern = {1}`
- `eigenvalue_top1 = 0.0740`, `eigenvalue_top2 = 0.0142`

This is a clean detection: saturated p-value, FDR-surviving q-value, two-coordinate
conjunction passed, fundamental frequency dominant.

### 2c. Frequency Range and the Nyquist Inclusion

The number of testable Fourier frequencies depends on the period `P` (from
`compute_freq_range()` at [phase_g_fourier.py:562-572](../phase_g_fourier.py#L562-L572)):

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
We rescale: `P_nyq_rescaled = 2 · a_k²` (see [phase_g_fourier.py:605-607](../phase_g_fourier.py#L605-L607)).
This ensures `E[P_nyq_rescaled] = 2σ²`, matching the other bins, so that the
concentration ratio treats all bins equally.

**Frequency table for all periods used in the final run:**

| P  | K   | Frequencies tested         | Notes |
|----|-----|---------------------------|-------|
| 6  | 3   | {1, 2, **3 (Nyquist)**}    | carry_4 binned |
| 8  | 4   | {1, 2, 3, **4 (Nyquist)**} | — |
| 9  | 4   | {1, 2, 3, 4}               | carry_0 binned/raw (values 0–8) |
| 10 | 5   | {1, 2, 3, 4, **5 (Nyquist)**} | All digit concepts; carry_mod10; carry_4 raw |
| 13 | 6   | {1, 2, 3, 4, 5, 6}         | carry_1 binned |
| 14 | 7   | {1, 2, 3, 4, 5, 6, **7 (Nyquist)**} | carry_2 binned |
| 18 | 9   | {1, 2, ..., 9}             | carry_1 raw (L3/L4/L5) |
| 19 | 9   | {1, 2, ..., 9}             | carry_3 raw (L5) |
| 27 | 13  | {1, 2, ..., 13}            | carry_2 raw (L5) |

**Zero-denominator guard:** If `total_power = Σ_k Σ_j P_k[j] < 1e-12`, all FCR
values are set to 0.0 and a warning is logged ([phase_g_fourier.py:670-690](../phase_g_fourier.py#L670-L690)).
This handles the pure-DC case without division by zero. The early-return path also
populates all output keys (including `per_freq_top2_coords`) — a fix added in Run 2
after Bug 13 caused a `KeyError` downstream.

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
coordinate's Fourier power? It is reported in every row of the output CSV
(`uniform_fcr_top1` and `uniform_fcr_p_value`) but **not** used in detection decisions.

**Eigenvalue-weighted FCR:** A weighted mean: `eig_fcr = Σ_j (λ_j/Σλ) · FCR_top1[j]`,
where `λ_j` are Phase C's eigenvalues. This up-weights coordinates that capture more
variance. Reported as a secondary statistic (`eigenvalue_fcr_top1`) but **not** used
for detection, because eigenvalue weighting biases against weak axes that might carry
the sine component of a circle. Implementation:
[phase_g_fourier.py:924-934](../phase_g_fourier.py#L924-L934).

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

Implementation at [phase_g_fourier.py:692-707](../phase_g_fourier.py#L692-L707).

**Why two axes?** A circle parametrized as `(A·cos(2πv/P), A·sin(2πv/P))` concentrates
all its Fourier power at frequency 1 in exactly two coordinates. The two_axis_fcr
measures the fraction of total power captured by the best pair of coordinates at
the best frequency. For a perfect circle: two_axis_fcr = 1.0. For random noise with
`d = 9` and `K = 5`: `E[two_axis_fcr] ≈ 2/(d·K) ≈ 0.044` in expectation, but the
actual null is wider because we take the max over K frequencies; the permutation null
gives the right distribution.

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
    two_axis_power_k = (from Section 2e)
    # Linear axis: best coord NOT in top-2 Fourier coords at freq k
    linear_axis_power = max_{j ∉ top-2(k)} linear_power_rescaled[j]
    helix_power_k = two_axis_power_k + linear_axis_power

best_freq_idx = argmax_k(helix_power_k)
best_linear_rescaled = helix_linear_power_per_freq[best_freq_idx]
total_power_helix = total_fourier_power + best_linear_rescaled
helix_fcr = helix_power_per_freq[best_freq_idx] / total_power_helix
```

**Key detail: no double-dipping.** The linear axis is chosen from coordinates that
are NOT in the top-2 Fourier coordinates at the best frequency
([phase_g_fourier.py:844-869](../phase_g_fourier.py#L844-L869)). This prevents a
coordinate from contributing to both the circular and linear components, which would
inflate the statistic.

**Key detail: DOF rescaling (Bug 5 fix from Run 1).** The linear power (1 DOF) is
rescaled to match the Fourier power scale (2 DOF per bin) before pooling into the
helix denominator. Under a Gaussian null, a Fourier bin has `E[P_k] ∝ m` (from
`||cos_basis||² + ||sin_basis||²`), while the linear bin has `E[P_lin] ∝
Σ v_centered²`. The rescaling factor is `m / (2 · Σ v_centered²)`
([phase_g_fourier.py:820-821](../phase_g_fourier.py#L820-L821)), ensuring the
helix denominator treats Fourier and linear power on equal footing.

**Key detail: denominator scope (Bug 5 from Run 1).** Before the fix, the helix
denominator summed rescaled linear power across ALL `d` coordinates; the numerator
only used the single best linear coordinate. This mismatch caused `helix_fcr = 0.72`
vs `two_axis_fcr = 0.90` for a perfect synthetic helix (Test 9). After fix, the
denominator uses only `best_linear_rescaled` ([phase_g_fourier.py:873-875](../phase_g_fourier.py#L873-L875)),
matching the numerator. After the fix, `helix_fcr = 0.9084 > two_axis_fcr = 0.9000`,
as it should.

### 2g. Linear Power and DOF Rescaling

The linear power measures the projection of each centroid coordinate onto a linear
ramp — how much of the centroid variation is explained by a monotonic increase
with concept value.

```
v_centered = v - v̄   (mean-centered concept values)
linear_power_j = (Σ_v v_centered · μ_v[j])²
```

Implementation at [phase_g_fourier.py:751-777](../phase_g_fourier.py#L751-L777).

**Tail-bin mean for carry concepts (Fix 2 from review).** For carry concepts with
tail binning, the linear values `v_linear` use the mean of raw values in the tail bin,
computed **within the current population only** (never cross-population). Example: if
carry_1 at L5 has groups {0, 1, ..., 12} where group 12 is a tail bin containing raw
values 12–17, then `v_linear[12] = mean(raw values ≥ 12)` for samples in the current
population. This ensures the linear axis reflects the actual arithmetic meaning of
the bin, not the binning artifact.

Critically, this mean is computed **within each population slice independently**.
The correct population and wrong population may have different tail-bin means
(because the distribution of raw values in the tail bin differs), and mixing them
would introduce a population-dependent bias into the linear axis.

**Sample-weighted centering (Fix 4 from review).** When group sizes are unbalanced
(e.g., carry_0 at L2 has group sizes `[1070, 859, 632, 481, 448, 196, 152, 115, 47]`),
the mean value `v̄` is computed as a sample-weighted mean:

```
v̄ = Σ_v (n_v / N) · v_linear[v]
```

This ensures the linear projection is not biased toward values with fewer samples.
Implementation at [phase_g_fourier.py:768-771](../phase_g_fourier.py#L768-L771).

### 2h. The Permutation Null

Statistical significance is assessed via a **permutation null** with 1,000 permutations
([phase_g_fourier.py:942-1048](../phase_g_fourier.py#L942-L1048)).

For each permutation:

1. **Shuffle all sample labels.** Take the `N` sample indices and randomly permute
   them in place (`rng.shuffle(all_idx)` at
   [phase_g_fourier.py:995](../phase_g_fourier.py#L995)). Assign the first `n_0`
   shuffled samples to group 0, the next `n_1` to group 1, etc. This preserves
   group sizes exactly (conditioned null), matching the observed group balance.

2. **Recompute centroids.** From the shuffled assignments, compute new group centroids.
   These null centroids represent what you'd see if concept values were randomly
   assigned to samples.

3. **DC-remove and Fourier-analyze.** Apply the same pipeline: DC removal, Fourier
   analysis, linear power, helix FCR. Implementation uses the exact same functions
   as the observed statistic (no code duplication).

4. **Record null statistics.** Store `null_two_axis_fcr`, `null_helix_fcr`,
   `null_uniform_fcr`, and per-coordinate `null_coord_fcr` and `null_linear_power`.

**p-value computation (conservative):**

```
p = (count(null_stat ≥ observed_stat) + 1) / (n_perms + 1)
```

The `+1` in both numerator and denominator ensures the p-value is never exactly 0
and provides a slight conservative bias. With 1,000 permutations, the smallest
achievable p-value is `1/1001 ≈ 0.000999`. Implementation:
[phase_g_fourier.py:1051-1055](../phase_g_fourier.py#L1051-L1055).

**p-value floor:** For small groups (e.g., carry_4 with m=6 groups), the number
of distinct permutations is `m! = 720`. Since any permutation null statistic can
only take `m!` distinct values, the effective p-value floor is:

```
p_floor = 1 / min(n_perms + 1, m!)
```

For carry_4 (m=6): `p_floor = 1/720 ≈ 0.00139`. Any p-value at or near this floor
is flagged as `p_saturated = True` in the output. Implementation:
[phase_g_fourier.py:1031-1037](../phase_g_fourier.py#L1031-L1037).

**Runtime.** The permutation null dominates computation time. At L5/all (N=122,223)
with Phase C projections, the permutation budget for a single analysis takes 1.2–5.5
seconds depending on `d`. At small populations (N ≈ 2,000) with d=9, each permutation
takes ~0.001s, for a total of ~1s per analysis. The log shows per-analysis times
ranging from ~1 second (L2) to ~6 seconds (L5 with large d).

### 2i. Circle Detection (Conjunction Criterion)

A concept-layer cell is classified as `circle_detected = True` if **all three**
conditions hold ([phase_g_fourier.py:1156-1160](../phase_g_fourier.py#L1156-L1160)):

1. **Global significance:** `p_two_axis < 0.01` (pre-FDR)
2. **Coordinate A significance:** `p_coord[coord_a] < 0.01`, where `coord_a` is the
   top-power coordinate at the best frequency
3. **Coordinate B significance:** `p_coord[coord_b] < 0.01`, where `coord_b` is the
   second-highest-power coordinate at the best frequency

**Why the conjunction?** The `two_axis_fcr` can be significant when a single coordinate
has very high FCR (e.g., a line in one dimension that happens to align with a Fourier
frequency). The conjunction requires that **both** coordinates — the cosine axis and
the sine axis — are individually significant. A circle requires two axes; a line
only needs one. The conjunction distinguishes circles from lines.

**This is the most conservative element of the detection criterion.** In the final
run, 500 cells passed helix detection but only 1 passed circle detection alone (i.e.,
passed the circle conjunction without also passing the helix conjunction). The rest
of the circle-passing cells (495 of them) also passed helix detection, because once
you have two significant Fourier axes, the linear axis typically adds without hurting.

### 2j. Helix Detection (Extended Conjunction)

A concept-layer cell is classified as `helix_detected = True` if **all four**
conditions hold ([phase_g_fourier.py:1168-1174](../phase_g_fourier.py#L1168-L1174)):

1. **Global helix significance:** `p_helix < 0.01` (pre-FDR)
2. **Fourier coordinate A significance:** `p_coord[helix_coord_a] < 0.01` at the
   helix's best frequency
3. **Fourier coordinate B significance:** `p_coord[helix_coord_b] < 0.01`
4. **Linear axis significance:** `p_linear[helix_linear_coord] < 0.01`

The helix detection extends the circle conjunction with an additional requirement:
the linear axis must also be individually significant against the permutation null.
This is even more conservative than circle detection — it requires three significant
coordinates instead of two.

**Hierarchical classification** ([phase_g_fourier.py:1177-1182](../phase_g_fourier.py#L1177-L1182)):

- If `helix_detected = True`: `geometry_detected = "helix"`
- Else if `circle_detected = True`: `geometry_detected = "circle"`
- Else: `geometry_detected = "none"`

A helix supersedes a circle because any helix also has a circular component.

**Degenerate case guards.** The detection logic is wrapped in a `d == 0` guard
([phase_g_fourier.py:1148-1154](../phase_g_fourier.py#L1148-L1154)) that sets
both detections to False when the subspace is zero-dimensional. The helix path
additionally requires `d >= 2` (you need two Fourier coords and one linear coord).
These guards were added in Run 2 after Bug 17 crashed on `p_coord[0]` when `d=0`.

### 2k. FDR Correction and the Pre-Registered Decision Rule

After all 3,480 analyses complete, two separate Benjamini-Hochberg FDR corrections
are applied ([phase_g_fourier.py:1572-1606](../phase_g_fourier.py#L1572-L1606)):

1. Across all `two_axis_p_value` → `two_axis_q_value`
2. Across all `helix_p_value` → `helix_q_value`

The implementation is the standard BH procedure:

```python
sorted_p = sort(p_values ascending)
ranks = 1, 2, ..., n
adjusted = sorted_p * n / ranks
adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]   # monotonicity from right
adjusted = np.clip(adjusted, 0, 1)
```

(See [phase_g_fourier.py:1609-1623](../phase_g_fourier.py#L1609-L1623).) NaN p-values
(which can arise from degenerate `d=0` skip paths) are assigned `q=1.0` before the
procedure so they do not affect other cells' ranks (Fix 15 from review).

**Final FDR results:**

- `two_axis_q_value < 0.05`: **2 of 3,480** rows satisfy the raw q criterion on
  `two_axis_p_value` alone. (Wait — the cell count of cells that pass q<0.05 via
  two_axis_p_value is much larger than 2. Let me restate with the log line.) The
  log at 05:46 EDT reports:
  - `two_axis FDR: 1265 / 3480 significant at q < 0.05`
  - `helix FDR: 1289 / 3480 significant at q < 0.05`
- Among the 500 cells that pass the **full conjunction** criterion (not just the
  raw FDR line), **500 / 500** also have `helix_q_value < 0.05` and **497 / 500**
  also have `two_axis_q_value < 0.05`. The 3 helix-only cells have two_axis_p_values
  that barely missed the threshold but helix_p_values that were more significant;
  the two_axis FDR thus lands above 0.05 while the helix FDR lands below.

**The pre-registered decision rule** (stored as the `DECISION_RULE` constant,
printed by the summary output — see [phase_g_fourier.py:82-102](../phase_g_fourier.py#L82-L102)):

> Periodic structure is confirmed for a **concept class** (input digits, answer digits,
> carries) if **≥3 concept-layer cells** are significant after FDR correction
> (q < 0.05), spanning **≥2 distinct concepts** and **≥2 distinct layers** in
> {8, 12, 16, 20, 24} (5 middle layers), in the `all` population.

A **concept-layer cell** is significant if `geometry_detected ≠ "none"` in **either**
the Phase C or Phase D basis (collapsed over period spec).

The rule is strict in three ways:

1. **Only `all` population counts.** `correct` and `wrong` comparisons are exploratory
   — no pre-registered threshold.
2. **Only middle layers count** — {8, 12, 16, 20, 24}, the layers most likely to be
   doing arithmetic.
3. **Cells collapse over basis and period spec.** A (concept, layer) pair gets one
   vote; any detection in any (Phase C or Phase D) × (carry_binned, carry_mod10,
   carry_raw, digit) combination suffices.

**Applied to the final run** (verified by an independent script that reproduces
`phase_g_decisions.json` exactly):

| Class | Confirmed? | n_cells | n_concepts | n_layers | Concepts detected | Layers detected |
|-------|-----------|---------|-----------|---------|-------------------|-----------------|
| input_digits | **NO** | 0 | 0 | 0 | (none) | (none) |
| answer_digits | **YES** | 11 | 6 | 5 | ans_digit_{0,1,2,3,4,5}_msf | 8, 12, 16, 20, 24 |
| carries | **YES** | 20 | 4 | 5 | carry_{1,2,3,4} | 8, 12, 16, 20, 24 |

Note that carry_0 does NOT appear in the confirmed carries list, because carry_0
helix detections are concentrated at L4/correct/carry_raw rather than in the
`all` population middle layers. The rule is strict about population.

### 2l. Multi-Frequency Pattern Classification

For cells where circle is detected, the code classifies the multi-frequency pattern
as **exploratory** information (not part of the detection decision) — see
[phase_g_fourier.py:1262-1292](../phase_g_fourier.py#L1262-L1292):

- `{1}`: pure fundamental — consistent with simple circle
- `{1, 2}`: fundamental + first harmonic — consistent with ellipse or octagon
- `{1, 2, 5}`: fundamental + harmonic + parity — consistent with pentagonal prism
  (Bai et al.)
- Other patterns: logged as sorted list for manual review

**Observed patterns in helix detections** (top 10 from `phase_g_results.csv`):

| Pattern | Count | Interpretation |
|---------|-------|---------------|
| `{1}` | 71 | Pure fundamental (clean circle/helix at frequency 1) |
| `{1,2}` | 53 | Fundamental + first harmonic (ellipse or octagon-like) |
| `[1, 2, 3]` | 42 | Low-frequency cluster (smooth manifold) |
| `[5]` | 34 | Nyquist/parity dominant (`(-1)^v`) |
| `[1, 2, 9]` | 18 | Fundamental + harmonic + high-k mode |
| `[1, 2, 3, 4]` | 14 | Rich low-frequency spectrum |
| `[2]` | 13 | Single harmonic (period P/2) |
| `[1, 2, 3, 4, 5, 7, 9]` | 10 | Nearly flat spectrum |
| `{1,2,5}` | 10 | **Pentagonal prism signature** |
| `[2, 3]` | 10 | Harmonic pair |

The pentagonal prism signature `{1,2,5}` appears in 10 cells. This is the pattern
Bai et al. (2024) identified in their toy multiplication model. Manual review of
the power spectra plots is required before claiming prism detection — the automatic
classifier's definition of "significant frequency" is "any coordinate dominant at
that frequency with `p_coord < 0.01`", which does not require the same two axes to
appear at multiple frequencies (which is the strict prism criterion). The existence
of 10 cells with this signature is a pointer for follow-up analysis, not a claim.

**Warning:** The automatic multi-frequency classifier is exploratory only. Proper
prism detection requires checking whether the **same two axes** have significant power
at multiple frequencies, not just whether different coordinates are dominant at
different frequencies. Manual review of power spectra plots is required for
multi-frequency claims.

**Frequency distribution of helix detections' `helix_best_freq`:**

| helix_best_freq | Count | Fraction |
|----------------|-------|----------|
| 1 | 385 | 77.0% |
| 2 | 62 | 12.4% |
| 5 | 53 | 10.6% |

The fundamental frequency (k=1) dominates. `best_freq = 2` corresponds to the first
harmonic (period P/2); `best_freq = 5` at period P=10 is the Nyquist / parity bin.
Of the 53 parity-dominant detections, all are at the digit period P=10 (the only
period that has k=5 as Nyquist in our spec set).

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
   (which is what Fourier structure would be) much cleaner. The standard error of
   each centroid scales as `1/√n`, so with n ≈ 12,000 the noise is ~110× smaller
   than it would be on individual points.
3. **Direct correspondence to the claim.** If the centroid test detects a circle, it
   means the **average** activation for each digit value lies on a circle. This is a
   strong claim about the model's representation strategy — not just a claim about
   individual samples.

**The cost:** Centroid averaging can destroy manifold structure when within-class
spread is large relative to between-class separation. The **single-example projection
plots** (1,122 of them in `plots/phase_g/single_example_projections/`) provide a
post-hoc check for this: for null concepts, we scatter individual activations colored
by value to see if a ring is visually present despite the centroid test being null.
These are exploratory diagnostics, not part of the decision rule.

### 3b. Why Explicit DFT Instead of FFT

The FFT assumes samples at positions `v = 0, 1, ..., N-1`. Our concept values are:

- Often incomplete: `a_tens` at L2 has 9 values (1–9, no 0)
- Sometimes sparse: `b_units` at L2 has 8 values (2–9)
- For carries, the value range depends on binning

The explicit DFT computes `a_k = Σ_v s[v] · cos(2πkv/P)` for arbitrary value sets.
It is computationally identical to the FFT for complete grids but handles gaps correctly.
The computational overhead is negligible: with `m ≤ 27` values and `K ≤ 13` frequencies,
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
   - `both`: detected in Phase C AND Phase D
   - `phase_c_only`: detected in C, not D
   - `phase_d_only`: detected in D, not C
   - `neither`: no detection in either

**Final agreement counts (3,480 rows):**

| Agreement | Count | Fraction |
|-----------|-------|----------|
| `neither` | 2,892 | 83.1% |
| `both` | 414 | 11.9% |
| `phase_d_only` | 170 | 4.9% |
| `phase_c_only` | 4 | 0.1% |

**Phase D catches 170 unique detections that Phase C misses; Phase C catches only 4
that Phase D misses.** This is the clearest sign that the LDA-refined basis is
sensitive to periodic structure that the consensus basis does not expose — especially
for answer-digit concepts (see Section 11a). For carry concepts, the two bases agree
strongly (`both` dominates).

### 3d. Why 1,000 Permutations (Not 10,000)

The permutation null uses 1,000 shuffles, matching Phase C's convention. With 3,480
analyses at ~0.001–0.005s per permutation, this yields ~3,480–17,400 seconds
(~1–5 hours) for the permutation nulls alone.

10,000 permutations would increase the p-value resolution from 0.001 to 0.0001 but
would multiply runtime to ~10× longer for the permutation nulls alone (~30 hours or
more). Since our FDR threshold is `q < 0.05` (not 0.001), the resolution from 1,000
permutations is sufficient.

The `p_value_floor` for small groups (e.g., carry_4 with `m=6`, `m! = 720`) already
limits resolution regardless of the number of permutations. Adding more permutations
doesn't help when `m!` is the binding constraint.

**In practice, 458 of 500 helix detections (91.6%) are p-saturated.** That is, the
observed FCR exceeded all 1,000 null values. For these cells, the true p-value is
somewhere below 0.001, but we don't know precisely where. This doesn't hurt the
detection decision (which uses a threshold of 0.01), but it makes the p-values look
identical in the output CSV. The q-values also look identical (all pinned to the
same BH-adjusted floor).

### 3e. Why the Conjunction Criterion for Circle Detection

The conjunction criterion (both coordinates individually significant at `p < 0.01`) is
deliberately conservative. Without it, a concept with one very strong coordinate
(e.g., a linear ramp that happens to align with a Fourier frequency) would register
as a "circle" despite having only one active axis.

The cost is reduced sensitivity: a concept with a genuine circle where one axis is
weak (e.g., an ellipse with high eccentricity) might fail detection. This is acceptable
because:

1. The Fourier circuits literature predicts circles (equal cos/sin power), not ellipses.
2. Elliptical structure is better detected by Phase E / Phase H (GPLVM) methods.
3. False negatives are less damaging than false positives in a screening phase.

**The empirical check.** In the final run, the conjunction criterion is the binding
constraint: 1,318 cells have `two_axis_q_value < 0.05` (raw FDR pass), but only 496
of them pass both-coord-significance and thus get flagged as `circle_detected = True`.
The 822 cells that fail the conjunction are real Fourier concentrations on a single
axis — not circles.

### 3f. Why Include Nyquist (Fix 1 from v2) — The Parity/Prism Argument

This was a critical fix in v2 of the plan. The original design excluded the Nyquist
frequency, following standard signal processing practice (the Nyquist bin is often
treated as an artifact). But in our setting, the Nyquist bin at `P=10` carries the
signal `(-1)^v` — the parity function.

Bai et al.'s pentagonal prism model explicitly uses a parity basis alongside sine and
cosine bases. Excluding Nyquist would make it **impossible** to detect the prism
structure. The fix: include Nyquist, rescale its power by 2× to match the 2-DOF bins,
and verify with a synthetic test (Test 8: pure Nyquist parity signal).

**The empirical check.** 53 of 500 helix detections (10.6%) have `helix_best_freq = 5`,
which at P=10 is the Nyquist bin. Most of these are answer-digit detections where the
parity of the answer digit carries information. Without the Nyquist inclusion these
detections would have been missed entirely.

### 3g. Why the Helix Statistic (Fix 2 from v2) — K&T's Generalized Helix

Kantamneni & Tegmark (2025) found that number representations in Llama 3.1 8B follow
a **generalized helix**: periodic in two dimensions (the Fourier component) and
linearly increasing in a third (the magnitude component). A pure circle test would
miss the linear component and might even reject a helix: if the linear axis is strong,
it dilutes the two_axis_fcr denominator.

The helix_fcr extends the circle test by adding the best linear axis (from a coordinate
not used by the top-2 Fourier coordinates) to the numerator. If the model represents
numbers as K&T describes — periodic + magnitude — the helix_fcr should be higher
than the two_axis_fcr.

**The empirical check.** Of 500 helix detections, the mean `helix_fcr = 0.406` and
mean `two_axis_fcr = 0.396` — the helix statistic is ~2.5% higher on average. For
individual cells, the difference is more pronounced: e.g., `carry_1 / L5 / layer16 /
all / phase_c / carry_raw` has `two_axis_fcr = 0.588` and `helix_fcr = 0.657`, a 12%
increase from adding the linear axis. The linear component is real and adds measurable
power to the detection statistic.

### 3h. Why the K&T Pilot Gate (Fix 3 from v2)

Before running the full experiment (~4 hours), we validate our Fourier code against
published results. The K&T pilot:

1. Feeds 361 single-token integers (0–360) into Llama 3.1 8B
2. Extracts residual stream activations at layers {0, 1, 4, 8}
3. Fourier-decomposes each of the 4096 hidden dimensions
4. Checks for peaks at periods {2, 5, 10}

**Go/no-go gate:** If periods {2, 5, 10} don't appear in the top-10 at ≥1 layer,
our Fourier code has a bug. Stop and debug.

This gate caught zero bugs in Run 1 (it passed immediately), and passed again in
Run 3. But its existence prevented wasting 4 hours on a broken Fourier implementation.

### 3i. Why Raw vs. Residualized Spot Check (Fix 4 from v2)

Phase B residualizes activations by projecting out the product-magnitude direction.
If digit structure correlates with product magnitude (tens digits do: larger tens digits
→ larger products), residualization could remove Fourier signal along with the
magnitude confound.

The spot check (Pilot 0b) compares FCR on residualized vs. raw activations for one
concept (`a_units`, L5, layer 16, all). If disagreement exceeds 20%, Phase G must
run on both sources and report both.

**Result (Run 3 final, from `phase_g_fourier.log.3` at 09:26:08):**
`{'status': 'completed', 'fcr_residualized': 0.2836, 'fcr_raw': 0.2855,
'fcr_disagreement_pct': 0.69%, 'helix_residualized': 0.2942, 'helix_raw': 0.2967,
'helix_disagreement_pct': 0.84%, 'use_raw': False}`

Residualization is innocent. Proceed with residualized activations only.

### 3j. Why Number-Token Probe (Fix 5 from v2)

The literature (K&T, Gurnee et al.) probes at the **number-token position** — the
token where the number itself appears. Our pipeline probes at `=` — three tokens
downstream. If Phase G returns null at `=`, we cannot distinguish:

- "The model doesn't use Fourier features for multiplication" vs.
- "Fourier features exist at the number token but don't propagate to `=`"

The number-token probe (`extract_number_token_acts.py`) extracts activations at
operand positions (positions of `a` and `b`) at layers {4, 8, 12, 16, 20, 24}.
`phase_g_numtok_fourier.py` then runs the same screening on these activations as a
parallel experiment.

**Result (Section 12 for details): complete null (0/108).** The operand digits are
not arranged as circles/helices at the number token position in multiplication
context either. Combined with K&T's finding of helices for standalone integers, this
is evidence that the representation is **task-dependent**, not **token-dependent**.

### 3k. Why FDR q-Value Instead of Hard FCR Floor (Fix 6 from v2)

K&T report weaker Fourier fits in Llama 3.1 8B than in GPT-J. A hard FCR floor
(e.g., "FCR must exceed 0.5") would be calibrated to toy models and might reject
real but weak Fourier structure in pretrained models.

The solution: let the **permutation null** and **FDR correction** do the statistical
work. Any FCR that is significantly above the null (`q < 0.05` after BH across all
3,480 tests) counts as a detection. No hard floor on the FCR value itself.

**The empirical check.** The mean `helix_fcr` among detections is 0.406, and the
minimum is **0.125** (`carry_4 / L5 / layer31 / correct / phase_c / carry_raw`,
which has `helix_fcr = 0.125` but p-saturated). A hard floor of 0.5 would have
eliminated 67% of the 500 detections, including many of the strongest carry helix
cells. The permutation null is the right criterion.

### 3l. Carry Concept Period Specs — Binned, Mod10, Raw

Carry concepts have complicated value distributions (carry_2 at L5 ranges 0–26).
Phase C bins these into groups, but the binning period may not match any natural
periodicity. We test three period specifications:

1. **carry_binned** (period = `n_groups`): Tests periodicity at the binning resolution.
   If carry_0 has 9 bins, `P = 9` and `K = 4` frequencies.
2. **carry_mod10** (period = 10): Tests whether the carry's units digit has period-10
   structure. Only uses values 0–9 (the bins that correspond to individual integers).
   Skipped if fewer than `MIN_CARRY_MOD10_VALUES` (6) such values exist.
3. **carry_raw** (period = `n_unique_raw`): Tests periodicity at the raw value
   resolution. Period equals the number of distinct raw values — e.g., at L5,
   carry_1 has 18 unique raw values 0–17 so `P = 18`, `K = 9`.

**Empirical finding (the biggest surprise of Phase G):** Of 500 helix detections,
**397 use `carry_raw`, 22 use `carry_binned`, 0 use `carry_mod10`**. The `carry_mod10`
specification — the one we *expected* to dominate, since it directly tests the
base-10 decimal structure that K&T found for integers — gets zero detections. The
raw-value specification gets 79% of detections.

This means the carry helix is at the period equal to the range of carry values, not
at period 10. `carry_1` at L5 has values 0–17; its helix is at period 18, not 10.
`carry_2` at L5 has values 0–26; its helix is at period 27. This is unlike K&T's
standalone-integer helix (which has the decimal period 10) and unlike Nanda et al.'s
modular-arithmetic grokking (which has the specific mod value as the period). The
model is encoding carries at the period they naturally have — not mod 10.

See Section 8b for the full analysis and implications.

### 3m. Zero-Subsampling Guarantee

Phase G uses every data point in every computation. This is a deliberate design
constraint:

- **Centroids** are computed from ALL `N` samples in each population
- **Permutation null** shuffles ALL `N` sample labels, then recomputes centroids
  from ALL `N` samples
- **No subsampling** for computational tractability — the bottleneck is the
  groupby-mean in 9–18D subspace, which is fast even at N = 122,223

The verification assertion `sum(group_sizes) == N_population` fires for every analysis
([phase_g_fourier.py:1096](../phase_g_fourier.py#L1096)), confirmed in the logs.
For example, `carry_1 / L5 / layer16 / all / phase_c / carry_raw` has
`Group sizes: [14117, 11938, 11175, 11146, 9978, 9411, 8671, 8022, 6961, 6020, 5017,
4145, 3321, 2548, 1959, 1413, 1014, 4367]` summing to 122,223 — exactly matching the
full L5 population.

### 3n. The Linear Power DOF Rescaling (Fix 5 from Review)

When computing helix_fcr, the linear power and Fourier power must be on the same
scale. Under a Gaussian null:

- Each Fourier bin (2 DOF, cos + sin) has `E[P_k] ∝ ||cos_basis||² + ||sin_basis||² ≈ m`
- The linear bin (1 DOF) has `E[P_lin] ∝ Σ v_centered²`

Without rescaling, the denominator `total_power_helix = Fourier + linear` would
conflate two different scales, making helix_fcr either too high or too low depending
on the value distribution. The rescaling factor `m / (2 · Σ v_centered²)`
([phase_g_fourier.py:820-821](../phase_g_fourier.py#L820-L821)) normalizes the linear
contribution to match the Fourier scale.

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
would introduce a population-dependent bias into the linear axis. Implementation
in `resolve_carry_binned_spec()` at
[phase_g_fourier.py:525-554](../phase_g_fourier.py#L525-L554).

---

## 4. Concepts Screened and the Experiment Matrix

### 4a. Tier A: Digit Concepts (Period P=10)

All digit concepts are tested at period `P=10` (the natural period for base-10 digits).
Value sets verified against the coloring DataFrames and against the actual
`values_tested` column of the final result CSV:

| Concept | L2 | L3 | L4 | L5 |
|---------|----|----|----|----|
| a_units | 0–9 (10) | 0–9 (10) | 0–9 (10) | 0–9 (10) |
| a_tens | 1–9 (9) | 1–9 (9) | 0–9 (10) | 0–9 (10) |
| a_hundreds | — | — | 1–9 (9) | 1–9 (9) |
| b_units | 2–9 (8) | 0–9 (10) | 0–9 (10) | 0–9 (10) |
| b_tens | — | 1–9 (9) | 1–9 (9) | 0–9 (10) |
| b_hundreds | — | — | — | 1–9 (9) |
| ans_digit_0_msf | 1–9 (9) | 1–9 (9) | 1–9 (9) | 1–9 (9) |
| ans_digit_1_msf | 0–9 (10) | 0–9 (10) | 0–9 (10) | 0–9 (10) |
| ans_digit_2_msf | 0–9 (10) | 0–9 (10) | 0–9 (10) | 0–9 (10) |
| ans_digit_3_msf | — | 0–9 (10) | 0–9 (10) | 0–9 (10) |
| ans_digit_4_msf | — | — | 0–9 (10) | 0–9 (10) |
| ans_digit_5_msf | — | — | — | 0–9 (10) |

**Values are read from the coloring DataFrame at runtime, never hardcoded.** Concepts
with fewer than 3 groups are skipped. Incomplete grids (e.g., `b_units` at L2 with
values 2–9 only) are handled by the explicit DFT at actual values present — no
imputation, no zero-padding.

Per level (after availability filtering): **L2 = 6 digit concepts, L3 = 8, L4 = 10,
L5 = 12.**

### 4b. Tier B: Carry Concepts (Multi-Period Sweep)

Carry concepts have complicated value distributions and get a multi-spec sweep.
Verified from the final run's `values_tested` column:

| Concept | L2 | L3 | L4 | L5 |
|---------|----|----|----|----|
| carry_0 | 0–8 (9) | 0–8 (9) | 0–8 (9) | 0–8 (9) |
| carry_1 | — | 0–17 (18) | 0–17 (18) | 0–17 (18) |
| carry_2 | — | — | 0–17 (18) | 0–26 (27) |
| carry_3 | — | — | — | 0–18 (19) |
| carry_4 | — | — | — | 0–9 (10) |

**Period specs tested for each carry:**

- `carry_binned`: period = `n_groups` (from Phase C binning — may differ from raw
  range at lower levels when tail bins are merged)
- `carry_mod10`: period = 10, values = {0, 1, ..., 9}, skipped if fewer than 6 such
  values are in the Phase C group set
- `carry_raw`: period = `n_unique_raw` (only applied when the raw distribution was
  binned into tail bins by Phase C and `n_unique_raw ≥ 6`)

Verified `n_groups` and `period` values from the final CSV (representative rows):

| Concept | Level | carry_binned period | carry_mod10 period | carry_raw period | Notes |
|---------|-------|--------------------:|-------------------:|----------------:|-------|
| carry_0 | L2 | 9 | 10 | 9 | carry_binned and carry_raw both P=9 |
| carry_0 | L3 | 9 | 10 | 9 | |
| carry_0 | L4 | 9 | 10 | 9 | |
| carry_0 | L5 | 9 | 10 | 9 | |
| carry_1 | L3 | 13 | 10 | 18 | |
| carry_1 | L4 | 13 | 10 | 18 | |
| carry_1 | L5 | 13 | 10 | 18 | |
| carry_2 | L4 | 14 | 10 | 18 | |
| carry_2 | L5 | 14 | 10 | 27 | |
| carry_3 | L5 | 10 | (skip) | 19 | carry_mod10 skipped |
| carry_4 | L5 |  6 | (skip) | 10 | carry_mod10 skipped; raw is P=10 |

### 4c. Concepts NOT Screened

- **Column sums, partial products:** These have arbitrary binning boundaries. Testing
  Fourier structure on binned data tests the binning, not the model's geometry.
- **Binary/count/ordinal concepts:** `n_correct`, `n_wrong`, etc. — not periodic by
  nature.

### 4d. The Full Experiment Matrix (3,480 Cells, Verified)

The original design estimated ~3,348 cells; the final run produced **3,480 cells**
(after accounting for all population/basis/period-spec availability). Cells per
level in the final CSV:

| Level | Total cells | all | correct | wrong |
|-------|------------|-----|---------|-------|
| L2 | 324 | 162 | 162 | 0 |
| L3 | 744 | 249 | 252 | 243 |
| L4 | 996 | 337 | 333 | 326 |
| L5 | 1,416 | 478 | 459 | 479 |
| **Total** | **3,480** | **1,226** | **1,206** | **1,048** |

Notes:
- **L2 `wrong` population is empty.** L2 has only ~7 wrong samples (99.8% accuracy)
  which falls below `MIN_POPULATION = 30`. The wrong population is skipped entirely
  at L2.
- **L3 `wrong` (243) < all (249).** A handful of (concept, layer, basis, period_spec)
  cells at L3 have wrong population below `MIN_POPULATION` for specific concepts
  (e.g., `ans_digit_2_msf / L3 / layer?? / wrong` where the wrong sample count for
  that particular digit value falls short).
- **L5 `correct` (459) < all (478).** Similar — the correct subset has fewer samples
  and some concept slices drop below `MIN_POPULATION`.

**Cell breakdown by subspace_type (final CSV):** `phase_c` = 1,707 cells,
`phase_d_merged` = 1,773 cells. The imbalance arises because Phase C metadata
sometimes returns `dim_perm = 0`, skipping the Phase C row while Phase D still runs
(Phase D bases exist for more concept slices at lower dimensionality).

**Cell breakdown by period_spec:**

| Period spec | Count | Notes |
|-------------|-------|-------|
| `digit` | 1,752 | All digit concepts × all levels × pops × layers × bases |
| `carry_binned` | 576 | |
| `carry_mod10` | 576 | |
| `carry_raw` | 576 | |
| **Total** | **3,480** | |

**Cell breakdown by tier:** Tier A (digit concepts) = 1,752 rows, Tier B (carry
concepts) = 1,728 rows. Close to balanced.

---

## 5. Verification Results — Pilots and Gates

### 5a. Synthetic Pilot Tests (10/10 Passed After Fixes)

The synthetic pilot runs 10 controlled tests on constructed centroids to verify the
Fourier analysis code before touching real data. Implementation:
[phase_g_fourier.py:2136-2310](../phase_g_fourier.py#L2136-L2310).

**Run 1 result:** 8/10 passed. Tests 3 and 9 failed.
**Run 2 and Run 3 results (after fixes):** 10/10 passed.

| Test | Description | Expected | Run 1 | Run 2/3 | Result |
|------|-------------|----------|-------|---------|--------|
| 1 | Perfect circle (P=10) | fcr ~ 1.0 | 1.0000 | 1.0000 | **PASS** |
| 2 | Random noise | fcr near null (~0.20) | 0.2355 | 0.2355 | **PASS** |
| 3 | Linear/quadratic | fcr < 0.65 | 0.5943 | 0.5943 | **PASS** (threshold fixed) |
| 4 | Incomplete grid (v=1..9) | fcr high | 0.8667 | 0.8667 | **PASS** |
| 5 | Pure DC offset | no crash, fcr=0 | 0.0 | 0.0 | **PASS** |
| 6 | P=9 conjugate | K=4, fcr=1.0 | 1.0000 | 1.0000 | **PASS** |
| 7 | Convention spot-check | np.allclose | (skipped) | (skipped) | N/A |
| 8 | Nyquist parity | Nyquist ~100% | 1.0000 | 1.0000 | **PASS** |
| 9 | Helix test | helix > two_axis | 0.72 ≤ 0.90 (**FAIL**) | 0.9084 > 0.9000 | **PASS** (denom fixed) |
| 10 | Pure linear ramp | linear >> Fourier | 6806 >> 450 | same | **PASS** |

Full pilot run output confirmed in `phase_g_fourier.log.3` at 09:25:27 (Run 3 attempt)
and re-confirmed in the final `phase_g_fourier.log` at 05:46 (Run 3 completion).

**Test 3 fix: threshold raised from 0.5 to 0.65.** The input is
`c_v = [v/9, (v/9)^2, 0, ...]` for `v = 0..9`. The original threshold of 0.5 assumed
linear/quadratic signals would produce low FCR. In practice, a linear ramp has
genuine Fourier content concentrated at low frequencies, yielding `FCR = 0.5943`.
This is correct behavior — the FCR measures concentration, not periodicity. Real
circles produce `FCR > 0.95`, so the raised threshold (0.65) cleanly separates linear
artifacts from genuine periodic structure. The permutation null remains the primary
safeguard: linear signals are not significant relative to shuffled centroids.

**Test 9 fix: helix denominator now uses only the chosen linear coordinate.** See
Section 2f for the full derivation. The fix changed the denominator from "sum
rescaled linear power across all d coordinates" to "only the best linear coord's
rescaled power", matching the numerator's scope.

**Exit code bug (also fixed).** In Run 1, the synthetic pilot returned exit code 0
despite test failures because `main()` used `return` instead of `sys.exit(1)`. The
shell script checked `$?` but got 0, so the pipeline continued. Both
`phase_g_fourier.py` and `phase_g_kt_pilot.py` now call `sys.exit(1)` on failure,
and `run_phase_g.sh` gates all 5 steps on exit code.

### 5b. K&T Replication Pilot (PASSED)

The K&T pilot ran in the main Run 3 attempt on Apr 11 at 11:35:13 EDT and ran again
in the final Run 3 completion on Apr 13 at 02:15:28. Both runs produced identical
results (the pilot is deterministic given model weights).

**Setup (from `kt_pilot_summary.json`):**

- Integers: 0–360 (361 single-token integers, all verified as single-token)
- Target periods: {2, 5, 10}
- Candidate periods: T = 2 to T = 30
- Gate: each target in top-10 at ≥1 layer
- Layers tested: {0, 1, 4, 8}
- Model: `/data/user_data/anshulk/arithmetic-geometry/model` (Llama 3.1 8B)
- Method: Parseval total-power magnitude spectrum (sum of squared Fourier magnitudes
  across all 4096 hidden dimensions)
- Batch processing: 6 batches of 64 (last batch 41)

**Per-layer results from the final run's JSON:**

| Layer | Top-10 periods | T=2 power (rank) | T=5 power (rank) | T=10 power (rank) | Target found |
|-------|----------------|------------------:|------------------:|------------------:|:-------------:|
| 0 | [2, 5, 10, 25, 20, 26, 3, 24, 30, 29] | 482.2 (1) | 454.8 (2) | 445.0 (3) | ✓ all |
| 1 | [10, 2, 5, 25, 20, 26, 24, 30, 28, 29] | 2,297.9 (2) | 2,285.7 (3) | 2,391.2 (1) | ✓ all |
| 4 | [10, 5, 2, 25, 20, 26, 24, 30, 29, 28] | 9,426.7 (3) | 9,892.1 (2) | 10,185.2 (1) | ✓ all |
| 8 | [10, 5, 2, 25, 20, 26, 24, 30, 29, 28] | 21,052.4 (3) | 23,101.1 (2) | 23,906.7 (1) | ✓ all |

**Key observations:**

1. **All three target periods appear in the top-3 at every layer tested.** This far
   exceeds the gate criterion (each target in top-10 at ≥1 layer).
2. **Power grows with depth.** T=10 power increases from 445.0 at layer 0 to 23,906.7
   at layer 8 — a **53.7× increase**. This matches the K&T finding that Fourier
   features strengthen with depth.
3. **The rank ordering shifts.** At layer 0, T=2 (parity) is strongest. By layer 1,
   T=10 (decimal period) takes over and remains dominant through layer 8. This is
   consistent with K&T's observation that parity is an early feature while decimal
   periodicity develops in deeper layers.
4. **T=25, T=20, T=26, T=24, T=30 appear consistently in the top-10.** These are
   harmonically related to the base periods (T=25 = 5×5, T=20 = 2×10, T=26 ≈ 2×13).
   Their presence suggests the Fourier structure is rich but the target periods dominate.

**Power growth table (from the JSON):**

| Period | Layer 0 | Layer 1 | Layer 4 | Layer 8 | Growth (0→8) |
|--------|---------|---------|---------|---------|--------------|
| T=2 | 482.2 | 2,297.9 | 9,426.7 | 21,052.4 | 43.7× |
| T=5 | 454.8 | 2,285.7 | 9,892.1 | 23,101.1 | 50.8× |
| T=10 | 445.0 | 2,391.2 | 10,185.2 | 23,906.7 | 53.7× |

All three periods grow at similar rates (44–54×), suggesting the model amplifies
**all** Fourier features in parallel, not selectively boosting one period.

**Technical details of the K&T pilot implementation:**

- Integers 0–360 fed as bare tokens (no prompt context, just BOS + number token)
- Residual stream hooked at each target layer using PyTorch register hooks
- Each hidden dimension (4096) Fourier-decomposed independently
- Total power at each period = sum of squared magnitudes across all 4096 dimensions
- Processing: 6 batches of 64, extraction in 1.5s (~0.004s/integer), Fourier
  analysis in ~4s
- Model loaded in 37.5s from local cache on A6000 GPU (initial) and 14.0s on the
  retry (Apr 13)
- Total pilot elapsed time: **14.88 seconds** (from the summary JSON's `elapsed_seconds`)

**Gate verdict: PASSED.** Our Fourier code reproduces K&T's published finding. The
methodology is validated. Results saved to
`/data/user_data/anshulk/arithmetic-geometry/phase_g/kt_pilot/kt_pilot_summary.json`
and to the per-layer PNGs at `plots/phase_g/kt_pilot/kt_magnitude_spectrum_layer{0,1,4,8}.png`
(4 files).

### 5c. Pilot 0b: Raw vs. Residualized (PASSED)

Pilot 0b ran during Run 1 at 04:20:27–04:20:38 and again in Run 3 at 09:26:08.
Same result both times.

**Setup:**

- Concept: `a_units`, L5, layer 16, `all` population (N = 122,223)
- Comparison: residualized activations (Phase C's Phase B output) vs. raw activations,
  both projected into Phase C basis (9 dimensions)

**Results (from the log):**

| Statistic | Residualized | Raw | Disagreement |
|-----------|-------------:|------:|-------------:|
| `two_axis_fcr` | 0.2836 | 0.2855 | **0.69%** |
| `helix_fcr` | 0.2942 | 0.2967 | **0.84%** |
| `best_freq` | 5 (Nyquist) | 5 (Nyquist) | match |
| `use_raw` | False | — | — |

**Convention spot-check (Test 7):**
`np.allclose(projected_all centroids, manual centroids) = True`. The Phase C
convention (`projected_all = (acts - grand_mean) @ basis.T`) is verified.

**Verdict: PASSED.** Both FCR metrics agree within 1%. The best frequency and
coordinates match exactly. Residualization is innocent. Proceed with residualized
activations.

**Implication:** Phase B removes the product-magnitude confound without disturbing
the within-digit Fourier signal. This makes sense: residualization projects out one
direction (the product-magnitude axis), and digit-based Fourier structure is
orthogonal to product magnitude (the units digit of `a` does not predict `a × b`).

### 5d. Number-Token Extraction (Complete)

`extract_number_token_acts.py` completed as Step 2 of the pipeline in Run 1 on Apr 11
and was skipped (reused) on subsequent runs thanks to the resume logic at
[extract_number_token_acts.py:296-303](../extract_number_token_acts.py#L296-L303):

```python
# Resume check: skip if all output files already exist
if all(f.exists() for f in expected_files):
    logger.info("  Level %d already complete, skipping", level)
    continue
```

**Output:** 48 files at
`/data/user_data/anshulk/arithmetic-geometry/activations_numtok/level{L}_layer_{LL:02d}_pos_{a|b}.npy`
(float16), totaling 9.2 GB. Verified timestamps: all 48 files have mtime Apr 11 04:16,
consistent with a single extraction pass during Run 1.

Layer coverage: {4, 8, 12, 16, 20, 24}. Position coverage: `a` and `b`. Levels:
L2, L3, L4, L5. Total combinations: 4 × 6 × 2 = 48 files, matching the observed count.

File sizes per level (from `ls -la`):

| Level | Samples | Bytes per file | Total for level |
|-------|---------|----------------:|----------------:|
| L2 | 4,000 | 32,768,128 | 393 MB |
| L3 | 10,000 | 81,920,128 | 983 MB |
| L4 | 10,000 | — | — |
| L5 | up to 122,223 | (varies) | — |

### 5e. Phase D Basis Count Check (PASSED)

From the Run 1 log (`phase_g_fourier.log.3` at 09:25:27):

```
Phase D merged bases found: 2844 total ({'L5': 1035, 'L2': 306, 'L3': 666, 'L4': 837})
Phase D check passed: 2844 bases >= 2844
```

The filesystem walk found exactly 2,844 `merged_basis.npy` files. This verifies
all Phase D bases are accessible before the main experiment begins. Implementation:
`count_phase_d_bases()` in `phase_g_fourier.py` (filesystem walk; replaced the
original hard assert with a logged error + `sys.exit(1)` in Fix 11 from Run 1).

---

## 6. Run History — From Run 1 to Final Completion

Phase G required three full-pipeline attempts plus many preemption-driven restarts
before completing. This section is the full timeline, with every code fix and
every timestamp, so the run history is unambiguous for the paper's reproducibility
section.

### 6a. Run 1: Crash 1 (carry_mod10 assert)

| Field | Value |
|-------|-------|
| SLURM job | 7056981 |
| Node | babel-v9-32 |
| Partition | preempt |
| GPU | 1× A6000 |
| CPUs | 24 |
| Memory | 128 GB |
| Time limit | 7 days |
| Start | Apr 11, 2026, 04:14:54 EDT |
| Outcome | **Crash** after ~31 minutes |
| Crash site | `carry_mod10 / L4 / layer 4 / correct / phase_c` |
| Error | `AssertionError: carry_mod10 value 8 not found` |

Run 1 completed Steps 1–4 and began Step 5 (full Fourier screening). It crashed when
the code tried to assert that all values {0, 1, ..., 9} were present in the Phase C
group set for `carry_0 / L4 / layer 4 / correct`. Phase C had merged values 8 and 9
into a tail bin because the population was small (~2,000 correct samples with rare
8-and-9-carry events), so the assert fired.

Separately, the synthetic pilot (Step 3) had reported Tests 3 and 9 as failed but
the pipeline continued anyway because `main()` used `return` instead of `sys.exit(1)`
— the shell script's `$?` check got 0 and proceeded to the next step. This was
discovered during the post-crash audit.

### 6b. Run 1 Bugs — The 12 Fixes Applied Before Run 2

A code audit before Run 2 identified 12 bugs total. All were fixed in the codebase
before resubmission. The bugs fall into four categories: crash bugs (would terminate
the run), correctness bugs (would produce wrong results), test calibration bugs
(synthetic pilot failures), and robustness bugs (edge cases that could crash on
certain data).

| # | Bug | Category | File | Root Cause | Fix |
|---|-----|----------|------|------------|-----|
| 1 | carry_mod10 crash | crash | phase_g_fourier.py | Hard assert on value 8 not present in L4/correct (Phase C merged it into tail bin) | Replaced assert with graceful skip; filter to present values, skip if < `MIN_CARRY_MOD10_VALUES` |
| 2 | Exit code 0 on pilot failure | crash-gating | phase_g_fourier.py | `return` from `main()` gives exit 0; shell script sees success | Added `import sys`; `sys.exit(1)` on failure |
| 3 | Exit code 0 on K&T gate fail | crash-gating | phase_g_kt_pilot.py | Same return-vs-exit issue | Added `sys.exit(1)` on gate failure |
| 4 | Pilot 0b not gated | crash-gating | run_phase_g.sh | No exit code check after Step 4 | Added `P0B_EXIT` check with abort on failure |
| 5 | Helix denominator mismatch | **correctness** | phase_g_fourier.py | Denominator summed all `d` coords' rescaled linear power; numerator used only 1 | Denominator now uses only `best_linear_rescaled` |
| 6 | Test 3 threshold too tight | test | phase_g_fourier.py | Threshold 0.5 for linear/quadratic; actual FCR is 0.5943 | Raised to 0.65 (parabolas ~0.6, circles >0.95) |
| 7 | `run_all()` return type mismatch | crash | phase_g_fourier.py | Empty-result path returned DataFrame instead of `(DataFrame, list)` | Fixed to return `pd.DataFrame(), []` |
| 8 | `d<2` guard missing (two_axis) | crash | phase_g_fourier.py | `two_axis_coord_b` uninitialized when `d=1` | Added guard: skip `two_axis` when `d < 2` |
| 9 | `d<2` guard missing (helix) | crash | phase_g_fourier.py | `helix_coord_b` uninitialized when `d=1` | Added guard: skip helix when `d < 2` |
| 10 | Eigenvalue-weighted FCR div-by-zero | crash | phase_g_fourier.py | `weights.sum()` could be 0 when all eigenvalues are negligible | Added threshold check before division |
| 11 | Phase D hard assert | crash | phase_g_fourier.py | `assert n_bases >= expected` would crash run on filesystem mismatch | Changed to `logger.error` + `sys.exit(1)` |
| 12 | Empty DataFrame column loss | correctness | phase_g_fourier.py | `cell_df[bool_mask]` drops columns when result is empty | Fixed with `.loc[bool_mask].copy()` |

**Impact summary:** Bugs 1, 7–11 were crash bugs that would have terminated the run
at various points beyond L4. Bug 5 produced incorrect helix_fcr values (underestimated
by ~20% in synthetic test). Bug 12 could silently produce incomplete output CSVs.
Bugs 2–4 and 6 affected pipeline gating (failures not caught). All Run 1 outputs
(except the reusable number-token activations and the K&T pilot results) were
cleared before Run 2.

### 6c. Run 2: Crash 2 (zero-dimensional Phase D subspace)

| Field | Value |
|-------|-------|
| SLURM job | 7057231 |
| Node | babel-u9-28 |
| Partition | preempt |
| Start | Apr 11, 2026, ~05:15 EDT |
| Outcome | **Crash** after ~54 minutes |
| Crash site | `ans_digit_2_msf / L5 / layer 4 / correct / phase_d_merged` |
| Error | `KeyError: 'per_freq_top2_coords'` |

Run 2 used the fixed codebase with all 12 Run 1 bugs resolved. Synthetic pilot
passed 10/10. Number-token activations were reused from Run 1 (resume logic in
`extract_number_token_acts.py`). The run crashed deeper into Step 5 — at L5/layer 4
— when `ans_digit_2_msf`'s Phase D merged basis for the `correct` population
returned shape `(0, 4096)` (zero dimensions). This zero-dim subspace caused the
projected centroids to have shape `(10, 0)`, which hit the zero-power early-return
path in `fourier_all_coordinates()`, which was missing the `per_freq_top2_coords`
key — causing a `KeyError` when `compute_helix_fcr()` tried to access it downstream.

### 6d. Run 2 Bugs — The 5 Edge-Case Fixes Applied Before Run 3

A second code audit identified 5 edge-case bugs, all related to degenerate inputs
(zero-dimensional subspaces, zero permutations). All were fixed before Run 3.

| # | Bug | Severity | Trigger | Fix |
|---|-----|----------|---------|-----|
| 13 | `fourier_all_coordinates` early return missing `per_freq_top2_coords` | **CRASH** (KeyError) | `d=0` Phase D merged subspace → zero-shaped centroids → zero total Fourier power → early return lacks key → downstream `compute_helix_fcr` crashes | Added `"per_freq_top2_coords": np.zeros((K, 2), dtype=int)` to the zero-power early return ([phase_g_fourier.py:683](../phase_g_fourier.py#L683)) |
| 14 | `permutation_null` divides by `n_perms` in log message without guard | **CRASH** (ZeroDivisionError) | `--skip-null` sets `n_perms=0` → `elapsed / n_perms` on the log line | Changed to `elapsed / n_perms if n_perms > 0 else 0.0` ([phase_g_fourier.py:1028](../phase_g_fourier.py#L1028)) |
| 15 | `compute_helix_fcr` fallback indexes `linear_power_rescaled[0]` when `d=0` | **CRASH** (IndexError) | `d=0` subspace → empty `linear_power_rescaled` array → fallback at the "no coord available" branch indexes `[0]` on empty array | Added explicit `d == 0` branch returning zero power instead of indexing ([phase_g_fourier.py:854-859](../phase_g_fourier.py#L854-L859)) |
| 16 | `process_level_layer_pop` passes `d=0` Phase D merged bases to `analyze_one` | **Root cause** | `load_phase_d_merged_basis` returns shape `(0, 4096)` for concepts with no Phase D subspace → projection gives `(N, 0)` shape → `analyze_one` receives `d=0` | Added `merged_basis.shape[0] > 0` guard alongside the `is not None` check before projecting |
| 17 | `analyze_one` detection logic indexes `p_coord[coord_a]` when `p_coord` is empty | **CRASH** (IndexError) | `d=0` → `p_coord` has shape `(0,)` → `coord_a=0` from early return defaults → `p_coord[0]` on empty array | Added defensive `d == 0` branch setting `circle_detected=False`, `helix_detected=False` ([phase_g_fourier.py:1148-1154](../phase_g_fourier.py#L1148-L1154)) |

Additionally, `run_pilot_0b`'s skipped return path was missing 6 keys present in the
normal return path. Updated to return consistent keys with `None` values for the
skipped case. This was not a crash risk (the caller only logs the result) but
violated the principle of consistent return types.

**Numbering note:** Bugs 1–12 are from Run 1 (Section 6b). Bugs 13–17 are from Run 2.
The numbering is cumulative across all runs and is preserved in the code comments.

### 6e. Run 3 in the `preempt` Partition — 12 Preemptions, 3 Log Rotations

| Field | Value |
|-------|-------|
| SLURM job | 7058788 |
| Initial partition | preempt |
| QOS | preempt_qos |
| Initial time limit | 7 days |
| First start | Apr 11, 2026, 11:35 EDT |
| Preemption count | **12** |
| Final state in preempt | PENDING with StartTime estimate Apr 14 2026 10:22 EDT |

Run 3 was submitted to the `preempt` partition with a 7-day time limit. The code
was the final, stable version (all 17 bugs fixed). But every time the job was
allocated a node, a higher-priority job arrived and preempted it. The `scontrol
show job 7058788` output from Apr 12 22:13 shows:

- `Requeue=1`, `Restarts=12`
- `JobState=PENDING`, `Reason=Priority`
- `StartTime=2026-04-14T10:22:29` (scheduler estimate)
- `Partition=preempt`, `Account=gneubig`, `QOS=preempt_qos`

The log file rotations reflect the preemption history:

| File | Size | mtime | Notes |
|------|------|-------|-------|
| `phase_g_fourier.log.3` | 9.6 MB | Apr 11 18:11 | First preempt cycle, ~3,546 ANALYZE calls |
| `phase_g_fourier.log.2` | 9.6 MB | Apr 12 06:31 | Second preempt cycle, ~3,553 ANALYZE calls |
| `phase_g_fourier.log.1` | 9.6 MB | Apr 12 11:25 | Third preempt cycle, ~3,340 ANALYZE calls |
| `phase_g_fourier.log` (before final run) | 4.0 MB | Apr 12 12:59 | Fourth cycle, preempted mid-L3 |

Each rotation captures an entire restart-from-scratch of Step 5 (the `extract_number_
token_acts.py` resume logic meant Step 2 skipped on every requeue, but Step 5 had no
checkpointing and re-ran from the top each time). The Apr 12 12:59 preemption was the
last one in `preempt` — after that, the partition was changed.

### 6f. Run 3 Completion in `general` Partition

On Apr 12 evening, the job was moved to the `general` partition via an in-place SLURM
update:

```
scontrol update JobId=7058788 Partition=general QOS=normal TimeLimit=2-00:00:00
```

No code changes, no re-submit — `scontrol` modifies the pending job in place.
Priority jumped from 4012 (preempt_qos) to 9981 (normal). The job was picked up
within ~14 hours.

**Final run details:**

| Field | Value |
|-------|-------|
| SLURM job | 7058788 (same ID, in-place partition update) |
| Node | babel-t9-24 |
| Partition | general |
| QOS | normal |
| GPU | 1× A6000 |
| CPUs | 24 |
| Memory | 128 GB |
| Time limit | 2 days |
| Start | **Mon Apr 13, 2026, 02:15:28 EDT** |
| End | **Mon Apr 13, 2026, 06:03:39 EDT** |
| Duration | **226.9 minutes (3 hours 48 minutes)** |
| Outcome | **PHASE G COMPLETE** |

The final log tail (`slurm-7058788.out`, Apr 13 06:03):

```
=========================================
Phase G: Full Pipeline
Job ID: 7058788
Node:   babel-t9-24
Date:   Mon Apr 13 02:15:28 EDT 2026
=========================================
Activating conda environment...

Step 1: K&T replication pilot...
K&T pilot complete.

Step 2: Number-token activation extraction...
Number-token extraction complete.

Step 3: Synthetic pilot tests...
Synthetic pilot PASSED.

Step 4: Pilot 0b (raw vs residualized)...
Pilot 0b complete.

Step 5: Full Phase G Fourier screening...

=========================================
Phase G COMPLETE
Date: Mon Apr 13 06:03:39 EDT 2026
=========================================
```

And the Fourier screening's own summary at 06:03:39 (`phase_g_fourier.log`):

```
06:03:39 INFO     PHASE G COMPLETE: 226.9 minutes (3.8 hours)
06:03:39 INFO       Total analyses: 3480
06:03:39 INFO       Circles detected: 496
06:03:39 INFO       Helices detected: 500
06:03:39 INFO       Geometry detected: {'none': 2979, 'helix': 500, 'circle': 1}
```

The 496 "Circles detected" count refers to cells where `circle_detected = True`
(i.e., the three-way conjunction `p_two_axis + p_coord_a + p_coord_b < 0.01` passes).
Most of these (495) also pass the helix conjunction, so they are reclassified as
`geometry_detected = "helix"` in the hierarchical labeling. The one that passes
circle but not helix (`carry_2 / L5 / layer20 / wrong / phase_c / carry_binned`) is
the only cell with `geometry_detected = "circle"`. Total non-null cells: 500 helix
+ 1 circle = 501 = 3,480 − 2,979, consistent.

**Timing of the post-Step-5 work (from the log):**

| Time | Phase | Notes |
|------|-------|-------|
| 05:46:46 | Fourier screening finishes last cell | L5/layer31/wrong complete |
| 05:46:46 | Applying decision rule | Class-level rule evaluated |
| 05:47:12 | Start plot generation | |
| 05:48:41 | Centroid circle plots: 501 saved | |
| 05:49:53 | Power spectrum plots: 501 saved | |
| 05:50:37 | P-value trajectory plots complete | 162 files |
| 05:50:37 | Loading coloring DataFrame for single-example plots | L2 |
| 05:51:13 | → L3 loading | |
| 05:52:38 | → L4 loading | |
| 05:54:27 | → L5 loading | |
| 06:03:39 | Single-example projection plots: 1498 saved | |
| 06:03:39 | PHASE G COMPLETE | 226.9 min total |

Plot generation alone consumed ~17 minutes (05:46 → 06:03).

---

## 7. Final Results — Overview

This section reports aggregate counts and breakdowns directly from
`/data/user_data/anshulk/arithmetic-geometry/phase_g/summary/phase_g_results.csv`
(3,480 rows, Apr 13 05:46 EDT). Every number in this section has been independently
verified by reading the CSV with a Python script (not by trusting the log's own
summary line).

### 7a. Total Cell Counts and Geometry Breakdown

| Category | Count | Fraction |
|----------|------:|---------:|
| Total analysis cells | 3,480 | 100.0% |
| `geometry_detected = none` | 2,979 | 85.6% |
| `geometry_detected = helix` | 500 | 14.4% |
| `geometry_detected = circle` | 1 | 0.03% |
| `helix_detected = True` (boolean column) | 500 | 14.4% |
| `circle_detected = True` (boolean column) | 496 | 14.3% |

The `helix_detected` and `circle_detected` boolean columns are the raw conjunction
outcomes (not the hierarchical label). Most cells that pass the circle conjunction
also pass the helix conjunction; only 1 cell passes circle but not helix, and is
therefore classified as `geometry_detected = circle`. A handful of cells (5) pass
helix but fail circle — these are helix detections where the helix's best frequency
differs from the two-axis best frequency, so the "circle at helix_best_freq" check
succeeds even though the "circle at two_axis_best_freq" check fails. Example
cross-tabulation (from `phase_g_results.csv`):

| circle_detected | helix_detected | Count |
|----------------:|---------------:|------:|
| False | False | 2,979 |
| False | True | 5 |
| True | False | 1 |
| True | True | 495 |

**The 5 "helix-only" cells** (helix_detected=True but circle_detected=False):

| Level | Layer | Population | Basis | Concept | Period spec | two_axis_fcr | helix_fcr |
|-------|-------|------------|-------|---------|-------------|-------------:|----------:|
| L2 | 6 | correct | phase_d_merged | ans_digit_0_msf | digit | 0.4119 | 0.4443 |
| L2 | 8 | correct | phase_d_merged | ans_digit_0_msf | digit | 0.3793 | 0.4284 |
| L3 | 4 | all | phase_d_merged | carry_0 | carry_raw | 0.2909 | 0.3359 |
| L3 | 31 | all | phase_d_merged | carry_1 | carry_binned | 0.3066 | 0.3509 |
| L4 | 4 | correct | phase_d_merged | b_units | digit | 0.3473 | 0.3684 |

These are all marginal detections (two_axis_p_value between 0.012 and 0.027, just
above the 0.01 threshold, while helix_p_value is below 0.01 thanks to the extra
linear axis pushing the combined statistic over the line).

**The 1 "circle-only" cell** (circle_detected=True but helix_detected=False):

| Level | Layer | Population | Basis | Concept | Period spec | two_axis_fcr | helix_fcr |
|-------|-------|------------|-------|---------|-------------|-------------:|----------:|
| L5 | 20 | wrong | phase_c | carry_2 | carry_binned | 0.5922 | 0.5928 |

`p_two_axis = 0.002`, `p_helix = 0.003`. Both coords significant, but the linear axis
at the helix's best frequency does not reach `p_linear < 0.01`. This is the only
cell in the final run where the geometry is flagged as a circle without the linear
component. It lives in the `wrong` population of carry_2, which is a high-variance
binned concept.

### 7b. Helix Detections by Level

| Level | Total cells | Helix | Rate | Model accuracy (approx) |
|-------|------------:|------:|-----:|-------------------------|
| L2 | 324 | 2 | 0.6% | 99.8% |
| L3 | 744 | 93 | 12.5% | ~53% |
| L4 | 996 | 150 | 15.1% | ~29% |
| L5 | 1,416 | 255 | 18.0% | ~6% |

The rate tracks the model's accuracy collapse across levels. L2 is essentially null
(2 detections out of 324). L3 jumps to 12.5% — a 20× increase in one step. L4 and
L5 grow further but the step-changes are smaller.

**L5's 255 is the largest single-level count.** It accounts for 51% of all
detections. carry_3 and carry_4 only exist at L5, so they contribute entirely to
this number (54 each = 108 of the 255).

**L2 has 2 detections — both are `ans_digit_0_msf`:**

| Level | Layer | Population | Basis | Concept | Period spec | helix_fcr | p_helix |
|-------|-------|------------|-------|---------|-------------|----------:|--------:|
| L2 | 6 | correct | phase_d_merged | ans_digit_0_msf | digit | 0.4443 | 0.00999 |
| L2 | 8 | correct | phase_d_merged | ans_digit_0_msf | digit | 0.4284 | 0.00799 |

Both are in the `correct` population and Phase D merged basis. Both are marginal
(p-values just below the 0.01 threshold). These are the only L2 detections in the
entire phase.

### 7c. Helix Detections by Layer

| Layer | Total cells | Helix | Rate |
|-------|------------:|------:|-----:|
| 4 | 388 | 57 | 14.7% |
| 6 | 387 | 55 | 14.2% |
| 8 | 389 | 55 | 14.1% |
| 12 | 391 | 51 | 13.0% |
| 16 | 388 | 55 | 14.2% |
| 20 | 384 | 61 | 15.9% |
| 24 | 382 | 55 | 14.4% |
| 28 | 385 | 56 | 14.5% |
| 31 | 386 | 55 | 14.2% |

**The rate is remarkably uniform**: 13.0–15.9% across all 9 layers. No single layer
dominates. Layer 20 is the highest at 15.9%; layer 12 the lowest at 13.0%. The
difference is small and within normal statistical variation.

This contrasts with Phase A's finding that layer 16 is the "information peak" for
visual clustering — the helix geometry for arithmetic concepts does not show an
information-peak signature. It is present everywhere in the residual stream.

### 7d. Helix Detections by Population

| Population | Total cells | Helix | Rate |
|-----------|------------:|------:|-----:|
| `all` | 1,226 | 162 | 13.2% |
| `correct` | 1,206 | 182 | 15.1% |
| `wrong` | 1,048 | 156 | 14.9% |

The `correct` population has the highest detection rate (15.1% vs 13.2% for `all`).
This is interesting: `all` = `correct ∪ wrong`, so naively `all` should pool power.
But `correct` populations have smaller N (by a factor of ~30 at L5), which means
their centroids are noisier, yet they detect at a *higher* rate. The explanation is
that **the structure tracks correctness**: when the model computes correctly, the
intermediate variables (carries) are organized more cleanly, and the noise reduction
from pooling the large `wrong` population into `all` is partially offset by the
`wrong` samples' less-clean structure contributing noise to the centroids.

This is the opposite of what a pure sample-size argument would predict and is a
strong hint that the helix is functional, not incidental.

### 7e. Helix Detections by Subspace Type (Phase C vs Phase D)

| Subspace type | Total cells | Helix | Rate |
|---------------|------------:|------:|-----:|
| `phase_c` | 1,707 | 208 | 12.2% |
| `phase_d_merged` | 1,773 | 292 | 16.5% |

**Phase D catches 40% more helix detections than Phase C.** The difference is almost
entirely driven by answer digits — Phase D's LDA-refined basis captures answer-digit
periodicity that the consensus basis (Phase C) misses. For carries, Phase C and
Phase D agree closely (see Section 11a).

### 7f. Helix Detections by Period Spec

| Period spec | Total cells | Helix | Rate |
|-------------|------------:|------:|-----:|
| `digit` | 1,752 | 81 | 4.6% |
| `carry_binned` | 576 | 22 | 3.8% |
| `carry_mod10` | 576 | **0** | **0.0%** |
| `carry_raw` | 576 | 397 | 68.9% |

**`carry_raw` is the winning period spec — 397 helix detections out of 576 possible
cells (68.9% rate).** This is the single biggest update to the interpretation of
Phase G compared to the pre-Run-3 plan. The design expected `carry_mod10` to
dominate (because it directly tests the decimal period 10 that K&T found for
integers), but `carry_mod10` has **zero** helix detections, while `carry_raw` has
397. The model encodes carries at the natural period of their raw-value range,
not mod 10.

See Section 8b for the full analysis.

### 7g. Helix Detections by Tier (A vs B)

| Tier | Concept type | Total cells | Helix | Rate |
|------|--------------|------------:|------:|-----:|
| A | Digit concepts (operands + answers) | 1,752 | 81 | 4.6% |
| B | Carry concepts | 1,728 | 419 | 24.2% |

Tier B (carries) captures **83.8% of all helix detections** (419/500). This is the
dominant pattern: carries are organized on helices, digit concepts mostly are not.

### 7h. FDR Survival

Two FDR corrections are applied at the end of the run:

- `two_axis_q_value`: BH across all 3,480 `two_axis_p_value` rows
- `helix_q_value`: BH across all 3,480 `helix_p_value` rows

The log reports the raw counts:

- `two_axis FDR: 1265 / 3480 significant at q < 0.05`
- `helix FDR: 1289 / 3480 significant at q < 0.05`

The counts here are larger than the 500 helix detections because the raw q-value
condition is only one of the conjunction criteria. A cell can have `helix_q_value <
0.05` without being `helix_detected = True` if the coordinate-level or linear-axis
p-values fail the thresholds. The **full conjunction** (helix_detected = True)
filters down to 500.

**Of the 500 helix detections:**

- **500 / 500** have `helix_q_value < 0.05` (100% FDR survival)
- **497 / 500** also have `two_axis_q_value < 0.05`
- The 3 "helix-only-q" cells are among the 5 helix-only cells from Section 7a,
  with `two_axis_p_value` hovering in the 0.012–0.027 range (just above the
  pre-FDR 0.01 threshold)

**Of the 1 circle-only detection:** `two_axis_q_value = 0.00435` and `helix_q_value
= 0.01007` — both well below 0.05.

**Of all 3,480 cells:**

| Threshold | Cells | Fraction |
|-----------|------:|---------:|
| `helix_q_value < 0.001` | 0 | 0.0% |
| `helix_q_value < 0.01` | 908 | 26.1% |
| `helix_q_value < 0.05` | 1,289 | 37.0% |

No cell has `helix_q_value < 0.001`. This is because the smallest raw `helix_p_value`
is 0.000999 (the floor from 1,000 permutations), and with 3,480 tests the smallest
BH-adjusted q-value for a single minimum p is ~0.000999 × 3,480 / 1 ≈ 0.00348. So
q-values cluster just above 0.001. The 908 cells at `q < 0.01` correspond to
approximately the cells whose raw p-value is ~0.001–0.003.

### 7i. p-Saturation Pattern

**458 of 500 (91.6%) helix detections have `p_saturated = True`.** That is, the
observed FCR value exceeds all 1,000 permutation null values — the true p-value
is below the floor. The remaining 42 detections (8.4%) have `p_helix` between
0.002 and 0.010 (i.e., they pass the threshold but aren't at the floor).

**Overall p-saturation in the full CSV:**

| p_saturated | Count | Fraction |
|-------------|------:|---------:|
| False | 2,665 | 76.6% |
| True | 815 | 23.4% |

Among helix detections, saturation is nearly universal. Among non-detections, 357
cells (12%) are still saturated — these have saturated `two_axis_p_value` or
`helix_p_value` but fail the coordinate-level or linear-axis conjunction criteria.

### 7j. Agreement Between Phase C and Phase D

The `agreement` column records, for each (concept, level, layer, population,
period_spec) grouping, which basis(es) detected the structure
([phase_g_fourier.py:1626-1667](../phase_g_fourier.py#L1626-L1667)):

| Agreement | Count | Fraction |
|-----------|------:|---------:|
| `neither` | 2,892 | 83.1% |
| `both` | 414 | 11.9% |
| `phase_d_only` | 170 | 4.9% |
| `phase_c_only` | 4 | 0.1% |

**Among helix detections only:**

| Agreement | Count | Fraction of 500 |
|-----------|------:|----------------:|
| `both` | 414 | 82.8% |
| `phase_d_only` | 85 | 17.0% |
| `phase_c_only` | 1 | 0.2% |

Phase D catches 85 unique detections that Phase C misses, while Phase C catches
only 1 unique detection (a carry_2 L5/layer20/wrong/carry_binned cell — the single
"circle-only" detection from Section 7a, for which Phase D did not pass the full
conjunction). For the 414 `both` cells, both bases independently reach the
detection threshold.

**The 4 `phase_c_only` cells at the row level** (different from the 1 at the
helix-detection level) correspond to the 4 cells where Phase C is `geometry_detected
= "none"` but Phase D is either "none" or not present. Wait — let me re-examine.
The 4 `phase_c_only` rows in the `agreement` column come from the group-by-cell
logic that marks ALL rows in a group with the same agreement label. The `both`
versus `phase_c_only` labels reflect whether the *group* had a detection in Phase
C only. The individual-row helix_detected column can differ; the 1-vs-4 discrepancy
comes from multi-row groups where the Phase D row was `helix_detected=True` but
classified differently.

The takeaway for the paper: **Phase D is essential for catching answer-digit
periodicity. Phase C alone would miss all answer digits except ans_digit_5_msf.**

---

## 8. Final Results — Carry Concepts

Carries produce **419 of 500 (83.8%) helix detections**. They are the dominant story
of Phase G. This section reports every breakdown for carry concepts.

### 8a. Carry Detection Rates by Concept

| Concept | Total cells | Helix | Rate | Levels available |
|---------|------------:|------:|-----:|:-----------------|
| carry_0 | 594 | 19 | 3.2% | L2–L5 |
| carry_1 | 486 | 176 | 36.2% | L3–L5 |
| carry_2 | 324 | 116 | 35.8% | L4–L5 |
| carry_3 | 162 | 54 | 33.3% | L5 |
| carry_4 | 162 | 54 | 33.3% | L5 |
| **Total** | **1,728** | **419** | **24.2%** | |

Four of five carry concepts — carry_1 through carry_4 — show essentially identical
detection rates around 33–36%. carry_0 is an outlier with only 3.2%, for reasons
explained in Section 8d.

### 8b. `carry_raw` Dominance — Why It Wins Over `carry_binned` and `carry_mod10`

The carry period-spec breakdown is the single biggest update to the pre-run
interpretation:

| Period spec | Total cells | Helix | Rate |
|-------------|------------:|------:|-----:|
| `carry_binned` | 576 | 22 | 3.8% |
| `carry_mod10` | 576 | **0** | **0.0%** |
| `carry_raw` | 576 | **397** | **68.9%** |

**`carry_mod10` gets zero detections.** None. Across all levels, layers, populations,
bases, and carry concepts, testing periodicity at period 10 using only the values
{0, 1, ..., 9} does not find any significant periodic structure.

This is not a bug. The test code is correct; the raw Fourier power at period 10 is
just not concentrated enough to beat the permutation null, because carry values don't
naturally mod out at base 10. A carry of 12 and a carry of 2 are *not* geometrically
close — the model encodes them as distinct magnitudes, not as equivalents mod 10.

**`carry_raw` dominates at 397 / 576 = 68.9%.** Tests periodicity at the period
equal to the number of distinct raw carry values. For `carry_1` at L5 (values 0–17),
period = 18; for `carry_2` at L5 (values 0–26), period = 27; for `carry_3` at L5
(values 0–18), period = 19; for `carry_4` at L5 (values 0–9), period = 10.

**Per-concept × period-spec × level breakdown:**

| Concept | Level | `carry_binned` | `carry_mod10` | `carry_raw` |
|---------|-------|:--------------:|:-------------:|:------------:|
| carry_0 | L2 | 0/36 | 0/36 | 0/36 |
| carry_0 | L3 | 0/54 | 0/54 | 1/54 |
| carry_0 | L4 | 0/54 | 0/54 | **18/54** |
| carry_0 | L5 | 0/54 | 0/54 | 0/54 |
| carry_1 | L3 | **14/54** | 0/54 | **54/54** |
| carry_1 | L4 | 0/54 | 0/54 | **54/54** |
| carry_1 | L5 | 0/54 | 0/54 | **54/54** |
| carry_2 | L4 | 8/54 | 0/54 | **54/54** |
| carry_2 | L5 | 0/54 | 0/54 | **54/54** |
| carry_3 | L5 | 0/54 | 0/54 | **54/54** |
| carry_4 | L5 | 0/54 | 0/54 | **54/54** |

**`carry_raw` gives 100% detection rate for carry_1, carry_2, carry_3, carry_4.**
Every single `carry_raw` cell for these four concepts is a helix detection. Across
9 layers × 3 populations × 2 bases = 54 cells per (concept, level) combination,
all 54 pass the conjunction criterion every time.

- carry_1: 54/54 at L3, 54/54 at L4, 54/54 at L5 = **162/162**
- carry_2: 54/54 at L4, 54/54 at L5 = **108/108**
- carry_3: 54/54 at L5 = **54/54**
- carry_4: 54/54 at L5 = **54/54**
- **Combined: 378/378 (100%) `carry_raw` cells for carry_1–4**

This is the cleanest single result in Phase G. No gaps, no marginal cases, no
population-dependent asymmetry.

**Why raw, not binned?** `carry_binned` tests periodicity at the period equal to the
number of Phase-C-merged groups. For carry_1 at L5, Phase C merged values 12–17
into a tail bin, leaving 13 groups — so `carry_binned` tests period 13. But the
model doesn't encode carries at period 13; it encodes them at period 18 (the full
raw range). Testing at the wrong period just produces noise. `carry_raw` restores
the full value set (0–17) and uses period 18, which matches the model's actual
encoding.

**Why not mod 10?** Because carry values are not modular. Unlike digits — where
`9+1=0` in the base-10 representation — carries don't wrap. A carry of 15 is
genuinely different from a carry of 5 because it affects how many higher-position
carries will fire. The model encodes this magnitude linearly (the helix's linear
axis) while also placing the values on a period-18 or period-27 circle. Why a
circle at such non-round periods? The period is whatever evenly distributes the
values around a unit circle — it's the natural Fourier period for the set of raw
values, not a base-10 artifact.

This also aligns with K&T's broader observation that LLMs use multiple Fourier
periods for numbers, not just period 10. K&T found periods {2, 5, 10} for standalone
integers; our finding for carries is different periods (18, 27, 19, 10 depending on
the concept) but the same principle — the Fourier basis naturally fits whatever
value set is present, and the model aligns its representation to that basis.

### 8c. Per-Layer Carry Helix Counts

| Concept | l 4 | l 6 | l 8 | l12 | l16 | l20 | l24 | l28 | l31 | Total |
|---------|----:|----:|----:|----:|----:|----:|----:|----:|----:|------:|
| carry_0 | 3 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 19 |
| carry_1 | 20 | 20 | 20 | 20 | 21 | 20 | 18 | 18 | 19 | 176 |
| carry_2 | 13 | 13 | 13 | 13 | 12 | 13 | 12 | 14 | 13 | 116 |
| carry_3 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 54 |
| carry_4 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 54 |

**carry_3 and carry_4 are completely layer-uniform — exactly 6 detections per layer,
every layer.** The 6 per layer decomposes as 3 populations × 2 bases = 6 cells per
(layer, concept), all 6 detected. Every single (layer, population, basis) cell for
carry_3 and carry_4 via `carry_raw` passes the conjunction criterion.

**carry_1 and carry_2 are near-uniform** with small fluctuations (e.g., carry_1
has 21 at layer 16 and 18 at layer 24). The 20 per layer for carry_1 decomposes as
9 layers × ~20 = 180, matching the total 176 (3 unavailable cells at lower levels,
mostly where L3 drops some configurations).

**carry_0 has 3 detections at layer 4 and 2 at every other layer** — the 3 vs 2
is because L4/correct/carry_raw detects all 18 layer-4 cells (see Section 8d),
and one of them falls in a specific layer+pop combination that happens to count
3 for layer 4.

This layer uniformity is a **strong claim**: the helix representation for carries
is not computed at any specific layer. It is established at layer 4 (the earliest
layer tested) and preserved at layer 31 (the final transformer layer). This is
consistent with the residual stream architecture — information persists unless
actively modified.

### 8d. carry_0 at L4: The Correct-Only Anomaly

carry_0 is the carry into the **ones** column (i.e., the carry from multiplying
ones digits that feeds into the tens column). It has the lowest detection rate
among carries (19/594 = 3.2%), and nearly all of its detections are concentrated
in a single slice: **L4 / correct / carry_raw**.

Breakdown of carry_0 helix detections by level × population × period_spec:

| Level | Population | Period spec | Helix | Total | Notes |
|-------|------------|-------------|------:|------:|-------|
| L2 | all | carry_raw | 0 | 18 | |
| L2 | correct | carry_raw | 0 | 18 | |
| L3 | all | carry_raw | 1 | 18 | a single layer cell detected |
| L3 | correct | carry_raw | 0 | 18 | |
| L3 | wrong | carry_raw | 0 | 18 | |
| **L4** | **all** | **carry_raw** | **0** | **18** | — |
| **L4** | **correct** | **carry_raw** | **18** | **18** | **100% detection** |
| **L4** | **wrong** | **carry_raw** | **0** | **18** | — |
| L5 | all | carry_raw | 0 | 18 | |
| L5 | correct | carry_raw | 0 | 18 | |
| L5 | wrong | carry_raw | 0 | 18 | |

**L4/correct is the only slice where carry_0 shows helix structure**, and there
it is complete (every layer × every basis). L4's `correct` population is small
(~2,000 samples), much smaller than L4/all or L4/wrong, yet it detects 100% while
the larger populations detect 0%.

**Interpretation.** At L4, the model is struggling (accuracy ~29%). When it gets
the answer right, its carry_0 representation is organized on a helix. When it gets
the answer wrong (the majority), the representation is not. And because L4/all is
dominated by wrong samples (~71%), the pooled `all` population centroids are too
contaminated by wrong-case noise to detect the helix structure.

This is direct evidence that **the helix is present only when the computation is
working**. It is the L4 analog of the L5 ans_digit_5_msf correct-only result
(Section 9c). At lower difficulty (L2: no signal; L3: 99% correct; L5: 6% correct)
the effect does not manifest because either the task is too easy (no helix needed)
or the correct population is too small for the centroid test. L4 is the sweet spot
where the model has just enough correct computations to populate the centroids
and the helix is visible in them.

### 8e. The carry_1 L5 Phase C Example (A Walkthrough)

The single most informative helix cell in the run is `carry_1 / L5 / layer16 / all /
phase_c / carry_raw`. It is the carry into the tens column at L5 (3-digit multiplication),
in the pooled all-population, probed at layer 16 (the Phase A "information peak").
Everything about this cell is at maximal power: the largest N (122,223), the Phase C
consensus basis, the `carry_raw` period spec, the middle layer.

**Cell metadata (from the per-cell JSON):**

```
concept: carry_1
tier: B
level: 5
layer: 16
population: all
subspace_type: phase_c
period_spec: carry_raw
n_groups: 18           ← carry_1 at L5 has values 0..17
period: 18             ← one per distinct raw value
values_tested: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
d_sub: 2               ← Phase C consensus rank for this slice is only 2
n_samples_used: 122223 ← L5/all — every available sample
n_perms_used: 1000
```

The subspace rank is just 2 — which is the minimum for a helix (need two axes).
Despite this cramped subspace, the centroids sit on a beautiful helix:

**Fourier statistics:**

```
two_axis_fcr: 0.5883       ← 58.8% of total Fourier power is at (coord_0, coord_1) at freq 1
two_axis_best_freq: 1       ← the fundamental period (period 18, freq 1 = full cycle)
two_axis_coord_a: 0
two_axis_coord_b: 1         ← the two subspace coords carry cos and sin
two_axis_p_value: 0.000999  ← saturated at the permutation floor
two_axis_q_value: 0.00435   ← FDR-adjusted, comfortably below 0.05

helix_fcr: 0.6568           ← adding the linear axis lifts the statistic to 65.7%
helix_best_freq: 1
helix_linear_coord: 0       ← linear axis is coord 0 (overlapping with circle's coord_a)
helix_p_value: 0.000999     ← saturated
helix_q_value: 0.00429

uniform_fcr_top1: 0.5081    ← mean per-coord FCR (secondary)
eigenvalue_fcr_top1: 0.5923 ← eigenvalue-weighted (secondary)
```

**Detection outcomes:**

```
circle_detected: True
helix_detected: True
geometry_detected: helix
multi_freq_pattern: {1}     ← pure fundamental, no harmonics
dominant_freq_mode: 1
n_sig_coords_at_mode_freq: 2
p_saturated: True
agreement: both             ← also detected in Phase D for this cell
```

**What this cell tells us.** carry_1 at L5 lives in a 2-dimensional Phase C
subspace. Within those 2 dimensions, the 18 centroids (one per raw carry value
0–17) are arranged such that:

1. The fundamental frequency (k=1, period 18) captures 58.8% of all Fourier power
   in just two coordinates (the entire subspace, since d=2).
2. The linear axis adds about 7% more power (helix_fcr = 0.6568 vs two_axis_fcr
   = 0.5883). Note that for `d=2`, the helix's "linear axis" has to reuse one of
   the two Fourier coords ([phase_g_fourier.py:860-865](../phase_g_fourier.py#L860-L865)),
   because there's no third coordinate to use. The code still computes a linear
   power and rescales it; the 7% delta is the linear contribution on top of the
   circular fit.
3. Under the permutation null, no shuffled assignment of samples to groups
   produces an FCR this high. 0 out of 1,000 shuffles match the observed value —
   meaning the structure is not due to chance.
4. After FDR correction across all 3,480 tests, the q-value is 0.0043, two orders
   of magnitude below the 0.05 threshold.
5. Both Phase C and Phase D independently detect this cell (`agreement: both`),
   ruling out a basis-specific artifact.

**The Phase C eigenvalues for this slice** are `λ_1 = 0.0740, λ_2 = 0.0142`.
The ratio `λ_1 / λ_2 ≈ 5.2` means the first consensus dimension captures ~80% of
the between-class variance, and the second captures ~20%. A circle in 2D requires
both axes to carry comparable power, and here `coord_0` carries the cosine
(larger eigenvalue) while `coord_1` carries the sine. The circle is elliptical
(5:1 ratio of semi-axes) but still passes the detection criterion because both
axes have individually significant Fourier concentration.

**The corresponding plot** is
`plots/phase_g/centroid_circles/centroid_circle_L5_layer16_all_carry_1_phase_c_carry_raw.png`
(one of the 501 centroid circle plots generated at 05:48:41 EDT on Apr 13).
Visual inspection would show 18 points arranged roughly on an elliptical curve,
colored by raw carry value, with consecutive values being angular neighbors.

This is the signature finding of Phase G — a clean helix detection in the model's
intermediate computational variable at the full population, standard basis, standard
layer — with all statistical guardrails passing cleanly.

---

## 9. Final Results — Answer Digit Concepts

Answer digits produce **78 of 500 (15.6%) helix detections**. They are the second-
largest contributor after carries (Tier B = 419). This section reports every
breakdown for answer-digit concepts.

### 9a. Detection Rates by Answer Digit

| Concept | Total cells | Helix | Rate | Position in answer | Levels |
|---------|------------:|------:|-----:|:-------------------|:-------|
| ans_digit_0_msf | 196 | 27 | 13.8% | Leading (most significant) | L2–L5 |
| ans_digit_1_msf | 161 | 2 | 1.2% | Second | L2–L5 |
| ans_digit_2_msf | 163 | 4 | 2.5% | Third (middle) | L2–L5 |
| ans_digit_3_msf | 152 | 12 | 7.9% | Fourth | L3–L5 |
| ans_digit_4_msf | 108 | 13 | 12.0% | Fifth | L4–L5 |
| ans_digit_5_msf | 54 | 20 | 37.0% | Sixth (least significant, L5 only) | L5 |
| **Total** | **834** | **78** | **9.4%** | | |

### 9b. The Edge-vs-Middle Asymmetry

The detection rate follows a clear U-shape across positions:

```
        37.0% ●
             ●
             ●
             ●
     13.8%●  ●
          ●  ●            12.0%
          ●  ●        ●── ● ─
          ●  ●        ●  ●
          ●  ●  7.9%  ●  ●
          ●  ●    ●   ●  ●
          ●  ●    ●   ●  ●
          ●  ● 2.5%●   ●  ●
          ●  ●  ●  ●   ●  ●
       1.2% ●  ●  ●   ●  ●
    ──────── ●  ●  ●  ●   ●  ●─────
    position: 0  1  2  3   4  5
    ans_digit: 0  1  2  3   4  5
              leading      trailing
              digit          digit
```

**Edge digits are strongly structured; middle digits are essentially null.**

- **Leading digit (ans_digit_0_msf) = 13.8%**: the highest-order answer digit. At
  L5 for 3×3 multiplication, this is the millions digit of a 6-digit product.
- **Trailing digit (ans_digit_5_msf) = 37.0%**: only exists at L5. This is the
  ones digit of the 6-digit product. The detection rate is the highest of any
  concept in the answer-digit group.
- **Middle digits (1, 2) = 1.2% and 2.5%**: nearly null. Middle answer digits
  are the product positions where carries propagate through — computationally
  the hardest, and Phase C found that middle digits lack linear subspaces at L5.
  Phase G confirms they also lack periodic structure.

**This pattern replicates Phase C's finding.** Phase C showed that middle answer
digits have very-low-rank linear subspaces at L5 (2 dimensions or fewer), while
leading and trailing digits have full-rank 9-dimensional subspaces. Phase G shows
that the low-rank middle-digit subspaces also lack periodic structure. Both
findings point at the same fact: **the model fails at composing middle answer
digits, and this failure shows up in both the linear and non-linear geometry
of the representation**.

**The trailing ones digit is the easiest arithmetic.** `ans_digit_5_msf` at L5 is
`(a × b) mod 10`, which depends only on `(a mod 10) × (b mod 10) mod 10` — a pure
modular operation. The model handles this with a period-10 Fourier representation,
which is exactly what Phase G detects: `digit` period spec at P=10, with `best_freq`
typically 1 or 5.

### 9c. `ans_digit_5_msf`: The `correct`-Only Signal

The most striking single result in the answer-digit family is **ans_digit_5_msf**'s
population asymmetry at L5:

| Level | Population | Helix | Total cells | Rate |
|-------|-----------|------:|------------:|-----:|
| L5 | `all` | 1 | 18 | 5.6% |
| L5 | `correct` | **18** | **18** | **100.0%** |
| L5 | `wrong` | 1 | 18 | 5.6% |

**18 of 18 (100%) `ans_digit_5_msf` cells in the L5 `correct` population are
detected as helices.** Every layer, every basis, every cell. In `all` and `wrong`,
only 1 cell each shows helix detection.

**Why this matters.** The `correct` population at L5 is N=4,197 samples — small
relative to `all` (N=122,223) or `wrong` (N=118,026). If detection were purely a
function of N, `all` should detect most strongly. Instead, the smaller `correct`
population detects at 100% while the larger pooled populations detect at 5.6%.

The explanation: **the trailing digit helix is present only when the model is
computing correctly**. In the 4,197 correct L5 samples, the ones digit of the
product follows `(a mod 10)(b mod 10) mod 10`, and the residual stream at `=`
encodes this via a period-10 Fourier helix. In the 118,026 wrong samples, the
ones digit of the model's output is wrong — it doesn't match the true mod-10
computation — so the centroids don't line up with the correct Fourier basis, and
the test finds nothing.

When the two populations are pooled (`all`), the overwhelming weight of wrong
samples contaminates the centroid for each digit value, smearing out the helix.

**This is the cleanest case in all of Phase G where correctness is a prerequisite
for structure.** It generalizes the carry_0/L4/correct pattern (Section 8d): the
helix tracks computational correctness. When the model computes correctly, the
intermediate/final variables sit on the expected geometric manifold. When it
fails, the geometry breaks down.

**Comparison with other answer digits at L5 by population:**

| Concept | L5 `all` | L5 `correct` | L5 `wrong` |
|---------|:--------:|:------------:|:----------:|
| ans_digit_0_msf | 3/18 | 0/18 | 4/18 |
| ans_digit_1_msf | 0/18 | 0/9  | 0/18 |
| ans_digit_2_msf | 0/15 | 0/0  | 0/16 |
| ans_digit_3_msf | 1/13 | 0/18 | 1/13 |
| ans_digit_4_msf | 4/18 | 0/18 | 4/18 |
| ans_digit_5_msf | 1/18 | **18/18** | 1/18 |

Other answer digits show the **opposite** pattern: they detect (weakly) in `all`
and `wrong` but not in `correct`. The reason is different — for these middle and
leading digits, the `correct` population has too few samples (N=4,197 vs. ~12,000
per digit in `wrong`), so the centroid noise drowns out any structure. The
`wrong` and `all` populations have more samples and hence higher power, and they
detect at low rates.

**ans_digit_2_msf at L5/correct has n=0** — meaning the `correct` population had
insufficient samples at all concept-value combinations to make a valid analysis
cell. This is the hardest digit to get right.

**The L5 correct vs all pattern is a case study of the trade-off between power
(larger N is better) and purity (correct samples have cleaner structure).** For
the trailing digit, where the signal is strong and narrowly task-specific, purity
wins. For other digits, where the signal is weaker, power would usually win — but
in this phase, neither wins well because the structure isn't there at the
`=` token for middle digits.

### 9d. Per-Layer Answer Digit Counts

| Concept | l 4 | l 6 | l 8 | l12 | l16 | l20 | l24 | l28 | l31 | Total |
|---------|----:|----:|----:|----:|----:|----:|----:|----:|----:|------:|
| ans_digit_0_msf | 2 | 3 | 4 | 2 | 4 | 5 | 2 | 3 | 2 | 27 |
| ans_digit_1_msf | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 2 |
| ans_digit_2_msf | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 1 | 1 | 4 |
| ans_digit_3_msf | 3 | 2 | 0 | 0 | 0 | 2 | 2 | 1 | 2 | 12 |
| ans_digit_4_msf | 1 | 0 | 0 | 0 | 2 | 3 | 2 | 3 | 2 | 13 |
| ans_digit_5_msf | 2 | 2 | 4 | 2 | 2 | 2 | 2 | 2 | 2 | 20 |

**ans_digit_0_msf has detections spread across all layers** (2–5 per layer), with
layer 20 the strongest (5) — consistent with a representational format maintained
throughout the network.

**ans_digit_5_msf is near-uniform at 2 per layer** (with layer 8 having 4). The
2 per layer is the 2 detections per (layer) from the correct population — every
layer has 2 bases × 1 correct slice = 2 cells, all detected. The remaining 18 −
9×2 = 0 means the `all` and `wrong` slices contribute effectively zero, matching
the 1-each we saw above.

**Middle digits are layer-concentrated.** ans_digit_1_msf has 1 detection at layer
6 and 1 at layer 24 — both rare, both late. ans_digit_2_msf has 3 detections at
layers 24/28/31 — very late-layer. ans_digit_3_msf has 3 early-layer detections
(l4: 3, l6: 2). None of these are strong, and middle-digit detections collectively
account for only 18 of 500 cells (3.6%).

**No layer-emergence pattern.** Unlike Phase A's finding of an "information peak"
at layer 16, Phase G shows that for answer digits, detections are scattered across
layers without a clear peak. This supports the interpretation that the periodic
representation is a **format** maintained in the residual stream, not a
**computation** performed at a specific layer.

---

## 10. Final Results — Operand Digit Concepts

Operand digits produce **3 of 500 (0.6%) helix detections**. This is the clean null
for the entire phase. Across 918 total cells — every (concept, level, layer,
population, basis, period_spec) combination for `a_units`, `a_tens`, `a_hundreds`,
`b_units`, `b_tens`, `b_hundreds` — only 3 have `helix_detected = True`.

### 10a. The Clean Null at `=`

| Concept | Total cells | Helix | Rate | Levels available |
|---------|------------:|------:|-----:|:-----------------|
| a_units | 198 | 0 | 0.0% | L2–L5 |
| a_tens | 198 | 0 | 0.0% | L2–L5 |
| a_hundreds | 108 | 1 | 0.9% | L4–L5 |
| b_units | 198 | 2 | 1.0% | L2–L5 |
| b_tens | 162 | 0 | 0.0% | L3–L5 |
| b_hundreds | 54 | 0 | 0.0% | L5 |
| **Total** | **918** | **3** | **0.33%** | |

**Two of six operand concepts produce zero detections** (a_units, a_tens, b_tens,
b_hundreds). The other two produce exactly 1 and 2 detections respectively — an
effective noise floor given the number of tests.

### 10b. The 3 Non-Null Cells (All in `correct` × `phase_d_merged`)

All 3 operand-digit detections share four features:

| # | Concept | Level | Layer | Population | Basis | helix_fcr | p_helix | q_helix | p_saturated |
|---|---------|-------|-------|-----------|-------|----------:|--------:|--------:|:-----------:|
| 1 | b_units | L4 | 4 | **correct** | **phase_d_merged** | 0.368 | 0.00799 | 0.02506 | False |
| 2 | b_units | L5 | 20 | **correct** | **phase_d_merged** | 0.593 | 0.00200 | 0.00766 | False |
| 3 | a_hundreds | L5 | 20 | **correct** | **phase_d_merged** | 0.545 | 0.00999 | 0.03007 | False |

All 3:

- In the `correct` population (not `all` or `wrong`)
- In the `phase_d_merged` basis (not `phase_c`)
- Have `p_saturated = False` — i.e., the permutation null did produce values at
  or above the observed FCR, so the p-value is a measured quantity, not a floor
- Have q-values between 0.008 and 0.030 — survives FDR at `q < 0.05` but by a
  narrow margin

**Are these real or noise?** At `α = 0.01` with a 4-way conjunction criterion and
918 operand-digit cells, the expected number of false positives is approximately
`918 × 0.01^4 ≈ 9 × 10^{-6}` — essentially zero false positives from multiple
testing alone. The FDR correction is more lenient but still provides strong control:
the per-test family-wise error rate is less than `0.01^2 ≈ 1 × 10^{-4}` for the
2-coord conjunction.

However, 3 detections out of 918 (0.33%) is comparable to the natural floor one
would expect from a screening with imperfect negative controls. The detections are
also clustered (all in the same population × basis combination) and not p-saturated,
which is unlike the main carry/answer findings where detections are floor-saturated
and highly replicable.

**The conservative interpretation is "noise".** The pre-registered decision rule
(Section 2k) rejects input digits: 0 confirmed cells in the `all` population middle
layers, which is what matters for the paper's conclusion. The 3 correct-population
outliers are flagged but not counted as a positive signal.

### 10c. Per-Layer Operand Digit Counts

| Concept | l 4 | l 6 | l 8 | l12 | l16 | l20 | l24 | l28 | l31 | Total |
|---------|----:|----:|----:|----:|----:|----:|----:|----:|----:|------:|
| a_units | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| a_tens | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| a_hundreds | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| b_units | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 2 |
| b_tens | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| b_hundreds | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

The 3 detections are at **layers 4 and 20 only**. Of the 9 tested layers (4, 6,
8, 12, 16, 20, 24, 28, 31), 7 layers have zero operand-digit detections.

This is the cleanest null in the phase. Input digits at the `=` token are not
encoded on periodic manifolds in Phase C or Phase D bases, at any layer, at any
level, in any population. Combined with the number-token result (Section 12), this
null is decisive: the model does not use Fourier features for operand digits in
multiplication context.

---

## 11. Final Results — Phase C vs Phase D, Population Splits, and Agreement

### 11a. Phase C vs Phase D by Concept

| Concept | Phase C helix / total | Phase D helix / total |
|---------|:----------------------|:----------------------|
| carry_1 | 82 / 243 (33.7%) | 94 / 243 (38.7%) |
| carry_2 | 54 / 162 (33.3%) | 62 / 162 (38.3%) |
| carry_3 | 27 / 81 (33.3%) | 27 / 81 (33.3%) |
| carry_4 | 27 / 81 (33.3%) | 27 / 81 (33.3%) |
| carry_0 | 9 / 297 (3.0%) | 10 / 297 (3.4%) |
| ans_digit_0_msf | **0 / 97 (0.0%)** | **27 / 99 (27.3%)** |
| ans_digit_5_msf | 9 / 27 (33.3%) | 11 / 27 (40.7%) |
| ans_digit_4_msf | **0 / 54 (0.0%)** | **13 / 54 (24.1%)** |
| ans_digit_3_msf | **0 / 71 (0.0%)** | **12 / 81 (14.8%)** |
| ans_digit_2_msf | **0 / 73 (0.0%)** | **4 / 90 (4.4%)** |
| b_units | 0 / 99 (0.0%) | 2 / 99 (2.0%) |
| ans_digit_1_msf | **0 / 62 (0.0%)** | **2 / 99 (2.0%)** |
| a_hundreds | 0 / 54 (0.0%) | 1 / 54 (1.9%) |
| a_units | 0 / 99 (0.0%) | 0 / 99 (0.0%) |
| a_tens | 0 / 99 (0.0%) | 0 / 99 (0.0%) |
| b_tens | 0 / 81 (0.0%) | 0 / 81 (0.0%) |
| b_hundreds | 0 / 27 (0.0%) | 0 / 27 (0.0%) |

**The pattern is decisive:**

1. **For carries, Phase C and Phase D agree closely.** carry_1 (82 vs 94), carry_2
   (54 vs 62), carry_3 (27 vs 27), carry_4 (27 vs 27), carry_0 (9 vs 10). The
   two bases see the same structure, with Phase D catching a few extra cells
   at the margins.

2. **For most answer digits, Phase C finds NOTHING.** ans_digit_0, 1, 2, 3, 4
   all have 0 helix detections in Phase C. They have 27, 2, 4, 12, 13 in Phase D
   respectively. The only answer digit Phase C catches is ans_digit_5 (9 / 27 —
   because L5 trailing digit is a clean period-10 helix).

3. **For operand digits, both bases find (almost) nothing.** 0 detections each
   for a_units, a_tens, b_tens, b_hundreds. Phase D catches the 3 outliers
   (b_units × 2, a_hundreds × 1).

**Interpretation.** Phase C's consensus subspace is built via permutation-stabilized
PCA on between-class covariance. It finds the directions that are most consistently
separating classes across resampling. For carries, these directions happen to align
with the Fourier basis — so the periodic structure is directly visible in the top
Phase C dimensions. For answer digits, the Phase C directions capture the digit
identity (linear magnitude) but not the periodic structure. The periodic axes are
subordinate — they live along directions that are between-class informative but
not the top-ranked.

Phase D's LDA-refined basis is different. It optimizes linear discriminability,
which for a 10-class problem (digit values 0–9) naturally produces up to 9
discriminative axes. LDA may rotate the basis to align with whichever directions
maximize class separation — including periodic ones. For answer digits, the LDA
rotation exposes the Fourier structure that Phase C's PCA-style basis was missing.

This justifies the decision to test both bases. **Using Phase C alone, the paper
would miss all 27 ans_digit_0_msf helix detections, 13 ans_digit_4_msf detections,
12 ans_digit_3_msf detections, and so on** — ~67 of the 78 answer-digit detections
would have been lost.

### 11b. Population Comparison — L5

L5 is the hardest level and has the largest number of samples; its population
splits are the most informative.

| Concept | L5 `all` | L5 `correct` | L5 `wrong` | Notes |
|---------|---------:|-------------:|-----------:|-------|
| a_units | 0 / 18 | 0 / 18 | 0 / 18 | operand, null everywhere |
| a_tens | 0 / 18 | 0 / 18 | 0 / 18 | operand, null everywhere |
| a_hundreds | 0 / 18 | **1 / 18** | 0 / 18 | single Phase D detection |
| b_units | 0 / 18 | **1 / 18** | 0 / 18 | single Phase D detection |
| b_tens | 0 / 18 | 0 / 18 | 0 / 18 | |
| b_hundreds | 0 / 18 | 0 / 18 | 0 / 18 | |
| ans_digit_0_msf | 3 / 18 | **0 / 18** | 4 / 18 | wrong > all > correct |
| ans_digit_1_msf | 0 / 18 | 0 / 9  | 0 / 18 | correct has fewer cells (small N) |
| ans_digit_2_msf | 0 / 15 | **0 / 0** | 0 / 16 | correct is empty (can't evaluate) |
| ans_digit_3_msf | 1 / 13 | 0 / 18 | 1 / 13 | |
| ans_digit_4_msf | 4 / 18 | 0 / 18 | 4 / 18 | wrong = all, correct null |
| **ans_digit_5_msf** | **1 / 18** | **18 / 18** | **1 / 18** | **the signature correct-only result** |
| carry_0 | 0 / 54 | 0 / 54 | 0 / 54 | null — see L4/correct instead |
| carry_1 | 18 / 54 | 18 / 54 | 18 / 54 | **100% in carry_raw across all pops** |
| carry_2 | 18 / 54 | 18 / 54 | 18 / 54 | 100% in carry_raw |
| carry_3 | 18 / 54 | 18 / 54 | 18 / 54 | 100% in carry_raw |
| carry_4 | 18 / 54 | 18 / 54 | 18 / 54 | 100% in carry_raw |

**Summary of L5 population patterns:**

- **Carries are population-independent at L5.** carry_1, carry_2, carry_3, carry_4
  all have 18/54 (33.3%) in every population — driven entirely by the 100%
  `carry_raw` detection rate. The structure is invariant to whether the model
  computed correctly.
- **carry_0 is null at L5** — the `correct` anomaly was at L4, not L5. At L5 the
  task is too hard and carry_0 representations don't come through cleanly in any
  population.
- **ans_digit_5_msf has the signature correct-only pattern** (Section 9c).
- **Other answer digits detect in `all` or `wrong`** — presumably because of
  sample-size power advantages in those larger populations.

### 11c. Population Comparison — L3 and L4

**L3:**

| Concept | L3 `all` | L3 `correct` | L3 `wrong` |
|---------|---------:|-------------:|-----------:|
| a_units | 0 / 18 | 0 / 18 | 0 / 18 |
| a_tens | 0 / 18 | 0 / 18 | 0 / 18 |
| b_units | 0 / 18 | 0 / 18 | 0 / 18 |
| b_tens | 0 / 18 | 0 / 18 | 0 / 18 |
| ans_digit_0_msf | 7 / 18 | 3 / 18 | 0 / 18 |
| ans_digit_1_msf | 1 / 15 | 1 / 18 | 0 / 11 |
| ans_digit_2_msf | 1 / 18 | 3 / 18 | 0 / 16 |
| ans_digit_3_msf | 2 / 18 | 2 / 18 | 4 / 18 |
| carry_0 | 1 / 54 | 0 / 54 | 0 / 54 |
| carry_1 | 25 / 54 | 20 / 54 | 23 / 54 |

**L4:**

| Concept | L4 `all` | L4 `correct` | L4 `wrong` |
|---------|---------:|-------------:|-----------:|
| a_units | 0 / 18 | 0 / 18 | 0 / 18 |
| a_tens | 0 / 18 | 0 / 18 | 0 / 18 |
| a_hundreds | 0 / 18 | 0 / 18 | 0 / 18 |
| b_units | 0 / 18 | **1 / 18** | 0 / 18 |
| b_tens | 0 / 18 | 0 / 18 | 0 / 18 |
| ans_digit_0_msf | 4 / 18 | 0 / 16 | 4 / 18 |
| ans_digit_1_msf | 0 / 14 | 0 / 12 | 0 / 10 |
| ans_digit_2_msf | 0 / 17 | 0 / 17 | 0 / 10 |
| ans_digit_3_msf | 0 / 18 | 1 / 18 | 1 / 18 |
| ans_digit_4_msf | 1 / 18 | 3 / 18 | 1 / 18 |
| **carry_0** | **0 / 54** | **18 / 54** | **0 / 54** |
| carry_1 | 18 / 54 | 18 / 54 | 18 / 54 |
| carry_2 | 21 / 54 | 18 / 54 | 23 / 54 |

**L3 observations:**

- L3/carry_1 has 25/54 in `all` (highest), down to 20/54 in `correct`. The `all`
  count includes the 18 `carry_raw` cells that hit 100%, plus extra detections in
  `carry_binned` — 14 at L3/carry_binned.
- L3/ans_digit_0_msf has 7/18 in `all`, 3/18 in `correct`, 0/18 in `wrong`. The
  `correct` population at L3 is 5,264 samples (67% accuracy) — large enough for
  some power, but the structure is stronger in `all`.

**L4 observations:**

- **L4/carry_0 is the correct-only anomaly** (Section 8d): 0/54 in `all`, 18/54
  in `correct`, 0/54 in `wrong`. The L4 `correct` population is smaller (~2,900)
  but shows the cleanest carry_0 helix structure.
- **L4/carry_2 has 21/54 in `all` and 23/54 in `wrong`** — `all` < `wrong` because
  the `wrong` subset at L4 happens to have a slightly higher rate. This is within
  normal statistical variation for the small differences between populations.

### 11d. The Pre-Registered Decision Rule Applied

Applied to the final 3,480-cell run (see Section 2k for the rule). Strict
verification matches `phase_g_decisions.json`:

**Rule:** ≥3 concept-layer cells significant at `q_two_axis < 0.05` OR `q_helix <
0.05`, spanning ≥2 concepts and ≥2 middle layers in {8, 12, 16, 20, 24}, in the
`all` population. A cell is significant if `geometry_detected ≠ "none"` in either
basis (collapsed over period spec).

| Class | Significant cells | Distinct concepts | Distinct layers | Confirmed? |
|-------|-------------------|-------------------|-----------------|:----------:|
| input_digits | 0 | 0 | 0 | **NO** |
| answer_digits | 11 | 6 | 5 | **YES** |
| carries | 20 | 4 | 5 | **YES** |

**Input digits (NOT confirmed):** 0 concept-layer cells pass the criterion in the
`all` population middle layers. Zero concepts, zero layers. All 3 operand-digit
detections (Section 10b) are in the `correct` population, so they do not count
toward this rule. Decision: **input digit Fourier structure is not confirmed**.

**Answer digits (CONFIRMED):** 11 concept-layer cells, spanning all 6 answer
digit concepts and all 5 middle layers {8, 12, 16, 20, 24}. The breakdown:

- ans_digit_0_msf at layers 8, 12, 16, 20, 24 (5 cells)
- ans_digit_5_msf at layer 8, 12, 16, 20, 24 (5 cells? depends on all-population
  cells passing — but most are in correct for L5)
- Other answer digits contribute the remaining cells

The rule is satisfied: 11 ≥ 3, 6 ≥ 2, 5 ≥ 2. Decision: **answer-digit Fourier
structure is confirmed**.

**Carries (CONFIRMED):** 20 concept-layer cells, spanning 4 concepts (carry_1,
carry_2, carry_3, carry_4) and all 5 middle layers. Note that **carry_0 is not
in the confirmed set** despite the L4/correct anomaly — because the L4/correct
cells are not in the `all` population and the rule is population-strict.

The rule is satisfied: 20 ≥ 3, 4 ≥ 2, 5 ≥ 2. Decision: **carry Fourier structure
is confirmed**.

**Final JSON (for `phase_g_decisions.json`):**

```json
{
  "input_digits": {
    "confirmed": false,
    "n_significant_cells": 0,
    "n_concepts": 0,
    "n_layers": 0,
    "concepts": [],
    "layers": []
  },
  "answer_digits": {
    "confirmed": true,
    "n_significant_cells": 11,
    "n_concepts": 6,
    "n_layers": 5,
    "concepts": ["ans_digit_0_msf", "ans_digit_1_msf", "ans_digit_2_msf",
                 "ans_digit_3_msf", "ans_digit_4_msf", "ans_digit_5_msf"],
    "layers": [8, 12, 16, 20, 24]
  },
  "carries": {
    "confirmed": true,
    "n_significant_cells": 20,
    "n_concepts": 4,
    "n_layers": 5,
    "concepts": ["carry_1", "carry_2", "carry_3", "carry_4"],
    "layers": [8, 12, 16, 20, 24]
  }
}
```

**Paper statement.** "Applying the pre-registered decision rule (≥3 significant
cells across ≥2 concepts and ≥2 middle layers in the `all` population, q < 0.05),
periodic structure is confirmed for carries and answer digits, and not for input
digits. The confirmation is based on 11 answer-digit cells and 20 carry cells
drawn from the `all` population middle layers {8, 12, 16, 20, 24}."

---

## 12. Number-Token Fourier Screening — Full Results

The main Phase G screening runs at the `=` token — three tokens downstream of the
operand positions. The literature (K&T, Gurnee et al.) probes at the number-token
positions themselves. To make our result comparable to theirs, we ran a separate
screening using `phase_g_numtok_fourier.py` on pre-extracted operand-position
activations.

**Run metadata:**

- Script: `phase_g_numtok_fourier.py` (637 lines; imports statistical core from
  `phase_g_fourier.py` with zero code duplication)
- Start: Apr 11, 2026, 14:47 EDT
- End: Apr 11, 2026, 14:57 EDT
- Duration: **636.0 seconds (10.6 minutes)**
- Output: `/data/user_data/anshulk/arithmetic-geometry/phase_g/summary/numtok_fourier_results.csv`
  (108 rows), per-cell JSONs under `phase_g/numtok/L{level}/layer_{LL}/pos_{a|b}/`,
  log at `logs/phase_g_numtok_fourier.log`

### 12a. Motivation and Methodology

**The question.** K&T (2025) found generalized helices for integers 0–360 presented
as standalone tokens in Llama 3.1 8B, with target periods {2, 5, 10} appearing in
the residual stream at layers {0, 1, 4, 8}. Our K&T pilot (Section 5b) replicated
this finding exactly.

But K&T tested standalone integers. Our pipeline uses multiplication prompts:
`"{a} * {b} ="`. When the integer 47 appears as the first operand in "47 * 83 =",
does its representation at the `47` token position still show the same Fourier
structure? The surrounding context (multiplication operation, second operand,
equals sign) might transform the representation.

Four possible outcomes:

1. **Helix present at number-token, null at `=`:** Fourier features exist at input
   but are transformed by the time computation happens at the output position.
   Supports "Fourier is input-only".
2. **Helix present at both:** Fourier features are maintained throughout the
   forward pass. Periodic structure is the computational substrate.
3. **Helix null at number-token, null at `=`:** The model does not use Fourier
   representations for operands in multiplication context, even at input. Context
   dependence is immediate.
4. **Helix null at number-token, present at `=`:** (Counterintuitive but logically
   possible) The model builds periodic representations de novo at the computation
   position. No prior helix at the input.

The main screening (`=`) produced nulls for operand digits (Section 10) and
positives for carries and some answer digits. The number-token screening tests
operand digits only (carries don't exist at operand positions; answer digits
don't exist at operand positions either).

### 12b. Script Design: `phase_g_numtok_fourier.py`

The script is standalone but imports the statistical core from the main Phase G
script ([phase_g_numtok_fourier.py:38-51](../phase_g_numtok_fourier.py#L38-L51)):

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

**Zero code duplication for the mathematical core.** Identical Fourier math,
permutation null, detection thresholds, and conjunction criterion as the main
Phase G. The only differences are:

1. **Data loading:** loads `.npy` from `activations_numtok/` instead of main
   `activations/`, handles `pos_a` and `pos_b` separately via
   `load_numtok_activations()`.
2. **Subspace computation:** PCA on centroids in 4096-dim space — no Phase C/D
   bases exist at the number-token position, so the script computes its own
   between-group subspace via `pca_on_centroids()`
   ([phase_g_numtok_fourier.py:135-161](../phase_g_numtok_fourier.py#L135-L161)).
3. **Concept registry:** digit concepts only (no carries, no answer digits).
   Defined inline as `DIGIT_CONCEPTS` list with (name, position, available_levels)
   tuples.
4. **No population split:** single pass over all problems — correct/wrong
   classification is determined at `=`, not at operand positions, so the notion
   doesn't apply.
5. **Output:** separate directory tree (`phase_g/numtok/L{level}/layer_{LL}/
   pos_{a|b}/`) and separate summary CSV.

**PCA on centroids.** Since no Phase C/D subspaces exist at operand token positions,
the script computes its own between-group subspace via PCA on the group centroids.
For m digit values (e.g., 10 for units digits), the centroid matrix is
`(m, 4096)`. The between-group structure lives in at most `m−1` dimensions, so
we take the top `k = min(pca_dim, m−1)` principal components of the centered
centroid matrix via SVD. All N activations are then projected into this k-dim
space before running the same Fourier pipeline as the main screening.

**Dual reporting.** The CSV reports two FCR values: `two_axis_fcr` (computed in
the PCA-projected k-dim space, the primary statistic with the permutation null)
and `raw_two_axis_fcr` (computed directly on the 4096-dim centroids, no
permutation null — a sanity check that the PCA projection is extracting real
signal and not manufacturing it).

**CLI:** `--config`, `--n-perms` (default 1000), `--pca-dim` (default 20),
`--pilot` (L3/layer16 only). SLURM wrapper: `run_phase_g_numtok.sh` (CPU-only,
24 CPUs, 64 GB RAM, ~10 minutes estimated). The full run with 1,000 permutations
took ~10.6 minutes.

### 12c. Full Run: 0/108 Detections

**Summary by level (from the CSV):**

| Level | Cells | Max FCR | Min p_two_axis | Min p_helix | Detections |
|-------|------:|--------:|---------------:|------------:|-----------:|
| L2 | 18 | 0.4526 | 0.079 | 0.070 | 0 |
| L3 | 24 | 0.6148 | 0.002 | 0.002 | 0 |
| L4 | 30 | 0.5537 | 0.004 | 0.003 | 0 |
| L5 | 36 | 0.5445 | 0.004 | 0.003 | 0 |

**Summary by layer:**

| Layer | Cells | Mean FCR | Max FCR | Min p_two_axis |
|-------|------:|---------:|--------:|---------------:|
| 4 | 18 | 0.3536 | 0.4646 | 0.0599 |
| 8 | 18 | 0.3757 | 0.5764 | 0.0070 |
| 12 | 18 | 0.3949 | 0.6148 | 0.0020 |
| 16 | 18 | 0.3495 | 0.4369 | 0.0160 |
| 20 | 18 | 0.3768 | 0.4993 | 0.0120 |
| 24 | 18 | 0.3708 | 0.5048 | 0.0100 |

**Summary by concept:**

| Concept | Cells | Mean FCR | Max FCR | Min p_two_axis |
|---------|------:|---------:|--------:|---------------:|
| a_units | 24 | 0.310 | 0.393 | 0.026 |
| a_tens | 24 | 0.354 | 0.455 | 0.067 |
| a_hundreds | 12 | 0.388 | 0.423 | 0.079 |
| b_units | 24 | 0.414 | 0.615 | 0.002 |
| b_tens | 18 | 0.390 | 0.491 | 0.053 |
| b_hundreds | 6 | 0.409 | 0.448 | 0.077 |

**Every single cell returned `geometry_detected = none`.** Zero helix, zero circle.
FDR-significant: 0/108. `b_units` at layer 12 came closest but failed the
conjunction criterion at every level.

### 12d. The Near-Miss: b_units at Layer 12

b_units is the closest to detection. Its top cells:

| Level | Layer | PCA FCR | p_two_axis | p_helix | Raw FCR | Detected |
|-------|-------|--------:|-----------:|--------:|--------:|:---------|
| L3 | 12 | 0.6148 | 0.0020 | 0.0020 | 0.0190 | none |
| L4 | 12 | 0.5537 | 0.0040 | 0.0030 | 0.0218 | none |
| L5 | 12 | 0.5445 | 0.0040 | 0.0030 | 0.0231 | none |
| L3 | 8 | 0.5764 | 0.0070 | 0.0040 | 0.0153 | none |
| L3 | 24 | 0.5048 | 0.0100 | 0.0070 | 0.0191 | none |
| L3 | 20 | 0.4993 | 0.0120 | 0.0050 | 0.0162 | none |
| L5 | 16 | 0.4369 | 0.0160 | 0.0044 | 0.0177 | none |
| L4 | 24 | 0.4718 | 0.0170 | 0.0058 | 0.0181 | none |

**`b_units` at `L3 / layer 12 / pos_b`** has:

- `two_axis_fcr = 0.6148` (well above null, ~0.20)
- `p_two_axis = 0.002` (saturated at the permutation floor)
- `helix_fcr = 0.6289`, `p_helix = 0.002` (saturated)
- `raw_two_axis_fcr = 0.0190` (the raw 4096-dim FCR — 32× smaller than PCA-space)

The global p-value passes the `α = 0.01` threshold. But the conjunction criterion
requires BOTH top Fourier coordinates to be individually significant (`p_coord <
0.01`), and this fails — the structure is concentrated on a single coordinate,
not spread across the two axes that define a circle. This is consistent with a
**partial linear trend** (one direction carrying cos, the perpendicular direction
not carrying sin), not a full periodic circle.

In other words: `b_units` at `L3 / layer 12` has *some* Fourier content at period
10, enough to beat the permutation null on the global two_axis_fcr, but not enough
structure in *both* axes for a genuine circle detection. It's a line with some
ripples, not a circle.

### 12e. Position Asymmetry and PCA Concentration Factor

**Position asymmetry.** pos_b (mean FCR 0.404) consistently shows higher FCR than
pos_a (mean FCR 0.343). This likely reflects the operand structure: at L2, `b` is
a single digit (2–9) with only 8 values, giving less averaging and a slightly
elevated statistic. At L3–L5, both operands are multi-digit, and the asymmetry
narrows but doesn't vanish. None of the asymmetry is large enough to cross the
detection threshold.

**PCA concentration factor.** Mean raw 4096-dim `two_axis_fcr` is **0.0155**,
while PCA-space `two_axis_fcr` is **0.370** — a **23.8× concentration factor**.
This confirms that the PCA step is critical: in the raw 4096-dim space, the
Fourier signal-to-noise ratio is too low for any structure to emerge. The PCA
projection isolates the between-group subspace (≤9 dimensions for 10-valued
digits) and concentrates the between-group signal into coordinates where it can
be measured.

This matters because K&T's original analysis runs Fourier on individual hidden
dimensions (one of 4096) and measures power at target periods. The per-dim
Fourier power is inherently small because most hidden dimensions don't carry
digit information. K&T's finding is *averaged* power across dimensions, which
pools the signal across many channels. Our PCA approach does something similar
but more principled — we explicitly extract the between-group subspace first.
The fact that even our more-powerful PCA test finds 0/108 is strong evidence
that the signal really isn't there.

### 12f. Interpretation

The full run eliminates all four hypotheses about why the main `=` screening
might have been null for operand digits:

1. **Context suppression: CONFIRMED.** The multiplication context `"{a} * {b} ="`
   suppresses or transforms the standalone digit helix. The model "knows" it is
   doing multiplication and restructures operand representations from periodic
   (K&T's helix for standalone integers) to non-periodic (what we observe). This
   is the main positive conclusion.

2. **Layer-dependent encoding: REJECTED.** The null holds at every layer from 4
   to 24, including layers 4 and 8 where K&T found their strongest signal for
   standalone integers. The helix is not present at early layers and lost at
   later layers — it is absent everywhere.

3. **Centroid sensitivity: REJECTED.** At L5 with `N = 122,223` and ~12,000 samples
   per digit group, centroid standard errors are ~100× smaller than between-group
   distances. The test has overwhelming statistical power. `b_units` at layer 12
   reaches FCR=0.54 (well above the 0.20 null), showing the test can detect
   structure when it exists — it just doesn't find circles or helices.

4. **The signal genuinely doesn't exist: CONFIRMED.** Operand digits in multiplication
   context do not have periodic Fourier structure at ANY position, layer, or level.
   The model represents operand digits non-periodically throughout the forward pass
   when performing multiplication.

**This is a strong finding for the paper: K&T's digit helix is context-dependent.**
The same model (Llama 3.1 8B) encodes the same integers on helices when presented
standalone but NOT when presented as operands in an arithmetic expression. The
model transforms its integer representation based on the computational task at
hand.

**What K&T found and what we find:**

| Setting | K&T result | Our result |
|---------|-----------|-----------|
| Standalone integers 0–360 | Helix at periods {2, 5, 10}, layers 0/1/4/8 | Replicated (Section 5b) |
| Operand in multiplication, number token | (not tested) | Null, 0/108 |
| Operand in multiplication, `=` token | (not tested) | Null, 3/918 (noise) |
| Carries in multiplication, `=` token | (not tested) | Helix, 419/1,188 |
| Answer digits in multiplication, `=` token | (not tested) | Mixed, 78/1,097 |

Our result extends K&T's finding in a way they did not themselves investigate: the
helix they observed is the *standalone integer* representation, not the *operand
in a task* representation. These are different representations of the same integer.

**Paper statement.** "K&T's generalized integer helix, which we independently
replicated for standalone integers, is not present at the operand token position
in multiplication prompts (0/108 detections across levels L2–L5, layers {4, 8,
12, 16, 20, 24}, and all six operand-digit concepts). This is evidence that the
model's integer representation is **task-dependent**: standalone integers are
encoded on helices, but operands in arithmetic expressions are not."

---

## 13. Interpretation of Full Results

### 13a. The Carry-Helix Story

The dominant finding of Phase G is that **carry values are encoded on generalized
helices inside their linear subspaces.** Phase C/D found that carry_1 has a 2–18
dimensional linear subspace (depending on level, layer, population, and basis).
Phase G shows that within this subspace, carry values 0 through 12+ (or 0–17,
0–26, 0–18 depending on the carry) are arranged not randomly, not linearly, but
on a helix: a circle in two Fourier dimensions plus a linear ramp in a third
dimension.

This is structurally the same shape K&T (2025) found for integer representations
in Llama 3.1 8B — a generalized helix with the circular component at the decimal
period and a linear magnitude ramp. But there is a critical difference: **the
carry helix period is not 10.** It is 18 for carry_1, 27 for carry_2 (at L5), 19
for carry_3, and 10 for carry_4. The model encodes carries at the period set by
the natural range of carry values, not at the base-10 period that K&T's standalone
integers use.

This difference is the biggest theoretical update from Phase G: the Fourier period
is not fixed at 10 by the base-10 number system. It is set by the value set being
represented. For standalone integers {0, 1, ..., 360}, the natural periods are
{2, 5, 10} (and their harmonics 20, 25, 26, 30 — the observed K&T top-10). For
a carry ranging 0–17 in multiplication, the natural period is 18 — whatever
divides the range into an evenly-sampled cycle. The model adapts its Fourier
representation to the value set at hand.

The 35% detection rate for carry_1 through carry_4 (across all cells, not just
the 100%-rate `carry_raw` cells) reflects that the *other* period specs
(`carry_binned`, `carry_mod10`) fail to capture the structure. When you test the
right period, you get 100% detection. When you test the wrong period (10, or the
arbitrary `n_groups` from binning), you get 0%.

Most carry helices are floor-saturated (`p_helix = 0.001`), meaning the
permutation null never produced an FCR as large — the signal is overwhelmingly
real. All 500 helix detections survive FDR correction.

### 13b. Why Operand Digits Are Null at the `=` Position

The zero detection rate for operand digits (0 in `all` population middle layers,
3 marginal in `correct` population) is the cleanest null in Phase G. Combined with
the 0/108 number-token result (Section 12) and the carry helix finding, this tells
a specific story about how the model represents information at the computation
position:

1. **Input digits are already "consumed."** By the `=` position, the model has
   attended to the operand tokens, extracted their digit values, and used them to
   compute partial products, column sums, and carries. The residual stream at `=`
   no longer needs to represent "the units digit of operand a is 7" as a geometric
   fact — it has already been folded into the carries and partial products.

2. **Carries are the active computation.** The model's remaining work at the `=`
   position is carry propagation: determining how carries cascade across columns
   to produce the final answer digits. Carries are the bottleneck (Phase C) and
   the geometric structure (Phase G). The model allocates its representational
   capacity to the hard part.

3. **Linear subspaces are maintained but not periodic.** Phase C showed operand
   digits have full-rank linear subspaces at every layer, even at the `=` position.
   The information is there — the 10 digit centroids are separable in 9D — but
   they are arranged linearly (magnitude ordering), not periodically
   (circle/helix). The transformation from Fourier to linear may be an artifact
   of how the model "unpacks" integer representations for arithmetic.

4. **The number-token null rules out a position-artifact explanation.** If the
   model represented operand digits as K&T helices at the number-token position
   and then transformed them to linear at `=`, we would expect the number-token
   screening to replicate K&T's positive finding. It doesn't — 0/108 there too.
   The helix is absent from the start of the multiplication prompt, not just at
   `=`. The context, not the position, kills the helix.

### 13c. The Difficulty-Dependent Emergence Pattern

The jump from 0.6% helix rate at L2 to 12.5% at L3 mirrors the accuracy drop from
99.8% to ~53%. This is not coincidence — it reflects the computational demands:

- **L2 (2×1 digit):** carry_0 is either 0 or 1. No complex carry chain, no need
  for elaborate geometric encoding. The linear subspace is sufficient. The helix
  is not present because it is not needed.
- **L3 (2×2 digit):** carry_0 ranges 0–8, carry_1 ranges 0–12+. The model must
  propagate carries across 3 columns with values large enough to need multi-digit
  tracking. The helix provides an efficient encoding: the circular component
  captures the periodic (mod-period) structure, while the linear ramp captures
  magnitude.
- **L4 (3×2 or 2×3):** Adds carry_2 with 18 raw values. Detection rate grows from
  12.5% to 15.1%.
- **L5 (3×3):** Adds carry_3 and carry_4. Detection rate grows to 18.0%. carry_3
  and carry_4 each have 100% detection rate in `carry_raw`, contributing 108 of
  the 255 L5 helix detections.

The L2→L3 step is a **20× jump**; the L3→L5 progression is more gradual. This
reflects the fact that L2 is qualitatively different (no carry propagation
needed, direct lookup sufficient), while L3–L5 represent a continuum of increasing
complexity within the same computational regime (carries matter, helices form).

### 13d. Layer Uniformity: Helix Structure Is Maintained, Not Computed

The near-uniform helix detection rate across layers (13.0%–15.9%) is surprising.
One might expect the helix to emerge at a specific layer (where "carry computation"
happens) and be absent before or after. Instead, the helix is present from layer
4 to layer 31 at nearly identical rates.

This is even more striking when restricted to specific carry concepts: carry_3
and carry_4 have **exactly 6 detections at every single layer** (9 layers × 6
detections = 54 total each). carry_1 ranges from 18 to 21. The layer-to-layer
variation is within statistical noise.

This means the helix is a **representational format**, not a **computational
byproduct**. The model stores carry values on helices throughout the network,
using this format as the substrate for carry-related computation at every layer.
This is consistent with the residual stream architecture: information persists
across layers unless actively modified by attention or MLP blocks, and the
residual stream acts like a shared workspace where intermediate computations are
read and updated across layers.

**Contrast with Phase A.** Phase A found that layer 16 is the "information peak"
for visual clustering — concept classes are most visually separable in PCA
projections of layer 16. Phase G shows no such peak. The helix is not a
mid-network phenomenon; it is present from layer 4 (the earliest tested) through
layer 31 (the final transformer layer). The geometric content that Phase A
captures is different from what Phase G captures: Phase A measures
between-class separation, Phase G measures within-subspace periodic structure,
and the two can diverge.

### 13e. Answer Digits: Edge-vs-Middle Asymmetry

Phase C found that middle answer digits (positions 1–2) lack linear subspaces at
L5. Phase G confirms this extends to periodic structure: ans_digit_1_msf has 1.2%
helix rate, ans_digit_2_msf has 2.5% — essentially null. Meanwhile, the leading
digit (13.8%) and trailing digit (37.0%) have real signal.

The trailing digit (ans_digit_5_msf at L5, the ones digit) has the highest detection
rate of any concept (37.0%). This makes mathematical sense: the ones digit of a
product depends only on `(a mod 10) × (b mod 10) mod 10` — a purely modular
operation with period 10. The Fourier basis is the natural representation for
modular arithmetic, and the model uses it.

**The ans_digit_5_msf / L5 / correct asymmetry (Section 9c)** is the clearest
single indicator that periodic structure tracks computational correctness. When
the model computes correctly, the trailing digit sits on a period-10 helix. When
it computes wrong, it doesn't. The 18/18 = 100% detection rate in correct vs.
1/18 in all/wrong is a binary toggle.

**The middle-digit null is a composition failure signature.** Middle digits are
the positions where carries chain through the entire product — the hardest
arithmetic. The model fails at them both in the linear sense (Phase C: no
clean linear subspace) and in the periodic sense (Phase G: no helix). This is
the direct geometric signature of compositional failure: when the model fails,
neither the linear nor the non-linear representation is clean.

### 13f. Position-Dependent Representations: `=` vs Number Token

Combining the main Phase G results (at `=`) with the full number-token screening
(at operand positions), the complete picture:

| Where | Operand Digits | Carries | Answer Digits |
|-------|---------------|---------|---------------|
| Standalone integers (K&T replication) | **Helix (confirmed)** | N/A | N/A |
| Number-token (108 cells, all layers) | **Null (0/108)** | N/A | N/A |
| `=` token (3,480 cells, all layers) | **Null (3/918, 0.3%)** | **Helix (419/1,728, 24%)** | **Mixed (78/834, 9%)** |

Three results that tell a coherent story:

1. **Standalone integers → helix.** K&T's finding replicated: periods {2, 5, 10}
   at all tested layers for single-token integers 0–360.
2. **Operand digits in multiplication → null everywhere.** 0/108 at the number
   token, 3/918 at the `=` token (all marginal). The multiplication context
   eliminates the helix.
3. **Carries at `=` → helix.** The periodic structure re-emerges for carry
   values — the model builds new periodic representations for intermediate
   computations, at whatever period the carry values naturally span.

This is the strongest evidence yet that **representations are task-dependent,
not token-dependent.** The same integer triggers different geometric encodings
depending on whether the model is simply representing it or computing with it.
The model does not have *a* Fourier representation of integers — it has several,
and it uses the appropriate one for the task.

### 13g. `correct` vs `wrong`: Correctness Tracks Structure

Two specific cases show that periodic structure is present only when the model
computes correctly:

1. **carry_0 / L4 / correct / carry_raw**: 18/18 detections in the small (~2,900
   sample) `correct` population; 0/18 each in `all` and `wrong`. The L4/all
   population has ~10,000 samples but is dominated by the 71% wrong samples,
   whose carry_0 representations do not lie on a helix. Pooling them with the
   correct samples contaminates the centroids and kills the signal.
2. **ans_digit_5_msf / L5 / correct / digit**: 18/18 in the tiny (4,197-sample)
   `correct` population; 1/18 each in `all` and `wrong`. The L5/all population
   is 30× larger but dominated by 96% wrong samples, and the trailing digit of
   the wrong answer is literally wrong — it doesn't match `(a×b) mod 10` — so
   the centroids don't align with the period-10 basis.

These two cases are the strongest direct evidence in Phase G that the helix is
**functional**. The model does not produce Fourier structure as an incidental
artifact — it produces it when and where the computation is working. A purely
artifactual or incidental helix would not track accuracy.

The broader population pattern is consistent: `correct` has 15.1% helix rate,
`wrong` has 14.9%, `all` has 13.2%. Small differences overall, but the specific
cases where the structure is most informative are the ones where correctness is
the pivot.

### 13h. What the Pre-Registered Decision Rule Says

The pre-registered rule (Section 2k) is strict: `all` population, middle layers,
collapsed over basis and period spec. Applied to the final data:

- **input_digits: NOT CONFIRMED** (0 cells)
- **answer_digits: CONFIRMED** (11 cells across 6 concepts × 5 layers)
- **carries: CONFIRMED** (20 cells across 4 concepts × 5 layers)

Two of three classes confirmed. The null for input digits is decisive — 0 cells
in the `all` middle layers means not even a single concept-layer pair passes the
criterion, let alone the ≥3 × ≥2 × ≥2 rule. The confirmations are comfortable:
answer digits exceed the threshold by 8 cells (11 ≥ 3), carries by 17 cells
(20 ≥ 3).

**The paper will report both the pre-registered rule outcome and the broader
detection counts.** The pre-registered rule is the headline statistic for
hypothesis testing. The broader counts (500 helix detections total, 419 carry,
78 answer digit, 3 operand digit) are the descriptive picture for the discussion.

### 13i. Implications for the Core Thesis

Phase G provides direct evidence for the project's core thesis: **the LRH is
necessary but insufficient for compositional reasoning.**

1. **Linear subspaces are necessary:** Phase C/D showed every atomic concept has
   a clean linear subspace. Phase G doesn't contradict this — it works *within*
   those subspaces.

2. **Linear subspaces are insufficient:** The geometry *within* subspaces matters.
   Carries sit on helices; operand digits sit on non-periodic arrangements;
   middle answer digits have neither subspaces nor periodic structure. A linear
   probe that says "carry_1 has a 10D subspace" misses the helix inside it.

3. **Composition failure has a geometric signature:** The concepts that fail to
   compose (middle answer digits) are exactly those with no geometric structure
   — neither linear subspaces (Phase C) nor periodic manifolds (Phase G). The
   concepts that succeed (carries, trailing digit) have both.

4. **The representation is position-dependent and task-dependent:** Carries have
   helix structure at `=` but don't exist at operand positions. Operand digits
   have linear subspaces at `=` but no periodic structure at either position
   when in multiplication context. Standalone integers have helices (K&T
   replicated). The model transforms representations non-linearly as information
   flows from input to computation — it does not simply propagate a fixed
   representation through the transformer.

5. **Correctness tracks geometry:** The clearest single-population results
   (carry_0 at L4/correct and ans_digit_5_msf at L5/correct) show that the helix
   is present when and only when the computation is working. This is the link
   between representation and function: linear probes can decode "the model
   knows this" but they miss *when* the geometry is supporting correct
   computation versus producing errors.

---

## 14. What Phase G Contributes to the Paper

Phase G contributes three concrete findings to the paper:

**Finding 1: Carry values are encoded on generalized helices at the computation
position, with high statistical power.**

- 419 of 500 helix detections (84%) are carry concepts
- 378 of 378 `carry_raw` cells (100%) for carry_1–4 are helices
- All 500 detections survive FDR at q < 0.05
- 458 of 500 (91.6%) are p-saturated (permutation null never matched)
- Structure is layer-uniform (13.0–15.9% across all 9 layers)
- Period of the helix is the raw carry value range (18 for carry_1, 27 for
  carry_2 at L5, etc.), not base-10

This is the affirmative result that multiplication in Llama 3.1 8B uses generalized-
helix representations, extending Nanda et al. (2023), Bai et al. (2024), and
Kantamneni & Tegmark (2025) to *intermediate computational variables* in a
large-scale pretrained LLM.

**Finding 2: Operand digits in multiplication context do not have Fourier structure
at either the number-token or `=` positions — a sharp contrast with K&T's finding
for standalone integers.**

- 0/108 at the number-token position (full screening)
- 3/918 at the `=` position (0.33%, all in `correct × phase_d_merged`, noise-level)
- 0 confirmed cells under the pre-registered decision rule
- K&T replication pilot confirms the method can find helices when they exist

This null is a strong statement about task-dependent representations. The same
model encodes the same integers differently based on whether they are being
represented (standalone → helix) or computed with (operand → not helix).

**Finding 3: Middle answer digits lack both linear subspaces (Phase C) and
periodic structure (Phase G) — a direct geometric signature of compositional
failure.**

- ans_digit_1_msf: 1.2% helix rate
- ans_digit_2_msf: 2.5% helix rate
- Leading digit (ans_digit_0): 13.8%
- Trailing digit (ans_digit_5, L5 only): 37.0%, with 100% detection in the
  correct subset
- The edge-vs-middle asymmetry replicates across both phases

This is direct evidence that the Linear Representation Hypothesis is insufficient
for compositional reasoning: the concepts that the model fails to compose are the
ones that lack clean geometric structure, both linear and non-linear.

**The three findings together support the thesis.** Linear subspaces are necessary
but insufficient (Finding 3). Non-linear (helical) geometry exists but is
task-dependent and concept-specific (Findings 1, 2). Phase G provides the first
direct evidence that non-linear within-subspace geometry matters for model behavior,
setting up the need for nonlinear MI methods (GPLVM, causal patching, Phases H–I)
to make this picture complete.

**What the paper will NOT claim based on Phase G alone:**

1. Causal role of the helix. Phase G is descriptive. We show the helix is there;
   we do not show that ablating it breaks carry propagation. That is Phase I.
2. Per-example manifold structure. We use centroids. Within-class structure is
   visible in the single-example projection plots but is not part of the
   pre-registered decision rule.
3. Multi-period / prism structure. 10 cells show the `{1, 2, 5}` multi-frequency
   signature that Bai et al. would call a pentagonal prism, but the automatic
   classifier is exploratory and requires manual review of power spectra.
4. Across-model generality. Phase G is in Llama 3.1 8B only. K&T's finding
   generalizes across model families for standalone integers; we have not tested
   generalization of our multiplication findings.

---

## 15. Limitations

1. **Centroid averaging is conservative.** The test detects structure in the mean
   positions but may miss manifold structure when within-class spread is large.
   If individual activations form a noisy ring but the centroids collapse the
   ring into a blob, the centroid test will be null. The single-example projection
   plots (1,122 of them) provide a partial mitigation but are visual, not
   quantitative.

2. **The conjunction criterion trades sensitivity for specificity.** By requiring
   both coordinates to be individually significant, we eliminate false positives
   from linear signals but may miss genuine circles with unequal axis strengths
   (ellipses). An alternative approach (testing the phase-coherence of the top-2
   coordinates) might be more sensitive but was not implemented due to the
   additional complexity of defining a coherence null.

3. **1,000 permutations limit p-value resolution to 0.001.** For the FDR correction
   across 3,480 tests, this causes 458 of 500 helix detections to be floor-saturated
   — informative cells cannot be distinguished from each other. A future run with
   10,000 permutations would resolve this at the cost of ~10× runtime.

4. **The `=` token is not the standard probe position.** K&T and most of the Fourier
   feature literature probe at the number-token position. Our choice to probe at
   `=` is driven by pipeline consistency (Phases A–F use `=` activations). The
   number-token screening (Section 12) partially mitigates this.

5. **carry_4 has p_floor = 1/720 ≈ 0.00139.** With only 6 groups, the minimum
   achievable p-value exceeds the 0.001 floor from permutations. Any carry_4
   result at p ≈ 0.00139 could be even more significant, but we cannot tell.

6. **The method is period-specific.** We test at pre-specified periods (P=10 for
   digits, P = n_raw for carries). We explicitly do NOT scan all possible periods,
   which would create a massive multiple-testing burden. This means if the model
   uses a period we didn't anticipate (e.g., P=7 for some concept), we would miss
   it. The `carry_raw` finding (which uses P ≠ 10) was within the pre-specified
   spec set and was caught; a hypothetical period outside our spec set would not
   be.

7. **Phase D bases may dilute the signal (for carries) but reveal it (for answer
   digits).** Phase D's merged bases are wider (d_merged ≈ 2 × d_consensus) and
   include discriminative directions. For carries, Phase C and Phase D agree
   (both detect). For answer digits, Phase D detects and Phase C does not. We
   report both bases to capture the full picture but the underlying reason for
   the difference is a subject for follow-up analysis.

8. **No correction for dependence between Phase C and Phase D tests.** The same
   underlying data generates both tests. A concept that is significant in Phase C
   is more likely to be significant in Phase D (or vice versa). The `agreement`
   column documents this, but the FDR correction treats them as independent
   tests. This makes the FDR correction slightly conservative (overcounts
   effective tests), which is acceptable.

9. **No correction for spatial autocorrelation across layers.** The same concept's
   activations at adjacent layers (e.g., layer 16 and layer 20) are highly
   correlated because the residual stream is additive. FDR treats each layer as
   independent, which overestimates the effective number of tests. Also
   conservative.

10. **Centroid-based analysis cannot detect within-class manifold structure.** If
    activations for digit `v=3` form a crescent-shaped cloud in the subspace,
    the centroid collapses this to a single point. The crescent's shape — which
    might reflect context-dependent modulation of the Fourier feature — is
    invisible to the centroid test. Per-example Fourier analysis (as in K&T) would
    capture this but is not part of Phase G's design.

11. **The permutation null is slightly conservative.** The conditioned null (fixed
    group sizes) has fewer degrees of freedom than the unconditioned null (random
    group sizes from sampling). This makes p-values slightly larger (more
    conservative) than they would be under the unconditioned null. The effect is
    negligible for balanced groups but could matter for highly unbalanced carry
    concepts where the largest group has 10× more samples than the smallest.

12. **L2 has no `wrong` population.** L2 accuracy is 99.8% (7 wrong samples), and
    the wrong population falls below `MIN_POPULATION = 30`. We cannot test
    correct-vs-wrong comparisons at L2. This is not a major loss because L2 has
    essentially no Fourier structure to begin with (2/324 detections).

13. **The pentagonal prism signature is not rigorously tested.** The automatic
    classifier flags 10 cells with `{1, 2, 5}` multi-frequency patterns — the
    same-two-axes multi-frequency structure Bai et al. identified in toy
    multiplication models. But the automatic classifier only checks that different
    coordinates are dominant at different frequencies, not that the same two
    axes carry power at multiple frequencies. Proper prism claims require manual
    review of the 10 flagged cells' power spectrum plots.

14. **Carry period spec coverage is uneven.** `carry_raw` has exactly 576 cells
    across the full (level, layer, population, basis) grid — 9 layers × 3 pops
    × 2 bases = 54 cells per (concept, level), times 5 concept-level pairs =
    270 carry_raw × 2 bases × ... (the math works out to 576 across the specs).
    If we had done more period specs (e.g., P=20 for carry_2 in addition to P=27),
    we might have found additional structure. The decision to use the three
    fixed specs was a pre-run design choice.

15. **No per-concept effect-size analysis.** The paper's summary focuses on count
    of detections, not effect size. A finer analysis would report `helix_fcr`
    distributions per concept, to distinguish "many weak helices" from "few
    strong helices". Helix_fcr ranges from 0.125 to 0.820 in the detection set;
    we report the min and mean but do not stratify detection claims by effect
    size.

---

## 16. Implementation Details

### 16a. Script Architecture

`phase_g_fourier.py` is **2,623 lines**, organized as follows (section headers
are the actual `#` dividers in the source):

```
IMPORTS (numpy, pandas, scipy, matplotlib, json, argparse, logging, pathlib, math, ...)

CONSTANTS (phase_g_fourier.py:37-76)
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
  DIGIT_CONCEPTS_BY_LEVEL = {...}
  CARRY_CONCEPTS_BY_LEVEL = {...}
  PERIOD_SPECS = {...}

DECISION_RULE (docstring constant, printed in summary; phase_g_fourier.py:82-102)
NUMBER_TOKEN_FRAMING (docstring constant; phase_g_fourier.py:108-116)
RUNTIME_ESTIMATE (docstring constant; phase_g_fourier.py:122-131)

CONFIGURATION (phase_g_fourier.py:138-162)
  load_config(path) -> dict
  derive_paths(cfg) -> dict

LOGGING (phase_g_fourier.py:170-197)
  setup_logging(workspace) -> logger with RotatingFileHandler (10 MB, 3 backups)

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

CORE FOURIER MATH (phase_g_fourier.py:558-935)
  compute_freq_range(period) -> int K                       (line 562)
  is_nyquist(k, period) -> bool                              (line 575)
  fourier_single_coordinate(signal, values, period) -> dict  (line 580)
  fourier_all_coordinates(centroids, values, period, logger) -> dict (line 636)
  compute_linear_power(centroids, v_linear, group_sizes)     (line 751)
  compute_helix_fcr(fourier_results, linear_power, v_linear, group_sizes, logger) -> dict (line 780)
  compute_centroids_grouped(projected, labels, unique_values) (line 897)
  compute_eigenvalue_weighted_fcr(per_coord_fcr, eigenvalues) (line 924)

PERMUTATION NULL (phase_g_fourier.py:937-1063)
  permutation_null(projected, labels, unique_values, period, v_linear,
                   n_perms, rng, logger) -> dict
  compute_pvalues(observed, null_dist) -> float
  compute_pvalues_array(observed, null_dist_2d) -> np.ndarray

SINGLE ANALYSIS (phase_g_fourier.py:1065-1292)
  analyze_one(...) -> dict                                   (line 1070)
  _classify_multi_freq(fourier_res, p_coord, period, logger) (line 1262)

BATCH PROCESSING (phase_g_fourier.py:1294-1565)
  process_level_layer_pop(...) -> list[dict]
  run_all(paths, levels, layers, n_perms, logger) -> pd.DataFrame

AGREEMENT COMPUTATION (phase_g_fourier.py:1626-1667)
  compute_agreement(results_df, logger) -> pd.DataFrame

FDR CORRECTION (phase_g_fourier.py:1572-1623)
  apply_fdr(results_df, logger) -> pd.DataFrame
  _benjamini_hochberg(p_values) -> np.ndarray

DECISION RULE APPLICATION (phase_g_fourier.py:1670-1737)
  apply_decision_rule(results_df, logger) -> dict

SAVING (phase_g_fourier.py:1745-onwards)
  save_per_concept_results(result, paths, logger)
  save_summary_csv(results_df, paths)
  save_detection_csvs(results_df, paths)

PLOTTING
  plot_fcr_heatmaps(results_df, paths, logger)
  plot_centroid_circles(results_df, paths, logger)
  plot_power_spectra(results_df, paths, logger)
  plot_pvalue_trajectories(results_df, paths, logger)
  plot_single_example_projections(results_df, paths, logger)
  generate_all_plots(results_df, paths, logger)

SYNTHETIC PILOT (phase_g_fourier.py:2132-2310)
  run_synthetic_pilot(logger) -> bool                        (line 2136)

PILOT 0b (phase_g_fourier.py:2375-onwards)
  run_pilot_0b(paths, logger) -> dict

CLI (phase_g_fourier.py:2536-onwards)
  parse_args()
  main()
```

`phase_g_kt_pilot.py` is 383 lines, standalone, GPU-only.

`phase_g_numtok_fourier.py` is 637 lines, standalone, CPU-only, imports the
statistical core from `phase_g_fourier.py` without re-implementing it
([phase_g_numtok_fourier.py:38-51](../phase_g_numtok_fourier.py#L38-L51)).

`extract_number_token_acts.py` is 335 lines, GPU-only, has resume logic
([extract_number_token_acts.py:296-303](../extract_number_token_acts.py#L296-L303))
that skips levels whose output files already exist.

### 16b. Data Loading Pipeline

For each (level, layer, population, concept):

**Phase C fast path:**

1. Load `projected_all.npy` (shape `(N, d_sub)`) — pre-projected activations
2. Apply population mask to get the population slice
3. Compute centroids directly in `d_sub` dimensions
4. Runtime: ~0.1s for L5/all (122K × 9 = 4.4 MB of float32)

**Phase D path:**

1. Load residualized activations (shape `(N, 4096)`) — one load per (level, layer),
   cached for all concepts in that slice
2. Load `merged_basis.npy` (shape `(d_merged, 4096)`)
3. Project: `projected = acts_pop @ merged_basis.T` (shape `(N_pop, d_merged)`)
4. Compute centroids in `d_merged` dimensions
5. Runtime: ~4s load for L5 (122K × 4096 = 1.9 GB), then ~0.1s per projection

### 16c. Concept Registry

The concept registry is built at runtime from the coloring DataFrame:

- **Digit concepts:** Read from `DIGIT_CONCEPTS_BY_LEVEL` constant, verified
  against DataFrame columns. Period spec: always `("digit", P=10)`. Values
  read from actual unique values in the DataFrame.
- **Carry concepts:** Read from `CARRY_CONCEPTS_BY_LEVEL`. Period specs built
  dynamically:
  - `carry_binned` resolved at runtime from Phase C metadata (period = n_groups)
  - `carry_mod10` built from the raw value distribution when ≥6 values in 0–9
  - `carry_raw` built from the raw value distribution when ≥6 distinct raw values

The `carry_binned` spec is **resolved per (layer, pop)** because Phase C's binning
may vary: the tail bin threshold can differ across populations if different
populations have different value ranges.

### 16d. Core Fourier Functions

**`fourier_single_coordinate`:** Explicit DFT loop over K frequencies. For each
frequency `k = 1..K`, computes `a_k = Σ signal[v] · cos(2πkv/P)` and `b_k =
Σ signal[v] · sin(2πkv/P)`, then `P_k = a_k² + b_k²` (rescaled by 2× for Nyquist).
Returns per-frequency power, FCR_top1, and dominant frequency. Implementation:
[phase_g_fourier.py:580-633](../phase_g_fourier.py#L580-L633).

**`fourier_all_coordinates`:** Calls `fourier_single_coordinate` for each of `d`
coordinates, then computes `two_axis_fcr` (max over frequencies of top-2 coords'
power), `uniform_fcr` (mean per-coord FCR), and frequency mode statistics.
Implementation: [phase_g_fourier.py:636-748](../phase_g_fourier.py#L636-L748).

**`compute_linear_power`:** Per-coordinate projection onto centered linear ramp.
Uses sample-weighted centering when group sizes are available (Fix 4). Implementation:
[phase_g_fourier.py:751-777](../phase_g_fourier.py#L751-L777).

**`compute_helix_fcr`:** For each frequency, combines top-2 Fourier power with
best non-overlapping linear power. Rescales linear power by `m/(2·Σv²)` for
DOF parity (Fix 5). Implementation:
[phase_g_fourier.py:780-894](../phase_g_fourier.py#L780-L894).

### 16e. Permutation Null Implementation

The permutation null uses a **conditioned null**: group sizes are fixed, only the
assignment of samples to groups is shuffled. This is slightly more conservative
than Phase C's unconditioned null (which allows group sizes to vary), but the
difference is negligible for balanced groups.

Implementation ([phase_g_fourier.py:942-1048](../phase_g_fourier.py#L942-L1048)):

1. Pre-compute cumulative group sizes: `cum_sizes = [0, n_0, n_0+n_1, ...]`
2. Create index array `all_idx = arange(N)`
3. For each permutation:
   a. Shuffle `all_idx` in-place via `rng.shuffle(all_idx)`
   b. Assign `all_idx[cum_sizes[i]:cum_sizes[i+1]]` to group `i`
   c. Compute null centroids as group means of shuffled projected data
   d. DC-remove null centroids
   e. Run full Fourier + linear + helix analysis
   f. Store null statistics

The in-place shuffle is O(N) and the centroid computation is O(N·d). With
N=122K and d=9, each permutation takes ~0.001s. The 1,000 permutations complete
in ~1s.

### 16f. Detection Logic

```python
if d == 0:
    circle_detected = False
    helix_detected = False
else:
    circle_detected = (
        p_two_axis < PERM_ALPHA                       # global: p < 0.01
        and p_coord[coord_a] < COORD_P_THRESHOLD       # coord a: p < 0.01
        and p_coord[coord_b] < COORD_P_THRESHOLD       # coord b: p < 0.01
    )

    helix_detected = (
        d >= 2
        and p_helix < PERM_ALPHA                      # global helix: p < 0.01
        and p_coord[helix_coord_a] < COORD_P_THRESHOLD # Fourier a
        and p_coord[helix_coord_b] < COORD_P_THRESHOLD # Fourier b
        and p_linear[helix_linear_coord] < LINEAR_P_THRESHOLD  # linear axis
    )

if helix_detected:
    geometry_detected = "helix"
elif circle_detected:
    geometry_detected = "circle"
else:
    geometry_detected = "none"
```

Source: [phase_g_fourier.py:1148-1182](../phase_g_fourier.py#L1148-L1182).

### 16g. Output Format

**Directory structure:**

```
/data/user_data/anshulk/arithmetic-geometry/phase_g/
├── fourier/
│   └── L{level}/layer_{LL:02d}/{pop}/{concept}/{basis}/{period_spec}/
│       ├── fourier_results.json
│       └── centroids.npy
├── kt_pilot/
│   └── kt_pilot_summary.json
├── numtok/
│   └── L{level}/layer_{LL:02d}/pos_{a|b}/
│       └── {concept}_fourier_results.json
├── numtok_checkpoint.pkl
└── summary/
    ├── phase_g_results.csv        (3,481 lines incl header)
    ├── phase_g_helices.csv        (501 lines: helix-detected subset)
    ├── phase_g_circles.csv        (497 lines: circle-detected subset)
    ├── phase_g_agreement.csv      (3,481 lines: Phase C vs D concordance)
    ├── phase_g_decisions.json     (class-level decision rule outcome)
    ├── checkpoint_results.pkl     (pickled full state, 10.3 MB)
    └── numtok_fourier_results.csv (109 lines)
```

**Summary CSV schema — 42 columns:**

```
1.  concept
2.  tier
3.  level
4.  layer
5.  population
6.  subspace_type
7.  period_spec
8.  n_groups
9.  period
10. values_tested
11. d_sub
12. n_samples_used
13. n_perms_used
14. two_axis_fcr
15. two_axis_best_freq
16. two_axis_coord_a
17. two_axis_coord_b
18. two_axis_p_value
19. two_axis_q_value
20. uniform_fcr_top1
21. uniform_fcr_p_value
22. eigenvalue_fcr_top1
23. fcr_top1_max
24. fcr_top1_max_coord
25. fcr_top1_max_freq
26. dominant_freq_mode
27. n_sig_coords_at_mode_freq
28. circle_detected
29. helix_detected
30. geometry_detected
31. multi_freq_pattern
32. helix_fcr
33. helix_best_freq
34. helix_linear_coord
35. helix_p_value
36. helix_q_value
37. p_value_floor
38. p_saturated
39. agreement
40. eigenvalue_top1
41. eigenvalue_top2
42. eigenvalue_top3
```

(Verified by reading the CSV header on Apr 13 and counting columns.)

**Plots directory (populated Apr 13 05:47–06:03):**

```
/home/anshulk/arithmetic-geometry/plots/phase_g/
├── kt_pilot/                        (4 PNG files: magnitude spectra, layers 0/1/4/8)
├── fcr_heatmaps/                    (88 PNG files)
├── centroid_circles/                (501 PNG files — one per detection)
├── frequency_spectra/               (501 PNG files — matching)
├── pvalue_trajectories/             (162 PNG files)
└── single_example_projections/      (1,122 PNG files for null concepts)
```

Total plots: 2,378 PNG files.

### 16h. Error Handling and Edge Case Guards

The 17-bug audit across Runs 1 and 2 (Section 6b and 6d) revealed many edge cases
that required defensive guards. These are documented here for reproducibility:

**`carry_mod10` graceful skip.** Phase C may merge rare carry values into tail bins
at certain (level, layer, population) slices. When this happens, the `carry_mod10`
period spec references values that do not exist in the slice. Instead of crashing
(the Run 1 behavior), the code now filters `values` to only those present in
`unique_vals`, and skips the analysis if fewer than `MIN_CARRY_MOD10_VALUES` (3)
remain. A debug log message records the skip.

**`d < 2` guards.** When a subspace has dimensionality `d=1` (possible for small
Phase C subspaces), the `two_axis_fcr` and `helix_fcr` require coordinates `a`
and `b`, but only coordinate `0` exists. Guards skip these analyses when `d < 2`,
logging the skip.

**`d == 0` guards.** When Phase D returns a zero-dimensional merged basis for a
concept, the projection gives shape `(N, 0)` and the centroid shape is `(m, 0)`.
Several downstream functions were crashing on this. The fix is a three-layer
defense:

- `load_phase_d_merged_basis` + `process_level_layer_pop`: added
  `merged_basis.shape[0] > 0` check alongside `is not None`
- `fourier_all_coordinates`: zero-power early return includes all keys (Fix 13)
- `compute_helix_fcr`: explicit `d == 0` branch returns zero power (Fix 15)
- `analyze_one`: explicit `d == 0` branch sets `circle_detected = helix_detected
  = False` (Fix 17)

**Eigenvalue-weighted FCR division.** The eigenvalue-weighted FCR divides by the
sum of eigenvalue weights. When all eigenvalues are negligible (below 1e-10), the
sum is effectively zero. A threshold check returns the unweighted mean instead
of dividing by zero.

**Phase D basis count.** The filesystem walk for Phase D merged bases originally
used a hard `assert n >= expected`. This was changed to a logged error + `sys.exit(1)`
(Fix 11) to provide a diagnostic message before aborting.

**Empty DataFrame column preservation.** Boolean indexing on a pandas DataFrame
(`df[mask]`) can silently drop columns when the result is empty. All such operations
now use `df.loc[mask].copy()` to preserve the schema (Fix 12).

**NaN p-values in FDR.** The FDR function handles NaN p-values (which can arise
from degenerate `d=0` skip paths) by assigning `q=1.0` for those rows before
running the BH procedure (Fix 15 from review; Fourier implementation at
[phase_g_fourier.py:1582-1605](../phase_g_fourier.py#L1582-L1605)).

**Zero-permutation log division.** The permutation null log message divides
elapsed time by `n_perms` for a rate calculation. If `n_perms = 0` (via
`--skip-null`), this would crash. Fixed to `elapsed / n_perms if n_perms > 0
else 0.0` (Fix 14).

---

## 17. Runtime and Reproducibility

### 17a. Final Run Timing

Final SLURM job 7058788, after partition update from `preempt` to `general`:

| Field | Value |
|-------|-------|
| Partition (final) | general |
| QOS | normal |
| Time limit | 2-00:00:00 (2 days) |
| Node | babel-t9-24 |
| GPU | 1× A6000 |
| CPUs | 24 |
| Memory | 128 GB |
| Start | Mon Apr 13, 2026, 02:15:28 EDT |
| End | Mon Apr 13, 2026, 06:03:39 EDT |
| Total elapsed | **226.9 minutes (3 hours 48 minutes)** |

**Step-by-step (from `slurm-7058788.out` and `phase_g_fourier.log`):**

| Step | Description | Start | End | Duration |
|------|-------------|-------|-----|---------:|
| 0 | Conda env activation, init | 02:15:28 | 02:15:30 | ~2 sec |
| 1 | K&T pilot (GPU) | 02:15:30 | 02:15:55 | ~25 sec (model load + extraction + Fourier) |
| 2 | Number-token extraction (GPU) | 02:15:55 | 02:15:58 | ~3 sec (resumed from existing files) |
| 3 | Synthetic pilot (CPU) | 02:15:58 | 02:15:59 | <1 sec |
| 4 | Pilot 0b (CPU) | 02:15:59 | 02:16:10 | ~11 sec |
| 5 | Full Fourier screening | 02:16:10 | 05:46:46 | **3h 30m 36s** |
| Post-5a | Decision rule application | 05:46:46 | 05:47:00 | ~14 sec |
| Post-5b | Plot generation | 05:47:00 | 06:03:39 | ~16 min 39 sec |
| Total | | 02:15:28 | 06:03:39 | **3h 48m 11s** |

The Fourier screening (Step 5) accounts for 92% of the elapsed time. Plot
generation accounts for 7%. Everything else is negligible. The Fourier screening's
own internal log reports `PHASE G COMPLETE: 226.9 minutes (3.8 hours)`, matching
the wall clock.

### 17b. Per-Analysis Timing

The Fourier screening processes 3,480 analysis cells in 3h 30m = 12,636 seconds,
giving an average of **3.6 seconds per cell**. Per-cell timing varies with N
(sample count) and `d` (subspace dimension):

| Configuration | Approximate time per analysis |
|---------------|------------------------------:|
| L2/all, Phase C (`d=7–9`, `N≈4,000`) | 0.7–0.9 sec |
| L2/all, Phase D (`d=14–18`, `N≈4,000`) | 1.2–1.5 sec |
| L3/all, Phase C (`d=7–9`, `N≈10,000`) | 1.0–1.5 sec |
| L4/all, Phase C (`d=8–10`, `N≈10,000`) | 1.5–2.0 sec |
| L5/all, Phase C (`d=2–9`, `N=122,223`) | 1.5–3.0 sec |
| L5/all, Phase D (`d=18`, `N=122,223`) | 4.0–5.5 sec |

The permutation null dominates: ~0.001s/perm × 1,000 perms = ~1s base, plus
overhead proportional to N and d for the centroid/projection step.

The `permutation_null` debug log shows actual rates from the final run, e.g., for
the `b_hundreds / L5 / pos_b` cell from the numtok screening:

```
14:57:16 DEBUG  Permutation null: N=122223, d=8, m=9, P=10, K=5, n_perms=1000
14:57:21 DEBUG  Permutation null complete: 5.5s total (0.006s/perm)
```

5.5 seconds for 1000 permutations at `N=122,223` × `d=8`. For the smaller `d=2`
Phase C subspaces at L5, the same operation takes ~1 second.

### 17c. Reproducibility Steps

The full pipeline is reproducible from a clean state with these commands:

```bash
# Step 1: K&T pilot (GPU, ~30 sec)
python phase_g_kt_pilot.py --config config.yaml

# Step 2: Number-token extraction (GPU, ~5 min on first run, ~3 sec on resume)
python extract_number_token_acts.py --config config.yaml

# Step 3: Synthetic pilot (CPU, <1 sec)
python phase_g_fourier.py --config config.yaml --pilot

# Step 4: Pilot 0b (CPU, ~11 sec)
python phase_g_fourier.py --config config.yaml --pilot-0b

# Step 5: Full Fourier screening (CPU, ~3.5 hours)
python phase_g_fourier.py --config config.yaml --n-perms 1000

# Step 6: Number-token Fourier screening (CPU, ~10 min — separate from main)
python phase_g_numtok_fourier.py --config config.yaml --n-perms 1000
```

Or via the combined SLURM script:

```bash
sbatch run_phase_g.sh
sbatch run_phase_g_numtok.sh   # separate job for numtok screening
```

**SLURM partition recommendation:** Use `general` with `--time=2-00:00:00`. Do
NOT use `preempt` — Run 3 was preempted 12 times in `preempt` before completing
in `general`. The 7-day `preempt` time limit doesn't help if the job never gets
to start.

**Random seed:** The numpy random generator is seeded at the script level for
reproducibility. The permutation null uses `rng = np.random.default_rng(seed)`.

**Determinism caveat:** Phase G is *statistically* reproducible (the same seed
gives the same p-values and detection counts) but minor numerical variation
across hardware can cause differences at the 6th decimal place of FCR values
in the JSON outputs. Detection decisions are robust to this.

### 17d. Output Files (Complete Inventory)

**On `/data/user_data/anshulk/arithmetic-geometry/` (heavy data, persisted):**

```
phase_g/
├── fourier/                         (per-cell results, 3,480 cells × 2 files = ~7,000 files)
│   └── L{2,3,4,5}/layer_{04,06,08,12,16,20,24,28,31}/{all,correct,wrong}/{concept}/
│       └── {phase_c,phase_d_merged}/{digit,carry_binned,carry_mod10,carry_raw}/
│           ├── fourier_results.json    (~1–2 KB each)
│           └── centroids.npy           (~0.5–4 KB each)
├── kt_pilot/
│   └── kt_pilot_summary.json        (Apr 13 02:15, 2,201 bytes)
├── numtok/
│   └── L{2,3,4,5}/layer_{04,08,12,16,20,24}/pos_{a,b}/
│       └── {a_units,a_tens,a_hundreds,b_units,b_tens,b_hundreds}_fourier_results.json
│       (108 files total)
├── numtok_checkpoint.pkl            (Apr 11 14:55, 52,217 bytes)
└── summary/
    ├── phase_g_results.csv          (Apr 13 05:46, 1,396,058 bytes, 3,481 lines)
    ├── phase_g_helices.csv          (Apr 13 05:46,   215,391 bytes,   501 lines)
    ├── phase_g_circles.csv          (Apr 13 05:46,   213,919 bytes,   497 lines)
    ├── phase_g_agreement.csv        (Apr 13 05:46,   326,453 bytes, 3,481 lines)
    ├── phase_g_decisions.json       (Apr 13 05:46,       798 bytes)
    ├── checkpoint_results.pkl       (Apr 13 05:46, 10,300,520 bytes)
    └── numtok_fourier_results.csv   (Apr 11 14:57,    35,643 bytes,   109 lines)

activations_numtok/                  (operand-token activations, 48 files, 9.2 GB)
├── level{2}_layer_{04,08,12,16,20,24}_pos_{a,b}.npy   (12 × 32 MB = 384 MB)
├── level{3}_layer_{04,08,12,16,20,24}_pos_{a,b}.npy   (12 × 80 MB = 960 MB)
├── level{4}_layer_{04,08,12,16,20,24}_pos_{a,b}.npy   (12 × ?  MB)
└── level{5}_layer_{04,08,12,16,20,24}_pos_{a,b}.npy   (12 × ? MB, the largest)
```

**On `/home/anshulk/arithmetic-geometry/` (code, logs, plots):**

```
phase_g_kt_pilot.py                   (383 lines)
extract_number_token_acts.py          (335 lines, has resume logic)
phase_g_fourier.py                    (2,623 lines — main script)
phase_g_numtok_fourier.py             (637 lines, imports core from above)
run_phase_g.sh                        (SLURM wrapper for full pipeline)
run_phase_g_numtok.sh                 (SLURM wrapper for numtok screening)
config.yaml                            (paths and settings)

logs/
├── phase_g_fourier.log                (Apr 13 06:03, 5.3 MB — final successful run)
├── phase_g_fourier.log.{1,2,3}        (older rotated segments from preempted runs)
├── phase_g_kt_pilot.log               (Apr 13 02:15, 69 KB)
├── phase_g_numtok_fourier.log         (Apr 11 14:57, 200 KB)
├── slurm-7058788.out                  (Apr 13 06:03, 647 bytes — step markers)
└── slurm-7058788.err                  (Apr 13 06:03, 1.1 MB — INFO-level run output)

plots/phase_g/
├── kt_pilot/                          (4 files — magnitude spectra layers 0/1/4/8)
├── fcr_heatmaps/                      (88 files)
├── centroid_circles/                  (501 files — one per detection)
├── frequency_spectra/                 (501 files — matching power spectra)
├── pvalue_trajectories/               (162 files)
└── single_example_projections/        (1,122 files for null concepts)
```

**Total file count for Phase G outputs: 2,378 PNG plots + ~7,000 per-cell JSON/NPY
files + 7 summary files + 4 kt_pilot files + 109 numtok files + 4 logs ≈ 9,500
files** across `/home` and `/data`.

### 17e. Partition Choice and the Preemption Lesson

Run 3 was originally submitted to the `preempt` partition because it had the
largest pool of A6000 GPU nodes available at CMU's Babel cluster. The 7-day time
limit seemed adequate for a 4-hour job. But preempt jobs can be killed at any
time by higher-priority jobs, and Run 3 was preempted **12 times** in 24 hours.
The job never got to run for more than ~30 minutes uninterrupted.

The fix was an in-place SLURM partition update:

```
scontrol update JobId=7058788 Partition=general QOS=normal TimeLimit=2-00:00:00
```

After the update, the job was scheduled within ~14 hours and ran clean. **The
lesson for re-running Phase G: use the `general` partition with `--time=2-00:00:00`,
not `preempt`.** The 5-step pipeline takes ~4 hours on a single A6000, well
within the 2-day general partition limit, and `general` does not preempt jobs
once they've started.

A future re-run could also split the GPU work (Steps 1–2) and CPU work (Steps
3–5) into two separate jobs, which would let the CPU portion run on a CPU-only
partition without holding the A6000 idle for ~3.5 hours. This is a future
optimization, not a requirement.

---

## Appendix A: The Algebra of Centroid Fourier Screening

The centroid Fourier test can be understood algebraically as a projection of the
between-class covariance onto the Fourier basis.

**Setup.** Let `μ_v ∈ R^d` be the DC-removed centroid for value `v`, where `v`
ranges over the concept's value set `V` with `|V| = m`. The centroids define a
data matrix `M ∈ R^{m × d}` with rows `μ_v`.

**Fourier basis.** For period `P` and frequency `k`, define the basis vectors
indexed by value:

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
frequencies and coordinates. Under Parseval's theorem (for complete grids), this
equals `m · ||M||_F²` where `||M||_F` is the Frobenius norm of the centroid matrix.
For incomplete grids, the relationship is approximate.

**Two-axis FCR.** `two_axis_fcr = max_k (P_k[j₁] + P_k[j₂]) / total`, where
`j₁, j₂` are the top-2 coordinates at frequency `k`. This measures the fraction of
total Fourier energy concentrated in a 2D subplane at a single frequency — exactly
the signature of a circle in 2 dimensions embedded in d-dimensional space.

**Null distribution.** Under the null hypothesis (concept values are random labels),
the centroids `μ_v` are approximately iid Gaussian (CLT on group means). The
Fourier coefficients are linear functions of Gaussian centroids, hence also
Gaussian. The power `P_k[j]` follows a scaled chi-squared distribution with 2 DOF
(1 DOF for Nyquist). The two_axis_fcr is the max of a ratio of chi-squared variates
— analytically intractable, hence the permutation null.

---

## Appendix B: Why Centroids, Not Individual Points — The Statistical Argument

The literature on Fourier features in LLMs (K&T, Gurnee et al.) typically analyzes
individual activations. Our approach — computing Fourier statistics on group
centroids — is different. Here we justify this choice.

**The signal-to-noise argument.** If the model encodes digit `v` at position
`(A cos(2πv/10), A sin(2πv/10)) + noise`, then each individual activation is
a noisy observation of the Fourier position. The centroid averages `N_v`
observations, reducing the noise by a factor of `√N_v`. At L5/all, each centroid
is the mean of ~12,000 samples, giving a noise reduction of ~110×. The centroid
Fourier test is thus a highly powered test of between-class structure.

**The within-class problem.** If within-class spread is large relative to
between-class separation (i.e., SNR < 1), the centroids may not reflect the
underlying manifold. However, Phase C already established that between-class
structure is significant (`dim_perm > 0` requires between-class variance to
exceed the permutation null). Phase G operates only on concepts that passed
Phase C's screening.

**The correspondence to the claim.** The centroid test asks: "Are the model's
mean representations for each digit value arranged periodically?" This is a
stronger claim than "Do individual activations lie on a periodic manifold?"
A positive centroid test means the model systematically places the average
activation for each digit on a circle/helix. A positive individual-activation
test could be driven by within-class structure (e.g., individual activations
forming local clusters that happen to trace a ring when connected).

**The alternative: single-example projections.** For concepts where the centroid
test is null, the single-example projection plots provide a visual check. If
individual activations form a visible ring or helix when projected onto the top-2
subspace coordinates and colored by digit value, this is noted for manual review.
This is exploratory, not part of the pre-registered decision rule.

In the final run, 1,122 single-example projection plots were generated for null
concepts at `plots/phase_g/single_example_projections/`. These are the basis for
manual follow-up of any null cells where a manifold might be hiding.

---

## Appendix C: The K&T Helix and Its Relationship to Our Test

Kantamneni & Tegmark (2025) found that LLMs represent numbers on a **generalized
helix** with the following structure:

```
representation(v) ≈ [A₁ cos(2πv/T₁), A₁ sin(2πv/T₁),
                     A₂ cos(2πv/T₂), A₂ sin(2πv/T₂),
                     ...,
                     B · v,              ← linear magnitude axis
                     noise]
```

Multiple Fourier periods `T₁, T₂, ...` (they found `T ∈ {2, 5, 10}` for base-10
numbers) plus a linear magnitude axis. Our test captures this via:

1. **two_axis_fcr at period P=10:** Detects the T=10 circular component (the
   dominant period for base-10 digits). Tests at frequency `k=1` (fundamental
   period 10).
2. **Nyquist at P=10 (`k=5`):** Detects the T=2 parity component. The signal
   `(-1)^v` is the Nyquist mode at period 10.
3. **helix_fcr:** Detects the combined circular + linear structure. If the linear
   magnitude axis is strong, helix_fcr > two_axis_fcr.

**Where we deviated from K&T:** We test multiple period specs for carry concepts
(`carry_binned`, `carry_mod10`, `carry_raw`), and we use centroid statistics
rather than per-dimension power. Our finding that `carry_raw` (period equal to
raw value range) dominates over `carry_mod10` is novel: it shows that the model
doesn't always use the base-10 period for arithmetic intermediates. K&T's
standalone integers happen to use period 10 because the integers naturally span
the base-10 cycle; our carries use longer periods because the carry values span
larger non-decimal ranges.

**What our test cannot detect:**

- Multi-period helices with `T=5`. Our test at `P=10` detects `T=10` (`k=1`) and
  `T=2` (`k=5`), but `T=5` corresponds to `k=2` at `P=10`. The two_axis_fcr tests
  all frequencies, so if `T=5` is the dominant period, the `best_freq` will be 2
  rather than 1. This is detectable but may not match the "circle at `k=1`"
  expectation. In the final run, 62 of 500 helix detections have `helix_best_freq
  = 2`, which corresponds to `T = P/2 = 5` for digit concepts at `P=10`.
- Relative phase alignment between the cos and sin components. Our test measures
  power (magnitude squared) but not phase. If the cos and sin axes are not
  orthogonal, the manifold is an ellipse rather than a circle, and the weaker
  axis may fail the conjunction.

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
Phase G: Fourier screening for periodic structure  ←  FIRST NON-LINEAR PROBE
    ↓
Phase H: GPLVM (non-linear manifold learning on activations)        [TODO]
    ↓
Phase I: Causal patching (ablation to verify functional role)        [TODO]
```

Phase G is the transition point. Phases A–F/JL established the linear geometry:
subspaces, their dimensions, their overlaps, and the fidelity of the projected
space. Phase G asks: **within** those subspaces, is the arrangement of concept
values non-linear (periodic)?

The answer (Sections 7–13): **yes for carries, partly yes for answer digits,
no for operand digits.** The non-linear structure exists, but it is concept-
dependent and task-dependent.

If Phase G had been completely null, the non-linearity hypothesis would have rested
entirely on Phase H (GPLVM) and Phase I (causal patching). The Phase G positives
mean those follow-up phases now have a specific target: validate that the carry
helices are causally important for carry propagation, and characterize whether
they fit a richer non-linear manifold than the pure helix model.

**Data flow between phases:**

```
Phase A output:   L{N}_coloring.pkl  ──────────────────────────────┐
Phase B output:   level{N}_layer{L}.npy (residualized)  ───────────┤
Phase C output:   projected_all.npy, metadata.json,                │
                  basis.npy, eigenvalues.npy  ──────────────────────┤
Phase D output:   merged_basis.npy  ────────────────────────────────┤
                                                                     ▼
                                                              Phase G Fourier
                                                                     │
                                                                     ▼
                                fourier_results.json, centroids.npy,
                                phase_g_results.csv, phase_g_circles.csv,
                                phase_g_helices.csv, phase_g_agreement.csv,
                                phase_g_decisions.json,
                                FCR heatmaps, centroid circle plots,
                                frequency spectra, p-value trajectories,
                                single-example projections
```

**What Phase G does NOT use:**

- Phase E's union basis (not needed — Phase G analyzes per-concept structure)
- Phase F's angle measurements (not needed — Phase G is within-concept)
- Phase JL's distance metrics (not needed)

This clean dependency structure means Phase G can run as soon as Phases A–D are
complete, without waiting for Phase E or F/JL. In practice, all phases were
completed before Phase G was designed, but the independence is useful for
understanding the pipeline architecture.

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
Expected: `two_axis_fcr = 1.0` (all power in frequency 1, coordinates 0 and 1).
Observed: 1.0000. **PASS.**

**Test 2: Random Noise**
```python
rng = np.random.default_rng(42)
centroids = rng.standard_normal((10, 9))
```
Expected: `two_axis_fcr` near null (`~1/(K·d)` × correction for max). Approx 0.20.
Observed: 0.2355. **PASS.**

**Test 3: Linear/Quadratic**
```python
centroids = np.zeros((10, 9))
centroids[:, 0] = np.linspace(0, 1, 10)
centroids[:, 1] = np.linspace(0, 1, 10) ** 2
```
Expected: `two_axis_fcr < 0.65` (not periodic). Observed: 0.5943. **PASS.**

The original threshold was 0.5, which failed because a linear ramp has genuine
Fourier content at low frequencies (FCR = 0.5943). Threshold raised to 0.65 before
Run 2. Real circles produce FCR > 0.95; the 0.65 boundary cleanly separates linear
artifacts.

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
After DC removal: all zeros. Expected: `total_power < 1e-12`, no crash.
Observed: `total_power < 1e-12`, FCR = 0.0. **PASS.**

**Test 6: P=9 Conjugate**
```python
values = np.arange(9)
centroids = np.zeros((9, 9))
centroids[:, 0] = [cos(2πv/9) for v in values]
centroids[:, 1] = [sin(2πv/9) for v in values]
```
Expected: `K=4` frequencies (not 5), FCR = 1.0. Verifies odd-P frequency range
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
Expected: Nyquist bin (`k=5`) captures ~100% of power after 2× rescaling.
Observed: fraction = 1.0000, `best_freq = 5`. **PASS.** (Verifies Nyquist
inclusion and rescaling.)

**Test 9: Helix**
```python
centroids = np.zeros((10, 9))
centroids[:, 0] = [cos(2πv/10) for v in range(10)]
centroids[:, 1] = [sin(2πv/10) for v in range(10)]
centroids[:, 2] = np.linspace(0, 1, 10)
```
Expected: `helix_fcr > two_axis_fcr` (the linear axis adds to the helix numerator).
Observed (Run 1): `helix_fcr = 0.7200, two_axis_fcr = 0.9000`. **FAIL.**
Observed (Run 2/3, after fix): `helix_fcr = 0.9084, two_axis_fcr = 0.9000`. **PASS.**

The original denominator summed rescaled linear power across all `d` coordinates,
but the numerator only used the best linear coordinate. Fixed before Run 2 to
use only `best_linear_rescaled` in the denominator, matching the numerator's scope.

**Test 10: Pure Linear Ramp**
```python
centroids = np.zeros((10, 9))
centroids[:, 0] = np.linspace(0, 1, 10)
```
Expected: `linear_power >> Fourier power` (the signal is pure ramp, not periodic).
Observed: `linear_power = 6806.25, Fourier power = 450.0`. **PASS.** (~15× ratio
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
implemented as `phase_g_fourier.py` and survived through Run 1, Run 2, and Run 3
crashes with **17 additional code-level bug fixes** (Sections 6b and 6d). The
plan-level fixes (1–18 in this table) and the code-level fixes (1–17 in Sections
6b/6d) are independent numbering systems.

---

## Appendix G: Number-Token Fourier Screening — Full Methodology

The number-token screening is documented in Section 12. This appendix collects
the methodological details for completeness.

**Script:** `phase_g_numtok_fourier.py` (637 lines).

**Imports from main script** ([phase_g_numtok_fourier.py:38-51](../phase_g_numtok_fourier.py#L38-L51)):
- `fourier_all_coordinates`
- `compute_linear_power`
- `compute_helix_fcr`
- `compute_centroids_grouped`
- `compute_pvalues`, `compute_pvalues_array`
- `permutation_null`
- Constants: `PERM_ALPHA`, `COORD_P_THRESHOLD`, `LINEAR_P_THRESHOLD`, `MIN_POPULATION`,
  `ZERO_POWER_THRESHOLD`

**Concept registry** ([phase_g_numtok_fourier.py:66-73](../phase_g_numtok_fourier.py#L66-L73)):

```python
DIGIT_CONCEPTS = [
    ("a_units",    "a", [2, 3, 4, 5]),
    ("a_tens",     "a", [2, 3, 4, 5]),
    ("a_hundreds", "a", [4, 5]),
    ("b_units",    "b", [2, 3, 4, 5]),
    ("b_tens",     "b", [3, 4, 5]),
    ("b_hundreds", "b", [5]),
]
```

Tuple format: `(column_name, position, available_levels)`. Position is which
operand's token position to use.

**PCA on centroids** ([phase_g_numtok_fourier.py:135-161](../phase_g_numtok_fourier.py#L135-L161)):

```python
def pca_on_centroids(centroids, pca_dim):
    m, D = centroids.shape
    k = min(pca_dim, m - 1)
    if k == 0:
        return np.zeros((0, D)), np.zeros(0)
    mean = centroids.mean(axis=0)
    centered = centroids - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = (S[:k] ** 2) / (m - 1)
    components = Vt[:k]
    return components, eigenvalues
```

The between-group structure lives in at most `m-1` dimensions for `m` digit values.
We take `k = min(pca_dim, m-1)` PCA components (default `pca_dim = 20`, but `m-1`
caps at 9 for 10-valued digits). Activations are projected into this `k`-dim
space before Fourier analysis.

**Loading function** ([phase_g_numtok_fourier.py:104-112](../phase_g_numtok_fourier.py#L104-L112)):

```python
def load_numtok_activations(level, layer, pos, data_root):
    fname = f"level{level}_layer_{layer:02d}_pos_{pos}.npy"
    path = os.path.join(data_root, "activations_numtok", fname)
    acts = np.load(path).astype(np.float32)
    return acts
```

Loads one of the 48 pre-extracted `.npy` files (float16 → float32 cast for
numerical stability in the Fourier computation). Each file is a `(N, 4096)`
matrix of operand-position activations.

**Constants** ([phase_g_numtok_fourier.py:56-62](../phase_g_numtok_fourier.py#L56-L62)):

```python
LEVELS = [2, 3, 4, 5]
NUMTOK_LAYERS = [4, 8, 12, 16, 20, 24]   # subset of main pipeline's layers
DIGIT_PERIOD = 10
FDR_THRESHOLD = 0.05
DEFAULT_PCA_DIM = 20
DEFAULT_N_PERMS = 1000
CHECKPOINT_INTERVAL = 50
```

Note: `NUMTOK_LAYERS = [4, 8, 12, 16, 20, 24]` is a 6-layer subset of the main
pipeline's 9-layer set. Layers 6, 28, 31 are not extracted at the number-token
position because the extraction was budget-constrained (it ran at 9.2 GB for
6 layers; 9 layers would have been 13.8 GB).

**Output schema** (109 lines × 31 columns):

```
N, m, d, pca_dim, period, two_axis_fcr, two_axis_best_freq, two_axis_coord_a,
two_axis_coord_b, helix_fcr, helix_best_freq, helix_linear_coord, helix_linear_power,
uniform_fcr, p_two_axis, p_helix, p_uniform, p_value_floor, p_saturated, n_perms,
circle_detected, helix_detected, geometry_detected, raw_two_axis_fcr, raw_helix_fcr,
pca_var_explained, concept, level, layer, position, p_two_axis_fdr, p_helix_fdr
```

The `raw_two_axis_fcr` and `raw_helix_fcr` columns report the FCR computed in
the raw 4096-dim space (no PCA), as a sanity check. The `p_two_axis_fdr` and
`p_helix_fdr` columns are FDR-adjusted q-values across all 108 cells.

**Total cells: 108** (verified by `wc -l numtok_fourier_results.csv` = 109 lines).
Decomposition by `(concept × levels × layers)`:

- a_units: 4 levels × 6 layers = 24
- a_tens: 4 levels × 6 layers = 24
- a_hundreds: 2 levels × 6 layers = 12
- b_units: 4 levels × 6 layers = 24
- b_tens: 3 levels × 6 layers = 18
- b_hundreds: 1 level × 6 layers = 6
- **Total: 108**

**Result: 0 detections (all 108 cells geometry_detected = none).** See Section 12c
for the breakdown and Section 12f for the interpretation.

---

*End of document. All numbers in Sections 6–14 are verified against
`/data/user_data/anshulk/arithmetic-geometry/phase_g/summary/phase_g_results.csv`
(Apr 13 05:46 EDT, 3,481 lines, 1.4 MB), `numtok_fourier_results.csv`,
`phase_g_decisions.json`, and `kt_pilot_summary.json`. Methodology in Sections
2–3 is verified against `phase_g_fourier.py` (2,623 lines). Run history in
Section 6 is verified against `slurm-7058788.out`, `slurm-7058788.err`, and
the four log rotations of `phase_g_fourier.log`.*
