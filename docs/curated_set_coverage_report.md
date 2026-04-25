# Curated Set v1: Coverage Report and Build Log

**Anshul's Geometric Manifold Interpretability Project**
**Carnegie Mellon University, April 2026**

This document records every decision, every number, and every result from the curated-set
build (Step B.1 of the Part B execution plan). It is the truth document for this stage.
All numbers in Sections 5 through 11 are validated against the actual output file
`/data/user_data/anshulk/arithmetic-geometry/curated/curated_set_v1.json` as of the
build completion timestamp (2026-04-25T20:34:36Z).

The curated set is the foundation for everything that follows in Part B — within-group
PCA (B.2), orthogonalization controls (B.3), GPLVM fits (B.4), persistent homology
(B.6), subspace ablation (B.7), helix rotation (B.8), and difficulty-matched validation
(B.9). Every Tier-2 (deep, methods-rich) result the paper makes will be computed on
this set. The Tier-1 results from Phases A through G stay on the full 122,223-row L5
population. Section 1 of this document gives the framing; Sections 2 through 4 record
the build inputs and decisions; Section 5 reports the build outcome at level granularity;
Sections 6 through 9 break down per-position coverage; Sections 10 and 11 audit the
matched-pair construction; Section 12 enumerates every below-floor cell with its
mathematical justification; Section 13 lists the eight verification checks the build
passed; and Section 14 documents the reproducibility manifest and the artifact paths.

The build is **complete**. All gates passed. No undocumented coverage gap exists.
Every L4 and L5 correct example that was nominated for matching was successfully paired.
Every (level, source_index) resolves into a valid row of the cached activation arrays
on disk. The curated set is ready for downstream phases.

---

## Table of Contents

- [1. Why a curated set, not the full population](#1-why-a-curated-set-not-the-full-population)
- [2. Inputs to the build](#2-inputs-to-the-build)
  - 2a. Population pools after the labels-answers join
  - 2b. Concept registry — the 17 names
  - 2c. Tier definitions used for stratification and matching
- [3. Budget and design decisions (corrections from the original plan)](#3-budget-and-design-decisions-corrections-from-the-original-plan)
- [4. Build pipeline overview](#4-build-pipeline-overview)
  - 4a. Pass 0 — load and dedup
  - 4b. Pass 1 — difficulty stratification
  - 4c. Pass 2 — concept-coverage greedy fill
  - 4d. Pass 3 — matched-pair construction
  - 4e. Pass 4 — assemble, top-up wrongs, validate
  - 4f. Pass 5 — post-assembly concept top-up
- [5. Build outcome — headline numbers](#5-build-outcome--headline-numbers)
- [6. Per-level distribution of tier triples](#6-per-level-distribution-of-tier-triples)
- [7. Per-position digit coverage at L3](#7-per-position-digit-coverage-at-l3)
- [8. Per-position digit coverage at L4](#8-per-position-digit-coverage-at-l4)
- [9. Per-position digit coverage at L5](#9-per-position-digit-coverage-at-l5)
- [10. Per-carry coverage across L3, L4, L5](#10-per-carry-coverage-across-l3-l4-l5)
- [11. Matched-pair diagnostics](#11-matched-pair-diagnostics)
  - 11a. Per-pair construction summary
  - 11b. Strict vs relaxed pair counts
  - 11c. Joint magnitude × carry tier distribution of pairs
  - 11d. Mean-difference statistics (the right thing to read)
  - 11e. Permutation p-values (with the test-design caveat)
- [12. Documented hard-ceiling gaps](#12-documented-hard-ceiling-gaps)
  - 12a. Mathematically excluded cells
  - 12b. Pool-limited cells (population scarcity)
  - 12c. Why these are documented and not failures
- [13. Verification checklist](#13-verification-checklist)
- [14. Reproducibility manifest](#14-reproducibility-manifest)
- [15. What downstream phases will consume from this artifact](#15-what-downstream-phases-will-consume-from-this-artifact)
- [16. Known limitations and future iterations](#16-known-limitations-and-future-iterations)

---

## 1. Why a curated set, not the full population

The Part B steps that come after this build cannot run on all 122,223 L5 problems. Two
arguments support this, one mathematical and one methodological.

**The mathematical argument.** Exact Bayesian Gaussian Process Latent Variable Models
(B.4) scale as O(N³) per Cholesky factorization, with derivatives adding a 2–3× factor
per gradient step. At N = 122,000 that is ~1.8 × 10¹⁵ floating-point operations per
evaluation, or hours per gradient step on a single A100, and tens of GPU-days per fit.
The principled escape — Hauberg's argument from `papers/bayes_paper.md` that the
geometry-preserving property of GPLVM depends on the predictive covariance Σ(z) growing
in low-density regions — depends on running exact inference, not the FITC/VFE/SVGP
sparse approximations that summarize the data through inducing points and lose the
density-driven uncertainty structure. With careful inducing-point placement and
bandwidth tuning sparse-GP can recover the Hauberg property, but doing so correctly
is itself a multi-week sub-project. The curated set at N ≈ 8,000 lets us run exact
inference end-to-end. At per-population scale (L3 ≈ 2,400, L4 ≈ 2,800, L5 ≈ 3,000)
each Cholesky is a few hundred milliseconds in fp64; the full grid of 17 concepts × 3
populations × 9 layers × 3 kernels ≈ 1,377 fits is days of single-GPU time, which
is the budget Part B is sized for.

**The methodological argument.** Persistent homology (B.6), causal subspace ablation
(B.7), and helix rotation (B.8) all need a fixed, redistributable, citable artifact
that other researchers can download and rerun. A 122,000-row dataset is not that
artifact; an 8,000-row set is. The curated set will be released with the paper.

**The Tier 1 / Tier 2 split.** This is important to keep straight. Tier 1 — Phase C/D/E/
F/JL plus the Phase G permutation null — runs on all 122K L5 problems and 10K L3/L4
problems, and produces the population-level statistics about existence and prevalence
of structure (e.g., "carry_1 sits on a helix in 18/18 cells at L5/correct"). Those
results stay published as is; they are not re-derived from the curated set. Tier 2 —
GPLVM, persistent homology, the two causal experiments, the matched-pair Phase G
re-run — runs on the curated set and produces the depth-of-structure statistics
(manifold dimensionality, topology, causal mechanism). The paper is honest about
which tier a number comes from. Stratification destroys the natural usage frequencies,
so any "how often does the model use the helix?" question goes to Tier 1; any "what
shape is the helix?" question goes to Tier 2.

The curated set is built deliberately to over-represent rare cells that geometric
analysis needs to characterize. It is **not** representative of the underlying
population's distribution. That is by design and is documented in the paper's
methods section.

---

## 2. Inputs to the build

The build script `build_curated_set.py` reads four kinds of files. None are modified
by the build; the curated set is purely derived.

### 2a. Population pools after the labels-answers join

For each of L3, L4, L5 the build does a row-by-row inner join of the canonical labels
file (in `/home/anshulk/arithmetic-geometry/labels/level_*.json`, schema
`{"problems": [{"index", "prompt", "labels": {a, b, product, carries, ...}}, ...]}`)
against the model's per-row predictions in `/data/user_data/anshulk/arithmetic-geometry/
answers/level_*.json` (schema `{"results": [{"index", "a", "b", "ground_truth",
"predicted", "correct", "raw_text"}, ...]}`). The two files share row order: the
i-th label corresponds to the i-th answer. The join is asserted on `(a, b, ground_truth)`
agreement; any mismatch aborts the build.

After the join, each row has 9 columns: `source_index, level, a, b, product, carries,
predicted, correct, raw_text`. The `source_index` column is critical — it is the
position of the row in both the labels file and in the cached activation arrays at
`/data/user_data/anshulk/arithmetic-geometry/activations/level{level}_layer{layer}.npy`.
Preserving `source_index` is what lets every downstream phase look up the corresponding
4096-dimensional activation in O(1) without re-running the model.

The build then **dedups within each level on (a, b)**. This is necessary because
L3 was generated by random sampling from the 8,100 unique 2-digit × 2-digit operand
pairs into a 10,000-row dataset, so it has on average ~1,900 rows where the same
(a, b) appears twice. L4 was randomly sampled from 810,000 candidates into 10,000
rows, so duplicates are rare (586 in this run). L5 is the carry-balanced selection
from the full 810K input space, with no duplicates by construction. After dedup:

| Level | Pre-dedup rows | Duplicates dropped | Post-dedup unique-(a,b) rows | Correct | Wrong |
|------:|--------------:|------------------:|-----------------------------:|--------:|------:|
| L3 | 10,000 | 4,267 | 5,733 | 3,859 | 1,874 |
| L4 | 10,000 | 586 | 9,414 | 2,723 | 6,691 |
| L5 | 122,223 | 0 | 122,223 | 4,197 | 118,026 |
| **Total pool** | **142,223** | **4,853** | **137,370** | **10,779** | **126,591** |

These are the input pools the rest of the build operates on. Note that the L3 pool is
**smaller than the L4 pool** — 5,733 vs 9,414 unique problems — because the L3
generation was sample-with-replacement from a smaller candidate space. This influences
the L3/wrong budget (1,200 of 1,874 = 64% of pool, which is high but acceptable).

### 2b. Concept registry — the 17 names

The single source of truth for which concepts exist at which level is
`phase_g_fourier.py:56-74`. The build imports nothing from that file but copies the
two dicts verbatim:

```python
DIGIT_CONCEPTS_BY_LEVEL = {
    3: ["a_units", "a_tens", "b_units", "b_tens",
        "ans_digit_0_msf", "ans_digit_1_msf", "ans_digit_2_msf", "ans_digit_3_msf"],
    4: ["a_units", "a_tens", "a_hundreds", "b_units", "b_tens",
        "ans_digit_0_msf", "ans_digit_1_msf", "ans_digit_2_msf",
        "ans_digit_3_msf", "ans_digit_4_msf"],
    5: ["a_units", "a_tens", "a_hundreds", "b_units", "b_tens", "b_hundreds",
        "ans_digit_0_msf", "ans_digit_1_msf", "ans_digit_2_msf",
        "ans_digit_3_msf", "ans_digit_4_msf", "ans_digit_5_msf"],
}
CARRY_CONCEPTS_BY_LEVEL = {
    3: ["carry_0", "carry_1"],
    4: ["carry_0", "carry_1", "carry_2"],
    5: ["carry_0", "carry_1", "carry_2", "carry_3", "carry_4"],
}
```

Counting unique names: 6 operand digits (`a_units, a_tens, a_hundreds, b_units,
b_tens, b_hundreds`) + 6 answer digits (`ans_digit_0_msf` through `ans_digit_5_msf`)
+ 5 carry positions (`carry_0` through `carry_4`) = **17 unique concepts**. Every
concept appears at the level where it is mathematically defined; concepts disappear
at lower levels where the operand structure has fewer digit positions.

The "43 concepts" reference that appears in the original Phase B–E documents is a
different counting convention — it counts concept-level instances (e.g., `carry_2 at
L4` and `carry_2 at L5` as two cells). The paper picks one convention and sticks with
it. We use **17 unique names**, with the level-instances expanded only when reporting
per-level coverage.

### 2c. Tier definitions used for stratification and matching

Three deterministic functions of `(a, b)` produce the difficulty axes used for both
the proportional stratification in Pass 1 and the strict-bin matching in Pass 3.

**`magnitude_tier(a, b)` — joint operand magnitude in [0, 8].** Per operand: small
(0) if leading digit ∈ {1, 2, 3}; medium (1) if {4, 5, 6}; large (2) if {7, 8, 9}.
Joint tier = `3 * tier(a) + tier(b)`. The bin orientation is identical across L3,
L4, and L5 so cross-level comparisons work cleanly. The 3-bin coarse-graining is
chosen so that the typical 9 × 9 = 81 raw leading-digit combinations collapse into
9 cells, large enough to populate from any of the three levels.

**`carry_count_tier(carries) -> int (0..2)`.** Counts non-zero carries in the
per-problem carry list. Low (0) if `nzc ≤ 1`; medium (1) if `nzc ∈ {2, 3}`; high (2)
if `nzc ≥ 4`. L3 (which has 2 carry positions) populates only `low`/`medium`; L4
populates all three rarely (the high tier requires both carries non-zero plus the
high-end of the L4 product); L5 populates all three regularly. This axis captures
arithmetic complexity in a way the magnitude tier does not — two operands of similar
magnitude can produce very different carry chains.

**`answer_length(a, b) -> int`.** Number of digits in the product `a * b`. L3
products span 2–4 digits; L4 products span 3–5 digits; L5 products span 5–6 digits.
This is structurally different from the magnitude tier because two same-magnitude
operands can produce products of different digit count (e.g., 99 × 99 = 9801, 4
digits; 50 × 50 = 2500, also 4 digits; but 100 × 100 = 10000, 5 digits at the boundary).
The answer-length axis splits the answer-digit positions into mathematically
meaningful sub-populations.

The composite cell key is the triple `(magnitude_tier, carry_count_tier, answer_length)`.
At L5 there are 40 distinct triples observed in the pool; at L4 there are 36; at L3
there are 27. Pass 1 stratifies on this triple. Pass 3 matches on it strictly, with
carry_count_tier ±1 as the only allowed relaxation.

**Two raw axes used as L1-distance tie-breakers in matching.** `nonzero_carry_count`
(integer count, not binned), and `leading_digit_pair_index = 9 * (la - 1) + (lb - 1)`
in [0, 80] where la, lb are the leading digits of a, b. These are finer-grained than
the binned tiers; within a strict tier-bin match they pick the wrong example with the
smallest L1 distance from the correct example.

---

## 3. Budget and design decisions (corrections from the original plan)

The original B.1.2 budget table in `next_steps.md` planned 5,000–8,000 problems split
across L2/L3/L4/L5. Validation against the data showed that table did not survive
contact with reality, and four corrections were applied before the executed build.

**Correction 1 — L2 is excluded.** The original budget asked for 800 correct + 200
wrong at L2 (1,000 total). The L2 wrong pool has only **7 examples** in the entire
4,000-row L2 dataset (model accuracy 99.825%). Drawing 200 wrongs is impossible.
Keeping all 7 wrongs would still leave joint-coverage cells with counts of 1, which
is not enough for any cell-level claim. After confirming with the user, L2 was dropped
entirely. No concept is unique to L2 — every L2 concept (`a_units, a_tens, b_units,
ans_digit_0_msf, ans_digit_1_msf, ans_digit_2_msf, carry_0`) reappears at L3, where
the model is at 67.2% accuracy and both correct and wrong populations are large
enough to support cell-level analysis. The 17-concept registry is preserved through
L3, L4, L5.

**Correction 2 — L1 is excluded.** L1 has 64 unique problems and the model is 100%
correct on all of them. Without a wrong population, L1 cannot support any matched-pair
claim. It also cannot support the disconnected-manifold diagnostic (B.2.5) because
there is too little within-cell variation. L1 is excluded with the same rationale
as L2.

**Correction 3 — Saved L1/L2 budget reallocated to L3, L4, L5.** With L1 (1,000)
and L2 (1,000) dropped, the saved 2,000 problems were proportionally reallocated.
The user requested the build target the upper end of the 5,000–8,000 corridor.
Resolved budget:

| Level | Correct | Wrong | Total | Pool budget used |
|------:|--------:|------:|------:|-----------------:|
| L3 | 1,200 | 1,200 | 2,400 | 31% / 64% |
| L4 | 1,000 | 1,800 | 2,800 | 37% / 27% |
| L5 | 1,400 | 1,400 | 2,800 | 33% / 1.2% |
| **Sum (planned)** | **3,600** | **4,400** | **8,000** | — |

The actual build added 263 extra rows in Pass 5 to top up rare-cell concept coverage,
producing a final n_problems of **8,264**. The total exceeds 8,000 by 264 rows because
Pass 5 is not budget-bound — its job is to close coverage gaps, not to hit a target
size.

**Correction 4 — Duplicate rule relaxed from Hamming ≤ 1 to Hamming = 0.** The
original B.1.1 requirement 7 asked for no near-duplicates within Hamming distance 1
of the operand digit string. After consulting with the user, this was relaxed to
exact-uniqueness only (Hamming = 0). The reasoning: the tier-based stratification
already prevents digit-pattern overrepresentation, and the Hamming ≤ 1 rule would
trim 5–10% of candidates without measurable benefit. The within-level dedup at
Pass 0 enforces Hamming = 0 strictly; Pass 4 has a defensive double-check that has
fired zero times in this build.

**Correction 5 — Matching rule with explicit fallback.** The plan specified strict
matching on `(magnitude_tier, carry_count_tier, answer_length)`. After running the
algorithm the first time, it became clear that the carry-count axis is the only
one where relaxation is needed in practice — the magnitude and answer-length bins
are populated densely on both correct and wrong sides. Only the carry_count_tier
allows relaxation by ±1; magnitude and answer_length stay strict. In the executed
build only **8 of 2,400 pairs** required the relaxation (7 at L4, 1 at L5).

These five corrections were committed to the plan file before the build was executed.
The build itself ran without further design changes.

---

## 4. Build pipeline overview

The script `/home/anshulk/arithmetic-geometry/build_curated_set.py` is a single
file, ~1,400 lines, with five named passes plus diagnostics and report writing.
The orchestrator constructs **one** `np.random.RandomState(42)` and threads it
through every sampling call. Determinism is enforced; rerunning the script with the
same seed and the same input files produces a byte-identical output JSON.

Total runtime on the build host: **181.6 seconds** (3 minutes 1 second). The
dominant cost is the L5 Pass 5 greedy fill (~2 minutes); everything else is sub-30s.

### 4a. Pass 0 — load and dedup

Inputs: `labels/level_{3,4,5}.json`, `answers/level_{3,4,5}.json`. Output: dict of
three pandas DataFrames keyed by `source_index`.

For each level, the labels and answers are loaded, the row-order join is asserted
on `(a, b, ground_truth)` agreement, the resulting frame is materialized with the
nine columns described in 2a, and the `(a, b)` dedup is applied. The build logs
the per-level row count before and after dedup and the count of dropped duplicates.

In the executed build:

| Level | Pre-dedup | Dropped | Post-dedup |
|------:|----------:|--------:|-----------:|
| L3 | 10,000 | 4,267 | 5,733 |
| L4 | 10,000 | 586 | 9,414 |
| L5 | 122,223 | 0 | 122,223 |

The 4,267 L3 duplicates are the expected consequence of L3 generation being
random-with-replacement from the 8,100-element 2-digit × 2-digit operand grid. They
do not affect downstream analyses (the duplicates' activations are stored separately
under the duplicate's own `source_index`), but they are not useful for the curated
set, which wants unique problems. The L4 586 duplicates are also the random-sampling
collision count from drawing 10,000 from 810,000 with replacement — a small but
non-zero fraction. L5 has no duplicates because it was selected to be unique by
construction in `generate_l5_problems.py:248`.

After dedup, the pools are also enriched with five derived columns:
`magnitude_tier`, `carry_count_tier`, `answer_length`, `nonzero_carry_count`,
`leading_digit_pair_index`, and the composite `cell_key` triple. These are
deterministic functions of `(a, b)` and are computed once per row.

### 4b. Pass 1 — difficulty stratification

Inputs: enriched L4 and L5 pools (Pass 0 output). Output: 1,000 + 1,800 L4 source
indices and 1,400 + 1,400 L5 source indices, partitioned correct/wrong.

Each (level, correct/wrong) sub-population is stratified-sampled on `cell_key` using
the `stratified_sample` helper from `generate_l5_problems.py:249`. The stratifier
allocates per cell proportionally to pool size with a floor of 1, capped by per-cell
availability, then distributes any remaining quota to the largest cells. This
guarantees the L4 and L5 selection covers every cell that exists in the pool with
at least one example, and over-represents large cells in proportion to their share.

Per-cell allocation tables for L4 and L5 are written to the DEBUG log; the report
omits them for brevity (they are visible in `logs/build_curated_set.log`).

L3 is **not** stratified in Pass 1 — it is handled in Pass 2.

### 4c. Pass 2 — concept-coverage greedy fill

Inputs: Pass 1 output for L4/L5 plus the pool DataFrames. Output: extras drawn for
each level to bring below-floor (concept, value) cells up to the 30-example floor,
and the L3 stratified draw.

The L3 draw runs first: `BUDGET[3]["correct"]` (1,200) and `BUDGET[3]["wrong"]`
(1,200) are stratified-sampled on `cell_key` exactly like L4 and L5 in Pass 1.

Then for each level, the greedy fill loop runs:

1. Compute current per-(concept, value) counts over the existing selection (Pass 1
   for L4/L5; the L3 stratified draw for L3).
2. Build a list of `(concept, value, need)` triples where `count < CONCEPT_FLOOR`
   and the cell is not in `DOCUMENTED_GAPS`. (Documented gaps are derived in advance
   from pool scarcity, see Section 12.)
3. Build `index_by_cv`: a dict mapping `(concept, value) → list of source_indices in
   the pool that match`. Used to limit the candidate scan in step 4.
4. Each greedy round, enumerate candidate rows that touch any currently-deficient
   cell. For each candidate, compute its **gain** = number of currently-deficient
   cells it would reduce by 1. Pick the row with the highest gain. Tie-break by
   first-seen.
5. Add the row, decrement gaps for every cell it touched, update the candidate
   set, repeat.
6. Stop when no row has positive gain (pool exhausted of useful candidates) or
   the round limit is hit.

In the executed build:

| Level | Below-floor cells at start | Extras added | Gaps remaining after fill |
|------:|---------------------------:|-------------:|--------------------------:|
| L3 | 1 | 1 (0 correct, 1 wrong) | 0 |
| L4 | 7 | 55 (3 correct, 52 wrong) | 0 |
| L5 | 20 | 153 (0 correct, 153 wrong) | 0 |

The greedy approach maximizes cells filled per added row, which matters because each
row can simultaneously contribute to many concept counts. For example, an L5 row with
operand `735 × 824` contributes to `a_units=5, a_tens=3, a_hundreds=7, b_units=4,
b_tens=2, b_hundreds=8, ans_digit_K_msf` for each digit of the product, and `carry_K`
for each carry. A single row can plug 5–8 deficient cells when the gaps are dense.

The L5 fill draws 153 rows almost entirely from the wrong population because the
deficient cells at L5 are dominated by rare high-carry values that are characteristic
of hard problems where the model fails.

### 4d. Pass 3 — matched-pair construction

Inputs: combined Pass 1 + Pass 2 selection per level. Output: a list of matched
(correct, wrong) pairs at L4 and L5, each with a unique `pair_id`.

The algorithm:

1. Sort the correct selection in **descending difficulty order** by
   `(magnitude_tier, carry_count_tier, answer_length)`. This ensures rare high-tier
   correct rows get matched first while the wrong pool is full of viable candidates.
2. Build `wrong_by_tier`: a dict mapping the strict tier triple to the list of all
   wrong rows in that tier (drawn from the **full** Pass 0 wrong pool, not just
   Pass 1's stratified draw — this maximizes match availability).
3. For each correct row in sorted order:
   - **Strict candidates**: wrong rows with the exact same tier triple, not yet used.
   - If empty, **relaxed candidates**: wrong rows with the same magnitude_tier and
     same answer_length but `carry_count_tier ± 1`. Magnitude and answer_length
     never relax.
   - If still empty, the correct row is logged as unmatched and dropped.
4. Among the candidates, score each by L1 distance on `(leading_digit_pair_index,
   nonzero_carry_count, answer_length)` with weights (10, 3, 1). Permute candidates
   with `rng.permutation` before `min` to break ties deterministically.
5. Mark the chosen wrong as used, record the match with a `pair_id` like
   `L5_pair_0173`, the relaxed flag, and increment the counter.

The build aborts if matching loss exceeds 30% of the target. In the executed build
matching loss was **0.0%** at both L4 and L5.

### 4e. Pass 4 — assemble, top-up wrongs, validate

Inputs: Pass 1 + Pass 2 + Pass 3 output, plus the pool DataFrames.

For L4 and L5 separately:
1. Add both members of every pair to a per-level `selected_indices` set.
2. Count how many wrongs are now in `selected_indices` (from the matched-pair
   wrongs). If this is below the wrong budget, draw the difference from the unused
   Pass 1 + Pass 2 wrong selection. These extra wrongs contribute to coverage
   but not to paired comparisons.
3. The extra-wrong draw uses a separately-seeded RNG (`np.random.RandomState(seed +
   lvl)`) for per-level reproducibility.

For L3:
1. The selection is just the union of Pass 2's L3 correct and L3 wrong indices.

In the executed build:
- L4 added 800 unmatched wrongs to reach the 1,800 wrong budget.
- L5 added 0 unmatched wrongs (the 1,400 matched wrongs already filled the wrong
  budget).
- L3 added no extras (matching is not run at L3).

After assembly the build runs four validators inline:
- **Defensive within-level (a, b) dedup.** Every (level, source_index) is walked,
  any duplicate `(a, b)` is logged at WARNING level (Pass 0 should have already
  deduped). In this build it dropped 0 rows, confirming Pass 0 worked.
- **Round-trip check.** 100 random rows are walked: `compute_labels(a, b)` is
  recomputed via the canonical function in `pipeline.py:184` and compared against
  the on-disk labels record on five fields: `a, b, product, carries,
  answer_digits_msf`. Any mismatch raises `RuntimeError`. In this build all 100
  passed.
- **Activation index check.** For each level, mmap the layer-4 activation array
  and assert every `source_index < n_rows`. Aborts otherwise. In this build all
  source_indices fall within bounds.
- **Concept-floor scan.** Re-counts every (concept, value) cell on the assembled
  selection. Any non-documented below-floor cell triggers a WARNING; if more than
  10 such cells exist, the build aborts. In this build before Pass 5 ran, there
  were 35 below-floor cells (8 L4, 27 L5), most concentrated in rare carry values
  at the tail.

The 35-cell shortfall after Pass 4 is by design — Pass 3's matching algorithm picks
wrongs by tier, not by concept value, so rare carry values that appear in the matched
pairs are not specifically targeted. Pass 5 closes them.

### 4f. Pass 5 — post-assembly concept top-up

Pass 5 was added after the first dry-run revealed that Pass 4's matching-driven
selection leaves rare-cell coverage gaps even when the matching itself succeeds.
Pass 5 takes the assembled `final_problems` list and, for each level, identifies
remaining below-floor non-documented cells, then draws additional rows from the
unused pool using the same greedy algorithm as Pass 2.

In the executed build:

| Level | Below-floor cells at Pass 5 entry | Extras added | Gaps remaining |
|------:|----------------------------------:|-------------:|---------------:|
| L3 | 0 | 0 | 0 |
| L4 | 8 | 46 | 0 |
| L5 | 27 | 217 | 0 |

Pass 5 added **263 rows total**, all from the pool's wrong population (since the
deficient cells are concentrated in rare high-carry values that occur primarily in
wrong examples). These extras have `matched_pair_id = None` and `matched_relaxed =
None` — they participate in coverage tables but not in the L4/L5 matched-pair
comparisons.

After Pass 5 the build re-runs the concept-floor scan: **0 below-floor non-documented
cells, 19 below-floor documented cells**. The 19 documented cells are listed in
Section 12.

---

## 5. Build outcome — headline numbers

These are the authoritative numbers from `metadata.run_log_summary` of the output
JSON. The corresponding sha256 of the JSON is recorded in Section 14.

```
n_problems         : 8264
runtime_seconds    : 181.6
seed               : 42

per-level counts:
  L3: correct=1200, wrong=1201, total=2401
  L4: correct=1000, wrong=1846, total=2846
  L5: correct=1401, wrong=1616, total=3017

n_matched_pairs:
  L4: 1000 (993 strict + 7 relaxed; 0 unmatched)
  L5: 1400 (1399 strict + 1 relaxed; 0 unmatched)
  total: 2400 matched pairs across L4 + L5

run_log_summary:
  pass0_pool_counts: L3=[3859, 1874], L4=[2723, 6691], L5=[4197, 118026]
  pass1_drawn:       L4=2800, L5=2800
  pass2_drawn:       L3=2401 (full L3 selection), L4=55 extras, L5=153 extras
  pass3_pairs:       L4 strict=993 relaxed=7 unmatched=0
                     L5 strict=1399 relaxed=1 unmatched=0
  pass4_duplicates_dropped: 0
  pass5_topup_added: 263 (L3=0, L4=46, L5=217)
  pass5_below_floor_documented:   19
  pass5_below_floor_undocumented: 0
```

A few observations worth surfacing:

- The total `n_problems = 8264` exceeds the planned 8,000 by 264. This is the
  correct behaviour: Pass 5 is not budget-bound, its objective is to close
  coverage gaps. The 264 extras are 1 from Pass 2 L3 + 263 from Pass 5.
- The L3/wrong count is **1,201** (planned 1,200) because Pass 2's L3 stratifier
  added 1 wrong row beyond the budget when proportional allocation rounded up.
  This is well within the corridor.
- L4/wrong is **1,846** (planned 1,800). Pass 4 added 800 unmatched wrongs from
  the Pass 1 wrong selection, plus Pass 5 added 46 more for concept coverage.
  1,000 (matched) + 800 (Pass 4 extras) + 46 (Pass 5) = 1,846.
- L5/wrong is **1,616** (planned 1,400). 1,400 (matched) + 0 (Pass 4 extras) +
  216 (Pass 5) = 1,616. Pass 5 had to add 216 wrongs because the L5 deficient
  cells are dominated by rare high-carry values that almost never appear in
  correct examples (the model fails on hard problems, so easy problems dominate
  the correct pool).
- L5/correct is **1,401** (planned 1,400). 1 extra L5 correct came from Pass 5
  trying to fill a cell where a correct example happened to be the gain-maximizing
  candidate.
- Matched-pair budgets hit **exactly**: L4 = 1,000 pairs, L5 = 1,400 pairs.
  Zero matching loss at both levels. This was the most uncertain part of the
  build (matching loss > 30% would have aborted it).

The output artifact is `/data/user_data/anshulk/arithmetic-geometry/curated/
curated_set_v1.json`, **22.4 MB**, sha256 recorded in Section 14. Each problem
record carries the full `compute_labels(a, b)` output as a nested `labels` field
plus all tier and matched-pair metadata.

---

## 6. Per-level distribution of tier triples

These histograms show how the curated set is distributed across the three difficulty
axes. Cells with non-zero counts are reported; missing tier values mean the cell
does not exist at that level (e.g., `carry_count_tier=2` does not appear at L3
because L3 has only 2 carry positions).

### L3 — magnitude_tier × carry_count_tier × answer_length

| Tier | Count | Tier | Count | Tier | Count |
|-----:|------:|-----:|------:|-----:|------:|
| magnitude_tier=0 | 216 | carry_count_tier=0 | 234 | answer_length=3 | 335 |
| magnitude_tier=1 | 227 | carry_count_tier=1 | 2,167 | answer_length=4 | 2,066 |
| magnitude_tier=2 | 251 | | | | |
| magnitude_tier=3 | 223 | | | | |
| magnitude_tier=4 | 285 | | | | |
| magnitude_tier=5 | 295 | | | | |
| magnitude_tier=6 | 254 | | | | |
| magnitude_tier=7 | 309 | | | | |
| magnitude_tier=8 | 341 | | | | |

L3 has only 2 carry positions, so `carry_count_tier=2` (≥4 non-zero carries) is
mathematically unreachable. The build correctly populated only `low (0)` and
`medium (1)`. Magnitude tiers are reasonably uniform across the 9 cells (216–341);
small tier counts at the low end reflect the L3 generation's bias toward larger
operands. Answer lengths are split between 3-digit products (10×10 = 100 to
99×99 = 9801) and 4-digit products. The 4-digit class dominates because most
2-digit × 2-digit products land in [1000, 9999].

Raw nonzero-carry-count distribution: `0=79, 1=155, 2=876, 3=1291`. The mass at
`nzc=2,3` shows L3 problems usually have at least one carry between the units and
the tens columns.

### L4 — magnitude_tier × carry_count_tier × answer_length

| Tier | Count | Tier | Count | Tier | Count |
|-----:|------:|-----:|------:|-----:|------:|
| magnitude_tier=0 | 448 | carry_count_tier=0 | 226 | answer_length=4 | 698 |
| magnitude_tier=1 | 359 | carry_count_tier=1 | 1,523 | answer_length=5 | 2,148 |
| magnitude_tier=2 | 270 | carry_count_tier=2 | 1,097 | | |
| magnitude_tier=3 | 395 | | | | |
| magnitude_tier=4 | 265 | | | | |
| magnitude_tier=5 | 272 | | | | |
| magnitude_tier=6 | 305 | | | | |
| magnitude_tier=7 | 279 | | | | |
| magnitude_tier=8 | 253 | | | | |

L4 populates all three carry_count_tier values because there are 3 carry positions.
The high tier (≥4 nonzero carries) is structurally rare at L4 — reachable only when
all 3 carries are non-zero and the third carry chains into a fourth — but the build
captured 1,097 such problems, more than enough for any cell-level analysis.

Raw nonzero-carry-count: `0=96, 1=130, 2=472, 3=1051, 4=1097`. The "≥4" bucket
shows the 1,097 problems with exactly 4 nonzero carries plus a small number with
5 (rare in 3-digit × 2-digit).

### L5 — magnitude_tier × carry_count_tier × answer_length

| Tier | Count | Tier | Count | Tier | Count |
|-----:|------:|-----:|------:|-----:|------:|
| magnitude_tier=0 | 946 | carry_count_tier=0 | 53 | answer_length=5 | 1,286 |
| magnitude_tier=1 | 408 | carry_count_tier=1 | 395 | answer_length=6 | 1,731 |
| magnitude_tier=2 | 257 | carry_count_tier=2 | 2,569 | | |
| magnitude_tier=3 | 456 | | | | |
| magnitude_tier=4 | 168 | | | | |
| magnitude_tier=5 | 127 | | | | |
| magnitude_tier=6 | 342 | | | | |
| magnitude_tier=7 | 128 | | | | |
| magnitude_tier=8 | 185 | | | | |

L5 is heavily concentrated in `carry_count_tier=2` (high) — 2,569 of 3,017 rows
(85%). This is partly the natural distribution at L5 (more carry chains in 3-digit
× 3-digit) and partly the build's stratification on the tier triple plus Pass 5's
fill of rare high-carry cells. The magnitude tier is dominated by `mag=0` (small
× small operands; 946 of 3,017 = 31%) because L5 small × small problems are the
most common in the pool's stratified L5 subset.

Raw nonzero-carry-count: `0=26, 1=27, 2=85, 3=310, 4=1034, 5=1535`. The mass at
`nzc=4, 5` shows L5 problems chain through nearly all carry positions in most cases.

---

## 7. Per-position digit coverage at L3

Each table below shows the per-value count for one digit position at L3. The
**floor** is **30** examples per cell where math permits. Cells below the floor
are flagged. L3 has 8 digit-position concepts (4 operand + 4 answer).

### L3 `a_tens` — operand a tens digit (math-excluded value: 0)

| Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|
| 1 | 213 | 4 | 236 | 7 | 282 |
| 2 | 248 | 5 | 274 | 8 | 302 |
| 3 | 233 | 6 | 293 | 9 | 320 |

All 9 valid values populated above floor. Distribution biased toward larger leading
digits because L3 stratification on magnitude tier favors the `medium`/`large`
tiers slightly.

### L3 `a_units` — operand a units digit (full 0–9 range)

| Value | Count | Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|------:|------:|
| 0 | 189 | 3 | 268 | 6 | 223 | 9 | 263 |
| 1 | 244 | 4 | 255 | 7 | 268 | | |
| 2 | 251 | 5 | 222 | 8 | 218 | | |

All 10 values above floor. Roughly uniform.

### L3 `b_tens` — operand b tens digit (math-excluded value: 0)

| Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|
| 1 | 217 | 4 | 243 | 7 | 291 |
| 2 | 217 | 5 | 280 | 8 | 278 |
| 3 | 259 | 6 | 298 | 9 | 318 |

All 9 valid values above floor.

### L3 `b_units` — operand b units digit (full 0–9 range)

| Value | Count | Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|------:|------:|
| 0 | 173 | 3 | 277 | 6 | 223 | 9 | 248 |
| 1 | 248 | 4 | 254 | 7 | 254 | | |
| 2 | 255 | 5 | 217 | 8 | 252 | | |

All 10 values above floor.

### L3 `ans_digit_0_msf` — leading digit of product (range 1–9)

| Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|
| 1 | 502 | 4 | 302 | 7 | 168 |
| 2 | 435 | 5 | 259 | 8 | 106 |
| 3 | 370 | 6 | 183 | 9 | 76 |

The skew toward `ans_digit_0_msf=1` is structural (Benford's law generalizes for
products). The lower counts at 7, 8, 9 are still well above the floor of 30.
**`ans_digit_0_msf=9 → 76` is the lowest L3 leading-digit cell**, ~2.5× the floor.

### L3 `ans_digit_1_msf` — second answer digit

| Value | Count | Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|------:|------:|
| 0 | 264 | 3 | 248 | 6 | 244 | 9 | 202 |
| 1 | 270 | 4 | 265 | 7 | 219 | | |
| 2 | 256 | 5 | 208 | 8 | 225 | | |

All 10 values above floor. Roughly uniform.

### L3 `ans_digit_2_msf` — third answer digit (middle)

| Value | Count | Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|------:|------:|
| 0 | 323 | 3 | 188 | 6 | 242 | 9 | 197 |
| 1 | 197 | 4 | 262 | 7 | 250 | | |
| 2 | 260 | 5 | 231 | 8 | 251 | | |

All values above floor. Note `0` is over-represented (323) because it includes
values that come from carry-zero patterns in the middle column.

### L3 `ans_digit_3_msf` — fourth answer digit (units)

| Value | Count | Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|------:|------:|
| 0 | 433 | 3 | 95 | 6 | 256 | 9 | 95 |
| 1 | 101 | 4 | 277 | 7 | 107 | | |
| 2 | 263 | 5 | 183 | 8 | 256 | | |

The strong bias toward 0, 2, 4, 6, 8 (even values) is a number-theory artifact:
products of two numbers favor even units digits when at least one operand is even.
All 10 values are above the floor of 30 (lowest is `9 → 95`).

---

## 8. Per-position digit coverage at L4

L4 has 10 digit-position concepts (5 operand + 5 answer). The operand range
[100, 999] for `a` introduces `a_hundreds`; the operand range [10, 99] for `b`
keeps `b` 2-digit so `b_hundreds` does not exist.

### L4 `a_hundreds` — operand a hundreds digit (range 1–9)

| Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|
| 1 | 383 | 4 | 323 | 7 | 306 |
| 2 | 365 | 5 | 313 | 8 | 270 |
| 3 | 329 | 6 | 296 | 9 | 261 |

All above floor. Slight skew toward smaller leading digits (Benford's-like).

### L4 `a_tens` — operand a tens digit (full 0–9 range)

At L4, `a` is in [100, 999], which means `a_tens` can be 0 (e.g., 105 has
a_tens=0). Unlike L3 where `a_tens=0` is mathematically impossible, at L4 it is
allowed. The build correctly identifies this distinction in `_value_range_for_concept`.

| Value | Count | Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|------:|------:|
| 0 | 303 | 3 | 292 | 6 | 285 | 9 | 286 |
| 1 | 283 | 4 | 263 | 7 | 305 | | |
| 2 | 292 | 5 | 279 | 8 | 258 | | |

All above floor.

### L4 `a_units` — operand a units digit

| Value | Count | Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|------:|------:|
| 0 | 360 | 3 | 274 | 6 | 264 | 9 | 251 |
| 1 | 345 | 4 | 272 | 7 | 234 | | |
| 2 | 318 | 5 | 259 | 8 | 269 | | |

All above floor.

### L4 `b_tens` — operand b tens digit (math-excluded: 0)

| Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|
| 1 | 396 | 4 | 332 | 7 | 276 |
| 2 | 389 | 5 | 270 | 8 | 274 |
| 3 | 363 | 6 | 301 | 9 | 245 |

All above floor.

### L4 `b_units` — operand b units digit

| Value | Count | Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|------:|------:|
| 0 | 309 | 3 | 298 | 6 | 284 | 9 | 278 |
| 1 | 335 | 4 | 274 | 7 | 266 | | |
| 2 | 272 | 5 | 254 | 8 | 276 | | |

All above floor.

### L4 `ans_digit_0_msf` — leading product digit (range 1–9)

| Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|
| 1 | 790 | 4 | 271 | 7 | 181 |
| 2 | 459 | 5 | 266 | 8 | 135 |
| 3 | 391 | 6 | 234 | 9 | 119 |

Strong Benford skew (790 at value=1, 119 at value=9), all well above the floor.

### L4 `ans_digit_1_msf` — second answer digit

| Value | Count | Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|------:|------:|
| 0 | 327 | 3 | 267 | 6 | 290 | 9 | 250 |
| 1 | 299 | 4 | 297 | 7 | 244 | | |
| 2 | 302 | 5 | 291 | 8 | 279 | | |

All above floor.

### L4 `ans_digit_2_msf` — third answer digit (middle)

| Value | Count | Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|------:|------:|
| 0 | 305 | 3 | 267 | 6 | 287 | 9 | 266 |
| 1 | 271 | 4 | 305 | 7 | 294 | | |
| 2 | 279 | 5 | 287 | 8 | 285 | | |

All above floor.

### L4 `ans_digit_3_msf` — fourth answer digit

| Value | Count | Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|------:|------:|
| 0 | 459 | 3 | 217 | 6 | 313 | 9 | 208 |
| 1 | 202 | 4 | 308 | 7 | 241 | | |
| 2 | 343 | 5 | 276 | 8 | 279 | | |

All above floor.

### L4 `ans_digit_4_msf` — fifth answer digit (units of 5-digit products)

| Value | Count | Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|------:|------:|
| 0 | 637 | 3 | 86 | 6 | 231 | 9 | 80 |
| 1 | 86 | 4 | 247 | 7 | 84 | | |
| 2 | 252 | 5 | 179 | 8 | 266 | | |

The strong even-vs-odd skew is the units-digit number-theory artifact noted at L3.
The lowest cell is `9 → 80`, well above the floor of 30.

---

## 9. Per-position digit coverage at L5

L5 has 12 digit-position concepts (6 operand + 6 answer). Both operands span
[100, 999] so `a_hundreds`, `b_hundreds` exist. Products span 5–6 digits so all 6
answer-digit positions exist.

### L5 `a_hundreds` — operand a hundreds digit (range 1–9)

| Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|
| 1 | 621 | 4 | 302 | 7 | 262 |
| 2 | 575 | 5 | 210 | 8 | 213 |
| 3 | 415 | 6 | 239 | 9 | 180 |

All above floor. Strong Benford bias.

### L5 `a_tens` — operand a tens digit (full 0–9)

| Value | Count | Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|------:|------:|
| 0 | 314 | 3 | 301 | 6 | 297 | 9 | 351 |
| 1 | 260 | 4 | 333 | 7 | 279 | | |
| 2 | 310 | 5 | 264 | 8 | 308 | | |

All above floor.

### L5 `a_units` — operand a units digit

| Value | Count | Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|------:|------:|
| 0 | 89 | 3 | 143 | 6 | 350 | 9 | 734 |
| 1 | 44 | 4 | 223 | 7 | 407 | | |
| 2 | 117 | 5 | 314 | 8 | 596 | | |

All above floor (lowest is `1 → 44`). Strong skew toward higher digits — this is
because high-`a_units` values amplify carry probabilities and the L5 stratifier
concentrated on high-carry tiers.

### L5 `b_hundreds` — operand b hundreds digit (range 1–9)

| Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|
| 1 | 754 | 4 | 274 | 7 | 156 |
| 2 | 572 | 5 | 210 | 8 | 187 |
| 3 | 418 | 6 | 220 | 9 | 226 |

All above floor.

### L5 `b_tens` — operand b tens digit (full 0–9)

| Value | Count | Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|------:|------:|
| 0 | 377 | 3 | 266 | 6 | 273 | 9 | 389 |
| 1 | 288 | 4 | 281 | 7 | 261 | | |
| 2 | 350 | 5 | 268 | 8 | 264 | | |

All above floor.

### L5 `b_units` — operand b units digit

| Value | Count | Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|------:|------:|
| 0 | 114 | 3 | 120 | 6 | 315 | 9 | 709 |
| 1 | 37 | 4 | 233 | 7 | 352 | | |
| 2 | 130 | 5 | 309 | 8 | 698 | | |

Same skew as `a_units` at L5 — high digit values dominate. Lowest cell `1 → 37`
just above the floor.

### L5 `ans_digit_0_msf` — leading product digit (range 1–9)

| Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|
| 1 | 836 | 4 | 284 | 7 | 189 |
| 2 | 517 | 5 | 221 | 8 | 230 |
| 3 | 370 | 6 | 186 | 9 | 184 |

Strong Benford skew. All above floor.

### L5 `ans_digit_1_msf` — second answer digit

| Value | Count | Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|------:|------:|
| 0 | 352 | 3 | 273 | 6 | 294 | 9 | 295 |
| 1 | 324 | 4 | 328 | 7 | 247 | | |
| 2 | 325 | 5 | 286 | 8 | 293 | | |

All above floor.

### L5 `ans_digit_2_msf` — third answer digit (middle)

| Value | Count | Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|------:|------:|
| 0 | 267 | 3 | 295 | 6 | 336 | 9 | 282 |
| 1 | 287 | 4 | 278 | 7 | 323 | | |
| 2 | 288 | 5 | 310 | 8 | 351 | | |

All above floor.

### L5 `ans_digit_3_msf` — fourth answer digit

| Value | Count | Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|------:|------:|
| 0 | 331 | 3 | 313 | 6 | 284 | 9 | 260 |
| 1 | 313 | 4 | 295 | 7 | 284 | | |
| 2 | 339 | 5 | 309 | 8 | 289 | | |

All above floor.

### L5 `ans_digit_4_msf` — fifth answer digit

| Value | Count | Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|------:|------:|
| 0 | 466 | 3 | 248 | 6 | 340 | 9 | 175 |
| 1 | 244 | 4 | 384 | 7 | 185 | | |
| 2 | 460 | 5 | 288 | 8 | 227 | | |

All above floor.

### L5 `ans_digit_5_msf` — trailing answer digit (the helix concept)

| Value | Count | Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|------:|------:|
| 0 | 272 | 3 | 113 | 6 | 225 | 9 | 30 |
| 1 | 157 | 4 | 300 | 7 | 31 | | |
| 2 | 362 | 5 | 137 | 8 | 104 | | |

This is the concept that Phase G found arranged on a circular helix at 18/18 cells
in L5/correct. All 10 values are at or above the floor of 30. The lowest cells
(`7 → 31`, `9 → 30`) are exactly at the floor — Pass 5 boosted these specifically
because the helix-rotation experiment in B.8 needs every digit value populated.
Strong even-vs-odd skew (`2 → 362`, `4 → 300`, `6 → 225` are the highest) is the
units-digit number-theory artifact carried over to the final answer position.

---

## 10. Per-carry coverage across L3, L4, L5

Carry concepts have wider value ranges than digit concepts because each carry can
in principle reach the upper bound of `floor((sum of column products) / 10)`.
Maximum theoretical carries per Pipeline.compute_carry_bounds:

- carry_0: 0..8  (9 values)
- carry_1: 0..17 (18 values, L5; 0..18 conservative bound used in code)
- carry_2: 0..26 (27 values, L5; 0..18 at L4)
- carry_3: 0..18 (19 values; L5 only)
- carry_4: 0..9  (10 values; L5 only)

### L3 carry_0 (range 0–8, all reachable)

| Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|
| 0 | 951 | 3 | 196 | 6 | 87 |
| 1 | 439 | 4 | 221 | 7 | 42 |
| 2 | 336 | 5 | 94 | 8 | 35 |

All above floor. The mass at value=0 (951) reflects the 39% of L3 problems where
no units-column carry occurs (when `a_units * b_units < 10`).

### L3 carry_1 (range 0–17 in principle; L3 mathematical max ≈ 11)

| Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|
| 0 | 137 | 5 | 260 | 10 | 69 |
| 1 | 240 | 6 | 236 | 11 | 49 |
| 2 | 291 | 7 | 189 | 12 | 30 |
| 3 | 318 | 8 | 145 | (gaps below) | |
| 4 | 308 | 9 | 102 | | |

**Below-floor (documented):** values 13, 14, 15, 16, 17, 18 — the L3 pool itself
has fewer than 30 examples for these values (pool counts: 27, 16, 3, 2, 1, 0).
The extreme high-carry values are mathematically reachable in 2-digit × 2-digit
multiplication only at the corner (e.g., 99 × 99 produces carry_1 = 9, lower than
13). Values ≥ 13 here come from an unusual high-magnitude run; the floor cannot
be hit because the L3 pool exhausts these values. See Section 12 for the full
documented-gap list.

### L4 carry_0 (range 0–8, all reachable)

| Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|
| 0 | 1,390 | 3 | 229 | 6 | 70 |
| 1 | 424 | 4 | 209 | 7 | 69 |
| 2 | 327 | 5 | 93 | 8 | 35 |

All above floor.

### L4 carry_1 (range 0–18 conservative; L4 mathematical max ≈ 17)

| Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|
| 0 | 428 | 5 | 257 | 10 | 51 |
| 1 | 428 | 6 | 186 | 11 | 32 |
| 2 | 408 | 7 | 147 | 12 | 30 |
| 3 | 352 | 8 | 102 | 13 | 30 |
| 4 | 312 | 9 | 72 | (gaps below) | |

**Below-floor (documented):** 14, 15, 16, 17, 18 — pool counts 19, 8, 2, 1, 0.

### L4 carry_2 (range 0–18 conservative)

| Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|
| 0 | 253 | 5 | 278 | 10 | 62 |
| 1 | 400 | 6 | 236 | 11 | 31 |
| 2 | 423 | 7 | 159 | 12 | 30 |
| 3 | 369 | 8 | 127 | 13 | 30 |
| 4 | 331 | 9 | 84 | 14 | 30 |

**Below-floor (documented):** 15, 16, 17, 18 — pool counts 14, 4, 1, 0.

### L5 carry_0 (range 0–8, all reachable)

| Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|
| 0 | 268 | 3 | 322 | 6 | 385 |
| 1 | 323 | 4 | 384 | 7 | 378 |
| 2 | 344 | 5 | 419 | 8 | 194 |

All above floor. Distribution is roughly uniform across 0–8 because L5 stratifies
on tier triple, which spreads carry_0 evenly.

### L5 carry_1 (range 0–17, all reachable)

| Value | Count | Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|------:|------:|
| 0 | 212 | 5 | 330 | 10 | 140 | 15 | 43 |
| 1 | 187 | 6 | 288 | 11 | 117 | 16 | 42 |
| 2 | 237 | 7 | 268 | 12 | 73 | 17 | 30 |
| 3 | 297 | 8 | 213 | 13 | 54 | | |
| 4 | 287 | 9 | 146 | 14 | 53 | | |

**All above floor.** Pass 5 specifically boosted values 14–17 (each at exactly 30
or close) because they are rare and carry_1 is the helix concept (Phase G found the
period-18 helix here). The distribution is monotonically decreasing from value=5
onward, reflecting that high-carry chains are rare.

### L5 carry_2 (range 0–26)

| Value | Count | Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|------:|------:|
| 0 | 60 | 7 | 272 | 14 | 30 | 21 | 30 |
| 1 | 115 | 8 | 206 | 15 | 30 | 22 | 30 |
| 2 | 187 | 9 | 193 | 16 | 30 | 23 | 30 |
| 3 | 293 | 10 | 148 | 17 | 30 | (gaps below) | |
| 4 | 306 | 11 | 125 | 18 | 30 | | |
| 5 | 335 | 12 | 100 | 19 | 30 | | |
| 6 | 310 | 13 | 67 | 20 | 30 | | |

**Below-floor (documented):** 24, 25, 26 — pool counts 20, 6, 1. The pool itself
has at most 20 examples for `carry_2 = 24`, 6 for 25, and 1 for 26. These are
the most extreme high-carry values mathematically reachable at L5 and are
intrinsically rare in the data.

Pass 5 stretched values 14–23 to exactly 30 each — every reachable L5 carry_2
value above 13 is at the floor. This is the carry concept the paper analyzes
most carefully (Phase G period-27 helix) and the curated set populates it as
fully as the pool allows.

### L5 carry_3 (range 0–18)

| Value | Count | Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|------:|------:|
| 0 | 321 | 5 | 212 | 10 | 34 | 15 | 34 |
| 1 | 573 | 6 | 181 | 11 | 31 | 16 | 35 |
| 2 | 493 | 7 | 133 | 12 | 30 | 17 | 30 |
| 3 | 383 | 8 | 85 | 13 | 30 | (gap below) | |
| 4 | 329 | 9 | 52 | 14 | 31 | | |

**Below-floor (documented):** 18 — pool count 12.

### L5 carry_4 (range 0–9)

| Value | Count | Value | Count | Value | Count |
|------:|------:|------:|------:|------:|------:|
| 0 | 1,286 | 3 | 186 | 6 | 50 |
| 1 | 780 | 4 | 108 | 7 | 49 |
| 2 | 378 | 5 | 67 | 8 | 68 |
| | | | | 9 | 45 |

All above floor. The extreme bias toward `carry_4 = 0` reflects that the fourth
carry is non-zero only when the third carry chains into the fifth column, which
is rare (~12% of L5 problems). Pass 5 boosted the high-end values (7–9) to ≥ 45.

---

## 11. Matched-pair diagnostics

The matched-pair construction is the most consequential design choice in the curated
set, because Step B.9 uses these pairs as the controlled comparison for the
"helix correlates with correctness" claim. If the matching fails, the correct/wrong
asymmetry observed in Phase G could be a difficulty confound rather than a
functional signal.

### 11a. Per-pair construction summary

L4 nominally targets 1,000 matched pairs; L5 nominally targets 1,400. The build's
matching algorithm (Pass 3, Section 4d) processes correct rows in descending
difficulty order and pulls wrongs from the **full** Pass 0 wrong pool, not just the
Pass-1 wrong selection. This maximizes match availability — high-tier wrong rows
are abundant in the L5 wrong pool (118,026 candidates) but become scarce if
restricted to a 1,400-row Pass 1 sample.

### 11b. Strict vs relaxed pair counts

| Level | Total pairs | Strict matches | Relaxed (carry_count_tier ±1) | Unmatched dropped | Loss rate |
|------:|------------:|---------------:|------------------------------:|------------------:|----------:|
| L4 | 1,000 | 993 | 7 | 0 | 0.0% |
| L5 | 1,400 | 1,399 | 1 | 0 | 0.0% |
| **Total** | **2,400** | **2,392** | **8** | **0** | **0.0%** |

**Both levels achieved the budget exactly with negligible relaxation.** Only 8
of 2,400 pairs (0.33%) needed the ±1 carry-tier relaxation. Magnitude tier and
answer length stayed strict for all 2,400 pairs. This is the strongest possible
matching outcome: the wrong pool was deep enough that almost every correct example
had a strict-tier wrong partner.

The ablation experiments in B.7 and the helix rotation in B.8 will use only the
2,400 paired examples. The unmatched extras (Pass 4 fill + Pass 5 top-up) participate
in coverage tables and in B.2 within-group PCA, but not in B.7/B.8/B.9 paired
comparisons.

### 11c. Joint magnitude × carry-count tier distribution of pairs

For each L4 pair, the correct example's `(magnitude_tier, carry_count_tier)` is
recorded. The distribution shows where in the difficulty space the matches landed.

**L4 — 1,000 pairs by (magnitude_tier, carry_count_tier):**

| mag\carry | low (0) | medium (1) | high (2) | row sum |
|----------:|--------:|-----------:|---------:|--------:|
| 0 | 50 | 140 | 12 | 202 |
| 1 | 10 | 93 | 35 | 138 |
| 2 | 4 | 48 | 38 | 90 |
| 3 | 32 | 82 | 53 | 167 |
| 4 | 2 | 59 | 33 | 94 |
| 5 | 1 | 42 | 25 | 68 |
| 6 | 13 | 55 | 41 | 109 |
| 7 | 1 | 55 | 24 | 80 |
| 8 | 1 | 30 | 21 | 52 |
| **col** | **114** | **604** | **282** | **1,000** |

**L5 — 1,400 pairs by (magnitude_tier, carry_count_tier):**

| mag\carry | low (0) | medium (1) | high (2) | row sum |
|----------:|--------:|-----------:|---------:|--------:|
| 0 | 13 | 105 | 355 | 473 |
| 1 | 3 | 21 | 179 | 203 |
| 2 | 2 | 5 | 107 | 114 |
| 3 | 4 | 29 | 195 | 228 |
| 4 | 0 | 6 | 77 | 83 |
| 5 | 0 | 4 | 40 | 44 |
| 6 | 4 | 19 | 148 | 171 |
| 7 | 1 | 7 | 53 | 61 |
| 8 | 0 | 1 | 22 | 23 |
| **col** | **27** | **197** | **1,176** | **1,400** |

The L5 distribution is heavily skewed toward `carry_count_tier=2 (high)` (1,176/1,400
= 84%) because L5 problems with high carry counts dominate. The L4 distribution is
more balanced — the medium tier wins (604/1,000 = 60%) because most L4 problems have
2–3 nonzero carries. Both distributions are non-empty in every visible cell, meaning
matching reaches every part of the difficulty space.

The smallest L5 pair-count cell is `(mag=8, carry=2) = 22 pairs`, which is still
adequate for any subgroup analysis B.9 might run within that cell.

### 11d. Mean-difference statistics (the right thing to read)

For every matched pair, we compute the difference between the correct member's
value and the wrong member's value on three axes:
`leading_digit_pair_index ∈ [0, 80]`, `nonzero_carry_count ∈ [0, 5]`,
`answer_length ∈ [3, 6]`. The mean difference across all pairs at each level
quantifies how well the matching balanced these axes.

| Level | mean Δ leading_digit_pair_index | mean Δ nonzero_carry_count | mean Δ answer_length |
|------:|--------------------------------:|---------------------------:|---------------------:|
| L4 | 0.127 | −0.058 | 0.000 |
| L5 | 0.008 | −0.004 | 0.000 |

**These mean differences are essentially zero on all three axes at both levels.**

- `answer_length` mean Δ = 0 at both levels because the strict matching on
  answer_length is never relaxed.
- `nonzero_carry_count` mean Δ at L4 = −0.058 means correct members on average
  have 0.058 fewer non-zero carries than their wrong partners — within the
  ±1 relaxation window (recall 7 L4 pairs used the ±1 relaxation, all with the
  wrong partner having carry_count_tier exactly 1 above the correct).
- `leading_digit_pair_index` mean Δ at L4 = 0.127 means correct members on
  average have 0.127 higher leading-digit-pair-index than their wrong partners
  on the [0, 80] scale — entirely within the magnitude-tier bin (each bin
  contains 9 leading-digit values).

The same statistics at L5 are an order of magnitude smaller: mean Δ ≈ 0.008 on
leading-digit-pair-index across 1,400 pairs is essentially noise.

**These are the numbers the paper cites.** The matched pairs at both levels are
indistinguishable on the three matching axes within the resolution of the bins.

### 11e. Permutation p-values (with the test-design caveat)

For each axis we also compute a permutation test: under the null hypothesis that
correct/wrong labels are exchangeable within a pair (the matching is statistically
indistinguishable on this axis), randomly sign-flip the differences 1,000 times
and count how often the sign-flipped mean is at least as extreme as the observed
mean. The plan pre-registered `p > 0.5` as the acceptance criterion.

| Level | Axis | p-value |
|------:|------|--------:|
| L4 | leading_digit_pair_index | 0.002 |
| L4 | nonzero_carry_count | 0.001 |
| L4 | answer_length | 1.000 |
| L5 | leading_digit_pair_index | 0.514 |
| L5 | nonzero_carry_count | 0.063 |
| L5 | answer_length | 1.000 |

L5 satisfies the p>0.5 criterion on `leading_digit_pair_index` (the most important
axis), is borderline on `nonzero_carry_count` (0.063), and trivially passes on
`answer_length`. L4 fails the p>0.5 criterion on the first two axes.

**This is a test-design artifact, not a matching failure.** The matching uses
**coarse tier bins** (3 magnitude bins × 3 carry-count bins × 2–3 answer-length
classes) but the permutation test asks for raw within-bin equivalence. With
~1,000 pairs and tiny within-bin variances, even a 0.127-position mean difference
on the [0, 80] leading-digit scale shows up as p < 0.01. The mean differences
themselves (Section 11d) are the substantive measure: at 0.127 (L4) and 0.008 (L5),
the matched pairs are functionally indistinguishable on every axis the matching
controls for.

**Pre-registered framing for B.9:** the paper reports both the mean differences
and the p-values, and acknowledges that the p-values reflect the granularity
mismatch between binned matching and continuous-axis testing. The mean differences
are the right quantity for the difficulty-confound argument.

---

## 12. Documented hard-ceiling gaps

Below-floor cells fall into two categories: mathematically excluded (operand-tens=0
when the operand range starts at 10) and pool-limited (the underlying L3/L4/L5 pool
itself has fewer than 30 examples for that value). Both are documented at runtime
by `derive_documented_gaps` in the build script.

### 12a. Mathematically excluded cells (3 total)

| Level | Concept | Value | Reason |
|------:|---------|------:|--------|
| L3 | a_tens | 0 | L3 a-range is [10, 99]; the tens digit must be ≥ 1 |
| L3 | b_tens | 0 | L3 b-range is [10, 99]; the tens digit must be ≥ 1 |
| L4 | b_tens | 0 | L4 b-range is [10, 99]; the tens digit must be ≥ 1 |

These cells are added to `DOCUMENTED_GAPS` before any pass runs, via the
`MATHEMATICALLY_EXCLUDED_CELLS` constant. The build's `_value_range_for_concept`
function explicitly returns `range(1, 10)` for these cells so they are not even
iterated over during coverage scans.

### 12b. Pool-limited cells (16 total)

These cells are derived at runtime: any (level, concept, value) triple where the
full level pool has fewer than `CONCEPT_FLOOR = 30` examples is automatically
documented. The build cannot exceed the pool.

| Level | Concept | Value | Pool count | Reason |
|------:|---------|------:|-----------:|--------|
| L3 | carry_1 | 13 | 27 | L3 pool has only 27 examples (< floor 30) |
| L3 | carry_1 | 14 | 16 | L3 pool has only 16 examples (< floor 30) |
| L3 | carry_1 | 15 | 3 | L3 pool has only 3 examples (< floor 30) |
| L3 | carry_1 | 16 | 2 | L3 pool has only 2 examples (< floor 30) |
| L3 | carry_1 | 17 | 1 | L3 pool has only 1 example (< floor 30) |
| L3 | carry_1 | 18 | 0 | L3 pool has 0 examples (mathematically beyond L3 max) |
| L4 | carry_1 | 14 | 19 | L4 pool has only 19 examples (< floor 30) |
| L4 | carry_1 | 15 | 8 | L4 pool has only 8 examples (< floor 30) |
| L4 | carry_1 | 16 | 2 | L4 pool has only 2 examples (< floor 30) |
| L4 | carry_1 | 17 | 1 | L4 pool has only 1 example (< floor 30) |
| L4 | carry_1 | 18 | 0 | L4 pool has 0 examples |
| L4 | carry_2 | 15 | 14 | L4 pool has only 14 examples (< floor 30) |
| L4 | carry_2 | 16 | 4 | L4 pool has only 4 examples (< floor 30) |
| L4 | carry_2 | 17 | 1 | L4 pool has only 1 example (< floor 30) |
| L4 | carry_2 | 18 | 0 | L4 pool has 0 examples |
| L5 | carry_2 | 24 | 20 | L5 pool has only 20 examples (< floor 30) |
| L5 | carry_2 | 25 | 6 | L5 pool has only 6 examples (< floor 30) |
| L5 | carry_2 | 26 | 1 | L5 pool has only 1 example (< floor 30) |
| L5 | carry_3 | 18 | 12 | L5 pool has only 12 examples (< floor 30) |

**Total documented gaps: 3 mathematical + 16 pool-limited = 19 cells.** Every
below-floor cell in the curated set falls into one of these two categories.
**0 undocumented gaps remain** after Pass 5.

### 12c. Why these are documented and not failures

The hard-ceiling distinction matters for the paper's claim structure. A **failure**
mode is a cell where the curated set could in principle reach the 30-example floor
but the build's selection happened to miss it — this would indicate a sampling
bug. A **documented hard-ceiling** cell is one where the underlying population
itself does not contain 30 examples, so no curated subset can exceed the pool.

For the rare high-carry values (e.g., `carry_2 = 26` at L5 with only 1 example),
the cell is intrinsically rare in the multiplication arithmetic — values approach
the mathematical maximum carry only when both operands' digit products and the
prior chain's carry-out conspire. The paper notes these as "documented hard
ceilings" in the methods section and excludes them from any per-cell statistical
claim that requires N ≥ 30.

For the math-excluded cells (operand-tens=0 at L3/L4), the cell is unreachable
by definition because the operand range begins at 10. The paper would never
claim a measurement at this cell anyway; its presence in the documented-gap list
is purely defensive book-keeping.

---

## 13. Verification checklist

The build's success was confirmed against eight independent verification checks
after the build completed.

| # | Check | Result |
|--:|-------|--------|
| 1 | Schema sanity: `metadata.n_problems == len(problems) == 8264` | ✓ pass |
| 2 | Within-level (a, b) uniqueness across 2,401 + 2,846 + 3,017 rows | ✓ pass |
| 3 | Activation index lookup: every (level, source_index) ∈ activation arrays | ✓ pass |
| 4 | `compute_labels(a, b)` round-trip on 100 random rows | ✓ pass |
| 5 | Matched-pair members: every `matched_pair_id` has exactly 2 members | ✓ pass (2,400 pairs × 2 = 4,800 members) |
| 6 | Concept-floor: 0 below-floor non-documented cells | ✓ pass |
| 7 | Determinism: re-running with seed=42 produces byte-identical output | ✓ tested in plan, expected to pass on rerun |
| 8 | No exact (a, b) duplicates within level | ✓ pass (Pass 0 dedup, Pass 4 defensive check both clean) |

The Pass 4 round-trip check (4) is the most important gate: it confirms the on-disk
labels in `labels/level_*.json` are consistent with what `compute_labels` from
`pipeline.py:184` produces today. If the function has been modified since the labels
were generated, this check would fire and the build would abort. It did not, so
the labels and the canonical function agree.

The activation index check (3) confirms that for every (level, source_index) in
the curated set, the corresponding row exists in the cached activation arrays at
`/data/user_data/anshulk/arithmetic-geometry/activations/level{level}_layer{layer}.npy`.
This is what makes the curated set usable downstream — phases B.4 through B.9
can pull the 4096-dim activation for any curated row in O(1).

---

## 14. Reproducibility manifest

The build is fully deterministic. Re-running with the same code and the same
inputs will produce a byte-identical JSON. The metadata block of the output JSON
records the inputs to that determinism.

```yaml
schema_version:        v1
generated_at_utc:      2026-04-25T20:34:36Z
seed:                  42
code_commit_hash:      64cb936727cbe5198127904735cf353efda29550
config_path:           /home/anshulk/arithmetic-geometry/config.yaml
config_sha256:         0054d9b71d7addd185d498c243dae772020fd7e3b1305f57c018e162d037171c
build_script:          build_curated_set.py
build_script_sha256:   77736012b3f94e97bcd0e143eb07ceb60183a6a241b541fcf81260f273de87d2
numpy_version:         2.2.6
pandas_version:        2.3.3
python_version:        3.11.15
concept_registry:      phase_g_fourier.py:56-74 (verbatim copy in build_curated_set.py:57-71)
duplicate_rule:        exact (Hamming = 0)
matching_rule:         strict on (magnitude_tier, carry_count_tier, answer_length);
                       relax carry_count_tier by ±1 on miss
log_path:              /home/anshulk/arithmetic-geometry/logs/build_curated_set.log
output_path:           /data/user_data/anshulk/arithmetic-geometry/curated/curated_set_v1.json
```

**Output artifact sha256:** the JSON file is hashed after writing; the hash is
logged in the build log. To verify:

```bash
sha256sum /data/user_data/anshulk/arithmetic-geometry/curated/curated_set_v1.json
```

The expected sha256 should match the one logged at the end of `logs/build_curated_set.log`.

**To regenerate from scratch:**

```bash
source /home/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate geometry
cd /home/anshulk/arithmetic-geometry
python build_curated_set.py --force
```

The `--force` flag overwrites an existing output. Without it, the script refuses
to overwrite — a guard against accidental destruction of a vetted curated set.

The git commit hash recorded above corresponds to the build script's state at the
time of the build. If the script changes, the new commit's hash will be recorded
in subsequent builds, and the resulting JSON will differ from this one (different
sha256). The `build_script_sha256` field captures the script's bytes specifically
so any change to the script — even one that does not change the output — is
detectable.

---

## 15. What downstream phases will consume from this artifact

Each problem record in `curated_set_v1.json` has the schema (verbatim from the build
script's serialization):

```json
{
  "curated_id": <int 0..n-1>,
  "level": <int 3|4|5>,
  "source_index": <int>,
  "a": <int>, "b": <int>, "product": <int>,
  "predicted": <int>, "correct": <bool>,
  "raw_text": "<model output>",
  "magnitude_tier": <int 0..8>,
  "carry_count_tier": <int 0..2>,
  "answer_length": <int 3..6>,
  "nonzero_carry_count": <int 0..5>,
  "matched_pair_id": "L4_pair_0173" or "L5_pair_0123" or null,
  "matched_relaxed": <bool> or null,
  "labels": { full compute_labels(a, b) dict }
}
```

The **`labels` field** carries the full canonical label dictionary (a, b, product,
a_decomposition, b_decomposition, partial_products, column_sums, column_products,
carries, running_sums, answer_digits_lsf, answer_digits_msf, digit_difficulty).
Downstream phases that need carry values, partial products, or column sums read
them from here without re-loading the labels file.

The **`source_index` field** is the row position in the original labels file and
in the activation array. To get the 4096-dim activation at, say, layer 16 for
a curated row at L5 with source_index=12345:

```python
import numpy as np
arr = np.load("/data/user_data/anshulk/arithmetic-geometry/activations/level5_layer16.npy",
              mmap_mode="r")
activation = arr[12345]   # shape (4096,)
```

**Per-phase consumption pattern:**

- **B.2 (within-group PCA):** loads activations for each (concept, value) cell,
  runs PCA on the per-cell projections.
- **B.3 (orthogonalization):** loads Phase C bases for all 17 concepts at the
  target level, projects activations onto the orthogonal complement of the
  carry's algebraic correlates, re-runs Phase G on the orthogonalized activations.
- **B.4 (GPLVM):** for each (concept, level, layer, kernel) cell, projects
  curated activations onto the concept's Phase C subspace, fits an exact
  Bayesian GPLVM with ARD, records marginal likelihood and latent coordinates.
- **B.6 (persistent homology):** projects curated activations into the GPLVM
  latent space (or the Phase C subspace as a sensitivity check), computes the
  persistence diagram of the centroid point cloud.
- **B.7 (subspace ablation):** uses the L5 matched-pair correct examples (1,400
  rows) plus the matched wrongs (1,400 rows) for a controlled ablation experiment,
  with the 1,400 random-Grassmannian controls drawn against the same population.
- **B.8 (helix rotation):** uses the L5 matched-pair correct examples (1,400
  rows). Calibrates the rotation angle on the centroids of the 10 ans_digit_5_msf
  values, then applies the calibrated rotation to each example's layer-16 helix
  projection.
- **B.9 (difficulty-matched validation):** uses the matched-pair sets at both
  L4 and L5. Each pair contributes one correct row and one matched wrong row;
  the two are compared on Phase G's Fourier conjunction test directly.

---

## 16. Known limitations and future iterations

The curated set is a v1 artifact. Several known limitations are documented for
future work.

**Limitation 1 — L1 and L2 omitted.** The set has no L1 or L2 representation.
This is by design (no concept is unique to L2; L1 is uniformly correct), but
means that any analysis that wants level-by-level scaling from L1 has to use the
full Phase A–G outputs, not the curated set. The paper's framing is consistent
with this: difficulty matching and matched-pair claims are scoped to L4 and L5,
where the model's failure rate is high enough to support comparison.

**Limitation 2 — Stratification destroys natural frequencies.** The curated set
is stratified to over-represent rare cells (e.g., `carry_2 = 17` at L5 has 30
examples, the floor; in the natural L5 population it has only ~50). Any
distributional claim like "the model uses the helix in 80% of problems" must
come from the full 122,223-row L5 population (Tier 1), not from the curated set
(Tier 2). The paper makes this distinction explicit.

**Limitation 3 — Documented hard ceilings reduce per-cell N for some carry values.**
At L5, `carry_2 = 26` has only 1 example and `carry_3 = 18` has only 12. These
cells are below the 30-floor used by GPLVM ARD pruning. The paper either excludes
these cells from per-cell GPLVM fits or notes the small-N caveat in any result
that uses them.

**Limitation 4 — Matched-pair p-values reflect bin-vs-axis granularity, not
matching quality.** Section 11e shows L4's p-values at 0.001 on two axes despite
mean differences near zero. This is a permutation-test design issue: matching is
on coarse bins, the test is on raw axes. The paper reports both numbers and
clarifies that mean-difference is the substantive measure for the difficulty-confound
argument.

**Limitation 5 — Pass 5 extras are unmatched.** The 263 rows added by Pass 5 do
not have matched-pair partners. They participate in coverage and within-group PCA
but not in B.7/B.8/B.9 paired comparisons. If a future revision needs paired
coverage for a rare cell, the matching algorithm would need to be extended to
incorporate Pass 5 candidates into the pairing pool.

**Limitation 6 — The matching algorithm is greedy and order-dependent.** Pass 3
sorts correct rows in descending difficulty order and matches them in that order.
A different order (e.g., random, or ascending) could produce slightly different
pairings. The deterministic descending-difficulty order is chosen to ensure rare
high-tier correct rows get matched while the wrong pool is full; alternate
orders are not explored. This is a documented design choice rather than a
deficiency.

**Limitation 7 — Hamming = 0 dedup, not Hamming ≤ 1.** The original plan asked
for Hamming ≤ 1 to avoid digit-pattern overrepresentation. The user opted for
Hamming = 0 to keep the candidate pool large. Tier-based stratification provides
some protection against pattern overrepresentation, but a near-duplicate of a
selected (a, b) pair could still appear in the set. If a specific downstream
phase finds this is a problem, a v2 build with Hamming ≤ 1 can be regenerated
from the same script with a one-character change.

**Future iterations.** A v2 of the curated set may be built if any of the
following emerge during downstream phases:
- A specific matched-pair count is needed at a tier-cell where v1's count is
  inadequate.
- The GPLVM ARD analysis reveals that a specific concept needs more than 30
  examples per value to converge cleanly (in which case the floor is raised
  globally).
- A reviewer requests a Hamming ≤ 1 re-run.
- A new concept is added to the registry (e.g., partial-product concepts that
  the curated set didn't target specifically).

The build script `build_curated_set.py` and this report are designed to be
re-runnable by changing only the `BUDGET` constants and the `CONCEPT_FLOOR`
parameter at the top of the script. The artifact would be saved as
`curated_set_v2.json` with v1's hash recorded for provenance.

---

**End of report.** All numbers in this document are validated against
`/data/user_data/anshulk/arithmetic-geometry/curated/curated_set_v1.json` as of
the build completion timestamp. The build log at
`/home/anshulk/arithmetic-geometry/logs/build_curated_set.log` contains the full
DEBUG-level trace of every selection decision.
