# Phase H / B.2: Orthogonalization Control for Carry Helix Superposition — Analysis (complete)

**Anshul's Geometric Manifold Interpretability Project**
**Carnegie Mellon University, May 2026**

This document records every decision, every number, and every result from Phase H —
the orthogonalization control that asks whether the carry helices reported in Phase G
belong to the carries themselves or are inherited from algebraically related concepts
that share residual-stream dimensions with them. It is the truth document for this
stage. Sections 1–7 are the design and pre-registration; Sections 8 onward include
the in-flight notes plus the completed, FDR-corrected result summaries from
`phase_h/summary/orthogonalization_results.csv`.

The run was launched as SLURM job `7666131` on Sun May 3 2026, 01:19:37 EDT, on
`babel-s9-28` (general partition, 24 CPU, 128 GB, 12 h budget). The toy harness ran
twice (interactively at 01:09:33 and again under SLURM at 01:19:35), passed 5/5 both
times, and the real run began at 01:19:37. Four sensitivity branches × 419 carry helix
targets = 1,676 cells total, 1,000 permutations each. The job completed at 02:30:05
EDT with 1,676 successful rows, 0 failures, and 70.5 minutes wall time. The final
branch-level verdict count is `{'inherited': 1676}`.

---

## 1. The question, in plain language

Phase G found 500 helix detections across the residual-stream subspaces of arithmetic
concepts in Llama-3.1-8B. 419 of those — 83.8% — were on `carry_0`..`carry_4`. Phase F
showed that 94% of concept pairs sit in superposition: the principal angle between
their subspaces is far below what a random pair of subspaces of the same dimensions
would give. At L5/layer 16 in particular, the `carry_1` and `col_sum_1` subspaces
overlap at 2.15° — they share over 99.9% of their span.

If two subspaces overlap that tightly, then any geometric structure inside one of them
is mathematically forced to also appear inside the other. So when Phase G reports a
helix in `carry_1`'s subspace, two distinct stories are consistent with the data:

- **Owned.** The model has a representation of the carry value, that representation
  has helical structure, and `carry_1`'s Phase C subspace is the lens we used to look
  at it. The helix really lives in the carry's representation.
- **Inherited.** The model has a representation of `col_sum_1` (the column sum at
  position 1, which carry_1 is algebraically derived from), that representation has
  helical structure, and `carry_1`'s subspace inherits the helix because the two
  subspaces overlap. The model never separately represents the carry; it computes
  it on the fly from the column-sum geometry whenever needed.

These have very different mechanistic implications. "Owned" says Llama allocates
representational machinery to carries as such. "Inherited" says Llama's carry concept
is not a representational primitive but a downstream read-out of something else.

The cheapest defensible test that distinguishes them is: project the activations onto
the orthogonal complement of the correlate subspaces, then re-run the Phase G Fourier
test. If the helix survives, it lives in the carry's own complement and is owned. If
the helix vanishes, it was inherited. Cells in between get a middle verdict and feed
the next-tier methods (B.5 persistent homology, B.6 ablation) for adjudication.

That test is Phase H / Step B.2.

## 2. Why the question is harder than it looks

There are five things that make this test substantially trickier than "subtract a
projection and see what happens":

**(a) Correlates overlap heavily.** carry_1 and col_sum_1 are at 2.15° at L5/layer 16,
so any direction inside the carry_1 subspace is also (with cosine 0.9994) inside the
col_sum_1 subspace. If we naively project away col_sum_1 we will also project away
almost everything in carry_1, and the orthogonalized FCR will be near zero by
construction, regardless of whether the helix was owned or inherited. This is not a
bug in the test — it is the test telling us something true about how tight the
superposition is — but it means we have to be careful about what we read off the FCR
drop alone. The mitigation is twofold: report a power metric alongside the ratio
metric (so we can distinguish "the FCR is meaningless because the amplitude is gone"
from "the FCR is preserved because real signal is still there"); and report a
loss-of-significance criterion (so a cell that drops to noise is classified
`inherited` even if its FCR ratio looks intact post-noise).

**(b) The correlate set is data-driven.** We do not hand-pick which concepts to
project away. Phase B's classified pair table tells us, per (level, population) cell,
which concepts have non-trivial residual correlation with each carry; we filter to
`classification == "structural"` and `|r_resid| ≥ 0.30` (the Phase B
deconfounding threshold) and use exactly that set, with scalar aggregates excluded
because they are 1-d derived quantities with no Phase C subspace. The set is therefore
small, principled, and reproducible. But it is also dependent on Phase B, and any bug
in Phase B's residualization or pair classification propagates downstream. The B.2.6
sensitivity grid (strict `0.50`, loose `0.20`, headline `0.30`, plus a Phase D
merged-basis branch) is designed to catch the cases where the verdict is brittle to
this choice.

**(c) Projector arithmetic is rank-deficient.** When we stack four correlate bases of
roughly 8 to 12 dimensions each and they are mutually overlapping, the stacked basis
has nominal rows ~37 but actual rank often 25–30. A naive matrix inverse would explode
the noise; we have to use rank-truncated QR with a relative tolerance, and we have to
verify per cell that the rank is sane and the projector idempotent. The plan registers
both the tolerance constant (`1e-8`) and the validation thresholds
(`||P_perp² - P_perp||_F < 1e-10`) explicitly.

**(d) The Phase G null was tuned to the full-population N.** Phase G ran on the
122,223-row L5 set with 1,000 permutations, and 458 of 500 helix detections hit the
permutation-floor (`p = 0.001`). Phase H runs on the curated 8,264-problem subset
(L5: 3,017 rows split 1,401 correct + 1,616 wrong; L4: 2,846; L3: 2,401), so the
permutation null is wider per cell. A cell whose Phase G p-value was floor-saturated
on the full data may not be saturated on the curated subset, even if the underlying
geometry is unchanged. This is not a bug — it is the cost of the curated design — but
it means the loss-of-significance criterion ("orthog `q ≥ 0.05`") will fire more often
on the curated run than on the full data, and we have to be honest about that.

**(e) Centering must match Phase G exactly.** Phase G projects with Phase C's
basis after subtracting Phase C's per-cell training mean. If we orthogonalize first
and then re-center with a different mean, the projected coordinates shift and the
centroids shift with them, and the FCR comparison stops being apples-to-apples. The
implementation sidesteps this by computing the orthogonalized projection
algebraically from the raw projection rather than by re-centering: if
`raw_projected = (X − μ) Bᵀ`, then the same centering pushed through the
orthogonalization gives
`orthog_projected = raw_projected − (X Q) (Qᵀ Bᵀ)`, with no mean materialization.
This is exact, not approximate; the bit-identical-raw-FCR gate is what catches any
divergence.

Each of these issues was handled explicitly in the design (sections 4–7 below).
Together they are why the plan has a sensitivity grid, a dual-metric reporting policy,
and a strict `unstable` downgrade rule. The point of B.2 is to give us a principled
verdict per cell, not to just report numbers.

## 3. The state of the world entering this phase

What we already know, with citations:

- **Phase F principal angles, L5/layer 16, `correct` population, carry vs. col_sum:**
  carry_1 ↔ col_sum_1: 2.15°. carry_3 ↔ col_sum_3: 2.57°. carry_2 ↔ col_sum_2: 2.94°.
  carry_4 ↔ col_sum_4: 4.31°. The angles are smaller than for any other concept pair
  in the registry. See `docs/phase_f_jl_analysis.md` Section "principal angles" for
  the full table.
- **Phase G helix counts, total 500:** carry_1 = 176, carry_2 = 116, carry_3 = 54,
  carry_4 = 54, carry_0 = 19, ans_digit = 78, operand_digit = 3 (noise),
  product_binned = 0. By level: L2 = 2, L3 = 93, L4 = 150, L5 = 255. By layer:
  approximately uniform across {4, 6, 8, 12, 16, 20, 24, 28, 31}. 458 of 500 are
  p-floor-saturated at `p = 0.001`. See `phase_g/summary/phase_g_helices.csv`.
- **Phase G null on operand digits: 0/846 detections.** This is the asymmetry that
  makes the carry result interesting in the first place: if Llama always packed
  helical structure into every binned integer that lived in the residual stream, we
  would see helices on `a_units` and `b_units` too. We do not. Whatever produces the
  carry helix is specific to carries (or to col_sums, or to partial products — the
  question B.2 answers).
- **Phase B classified pairs:** 2,677 pair classifications across the 43-name registry.
  The carry-correlate edges that B.2 uses are a small subset. For carry_1 at L3/all:
  `col_sum_1` (r_resid = 0.938), `pp_a0_x_b1` (0.586), `pp_a1_x_b0` (0.584),
  `col_sum_0` (0.805), plus three scalar aggregates excluded by design.
- **Phase D LDA at L5:** 1,026 of 1,035 cells significant. Most carry cells gained
  novel directions beyond Phase C's basis. Phase D's `merged_basis.npy` is the
  sensitivity branch in B.2.6: it tests whether the additional LDA directions soak up
  variance that the Phase C basis misses.
- **Curated set v1:** 8,264 problems with rounded difficulty matching. L3: 1,200
  correct + 1,201 wrong = 2,401. L4: 1,000 correct + 1,846 wrong = 2,846. L5: 1,401
  correct + 1,616 wrong = 3,017. Includes the 2,400 difficulty-matched correct/wrong
  pairs that B.8 will use, so B.2 verdicts attach naturally to the same row set.

What we don't know yet, and what B.2 might or might not tell us:

- Whether each carry helix is owned, inherited, or split between the two.
- Whether the answer is the same across layers and populations (a single global
  verdict would be cleaner, layer-dependent verdicts more biologically realistic).
- Whether the verdict depends on which basis (Phase C vs. Phase D merged) we use to
  define the carry's subspace. If the two bases give different verdicts, the
  representation is somewhere between the two and B.2 alone cannot resolve it.
- How robust the verdict is to the correlate-selection threshold (0.20 vs. 0.30 vs.
  0.50). If the strict and loose branches disagree, we mark the cell `unstable` and
  exclude it from the headline count.

## 4. Pre-registered design

The full design is in `docs/b2_plan.md`. The pre-registration in this document is
intentionally narrower; it locks in only the things we will not change in response to
results.

### 4.1 Targets

419 carry helix rows from `phase_g/summary/phase_g_helices.csv` (concept ∈
{carry_0, carry_1, carry_2, carry_3, carry_4}). Optionally 200 stratified `none`
controls from `phase_g_results.csv` (this run includes them only if `--include-controls`
is passed; the launched run does not pass it, so the headline numbers are based on
the 419 helix targets only).

Per-target inputs are joined to:
- the curated row set for the target's level (3,017 / 2,846 / 2,401 rows for L5 / L4 /
  L3),
- the population mask (`all`, `correct`, `wrong`) read from
  `coloring_dfs/L{L}_coloring.pkl[correct]`,
- the residualized activations at the target's layer
  (`phase_c/residualized/level{L}_layer{ly}.npy`).

### 4.2 Statistic

Two metrics, both reported per cell, both computed by Phase G's `analyze_one`:

- `helix_fcr` — the conjunction-test "Fourier concentration ratio" used by Phase G.
  This is the headline statistic for the verdict rule.
- `total_helix_power` — the un-normalized power on the helix axis. Reported alongside
  the FCR because the FCR is a ratio and can be artifactually preserved (or even
  elevated) when the absolute amplitude is removed.

The drop is computed in three forms:
- absolute `raw − orthog`,
- relative `(raw − orthog) / raw`,
- power relative `(raw_power − orthog_power) / raw_power`.

Significance is via Benjamini-Hochberg FDR over the *Phase H* p-value pool — that is,
all raw and orthog p-values from the 1,676 cells on this run. We do not borrow Phase
G's FDR pool; the two phases are separately corrected.

### 4.3 Verdict rule

Applied per cell, on the headline branch (`threshold = 0.30`,
`correlate_basis = phase_c`):

| Condition | Verdict |
|---|---|
| `helix_fcr_rel < 0.30` | `own_structure` |
| `helix_fcr_rel ≥ 0.30` and `helix_fcr_rel ≤ 0.50` | `ambiguous` |
| `helix_fcr_rel > 0.50` **or** `orthog_helix_q ≥ 0.05` | `inherited` |
| `helix_fcr_rel` is `NaN` (raw `helix_fcr` ≈ 0) | `unclassified` |

Then a sensitivity downgrade:
- if the strict (`0.50`) and loose (`0.20`) branches give a different verdict from the
  headline (`0.30`), the cell is reclassified `unstable` and excluded from the headline
  count. It is reported in the appendix.
- if Phase D merged-basis branch disagrees with Phase C branch, both verdicts are
  reported and the headline takes Phase G's basis-of-detection (414/500 helices in
  Phase G agreed across both bases, so this should affect a minority of cells).

Powerful side note: the loss-of-significance branch is not a redundant restatement of
the FCR-ratio branch. They cover different failure modes. A cell can preserve `helix_fcr`
near 1.0 on a tiny absolute amplitude (post-orthogonalization residual is mostly the
helix coordinate by construction, even though it is at the noise floor) — the
ratio survives but the q-value collapses. The `q ≥ 0.05` clause catches that case;
otherwise we would call it `own_structure` when it is actually `inherited`.

### 4.4 Sensitivity grid

Four branches, all 419 targets, all 1,000 permutations, fixed RNG seed `42`:

- `(threshold = 0.30, correlate_basis = phase_c)` — the headline
- `(threshold = 0.50, correlate_basis = phase_c)` — strict (smaller correlate set)
- `(threshold = 0.20, correlate_basis = phase_c)` — loose (larger correlate set)
- `(threshold = 0.30, correlate_basis = phase_d_merged)` — Phase D LDA basis

The choice to make the headline use Phase C's basis matches Phase G's primary
detection basis (Phase G ran both, but the carry helix agreement story rests
predominantly on Phase C). Phase D is the sensitivity check that asks whether the LDA
directions add or subtract correlate variance.

### 4.5 Acceptance gates

The run is gated on five checks; failing any one halts publication of the headline:

1. Toy harness 5/5 pass.
2. Bit-identical raw FCR for at least one cell (carry_1 / L5 / layer 16 / correct /
   phase_c / carry_raw is the canonical check) against `phase_g_helices.csv`. If the
   raw branch does not reproduce Phase G to within float32 epsilon, then the curated
   row filter or the projection mean is being applied differently from Phase G; fix
   before publishing the orthog branch.
3. Per-cell projector idempotency `||P_perp² − P_perp||_F < 1e-10`.
4. Per-cell null-space residual `||B_correlates · P_perp||_F / ||B_correlates||_F < 1e-6`.
5. Cross-method hand-off: every row in Phase H summary CSV joins to a row in
   `phase_g_helices.csv` on `(concept, level, layer, population, subspace_type,
   period_spec)` keys. (The 419 helix targets define the cell list, so this should
   hold trivially; failure indicates a bug in target loading.)

### 4.6 What the verdicts mean for the paper

Three publishable framings, decided in advance:

- If most cells come back `own_structure`, Phase G's headline survives. The carry
  helix is a property of how Llama encodes the carry value, not a shadow of col_sum
  or partial-product geometry. The framing for the paper would be conservative, but
  the methodological novelty (running an orthogonalization control on production-LLM
  internals) is what makes the contribution in this case.
- If most cells come back `inherited`, the framing flips. The model does not
  represent the carry as such; the helix is an emergent property of the column-sum
  / partial-product computation and we read it off the carry subspace because of
  superposition. This is the more scientifically interesting outcome, because it
  reveals a fact about how Llama organizes arithmetic that no prior paper in the
  literature established for production LLMs.
- If the verdicts split — say carry_0 is `inherited` while carry_3 is `own_structure`
  — that is also a substantive finding: it would mean the model owns harder carries
  but inherits easier ones, or vice versa, and the carry-difficulty axis is a
  representational dimension we did not anticipate.

All three framings are pre-registered. We are not adjusting the headline choice
post-hoc.

## 5. Implementation cross-reference

The script is [phase_h_orthogonalize.py](../../../arithmetic-geometry/phase_h_orthogonalize.py).
Single file, follows `phase_g_fourier.py` convention, imports rather than copies
the Phase G machinery. The SLURM wrapper is
[run_phase_h.sh](../../../arithmetic-geometry/run_phase_h.sh), CPU-only, `general`
partition, 12 h budget. Both pass `python -m py_compile` and `bash -n` cleanly.

The pre-registered four-branch grid is at
[phase_h_orthogonalize.py:51-56](../../../arithmetic-geometry/phase_h_orthogonalize.py#L51-L56).
The verdict rule is at
[phase_h_orthogonalize.py:593-609](../../../arithmetic-geometry/phase_h_orthogonalize.py#L593-L609).
The QR projector is at
[phase_h_orthogonalize.py:296-339](../../../arithmetic-geometry/phase_h_orthogonalize.py#L296-L339).
The trick that preserves Phase C centering exactly is at
[phase_h_orthogonalize.py:349-360](../../../arithmetic-geometry/phase_h_orthogonalize.py#L349-L360):

```
def project_orthogonalized_from_raw(raw_projected, x_full, q_rank, target_basis):
    if q_rank.shape[1] == 0:
        return raw_projected.copy()
    return raw_projected - (x_full @ q_rank) @ (q_rank.T @ target_basis.T)
```

The algebra: if Phase C stores its projected coordinates as
`raw_projected = (X − μ) Bᵀ`, then for the orthogonalized activations
`X_orth = X − X Q Qᵀ`, the projected coordinates are
`(X_orth − μ) Bᵀ = (X − μ) Bᵀ − X Q Qᵀ Bᵀ = raw_projected − X Q Qᵀ Bᵀ`.
The mean μ never has to be loaded explicitly because it cancels through; we only need
the raw projection and the original activations. This guarantees that the raw branch
of `analyze_one` reproduces Phase G's number bit-for-bit (gate #2 in §4.5), and the
orthog branch differs from it by exactly the orthogonalization step, with no
double-centering or other off-by-something errors.

The four-branch grid is iterated inner-most in `run_all`
([phase_h_orthogonalize.py:884-981](../../../arithmetic-geometry/phase_h_orthogonalize.py#L884-L981)),
so the log walks targets in `(target_kind, level, layer, population, concept,
subspace_type, period_spec)` order and prints four `B.2 cell:` lines per target. This
is the order the in-flight log lines below appear.

## 6. Toy validation results

Two toy runs were done; both passed 5/5. The runs are recorded in
`logs/phase_h_orthogonalize.log` at `01:09:33` (interactive, before SLURM submission)
and at `01:19:35` (inside the SLURM job, before the real run started).

Numbers from the SLURM-job toy run:

| Toy | Construction | Raw helix FCR | Orthog helix FCR | FCR drop | Power drop | Validation | Verdict |
|---|---|---|---|---|---|---|---|
| **T1 — own helix** | helix in carry-only complement, correlates orthogonal | 0.603 | 0.603 | 0.000 | 0.000 | nullspace 0 | PASS |
| **T2 — inherited helix** | helix lies entirely in correlate subspace | 0.603 | 0.000 | 1.000 | 1.000 | nullspace 0 | PASS |
| **T3 — split power** | helix amplitude split 50/50 between carry and correlate | 0.349 | 0.602 | −0.726 | 0.467 | nullspace 0 | PASS |
| **T4 — bleed-through** | null carry, real correlate helix bleeds in | 1.000 | 0.498 | 0.502 | 1.000 | nullspace 0 | PASS |
| **T5 — projector null-space** | random correlate basis, identity acts | n/a | n/a | n/a | n/a | residual 4.812e-16, q-orth 2.184e-15 | PASS |

Three details that matter:

- **T3 is the FCR-vs-power split case the design anticipates.** When half the helix
  amplitude lives in the carry-only complement, projecting away the correlate-shared
  half *removes* the contaminating signal and the residual helix has higher FCR than
  the raw mixed signal — `−72.6%` "drop" in the table is just `−`(0.602 − 0.349)/0.349.
  The power axis is the honest one here: it dropped 47%, exactly half. If the verdict
  were FCR-only, T3 would be classified `own_structure` (drop < 30%); with the power
  axis reported alongside it, the reader can see the amplitude story too. This is the
  case that motivated adding `helix_power_rel` to the output schema; the original
  plan listed only `helix_fcr_rel`.
- **T2 and T4 both look like `inherited` in the FCR axis (drop > 0.50) and the power
  axis (drop > 0.95).** They are the cleanest verdict cases.
- **T5's residual norm is at machine epsilon** (`4.8e-16`), well below the `1e-6`
  gate. The QR construction is numerically clean for the synthetic case; per-cell
  validation in §8 will tell us whether real Phase B / Phase C bases stay this clean.

The toy harness is not a test of the test — it cannot tell us whether the Phase B
correlate set is the right correlate set, or whether the Phase C basis is the right
basis. It is a test of the *projector arithmetic and FCR-comparison logic*: given a
correlate basis we know is correct, do we get the verdicts we expect for the four
ground-truth cases (own / inherited / split / bleed-through). 5/5 says the math is
implemented correctly. The harder question — does B.2 give the right *scientific*
verdict on real data — is what the rest of this document will address.

## 7. Run history and progress as of writing

This section records the completed run.

- **Sun May 3 2026, 01:09:33 EDT** — interactive toy validation run. 5/5 PASS.
- **Sun May 3 2026, 01:19:35 EDT** — SLURM job 7666131 starts on `babel-s9-28`,
  general partition, 24 CPU, 128 GB. Toy harness re-runs inside the job. 5/5 PASS.
- **Sun May 3 2026, 01:19:37 EDT** — real run begins. Loaded 419 helix targets;
  4 sensitivity branches (`(0.30, phase_c)`, `(0.50, phase_c)`, `(0.20, phase_c)`,
  `(0.30, phase_d_merged)`); 1,000 permutations per branch.
- **Sun May 3 2026, 01:31:59 EDT** — 12 minutes in. 212 cells started, ~13% of the
  1,676-cell budget. Pace ≈ 1 cell every 3.4 s with 1,000 permutations included.
  Linear extrapolation: total wall time ≈ 95 minutes. Well within the 12 h budget.
- **Sun May 3 2026, 02:30:05 EDT** — run complete. Summary CSV saved at
  `/data/user_data/anshulk/arithmetic-geometry/phase_h/summary/orthogonalization_results.csv`
  with 1,676 rows and 83 columns. Correlate audit saved at
  `/data/user_data/anshulk/arithmetic-geometry/phase_h/summary/correlate_sets.json`.
  Runtime was 70.5 minutes. Failure count was 0.
- **Final branch-level verdict count:** 1,676 `inherited`, 0 `own_structure`,
  0 `ambiguous`, 0 `unclassified`, 0 failed rows. This is the literal `verdict`
  column in the completed CSV after the FDR pass.

## 8. Intermediate per-cell observations (in-flight, not the final verdict)

The summary CSV and per-cell JSONs are written only after all 1,676 cells finish and
the FDR pass runs. So nothing in this section is a verdict — these are the raw and
orthog FCR / power numbers that `analyze_one` is printing to the log per cell. They
are useful as a sanity check on the math and as an early read on the direction of the
result, but the "inherited" / "own_structure" / "ambiguous" labels here are
provisional and may shift after FDR.

The table format below is one row per `(concept, level, layer, population, subspace_type,
period_spec, branch)` tuple, with raw and orthog FCR and power side by side and the
provisional verdict assuming the FDR loss-of-significance test is benign.

### 8.1 carry_0 / L3 / layer 4 / all / phase_d_merged / carry_raw

(Raw `helix_fcr = 0.326`, raw `total_helix_power = 0.705` — same across all four
branches because the raw signal does not depend on which correlate set we project
against.)

| Branch | Orthog `helix_fcr` | Orthog `total_helix_power` | FCR drop | Power drop | Provisional |
|---|---|---|---|---|---|
| (0.30, phase_c) | 0.361 | 0.000152 | −0.107 | 0.99978 | inherited (power) |
| (0.50, phase_c) | 0.278 | 0.000240 | 0.147 | 0.99966 | inherited (power) |
| (0.20, phase_c) | 0.374 | 0.000123 | −0.147 | 0.99983 | inherited (power) |
| (0.30, phase_d_merged) | 0.317 | 0.000130 | 0.028 | 0.99982 | inherited (power) |

Reading: the FCR ratio looks preserved (sometimes elevated above raw), the power axis
shows the helix amplitude has been almost entirely removed, four orders of magnitude.
This is the toy-T3-meets-real-data case: the FCR is meaningless when the numerator is
this small relative to the noise floor. The post-FDR loss-of-significance branch
should fire and classify this `inherited`. Power-axis verdict already says the same
thing.

### 8.2 carry_1 / L3 / layer 4 / all / phase_c / carry_raw

(Raw `helix_fcr = 0.548`, raw `total_helix_power = 2.774`.)

| Branch | Orthog `helix_fcr` | Orthog `total_helix_power` | FCR drop | Power drop | Provisional |
|---|---|---|---|---|---|
| (0.30, phase_c) | 0.169 | 0.0041 | 0.692 | 0.99852 | inherited |
| (0.50, phase_c) | 0.169 | 0.0041 | 0.692 | 0.99852 | inherited |
| (0.20, phase_c) | 0.171 | 0.0030 | 0.688 | 0.99892 | inherited |
| (0.30, phase_d_merged) | 0.164 | 0.0047 | 0.701 | 0.99830 | inherited |

Reading: this is the cleanest "inherited" pattern. FCR drop > 0.50 in all four
branches; power drop > 0.998 in all four; sensitivity branches all agree. Verdict
robust to threshold and basis choice.

### 8.3 carry_1 / L3 / layer 4 / all / phase_d_merged / carry_binned

(Raw `helix_fcr = 0.444`, raw `total_helix_power = 2.335`. The binned spec has period
P=15 instead of P=18 because Phase C's `dim_consensus` for carry_1 at L3 is 14.)

| Branch | Orthog `helix_fcr` | Orthog `total_helix_power` | FCR drop | Power drop | Provisional |
|---|---|---|---|---|---|
| (0.30, phase_c) | 0.140 | 0.0034 | 0.685 | 0.99854 | inherited |
| (0.50, phase_c) | 0.140 | 0.0034 | 0.685 | 0.99854 | inherited |
| (0.20, phase_c) | 0.142 | 0.0025 | 0.680 | 0.99893 | inherited |
| (0.30, phase_d_merged) | 0.141 | 0.0036 | 0.683 | 0.99846 | inherited |

Same shape, same verdict. This one matters because it cross-validates: Phase C basis
+ `carry_raw` spec (8.2) and Phase D merged basis + `carry_binned` spec (8.3) are
algebraically distinct probes of the same underlying carry_1 cell, and they agree.

### 8.4 carry_1 / L3 / layer 4 / all / phase_d_merged / carry_raw

(Raw `helix_fcr = 0.390`, raw `total_helix_power = 2.919`. Period 18 with binned
groups dropped.)

| Branch | Orthog `helix_fcr` | Orthog `total_helix_power` | FCR drop | Power drop | Provisional |
|---|---|---|---|---|---|
| (0.30, phase_c) | 0.102 | 0.00449 | 0.738 | 0.99846 | inherited |
| (0.50, phase_c) | (filling in) | (filling in) | | | |
| (0.20, phase_c) | (filling in) | | | | |
| (0.30, phase_d_merged) | (filling in) | | | | |

(Will be filled in after the run completes — the cells are still streaming.)

### 8.5 Cross-check: raw FCR is invariant across branches per target

A key sanity check: the raw `helix_fcr` value for a given target cell should be
identical across the four branches because the raw projection does not depend on the
correlate set. The log shows this is the case to 4 decimal places of precision in
every cell so far:
- carry_0 / L3 / layer 4 / all / phase_d_merged / carry_raw: raw FCR = 0.3264 across
  all four branches.
- carry_1 / L3 / layer 4 / all / phase_c / carry_raw: raw FCR = 0.5484 across all four
  branches.
- carry_1 / L3 / layer 4 / all / phase_d_merged / carry_binned: raw FCR = 0.4437
  across all four branches.

This is gate #2 from §4.5 (bit-identical raw match). The full bit-identical match
against `phase_g_helices.csv` will be verified in §10 once we can inspect the per-cell
JSON output, but the cross-branch invariance is the necessary condition: if the raw
FCR were drifting between branches, the sensitivity comparison would be incoherent.

### 8.6 Plain-language status snapshot (early read, ~25% of run)

What we can say in plain language as of Sun May 3 2026 ~01:42 EDT, 442 cells started
of 1,676:

- The run is healthy. Toy harness 5/5 PASS, twice. No exceptions on real cells. SLURM
  job stable on the general partition (no preemption like Phase G hit). Pace is
  ~1 cell per 3.4 s; full run projects to ~95 minutes wall.
- For the first L3 cells we have looked at, the carry helix mostly disappears after
  projecting away the structurally related concepts (col_sum_1, col_sum_0, partial
  products). carry_1 / L3 / layer 4 / all is the cleanest example: helix_fcr drops
  from 0.548 to 0.169, total helix power drops from 2.774 to 0.004. Both the shape
  metric and the amplitude metric agree: the helix structure was not in the
  carry-only complement of the correlate subspaces.
- For carry_0 / L3 / layer 4 / all, the picture is more confusing on the FCR axis
  alone (FCR sometimes goes up after orthogonalization), but the power axis is
  unambiguous (drop > 99.97%). This is the case the dual-metric reporting was
  designed for.
- These early reads are consistent with the `inherited` hypothesis at L3, but L3 is
  not the headline. The headline cell is L5/layer 16, where Phase F measured the
  tightest carry-vs-col_sum angle (2.15° for carry_1, 2.57°–4.31° for carry_2..4).
  L5/layer 16 has not been reached in the log yet (current pointer: L4/layer 8 /
  carry_1).
- Final verdicts cannot be called yet. They land only after all 1,676 cells finish
  and the FDR pass runs across the full Phase H p-value pool. Anything tagged
  "provisional inherited" in §8.1–§8.4 is the power-axis read, not the official
  verdict.

The narrative through-line: Phase G found the helices, Phase F flagged the
superposition that meant the helices might not belong to the carries, B.2 is the
test that decides — and the test is currently producing what looks like the more
scientifically interesting outcome (`inherited`) for the easy L3 cells. Whether the
same picture holds at the L5/layer-16 headline cell is the question the rest of the
run is answering.

### 8.7 The FCR-vs-power policy, written out longhand

Two separate questions, two separate metrics. Both reported per cell. Verdict rule
unchanged from pre-registration; interpretation policy expanded as below.

**FCR (`helix_fcr_rel`).** Asks: of whatever signal remains after projection, how
helix-shaped is it? FCR is a ratio: numerator is the squared-amplitude on the helix
axis, denominator is total squared-amplitude. Insensitive to scaling. Robust against
"the signal got smaller" if the residual is still proportionally helix-shaped. Fails
when the residual is at the noise floor and the ratio is just measuring the shape of
noise.

**Power (`helix_power_rel`).** Asks: how much actual signal is left? `total_helix_power`
is un-normalized. Sensitive to scaling. Goes to zero when the signal is removed,
regardless of what the residual noise looks like. Fails when there is genuinely a
small amplitude signal that survives projection — power says "small" even if FCR
says "high-quality."

**Together, they cover four cases:**

| FCR drop | Power drop | Reading |
|---|---|---|
| Small | Small | Helix is in carry's own complement. `own_structure` (provisional). |
| Large | Large | Helix is in the correlate subspaces. `inherited`. |
| Small | Large | Amplitude collapsed but residual noise happens to be helix-shaped. `inherited` (caught by FDR / power-collapse diagnostic). |
| Large | Small | Amplitude preserved but the residual is no longer helix-shaped. Rare; would suggest the projection moved structure to a different Fourier basis. Flag for inspection. |

**Pre-registered verdict rule, unchanged.** The thresholds are 0.30 (own/ambiguous
boundary) and 0.50 (ambiguous/inherited boundary), and a loss-of-FDR-significance
trigger that overrides the FCR rule. These thresholds are not adjusted from this
run. They were in `docs/b2_plan.md` before the run launched.

**Interpretation policy, added now.** When reading the final results CSV:
- The verdict column is the headline. Trust the pre-registered rule.
- The `helix_power_rel` column is the diagnostic. Inspect any cell where
  `helix_fcr_rel < 0.30` (would-be `own_structure`) but `helix_power_rel > 0.90`
  (amplitude collapsed). Those are the "amplitude-collapsed" cells. They should be
  classified `inherited` by the FDR loss-of-significance trigger; if any escape that
  trigger and end up `own_structure`, list them in the appendix as "amplitude-collapsed,
  FDR-survived" and treat them as ambiguous in B.9 cross-method discussion.
- The mirror case — `helix_fcr_rel > 0.50` with `helix_power_rel < 0.30` — would mean
  the FCR test is flagging a shape change without a substantial amplitude change. So
  far we have not seen this; if it appears, flag for inspection rather than calling
  it `inherited` automatically.

**Why the policy matters.** Without it, the dual reporting is just two columns side
by side and the reader has to do their own integration. With it, the reader has a
documented rule for handling the corner cases that the pre-registered FCR rule alone
does not handle, and there is no after-the-fact interpretation drift if a cell
surprises us.

### 8.8 Curated coverage warnings

Multiple cells trigger the "curated subset missing N value groups for `carry_1/carry_raw`"
warning. Examples from the log:
- carry_1 / L3 / layer 4 / all / carry_raw: missing values [15, 16, 17] (period 18,
  so 3 of 18 groups have zero examples in the curated subset).
- carry_1 / L3 / layer 4 / correct / carry_raw: missing values [12, 13, 14, 15, 16,
  17] (correct-only filter is much smaller; 6 of 18 groups empty).
- carry_1 / L3 / layer 4 / wrong / carry_raw: missing [15, 16, 17].

The script handles this by retaining the original period (18) and dropping the
empty-value-group rows from the centroid set
([phase_h_orthogonalize.py:679-698](../../../arithmetic-geometry/phase_h_orthogonalize.py#L679-L698)).
The reduced centroid count `m = 12` (for the L3/correct/carry_raw cell) is logged
explicitly. `len(unique_values) < 3` would raise; this has not happened in any cell so
far.

This warning is structurally inevitable for `carry_raw` at L3 because the L3
multiplication arithmetic makes carry_1 ≥ 15 extremely rare (it requires very
specific operand combinations); the curated set's source-index sampling did not
guarantee uniform value coverage. The `carry_binned` spec (P=15) sidesteps this
entirely by collapsing the long tail into a single bin. For the L3 cells, the
`carry_binned` branch is therefore more in-distribution and the verdict is more
trustworthy.

## 9. Key statistical considerations

### 9.1 Why we can read the power drop without an explicit statistical test

The plan does not register a permutation test on `total_helix_power` itself; the
permutation null is on the FCR. So at first glance it might look unprincipled to use
the power drop as an alternate verdict criterion. The reasoning is that power and FCR
are mathematically linked: when power drops by orders of magnitude, the residual FCR
is computed on a denominator that is approximately the noise floor, and the
permutation distribution of FCR on noise is centered near `1/d`. A high FCR ratio on
an essentially zero-amplitude signal is therefore not significant — it is the
permutation test that catches that, and the q-value will be ≥ 0.05 for those cells.

In other words, the loss-of-significance criterion is the principled enforcement of
the power-drop intuition. We use the power-drop column for early reading and for
diagnostic reporting; we use the q-value for the verdict.

### 9.2 Why we use Phase H's own FDR pool, not Phase G's

Phase G corrected its 3,480-test pool. Phase H corrects its 1,676-cell × 4-test
(raw two_axis, raw helix, orthog two_axis, orthog helix) pool = 13,408 tests
(though many will saturate). If we borrowed Phase G's q-values for the raw branch
and computed Phase H q-values for the orthog branch, the two would not be on the
same FDR scale and the loss-of-significance verdict would be incoherent. By
correcting both inside Phase H, every cell's raw and orthog q-values are directly
comparable.

### 9.3 The Phase F angle as a back-of-envelope upper bound

Phase F gives carry_1 ↔ col_sum_1 angle 2.15° at L5/layer 16. cos²(2.15°) ≈ 0.9986.
That means roughly 0.14% of carry_1's variance lies in the col_sum_1-orthogonal
complement. So projecting away col_sum_1 from carry_1 should remove ~99.86% of any
signal that happens to align with col_sum_1. If carry_1's helix is owned, it lives
in that 0.14% complement and survives projection. If carry_1's helix is inherited,
it lives in the col_sum_1-aligned 99.86% and disappears.

The intermediate observations in §8.2 say: at L3/layer 4, the helix lies in the
~99.85% removed by projection. That is consistent with the `inherited` interpretation.
But L3/layer 4 is not the headline cell. The headline cell is L5/layer 16, where
Phase F gives that 2.15° angle, and where Phase G gave the highest helix FCR for
carry_1. We need to wait for that cell in the log before drawing the headline
verdict.

### 9.4 The wider-null artifact concern, quantitatively

Phase G ran with N ≈ 122,223 at L5; Phase H runs with N ≈ 3,017 at L5 / `all`. The
permutation null for the FCR test scales roughly as `1 / sqrt(N)` for the standard
deviation of the null distribution (the FCR statistic is approximately a sum of
chi-square ratios). So Phase H's null at L5 is roughly `sqrt(122223 / 3017) ≈ 6.4×`
wider than Phase G's. Cells whose raw `helix_fcr` was 0.55 with Phase G's tight null
(`p < 0.001`) might come back at Phase H raw `helix_fcr ≈ 0.55` but with a much
wider null and `p ≈ 0.005` instead of `< 0.001`. This is fine for the *raw* branch
because we still expect raw to clear FDR by a wide margin. But for the *orthog*
branch, where the signal might be small to begin with, the wider null could push
borderline cells over the q ≥ 0.05 line that they would have cleared on the full
data.

This is why §4.6 frames the decision criteria with a sensitivity grid rather than a
single threshold: if the orthog branch is borderline, the strict `0.50` threshold
(smaller correlate set, less collateral signal removed) should give an FCR drop
closer to zero, and the loose `0.20` threshold (larger correlate set, more removed)
should give a drop closer to 100%. If both branches give the same verdict as the
headline, the verdict is robust to the wider null. If they disagree, the cell is
`unstable` and excluded.

## 10. Validation diagnostics

The completed CSV has 1,676 rows and 83 columns. Each row records the QR rank,
null-space residual, orthonormality residual, idempotency residual, and missing-basis
list for that branch.

### 10.1 Headline branch diagnostics

Headline branch means `correlate_threshold = 0.30` and `correlate_basis = phase_c`.
This is the branch used for the main B.2 verdict table.

| Diagnostic | Min | Median | Max | Gate status |
|---|---:|---:|---:|---|
| `validation_residual_norm` | 3.37e-16 | 6.08e-16 | 2.74e-15 | all 419 < 1e-6 |
| `validation_nullspace_norm` | 3.37e-16 | 6.08e-16 | 2.74e-15 | all 419 < 1e-6 |
| `validation_q_orthonormality_norm` | 4.58e-16 | 3.27e-15 | 4.74e-15 | all 419 < 1e-10 |
| `validation_idempotent_norm` | 4.58e-16 | 3.27e-15 | 4.74e-15 | all 419 < 1e-10 |

Missing correlate bases in the headline branch: 0 / 419.

QR rank in the headline branch:

| Quantity | Min | P10 | P25 | Median | P75 | P90 | Max | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `d_correlates_nominal` | 7 | 16 | 71 | 86 | 105 | 117 | 143 | 82.29 |
| `d_correlates_rank` | 7 | 16 | 71 | 86 | 105 | 117 | 143 | 82.29 |

For the headline branch, rank equals nominal in every row. The projector diagnostics
are at machine precision. This means the Phase C headline branch did the intended
linear algebra: it projected against the correlate span and left no measurable
correlate component behind.

### 10.2 Phase D sensitivity diagnostics

The Phase D merged-basis sensitivity branch is different. In that branch,
293 / 419 rows have `d_correlates_rank != d_correlates_nominal`, and 285 / 419 rows
have `validation_residual_norm >= 1e-6`. These rows are all in the
`correlate_basis = phase_d_merged` sensitivity branch.

This does not affect the headline Phase C result. It means the Phase D merged bases
contain redundant LDA-augmented directions that QR truncates. The projector is still
orthonormal and idempotent after truncation, but the diagnostic measured against the
full nominal stacked Phase D basis is no longer near zero because discarded redundant
directions are included in that denominator.

Phase D branch rank summary:

| Quantity | Count | Mean | Std | Min | P25 | Median | P75 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `d_correlates_nominal` | 419 | 185.27 | 77.25 | 16 | 162 | 191 | 236 | 310 |
| `d_correlates_rank` | 419 | 181.89 | 75.65 | 16 | 156 | 191 | 232 | 304 |

The important reporting choice is therefore:

- Headline claims use the Phase C branch, whose projector diagnostics pass all gates.
- Phase D is kept as a sensitivity branch and reported literally.
- No Phase D sensitivity number is used to rescue or override the Phase C headline.

### 10.3 Missing-basis and failure diagnostics

Missing correlate bases:

| Scope | Missing-basis rows |
|---|---:|
| All 1,676 rows | 0 |
| Headline branch | 0 |

Failures:

| Scope | Successful rows | Failed rows |
|---|---:|---:|
| Full Phase H run | 1,676 | 0 |

No `orthogonalization_failures.json` was produced.

## 11. Headline verdicts

The headline branch is `threshold = 0.30`, `correlate_basis = phase_c`. It has
419 rows: one for every carry helix target imported from `phase_g_helices.csv`.

### 11.1 Overall result

| Verdict | Count | Fraction |
|---|---:|---:|
| `inherited` | 419 | 100.0% |
| `own_structure` | 0 | 0.0% |
| `ambiguous` | 0 | 0.0% |
| `unclassified` | 0 | 0.0% |

All four branches give the same branch-level verdict count:

| Branch | Rows | `inherited` |
|---|---:|---:|
| threshold 0.20 / Phase C correlates | 419 | 419 |
| threshold 0.30 / Phase C correlates | 419 | 419 |
| threshold 0.50 / Phase C correlates | 419 | 419 |
| threshold 0.30 / Phase D merged correlates | 419 | 419 |

### 11.2 By concept

| Concept | Rows | `inherited` | `own_structure` | `ambiguous` |
|---|---:|---:|---:|---:|
| `carry_0` | 19 | 19 | 0 | 0 |
| `carry_1` | 176 | 176 | 0 | 0 |
| `carry_2` | 116 | 116 | 0 | 0 |
| `carry_3` | 54 | 54 | 0 | 0 |
| `carry_4` | 54 | 54 | 0 | 0 |

### 11.3 By level

| Level | Rows | `inherited` |
|---|---:|---:|
| L3 | 69 | 69 |
| L4 | 134 | 134 |
| L5 | 216 | 216 |

No L2 rows appear in Phase H because the carry helix target list imported from Phase G
for this run contains L3-L5 carry targets.

### 11.4 By layer

| Layer | Rows | `inherited` |
|---|---:|---:|
| 4 | 48 | 48 |
| 6 | 47 | 47 |
| 8 | 47 | 47 |
| 12 | 47 | 47 |
| 16 | 47 | 47 |
| 20 | 47 | 47 |
| 24 | 44 | 44 |
| 28 | 46 | 46 |
| 31 | 46 | 46 |

### 11.5 By population

| Population | Rows | `inherited` |
|---|---:|---:|
| `all` | 137 | 137 |
| `correct` | 146 | 146 |
| `wrong` | 136 | 136 |

### 11.6 What exactly caused the verdicts

The final verdict rule allows `inherited` through either a large FCR drop or
loss of orthogonalized helix significance after FDR. The headline branch has:

| Statistic | Count |
|---|---:|
| Raw helix q < 0.05 | 237 / 419 |
| Orthogonalized helix q < 0.05 | 0 / 419 |
| Raw two-axis q < 0.05 | 221 / 419 |
| Orthogonalized two-axis q < 0.05 | 0 / 419 |
| Orthogonalized `geometry_detected = helix` | 0 / 419 |
| Orthogonalized `geometry_detected = circle` | 0 / 419 |
| Orthogonalized `geometry_detected = none` | 419 / 419 |

This is the most direct final result: after projecting away the selected correlate
subspaces, no carry cell remains helix-significant or circle-significant under the
Phase H FDR pool.

### 11.7 Raw curated replay caveat

The raw branch in Phase H is not identical to Phase G's full-population run. It uses
the same Fourier code, but only on the curated 8,264-problem subset. On the headline
branch, raw curated `geometry_detected` is:

| Raw curated geometry | Rows |
|---|---:|
| `helix` | 24 |
| `circle` | 2 |
| `none` | 393 |

So the careful statement is not "419 strong raw curated helices vanished." The careful
statement is:

> Phase G supplied 419 full-population carry helix targets. On the curated subset, the
> raw Fourier FCR remains highly correlated with the full-population FCR, but the
> stricter conjunction verdict reproduces as `helix` for only 24 of those rows and
> as `circle` for 2 more. After correlate orthogonalization, 0 / 419 rows retain
> helix or circle significance.

This distinction matters for writing. Phase H is a superposition control on the
curated set, not a second full-population Phase G run.

## 12. Critical perspective and caveats

This section is the devil's-advocate read on what B.2 can and cannot show, written
before the results to keep us honest.

### 12.1 What B.2 cannot rule out

- **Unmeasured confounders.** B.2 only projects away concepts that are in the 43-name
  registry. If Llama has an internal representation of, say, "the sum of all carries"
  or "the magnitude of the larger operand" that has helical structure and is in
  superposition with the carry subspaces, B.2 will not catch it. The helix would look
  `own_structure` in our test even though it is inherited from this unmeasured
  concept. Mitigation: B.2 is explicit about being a "known-correlates only" control;
  the broader claim ("the carry helix is owned") requires the SAE-based Paper 4
  decomposition to handle unmeasured features, and we do not make the broader claim
  on B.2 alone.
- **Causal direction.** A cell that comes back `own_structure` does not prove the
  model uses the carry helix in any computation. It proves the helix lives in carry's
  own complement, but that complement is still a downstream readout of the column-sum
  computation; the helix could be functionally vestigial. Mitigation: B.6 (causal
  ablation) is the gold standard for this question; B.2 is upstream of it. A B.2
  `own_structure` verdict licenses asking the causal question; an `inherited` verdict
  closes the question.
- **Phase-G-specific artifacts.** B.2 shares Phase G's `analyze_one` machinery, so
  any bug in Phase G's permutation null or FDR procedure also affects B.2. Phase G
  was independently validated against synthetic toys and against K&T's published
  result (see `docs/phase_g_analysis.md` Section "K&T pilot"); this is mitigation but
  not proof.
- **Curated-set selection bias.** The curated set was sampled by `source_index` from
  the difficulty-matched correct/wrong pairs. If the matching procedure systematically
  biases the carry-value distribution (e.g. biases toward small carries, where the
  helix shape is different), the verdict could differ from what we would have got on
  the full population. Mitigation: §8.6 documents the missing-value-group warnings
  per cell; the headline carry_1 / L5 / layer 16 cell is the most-covered and least
  biased. A confirmatory follow-up on the full data for the headline cell only is
  planned if any verdict comes back ambiguous (cheap; one cell × one branch ≈ 30 s).

### 12.2 What an `inherited` verdict does not mean

If B.2 returns `inherited` for most carries, it would be tempting to say "Llama does
not represent carries." That is too strong a conclusion. What it *would* mean:

- The carry's helical structure is not a property of the carry-only complement of the
  correlate subspaces. The structure lives in the shared dimensions.
- The model's internal computation of the carry value is geometrically inseparable
  from the column-sum / partial-product computation at this layer.
- Reading off the carry value via a linear probe on the carry's Phase C basis works
  *because* of superposition, not because there is an independent carry channel.

What it *would not* mean:

- That Llama does not use carries downstream. It might still ablate-cleanly when
  patched.
- That carries are not represented at all. They are represented; the representation
  is just not separate from col_sum.
- That the Phase G headline is wrong. The helix exists; the question of where it
  *belongs* is the new finding.

### 12.3 What an `own_structure` verdict does not mean

Similarly, if most cells come back `own_structure`:

- It would not mean Llama has a dedicated "carry circuit." It would mean the carry's
  Phase C subspace contains structure beyond what the correlate subspaces can
  account for.
- It would not mean Phase F's 2.15° angle is wrong. The angle is what it is; the
  structure beyond the angle is what we report.
- It would not rule out that some other concept (say, a partial product we do not
  project against because its `r_resid` is below 0.30) carries the helix and bleeds
  in. The 0.20 sensitivity branch is the test for that; if it gives `inherited`
  while 0.30 gives `own_structure`, the cell is `unstable`.

### 12.4 Limits of the FCR / power dual reporting

The FCR-and-power dual is robust against the "high FCR on tiny amplitude" failure mode
(toy T3) and against the "amplitude removed but ratio elevated" pattern in §8.1. It is
*not* robust against:

- A cell where the helix is genuinely split, with most amplitude in the carry
  complement and a small but real amplitude in the correlate-shared dimensions. The
  FCR ratio survives; the power ratio drops modestly; the q-value is preserved.
  Verdict would be `own_structure` and that is correct, but the appendix should still
  report the partial inheritance.
- A cell where the helix is very weak to begin with (raw `helix_fcr` near
  Phase G's significance threshold of about 0.30). If raw is at 0.30 and orthog is at
  0.20, the relative drop is 33% (`ambiguous`), but with the wider Phase H null both
  raw and orthog might fail FDR, in which case the verdict becomes `unclassified`. This
  is a low-power cell and will be flagged in the appendix as such.

### 12.5 Pre-mortem on the worst plausible failure modes

- **Most cells `unstable`.** If the strict and loose branches frequently disagree
  with the headline, we cannot make a sensitivity-stable claim and the headline-cell
  count drops sharply. This would mean the verdict is fragile to correlate-set
  composition, which is itself a real finding (the model's carry representation is
  not cleanly separable from the correlate set under any threshold). The framing
  would shift from "owned vs. inherited" to "the boundary is not crisp," and we would
  fall back on B.5/B.6 for the deciding evidence.
- **Pre-flight numerical failures.** If many cells trip the rank-deficiency or
  null-space residual gates, the QR construction is failing on real bases. The
  fallback would be to use SVD with the same relative tolerance instead of QR. The
  cost is negligible (SVD is 2–3× slower per cell at d ≤ 100, which is what the
  correlate bases are).
- **The wider null kills the orthog branch.** If most cells fail orthog FDR
  regardless of the FCR ratio, we cannot distinguish `inherited` from `low-power
  noise`. The fallback would be to re-run only the post-orthog `analyze_one` on the
  full data for the headline cells, restoring Phase G's tight null. This would
  break the apples-to-apples N matching but is the cleanest way to answer the
  question for the few cells where it matters most.

## 13. Open methodological questions for follow-up

Even with a clean B.2 result, several follow-ups would sharpen the conclusions:

- **Per-correlate ablation.** Instead of stacking all correlates and projecting once,
  project against each correlate alone and see which one drives the most FCR drop.
  This is the diagnostic for "which concept does the helix actually live in" — if
  carry_1 → col_sum_1 alone removes 95% of the helix while carry_1 → pp_a0_x_b1 alone
  removes 5%, the answer is "the helix is in col_sum_1." Cheap to add: one extra pass
  per correlate per cell, or 4× the headline cost.
- **Add a non-Phase-B null correlate.** Project away an irrelevant concept (say,
  `a_units` for carry_3 at L5/layer 16, where the Phase B `r_resid` is near zero)
  and confirm the FCR drop is near zero. This is the negative control for the test:
  it says "your projector does not always destroy structure; it destroys it
  selectively for genuine correlates." This would add weight to the negative
  (`inherited`) verdicts.
- **Layer-by-layer trajectory.** Rather than reporting the verdict per cell, track
  the FCR drop curve across layers {4..31} for each carry concept. If the drop is
  layer-dependent (e.g. carry_1 is owned at layer 4 and inherited at layer 16),
  that is a substantive finding about where in the model the carry representation
  forms vs. where it merges with col_sum.
- **Difficulty-stratified verdict.** Use the difficulty-matched pairs from the
  curated set (B.8 is built on these) to split the verdict by problem difficulty.
  If the carry helix is owned on easy problems and inherited on hard ones (or vice
  versa), the carry-difficulty axis is a representational dimension we did not
  anticipate.

These are deferred to follow-up runs; the current B.2 run executes the headline
design only.

## 14. Glossary

- **Carry helix.** A pattern in the projected residual-stream activations where, when
  the activations are projected into a concept's Phase C subspace and the centroid
  per concept value is computed, the centroids trace a 3-d helix as the concept
  value increases. Phase G detects this by Fourier-analyzing the centroid sequence;
  see `docs/phase_g_analysis.md` Section "FCR statistic" for the full definition.
- **Correlate set.** The set of concepts (from the 43-name registry) that share
  algebraic / structural relationships with a target carry concept, identified by
  Phase B's residual-correlation classification. For carry_1 at L3, this is
  {col_sum_1, pp_a0_x_b1, pp_a1_x_b0, col_sum_0}.
- **FCR.** Fourier Concentration Ratio. The fraction of squared-norm power in a
  centroid set that lies on the best two Fourier coefficients (the conjunction
  test). High FCR = strongly periodic centroid pattern.
- **FCR drop.** `(raw_fcr − orthog_fcr) / raw_fcr`. The relative reduction in FCR
  after orthogonalization. Used for the verdict rule.
- **Inherited.** A verdict for cells where the helix structure lies inside the
  correlate subspaces; orthogonalization removes most of the FCR or all of the
  amplitude. See §4.3.
- **Own structure.** A verdict for cells where the helix structure lies inside
  the carry's complement of the correlate subspaces; orthogonalization preserves
  most of the FCR. See §4.3.
- **P_perp / P⊥.** The orthogonal-complement projector: `P⊥ = I − Q Qᵀ` where Q is
  an orthonormal basis for the correlate subspaces. Applied per cell to the residual
  activations.
- **Power.** `total_helix_power` from `analyze_one`. The un-normalized squared-amplitude
  on the helix axis.
- **Sensitivity-stable.** A cell whose verdict is the same across the strict, loose,
  and headline correlate-threshold branches.
- **Subspace_type.** Either `phase_c` (Phase C basis from the cv consensus PCA) or
  `phase_d_merged` (Phase D's LDA-augmented basis).

## 15. Appendix: derivation of the centering-preserving projection

This is in the implementation comments, but worth writing out longhand because it is
the trick that makes the bit-identical-raw-FCR gate work.

Phase C stores per-cell projected coordinates as
`raw_projected = (X − μ) Bᵀ`,
where `X ∈ R^(N × 4096)` is the residualized activation matrix, `μ ∈ R^4096` is the
per-cell training mean, and `B ∈ R^(d × 4096)` is the basis (rows are basis vectors).

After orthogonalization, the activations are `X_orth = X (I − Q Qᵀ) = X − X Q Qᵀ`.

The orthogonalized projected coordinates, computed with the same centering, are
`(X_orth − μ) Bᵀ = (X − X Q Qᵀ − μ) Bᵀ = (X − μ) Bᵀ − X Q Qᵀ Bᵀ
                = raw_projected − X Q Qᵀ Bᵀ
                = raw_projected − (X Q) (Qᵀ Bᵀ)`.

So we compute the orthogonalized projection from the raw projection plus an
inexpensive `(N × 4096) · (4096 × r) · (r × d)` correction, where `r` is the rank of
the correlate basis (typically 25–35) and `d` is the target subspace dimension
(typically 5–20). No mean materialization required; no double-centering risk; the
raw branch reproduces Phase G bit-for-bit by construction.

(The `if q_rank.shape[1] == 0: return raw_projected.copy()` guard at
[phase_h_orthogonalize.py:358](../../../arithmetic-geometry/phase_h_orthogonalize.py#L358)
handles the edge case where no correlate has a basis on disk. Such cells are then
recorded as `correlate_set_missing_bases` non-empty and would still produce a result,
though the orthog branch trivially equals the raw branch.)

## 16. Appendix: per-concept correlate-set summary

This section summarizes the correlate sets used in the headline branch
(`threshold = 0.30`, `correlate_basis = phase_c`). Full per-cell correlate sets and
the Phase B audit trail are in
`/data/user_data/anshulk/arithmetic-geometry/phase_h/summary/correlate_sets.json`
with 1,676 keyed entries.

### 16.1 Correlate-set sizes

Headline correlate-set size summary:

| Statistic | Value |
|---|---:|
| Min | 1 |
| P10 | 2 |
| P25 | 10 |
| Median | 13 |
| P75 | 16 |
| P90 | 19 |
| Max | 20 |
| Mean | 12.31 |

By target concept:

| Concept | Rows | Mean | Std | Min | P25 | Median | P75 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `carry_0` | 19 | 11.84 | 0.69 | 9 | 12 | 12 | 12 | 12 |
| `carry_1` | 176 | 13.23 | 3.27 | 9 | 10 | 14 | 16 | 18 |
| `carry_2` | 116 | 15.77 | 3.69 | 12 | 12 | 13 | 20 | 20 |
| `carry_3` | 54 | 13.00 | 0.00 | 13 | 13 | 13 | 13 | 13 |
| `carry_4` | 54 | 1.33 | 0.48 | 1 | 1 | 1 | 2 | 2 |

`carry_4` has the smallest correlate sets in the headline branch. That is a data
fact from Phase B's structural-pair table, not a hand choice in Phase H.

### 16.2 Most common correlates

Top headline-branch correlates by number of rows in which they appear:

| Correlate | Rows |
|---|---:|
| `col_sum_1` | 365 |
| `pp_a1_x_b0` | 365 |
| `pp_a1_x_b1` | 347 |
| `col_sum_2` | 319 |
| `b_units` | 311 |
| `col_sum_0` | 311 |
| `pp_a0_x_b0` | 311 |
| `pp_a0_x_b1` | 303 |
| `carry_0` | 292 |
| `a_tens` | 278 |
| `pp_a2_x_b1` | 260 |
| `a_units` | 249 |
| `carry_1` | 189 |
| `col_sum_3` | 185 |
| `carry_2` | 180 |
| `b_tens` | 162 |
| `pp_a2_x_b2` | 162 |
| `pp_a2_x_b0` | 134 |
| `pp_a1_x_b2` | 126 |
| `carry_3` | 108 |
| `pp_a0_x_b2` | 54 |
| `ans_digit_1_msf` | 54 |
| `ans_digit_4_msf` | 36 |
| `ans_digit_3_msf` | 20 |
| `a_hundreds` | 18 |
| `ans_digit_0_msf` | 18 |

Scalar aggregate concepts (`product`, `product_binned`, `total_carry_sum`,
`max_carry_value`, `n_nonzero_carries`, `n_answer_digits`, `correct`) are excluded
by design even when Phase B marks them as structurally correlated.

### 16.3 QR rank versus nominal dimension

In the headline Phase C branch, `d_correlates_rank == d_correlates_nominal` for all
419 rows. In the Phase D merged-basis sensitivity branch, 293 / 419 rows are
rank-deficient after QR truncation. That is reported in §10.2 and is not used as the
headline linear-algebra gate.

## 17. Appendix: final per-cell table notes

The complete per-cell table is the CSV:

`/data/user_data/anshulk/arithmetic-geometry/phase_h/summary/orthogonalization_results.csv`

It has 1,676 rows and 83 columns. The columns are grouped as:

- target identity: `target_kind`, `level`, `layer`, `population`, `concept`,
  `subspace_type`, `period_spec`, `period`;
- branch identity: `correlate_threshold`, `correlate_basis`;
- correlate audit: `d_correlates_nominal`, `d_correlates_rank`, `correlate_set`,
  `correlate_set_missing_bases`;
- raw Fourier statistics: `raw_*`;
- orthogonalized Fourier statistics: `orthog_*`;
- drop statistics: `drop_two_axis_*`, `drop_helix_*`;
- validation diagnostics: `validation_*`.

The most useful columns for inspection are:

| Column | Meaning |
|---|---|
| `raw_helix_fcr` | Phase G helix FCR recomputed on the curated subset |
| `orthog_helix_fcr` | same statistic after correlate orthogonalization |
| `drop_helix_fcr_rel` | `(raw - orthog) / raw` |
| `raw_total_power_helix` | unnormalized raw helix-axis power |
| `orthog_total_power_helix` | unnormalized orthogonalized helix-axis power |
| `drop_helix_power_rel` | `(raw_power - orthog_power) / raw_power` |
| `raw_helix_q_value` | Phase H FDR q-value for raw helix statistic |
| `orthog_helix_q_value` | Phase H FDR q-value for orthogonalized helix statistic |
| `verdict` | final branch-level verdict after Phase H FDR |

The headline branch can be extracted with:

```
df[(df.correlate_threshold == 0.30) & (df.correlate_basis == "phase_c")]
```

All headline rows have `verdict == "inherited"`.

## 18. Appendix: code provenance

- `phase_h_orthogonalize.py`: 1,195 lines (as of run launch). Single-file design,
  all helpers + driver + toy harness. Imports from `phase_g_fourier` for `analyze_one`,
  `compute_centroids_grouped`, `permutation_null`, `_benjamini_hochberg`,
  `load_residualized`, `load_phase_c_projected`, `load_phase_c_eigenvalues`,
  `load_phase_c_metadata`, `load_phase_d_merged_basis`, `load_coloring_df`,
  `_get_phase_c_group_labels`, `compute_labels_and_linear_values`,
  `resolve_carry_binned_spec`, `get_fourier_concepts`, `get_population_mask`,
  and the constants `LAYERS, MIN_POPULATION, MIN_CARRY_MOD10_VALUES,
  CARRY_CONCEPTS_BY_LEVEL, FDR_THRESHOLD, N_PERMUTATIONS, POPULATIONS`.
  No copies of those functions are maintained inside `phase_h_orthogonalize.py`.
- `run_phase_h.sh`: SLURM wrapper. `set -euo pipefail`; pre-flight checks for
  config, curated set, classified_pairs, helices CSV, residualized count ≥ 27,
  Phase C basis count ≥ 100, Phase D basis count > 0; toy run first; main run with
  `--n-perms 1000`. Mail on `BEGIN, END, FAIL` to `anshulk@andrew.cmu.edu`.
- Output paths:
  - per-cell JSON: `/data/user_data/anshulk/arithmetic-geometry/phase_h/orthogonalize/L{L}/layer_{ly:02d}/{pop}/{concept}/<filename>.json`
  - summary CSV: `/data/user_data/anshulk/arithmetic-geometry/phase_h/summary/orthogonalization_results.csv`
  - correlate-set audit: `/data/user_data/anshulk/arithmetic-geometry/phase_h/summary/correlate_sets.json`
  - failures (if any): `/data/user_data/anshulk/arithmetic-geometry/phase_h/summary/orthogonalization_failures.json`
- Log: `/home/anshulk/arithmetic-geometry/logs/phase_h_orthogonalize.log` (rotating,
  10 MB per file, 3 backups). The full SLURM stdout/stderr go to
  `/home/anshulk/arithmetic-geometry/logs/slurm-7666131.{out,err}`.

## 19. Appendix: dependency graph for downstream phases

B.2 / Phase H is upstream of:

- **B.3 (GPLVM admissibility).** GPLVM kernel choice will use B.2 verdicts as priors:
  cells classified `inherited` are not promising candidates for a "carry-only"
  manifold model and would be either skipped or modeled jointly with the correlate
  subspace.
- **B.5 (Persistent homology, full point-cloud).** PH on the carry's full point
  cloud will be cross-tabulated against B.2 verdicts. If PH says "loop" and B.2 says
  `inherited`, the loop is the col_sum loop seen through carry's lens.
- **B.6 (Subspace ablation).** Causal ablation will target the carry directions
  identified as `own_structure` first, since those are the candidates for actually
  carrying functional information. `inherited` cells are deprioritized.
- **B.9 (Cross-method consistency table).** B.2 verdicts are one row per cell in
  this table, joined to PH verdicts (B.5) and ablation verdicts (B.6). The table
  is the deliverable that lets us say "for the L5/layer 16 carry_1 helix, three
  independent methods agree it is owned" (or do not agree, in which case the
  per-method discrepancy becomes the finding).

Updates to `docs/next_steps.md` upon run completion will mark B.2 as done in the
Step 2 status, populate B.15 sensitivity rows with realized numbers, and update the
B.13 dependency entries that mark B.3 as unblocked.

## 20. Running-tally of the final-verdict counts

Final counts from `orthogonalization_results.csv`:

```
All branches combined:
  inherited:     1676 / 1676
  own_structure:    0 / 1676
  ambiguous:        0 / 1676
  unclassified:     0 / 1676

Headline branch (threshold 0.30, Phase C basis):
  inherited:      419 / 419
  own_structure:    0 / 419
  ambiguous:        0 / 419
  unclassified:     0 / 419

Strict branch (threshold 0.50, Phase C basis):
  inherited:      419 / 419

Loose branch (threshold 0.20, Phase C basis):
  inherited:      419 / 419

Phase D merged-basis branch (threshold 0.30, Phase D merged basis):
  inherited:      419 / 419
```

By concept, headline branch:

```
carry_0: inherited=19,  own_structure=0, ambiguous=0
carry_1: inherited=176, own_structure=0, ambiguous=0
carry_2: inherited=116, own_structure=0, ambiguous=0
carry_3: inherited=54,  own_structure=0, ambiguous=0
carry_4: inherited=54,  own_structure=0, ambiguous=0
```

By level, headline branch:

```
L3: inherited=69
L4: inherited=134
L5: inherited=216
```

By population, headline branch:

```
all:     inherited=137
correct: inherited=146
wrong:   inherited=136
```

By layer, headline branch:

```
layer 4:  inherited=48
layer 6:  inherited=47
layer 8:  inherited=47
layer 12: inherited=47
layer 16: inherited=47
layer 20: inherited=47
layer 24: inherited=44
layer 28: inherited=46
layer 31: inherited=46
```

## 21. Algebraic appendix: the FCR statistic, longhand

This section restates Phase G's FCR algebra in the notation we use here, so the
verdict rule does not have to refer back to `phase_g_analysis.md`. Anyone reading
B.2 in isolation should be able to follow the derivation.

### 21.1 Setup

Fix a target cell `(level, layer, population, concept, period_spec, subspace_type)`.
The subspace_type determines a basis matrix `B ∈ R^(d × 4096)` (rows are basis
vectors, `d` is the subspace dimension — typically 5–20). The period_spec determines
the period `P` (e.g. P=18 for `carry_1` raw, P=15 for `carry_1` binned at L3) and the
ordered set of unique values `v₁ < v₂ < … < v_m` along with their `v_linear` mapping
(usually identity, but for binned specs the largest bin is at the centroid of the
collapsed values).

Activations enter as `X ∈ R^(N × 4096)` (residualized at the cell's level/layer);
projected coordinates are `Y = (X − μ) Bᵀ ∈ R^(N × d)`, with `μ` Phase C's per-cell
training mean.

Labels `ℓ ∈ {1, …, m}^N` assign each row to its value group; group sizes are
`n_k = |{i : ℓ_i = k}|`. Centroids are computed group-wise:
```
c_k = (1/n_k) ∑_{i: ℓ_i = k} Y_i ∈ R^d
```
Then DC-removed:
```
c'_k = c_k − (1/m) ∑_j c_j
```
giving the centroid matrix `C ∈ R^(m × d)`.

### 21.2 The Fourier transform on the centroid sequence

For each coordinate `j ∈ {1, …, d}`, take the discrete Fourier transform of the
centroid sequence `{c'_{k,j}}_{k=1}^m`:
```
F_j(f) = ∑_{k=1}^m c'_{k,j} · exp(-2πi · f · k / m)
```
for frequencies `f ∈ {0, 1, …, ⌊m/2⌋}`. The squared magnitude `|F_j(f)|²` is the
Fourier power at frequency `f` on coordinate `j`.

The total power across all (frequency, coordinate) pairs is:
```
T = ∑_f ∑_j |F_j(f)|²
```

### 21.3 Two-axis FCR

Phase G's "two-axis FCR" answers: of the total Fourier power, what fraction is
concentrated on a single (frequency, two-coordinate-pair)?

For each frequency `f` and each pair of coordinates `(j₁, j₂)` (`j₁ ≠ j₂`),
compute the pair's power:
```
P(f, j₁, j₂) = |F_{j₁}(f)|² + |F_{j₂}(f)|²
```
The two-axis FCR is the maximum over all such triples normalized by total power:
```
two_axis_fcr = max_{f, (j₁, j₂)} P(f, j₁, j₂) / T
```
Returned alongside the maximizer `(f*, j₁*, j₂*)`.

A high two-axis FCR means there is a single Fourier frequency for which two
coordinates carry most of the centroid power — geometrically, a circle in the
`(j₁, j₂)` plane traced out as the value sweeps from `v₁` to `v_m`. This is the
"circle" or "rotation" signal.

### 21.4 Helix FCR

Helix FCR adds a third coordinate carrying the *linear* dimension. For each candidate
linear coordinate `j_L`, compute:
```
linear_power(j_L) = ∑_k (v_linear[k] · c'_{k,j_L})²
```
that is, the inner product of the centroid coordinate sequence with the value-linear
ramp.

Then the helix FCR for a given frequency `f` and circle pair `(j₁, j₂)` and linear
coordinate `j_L` is:
```
helix_fcr(f, j₁, j₂, j_L) = (P(f, j₁, j₂) + linear_power(j_L)) / (T + total_linear_power)
```
where `total_linear_power = ∑_j linear_power(j)`. The function returns the maximum
helix FCR over all `(f, j₁, j₂, j_L)` triples and the maximizer.

A high helix FCR means there is a (frequency, circle, linear) triple such that two
coordinates carry the rotation and a third carries the linear ramp — geometrically,
a 3-d helix.

### 21.5 The permutation null

The null distribution is generated by repeatedly:
1. Permuting the labels `ℓ` while preserving group sizes `{n_k}`.
2. Recomputing centroids and the FCR statistic.
3. Repeating 1,000 times.

Phase G uses 1,000 permutations as a hard-coded constant (`N_PERMUTATIONS = 1000` in
`phase_g_fourier.py`). The conservative p-value is:
```
p_value = (count of null FCRs ≥ observed FCR + 1) / (1000 + 1)
```
The minimum reportable p-value is `1 / 1001 ≈ 0.000999`. When the observed FCR is
greater than all 1,000 null draws, `p_saturated = True` and the reported p-value is
this floor.

### 21.6 Significance and FDR

After all cells are tested, p-values are FDR-corrected via Benjamini–Hochberg
(`_benjamini_hochberg` in `phase_g_fourier.py`). Cells with `q < 0.05` are
significant. Phase G's pool was 3,480 tests; Phase H's pool is 13,408 tests
(1,676 cells × 4 statistics each).

### 21.7 Connection to the orthogonalization test

The orthogonalization test asks whether the helix FCR survives projection onto the
correlate-orthogonal complement of the carry's basis. Algebraically, if `Q` is an
orthonormal basis for the correlate row-space and `B` is the carry's Phase C basis,
then orthogonalized projected coordinates are:
```
Y_orth = Y − (X Q)(Qᵀ Bᵀ)
```
(see §15 for the centering-preservation derivation). The helix FCR is recomputed on
`Y_orth` instead of `Y`. The drop `(raw_fcr − orthog_fcr) / raw_fcr` is the verdict
input.

The key observation: this is *not* a test of whether the carry's Phase C basis lies
inside the correlate subspace. We expect substantial overlap — Phase F measured 2.15°
between carry_1 and col_sum_1 at L5/layer 16. The test is whether the *helical
structure* on the centroids lives in the overlap or in the (small) complement.
Because the complement is geometrically small (cos²(2.15°) ≈ 0.14% of variance), even
a real "owned" helix has to be quite tight in the carry-only direction to register.
The verdict rule's 0.30 / 0.50 thresholds were calibrated for this asymmetry — they
demand a pre-registered drop magnitude rather than just "any drop," because a small
drop is consistent with both "owned with most amplitude in the overlap" and "owned
in the complement." Pre-registering the threshold prevents post-hoc verdict
adjustment.

## 22. Algebraic appendix: the QR projector and rank truncation

The implementation builds the orthogonal-complement projector via QR decomposition
of the stacked correlate basis transpose, with relative-tolerance rank truncation.
This section walks through why each step is the way it is.

### 22.1 The naive approach and why it fails

The straightforward way to project onto the orthogonal complement of a subspace
spanned by the rows of `B_correlates ∈ R^(r × 4096)` is:
```
P_perp = I − B_correlatesᵀ (B_correlates B_correlatesᵀ)⁻¹ B_correlates
```
This works when `B_correlates` has full row rank. For our correlate sets, the rows
are concatenated bases of multiple highly-correlated concepts (col_sum_1, col_sum_0,
pp_a0_x_b1, pp_a1_x_b0 for carry_1 at L3), and the row space has substantial
redundancy. `B_correlates B_correlatesᵀ` becomes ill-conditioned or singular and the
naive inverse explodes.

### 22.2 QR with rank truncation

The fix is to compute `Q, R = qr(B_correlatesᵀ, mode="reduced")`. `Q` is
`4096 × r`, orthonormal; `R` is `r × r`, upper triangular. The diagonal of `R`
encodes the singular structure of `B_correlatesᵀ`.

We truncate to numerically full-rank columns:
```
rank = ∑(|diag(R)| > 1e-8 × max|diag(R)|)
Q_rank = Q[:, :rank]
```
The relative tolerance `1e-8` is the same constant used by NumPy's default
matrix_rank function. Then `P_perp = I − Q_rank Q_rankᵀ` is the projector onto the
orthogonal complement of the *numerically full-rank* portion of the correlate row
space.

### 22.3 Why we never materialize P_perp

`P_perp` is a `4096 × 4096` matrix — 128 MB in float64. We never form it. Instead:
- Apply to a vector `x`: `P_perp · x = x − Q_rank (Q_rankᵀ · x)`. Two matrix-vector
  multiplications, one of size `4096 × rank` and one of size `rank × 4096`. Total
  cost `O(rank × 4096)`, vs `O(4096²)` for the materialized projector. Memory: zero
  extra.
- Apply to a matrix `X ∈ R^(N × 4096)`: `P_perp X = X − (X Q_rank)(Q_rankᵀ)`. Two
  matrix-matrix products. Cost `O(N · 4096 · rank)`, memory `O(N · rank)` for the
  intermediate, then back to `O(N · 4096)`.

This makes the per-cell projector application linear in `rank` rather than quadratic
in `4096`, which is the difference between B.2 fitting in 4 hours vs taking days.

### 22.4 Validation per cell

Every cell records four diagnostic numbers:

| Diagnostic | Formula | Threshold | Meaning |
|---|---|---|---|
| `d_correlates_nominal` | row count of stacked basis | informational | how many basis rows we started with |
| `d_correlates_rank` | post-truncation rank | informational; flag if `< nominal/2` | numerical full-rank dimension |
| `residual_norm` | `‖B_correlates · P_perp‖_F / ‖B_correlates‖_F` | < 1e-6 | confirms correlates lie in P⊥'s null space |
| `q_orthonormality_norm` | `‖Qᵀ Q − I‖_F` | < 1e-10 | confirms QR decomposition gave a valid orthonormal basis |

The toy harness (T5) confirmed these are at machine epsilon for synthetic bases:
`residual = 4.812e-16`, `q_orth = 2.184e-15`. For real bases, the gates are looser
(`1e-6` and `1e-10`) to allow for accumulated float64 error across the multiple
basis loads and the `vstack`. If any cell trips either gate, it is logged and the
cell is excluded from the headline; we do not silently propagate numerically-bad
projectors.

### 22.5 The rank-deficiency interpretation

When `d_correlates_rank ≪ d_correlates_nominal`, it means the supposedly-distinct
correlate concepts share most of their basis — i.e., they are themselves in
superposition. This is informative on its own:

- For carry_1 at L3, the four correlates {col_sum_1, pp_a0_x_b1, pp_a1_x_b0, col_sum_0}
  have nominal dimensions roughly 11+8+8+10 = 37, with expected rank around 25–32 if
  they share a couple of dimensions. Anything below 20 would be surprising.
- The Phase D merged basis includes LDA-augmented dimensions per concept, so the
  nominal totals are larger. Expect nominal ≈ 60–80, rank ≈ 35–50.

We will tabulate the per-cell rank distribution in §10 (after the run) to see
whether the rank-deficiency story has anything to add to Phase F's principal-angle
story.

## 23. Sensitivity branches in detail

The four-branch grid is the principal defense against threshold-pinned verdict
fragility. This section explains what each branch is testing and what specifically
would invalidate the verdict.

### 23.1 Headline branch (threshold = 0.30, Phase C basis)

The 0.30 threshold matches the Phase B `r_resid` cutoff for the deconfounding action
classification. Concepts with `r_resid` between 0.30 and 0.50 are deconfounded but
not removed in Phase B; concepts with `r_resid > 0.50` are flagged as candidates for
removal. Using 0.30 as the correlate threshold means we project against every
deconfoundable concept, which is the most natural definition of "structural correlate."

Phase C basis is the cv-consensus PCA basis from `phase_c_subspaces.py`. This is
Phase G's primary detection basis (414 of 500 helices were detected on it).

### 23.2 Strict branch (threshold = 0.50, Phase C basis)

Tests whether the verdict depends on weak correlates. Only includes concepts with
`r_resid > 0.50` — typically just col_sum_1 and the partial products for carry_1.
The correlate set is smaller, so the projector removes less. If the headline is
`inherited` and the strict branch is also `inherited`, the carry helix is in the
top correlates' subspace, not in the borderline ones.

### 23.3 Loose branch (threshold = 0.20, Phase C basis)

Tests whether the verdict depends on excluding sub-threshold correlates. Includes
concepts with `r_resid > 0.20`, picking up two or three additional adjacent
column sums and partial products for each carry. The projector removes more. If
the headline is `own_structure` and the loose branch is `inherited`, then the
"owned" verdict was conditional on us not projecting out a wider ring of
correlates — a genuine but milder superposition.

### 23.4 Phase D branch (threshold = 0.30, Phase D merged basis)

Tests whether the verdict depends on the basis used to define the carry's subspace.
Phase D's `merged_basis.npy` adds LDA-novel directions to the Phase C consensus PCA
basis. If the carry's "owned" structure lives in a Phase D LDA direction not
captured by Phase C, this branch will give a `own_structure` verdict where the
Phase C branch gave `inherited` — meaning the helix is owned, just not by the
conservative basis.

### 23.5 Verdict downgrade rule, written out

| Headline | Strict | Loose | Phase D | Final verdict |
|---|---|---|---|---|
| own | own | own | own | `own_structure` (sensitivity-stable) |
| inherited | inherited | inherited | inherited | `inherited` (sensitivity-stable) |
| own | own | inherited | own | `unstable` (loose disagrees) |
| own | inherited | own | own | `unstable` (strict disagrees) |
| inherited | inherited | inherited | own | `inherited`, with Phase D footnote |
| ambiguous | own | inherited | * | `unstable` |
| ambiguous | ambiguous | ambiguous | * | `ambiguous` (sensitivity-stable) |
| any | * | * | * mismatching * | record both, headline takes Phase G's basis |

The rule was set in `docs/b2_plan.md` before the run and we are not changing it
post-hoc. The published table in §11 will use these final verdicts; the appendix
will list every per-branch verdict so a reader can audit any cell.

## 24. Why this test is the right test for this question

The question — "is the carry helix owned or inherited?" — could in principle be
answered by several different methods. We chose orthogonalization. Why?

### 24.1 Alternatives we considered

- **Per-correlate ablation** (project against one correlate at a time and rank by
  drop). This would tell us *which* correlate "owns" the helix, but it does not
  give us a clean "own vs. inherited" verdict at the headline level. It is in §13
  as a follow-up.
- **Causal patching** (replace the carry's activations with a permuted version,
  measure downstream loss). This is the gold standard for "does the model use the
  carry," but it answers a different question — functional use, not representational
  ownership. It is B.6.
- **Linear probe transfer** (train a probe to predict the carry on activations
  with col_sum_1 projected out, see if it still works). This tests whether the
  carry value is *linearly readable* from the orthogonal complement; if yes, the
  carry is independently encoded. But it does not directly test whether the
  *helix* lives in the complement — a non-helical linear encoding would give a
  positive verdict. We want the helix-specific test.
- **Subspace nullification** (replace the carry's projection with a permuted
  version, see whether downstream behavior changes). Also a causal test, also
  the wrong question.

Orthogonalization tests the helix-specific question directly, with no train/test
split, and re-uses the Phase G machinery for an apples-to-apples comparison.

### 24.2 What this test does *not* test

- Whether the carry has a representation in the model. (The carry has *some*
  representation that supports a 0.55 helix FCR — we know the representation
  exists. The question is whether it is *separate* from col_sum_1.)
- Whether the model uses the carry computationally. (B.6's job.)
- Whether unmeasured features confound the verdict. (See §12.1; deferred to
  Paper 4.)
- Whether the helix is the right shape for K&T's (2025) parametric helix model.
  (Phase G's `helix_fcr` is a generic conjunction test that detects a circle plus
  a linear ramp; K&T's model is a specific 5-parameter helix. The two are
  approximately equivalent for the cells we care about, but only approximately.)

## 25. Literature comparison

How does B.2 sit among the existing methods literature in interpretability?

### 25.1 Gurnee et al. (2024) — "Universal Neurons"

Gurnee et al. observed that some neurons in production LLMs respond to specific
input features in an essentially orthogonal way (their "feature-specific" neurons).
Their definition of orthogonality was a per-pair test: two features are orthogonal
if their direction overlap is below a threshold. This is descriptive — they observe
which features happen to be orthogonal — not prescriptive.

B.2 is the prescriptive complement: given a candidate "geometric structure" claim,
*enforce* orthogonality to known correlates and see whether the claim survives. The
two methods answer different questions and could in principle be combined: Gurnee-
style observation tells us which neurons are already orthogonal; B.2-style
projection tests whether a sub-population is orthogonal to a specified set.

### 25.2 Kantamneni & Tegmark (2025)

K&T fit a parametric 5-parameter helix model to single-token integer activations and
report Fourier components at frequencies {2, 5, 10}. Their test is parametric and
positive (they assume the helix and find its parameters); B.2 is non-parametric and
negative (we assume nothing about the helix shape and test whether the FCR survives
projection).

K&T's helix is at the *number-token* position; ours is at the `=` position. K&T
worked on standalone integers; we work on multiplication operands and outputs. The
two findings are not in conflict — Phase G's number-token screening (Appendix G of
`phase_g_analysis.md`) confirms the K&T helix at the number-token position and finds
no helix at `=` for digit concepts. The carry helix at `=` is a separate finding
that K&T did not make.

### 25.3 Bai et al. (2024)

Bai et al. trained small Transformers on multiplication and observed circular and
helical structure in intermediate computations. Their setup is a synthetic toy; they
have ground truth and can verify causally. We cannot — we have a production LLM and
no ground truth, so the orthogonalization test is our best available substitute for
their causal verification.

### 25.4 Bricken et al. (2023) — Anthropic SAE

Anthropic's sparse autoencoder work decomposes activations into a (much larger,
sparse) set of features. The claim is that the right unit of representation is the
SAE feature, not the residual-stream direction. Our Phase B/C/D registry concepts
are interpretable arithmetic concepts — col_sum_1, pp_a0_x_b1, etc. — which are not
SAE features but are arguably cleaner for the multiplication task.

The two methodologies have different trade-offs: SAE features are model-discovered
and may not align with the analyst's concepts; registry concepts align with the
analyst's concepts but may not match the model's internal decomposition. B.2 tests
the registry concepts; Paper 4 will redo this test against SAE features.

### 25.5 Templeton et al. (2024)

Templeton et al. observed that SAE features are themselves often in superposition
with other SAE features (a "dictionary collision" problem). The B.2 test, if
applied to SAE features, would have to handle the same problem — orthogonalizing
against an SAE feature might project away other SAE features that are in
superposition with it. The 4096-dim residual stream means there is no shortage of
overlap.

For B.2 against registry concepts, the situation is somewhat better: the
registry has 43 concepts vs. SAE's tens of thousands, and the registry concepts
have known algebraic relationships that we can use to choose which to project.
But the same fundamental issue is there.

### 25.6 Conmy et al. (2023) — ACDC

ACDC is a causal subgraph discovery method for finding which model components are
needed to perform a task. It is causal, not representational; it finds attention
heads and MLP components, not residual-stream directions. Complementary to B.2.

### 25.7 Position in the literature

To our knowledge no published paper has run a representational-ownership test
of the form "geometric structure X is observed in subspace Y; project against
the algebraically-correlated subspaces and see whether X survives" on a
production LLM. Bricken et al. and Templeton et al. acknowledge superposition as
a confound; ACDC, K&T, and Gurnee et al. each take a different angle on
mitigation. B.2 is the missing direct test.

## 26. Test designs we considered and rejected

This section documents methodological choices that did not make it into B.2, with
the reasons for rejection. It is here for the record.

### 26.1 Gradient-based attribution

Idea: take the gradient of the helix-FCR loss with respect to the activations,
attribute the helix to the residual-stream directions with the largest gradient
norm. Rejected because:
- The FCR is a non-smooth function of the centroid coordinates (max over discrete
  frequencies and pairs), so gradient signal is noisy.
- The attribution is to individual coordinates, not subspaces; would have to
  cluster after the fact.
- The result is correlational, not causal, and not directly comparable to other
  ownership claims.

### 26.2 Information-bottleneck regularization

Idea: train a probe that predicts the carry value from the activations with an
information-bottleneck on the dimension count. Rejected because:
- Requires probe training, which means train/test splits, regularization choices,
  hyperparameter sweep — all of which inject more degrees of freedom than B.2.
- The result is "what's the smallest probe that decodes the carry," not "is the
  helix owned by the carry."
- Would have to re-run for each layer × population.

### 26.3 Causal mediation analysis

Idea: replace the carry-only complement of the activations with a permuted version,
measure the effect on downstream helix detection in later layers. Rejected because:
- Would require running the full forward pass for each cell × permutation, which
  is computationally orders of magnitude more expensive than the centroid-FCR
  recompute. (At 1,676 cells × 1,000 permutations × 31 layers of forward pass,
  this is infeasible on the project budget.)
- It tests "downstream geometry depends on the carry's own complement," which is
  weaker than "the helix lives in the carry's own complement."

### 26.4 Subspace SVD comparison

Idea: take the top singular vector of the carry's centroid Fourier transform on
the helix axis; compare to the top singular vector of col_sum_1's centroid Fourier
transform. If they are nearly parallel, "inherited"; if orthogonal, "owned."
Rejected because:
- Compares two specific singular vectors instead of the full structure.
- Sensitive to scale and centering.
- No natural verdict threshold — would have to pick another arbitrary cosine cutoff.

### 26.5 Reading off Phase F angles directly

Idea: skip the orthogonalization test; just use the Phase F principal angles. If
the angle is small, the carry helix is inherited. If large, owned. Rejected because:
- The angle measures subspace overlap globally, not whether the helix specifically
  lives in the overlap. Two subspaces could share most of their direction without
  sharing the helix structure.
- The 2.15° angle for carry_1 ↔ col_sum_1 at L5/layer 16 is a property of the
  Phase C bases, not of the centroids. The helix lives on the centroids. Need a
  centroid-level test.

## 27. The L5 / layer 16 / carry_1 headline cell, in advance

Phase G's L5/layer 16 carry_1 detections are all `carry_raw` in the final
`phase_g_helices.csv` rows. The `correct / phase_c / carry_raw` row has
`helix_fcr = 0.432781`, period 18, and `p_saturated = True`; the `all / phase_c /
carry_raw` row has `helix_fcr = 0.656754`. Phase F gives the carry_1 ↔ col_sum_1 angle
of 2.15° at this exact cell. So this is the cell where B.2's verdict matters most:
if the helix vanishes here, the carry-helix story is inherited at the most central
cell, and Phase G's headline reframes; if the helix survives here, carry_1 has its
own helical structure even at the tightest possible superposition.

What we know in advance:

- **Raw N (curated, correct).** 1,401 rows. ~12 / value group on average for the
  binned spec at L5; the smallest group will be in the tail of the carry_1
  distribution.
- **Correlate set (predicted from Phase B).** col_sum_1, col_sum_0, pp_a0_x_b1,
  pp_a1_x_b0, pp_a1_x_b1 (this last one is L5-only and may add for L5 specifically).
  Possibly also col_sum_2 if `r_resid > 0.30` at L5/correct.
- **Stacked correlate basis nominal dimension.** ~12 + 9 + 8 + 8 + 8 = 45 (rough,
  Phase C dim_consensus values vary). Expected QR rank: ~30–35.
- **Phase F angle.** 2.15° between carry_1 and col_sum_1 alone. With four
  correlates stacked, the *minimum* principal angle between carry_1 and the joint
  span will be even smaller — possibly under 1°. This means cos²(θ) ≈ 0.9997, so
  >99.97% of carry_1's variance is in the joint correlate span. The helix has to
  live in the remaining 0.03% to be "owned."
- **Predicted FCR drop if owned.** `(0.59 − x) / 0.59 < 0.30` requires `x > 0.41`.
  So the orthog `helix_fcr` would have to stay above 0.41 to register `own_structure`.
  Given the geometry, we should not be confident that even a true "owned" helix
  would clear that threshold at this cell — the projection is removing nearly all
  of carry_1's variance.
- **Predicted FCR drop if inherited.** Power should drop to ~10⁻³ × raw, and FCR
  should fall to noise (~0.1–0.2). FDR significance lost.

So the headline-cell verdict has limited resolution at L5/layer 16: the geometry
is so tight that "owned" requires almost all of the helix to live in the very
small carry-only complement. A more diagnostic cell is L5 at an earlier layer
(say layer 4 or 8) where Phase F's angle is somewhat looser and the test can
distinguish the two hypotheses with more headroom. We will report verdicts at
all layers and look at the layer-trajectory specifically.

### 27.1 Completed L5 / layer 16 / carry_1 rows

The completed run has 24 rows for `carry_1 / L5 / layer 16`: three populations
(`all`, `correct`, `wrong`) × two target bases (`phase_c`, `phase_d_merged`) × four
sensitivity branches. Every row has verdict `inherited`.

The Phase C target-basis rows are the closest match to the Phase G headline basis:

| Population | Branch | Raw FCR | Orthog FCR | FCR drop | Raw power | Orthog power | Power drop | Raw q | Orthog q | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| all | 0.20 / phase_c | 0.685845 | 0.333839 | 0.513245 | 22.906360 | 0.000010 | 1.000000 | 0.006991 | 1.0 | inherited |
| all | 0.30 / phase_c | 0.685845 | 0.299026 | 0.564004 | 22.906360 | 0.000009 | 1.000000 | 0.014392 | 1.0 | inherited |
| all | 0.50 / phase_c | 0.685845 | 0.278955 | 0.593269 | 22.906360 | 0.000016 | 0.999999 | 0.006991 | 1.0 | inherited |
| all | 0.30 / phase_d_merged | 0.685845 | 0.275295 | 0.598605 | 22.906360 | 0.000012 | 0.999999 | 0.004625 | 1.0 | inherited |
| correct | 0.20 / phase_c | 0.469039 | 0.492755 | -0.050564 | 24.035757 | 0.003147 | 0.999869 | 0.024835 | 1.0 | inherited |
| correct | 0.30 / phase_c | 0.469039 | 0.574694 | -0.225258 | 24.035757 | 0.004577 | 0.999810 | 0.012704 | 1.0 | inherited |
| correct | 0.50 / phase_c | 0.469039 | 0.546930 | -0.166065 | 24.035757 | 0.007861 | 0.999673 | 0.023196 | 1.0 | inherited |
| correct | 0.30 / phase_d_merged | 0.469039 | 0.549633 | -0.171829 | 24.035757 | 0.004569 | 0.999810 | 0.016121 | 1.0 | inherited |
| wrong | 0.20 / phase_c | 0.645040 | 0.211073 | 0.672775 | 22.322959 | 0.000009 | 1.000000 | 0.010979 | 1.0 | inherited |
| wrong | 0.30 / phase_c | 0.645040 | 0.211756 | 0.671716 | 22.322959 | 0.000010 | 1.000000 | 0.009034 | 1.0 | inherited |
| wrong | 0.50 / phase_c | 0.645040 | 0.273149 | 0.576540 | 22.322959 | 0.000013 | 0.999999 | 0.004625 | 1.0 | inherited |
| wrong | 0.30 / phase_d_merged | 0.645040 | 0.200150 | 0.689710 | 22.322959 | 0.000012 | 0.999999 | 0.006991 | 1.0 | inherited |

For `correct`, the FCR ratio rises after orthogonalization in the Phase C target-basis
rows. That is why the power and q-value columns are necessary. The absolute helix
power drops from 24.035757 to at most 0.007861, and the orthogonalized q-value is
1.0 in every branch. The verdict is therefore inherited by the loss-of-significance
branch of the pre-registered rule, not by the FCR-drop branch.

## 28. Sample table layout for the final report (template)

The final per-cell table will look approximately like:

| Concept | Level | Layer | Pop | Sub | Spec | Raw FCR | Orth FCR | FCR drop | Raw pwr | Orth pwr | Pwr drop | Raw q | Orth q | Verdict | Stable? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carry_0 | 3 | 4 | all | phC | raw | 0.32 | 0.05 | 0.84 | 0.70 | 0.001 | 1.00 | <.001 | 0.18 | inherited | yes |
| carry_1 | 3 | 4 | all | phC | raw | 0.55 | 0.17 | 0.69 | 2.77 | 0.004 | 1.00 | <.001 | 0.42 | inherited | yes |
| ... | | | | | | | | | | | | | | | |

Aggregate counts will look like:

```
Total target cells: 419
By verdict (headline branch only):
  own_structure: ___
  ambiguous:     ___
  inherited:     ___
  unclassified:  ___

After sensitivity downgrade:
  sensitivity-stable: ___ / 419
  unstable:           ___ / 419

By concept:
  carry_0: own=___, amb=___, inh=___, unstable=___, total=19
  carry_1: own=___, amb=___, inh=___, unstable=___, total=176
  carry_2: own=___, amb=___, inh=___, unstable=___, total=116
  carry_3: own=___, amb=___, inh=___, unstable=___, total=54
  carry_4: own=___, amb=___, inh=___, unstable=___, total=54

By level:
  L2: own=___, amb=___, inh=___, unstable=___, total=2
  L3: own=___, amb=___, inh=___, unstable=___, total=93
  L4: own=___, amb=___, inh=___, unstable=___, total=150
  L5: own=___, amb=___, inh=___, unstable=___, total=255

By layer (collapsed across concepts and levels):
  layer 4:  own=___, amb=___, inh=___, unstable=___
  layer 6:  own=___, amb=___, inh=___, unstable=___
  layer 8:  own=___, amb=___, inh=___, unstable=___
  layer 12: own=___, amb=___, inh=___, unstable=___
  layer 16: own=___, amb=___, inh=___, unstable=___
  layer 20: own=___, amb=___, inh=___, unstable=___
  layer 24: own=___, amb=___, inh=___, unstable=___
  layer 28: own=___, amb=___, inh=___, unstable=___
  layer 31: own=___, amb=___, inh=___, unstable=___

Phase D merged basis branch (sensitivity):
  ...

Amplitude-collapsed cells (FCR rule says own_structure but power_rel > 0.90):
  Listed individually in §28.x
```

These templates exist so that whoever fills in the final numbers does not have to
re-derive the structure of the report from scratch.

### 28.1 Final compact headline table

The completed headline branch can be summarized compactly as:

| Concept | Rows | Raw helix q<.05 | Orthog helix q<.05 | Median raw FCR | Median orthog FCR | Median FCR drop | Median power drop | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `carry_0` | 19 | 3 | 0 | 0.326380 | 0.291768 | 0.121298 | 0.998711 | inherited |
| `carry_1` | 176 | 123 | 0 | 0.390388 | 0.138644 | 0.607908 | 0.997163 | inherited |
| `carry_2` | 116 | 88 | 0 | 0.315808 | 0.102687 | 0.693218 | 0.998769 | inherited |
| `carry_3` | 54 | 23 | 0 | 0.275028 | 0.146687 | 0.460513 | 0.999774 | inherited |
| `carry_4` | 54 | 0 | 0 | 0.371408 | 0.300213 | 0.122441 | 0.650857 | inherited |

The raw significance column is included to keep the result honest. `carry_4` has
0 / 54 raw curated helix-q significant rows in the Phase H FDR pool, despite all
54 being Phase G full-population helix targets. Its final Phase H verdict is still
`inherited` because the orthogonalized branch is non-significant, but the raw curated
replay is weak for this concept.

By level:

| Level | Rows | Raw helix q<.05 | Orthog helix q<.05 | Median raw FCR | Median orthog FCR | Median FCR drop | Median power drop |
|---|---:|---:|---:|---:|---:|---:|---:|
| L3 | 69 | 50 | 0 | 0.319316 | 0.122056 | 0.653482 | 0.993433 |
| L4 | 134 | 84 | 0 | 0.398978 | 0.133243 | 0.691658 | 0.997178 |
| L5 | 216 | 103 | 0 | 0.316522 | 0.184095 | 0.478369 | 0.999550 |

By population:

| Population | Rows | Raw helix q<.05 | Orthog helix q<.05 | Median raw FCR | Median orthog FCR | Median FCR drop | Median power drop |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 137 | 91 | 0 | 0.382329 | 0.115933 | 0.684514 | 0.995720 |
| correct | 146 | 65 | 0 | 0.299477 | 0.192657 | 0.368934 | 0.998069 |
| wrong | 136 | 81 | 0 | 0.361785 | 0.124409 | 0.620400 | 0.997796 |

### 28.2 FCR-drop bins and power-drop bins

Headline branch FCR-drop bins:

| `drop_helix_fcr_rel` bin | Rows |
|---|---:|
| < 0 | 28 |
| 0 to 0.30 | 88 |
| 0.30 to 0.50 | 58 |
| 0.50 to 0.80 | 216 |
| ≥ 0.80 | 29 |

Headline branch power-drop bins:

| `drop_helix_power_rel` bin | Rows |
|---|---:|
| < 0.30 | 0 |
| 0.30 to 0.50 | 2 |
| 0.50 to 0.80 | 38 |
| 0.80 to 0.90 | 14 |
| 0.90 to 0.99 | 21 |
| 0.99 to 0.999 | 190 |
| ≥ 0.999 | 154 |

If the verdict used FCR drop alone, the headline branch would have been:

| FCR-only verdict | Rows |
|---|---:|
| `own_structure` | 116 |
| `ambiguous` | 58 |
| `inherited` | 245 |

The actual final verdict is 419 `inherited` because the orthogonalized helix q-value
is ≥ 0.05 for every row. There are 65 rows where the FCR-only rule would say
`own_structure` (`drop_helix_fcr_rel < 0.30`) but the power drop is > 0.90. These are
the amplitude-collapsed rows that the loss-of-significance rule was meant to catch.

Mirror case check: rows with `drop_helix_fcr_rel > 0.50` but `drop_helix_power_rel < 0.30`
= 0.

## 29. Difficulty stratification — deferred but planned

The curated set includes 2,400 difficulty-matched correct/wrong pairs. B.8 will
use these for the difficulty-matched correct vs. wrong comparison; B.2 does not
stratify by difficulty in its headline. But we can in principle also report B.2
verdicts on the matched pairs:

- If the carry helix at carry_1 / L5 / layer 16 is `inherited` on `all`, but
  `own_structure` on the difficulty-matched easy pairs and `inherited` on the
  matched hard pairs (or vice versa), the carry-difficulty axis is a
  representational dimension we did not anticipate.
- If the verdict is the same across difficulty buckets, the verdict is
  difficulty-robust and the headline is clean.

We do not run this stratification in the current B.2 launch (the curated set is
big enough for the headline run but slicing to difficulty buckets cuts N by
roughly half again). It is in the follow-up plan; will run after B.8.

## 30. Deliverable schedule

The following sequencing is planned:

1. **Run completion** (Sun May 3 ~03:00 EDT, ~95 min from launch).
2. **Output validation** (gates from §4.5 verified against
   `orthogonalization_results.csv`).
3. **§10 (validation diagnostics) and §11 (headline verdicts)** filled in from
   the summary CSV.
4. **§17 (per-cell intermediate tables)** replaced with final per-cell tables.
5. **§16 (correlate-set summary)** filled in from `correlate_sets.json`.
6. **§20 (verdict counts)** filled in.
7. **Phase G Appendix H (`phase_g_analysis.md`)** updated with curated-vs-full
   FCR comparison for all 419 cells.
8. **`docs/next_steps.md`** updated to mark Step B.2 done.
9. **`README.md`** updated to reflect B.2 results in the "What's Next" section.
10. **B.3 (GPLVM admissibility)** can begin once verdicts are cross-referenced
    against B.2.

If the run hits an unexpected failure mode (failure rate above ~5% of cells, or
FDR pool collapses to noise), the backup plan is to re-run only the failing cells
with denser permutations or to fall back to per-correlate ablation as the headline
test. Either fallback is documented in §12.5 in advance.

## 31. Reading guide for the final document

When this document is in its final form (post-run), readers interested in:

- **The headline number** should jump to §11 (verdict counts) and §27 (the
  L5/layer 16 cell specifically).
- **The methodological contribution** should read §1 (the question), §4 (the
  pre-registered design), §22 (the QR projector), and §25 (literature comparison).
- **The math** should read §15, §21, and §22 in sequence.
- **The skepticism** should read §12 (caveats) and §26 (rejected designs) before
  the headline.
- **The robustness** should read §23 (sensitivity branches), §10 (validation
  diagnostics), and §H of `phase_g_analysis.md` (curated vs. full).
- **The next steps** should read §13 (open questions), §29 (difficulty
  stratification), and §30 (deliverable schedule).

The document is intentionally redundant — the same key facts appear in multiple
places — so that any single section can be read in isolation without losing
context. This is the same convention as `phase_g_analysis.md` and is deliberate.

---

## 32. Final result statement

Literal result:

- Phase H processed 419 Phase G carry helix targets across four sensitivity branches,
  producing 1,676 rows.
- The job completed with 1,676 successful rows and 0 failures.
- The headline Phase C branch passes projector diagnostics at machine precision.
- The orthogonalized branch has 0 / 419 helix-significant rows and 0 / 419
  two-axis-significant rows after Phase H FDR.
- The `verdict` column is `inherited` for all 419 headline rows and all 1,676 branch
  rows.

Careful interpretation:

- Phase H supports the statement that, on the curated 8,264-problem set, the carry
  Fourier structure targeted by B.2 does not survive projection away from the selected
  structurally related correlate subspaces.
- Phase H does not by itself prove mechanism or causal use.
- Phase H does not say "carries are not represented." It says the tested Fourier
  geometry is not separable from the selected correlate subspaces under this
  orthogonalization control.
- Because the raw curated replay is weaker than the full Phase G population result,
  the paper should report the curated raw replay and the orthogonalized result
  separately.

Downstream consequence:

- B.3 / GPLVM should not frame carry manifolds as primarily "carry-private" until
  it explicitly tests the raw-vs-orthogonalized branch.
- The cleaner next question is: what shared manifold is represented across
  carries, column sums, and partial products, and whether any private carry geometry
  remains after removing the shared span.

---

End of completed Phase H analysis.
