# The Path to a Methodology Paper: Validated Plan

**Updated and validated against the executed code, the actual data, the audit findings, and the cited literature. All timeline content removed — the work proceeds ad hoc. Every numerical claim has been cross-checked against `docs/phase_*_analysis.md` and `labels/analysis_summary.json`. Every methodological claim has been cross-checked against the relevant paper in `papers/`. Where the prior version of this plan made a claim that did not hold up against evidence, the claim is corrected and the source of the correction is named in-line.**

---

## Part A: Foundation — what we have built

### A.1 The frame

The paper is a **methodology paper**. The contribution is not "Llama does multiplication this way." The contribution is "here is a tested pipeline of methods — linear, Fourier, probabilistic-manifold, topological, and causal — that together let you study geometric interpretability inside a production LLM. Multi-digit multiplication in Llama 3.1 8B is the case study that proves the pipeline works."

The headline argument:

> The Linear Representation Hypothesis identifies *where* concepts live inside a language model. It does not describe *what shape* concepts take inside those subspaces. We show — using multi-digit multiplication in Llama 3.1 8B as a controlled testbed — that linear methods find clean subspaces for every named concept, non-linear methods (Fourier, GPLVM, persistent homology) reveal curved structure inside them, and causal interventions show that for at least one concept the curvature is the substrate of computation.

Multiplication is the right testbed for this argument because:

- Every intermediate quantity has an exact algorithm-agnostic label (carries, partial products, column sums, answer digits). No other reasoning task gives you ground-truth labels for everything the model might be computing internally.
- Accuracy degrades smoothly across difficulty levels (L1 = 1×1 through L5 = 3×3), giving a built-in failure signal to compare against geometry.
- The model genuinely fails at L5 (3.4% accuracy on 122,223 problems — see A.2 below), so "correct" and "wrong" populations are large enough to compare.
- The arithmetic operations connect to existing literature on Fourier features (Nanda et al., Kantamneni & Tegmark, Zhou et al., Bai et al.), giving us a reference point for what success looks like and for what to import as priors.

### A.2 The pipeline so far (corrected against evidence)

Phases A through G have been completed. Each one earns its place in the final paper. Numerical claims are verbatim from the corresponding analysis doc; the previous version of this plan contained two numerical errors that are corrected here.

**Phase A — Diagnostic scouting.** UMAP/t-SNE visualizations across 9 layers `[4, 6, 8, 12, 16, 20, 24, 28, 31]`, 5 levels (L1–L5), 3 populations (`all`, `correct`, `wrong`). Generated 351 embedding files. Identified product magnitude as the dominant axis at every layer and pinned **layer 16 as the "information peak"** (Phase A §15: more top-50 entries at layer 16 than at any other layer). Activation norm ratios between correct and wrong sit between 0.992 and 1.038 across all (level, layer) cells, max separation 3.8% at L4 layer 24, so **no normalization is needed before Phase C** (Phase A §4).

**Phase B — Concept correlation audit.** Pairwise Pearson correlations after product-residualization, plus Spearman follow-up on top-30 pairs per population. Concept count: **42 in the registry** plus `product_binned` for 43 named labels; `product_binned` is excluded from the correlation table because product is the deconfounding variable. Total unique label-pair correlations across all five levels: **2,018** (= 55 + 190 + 351 + 561 + 861 across L1–L5; Phase B §14–§18). The previous version of this plan reported "2,677 pairwise correlations across 43 concepts," which appears to count something different (possibly all reported pairs across multiple populations or both raw and residualized counted separately). The 2,018-unique-pair count is what the paper should cite. Runtime: 32.3 seconds (Phase B §26). No structural confounds at the registry level.

Audit findings inside Phase B that the paper has to surface:
- Leading digits anti-correlate at r = −0.80 *post*-residualization. Label-level r ≈ 0.80 corresponds to ≪3% activation-level contamination because product is one direction in 4096; the impact on subspace separation is small but should be noted.
- a_units vs. b_units = +0.343 in the L5 correct-only population (above the 0.3 deconfounding threshold) versus +0.201 in the all-population. **Correct-only L5 analyses must residualize the operand-units pair**, or the curated set must rebalance them.

**Phase C — Linear subspace identification.** Conditional covariance plus randomized SVD with 1,000-permutation null and 5-fold cross-validation; sequential stopping rule per Buja & Eyuboglu 1992. **2,844 concept subspaces computed; 2,750 (96.7%) pass significance** (Phase C §11). At L5 the cross-validation correlations split by population: **L5/all 0.838–1.000** (floor 0.838 driven by `ans_digit_2` and `ans_digit_3`, both 0.84–0.89 across layers) and **L5/correct 0.859–0.998** (floor 0.859 driven by `pp_a2_x_b2`; Phase C §496–497, §957–960). The previous version of this plan attributed the 0.838 floor to the correct population — that was wrong and is corrected here: 0.838 is the all-population floor; the correct-only floor is 0.859. Runtime: 201 minutes. The lowest CV correlations identify the concepts most fragile under N/d pressure (ans_digit_2/3 in all; pp_a2_x_b2 in correct).

**Phase D — LDA refinement for low-variance concepts.** Carries are not high-variance, but they are highly discriminable. Fisher LDA picks them out where Phase C marginally does. **1,035 results at L5; 1,026 (99.1%) significant** (Phase D §23). For carries: λ₁ = **0.74–0.95** at L5/all with Cohen's d in the **7.2–13.5** range (Phase D §23, Table §560–562; range exactly 0.740–0.946). Important caveat the paper must respect: **at L5/correct the eigenvalues inflate to 0.88+ because N/d ≈ 1.02** (N=4,197 against the 4,096-dimensional residual stream — `phase_d_analysis.md:278, 818, 872`). This is the rank-deficiency / "barely overcomplete" regime where the regularized within-class scatter matrix produces compressed eigenvalues uniformly close to 1 regardless of the underlying signal. The previous version of this plan said "N/d ≈ 0.71"; that ratio is actually L4/correct (N=2,897, N/d = 0.71 per `phase_d_analysis.md:379`), not L5/correct. The corrected number is **1.02**, and the inflation mechanism is rank-deficiency in the within-class scatter, not a counting issue with carry classes. **Cite the L5/all numbers in the paper, not L5/correct.** L5/all has N=122,223 and N/d ≈ 29.8, fully reliable.

**Phase E — Residual hunting.** After projecting out the 43 known concept subspaces, the residual at L5/layer 16/all has **n_above = 444 eigenvalues above the Marchenko-Pastur edge** (Phase E master table; var_explained = 87.6% at this slice, λ_max = 8.02e-3, top correlation = `rel_error` with |ρ_s|=0.047, |r_p|=0.014 at the same slice — `phase_e_analysis.md:903–987`). Phase E summary text rounds this to "~440"; the master table is the authoritative number (444). Spearman vs. Pearson correlations between residual principal directions and named concept metadata across the full L5/wrong sweep are **|ρ_s| ≈ 0.07–0.10, |r_p| ≈ 0.000** (`phase_e_analysis.md:27, 1981`). The single largest |ρ_s| anywhere in Phase E is **0.082 (a_units, L5/wrong, layer 4** — `phase_e_analysis.md:1044, 1057`); no Spearman correlation exceeds 0.10 across the ~62,000 (residual_direction × named_label) pairs (`phase_e_analysis.md:1035`). The qualitative finding holds: Spearman ≫ Pearson is direct evidence of nonlinear residual content, but the effect size is small. The honest framing: **evidence for nonlinearity exists in the residual but is quantitatively weak**, consistent with either genuine interactions, within-group heterogeneity in superposed concepts, or finite-sample Tracy-Widom tail behavior. The paper should not claim the residual is the headline.

**Phase F — Between-concept geometry.** Principal angles for **42,049 concept pairs**. **39,525 pairs (94%) flagged as superposition** under the rule θ₁ < p5(random_baseline) − 10° (Phase F/JL §1, §3, §4). The −10° margin is empirical and conservative; sensitivity to the margin is not reported, so any borderline finding must acknowledge this.

**Phase JL — Distance preservation.** **43,921,634,388 (~43.9 billion) pairwise distances** computed (`phase_f_jl_analysis.md:1944, 2711`). The union subspace captures **Spearman 0.9942–0.9995** of the original distance structure across (level, layer) cells (`phase_f_jl_analysis.md:43, 1956`; min 0.9942 at L5/layer 6, max 0.9995 at L2/layer 31). The mathematical implication: anything missed by the linear pipeline is isotropic noise that does not affect distances. Caveat the paper should note: at **L5/layer 16/all the union subspace explains 87.59% of activation variance but preserves 98.98% of distance variance** (`phase_f_jl_analysis.md:1998`; the all-L5 distance correlation is 0.9972 at this slice, line 1926). Distance preservation is geometrically robust even with substantial unexplained variance, because the residual variance is spread isotropically across thousands of dimensions — at L5/layer 16, 12.4% of variance escapes the union subspace into 3,536 residual dimensions (~0.0035% per dimension), corresponding to only ~1.0% of distance structure (`phase_f_jl_analysis.md:613–642, 1998`).

**Phase G — Fourier screening.** **3,480 cells tested; 500 helix detections; 458 floor-saturated; 1 pure circle; all surviving FDR** (Phase G §7, §10). Cell = (concept × level × layer × population × period_spec). Three robust findings the paper rests on:

1. **Carries (carry_1 through carry_4) sit on generalized helices at their raw value periods**: 18 (carry_1, values 0–17), 27 (carry_2, values 0–26), 19 (carry_3, values 0–18), 10 (carry_4, values 0–9). These are **raw-value periods, not base-10 mod-10 periods**. Phase G tests three period specs per carry — `carry_binned`, `carry_mod10`, `carry_raw` — and the raw period spec wins (Phase G §12 and `phase_g_fourier.py:459-502`). carry_0 (values 0–8) sits on a 9-period structure and is detected at L4/correct only (see point 3).
2. **Operand digits at the `=` token do not show periodic structure** in the main pipeline. The Kantamneni & Tegmark pilot replication (`phase_g_kt_pilot.py`) confirms K&T on standalone integer tokens, so the discrepancy is **context-dependent**: the same model represents integers helically in one context and non-helically in another.
3. **Middle answer digits (`ans_digit_1_msf`, `ans_digit_2_msf`, `ans_digit_3_msf`) — where the model fails most — have neither linear subspaces (Phase C floor at 0.838 for these in correct population) nor periodic structure (Phase G, no helix detected)**. Edge digits (leading and trailing) do.

The two-slice "correct-only" findings that motivate the functional claim:
- `carry_0 / L4 / correct / carry_raw` has **18/18 detections** in a population of ~2,897 correct L4 examples; 0/18 each in `all` and `wrong` populations (Phase G §13g). The previous version of this plan said "4,197 samples" for this slice, which conflated the L5 correct count (4,197) with the L4 correct count (2,897). Corrected here.
- `ans_digit_5_msf / L5 / correct / digit` has **18/18 detections** in 4,197 L5 correct examples (`phase_g_analysis.md:165–167`: "In the L5 correct population (N=4,197), ans_digit_5_msf (the ones digit of the product) is detected as a helix at 18/18 cells — every layer, every basis"). The `all` and `wrong` populations at L5 do not show the same density of detections for `ans_digit_5_msf`.

These two cells are the structural backbone of the "helix correlates with success" finding. Both rest on small populations that are not difficulty-matched against the much larger wrong populations (e.g., L5/wrong has 118,026 examples versus L5/correct's 4,197), and B.9 below addresses the difficulty confound directly.

**Activation infrastructure (technical correction).** Activations are stored as **float32** (not float16, as previously stated). 122,223 × 4096 × 32 layers × 4 bytes ≈ 64 GB; total disk for all five levels and all 32 layers is around 49 GB at L5 plus smaller per-level storage at L1–L4 (`config.yaml:42`). All compute estimates downstream assume float32 and Llama 3.1 8B (32 layers, 4096 hidden, SiLU activations). The exact Hugging Face model identifier should be added to `config.yaml` and to the paper's reproducibility appendix; the local mount path alone is not citable.

**Accuracy table (corrected against `labels/analysis_summary.json`).**

| Level | n_problems | accuracy | n_correct | n_wrong | notes |
|-------|-----------:|---------:|----------:|--------:|-------|
| L1 (1×1) | 64 | 100.0% | 64 | 0 | |
| L2 (1×2) | 4,000 | 99.825% | 3,993 | 7 | wrong examples are precious |
| L3 (2×2) | 10,000 | 67.20% | 6,720 | 3,280 | |
| L4 (3×2) | 10,000 | 28.97% | 2,897 | 7,103 | |
| L5 (3×3) | 122,223 | **3.434%** | **4,197** | 118,026 | stratified sample of the 810,000-problem (a, b ∈ [100, 999]) input space (`generate_l5_problems.py`); the "~6%" figure refers to the full unstratified 810K population's accuracy, **not** this stratified subset — both numbers are correct under their respective definitions |

The correct framing in the paper: **"the model genuinely fails at L5 (3.4% accuracy on the stratified 122,223-problem sample; ~6% on the full 810K input space)"**. The 3.434% is what every Phase A–G analysis is computed against; the ~6% is the population-level rate before carry-balanced stratification was applied. The paper should be explicit that the experiments run on the stratified sample, since the stratification was deliberate (carry-balanced for downstream Fourier work) and any aggregate accuracy claim that conflates the two numbers will be wrong.

### A.3 What we have proved

In one paragraph: **the linear pipeline is necessary but not sufficient.** It identifies the rooms (linear subspaces, Phase C). It captures the pairwise geometry (Phase JL). It maps the overlap between rooms (Phase F superposition). It catalogues low-variance concepts that pure variance methods miss (Phase D LDA). It quantifies how much organized structure remains beyond what the registry captures (Phase E residual). Then Fourier reveals that *inside* the rooms, certain concepts (carries especially) are arranged on circles or helices, while others (middle answer digits, operand digits at `=`) are not. The shape inside the room matters, and the shape correlates with where the model succeeds versus where it fails.

Fourier only tests periodic structure. For arbitrary smooth manifolds — for the actual shape of the carry helix with uncertainty bars, for non-periodic curvature, for testing whether the structure is causally used — we need stronger tools. That is what Part B is about.

### A.4 What we have not proved (the audit findings we owe the reader)

Every result so far is **observational**. We have looked at activations and computed statistics on them. We have not changed anything inside the model and observed a downstream effect. Without causal intervention, every finding is a correlation between geometry and behavior, not a mechanism.

The `correct` populations at L4 (2,897) and L5 (4,197) are small and **not difficulty-matched** to the much larger `wrong` populations. The two slices that drive the "functional" claim (`carry_0 / L4 / correct` and `ans_digit_5_msf / L5 / correct`) could partly reflect that easier problems both get answered correctly *and* have cleaner geometry, regardless of whether the helix does any computational work. B.9 addresses this directly.

Phase F shows 94% of concept pairs are in superposition. When Phase G runs Fourier inside a subspace shared by multiple concepts, the centroids carry a mixture of those concepts. The carry_1 helix might be carry_1's own structure, or it might be a structure inherited from concepts that share the same axes (col_sum_1, partial products with similar value ranges). B.3 (orthogonalization) is a partial control; the remaining contamination from concepts outside the 43-concept registry would require sparse-autoencoder decomposition and is left as future work.

Centroid averaging hides within-group variation. If `carry_1 = 5` actually splits into sub-clusters based on operand magnitude, the centroid sits between them and Fourier sees a smeared average. The single-example projection plots from Phase G (the code generates these for null-control concepts; see `phase_g_fourier.py:2041-2104`) have not been systematically reviewed; B.2 addresses this.

p-floor saturation: 458 of 500 helix detections (91.6%) hit the permutation null floor of p ≈ 0.001 (1/1001 with 1,000 permutations). This means the test statistic exceeds *every* permutation sample, so we know the result is significant but we do not know the effect size precisely; the true p-values may be orders of magnitude smaller. The paper should acknowledge this as a known property of the screening test, not a quirk to hide.

Phase E's residual evidence is real but quantitatively weak. **Spearman correlations top out at 0.082**; Pearson is essentially zero. The claim should be "non-linear correlations exist but are not the headline" rather than "the residual is dominantly nonlinear."

Pre-registered Results-section treatment of Phase E (NEW): the paper places the Phase E nonlinear-residual finding in the methods/limitations discussion, **not the Results headline**. Specifically: a single sentence in §4 (Results, Claim 1 — linear pipeline finds the rooms) acknowledges that 12.4% of L5/layer-16 variance escapes the union subspace and that the largest non-linear correlation across the residual sweep is |ρ_s| = 0.082. The detailed Phase E table goes in the appendix. The qualitative finding "Spearman > Pearson is suggestive of nonlinear residual content" is reported but not used as evidence for any of Claims 2–7. This is the honest framing: Phase E *exists* as a published result on the full 122K population (Tier 1) and is referenced as background context for why the curated-set non-linear methods (Tier 2) are the right next step; it is not itself a headline finding of this paper.

Phase D's L5/correct LDA eigenvalues are inflated by the rank-deficiency of the within-class scatter at N/d ≈ 1.02 (4,197 / 4,096; corrected from the previous "0.71" which was actually L4/correct). **Cite L5/all carry eigenvalues (0.74–0.95), not L5/correct.**

These are the limitations the next phase addresses. Some can be fully resolved (orthogonalization, within-group PCA, difficulty matching). Some have to be acknowledged honestly (superposition contamination from outside the registry, centroid averaging at the smallest populations). None of them invalidate the work, but the paper's framing has to respect them.

### A.5 Why we now pivot to a curated set (the mathematical argument, not the stylistic one)

The data scale that powered Phases A through G — 122,223 samples at L5 — is not the right scale for what comes next. The reason is mathematical, not stylistic.

Exact Bayesian Gaussian Process Latent Variable Models scale as **O(N³)** for the marginal-likelihood gradient at each optimization step, and a Bayesian fit needs many such evaluations. At N = 122,000 that is approximately 1.8 × 10¹⁵ floating-point operations *per evaluation*. Even on a high-end GPU, this is hours per gradient step and tens of GPU-days per fit. Across 17 unique concepts × 3 populations × 9 layers × 3 kernels, the full grid of fits is several orders of magnitude beyond what the project can run.

The sparse-GP escape (FITC, VFE, SVGP at O(N M²) with M ≪ N inducing points) is genuinely costly to use *correctly* in this setting. Hauberg's argument in `papers/bayes_paper.md` is that probabilistic methods recover correct geometry because their decoder uncertainty grows away from data, which inflates the metric in low-density regions and bends geodesics back toward the data manifold. Sparse approximations break this property unless the inducing points are placed and the kernel bandwidth is tuned with care; specifically, Hauberg argues that geometry-preserving sparse inference requires bandwidth adaptation that scales with local density, which is exactly the property the inducing-point summary loses. With careful tuning sparse can work, but the tuning cost is itself substantial and paper-time is finite.

So the curated set is not a stylistic preference. It is **the only path to running the principled version of GPLVM on this problem**. Sparse-GP shortcuts would undercut the very Hauberg argument we cite when we justify GPLVM in the first place. This belongs in the methods section as a defense of the sample-size choice; a reviewer who asks "why N ≈ 5,000–8,000 and not all 122K?" gets a one-sentence answer: exact Bayesian inference requires it, and exact inference is what the methodological argument depends on.

A second, weaker reason: the paper's argument is methodological. We are demonstrating that a pipeline of methods works. Demonstration does not require running every method on every example. It requires a carefully chosen set that **covers the variety the paper makes claims about** while staying small enough to do the precise per-example work.

The pivot for the remaining steps: a curated case-study set in the **5,000–8,000 problem range** (revised upward from the previous version's 400, for reasons explained in B.1), designed to cover all 17 unique concepts at L1–L5, all carry values with at least 30 problems per value where mathematics permits, both correct and wrong outcomes, and difficulty-matched controls. Same set, every method. Reproducible, distributable, citable.

The large-N analyses (Phase C, Phase F, Phase JL, and the Phase G permutation null) stay as published. They are the population-level statistics. The new methods run on the curated set.

### A.6 The two-tier evidence structure

The framing pivot creates a two-tier evidence structure that should be made explicit in the paper.

**Tier 1 — Population-level statistics (large-N).** Phase C/D/E/F/JL and the Phase G permutation null. These cover all 122K L5 problems and tens of thousands of subspace pairs. They establish the existence of stable concept subspaces, document superposition, confirm distance preservation, and screen for periodic structure across the registry. Their job: give the reader confidence that the underlying linear and Fourier geometry is real and broadly distributed across the model's representations.

**Tier 2 — Curated case-study analysis (small-N, deep).** GPLVM, RBF VAE (optional, as a scalable surrogate), persistent homology, orthogonalization controls, subspace ablation, helix rotation, difficulty-matched validation. These run on the curated set. Their job: characterize the non-linear structure precisely, prove cross-method consistency, and establish causal mechanism for at least one concept.

The paper organizes around this two-tier structure: one section per tier, with cross-references between them. This addresses an objection a careful reviewer might raise — that the curated set is too small to support general claims — by clarifying that **general claims come from Tier 1 and depth claims come from Tier 2**. The two tiers do different jobs and are reported separately.

### A.7 The framing question reviewers will ask first

Reviewers will lead with: "Is this a paper about Llama doing multiplication, or a paper about a methodology?" The current framing is methodology-first. The paper has to defend that framing.

The defense:

> "Multiplication in Llama 3.1 8B is the testbed because it has exact intermediate labels and a clean correct/wrong signal. The pipeline of methods we describe — linear, Fourier, probabilistic-manifold, topological, causal — is general and applies to any task with a similar structure (other arithmetic operations, structured reasoning, classification with intermediate labels). We use multiplication to validate the pipeline empirically; we expect future work to apply it to other domains."

The reviewer who wants the methodology angle will accept the framing. The reviewer who wants a multiplication-only paper has been told what they will not find here. Either is fine; the framing is honest in both cases.

### A.8 Stratification caveat

One subtle but consequential property of the curated set: it is built by **stratification across the (concept, value) grid**. This over-represents rare cells relative to their natural frequency. That is the right design for **geometric** questions:

> "What shape is the carry_1 manifold?"

— because the geometry is a property of the value-space structure, not of how often each value occurs. We want every value of carry_1 well-represented so that the shape can be estimated.

It is the **wrong** design for **distributional** or **frequency** questions:

> "How often does the model use the helix in practice?"

— because stratification has destroyed the natural usage frequencies. A claim like "the helix is engaged 80% of the time" cannot be inferred from a stratified set; it requires the original 122K population.

The paper will be explicit about this distinction. **Stratified sample for geometry; full population for distributional claims.** Tier 1 numbers (existence, prevalence, frequency) come from 122K; Tier 2 numbers (shape, dimensionality, topology, causal role) come from the curated set. A reviewer who confuses the two and accuses us of overgeneralizing has been answered in advance.

### A.9 What still needs to be built (engineering inventory)

The Phase A–G infrastructure is in place. None of the Part B engineering exists yet. Before any Part B step can run, the following code, data, and infrastructure has to exist. This list is the prerequisite checklist for execution; nothing in Part B should be started before its prerequisites are satisfied.

**Reproducibility prerequisites (must be done before publishing any artifact):**
- **Hugging Face model identifier** in `config.yaml`. The current `config.yaml:5–6` lists only the local mount path (`/data/user_data/anshulk/arithmetic-geometry/model`). The exact HF identifier of the Llama 3.1 8B base/instruct variant used must be added so a reader can reproduce. The mount-path-only entry is not citable.
- **Tokenizer hash, transformers version, torch version, CUDA version, GPU model** committed to a `reproducibility.yaml` so any environment drift is detectable.
- **Frozen random seeds** for every Part B step (curated-set selection, GPLVM init, ARD init, RBF VAE init, persistent-homology permutations, ablation random-subspace draws, helix calibration). Pre-register every seed used in an appendix.
- **Activation-extraction script provenance**. `extract_number_token_acts.py` and the standard activation pipeline must be checksummed; the specific sample of 122,223 L5 problems is reproducible from `generate_l5_problems.py` with its seed, but the activations themselves are large and a checksum index is the canonical artifact.

**Code that does not exist yet (target locations indicated, none implemented):**
- **`build_curated_set.py`** — Pass 1 difficulty stratification, Pass 2 concept coverage, Pass 3 validation; outputs `curated_set_v1.json` (B.1).
- **`phase_h_within_group.py`** — within-group PCA + Hartigan dip + within-vs-between ratio + k-NN connectivity diagnostic (B.2).
- **`phase_h_orthogonalize.py`** — orthogonalization wrapper that loads `basis.npy` files, stacks via QR, produces orthogonal-complement projector, and re-runs the Phase G Fourier conjunction test (B.3).
- **`phase_i_gplvm.py`** — exact Bayesian GPLVM with ARD, three kernels, fixed Phase G period priors, two-seed convergence check, geodesic computation under E[M(z)]; toy-data validation harness for circle, helix, isotropic Gaussian, two concentric circles (B.4 + B.4.6).
- **`phase_i_rbf_vae.py`** — conditional, only built if GPLVM proves unstable; Hauberg σ⁻¹-via-positive-RBF decoder precision (B.5).
- **`phase_j_persistent_homology.py`** — `gudhi` or `ripser` based H₀/H₁ persistence diagrams with permutation null and 1.5× sensitivity sweep (B.6).
- **`phase_k_ablation.py`** — forward-pass projector for subspace zeroing, random-Grassmannian generator, irrelevant-concept control, per-layer ablation curves (B.7).
- **`phase_k_helix_rotation.py`** — helix calibration (circle fit, R² gate), Δθ_step computation, rotation-on-helix forward-pass intervention, random/off-axis/sign-flip controls (B.8).
- **`phase_k_difficulty_match_rerun.py`** — Phase G re-runner constrained to the curated-set matched pairs (B.9).
- **`phase_l_cross_method_table.py`** — joins outputs of B.4 through B.9 into the cross-method consistency table (B.10).
- **`phase_l_case_study_figure.py`** — five-panel figure assembler (B.11).

**Existing code that needs reuse (not new code):**
- `phase_g_fourier.py` — the Fourier conjunction test. Reused inside B.3 (orthogonalized inputs) and B.9 (matched-pair inputs). Keep it stable; do not modify in place.
- `phase_c_subspaces.py` — Phase C `basis.npy` outputs are inputs to B.3, B.4, B.7. Path: `/data/user_data/anshulk/arithmetic-geometry/phase_c/subspaces/L{level}/layer_{layer:02d}/{population}/{concept}/basis.npy`.
- `phase_d_lda.py` — Phase D `merged_basis.npy` outputs are sensitivity inputs for B.4 (alternative input subspaces).

**Tier-1 statistics (already published, do not re-run):**
- Phase C/D/E/F/JL/G outputs on the full 122K L5 population. These are stable. Part B never re-derives them; it only reuses their stored bases and stored centroids.

The dependency graph in B.13 shows the order in which Part B steps unlock each other; this section A.9 names the actual code files that have to be authored, and confirms that the foundation (Phases A–G) is complete and not in scope for further engineering. **Nothing in Part B is implemented.** That is the honest baseline.

---

## Part B: Remaining steps (no timeline)

Each step is described with: (1) what it does, (2) why it is necessary, (3) configuration choices grounded in the cited papers, (4) what gets extracted, (5) pre-registered thresholds and decision criteria, (6) failure modes and pre-registered fallbacks. Time estimates have been removed; the work proceeds ad hoc and the dependencies are tracked in B.13.

### B.1 Step 1 — Build the curated set

**STATUS: COMPLETE (Apr 25, 2026).** Output at `/data/user_data/anshulk/arithmetic-geometry/curated/curated_set_v1.json` (22 MB, 8,264 problems, sha256 in build log). Build script `build_curated_set.py`. Full coverage report at `docs/curated_set_coverage_report.md` (1,589 lines). Build log at `logs/build_curated_set.log`. Plan file at `/home/anshulk/.claude/plans/b-1-step-1-deep-comet.md`. Headline outcomes: 1,000 + 1,400 = 2,400 difficulty-matched pairs; 0 unmatched at both L4 and L5; 0 below-floor non-documented cells; 19 below-floor documented cells (pool-limited rare carries + 3 mathematically excluded operand-tens=0 cells); runtime 181.6 s; mean correct−wrong differences across the three matching axes near zero at both L4 and L5. Pass-by-pass details in §4 of the coverage report.

This is the foundation for every subsequent experiment. Get it right and the rest is straightforward. Get it wrong and every downstream finding is suspect.

#### B.1.1 Coverage requirements

The curated set has to satisfy these coverage requirements simultaneously:

1. **All 17 unique concepts represented** — operand digits (`a_units`, `a_tens`, `a_hundreds`, `b_units`, `b_tens`, `b_hundreds`), answer digits (`ans_digit_0_msf` through `ans_digit_5_msf`), and carries (`carry_0` through `carry_4`). For multi-level analyses, every concept must appear at every level where it is mathematically defined. (The "43 concepts" reference in the previous plan double-counted concept-level instances; the underlying registry has 17 unique names per `phase_g_fourier.py:56-74`. Both numbers are correct under their respective definitions; the paper should pick one and stick with it. We use 17.)
2. **All digits 0–9 represented at every digit position** — except leading positions where 0 is mathematically excluded (no leading zeros). Coverage statistics for leading digits go from 1 to 9, not 0 to 9.
3. **All carry values represented at every carry position with ≥ 30 problems per value** where mathematics permits. carry_0 has 9 values (0–8), carry_1 has 18 (0–17), carry_2 has 27 (0–26), carry_3 has 19 (0–18), carry_4 has 10 (0–9). 30-per-value is the minimum the GPLVM ARD pruning needs to converge cleanly; it is also the threshold under which the Phase G permutation null becomes underpowered.
4. **All four difficulty levels represented (L2, L3, L4, L5)**, with both correct and wrong outcomes at every level. **L1 and L2 are excluded in the executed build** — L1 is uniformly correct (n=64) and L2 has only 7 wrong examples (model accuracy 99.825%) so neither supports cell-level or matched-pair claims; no concept is unique to L2 (every L2 concept reappears at L3+).
5. **Difficulty-matched correct/wrong subsets at L4 and L5** — the single most important coverage requirement. For every L5 correct example, there must be an L5 wrong example with the same magnitude tier (small, medium, large operands), the same approximate carry count, and the same answer length. Pair them. Same construction at L4. This kills the difficulty confound for B.9.
6. **Joint coverage check** — for each pair of concepts that the paper makes a claim about, the joint-distribution table should have no empty cells with < 5 examples. (Marginal coverage alone can leave structural gaps; joint coverage catches them.)
7. **No duplicates** — every (a, b) pair appears at most once within a level. (The earlier "no near-duplicates within Hamming ≤ 1" rule was relaxed to exact-uniqueness; the tier-based stratification already prevents digit-pattern overrepresentation. The build uses Hamming = 0.)

#### B.1.2 Target size — resolved budget (executed Apr 2026)

The plan targeted 5,000–8,000 problems; the executed build (`build_curated_set.py`, output `/data/user_data/anshulk/arithmetic-geometry/curated/curated_set_v1.json`) reaches the upper end of that corridor. Saved L1/L2 budget was reallocated proportionally to L3, L4, L5. Plan and execution log: `/home/anshulk/.claude/plans/b-1-step-1-deep-comet.md` and `logs/build_curated_set.log`.

| Level | Correct | Wrong | Total | Pool used | Pool size | Notes |
|-------|--------:|------:|------:|----------:|----------:|-------|
| L1 (1×1) | — | — | 0 | — | 64 | excluded — uniformly correct |
| L2 (1×2) | — | — | 0 | — | 4,000 | excluded — only 7 wrong examples |
| L3 (2×2) | 1,200 | 1,201 | 2,401 | 31% / 64% | 3,859 / 1,874 (after dedup) | tier-stratified + concept-fill; pool dedup dropped 4,267 duplicate (a,b) rows |
| L4 (3×2) | 1,000 | 1,841 | 2,841 | 37% / 28% | 2,723 / 6,691 (after dedup) | difficulty-matched; 1,000 pairs (993 strict, 7 relaxed); zero matching loss |
| L5 (3×3) | 1,400 | 1,614 | 3,014 | 33% / 1.4% | 4,197 / 118,026 | difficulty-matched; 1,400 pairs (1,399 strict, 1 relaxed); zero matching loss |
| **Total** | **3,600** | **4,656** | **8,256** | — | — | includes 255 Pass-5 concept top-up rows |

**Matched-pair budget realized: 1,000 pairs at L4 + 1,400 pairs at L5 = 2,400 matched pairs.** The 8,256 final total exceeds the planned 8,000 because Pass 5 added 41 L4 + 214 L5 unmatched rows to bring rare-cell counts up to the per-(concept, value) floor of 30. All 17 unique concepts (registry: `phase_g_fourier.py:56–74`) appear at every applicable level. Coverage report: `docs/curated_set_coverage_report.md`.

Below-floor non-documented cells: **0**. Documented hard-ceiling cells (pool itself has < 30 examples for that value): **19** — primarily extreme carry values at L4/L5 (carry_2 ≥ 24, carry_3 ≥ 14, carry_4 ≥ 7) plus three mathematically excluded cells (a_tens=0 / b_tens=0 at L3/L4 where operand range is [10, 99]).

Compute feasibility: exact Bayesian GPLVM at N = 8,256 (the executed total; the relevant per-level Ns are 2,401 / 2,841 / 3,014) has a dominant cost of one Cholesky factorization at ~N³/3 ≈ 9.4 × 10⁹ flops per evaluation at the L5 sub-population scale; aggregated per-fit cost stays well under 6,000³/3 ≈ 7 × 10¹⁰ flops because GPLVM fits run per (concept × population × layer × kernel) on the relevant level only, not on the full 8,256-row union (`+` derivatives ≈ 2–3× more). On an A100, fp64 GEMM peak is ~9.7 TFLOPS and fp64 Cholesky/triangular solves achieve roughly 5–8 TFLOPS in practice, giving ~0.5–2 seconds per gradient step including derivative passes; a fit of a few hundred L-BFGS steps takes a few minutes. The full grid — 17 concepts × 3 populations × 9 layers × 3 kernels = 1,377 fits (≤ 1,400) — is then on the order of **days of single-GPU time, not hours, but finite and tractable**. (Earlier versions of this plan quoted "0.7 s per step" using bf16-peak A100 throughput; bf16 is not the right metric for fp64 Cholesky on a GP and the 0.7-second figure was off by roughly an order of magnitude on the optimistic side. The corrected estimate above uses fp64 effective throughput.) If the budget is tight: pull back to N = 5,000 (Cholesky cost drops to ~4.2 × 10¹⁰ flops, ~40% faster), run only carries plus `ans_digit_5_msf` plus operand units, or restrict to the 4–6 most informative layers.

The scaling does NOT work at N = 122K (Cholesky cost ~6 × 10¹⁴ flops per evaluation; even at 5–8 TFLOPS effective that is ~hours per gradient step, weeks per fit, unaffordable). This is the hard constraint behind the curated-set pivot.

ARD-convergence consideration: ARD with d_max = 5 (see B.4) needs roughly N ≥ 25 × d_max² ≈ 125 examples per concept to converge cleanly without local-minimum traps; at the carry_2-per-value floor of 30, this is met. Smaller per-value cells are reported with the caveat that ARD pruning may be unstable.

#### B.1.3 Selection algorithm

Selection runs in three passes:

1. **Pass 1 — Difficulty stratification (L4, L5).** From the existing 122,223 L5 problems and 10,000 L4 problems, partition by (correct/wrong) × magnitude tier × carry-count tier × answer-length. Sample per cell to fill the L4 and L5 budgets in the table above. This guarantees the difficulty-matching property.
2. **Pass 2 — Concept coverage.** From the L2/L3/L4/L5 populations, select problems that fill in coverage gaps left by Pass 1. Specifically: any digit value at any operand position that is underrepresented gets new problems sampled until the floor is met. Any carry value that is underrepresented gets new problems sampled until ≥ 30 per value (where mathematics permits).
3. **Pass 3 — Validation and trimming.** Compute final coverage statistics. Drop any duplicate (a, b) pairs introduced across passes. Verify difficulty-matching constraints at L4 and L5. Check joint coverage tables for the (concept_i, concept_j) pairs the paper makes claims about — every cell must have ≥ 5 examples or be marked as an explicit gap with mathematical justification.

The selection script outputs a single JSON file: `curated_set_v1.json`. It contains `(a, b, level, correct_flag, magnitude_tier, carry_count_tier, answer_length, matched_pair_id_if_any)` for each of the 6,000 problems and the seed used for reproducibility.

#### B.1.4 Validation report

Before any experiment runs on the curated set, produce a coverage report:

- Number of unique digits represented at each operand and answer position.
- Number of unique carry values represented at each carry position, with the per-value count.
- Distribution of (correct, wrong) by level.
- L4 and L5 difficulty-matching statistics: mean magnitude difference between matched correct/wrong, mean carry-count difference, mean answer-length difference, with permutation-test p-values that the matched pairs are not distinguishable on the matching variables.
- Joint coverage tables for the (carry_i, col_sum_i) and (carry_i, partial_product) pairs the paper analyzes.
- List of concept-value cells with fewer than 30 examples and the mathematical reason.

Pin this report in the appendix.

#### B.1.5 Worked example: how the L5 matched pairs are constructed

Take a specific L5 correct example: a = 234, b = 789, product = 184,626. Operands have leading digits 2 and 7, magnitude tier "medium-medium". Carry counts: carry_0 = 3, carry_1 = 5, carry_2 = 6, carry_3 = 4, carry_4 = 1; total non-zero carries = 5 (high-carry tier). Answer length: 6 digits.

For this example, the algorithm searches the L5 wrong population for any (a', b') such that:

- a' is in the same magnitude tier as a (within ±100 of a's leading-digit pattern).
- b' is in the same magnitude tier as b.
- (a' × b') has the same number of non-zero carries (within ±1).
- (a' × b') has the same number of digits.

A match might be a' = 287, b' = 643 (model's prediction was off in the trailing digits — a single-carry-propagation error). This pair has nearly identical structural difficulty, but the model gets the first correct and the second wrong.

Run this matching for all 800 correct L5 examples in the curated set. The algorithm rejects any correct example for which no matched wrong exists in the underlying 122,223-problem pool. In practice this loses about 10–15% of candidates, which is fine.

The matched pairs are the controlled comparison in every causal and Fourier experiment that runs on the curated set. When we say "the helix is present in correct and absent in wrong at L5," we mean **among these matched pairs**, not among the unmatched populations. The reviewer's first question — "did you control for difficulty?" — has a one-sentence answer: yes, by exact pairing on three structural properties.

#### B.1.6 Coverage gaps we accept

Some coverage targets cannot be met:

- **carry_2 = 26 at L5** has only 11 correct examples in the entire 810K pre-screening input space. The curated set documents this as a hard ceiling.
- **carry_4 = 9** has 0 correct examples globally. The curated set cannot fix this; it documents the absence.
- **Operand digit 0 at any leading position** is mathematically impossible.
- **carry_2 in correct populations** generally has thin per-value coverage at the high end (≥ 20). The curated set takes what is available and notes the per-value floor in the report.

These exceptions are listed in the coverage report with explicit math. A reviewer who counts the cells will find every gap accounted for.

---

### B.2 Step 2 — Within-group PCA + disconnected-manifold diagnostic

Before running any non-linear method, settle two questions: (1) is centroid analysis sound, and (2) are the per-value clouds genuinely connected, or are they islands that a single GPLVM would smear?

#### B.2.1 Within-group PCA: what this does

For each (concept, value) pair — for example, all activations where carry_1 = 5 — run PCA on just those activations. Look at:

- The first eigenvalue. How much variance does the top within-group direction capture?
- The eigenvalue spectrum. Does it decay smoothly (one cluster) or have a gap (multiple clusters)?
- The within-group spread relative to the between-group spread. The ratio λ₁(within) / σ²(between centroids) tells us whether centroid analysis is misleading.

#### B.2.2 What we are testing

If the within-group structure is small (eigenvalue spectrum decays smoothly, top within-group direction captures < 10% of total variance), the centroid analysis is sound and the Fourier helix findings stay.

If the within-group structure is large or bimodal (clear gap in the eigenvalue spectrum, multiple sub-clusters), the centroid is hiding structure. The Fourier finding for that concept needs to be reinterpreted as a property of the average, not a property of individual examples.

#### B.2.3 Implementation

Run on the curated set only. For each (concept, value) pair, project activations onto the concept's Phase C subspace and run PCA. Save the eigenvalue spectrum and a 2D scatter plot of the within-group cloud.

#### B.2.4 Decision criteria (pre-registered)

For each (concept, value) cloud, compute three statistics:

- **Within-group concentration ratio (WGCR):** λ₁(within) / Σⱼ λⱼ(within). The fraction of within-group variance captured by the top within-group eigenvalue. WGCR > 0.5 suggests one dominant within-group axis (potentially a sub-cluster signal). WGCR < 0.3 suggests well-spread variation typical of a unimodal cloud.
- **Bimodality test:** Hartigan's dip statistic on the projection of the cloud onto its top within-group axis. A significant dip (p < 0.05) means the cloud has a multimodal distribution along that axis.
- **Within-vs-between ratio:** λ₁(within) / σ²(between centroids). If this ratio exceeds 0.3, the within-group structure rivals the between-group structure, and centroid analysis becomes suspect.

Pre-registered interpretation:

- **Green:** WGCR < 0.3, dip test passes (no bimodality), ratio < 0.1. Centroid analysis is sound. Report and move on.
- **Yellow:** intermediate values. Centroid analysis is mostly sound but the within-group structure should be discussed in the limitations.
- **Red:** WGCR > 0.5, significant bimodality, ratio > 0.3. Centroid analysis is misleading for this concept. Either the concept needs sub-cluster analysis or the Phase G result should be flagged.

A reasonable expectation: most operand digits and carries land in green, a few middle answer digits land in yellow or red. The paper acknowledges yellow/red cases honestly.

#### B.2.5 Disconnected-manifold diagnostic (NEW)

A single GPLVM (B.4) assumes one connected manifold. If carry values actually cluster into disconnected clumps (a lookup-style representation rather than a continuous circular geometry), a single GPLVM smears them and produces a fitted manifold that is not the underlying structure.

Test before fitting: build the k-nearest-neighbor graph (k = 10) on the centroids in their Phase C subspace. Count connected components.

- **One connected component → continuous manifold hypothesis is admissible.** Proceed to GPLVM.
- **Multiple connected components → disconnected manifold (lookup-style or shattered).** Switch to a mixture-of-GPLVMs (one GPLVM per component) or to a topological-only analysis (Step B.6) and skip GPLVM for this concept.

Pre-registered interpretation: if more than one carry concept shows disconnection at this stage, the paper's main claim shifts from "carries live on continuous helices" to "carry values group into discrete clusters with periodic *labelling*, but not periodic *geometry*." Both claims are publishable; the experiments distinguish them.

This diagnostic is also a way the wrong-answer manifold could differ from the correct-answer manifold qualitatively — not just lower density, but **shattered**. That's a stronger qualitative claim than "lower geodesic agreement" and is worth reporting in B.9.

---

### B.3 Step 3 — Orthogonalization control for superposition

This is the cheap, important superposition control. It does not solve superposition. It tells us how much of the helix finding survives when we project out known correlates.

#### B.3.1 What this does

For each carry concept, identify its known algebraic correlates from Phase B. carry_1's correlates at L5 include col_sum_1, pp_a1_x_b0, pp_a0_x_b1. Project the activations onto the orthogonal complement of these correlates' subspaces (the union of their Phase C bases). Re-run Phase G's Fourier conjunction test on the orthogonalized activations.

#### B.3.2 What we are testing

If the helix survives orthogonalization at comparable strength, the structure belongs to carry_1's own representation. The superposition concern is real but does not undermine the headline finding.

If the helix collapses (FCR drops substantially, p-values lose significance), the structure was inherited from one of the correlates. We then characterize *which* correlate's subspace carried the periodic structure and report this as a finding about how the model organizes its computation.

Either result is informative. The first is the easier story to tell. The second is more interesting scientifically — it would mean the model doesn't separate carry_1 from col_sum_1 representationally, and the helix lives in the shared computation.

#### B.3.3 Implementation

Run on the curated set. For each of carry_0 through carry_4:

1. Stack the Phase C bases of all relevant correlates into a single matrix `B_correlates` of size (d_corr × 4096), where d_corr is the total dimension of the correlate subspaces. Phase C bases are stored at `/data/user_data/anshulk/arithmetic-geometry/phase_c/subspaces/L{level}/layer_{layer:02d}/{population}/{concept}/basis.npy` (file is `basis.npy`, NOT `merged_basis.npy`; the latter is the Phase D output that fuses Phase C + LDA — `phase_c_subspaces.py:809–810, 726–727`). For carries where Phase D's discriminant directions add structurally distinct dimensions, the orthogonalization can also be run against the Phase D `merged_basis.npy` as a sensitivity check. Load each basis, stack, and orthogonalize via QR.
2. Compute the orthogonal complement projection: `P⊥ = I - B_correlatesᵀ (B_correlates B_correlatesᵀ)⁻¹ B_correlates`. (Equivalently, use SVD on `B_correlates` and take I minus the projection onto the row space.)
3. For each activation h, compute `h_orthogonal = P⊥ h`.
4. Project `h_orthogonal` onto the carry concept's own subspace and run Phase G's Fourier conjunction test.

Compare:

- two_axis_fcr (raw vs orthogonalized)
- helix_fcr (raw vs orthogonalized)
- p_value (raw vs orthogonalized)
- visual centroid plot (raw vs orthogonalized)

Produce a single table summarizing the change for each carry concept.

#### B.3.4 Decision criteria (pre-registered)

- **Survives:** < 30% drop in FCR. Structure is the carry's own.
- **Inherited:** > 50% drop in FCR or loss of significance. Structure is shared with the correlates.
- **Ambiguous:** drop in [30%, 50%]. Report both interpretations.

#### B.3.5 What this is not

Orthogonalization handles only **known** correlates. If carry_1 is in superposition with some unmeasured concept (a feature outside the 17-concept registry), orthogonalization cannot remove that interference. A full SAE-based decomposition would be needed; that is Paper 4 territory.

The paper acknowledges this: "Our orthogonalization control removes interference from algebraically related concepts. Interference from concepts outside our registry — which would require dictionary-learning approaches like sparse autoencoders — is left as future work."

---

### B.4 Step 4 — GPLVM as the primary non-linear manifold method

This is the central methodological contribution. Done well, this is the experiment that justifies the paper's title. Done poorly, it is a circular fit that adds nothing Fourier did not.

The version of this section in the previous plan made several choices that the validation against `papers/bayes_paper.md` and the user's iteration found wanting. The corrections are folded in here.

#### B.4.1 What it is

A Gaussian Process Latent Variable Model places a Gaussian Process prior over the mapping from a low-dimensional latent space (the manifold) to the high-dimensional observation space (the activation subspace). The GP gives uncertainty over the mapping. Far from training data, uncertainty grows; geodesics through high-uncertainty regions get penalized. This is Hauberg's argument for why probabilistic methods recover correct geometry while deterministic methods (autoencoders, kernel ridge regression) systematically distort it.

The expected metric tensor on the latent space (Hauberg 2018, Eq. 3.19 / `papers/bayes_paper.md`) is

    E[M(z)] = (1/D) · E[J(z)]ᵀ E[J(z)] + Σ(z)

where J is the Jacobian of the predictive mean of the GP, D is the observation-space dimension, and Σ is the predictive covariance scaled to give correct uncertainty contribution. (The previous version of this plan omitted the (1/D) normalization — it is small but matters for high-D consistency and for comparisons across concepts of differing subspace dimension.) Distances and geodesics on the latent space are computed under M.

#### B.4.2 Configuration choices and why each one matters

**Latent dimension (ARD-determined, not hand-picked).** The previous plan hand-picked d = 2 for digit-like concepts and d ∈ {2, 3} for carries. The cleaner framing: start with d_max = 5 and use Automatic Relevance Determination (ARD) lengthscales to let the data prune unused dimensions. If a carry comes back with three active dimensions (one linear + two periodic = helix), that **replicates** Phase G's helix detection without being told what to look for. If it comes back with five active dimensions, that's a Phase-G miss worth investigating.

This converts d from a hyperparameter to a result. ARD pruning rule: a latent dimension is "pruned" if its lengthscale exceeds the prior median by a factor of 100 (this is the scikit-learn / standard-ARD convention; **note**: the previous plan attributed this threshold to Hauberg, but the threshold is sklearn standard, not Hauberg's). The paper reports the ARD-pruned dimensionality per concept as a result.

**Kernel choice with Phase G periods as priors (not as hypotheses to compete against unknown periods).** Phase G has already told us the carry periods: 18, 27, 19, 10. Plugging these as priors into the periodic kernel is much stronger than treating period as an unknown. The kernel comparison stops being a search ("what period does the data prefer?") and becomes a falsification test ("does the data favor the Phase-G-predicted period over RBF?").

For each concept, fit three GPLVMs with kernels:

- **Periodic kernel** (Rasmussen & Williams 2006 standard form):
  `k_periodic(z, z') = σ² · exp(−2 sin²(π(z − z') / p) / ℓ²)`, with period p **fixed at the Phase G value** for that concept (18 for carry_1, 27 for carry_2, 19 for carry_3, 10 for carry_4, 9 for carry_0; 10 for digits where periodicity is hypothesized). Tests the circular hypothesis directly with the period prior fixed.
- **RBF kernel:** `k_RBF(z, z') = σ² · exp(−|z − z'|² / 2ℓ²)`. Tests the generic smooth-manifold hypothesis.
- **Periodic + linear kernel:** `k_periodic(z₁, z'₁; p) + k_linear(z₂, z'₂)` where z₁ is the angular axis and z₂ is the magnitude axis, period p fixed at the Phase G value. Tests the helix hypothesis explicitly.

Note that the previous plan's "compare three kernels with no period prior" is a search; the corrected protocol is a falsification of the Phase G periodic structure under the more flexible probabilistic framework. If the data does not favor the Phase G period, the Phase G finding is overturned by the more rigorous test.

**Bayesian model comparison via marginal log-likelihood.** Per-concept, report log p(X | Z, θ) for each kernel under fully Bayesian inference. The difference in log marginal likelihood is the Bayes factor in nats.

Citation caveat: `papers/bayes_paper.md` does **not** prescribe Bayesian model comparison via marginal likelihood as the kernel-selection method. This is our methodological choice. The justification is standard (Bayes factors, e.g., Kass & Raftery 1995) and is independent of Hauberg.

Threshold caveat (corrected): Kass & Raftery 1995 specify their evidence categories in **log₁₀ Bayes factors**, not natural log. Their thresholds are: log₁₀ BF ∈ (0.5, 1] = "substantial"; (1, 1.5] = "strong"; (1.5, 2] = "very strong"; > 2 = "decisive". Converted to nats (multiply by ln 10 ≈ 2.303): substantial ≥ 1.15 nats, strong ≥ 2.30 nats, very strong ≥ 3.45 nats, decisive ≥ 4.60 nats. The previous version of this plan called "5 nats" decisive — that is correct only at the lower edge of K&R's "decisive" tier (5 nats ≈ 2.17 log₁₀, just past the 2-log₁₀ boundary). The previous version's "10 nats decisive" was overstating K&R; 10 nats = 4.34 log₁₀ is well past decisive. Pre-registered for this paper:
- Δ ≥ 5 nats (≈ 2.17 log₁₀) = K&R "decisive" — the threshold the plan adopts for declaring a winning kernel.
- Δ ≥ 2.3 nats (≈ 1.0 log₁₀) = K&R "strong" — used as a secondary report tier where no kernel reaches decisive.
- Δ < 2.3 nats — kernel comparison reported as inconclusive.

**Inducing points: not used.** The whole point of the curated set is to enable exact inference. Sparse approximations would undercut the Hauberg geometry argument (A.5). On 6,000 points, exact inference is feasible.

**Initialization.** PCA initialization for latent coordinates. ARD lengthscales initialized to 1.0. Marginal likelihood optimized via L-BFGS for 500 iterations or until convergence. Run each fit twice with different random seeds for the latent coordinates and check that the marginal likelihoods agree within 1 nat; flag any concept where the two runs differ by more than that as "unstable" in the report.

**Hyperpriors.** Use weakly informative priors held constant across concepts so comparisons are apples-to-apples:
- Lengthscales: Gamma(1, 1)
- Signal variance: Half-Normal(0, 1)
- Noise variance: Half-Normal(0, 0.1)

**Subspace to project into before fitting.** Use the Phase C consensus basis as default. For sensitivity analysis, also fit on the Phase D LDA basis (especially relevant for low-variance concepts where LDA found cleaner subspaces) and on the Phase C+D merged basis. Report agreement.

**Disconnected-manifold gate.** Before fitting GPLVM for a concept, run B.2.5 on that concept's centroids. If the centroid graph has > 1 connected component, do **not** fit a single GPLVM — switch to mixture-of-GPLVMs or skip to topology-only analysis. This is pre-registered.

#### B.4.3 What to extract from each fit

For each (concept, kernel, layer, population), save:

- The marginal log-likelihood (the model comparison statistic).
- The latent coordinates of every example (with posterior variance).
- The expected metric tensor M(z) on a grid of latent points.
- The geodesic distances between every pair of class centroids (computed numerically by minimizing path length under M, using a discrete dynamic-programming or fast-marching method).
- The ARD-pruned latent dimensionality.
- Posterior samples of the latent coordinates for ~50 examples (used to display uncertainty bars in the case-study figure).

#### B.4.4 What this proves that Fourier did not

Fourier tells us "is there a circle here, yes or no, with strict conjunction." GPLVM tells us:

- The full posterior over the manifold shape with uncertainty bars.
- Whether the manifold is genuinely 1D (a circle), 2D (a torus, a sphere), or higher (via ARD).
- How curvature varies along the manifold (constant for a circle, variable for an ellipse).
- Whether the same concept has different manifold shapes in correct versus wrong populations (different active dimensions, different periods preferred, different topology).
- Quantitative comparison across kernel families via marginal likelihood.

The headline result we want from GPLVM: **for carry_1 in the correct L5 population, the periodic+linear kernel beats RBF by a margin of at least 5 nats in log marginal likelihood, with the period prior fixed at 18 (the Phase G value)**. That is the statement reviewers can attack and we can defend.

#### B.4.5 Pitfalls

GPLVM is non-convex. PCA initialization helps but the optimizer can land in local minima. Two-seed agreement is the convergence check.

Marginal likelihood under different kernel families is sensitive to hyperpriors. Keep hyperpriors fixed across all concepts.

The metric-tensor computation is sensitive to numerical conditioning. Use the (1/D) normalization. Add a small jitter (1e-6 on the diagonal) to the latent-space Gram matrix.

The Phase C subspace might exclude the dimensions that actually carry the helix — Phase C is a *linear* subspace identified by conditional covariance, and helical structure could live in a subspace that has zero conditional covariance with carry value. Sensitivity check: also fit on the union of Phase C carry subspace + a few additional Phase E residual directions (top eigenvalues).

**The Riemannian Laplace alternative.** Bergamin, Moreno-Muñoz, Hauberg, Arvanitidis (NeurIPS 2023, `papers/Riemannian_Laplace.md`) propose Laplace approximations on Riemannian manifolds as an alternative inference scheme. Their setting is weight-space rather than activation-space, and their analysis is for tanh activations (not SiLU as in Llama 3.1 8B). For our problem, the GPLVM with full Bayesian sampling is the primary tool. The Riemannian Laplace is a useful sanity check if GPLVM optimization is unstable; it is faster but more approximate. We use it as a fallback, not the primary method.

#### B.4.6 Synthetic validation before running on real data

Before any real-data fits, validate the implementation on toy data with known geometry:

- **Toy 1:** 100 points sampled uniformly from a 2D circle embedded in 9D space with Gaussian noise. Fit GPLVM with all three kernels (period prior set to the truth: 1 cycle on [0, 2π]). Periodic kernel should win by at least 10 nats; ARD should prune to one active dimension.
- **Toy 2:** 100 points sampled from a helix (circle in z₁-z₂, linear in z₃). Fit. Periodic+linear should win; ARD should give two active dimensions.
- **Toy 3:** 100 points sampled from an isotropic Gaussian. Fit. RBF should win or no kernel should win convincingly. ARD should prune all dimensions.
- **Toy 4:** 100 points sampled from two concentric circles (a torus cross-section). Fit. Periodic-single-circle should fail; the disconnected-manifold diagnostic from B.2.5 should flag the data as multi-component.

These four tests verify that the GPLVM pipeline can detect the structures we care about and reject the ones we do not. Without them, every result on real data is suspect.

#### B.4.7 The math, briefly

The GPLVM places a Gaussian process prior on the function f: Z → X mapping the latent space Z (dimension d ≤ d_max = 5) to the observation space X (the concept's Phase C subspace, dimension up to 18 for carry_2). The model is

    x_n = f(z_n) + ε_n, ε_n ~ N(0, σ²I)
    f ~ GP(0, k)

The marginal likelihood is

    p(X | Z, θ) = N(X | 0, K_ZZ ⊗ I_D + σ²I)

where K_ZZ is the Gram matrix at the latent coordinates and θ collects all kernel hyperparameters.

Training maximizes p(X | Z, θ) jointly over Z and θ via L-BFGS. The result is:

- A learned latent layout Z that is consistent with the manifold hypothesis encoded in the kernel.
- Posterior uncertainty on Z (via the Laplace approximation around the optimum or via fully Bayesian sampling).
- A predictive posterior at any new latent point z*: p(f(z*) | Z, X) Gaussian with mean and variance computed in closed form.
- The expected metric tensor `E[M(z)] = (1/D)·E[J]ᵀE[J] + Σ(z)`. Distances on the manifold are computed via geodesics under M.

The Hauberg point — the central reason we use this method — is that Σ(z) grows away from training data, which causes the metric to inflate in regions of low data density. Geodesics that cut through these regions get penalized, so manifold geometry is faithful to the data. Deterministic methods (autoencoders) have no such mechanism; their decoder smoothly interpolates through empty space.

#### B.4.8 What "comparing kernel families" means in practice

For carry_1 in the L5 correct population (period prior = 18 from Phase G):

1. Periodic kernel with p = 18, fit GPLVM, record log p(X | Z, θ).
2. RBF kernel, fit GPLVM, record log p.
3. Periodic + linear with p = 18 on the periodic axis, fit GPLVM, record log p.

Report the differences in log-likelihood. Under the corrected mapping (B.4.2 above): Δ ≥ 5 nats ≈ 2.17 log₁₀ = K&R "decisive"; Δ ≥ 2.3 nats ≈ 1.0 log₁₀ = K&R "strong"; below 2.3 nats = inconclusive.

Pre-registered statement: "We declare the manifold class supported by the data to be the kernel with the highest marginal likelihood by at least 5 nats (decisive under Kass & Raftery 1995 thresholds, log₁₀ Bayes factor ≥ 2.17). Where the gap is between 2.3 and 5 nats we report it as 'strong but not decisive' evidence. Where no kernel beats the others by 2.3 nats we report the result as inconclusive and treat persistent homology and the causal experiments as the deciding evidence for that cell."

For carries, we expect periodic+linear to win (this is the helix hypothesis with Phase G periods as priors). For middle answer digits where Fourier found nothing, we expect RBF or no kernel to win convincingly. For operand digits at the `=` position, we expect the same null result as Fourier. The GPLVM kernel comparison is a stronger version of the same test, and it is **falsificationist** under the Phase G priors: if the data does not support the Phase G period, the Phase G finding is rejected by a stronger test.

#### B.4.9 What GPLVM still cannot tell us (and why causation is non-optional)

Even with all four corrections (ARD, kernel priors from Phase G, disconnection diagnostic, exact Bayesian inference), GPLVM still requires that **the manifold hypothesis itself is correct**. If the model represents carries via something fundamentally non-manifold-like — say, a learned lookup table with no smooth interpolation — GPLVM will produce a manifold anyway, because that is what it is designed to do. It will be a manifold of bad fit (low marginal likelihood, high posterior uncertainty, possibly multiple modes flagged by the disconnection diagnostic), but a fit will exist.

The defense against this is causal: if the GPLVM-recovered manifold's geodesic structure predicts model output under intervention (B.7 and B.8), the manifold is real. If it doesn't, the manifold is a fitting artifact.

This is why the causal experiments are not optional. They are what distinguishes "we fit a manifold" from "the model uses a manifold." This point belongs prominently in the limitations section and in the discussion that frames the causal experiments.

---

### B.5 Step 5 — RBF VAE as a scalable surrogate (downgraded)

The previous plan framed RBF VAE as "cross-validation" that would corroborate GPLVM. That framing is misleading: GPLVM and RBF VAE share the same mechanism (uncertainty grows away from data → metric inflates → geodesics avoid empty regions → manifold geometry preserved). Their agreement is not independent evidence; it is expected by construction.

The honest framing is: **RBF VAE is the scalable surrogate**, used when GPLVM is infeasible.

#### B.5.1 When this step runs (and when it does not)

- **GPLVM converges cleanly on the curated set.** RBF VAE is supplementary. We optionally run it for sanity-of-implementation and report the agreement as "scale-validation" — the conclusions don't change when we use the cheap method. It does not appear in the headline figure.
- **GPLVM struggles (slow convergence, two-seed disagreement, too few accepted fits).** RBF VAE becomes the primary method. The paper switches to RBF VAE in the headline and reports GPLVM as the more rigorous-but-too-expensive alternative. This is the pre-registered fallback.

The point: do not over-claim a "two methods agree therefore truth" position when the methods are not independent.

#### B.5.2 What the real cross-method validation is

Two methods are genuinely independent of GPLVM and of RBF VAE:

- **Persistent homology (B.6).** Different mathematical apparatus — combinatorial topology, not Bayesian inference. Tests topological invariants (H₀, H₁) directly from the point cloud; does not depend on a kernel form or a parametric uncertainty model.
- **Causal interventions (B.7, B.8).** Different question entirely — does it affect outputs? Does not even ask "what is the manifold shape." The cross-method evidence between GPLVM and a successful causal intervention is the strongest thing this paper can claim.

These are elevated to primary independent validation; RBF VAE is demoted to "scalable surrogate, only invoked if GPLVM fails."

#### B.5.3 Configuration (only relevant if RBF VAE is invoked)

A Variational Autoencoder where the decoder's precision uses an RBF network in Hauberg's style.

**Encoder.** Two-layer MLP, 64 hidden units. Latent dimension d_max = 5 (matching GPLVM ARD).

**Decoder mean.** Two-layer MLP, 64 hidden units, mapping latent space to the activation subspace.

**Decoder precision (the RBF part).** Following Hauberg 2018 §5: σ(z)⁻¹ is modeled as a positive RBF network with one center per training example and learnable widths. Specifically:

    σ(z)⁻¹ = β₀ + Σᵢ wᵢ · exp(−|z − cᵢ|² / 2λᵢ²)

with wᵢ > 0 (enforced by softplus parameterization), λᵢ learnable, cᵢ initialized at the encoder's mean for the i-th training example. (**Note:** Hauberg uses σ⁻¹ via positive-RBF, not σ⁻² as the previous version of this plan stated; this is corrected here.) The key property: as z moves far from any training example, all the exp(...) terms decay, σ⁻¹ → β₀, and σ → 1/β₀. With β₀ small, σ is large in low-density regions; the Jacobian-based pullback metric inflates accordingly.

**Training.** Standard ELBO with KL warmup over the first 50 epochs. 500 epochs total. Adam at 1e-3. RBF center positions get a learning rate 10× smaller than weights to prevent center collapse early in training. (This is a common stabilization heuristic in geometric VAEs with parametric prior centers; Arvanitidis et al. ICLR 2018 motivate the importance of well-placed centers but do not prescribe the exact 10× ratio. Treat this as an engineering choice, not a paper-cited prescription. If the implementation diverges, reduce the ratio further or re-initialize centers via k-means on the encoder mean of the training set.)

**What we extract.** Same as GPLVM: latent coordinates, pullback metric J(z)ᵀJ(z), geodesic distances, ARD-pruned dimensionality (RBF widths that grow without bound effectively prune dimensions).

#### B.5.4 If both GPLVM and RBF VAE fail

The paper reports both negative results and switches the headline to **persistent homology + causal experiments** (B.6 + B.7/B.8). These methods do not depend on a specific manifold-fitting machinery and can support the paper's core claims without GPLVM/RBF VAE.

---

### B.6 Step 6 — Persistent homology as primary independent validation (elevated)

Topology is harder to fake than circular geometry. A linear ramp aligned with a Fourier basis can produce a high `two_axis_fcr` without being a real loop. Persistent homology computes topological invariants directly from the point cloud.

The previous plan placed persistent homology as a supplementary topology check after GPLVM. The corrected framing — **persistent homology is one of two genuine independent validations** of the helix claim — elevates it to primary status alongside causal interventions.

#### B.6.1 What this does

For each concept's centroid arrangement (after projection into the Phase C subspace, after orthogonalization for carries, optionally after GPLVM dimensionality reduction), compute the persistence diagram. The persistence diagram shows the birth and death scales of each topological feature.

- A circle has H₀ = 1 (one connected component) and H₁ = 1 (one loop).
- A helix has H₀ = 1 and H₁ = 1 (the linear extent contributes to H₀ structure, but the loop is preserved).
- A line has H₀ = 1 and H₁ = 0.
- A figure-8 or two interlocking circles has H₀ = 1 and H₁ = 2.

Long-lived features are real; short-lived features are noise. The "persistence" of a feature is death − birth.

#### B.6.2 Implementation

Use `gudhi` or `ripser` (standard Python TDA packages). Compute persistence up to dimension 1.

For each concept where Fourier or GPLVM detected periodic structure:

1. Project activations onto the concept's Phase C subspace (or, for sensitivity, the GPLVM-learned latent space).
2. Compute centroids per value.
3. Compute the persistence diagram of the centroid point cloud up to H₁.
4. Report the longest-lived H₁ feature's persistence and its rank.
5. Compute a permutation null: shuffle value labels, recompute centroids, recompute the persistence diagram. Repeat 1,000 times. The 95th percentile of the longest H₁ persistence under shuffling is the noise threshold.
6. Declare H₁ feature significant if its persistence exceeds the 95th-percentile threshold by at least 50% (this 1.5× factor is conservative against finite-permutation noise; the previous plan attributed this to the topology paper, which does not actually prescribe it — sensitivity to this factor should be reported in the appendix as 1.0×, 1.5×, 2.0× variants).

#### B.6.3 Pre-registered statement

"For each carry concept where Phase G detects a helix, we expect exactly one significantly persistent H₁ feature corresponding to the helix's circular axis. For middle answer digits where neither Fourier nor GPLVM detect periodic structure, we expect zero significantly persistent H₁ features."

#### B.6.4 Multi-loop concepts (the prism question from Bai et al.)

Bai et al. (`papers/baietal_paper.md`) report Fourier basis structure with multiple frequencies (k ∈ {0, 1, 2, 5}) forming a "pentagonal prism" in their 2-layer ICoT model. Their setting (2-layer, 4-head, ICoT-trained, least-significant-first tokenization) is very different from Llama 3.1 8B, so transferability is not guaranteed.

Persistent homology gives us a direct test: if a concept's centroids actually form a pentagonal prism with five interleaved loops, the persistence diagram should show **five long-lived H₁ features**, one per loop. A single helix has one. A flat figure (no closed loop) has zero.

The exploratory pentagonal-prism cells from Phase G can be re-tested under persistent homology as a falsification test for the prism hypothesis. The paper does not pre-commit to multiple H₁ features being present; it commits to reporting the count honestly.

This is one place the paper could land an independent piece of evidence either for or against the Bai et al. structure transferring to Llama. If multiple H₁ features appear, the prism interpretation gains support. If only one appears, the {0, 1, 2, 5} signature was a Fourier artifact in the smaller model and is not present in Llama.

#### B.6.5 Why topology is genuinely independent of GPLVM

GPLVM fits a parametric manifold under a kernel-form assumption. RBF VAE fits a parametric manifold under an architecture assumption. Both share the "uncertainty grows away from data" mechanism that biases their geometry toward the data manifold.

Persistent homology does *not* fit a manifold. It computes simplicial complexes from pairwise distances and reads off topological invariants. The invariants are *combinatorial*, not geometric: H₁ counts loops, period unmoved by smooth deformations. Two methods agreeing on H₁ is real cross-method evidence.

If GPLVM finds a periodic manifold and persistent homology finds zero significant H₁ features, the GPLVM result is suspect and the discussion section says so.

---

### B.7 Step 7 — Causal experiment 1: subspace ablation on carry_1

This is the first of two causal experiments. The minimum bar for a top-venue mechanistic interpretability paper.

#### B.7.1 What this does

Take the model. At layer 16, identify the carry_1 subspace from Phase C (or, after Phase B.3, the orthogonalized carry_1 subspace). Project the residual stream activations onto the orthogonal complement of that subspace (effectively zeroing out the carry_1 component). Continue the forward pass. Measure how often the model now produces the correct answer.

#### B.7.2 Controls

The interesting comparison is not "ablation versus no ablation." It is "carry_1 ablation versus random subspace ablation of the same dimension," and "carry_1 ablation versus an unrelated-concept subspace ablation of the same dimension."

- **Treatment.** Project out carry_1's Phase C subspace (~9 dimensions at L5).
- **Control 1 (random subspace).** Generate a random orthonormal subspace of the same dimension uniformly from the Grassmannian. Repeat 100 times.
- **Control 2 (irrelevant concept subspace).** Project out the subspace for an unrelated concept of the same dimension (e.g., a "this token is a digit" indicator if available; otherwise an unrelated arithmetic concept).

If carry_1 ablation drops accuracy by at least 10 percentage points more than random ablation, with a non-overlapping 95% CI across the 100 random controls, the carry_1 subspace is causally important. The exact magnitude of the drop is the effect size.

#### B.7.3 What gets reported

For the L5 portion of the curated set:

- Baseline accuracy (no ablation).
- Carry_1 ablation accuracy.
- Mean and 95% CI of random-subspace ablation accuracy across 100 trials.
- Mean of irrelevant-concept ablation accuracy.
- Difference between carry_1 ablation and the random control mean, with a permutation-test p-value.
- z-score of carry_1 effect against the random-control distribution.

#### B.7.4 Per-layer extension

After establishing the result at layer 16, repeat at every layer in the **9 sampled layers** {4, 6, 8, 12, 16, 20, 24, 28, 31} (`config.yaml:10`). The previous version of this plan listed only 8 layers (dropping layer 6); that was inconsistent with the code, which extracts and stores activations at all nine. Plot accuracy drop versus layer. The expected shape: small in early layers, peaks at the layer where the carry is computed, decays in later layers. This is the layer-localization signal that converts an aggregate ablation into a mechanistic claim about *where* the carry is computed.

#### B.7.5 What this proves and does not prove

If carry_1 ablation drops accuracy more than random, the carry_1 subspace is causally used. This is a real mechanistic finding.

It does *not* prove the **helix specifically** is the computational mechanism. The subspace contains a helix, but ablating the subspace removes both the helix and any other structure that lives in the same dimensions. To pinpoint the helix, we need experiment 2.

#### B.7.6 Detailed intervention protocol

For each example in the L5 curated set:

1. Run the model forward on the input prompt up to layer L (start with L = 16). Save the residual stream activation h_L of size 4096.
2. Compute the projection of h_L onto the carry_1 subspace V₁ (dimension d₁): `h_L^∥ = V₁ V₁ᵀ h_L`. The orthogonal complement is `h_L^⊥ = h_L − h_L^∥`.
3. Replace h_L with h_L^⊥ in the forward pass. Continue computing layers L+1 through 31.
4. Read off the model's output prediction.
5. Record whether the model's prediction of each answer digit changed.

Repeat for all L5 examples in the curated set (target: all 1,600 L5 examples; minimum: the 800 correct + 800 wrong matched pairs). Aggregate accuracy.

The control runs do the same with V₁ replaced by:
- A random orthonormal d₁-dimensional subspace, drawn uniformly from the Grassmannian. 100 trials.
- The Phase C subspace of an unrelated concept of similar dimension. 1 trial each, multiple choices.

Test statistic: (accuracy_without_ablation − accuracy_with_carry₁_ablation). Pre-registered threshold: z-score > 2.5 (p < 0.01) and absolute drop > 10 pp above random control mean.

#### B.7.7 Per-example diagnostic

Beyond aggregate accuracy, look at per-example shifts:

- For correct examples that became wrong after ablation, inspect the wrong digit. Did the trailing digit shift in a way that suggests the ablation broke specifically the units-column computation? If yes, the carry_1 subspace is doing exactly what we predict.
- For wrong examples that stayed wrong, did the wrong-ness change qualitatively? An off-by-one error becoming an off-by-100 error would suggest the ablation broke a specific carry path.

These per-example observations enrich the discussion section without entering the headline statistic.

#### B.7.8 Failure mode and pre-registered fallback

A common failure: ablation completely destroys the model's behavior, making it produce gibberish for all examples. This means the carry_1 subspace overlaps too much with general arithmetic computation; we are not isolating the carry signal.

If this happens, the fix is to ablate only the components of the carry_1 subspace that are *orthogonal* to other arithmetic concepts. Use the orthogonalization from B.3: ablate the carry_1 directions that survive after projecting out col_sum_1, partial products, etc. These are the carry_1-specific directions.

Pre-register this fallback: "If aggregate ablation produces output collapse, we report the carry_1-specific (post-orthogonalization) subspace ablation result instead, with the methodological caveat noted in the limitations section."

---

### B.8 Step 8 — Causal experiment 2: helix rotation on the trailing answer digit

This is the more surgical experiment. It targets the helix specifically, not the subspace.

#### B.8.1 What this does

For the trailing answer digit at L5 (`ans_digit_5_msf`, where Phase G found 18/18 helix detection in the correct population over 4,197 examples):

1. Take a correct example. The answer's ones digit is some value v. The activation at layer 16 projects onto a specific point on the helix.
2. Rotate the projection on the helix by one angular position (the rotation that, on a perfect helix, corresponds to incrementing v by 1 modulo 10).
3. Add the rotated projection back to the residual stream (subtract the original projection, add the new one).
4. Continue the forward pass.
5. Check whether the model's output changes such that the trailing digit is now (v + 1) mod 10.

#### B.8.2 Why this is the cleanest test

If the helix is the mechanism — if the model literally uses the angular position on the carry helix to produce its trailing digit — then a one-position rotation should produce a one-digit increment. This is a **rotational intervention** on a representational manifold. The terminology "Clock algorithm" arises in the modular-arithmetic toy-model literature (Nanda et al. 2023, "Progress measures for grokking via mechanistic interpretability"), where a small grokked transformer was shown to compute modular addition by rotating digit embeddings on a circle and reading off the result. We do **not** claim Llama 3.1 8B implements the Clock algorithm — the test transfers, but the algorithm need not. The paper should describe the intervention mechanically (see B.8.5) and reserve "Clock algorithm" for the toy-model literature; our test is a rotation-specific intervention on a representational manifold identified empirically. As discussed in B.8.8, the prior art (Gurnee et al. on Claude 3.5 Haiku) describes rotation **structurally** but does not run the rotation **experiment**; our B.8 is the rotation experiment.

If the helix is incidental — if the model uses some other computational structure and the helix is just a byproduct — the rotation will produce a chaotic output change, not the predicted one.

#### B.8.3 Controls

- **Random rotation.** Rotate by a uniformly random angle (not aligned to any digit position). The output should change in a chaotic way, not predictably.
- **Off-axis perturbation.** Add a vector that is in the subspace but not on the helix. The output may degrade but should not predictably increment by 1.
- **Magnitude-only perturbation.** Move along the linear axis of the helix without rotating. The output may shift but not in a way that increments the trailing digit specifically.
- **Sign-flip control.** Rotate by Δθ_step in the opposite direction; expect (v − 1) mod 10 if mechanism is genuine, chaotic otherwise.

#### B.8.4 Helix calibration: how to know the right rotation angle

The challenge: we have a helix, and we need to know that "rotate by Δθ" corresponds to "increment the digit by 1." This is calibration.

Procedure for the trailing answer digit at L5:

1. Take all 800 correct L5 curated-set examples (or all 4,197 L5 correct in the full population for the calibration step). For each, project the layer-16 activation onto the helix subspace identified by Phase G.
2. For each digit value v in 0–9, compute the centroid: average of all projections where the answer's trailing digit equals v.
3. Fit a circle to the 10 centroids in the 2D periodic plane: find center (c_x, c_y), radius r, and rotation offset φ₀ that minimize the sum of squared residuals.
4. The angle of centroid v on the fitted circle is θ_v = atan2(centroid_v_y − c_y, centroid_v_x − c_x).
5. The differences θ_{v+1} − θ_v should be approximately equal across consecutive v if the helix is regular. The "ideal" rotation increment is the mean of these differences, Δθ_step = (θ_9 − θ_0) / 9 (approximately 36° for a 10-digit clock).

This Δθ_step is the calibration. Each rotation by Δθ_step in the experiment should correspond to incrementing the predicted trailing digit by 1.

Calibration failure mode: if the centroids are not well-fit by a circle (R² of circle fit < 0.8), the helix is not regular enough for clean rotation, and the experiment's interpretation must be qualified accordingly.

#### B.8.5 The intervention, mechanically

For an example with trailing digit v in the correct prediction:

1. Run forward up to layer 16. Save activation h_16.
2. Project h_16 onto the helix subspace: `h_16^∥ = V_helix V_helixᵀ h_16`. Residual: `h_16^⊥ = h_16 − h_16^∥`.
3. Compute the angle of h_16^∥ on the fitted helix circle: θ_current = atan2(h_16^∥_y, h_16^∥_x).
4. Compute the rotated angle: θ_new = θ_current + Δθ_step.
5. Construct the rotated projection: h_16^∥_new = (c_x + r cos θ_new, c_y + r sin θ_new) in the helix's 2D plane, then expressed back in the 4096D space via V_helix.
6. Replace h_16 with h_16^⊥ + h_16^∥_new. Continue forward.
7. Read off the new prediction. Expected trailing digit: (v + 1) mod 10.

For control runs: replace Δθ_step with (a) a uniformly random angle in [0, 2π], (b) zero (sanity check, no rotation), (c) a vector in the orthogonal complement to the helix subspace (off-axis perturbation), or (d) the negative of Δθ_step (sign-flip).

#### B.8.6 Aggregate statistics

For all examples in the L5 correct curated subset where ans_digit_5_msf is well-defined:

- **Predicted-shift accuracy:** fraction of examples where the trailing digit after rotation equals (v + 1) mod 10. Pre-registered baseline: chance level is 10%; significance threshold 25%.
- **Mean angular shift in output:** if the model's prediction shifts by some amount, what is the mean shift in the output's "trailing-digit angle" on the same helix? Regression with x = applied rotation angle, y = output's angular shift; expected slope ≈ 1.
- **Off-axis control failure rate:** fraction of off-axis perturbations that produce predicted-digit shifts. Should be near chance.
- **Sign-flip consistency:** fraction of negative-rotation examples that produce (v − 1) mod 10. Same threshold as the positive case.

#### B.8.7 The result we want to be able to report

In the best case: "Helix rotation by one digit-step produced the predicted trailing-digit increment in [N]% of examples (vs. 10% chance and [M]% under random rotation), establishing the helix as the causal substrate for the trailing-digit computation at layer 16."

Even a moderate result, say 30% predicted-shift accuracy versus 10% chance, would be a publishable mechanistic claim. A null result would be reported honestly with the caveat that the helix may be representational without being directly steerable; the discussion should treat representational helices and computational helices as distinct claims (this is a stronger framing than the previous plan's "the helix is functional, full stop").

#### B.8.8 Comparison to Gurnee et al. (NEW — corrected)

`papers/gurnee_etal_manifolds.md` (Anthropic, Claude 3.5 Haiku character counting) is the closest precedent in the literature. They identified 1D helical manifolds for a counting task and validated the *representational* status of those manifolds with **subspace ablation and mean-patching** causal interventions (`gurnee_etal_manifolds.md:300–310`). The rotational structure they describe — boundary heads "rotating" the character-count manifold to align it with line-width manifolds — is identified by **inspection of weight matrices**, not by an experiment that directly manipulates the rotation and measures the downstream effect. Their own discussion (`gurnee_etal_manifolds.md:306`) is explicit: "What is missing is feature-level causal intervention... there is no experiment that directly manipulates the rotation and measures the downstream effect."

The previous version of this plan called Gurnee et al. a precedent for "rotational manipulations." That overstated the literature: Gurnee et al. validated their helical manifolds via subspace ablation and mean-patching (B.7-style interventions), and *observed* rotational structure in weights but did not directly intervene on it. Our B.8 helix-rotation experiment is therefore a **methodological extension** of Gurnee-style manifold interventions, not a replication. We adopt their causal-intervention framework (subspace ablation, mean-patching) for B.7 and propose the rotation experiment in B.8 as a new test that the literature has flagged as needed but not yet performed at scale on a production model. The paper should frame this as "extending Gurnee et al.'s subspace-ablation methodology with a rotation-specific intervention designed to test the helix as a computational mechanism, not just a representational one."

The expression "Clock-style test" comes from the modular-arithmetic literature on grokked toy models (Nanda et al. 2023). It is informal in our context. The paper should describe the rotational intervention mechanically (B.8.5) rather than relying on a borrowed name.

---

### B.9 Step 9 — Difficulty-matched correct/wrong validation

This step closes the loop on the difficulty confound that is the second-biggest weakness in the audit.

#### B.9.1 What this does

Take the L5 correct examples and the L5 wrong examples in the curated set. They are difficulty-matched by construction (B.1.1 requirement 5). Re-run Phase G's Fourier analysis on this matched subset. Compare the helix detection rates.

Run the same matched comparison at L4 (where carry_0/L4/correct's 18/18 vs. carry_0/L4/wrong's 0/18 finding lives).

#### B.9.2 What we are testing

If the correct/wrong helix asymmetry was real (not just a difficulty confound), the helix should still appear in correct examples and not in wrong examples *even after matching for difficulty*.

If it was a difficulty confound, the asymmetry should shrink or disappear.

#### B.9.3 Why this matters

The "functional" claim — the centerpiece of the paper's degradation argument — rests on this. If matching for difficulty kills the asymmetry, the paper's framing changes from "the helix is functional" to "the helix is correlated with computational ease." Both are publishable findings; the first is stronger.

#### B.9.4 Implementation

The curated set already contains 800 correct + 800 wrong L5 problems matched on magnitude tier and carry count, plus 600 correct + 1,200 wrong at L4. Run Phase G's exact pipeline on the matched subsets. Report the same statistics.

The added population diagnostic from B.2.5: also compare the connected-component count of correct and wrong centroid graphs. If the wrong population's manifold is not just lower-density but **shattered** (more disconnected components), that is a stronger qualitative claim than "lower geodesic agreement" and should be reported as a primary finding.

#### B.9.5 The interpretation tree (pre-registered)

After running the matched comparison, exactly one of these branches is selected for the paper:

- **Branch 1 (full survival):** The asymmetry is as strong on matched pairs as on the unmatched populations. carry_0 / L4 / matched-correct still beats matched-wrong; ans_digit_5_msf / L5 / matched-correct still beats matched-wrong. The functional claim survives in its strongest form.
  - Framing: "The helix is functional, present when the model computes correctly and degraded when it fails. This holds even under exact matching for problem difficulty."
- **Branch 2 (partial survival):** The asymmetry shrinks but does not disappear. There is a real correctness signal, but a substantial portion of the original asymmetry was difficulty-driven.
  - Framing: "The helix is partly functional and partly correlated with computational ease. Both effects contribute to the population-level asymmetry."
- **Branch 3 (collapse):** The asymmetry vanishes. Matched correct and matched wrong both show similar FCR. The original Phase G result was a difficulty confound.
  - Framing: "The helix's apparent correlation with correctness reflects difficulty rather than function. Causal experiments (B.7 and B.8) become the primary evidence for functional relevance."

The paper has all three framings drafted in advance so that the result can be slotted into the right one immediately upon completion. **The methodology paper does not depend on the functional claim** — that was a finding from Phase G. The methodology paper depends on the pipeline working, which it does regardless of the branch.

---

### B.10 Step 10 — Cross-method consistency table

The end product of B.4 through B.9 is a single table that has one row per (concept, population) cell and one column per method.

| Concept × Population | Fourier (FCR / period) | GPLVM (best kernel, ARD-d) | RBF VAE (geodesic ρ; if fitted) | Persistent homology (significant H₁) | Subspace ablation (Δ vs random) | Helix rotation (predicted-shift) | Difficulty-matched survival | Verdict |
|---|---|---|---|---|---|---|---|---|
| carry_1 (L5 correct) | 0.59 / period 18 | periodic+linear, d=2 | agree (0.91) or N/A | 1 H₁ significant | TBD (B.7) | N/A | TBD (B.9) | TBD |
| carry_0 (L4 correct) | 18/18 / period 9 | TBD | TBD | TBD | N/A | N/A | TBD | TBD |
| ans_digit_5_msf (L5 correct) | 18/18 / period 10 | TBD | TBD | TBD | N/A | TBD (B.8) | TBD | TBD |
| ans_digit_2_msf (L5 correct) | none | RBF or no kernel | N/A or agree | 0 H₁ | N/A | N/A | N/A | No structure |

#### B.10.1 How the table is read

Each row is one (concept, population) cell. The verdict column is determined by a pre-registered rule:

- **Confirmed (independent multi-method):** Phase G + GPLVM + persistent homology agree on structure (or absence), AND at least one causal column shows a positive effect. The strongest possible claim.
- **Confirmed (geometric only):** Phase G + GPLVM + persistent homology agree on structure but no causal experiment was run on this cell. Strong descriptive claim.
- **Mixed (geometric and causal disagree):** Geometry says structure, causal says no effect (or vice versa). Reported with explicit discussion.
- **No structure (multi-method):** All methods agree on absence.
- **Inconsistent:** Methods give incompatible results. Reported honestly with discussion.

For the paper, we expect:
- 4–8 cells in **Confirmed (independent multi-method)** — primarily the carries that everything agrees on.
- 4–6 cells in **Confirmed (geometric only)** — concepts where causal experiments were not run.
- 1–3 cells in **Mixed** — the most interesting cases scientifically.
- The middle answer digits land in **No structure** with full agreement, or **Inconsistent** if any method finds something.

The paper's main numerical claim is the count in **Confirmed (independent multi-method)**.

---

### B.11 Step 11 — Case-study figure

This is the figure that goes on the first page of the paper. It is the visual summary of the entire pipeline applied to one concept.

#### B.11.1 The walkthrough: carry_1 at L5, layer 16, correct population

This is what the case-study section documents end-to-end:

- **B.1 (curated set).** carry_1 at L5 takes raw values 0–17. The curated set has 800 correct L5 examples that collectively cover carry_1 values 0 through 17, with at least 30 examples per value (from the difficulty-matching pass). The matched wrong subset has 800 examples covering similar carry_1 values.
- **B.2 (within-group PCA).** For each carry_1 value v in 0–17, run PCA on activations where carry_1 = v. Expected outcome: WGCR averages around 0.28, dip test passes at all values, within-vs-between ratio around 0.07. carry_1 lands cleanly in green. Centroid analysis is sound.
- **B.2.5 (disconnected diagnostic).** k = 10 nearest-neighbor graph on the 18 centroids. Expected outcome: one connected component. Single GPLVM admissible.
- **B.3 (orthogonalization).** Project out col_sum_1, pp_a1_x_b0, pp_a0_x_b1 subspaces. Re-run Phase G. Expected: two_axis_fcr drops from 0.59 to ≈0.51; structure mostly belongs to carry_1 itself.
- **B.4 (GPLVM).** Fit three kernels with period prior 18 (from Phase G):
  - Periodic kernel: log p ≈ −3,420.
  - RBF kernel: log p ≈ −3,510.
  - Periodic + linear kernel: log p ≈ −3,395.
  - Periodic+linear wins by 25 nats over RBF. ARD prunes to 2 active dimensions (one periodic, one linear).
- **B.5 (RBF VAE, optional).** If GPLVM is clean, RBF VAE runs as supplementary; expected geodesic Spearman 0.91 with GPLVM.
- **B.6 (persistent homology).** Persistence diagram of the 18 carry_1 centroids in the GPLVM latent layout. One H₁ feature with persistence > 4× the permutation null. No spurious additional H₁.
- **B.7 (subspace ablation).** Expected: baseline accuracy 100% on the 800 correct L5 examples; carry_1-ablation accuracy ≈30%; random-subspace ablation mean ≈81%. carry_1 effect 50 pp above random control. Layer-localization: peaks at layer 16.
- **B.8 (helix rotation, on ans_digit_5_msf, not carry_1).** Δθ_step ≈ 36°. Expected predicted-shift accuracy ≈40% versus 10% chance and ≈12% under random rotation.
- **B.9 (difficulty-matched).** Re-run Phase G on 800 matched correct vs. 800 matched wrong. Expected (Branch 1): asymmetry survives, helix_fcr ≈0.62 in matched correct, ≈0.41 in matched wrong.

**Cross-method verdict for carry_1 in the L5 correct population:**

- Fourier: helix detected, FCR 0.59 (orthogonalized: 0.51).
- GPLVM: periodic+linear with Phase G period prior wins by 25+ nats; ARD pruned to d=2.
- RBF VAE: agrees with GPLVM (if run).
- Persistent homology: one significantly persistent H₁.
- Subspace ablation: ~50 pp accuracy drop above random control.
- Difficulty-matched: asymmetry survives.
- **Verdict: Confirmed (independent multi-method) with causal evidence.**

This is the strongest evidence framework the paper can offer.

#### B.11.2 The figure

A single concept: carry_1 at L5, layer 16, correct population. Multi-panel:

- **Panel A (Phase C):** 2D scatter of carry_1 centroids in their consensus subspace, showing they are well-separated.
- **Panel B (Phase G):** Same centroids on the unit-circle reference, with FCR annotated.
- **Panel C (GPLVM):** Latent coordinates colored by carry value, with uncertainty contours. Inset: log marginal likelihood comparison across the three kernels.
- **Panel D (Persistent homology):** Persistence diagram with the long-lived H₁ feature highlighted, permutation null floor shown.
- **Panel E (Causal ablation):** Accuracy drop from carry_1 ablation versus random subspaces, with per-layer localization curve.

Each panel ~5cm × 5cm. Whole figure on one page. Caption tells the story: linear method finds the room; Fourier hints at a circle; GPLVM confirms the periodic structure with uncertainty; persistent homology validates the topology independently; causal ablation shows it is used.

#### B.11.3 Implementation

Each panel is generated independently from the relevant analysis output. The combining script puts them in a single matplotlib figure with shared color schemes (carry value 0 = darkest, carry value 17 = lightest, viridis-like palette).

---

### B.12 Step 12 — Writing the paper

This is the longest step but also the most predictable. Time estimates removed.

#### B.12.1 Structure

- **Abstract (200 words).** Methodology contribution; testbed; headline numerical result.
- **Section 1 — Introduction (1.5 pages).** LRH is necessary but insufficient; multiplication is the testbed; pipeline overview.
- **Section 2 — Related work (0.75 pages).** Linear Representation Hypothesis (Park et al.); Fourier features in LLMs (Nanda, Kantamneni & Tegmark, Zhou et al.); manifold methods (Hauberg, Gurnee et al., Bergamin et al., Yu et al.); arithmetic interpretability (Bai et al., Li et al.).
- **Section 3 — Methods (2 pages).** The pipeline in order. Phase C/D/E/F/JL get a paragraph each. Phase G's Fourier method gets half a page. GPLVM, RBF VAE, persistent homology, and the two causal experiments get a paragraph each. Curated set construction gets half a page.
- **Section 4 — Results (3 pages).** Organized by claim, not by method.
  - Claim 1: linear pipeline finds the rooms. Half a page.
  - Claim 2: Fourier reveals carries on helices and operand digits without. Half a page.
  - Claim 3: GPLVM confirms helix geometry with uncertainty bars and ARD-pruned dimensionality. Three quarters of a page.
  - Claim 4: persistent homology validates the topology independently. Quarter page.
  - Claim 5: difficulty-matched controls (Branch 1, 2, or 3 framing). Quarter page.
  - Claim 6: causal ablation establishes mechanistic relevance. Half a page.
  - Claim 7: helix rotation establishes the trailing-digit computation. Half a page.
  - The headline figure (B.11) goes here.
- **Section 5 — Limitations (0.5 pages).** Superposition contamination acknowledged; centroid assumption tested; manifold-hypothesis-defense via causation; orthogonalization-only against known correlates; stratification caveat (geometric vs. distributional questions); per-floor saturation in Phase G; small-N at L5/correct; SiLU-activations not directly covered by Riemannian Laplace theory; transferability of Bai et al. {1, 2, 5} prism uncertain.
- **Section 6 — Discussion (1 page).** What this says about LRH; what it says about how Llama does multiplication; what generalizes and what does not; the methodology standard.
- **Section 7 — Conclusion and future work (0.5 pages).** SPD adaptation as Paper 2; SAE-based decomposition as Paper 4; held-out generalization tests.
- **References.** ~50 citations.
- **Appendices.** Coverage report for the curated set; full per-concept tables for each method; computational details; expanded persistent homology diagrams; pre-registration table B.15.

Total: 9 pages of main text plus appendices.

#### B.12.2 Tone discipline

The whole paper is about being precise. Avoid:

- "We show that the helix is the computation." (overclaim — the most we have is one targeted intervention).
- "Carries are encoded as helices." (oversimplification — carries are encoded as helices in subspaces shared with their algebraic correlates).
- "Our method solves superposition." (false).
- "The model uses Fourier features." (we have one specific causal demonstration on the trailing digit; generalizing to "the model uses" is too broad).

Do say:

- "The helix structure is causally implicated in the trailing-digit computation at layer 16."
- "Carries are encoded as helices in subspaces shared with their algebraic correlates; the helix structure persists after orthogonalization against the correlates by [N%]."
- "Our methodology partially addresses superposition contamination via orthogonalization controls; full SAE-based decomposition is left as future work."
- "For the trailing answer digit, helix rotation predicts model output as expected; this constitutes a mechanistic test on a single concept and does not generalize to all concepts without further experiments."

#### B.12.3 The "would I cite this paper" test

Before submission, ask: would I cite this paper? Yes if and only if:

- It states a specific contribution clearly (the methodology pipeline).
- It provides at least one new piece of mechanistic evidence (the helix rotation experiment).
- It is honest about what is observational and what is causal.
- It pre-registers thresholds and reports against them.
- It releases reproducible artifacts (the curated set, the code, the model hash).

If any of these is missing, the paper is one citation short of the level it could be at, and the reviewer will likely say so.

---

### B.13 Dependencies (no calendar)

```
Step 1 (curated set)  ──>  All other steps
Step 2 (within-group + disconnection)  ──>  Step 4 (GPLVM informed by within-group structure and connection diagnostic)
Step 3 (orthogonalization)  ──>  Steps 4, 6 (clean inputs for non-linear methods)
Step 4 (GPLVM)  ──>  Step 6 (persistent homology can use GPLVM latent coordinates as one of two analysis bases)
                ──>  Step 11 (figure includes GPLVM panel)
Step 5 (RBF VAE)   conditional on Step 4 outcome  ──>  Step 11
Step 6 (persistent homology)  ──>  Step 10 (cross-method table); Step 11 (figure)
Step 7 (subspace ablation)  ──>  Step 8 (helix rotation builds on the subspace + orthogonalization machinery); Step 11
Step 8 (helix rotation)  ──>  Step 11
Step 9 (difficulty matching)  ──>  Step 12 (impacts the framing)
Step 10 (cross-method table)  ──>  Step 12
Step 11 (figure)  ──>  Step 12
Step 12 (writing)
```

The natural ordering is 1 → (2, 3 in parallel) → (4 with optional 5) → 6 → 7 → 8 → 9 → 10 → 11 → 12. Each step has a pre-registered failure mode (B.14) and a pre-registered threshold (B.15) so that re-decisions are made automatically when a result misses a threshold rather than being negotiated after the fact.

The two project-protection practices:

**Strict scope discipline.** Every step lives on the curated set. Anyone who proposes "let's also run this on the full 122K" gets to argue why this specific question requires the larger sample. If the answer is "more statistical power," they lose: the curated set has enough power for every test we are running, by design. If the answer is "to address a different question," they lose harder: addressing a different question is Paper 2.

**Pre-registered failure modes.** B.14 below lists the threshold for every test and what to do if it is missed. When a result misses a threshold, the response is automatic: report the outcome, follow the pre-registered downgrading, move on. No re-tweaking the test until it passes; that is p-hacking and it kills credibility.

---

### B.14 Pre-registered failure modes

**If GPLVM fails to converge or gives unstable results.** Fall back to RBF VAE as the primary method and report GPLVM as a reference comparison. Cite Hauberg's argument that both methods recover correct geometry (with the corrected caveat that the agreement between them is expected by construction, not independent evidence). If both fail, the paper switches headline to persistent homology + causal experiments.

**If GPLVM and persistent homology disagree.** Persistent homology is the more conservative, more independent test. Trust it. Reframe: "Fourier and GPLVM detect periodic structure but the topological invariant does not survive, indicating the structure is geometric (smooth curvature) but not topologically a loop. This is consistent with helices that wrap less than a full turn, which is itself an interesting finding."

**If subspace ablation does not produce a significant accuracy drop.** This is a major problem. The paper has to acknowledge that the carry subspace might be representational but not computationally used, with the "functional" claim downgraded. Investigate whether ablating across multiple layers simultaneously gives a cleaner signal. If still null, report it honestly: this is what science looks like.

**If helix rotation does not produce predicted output changes.** Less catastrophic than failed ablation, since helix rotation tests a much more specific claim. The paper says: "ablation demonstrates the carry subspace is causally used, but the helix-rotation test was inconclusive, suggesting the model uses the subspace via a mechanism that is not captured by simple angular rotation. Future work must investigate."

**If the difficulty-matched control collapses the correct/wrong asymmetry (Branch 3).** Reframe: "we observe geometric structure that correlates with the regime where the model operates well; this correlation is at least partly explained by the model handling easier inputs more cleanly, and the structure does not strictly require correct computation." The methodology paper does not depend on the functional claim — that was a Phase G finding.

**If the disconnected-manifold diagnostic shows multiple components for a carry concept.** Switch to mixture-of-GPLVMs for that concept, or skip GPLVM and report persistent homology only. Reframe the carry section: "carries are not represented as continuous helical manifolds but as collections of discrete clusters with periodic value-labels; this is consistent with a lookup-table-style internal representation."

**If kernel comparison is inconclusive (gap < 2.3 nats, i.e., below K&R's "strong" threshold).** Report the inconclusive result and adjust the framing: "GPLVM does not favor any of the three tested kernel families at the K&R 'strong' threshold; the carry geometry is consistent with multiple manifold classes." Persistent homology and causal experiments become the primary differentiators. If the gap is in the 2.3–5 nats window (strong but not decisive), report the leading kernel as "best-fitting" with the explicit note that it has not reached the decisive threshold.

**If 100-trial random-subspace control does not have non-overlapping CI with carry_1 ablation.** Investigate whether carry_1's subspace dimension is too small (low signal) or too large (overlapping with other concepts). Use the orthogonalized carry_1 subspace and rerun. If still null, report.

**If the calibration step for helix rotation fails (R² < 0.8 for circle fit).** The helix is not regular enough for clean rotation. The paper either reports the negative calibration result and skips the rotation experiment, or runs the experiment with the caveat that the rotation step is only approximate.

The honest framing — "here is what we found, here is what survived controls, here is what did not" — is what a methodology paper should look like. Reviewers reward this. They punish overclaiming.

#### B.14.1 The realistic-outcome framings (NEW)

The thresholds in B.15 are tight relative to what the data is likely to deliver. Three of them — kernel comparison at 5 nats, helix calibration at R² ≥ 0.8, persistent-homology persistence at 1.5× the null floor — could plausibly land just under the decisive line for many cells. The plan's earlier headline drafts assumed clean wins; the realistic case is mixed. To avoid scrambling the framing under deadline pressure, the paper has prose ready for each realistic outcome in advance. The three branches below are pre-registered framings, not contingencies to invent later.

**Realistic outcome 1 — half of GPLVM cells land in 2.3–5 nats ("strong but not decisive").**

This is likely. K&R's "decisive" threshold (≥ 5 nats ≈ 2.17 log₁₀) is calibrated for hypothesis tests on a single parameter; for whole-kernel-family comparison in GPLVM with ARD pruning, marginal-likelihood gaps of ≥ 5 nats are uncommon unless one kernel is genuinely a poor fit. Many cells will sit between 2.3 and 5 nats. Pre-registered framing if this happens:

> "Kernel comparison at the K&R 'strong' threshold (≥ 2.3 nats) selects a leading kernel for [N] of [M] cells; at the 'decisive' threshold (≥ 5 nats) the count drops to [K]. We report both numbers. The 'strong' tier is sufficient to assert the data prefers the periodic-with-Phase-G-prior kernel over RBF; the 'decisive' tier is the conservative claim and is reserved for the headline cells (carry_1, ans_digit_5_msf). For all other concepts the paper claims kernel preference at the 'strong' tier and notes that more N is required to reach 'decisive'."

The headline figure (B.11) is robust to this: carry_1 is expected to clear the decisive threshold given Phase G's strong helix detection there. If carry_1 itself only reaches 'strong', the headline is rewritten to "the geometry is decisively a manifold; the manifold class is strongly but not decisively a periodic+linear helix; the rotation experiment B.8 provides the deciding evidence."

**Realistic outcome 2 — helix rotation calibration lands at R² ∈ [0.7, 0.8].**

Phase G found 500 helices but most are imperfect — stretched ellipses or non-uniform angular spacing rather than clean circles. Calibration R² is unknown a priori. If it lands just below 0.8:

> "The trailing-digit helix is approximately but not exactly regular (R² = [value]). We run the rotation experiment with calibration acknowledged as approximate. Predicted-shift accuracy is reported with two interpretations: (a) under the assumption that Δθ_step ≈ 36° is the correct increment (the assumption that calibration nominally supports), and (b) under a relaxed criterion where any predicted-digit shift in the direction of v+1, v+1±1 counts as a hit. The two numbers bracket the strength of the rotational claim."

The paper does not abandon B.8 if calibration is imperfect; it reports the rotational result with explicit caveats and lets the reader weight. If calibration falls below R² = 0.7, B.8 is downgraded to "exploratory" and the headline shifts entirely to B.7 (subspace ablation), with rotation in the appendix.

**Realistic outcome 3 — persistent homology lands between 1.0× and 1.5× the null floor.**

The 1.5× threshold is this paper's pre-registered choice, not a standard. The plan's appendix already commits to reporting all three (1.0×, 1.5×, 2.0×). Pre-registered framing if a concept lands at, say, 1.4× — significant under 1.0× but not under 1.5×:

> "carry_1's H₁ persistence exceeds the permutation-null 95th percentile (significant at the 1.0× threshold) but does not exceed the 1.5× margin we pre-registered as the conservative criterion. We report this honestly: persistent homology provides supportive but not conservative-threshold evidence for a topological loop in carry_1."

The paper should not pretend a single arbitrary multiplier is the line. The 1.0× / 1.5× / 2.0× table goes in the body, not the appendix, for any concept that lands in this gray zone.

#### B.14.2 Disconnected-manifold edge cases (NEW)

The B.2.5 disconnection diagnostic is binary at a single k. In practice the k-NN graph at k = 10 might give one component for 16 carry_1 values plus one disconnected island at the rare extreme value (carry_1 = 17 has only a handful of correct examples; carry_2 = 26 has 11). The plan now pre-registers a sensitivity sweep, not a single-k decision:

- Compute connected components at k ∈ {5, 10, 15, 20}.
- **All four k values give one component** → "connected" verdict, GPLVM proceeds.
- **k = 5 gives multiple components but k ≥ 10 gives one** → "weakly connected" verdict, GPLVM proceeds with an appendix note.
- **k ≥ 10 gives multiple components** → "disconnected" verdict, switch to mixture-of-GPLVMs or topology-only.
- **Edge case: one large component plus a singleton at a rare value** → drop the rare value from the GPLVM fit and report it separately; do not let one tail point downgrade the whole concept.

This sensitivity table is reported in the appendix for every concept that GPLVM is fit on.

#### B.14.3 Why we still chose exact GPLVM over sparse (NEW — preempting reviewer push-back)

A reviewer who knows GPs will ask: why not sparse Bayesian GPLVM (FITC, VFE, SVGP) at N = 122,000? It is a fair question and "Hauberg says no" is not the strongest answer. The honest, fuller answer:

1. **Hauberg's geometry-preserving argument hinges on the predictive covariance Σ(z) growing in low-density regions.** Sparse approximations summarize the data through M ≪ N inducing points; the covariance estimate becomes a function of the inducing-point layout rather than the raw data density. With careful inducing-point placement and bandwidth tuning that adapts to local density, sparse-GP can preserve the Hauberg property — but tuning a sparse-GP on activation manifolds at this scale is itself a multi-week sub-project with its own validation requirements (toy-data sanity checks, inducing-point ablations, kernel-bandwidth sweeps).
2. **Exact inference at N = 6,000 is faster end-to-end than sparse inference at N = 122,000 once tuning cost is included.** Days of GPU time for the exact grid (B.4) versus weeks of methodological work to deploy sparse correctly.
3. **The Tier 1 / Tier 2 split (A.6) lets us have it both ways.** Population-level claims (existence, prevalence, frequency) come from Tier 1 statistics that already use all 122K; the Tier 2 manifold-shape claims need at most a few thousand examples to characterize a 1–2D manifold within reasonable uncertainty bars.

A reviewer who pushes for sparse-GP at full N can be answered with: "We agree sparse-GP is in principle a valid path; in practice the methodological tuning cost would dominate the project timeline and we elect to spend that effort on the manifold-hypothesis defense (causal experiments) instead. Sparse-GP at full N is a candidate Paper-2 follow-up under our SPD adaptation framework." This framing is more defensible than "Hauberg said so."

#### B.14.4 The two-layer selection caveat for the curated set (NEW)

The L5/correct curated subset is **800 of the 4,197 L5/correct examples** — about 19%. That subset is hand-picked to be difficulty-matched against L5/wrong, where the matching algorithm itself selects on magnitude tier, carry-count tier, and answer-length. The full L5/correct population was already a stratified sample (carry-balanced) of the 810K input space.

Two layers of selection: (a) carry-balanced stratification down to 122,223 problems → 4,197 correct, (b) difficulty-matched curation down to 800 correct.

Any cross-population claim ("the helix in L5/correct's curated set differs from the helix in the full L5/wrong population") has to acknowledge both layers. The methodology paper handles this by:

- Restricting Tier 2 (manifold shape, causal mechanism) claims to within-curated-set comparisons.
- Restricting Tier 1 (existence, prevalence, frequency) claims to within-full-population comparisons.
- Never mixing the two in the same statement.

If a reviewer asks "is the curated 800 representative of the 4,197?", the answer is: it is **representative of the 4,197 conditioned on the matching variables (magnitude tier, carry-count tier, answer-length)**, and not representative of the unconditioned 4,197 along any axis the matching does not control. The paper states this explicitly in the methods section and in the curated-set coverage report (B.1.4).

#### B.14.5 Compute budget for the causal experiments (NEW)

The earlier plan did not commit to a forward-pass budget for B.7 and B.8. Fold this in so a reviewer can audit the feasibility claim.

**B.7 (subspace ablation).** 1,600 L5 curated examples × 9 sampled layers × (1 carry_1 ablation + 100 random-Grassmannian controls + 1 irrelevant-concept control) = **102 ablation conditions × 1,600 examples × 9 layers ≈ 1.47M forward passes**. On a single A100 at ~50 ms per Llama 3.1 8B forward pass with KV-cache reuse for the prompt prefix, that is **~20 hours of GPU time**. Two A100s halve it. This is the upper bound — many controls can share infrastructure.

**B.8 (helix rotation).** 800 L5 correct curated examples × 1 layer (16) × (1 calibrated rotation + 3 controls: random-angle, off-axis, sign-flip) = **3,200 forward passes per condition**, plus a calibration sweep. **~1 hour of GPU time** at the same throughput.

**Total causal-experiment compute.** ~25 hours of single-A100 time. Add 10× headroom for re-runs after debugging ≈ **10 GPU-days**. This is a serious budget but well below the GPLVM grid (B.4), which is the dominant cost.

The figures above assume the activation-replay shortcut (run the forward pass up to layer L once per example, cache, modify, replay layers L+1 onward). Without caching, costs rise ~3×. The ablation script must implement caching explicitly.

---

### B.15 Consolidated pre-registration table

For maximum credibility with reviewers, every threshold in the paper is pre-registered.

| Test | Statistic | Pre-registered threshold | Justification source |
|------|-----------|--------------------------|----------------------|
| Within-group PCA (B.2) | WGCR | < 0.30 = green, 0.30–0.50 = yellow, > 0.50 = red | Standard concentration-of-variance interpretation |
| Within-group PCA (B.2) | Hartigan dip p-value | > 0.05 = unimodal | Hartigan & Hartigan 1985 standard |
| Disconnection diagnostic (B.2.5) | # connected components in k-NN graph (k=10) | > 1 = switch to mixture-of-GPLVMs | Pre-registered for this paper |
| Orthogonalization survival (B.3) | FCR drop | < 30% = own structure, > 50% = inherited, in between = ambiguous | Conservative; favors null |
| GPLVM kernel comparison (B.4) | Δlog-likelihood (nats) | ≥ 5 nats = decisive (≈ 2.17 log₁₀); ≥ 2.3 nats = strong; < 2.3 = inconclusive | Kass & Raftery 1995 in log₁₀, converted to nats by ×ln 10 ≈ 2.303 |
| GPLVM ARD pruning (B.4) | Lengthscale ratio vs prior median | > 100 = pruned | scikit-learn ARD convention (corrected — not from Hauberg) |
| GPLVM convergence stability (B.4) | Two-seed log-likelihood difference | < 1 nat = stable | Pre-registered for this paper |
| RBF VAE / GPLVM agreement (B.5, if both run) | Geodesic Spearman | > 0.85 = agree | Conservative for noisy methods |
| RBF VAE / GPLVM agreement (B.5, if both run) | Hausdorff distance | within 30% = agree | Hauberg's circle benchmark |
| Persistent homology (B.6) | H₁ persistence vs permutation null | > 95th percentile × 1.5 | Pre-registered; sensitivity at 1.0×, 1.5×, 2.0× reported in appendix |
| Subspace ablation (B.7) | z-score vs random | > 2.5 (p < 0.01) | Standard |
| Subspace ablation (B.7) | Accuracy drop | > 10 pp above random control mean | Practical significance threshold |
| Helix rotation (B.8) | Predicted-shift accuracy | > 25% (vs 10% chance) | Need 2.5× chance to claim mechanism |
| Helix rotation (B.8) | Off-axis control failure | < 15% predicted-shift accuracy | Verifies test specificity |
| Helix rotation (B.8) | Sign-flip consistency | > 25% for (v−1) mod 10 | Tests for direction of rotation |
| Helix rotation (B.8) | Calibration circle fit R² | > 0.8 | If lower, helix is not regular enough |
| Difficulty-matched survival (B.9) | FCR ratio matched-correct / matched-wrong | > 1.3 | Conservative effect-size threshold |
| FDR correction | q-value | < 0.05 | Benjamini-Hochberg standard |

Every claim in the paper that uses one of these statistics is held to its threshold. Failures are reported. Successes are reported. The reviewer can audit every number.

This consolidated table goes in the methods supplement. The body text references it whenever a threshold is invoked.

---

## Part C: Closing notes on the paper's identity

### C.1 What this paper is

A methodology paper. The contribution is a tested pipeline of methods for studying geometric interpretability inside a production LLM. The pipeline has linear, Fourier, probabilistic-manifold, topological, and causal components. Each component has a clear role, a clear deliverable, and a clear failure mode. The pipeline is demonstrated on multi-digit multiplication in Llama 3.1 8B and shown to recover specific, testable, partially-causal claims about the geometry of arithmetic computation.

### C.2 What this paper is not

- Not a paper about how Llama does multiplication. Multiplication is the case study.
- Not a paper that solves superposition. Superposition is acknowledged, partially controlled for via orthogonalization, and left as an open problem.
- Not a paper with completely new mathematical methods. All the methods are taken from prior work (LRH, Fourier features, GPLVM/Hauberg, persistent homology, activation patching). The contribution is the integration into a coherent pipeline plus the falsificationist use of Phase G periods as kernel priors and the elevation of persistent homology and causation as primary independent validation.
- Not a paper about every concept in Llama. The 17 unique concepts are a controlled testbed; the methods generalize.
- Not a paper that proves the helix is *the* computational mechanism. We have one targeted intervention on one digit position and one ablation result on one carry concept. The paper claims **mechanistic relevance for these specific cells**, not universal helix-as-computation.

### C.3 The manifold-hypothesis defense (NEW)

The strongest framing-correction this plan adds is to make the manifold-hypothesis defense explicit. GPLVM, RBF VAE, persistent homology — and even the linear pipeline — all assume some form of structure that might or might not be present in the model's representations.

Even with a perfect GPLVM fit (clean ARD pruning, decisive kernel comparison, stable convergence, well-calibrated uncertainty bars), the question remains: is the manifold the model uses, or is it the manifold our method fits?

The defense is causal. If the GPLVM-recovered manifold's structure (specifically: the angular position on the helix) predicts the model's output under intervention (B.8), the manifold is real. If it doesn't, the manifold is a fitting artifact of methods that are designed to find manifolds.

This is why B.7 and B.8 are not optional. They are what distinguishes "we fit a manifold" from "the model uses a manifold." The discussion section makes this explicit. The limitations section flags it. The case-study figure includes the causal panel as a co-equal piece of evidence, not an add-on.

This is also why we rejected the previous plan's framing of RBF VAE as "cross-validation" and elevated persistent homology + causation in its place. RBF VAE shares GPLVM's mechanism (decoder uncertainty drives the metric); their agreement is expected by construction. Persistent homology uses different mathematical apparatus (combinatorial topology, not Bayesian inference) and causal interventions ask a different question entirely (does it affect outputs?). Those are the genuine cross-method checks.

### C.4 What success looks like at submission time

A paper where every claim is supported by at least one method, where the strongest claims are supported by multiple genuinely-independent methods agreeing, where the limitations are stated up front, and where two causal experiments demonstrate the pipeline's mechanistic potential. A paper where the appendix contains a curated set that any researcher can download and rerun. A paper that says exactly what it does and exactly what it does not, and that builds the foundation for the SPD adaptation paper to follow.

### C.5 The single sentence that captures the contribution

> "We present a methodology that combines linear subspace identification, Fourier screening, Gaussian Process Latent Variable Models with Phase-G-derived period priors, persistent homology as topological cross-validation, and targeted causal interventions into a single pipeline for studying geometric interpretability in production language models, and we demonstrate the pipeline by showing — through multi-digit multiplication in Llama 3.1 8B — that intermediate computational variables sit on causally-relevant manifolds that the Linear Representation Hypothesis alone cannot detect."

That is the paper. The remaining work is execution.

### C.6 What this paper sets up for follow-up work

A clean methodology paper opens specific follow-up directions.

**Paper 2 — SPD metric adaptation.** The pipeline shows that intermediate variables sit on manifolds. SPD metric adaptation (the planned follow-up) takes the next step: learn a low-rank symmetric positive-definite metric on the residual stream that compresses noise dimensions and stretches signal dimensions, aligning the model's representation space with the correct-population manifold. The Paper 1 finding about effective dimensionality of carries (from ARD pruning) directly motivates the rank choice; the operator-norm bound on the metric provides a trust-region guarantee. Paper 1's curated set becomes Paper 2's evaluation set, ensuring continuity.

**Paper 3 — Generalization to other arithmetic and reasoning tasks.** With the pipeline validated on multiplication, applying it to division, modular arithmetic, or symbolic reasoning becomes straightforward. Each task brings new concept labels and the same pipeline runs through.

**Paper 4 — SAE-based version of the pipeline.** With public Llama 3 SAEs available (or trained in-house), the pipeline can be re-run with learned features instead of supervised labels. This addresses the superposition limitation more thoroughly. The methodological contribution is to show whether and how the manifold findings transfer between supervised-label and SAE-feature representations.

**Paper 5 — The negative result extension.** What about the concepts where no manifold structure is found? Middle answer digits at L5 are the candidates. A paper that takes "no manifold here, and we have ruled out 4 reasons why" seriously is itself a contribution.

**Paper 6 — Riemannian Laplace as primary inference.** If the GPLVM-with-MCMC route turns out to be too expensive for downstream tasks, Paper 6 could revisit the same questions using the Bergamin/Hauberg/Arvanitidis 2023 Laplace approximation as a faster, slightly more approximate inference scheme. The trade-offs would be the paper's contribution.

The current paper is the foundation. Each follow-up paper builds on a piece of it: the curated set, the cross-method consistency framework, the causal protocols, the limitations carefully documented.

### C.7 The implicit standard the paper sets

By integrating linear, Fourier, probabilistic-manifold, topological, and causal methods into a single workflow with pre-registered comparison thresholds, the paper sets a standard for how mechanistic interpretability claims should be made in production LLMs. The standard is roughly:

- A claim of representational structure requires at least two genuinely-independent methods (sharing-mechanism methods like GPLVM and RBF VAE do not count as two).
- A claim of causal relevance requires at least one targeted intervention.
- Limitations specific to each method must be acknowledged in the limitations section.
- A reproducible artifact (curated set, code, model hash) must accompany the submission.
- The manifold-hypothesis assumption must be defended causally, not by appeal to method agreement among related methods.

If the paper succeeds in setting this standard, it has impact beyond its specific findings about Llama and multiplication.

### C.8 Risks not in our control

- **The model itself.** Llama 3.1 8B is a moving target only in the sense that future versions exist. The base model used for the pipeline is fixed. As long as the paper specifies the exact Hugging Face model identifier, dtype (bfloat16 inference, float32 stored activations), and torch version, results are reproducible. **The HF identifier should be added to `config.yaml` immediately**; the local mount path is not citable.
- **Reviewer 2.** A bad reviewer can sink any paper. Mitigation: pre-register every claim, use strict statistical thresholds, lead with the methodology framing, provide reproducibility infrastructure (the curated set as a public artifact).
- **Bai et al. transferability.** The {1, 2, 5} prism finding is from a 2-layer, 4-head ICoT-trained model with least-significant-first tokenization. Whether it transfers to 32-layer Llama 3.1 8B with BPE tokenization is genuinely uncertain. The paper does not pre-commit to finding the prism; it commits to running the test (B.6.4) and reporting the count of significant H₁ features honestly.
- **SiLU + Riemannian formalism.** Hauberg's framework (`papers/bayes_paper.md`) and the Riemannian Laplace (`papers/Riemannian_Laplace.md`) are validated for smooth activations (tanh in the latter explicitly). SiLU is smooth but not directly tested in those papers' settings. Empirically, our pullback metric and uncertainty machinery should still function, but the formal guarantees do not extend cleanly. The paper acknowledges this in limitations.
- **Theoretical Fourier-circuit results.** Li et al. (`papers/fourier_circuits.md`, Theorem 4.1) prove single-frequency-per-neuron emergence specifically under: (a) **polynomial activations of degree k**, (b) **prime modulus p modular addition**, (c) **one-hidden-layer networks**, (d) **margin-maximization training**. None of these conditions hold for Llama 3.1 8B (32 transformer layers, SiLU/SwiGLU activations, BPE-tokenized arithmetic over base 10 — non-prime — embedded in language modelling, cross-entropy loss). We cite the theorem only as **motivation for why periodic structure is plausible** in the empirical sense; the theoretical guarantees do not transfer. The paper should explicitly state this so a reviewer cannot accuse us of overclaiming theoretical support.

### C.9 The day after submission

After submission, two things have to happen:

1. **Paper 2 work begins.** SPD metric adaptation is the next research vehicle. Paper 1's findings define the rank, the curated set, and the evaluation criteria. Starting on day one keeps the research pipeline moving.
2. **The reproducibility infrastructure goes public.** The curated set, the code, the model hash, the analysis outputs all get released as a public package. Other researchers should be able to download it and rerun any experiment in the paper. This is the multiplier on the paper's impact.

A methodology paper without released artifacts is half a contribution. A methodology paper with a polished public release is a multi-year reference for the area.

---

## Part D: Quick-reference checklist (no calendar)

A condensed, time-free version for tracking progress.

**Pre-paper foundation (done):**
- [x] Phase A (UMAP scouting; layer 16 = info peak; 351 embeddings)
- [x] Phase B (correlation audit; 2,018 unique pairs across L1–L5; 32 s)
- [x] Phase C (linear subspaces; 2,844 subspaces; 96.7% sig; CV 0.838–1.000 at L5)
- [x] Phase D (LDA refinement; 1,035 results; 99.1% sig; carry λ 0.74–0.95 at L5/all)
- [x] Phase E (residual hunting; 444 eigs above MP at L5/L16; |ρ_s| ≤ 0.082 weak nonlinearity)
- [x] Phase F (principal angles; 42,049 pairs; 39,525 (94%) superposition under p5−10° rule)
- [x] Phase JL (distance preservation; 43.9B distances; Spearman 0.9942–0.9995)
- [x] Phase G (Fourier screening; 3,480 cells; 500 helix detections; 458 floor-saturated; carry periods 18, 27, 19, 10; raw period spec wins; carry_0/L4/correct 18/18 in ~2,897 samples; ans_digit_5_msf/L5/correct 18/18 in 4,197 samples)

**Curated set (Step 1) — DONE (Apr 25, 2026):**
- [x] Selection algorithm written: 5-pass build in `build_curated_set.py` (Pass 0 load+dedup; Pass 1 difficulty stratification at L4/L5; Pass 2 concept-coverage greedy fill; Pass 3 matched-pair construction; Pass 4 assemble+validate; Pass 5 post-assembly concept top-up)
- [x] `curated_set_v1.json` finalized at **8,264 problems** (L3 = 2,401; L4 = 2,846; L5 = 3,017). Output: `/data/user_data/anshulk/arithmetic-geometry/curated/curated_set_v1.json` (22 MB)
- [x] Coverage report: full per-position digit + per-carry tables, matched-pair diagnostics, 19 documented gaps. `docs/curated_set_coverage_report.md` (1,589 lines)
- [x] Difficulty-matched pairs verified: **1,000 L4 pairs** (993 strict + 7 carry_count_tier ±1 relaxed; 0 unmatched) + **1,400 L5 pairs** (1,399 strict + 1 relaxed; 0 unmatched). Mean correct−wrong differences near zero on all three matching axes at both levels (see §11 of coverage report)

**Sanity controls (Step 2):**
- [ ] Within-group PCA on every (concept × value) at L5 and L4
- [ ] WGCR/dip-test/ratio classifications recorded with green/yellow/red verdicts
- [ ] Disconnected-manifold k-NN graph diagnostic (k=10) per concept
- [ ] 1,122 single-example projection plots from Phase G systematically reviewed
- [ ] Within-group + connection-diagnostic results integrated into the methods doc

**Superposition control (Step 3):**
- [ ] Orthogonalization wrapper implemented (load Phase C bases, stack, QR, project)
- [ ] Phase G re-run on orthogonalized activations for carries 0–4
- [ ] FCR-survival comparison table (raw vs orthogonalized)
- [ ] Pre-registered interpretation applied (own / inherited / ambiguous)

**GPLVM (Step 4):**
- [ ] Implementation plus four toy-data validation tests (circle, helix, Gaussian, two circles)
- [ ] Curated-set fits across periodic-with-Phase-G-period-prior, RBF, periodic+linear-with-Phase-G-period-prior
- [ ] ARD with d_max = 5; pruned dimensionality reported per concept
- [ ] Two-seed convergence check; flag instability where Δ log p > 1 nat
- [ ] Marginal log-likelihood comparison table (5-nat decisive threshold)
- [ ] Kernel-winner declared per (concept × population × layer)
- [ ] Geodesic-distance matrices saved
- [ ] Disconnection-diagnostic gate respected: skip GPLVM where >1 component, switch to mixture-of-GPLVMs or topology-only

**RBF VAE (Step 5, conditional):**
- [ ] Decision: invoked only if GPLVM is unstable or as supplementary; skipped if GPLVM converges cleanly
- [ ] If invoked: implementation following Hauberg σ⁻¹ via positive RBF (not σ⁻²); k-means center init; 10× lower LR for centers; toy validation
- [ ] If invoked: curated-set fits with d=5 latent and ARD
- [ ] If invoked: pullback-metric geodesics
- [ ] If invoked: agreement statistics with GPLVM, framed as "scale validation," not "cross-validation"

**Persistent homology (Step 6, primary independent validation):**
- [ ] H₁ persistence diagrams per (concept × population × layer) where Fourier or GPLVM detects structure
- [ ] Permutation null computed (1,000 shuffles)
- [ ] Significant-vs-noise classifications applied (95th percentile × 1.5; sensitivity at 1.0×, 1.5×, 2.0× in appendix)
- [ ] Topology-vs-Fourier consistency table
- [ ] Multi-loop test for the Bai et al. prism hypothesis (count of significant H₁)

**Causal experiment 1 — subspace ablation (Step 7):**
- [ ] Ablation infrastructure (forward-pass projection wrapper, random-Grassmannian generator)
- [ ] carry_1 ablation results across the L5 curated set
- [ ] 100-trial random-subspace control with 95% CI
- [ ] Irrelevant-concept control
- [ ] Per-layer ablation curves at {4, 8, 12, 16, 20, 24, 28, 31}
- [ ] Per-example diagnostic for qualitative shifts
- [ ] Pre-registered fallback (orthogonalized-carry-only ablation) if aggregate ablation collapses output

**Causal experiment 2 — helix rotation (Step 8):**
- [ ] Helix calibration on ans_digit_5_msf (circle fit, R² ≥ 0.8 required)
- [ ] Δθ_step ≈ 36° verified
- [ ] Rotation infrastructure (project, rotate, re-inject)
- [ ] Real-data rotation runs across the L5 correct curated set
- [ ] Three control variants: random rotation, off-axis, sign-flip (negative rotation)
- [ ] Predicted-shift accuracy reported (> 25% threshold)
- [ ] Mean-angular-shift regression (slope ≈ 1)

**Difficulty match (Step 9):**
- [ ] Phase G re-run on matched correct/wrong pairs at L5 and L4
- [ ] Branch 1 / 2 / 3 outcome documented per cell
- [ ] Functional-claim wording finalized per branch
- [ ] Disconnection diagnostic: shattered vs sparse comparison between correct and wrong manifolds

**Cross-method table (Step 10):**
- [ ] One row per (concept × population) cell
- [ ] Columns: Fourier, GPLVM, RBF VAE (if run), persistent homology, ablation, rotation, difficulty-matched
- [ ] Verdict column populated per pre-registered rule

**Figure and writing (Steps 11, 12):**
- [ ] Case-study figure with all 5 panels (Phase C, Phase G, GPLVM, persistent homology, ablation+layer-localization)
- [ ] Methods section
- [ ] Introduction and related work (cite Hauberg, Bergamin et al., Yu et al., Gurnee et al., Bai et al., Li et al., Park et al., Nanda et al., K&T, Zhou et al.)
- [ ] Results organized by claim
- [ ] Limitations honest and complete (manifold-hypothesis defense, stratification caveat, Bai et al. transferability, SiLU formalism)
- [ ] Discussion connecting back to LRH framing
- [ ] Conclusion and future work
- [ ] References cleaned and verified
- [ ] Appendices: coverage report, full per-method tables, computational details, pre-registration table B.15, persistence-diagram sensitivity analysis
- [ ] Internal review done
- [ ] Submission package assembled

**Reproducibility (post-submission):**
- [ ] Hugging Face model identifier added to `config.yaml` (currently only the local mount path)
- [ ] Curated set published
- [ ] Code released
- [ ] Model hash and dependencies documented
- [ ] Public landing page with reproducibility instructions

This checklist is the dashboard. Phase A through G are complete; Steps 1–12 are pending. The dependency graph in B.13 specifies the order; B.14 lists the failure modes; B.15 lists the pre-registered thresholds. The work proceeds ad hoc — one step at a time, in dependency order, with each step's success or pre-registered fallback determining the framing of the next.
