# Arithmetic Geometry

The Linear Representation Hypothesis (LRH) claims that concepts in LLMs are encoded as linear directions in activation space. We show that LRH is necessary but insufficient for understanding compositional reasoning. Linear subspace methods accurately capture *atomic* concepts — individual digits, carries, partial products — but fundamentally miss how these concepts *compose*. The actual computational structure lives on nonlinear manifolds *within* those linear subspaces: a 9-dimensional digit subspace may contain a 1-dimensional circle encoding digit identity via Fourier features; computation operates as geometric transformations (rotations, translations) on these manifolds; and failure occurs when these transformations break. A linear probe finds the 9D room but completely misses that the data sits on a 1D circle inside it.

We use multi-digit multiplication in Llama 3.1 8B as a controlled testbed. The computational graph is fully known (partial products, column sums, carries, answer assembly), difficulty scales from trivial to near-impossible, and errors are structured — so we can study exactly where and how composition breaks down, and build the nonlinear geometric tools to characterize it.

## Research Question

The LRH finds the ingredients. But does it find the recipe? Multiplication requires composing multiple intermediate steps — carry propagation across columns — and it is precisely this compositional structure that breaks in the model's representations. Linear probes can detect failure (correct vs. wrong) but cannot explain it. We ask: what is the geometric structure *within* concept subspaces, how do these structures interact during composition, and what changes geometrically when computation fails?

Bai et al. (2025) found Fourier-basis digit representations in tiny 2-layer models. Gurnee et al. (2025) found helical manifolds in Claude 3.5 Haiku. Qian et al. (2024) showed carry propagation is the bottleneck. Nobody has systematically tested whether these structures exist in a large pre-trained model, mapped the full computational graph geometrically, or characterized what changes in the manifold geometry between correct and wrong answers. We fill that gap using Llama 3.1 8B with a difficulty gradient from trivial (1x1 digit, 100% accuracy) to near-impossible (3x3 digit, 6.1% accuracy).

## Dataset

| Level | Type | Problems | Correct | Wrong | Accuracy |
|-------|------|----------|---------|-------|----------|
| 1 | 1x1 digit | 64 | 64 | 0 | 100.0% |
| 2 | 2x1 digit | 4,000 | 3,993 | 7 | 99.8% |
| 3 | 2x2 digit | 10,000 | 6,720 | 3,280 | 67.2% |
| 4 | 3x2 digit | 10,000 | 2,897 | 7,103 | 29.0% |
| 5 | 3x3 digit | 122,223 | 4,197 | 118,026 | 3.4%* |

**146,287 problems total. 17,871 correct, 128,416 wrong.** Levels 3-5 are the effective range for correct/wrong geometric comparison.

*L5 uses a two-phase approach: all 810,000 possible 3-digit x 3-digit problems were screened for correctness (6.11% accuracy on the full space), then a carry-balanced subset was selected targeting 500 correct per carry_0 value. The 3.4% dataset accuracy is lower than the 6.11% true accuracy because the selection deliberately oversamples high-carry problems where the model fails. **Report 6.11% as the model's L5 accuracy.**

## Model

- **Llama 3.1 8B base** (not instruct) — base model avoids RLHF confounds
- 32 transformer layers, 4096-dim hidden state, loaded in bfloat16 (~16 GB VRAM)
- Activations extracted at layers **[4, 6, 8, 12, 16, 20, 24, 28, 31]** spanning early/mid/late
- Residual stream captured at the `" ="` token (last prompt position, attends to full input)
- Prompt format: `"{a} * {b} ="` — MSF to match pre-training distribution
- Greedy decoding (`do_sample=False`, deterministic) — verified bit-identical across screening and pipeline runs

## Labels

Every problem gets algorithm-agnostic mathematical labels: all pairwise partial products, column sums grouped by output position (`i+j=k`), carries and running sums via LSF propagation, answer digits in both MSF and LSF order, and per-digit difficulty annotations (partial product count, max column sum, carry chain length). These are mathematical facts about the product, not steps of any specific algorithm. The model may use long multiplication, Fourier-space computation, or something entirely alien — the labels test whether it represents these quantities regardless of method. All 146,287 label sets are verified against tight per-column carry bounds.

For L5 carry analysis, rare carry values are binned: carry_1 >= 12, carry_2 >= 13, carry_3 >= 9, carry_4 >= 5 are grouped into single "high" classes. This ensures >= 100 correct samples per class for Phase C/D. Raw labels store exact values; binning is applied only by analysis code.

## Pipeline Status

| Stage | Status | Script | Output |
|-------|--------|--------|--------|
| Data Generation | Complete | `pipeline.py`, `generate_l5_problems.py`, `analysis.py` | 45 activation files (20.09 GB), labels, error analysis |
| L5 Screening | Complete | `generate_l5_problems.py` | 810K evaluations cached, 122,223 selected |
| Phase A: Visual Reconnaissance | Complete (L1-L5) | `phase_a_embeddings.py`, `phase_a_analysis.py` | 351 UMAP/t-SNE embeddings, interestingness scores |
| Phase B: Concept Deconfounding | Complete | `phase_b_deconfounding.py` | 2,677 classified pairs, correlation matrices, deconfounding plan |
| Phase C: Concept Subspaces | Complete (rerun Mar 19) | `phase_c_subspaces.py` | 2,844 subspaces, 2,750 significant (96.7%), GPU-accelerated |
| Phase D: LDA Refinement | Complete (L1-L5, 18h on 4×A6000) | `phase_d_lda.py`, `run_phase_d.sh` | 1,035 LDA results, 1,026 significant (99.1%), 247 plots |
| Phase E: Residual Hunting | Complete (L1-L5) | `phase_e_residual_hunting.py`, `run_phase_e.sh` | Union bases, var_explained 80.8-96.6%, Marchenko-Pastur validation |
| Phase F/JL: Between-Concept Angles & JL Check | Complete (L1-L5, 17.1h on A6000) | `phase_f_jl.py`, `run_phase_f_jl.sh` | 42,049 angle pairs, 39,525 superposition flags, 99 JL slices, 53 plots |
| Phase G: Fourier Screening | **Complete** | `phase_g_fourier.py`, `phase_g_kt_pilot.py`, `run_phase_g.sh` | 3,480 cells, 500 helix, 1 circle; 458 floor-saturated; carries dominant (419/500 = 83.8%) |
| Phase G: Number-Token Screening | **Complete** (0/108) | `phase_g_numtok_fourier.py`, `run_phase_g_numtok.sh` | K&T digit helix absent in multiplication context |
| Step B.1: Curated Set v1 | **Complete** | `build_curated_set.py` | 8,264 problems (L3/L4/L5), 2,400 difficulty-matched pairs, 0 undocumented gaps; 22 MB JSON |

## Key Findings

### Data Generation

- **Accuracy gradient works**: monotonic drop from 100% (1x1) to 6.11% (3x3, full space)
- **U-shaped per-digit accuracy**: first and last digits are easy (80-99%), middle digits collapse (13.0% at L5 position 3) — confirms Bai et al.'s carry-chain bottleneck in a production model
- **Carries dominate difficulty**: monotonically decreasing accuracy with carry count at every level
- **Errors are structured, not random**: median relative error 0.24% at L5; 83-95% close arithmetic
- **Even error bias**: 90-95% of errors preserve the ground truth's parity
- **L5 exhaustive screening**: 810,000 problems evaluated, 49,504 correct (6.11%), 25 carry values at hard mathematical ceilings

### Phase A: Visual Reconnaissance

- **Layer 16 is the information peak**: most label variables create visible clusters here
- **Product magnitude is the dominant axis**: must be controlled in downstream analysis
- **Correct/wrong answers separate clearly at L3-L5**: maximum divergence score |Δ| = 0.695
- **CKA reveals layer groups**: layers 20-24-28 are globally similar (CKA > 0.98), but concept-specific subspaces can still rotate between them

### Phase B: Concept Deconfounding

- **2,677 correlated pairs classified across all levels**, zero unexplained — every pair above |r| = 0.1 falls into a known category
- **Product residualization is sufficient**: no additional multi-concept deconfounding needed for Phase C's primary ("all") population
- **Suppression effect discovered**: product residualization creates r ≈ -0.80 anti-correlation between leading-digit pairs (a_tens↔b_tens at L3, a_hundreds↔b_hundreds at L5) that does not exist in raw data. Label-level impact is 63% shared variance; activation-level impact is estimated ~3% because product occupies 1 of 4096 dimensions while digit encodings use 8+
- **L5 sampling bias is small**: carry-stratified sampling induces r = +0.20 between a_units and b_units (below action threshold in "all" population; rises to r = +0.34 in the 4,197-problem correct-only population)
- **Structural correlations dominate**: 92-204 pairs per level are arithmetic relationships (carry chains, column-sum dependencies, digit-to-partial-product links). These are features of multiplication, not confounds

### Phase C: LRH Works for Atomic Concepts (confirmed with 17x scale-up)

- **2,844 subspaces identified, 2,750 significant (96.7%)** by permutation null (1,000 shuffles, α=0.01) — up from 88.6% in the first run, driven by L5
- **Input digits use maximum rank at all levels, layers, and populations**: a_units (10 values) → 9D subspace, a_tens (9 values) → 8D, all with CV > 0.985. Confirmed in the L5 correct population (4,197 samples) — the "vanishing subspaces" from the first run were sample-size artifacts
- **Column sums are strongly represented**: col_sum_0 through col_sum_4 all significant at maximum rank (CV 0.998-1.000 at L3), confirming the model tracks intermediate computation
- **Carries survive in the correct population**: carry_0 through carry_4 are significant at all layers even for L5 correct. The first run showed carry_0 with dim_perm=0 at layers 16+ — now dim_perm=8 everywhere. The model maintains full intermediate computation representations for problems it gets right
- **Partial products, product magnitude** — all have significant linear subspaces. The LRH is validated for every atomic concept we tested, including in the failure regime

### Phase C: LRH Fails for Composition (confirmed null results)

- **Middle answer digits have no linear subspace at L5 correct**: ans_digit_1 and ans_digit_2 have dim_perm=0 at every single layer with 4,197 samples. This is not a power issue — it is confirmed with ~300-900 samples per digit value. The model has all the ingredients (digits, carries, column sums) in clean linear subspaces, but the composed middle output digits are not linearly encoded
- **The bottleneck is output, not representation**: the model maintains full-rank input digits, carries, column sums, and partial products for correct L5 problems. Only middle answer digits are absent. The computation pipeline is intact; the failure is in composing intermediates into outputs for the hardest positions
- **Edge-vs-middle asymmetry scales with difficulty**: L3 has 1 weak answer position, L4 has 2, L5 has 2-3. The middle digits requiring carry chain propagation are hardest. Leading (magnitude) and trailing (modular arithmetic) digits are always represented
- **Correct/wrong divergence peaks at the output stage**: principal angles between correct and wrong subspaces are 10-18° for input digits, 13-31° for intermediates, 38-48° for middle answer digits. The failure manifests in output encoding, not input encoding
- **Subspace dimensionality is flat across layers**: carries hold the same linear rank from layer 4 to layer 31, yet the model is clearly computing with them. The computation transforms nonlinearly between layers while projecting to the same linear shadow

### Phase E: The Union Subspace Captures 81-97% of Activation Variance

- **Union subspace dimensionality scales with difficulty**: k ≈ 240 (L2), 380 (L3), 490 (L4), 540 (L5) — sub-linear growth despite 8× increase in arithmetic complexity
- **Variance explained ranges from 80.8% (L5/layer06) to 96.6% (L2/layer04)** across all 36 level×layer combinations
- **440 residual eigenvalues above Marchenko-Pastur at L5**: the residual is not pure noise — there are structured components, including Spearman >> Pearson nonlinear encoding signatures for partial product interactions
- **The residual variance is isotropic**: spread across ~3,500 dimensions, each residual dimension carries 28-82× less variance than each projected dimension

### Phase F/JL: Concepts Share Infrastructure, and the Subspace Is Geometrically Complete (subspace-finding pipeline complete)

- **Universal superposition: 39,525 of 42,049 concept pairs (94.0%) share subspace dimensions** significantly below random baselines. At L2, 100% of pairs; at L3-L5, 86-100% depending on layer and population
- **Algebraic gradient in angles**: carry/colsum/partial-product pairs (T2×T2) have mean θ₁ = 16-32° across levels; input/output digit pairs (T1×T3) have 39-51°. Concepts in the same computational chain share 2× more subspace structure
- **θ₁ ≈ 0° sanity checks pass**: col_sum_0 ↔ pp_a0_x_b0, carry_0 ↔ col_sum_0, and col_sum_4 ↔ pp_a2_x_b2 share directions exactly (θ₁ < 0.0001°) at every level — confirming Phase D bases capture real algebraic structure
- **Correct computations are more superposed**: correct population has mean θ₁ 26-35° vs wrong's 34-43° at L3-L5. The model packs concepts tighter when it gets the answer right
- **Near-perfect JL distance preservation**: Spearman correlations between full-space and projected distances range from 0.9942 to 0.9995 across all 99 slices. The union subspace preserves >98.7% of pairwise distance structure at every level, layer, and population
- **The variance-vs-distance gap confirms residual is noise**: Phase E's 3-19% residual variance translates to only 0.02-1.28% distance structure lost. The residual is isotropic — it changes activation magnitude but not relative positions
- **L5 passes the critical test**: with N=122,223 and 7.47 billion pairs per slice, Spearman ≥ 0.9942 at every layer. The nonlinear encoding detected by Phase E is real but low-amplitude — geometrically minor
- **43.9 billion pairwise distances computed, zero subsampling**: every pair at every level, layer, and population. Pythagorean validation errors at machine epsilon (1.5e-15 to 5.4e-15) across all 99 slices

### Phase G: Carries Have Helix Geometry, Operand Digits Do Not (Run 3 at 92%)

- **417 helix detections out of 3,084 completed analyses (13.5%)** across L2–L5, all layers, all populations. 1 circle detection, 2,666 none. Carries dominate: 311 of 417 helices (74.6%) come from carry_1 through carry_4
- **Carry helix structure is the dominant signal.** carry_1: 148/432 (34.3%), carry_2: 97/270 (35.9%), carry_3: 32/108 (29.6%), carry_4: 34/108 (31.5%). Most are floor-saturated (p_helix = 0.001, permutation null never produced an FCR this large). The generalized helix matches K&T's structure: circular Fourier component at period 10 + linear magnitude ramp
- **Operand digits are completely null at the `=` position.** 0/846 analyses detected any geometry for a_units, a_tens, a_hundreds, b_units, b_tens, b_hundreds. Not a power issue — N up to 122,223 with 1,000 permutations. The model does not use periodic representations for operand digits at the computation position, despite clean 8–9D linear subspaces (Phase C)
- **Difficulty-dependent emergence.** L2: 2/324 helix (0.6%). L3: 93/774 (12.0%). L4: 150/996 (15.1%). L5: 155/991 (15.6%). Helix geometry emerges exactly where carry propagation becomes error-prone
- **Layer uniformity.** Detection rates range 11.7%–15.1% across layers 4 through 31. No single layer dominates — the helix is a representational format maintained throughout the network, not a computation-layer artifact
- **Answer digit edge-vs-middle asymmetry replicates Phase C.** ans_digit_0 (leading): 15.2% helix. ans_digit_5 (trailing/ones): 38.9%. Middle digits (positions 1–2): 1.4%–2.6%. The ones digit — determined by purely modular arithmetic — has the strongest Fourier structure
- **K&T replication: PASSED.** Periods {2, 5, 10} confirmed in top-3 at layers {0, 1, 4, 8} for single-token integers 0–360. Synthetic pilot 10/10. Raw vs. residualized spot-check passed (FCR disagreement < 1%)

### Phase G Number-Token Screening: K&T's Digit Helix Is Context-Dependent (Complete)

- **0/108 detections across all levels, layers, and digit concepts.** 108 analysis cells (4 levels × 6 layers × 6 concepts), 1,000 permutations each, 636 seconds total. Zero helix, zero circle, zero FDR-significant. K&T's digit helix does NOT exist at operand token positions in multiplication context
- **b_units at layer 12 came closest** — L3: FCR=0.61, p=0.002; L4: FCR=0.55, p=0.004; L5: FCR=0.54, p=0.004. Global p-value passes α=0.01 but conjunction criterion fails (structure on one coordinate, not two). Consistent with partial linear trend, not a circle
- **Layer gradient: null everywhere.** Layer 4 and 8 (where K&T found strongest signal for standalone integers): null. Layer 12 (closest to detection): still null. Layers 16–24: null. The helix is absent at all depths, not just suppressed at late layers
- **K&T's helix is context-dependent.** Same model, same integers, different context → different geometry. Standalone integers sit on helices (K&T confirmed). Operand digits in multiplication do not — neither at the number token (0/108) nor at the `=` token (0/846). The model transforms representations based on computational task, not token identity
- **The three-way comparison tells the story:** (1) Standalone integers → helix (K&T replicated); (2) Operand digits at number token → null (0/108); (3) Carries at `=` → helix (311/918, 33.9%). Periodic structure is destroyed for inputs but rebuilt for intermediate computations

### Step B.1: Curated Set v1 (Complete)

- **8,264 problems selected** from the L3, L4, L5 pools (L1 and L2 excluded — L1 is uniformly correct, L2 has only 7 wrong examples). Per-level: L3 = 1,200 c / 1,201 w / 2,401 t; L4 = 1,000 c / 1,846 w / 2,846 t; L5 = 1,401 c / 1,616 w / 3,017 t. 22 MB JSON at `/data/user_data/anshulk/arithmetic-geometry/curated/curated_set_v1.json`
- **2,400 difficulty-matched pairs** (1,000 at L4 + 1,400 at L5). Matching is strict on (`magnitude_tier`, `carry_count_tier`, `answer_length`); only `carry_count_tier ±1` relaxes on miss. **993/1,000 L4 pairs strict + 7 relaxed; 1,399/1,400 L5 pairs strict + 1 relaxed; 0 unmatched at both levels.** Mean correct−wrong difference per axis at L5: leading_digit_pair_index = 0.008, nonzero_carry_count = −0.004, answer_length = 0.0. Matched pairs are functionally indistinguishable on every axis
- **All 17 concepts × per-value floor of 30 examples met where math allows.** 0 below-floor non-documented cells. 19 below-floor documented cells (3 mathematically excluded — operand-tens=0 at L3/L4; 16 pool-limited — extreme high-carry values whose underlying pool has < 30 examples)
- **Reproducibility:** seed = 42, single threaded RNG, sha256 of script + config + output JSON recorded in metadata. Re-running with same seed yields byte-identical output. Build runtime: 181.6s on the login node
- **Schema:** every problem record carries `(curated_id, level, source_index, a, b, product, predicted, correct, raw_text, magnitude_tier, carry_count_tier, answer_length, nonzero_carry_count, matched_pair_id, matched_relaxed, labels)`. The `source_index` resolves directly into `level{level}_layer{layer}.npy[source_index]` for activation lookup. The full `compute_labels(a, b)` dict is nested under `labels`

## Project Structure

```
arithmetic-geometry/                       (workspace, in git)
├── pipeline.py                            # generate problems, extract activations, evaluate
├── generate_l5_problems.py                # L5 two-phase screening + carry-balanced selection
├── analysis.py                            # error pattern analysis (CPU only)
├── phase_a_embeddings.py                  # UMAP/t-SNE embeddings + interestingness scoring
├── phase_a_analysis.py                    # Phase A summary analysis
├── phase_b_deconfounding.py                # label-level correlation diagnostics (CPU, <1 min)
├── phase_c_subspaces.py                   # concept subspace identification via cond. covariance + SVD (GPU-accelerated)
├── phase_d_lda.py                         # LDA refinement of Phase C subspaces (GPU-accelerated)
├── phase_e_residual_hunting.py            # union subspace + variance decomposition (GPU)
├── phase_f_jl.py                          # between-concept angles + JL distance check (GPU)
├── phase_g_kt_pilot.py                    # K&T replication: Fourier on single-token integers (GPU)
├── phase_g_fourier.py                     # Fourier screening: periodic structure in subspaces (CPU)
├── extract_number_token_acts.py           # number-token activation extraction (GPU)
├── phase_g_numtok_fourier.py              # number-token Fourier screening (CPU)
├── build_curated_set.py                   # B.1: build the 8,264-problem curated set with matched pairs (CPU, ~3 min)
├── config.yaml                            # all parameters, paths, model config
├── run.sh                                 # SLURM: main pipeline (GPU, ~14 min)
├── run_l5_screen.sh                       # SLURM: L5 screening (GPU, ~50 min)
├── run_phase_b.sh                         # SLURM: Phase B deconfounding (CPU, <1 min)
├── run_phase_c.sh                         # SLURM: Phase C subspaces (GPU, ~3.5 hours)
├── run_phase_d.sh                         # SLURM: Phase D LDA refinement (4×GPU, ~18 hours L5)
├── run_phase_e.sh                         # SLURM: Phase E residual hunting (GPU)
├── run_phase_f_jl.sh                      # SLURM: Phase F/JL angles + distances (GPU, ~17 hours)
├── run_phase_g.sh                         # SLURM: Phase G full pipeline (GPU+CPU, ~8 hours)
├── run_phase_g_numtok.sh                  # SLURM: number-token Fourier screening (CPU, ~1-2 hours)
├── labels/                                # per-level labels + analysis summary
│   ├── level_{1-5}.json                   # L5 is 160 MB (122,223 problems)
│   └── analysis_summary.json
├── plots/                                 # all generated plots
│   └── *.png                              # 9 data generation diagnostic plots
├── logs/                                  # execution logs and SLURM output
└── docs/
    ├── datageneration_analysis.md         # complete data generation reference
    ├── phase_a_analysis.md                # complete Phase A reference
    ├── phase_b_analysis.md                # complete Phase B reference
    ├── phase_c_analysis.md                # complete Phase C reference
    ├── phase_d_analysis.md                # Phase D LDA reference
    ├── phase_e_analysis.md                # Phase E residual hunting reference
    ├── phase_f_jl_analysis.md             # Phase F/JL reference
    ├── phase_g_analysis.md                # Phase G Fourier screening reference
    ├── next_steps.md                      # Part B execution plan (B.1 done; B.2-B.12 pending)
    └── curated_set_coverage_report.md     # B.1 build report: tables, gaps, matched-pair diagnostics, reproducibility manifest

/data/user_data/anshulk/arithmetic-geometry/  (heavy files, not in git)
├── model/                                 # Llama 3.1 8B weights (~30 GB)
├── l5_screening/                          # L5 two-phase outputs
│   ├── l5_evaluation_cache.npz            # 810K evaluations (2.4 MB)
│   └── l5_selected_problems.json          # 122,223 selected problems (1.2 MB)
├── activations/                           # 45 .npy files, 20.09 GB total
│   └── level{N}_layer{L}.npy             # shape (n_problems, 4096), float32
├── answers/                               # per-level predictions + correctness (20.7 MB)
│   └── level_{1-5}.json
├── phase_a/                               # Phase A outputs
│   ├── coloring_dfs/                      # 5 .pkl coloring DataFrames
│   ├── embeddings/                        # 117 CSVs with 2D coordinates
│   └── interestingness/                   # scoring results
├── phase_b/                               # Phase B outputs
│   ├── correlation_matrices/              # 26 CSV files (raw + residualized per level/pop)
│   ├── classified_pairs.csv               # 2,677 pairs with classification + action
│   ├── deconfounding_plan.json            # per-level confound lists for Phase C
│   └── summary.json                       # aggregate statistics
├── phase_c/                               # Phase C outputs (26 GB)
│   ├── residualized/                      # 45 product-residualized activation files (21 GB)
│   ├── subspaces/                         # concept subspaces (basis, eigenvalues, null) (5.5 GB)
│   │   └── L{N}/layer_{LL}/{pop}/{concept}/
│   └── summary/                           # master CSVs (results, divergence, alignment)
├── phase_d/                               # Phase D LDA outputs
│   └── L{N}/layer_{LL}/{pop}/{concept}/   # merged bases, eigenvalues, metadata
├── phase_e/                               # Phase E outputs
│   └── L{N}/layer_{LL}/{pop}/             # union bases, var_explained, eigenvalue spectra
├── phase_f/                               # Phase F/JL outputs
│   ├── principal_angles/                  # pairwise angle CSVs per (level, layer, pop)
│   ├── jl_check/                          # JL metrics per (level, layer, pop)
│   └── summary/                           # phase_f_principal_angles.csv (42,049 rows),
│                                          # superposition_summary.csv (39,525 rows),
│                                          # jl_distance_preservation.csv (99 rows)
├── phase_g/                               # Phase G Fourier screening outputs
│   ├── kt_pilot/                          # K&T replication results
│   ├── fourier/                           # per-concept Fourier results (=position)
│   │   └── L{N}/layer_{LL}/{pop}/{concept}/  # FCR, p-values, detection flags
│   ├── numtok/                            # number-token Fourier results
│   │   └── L{N}/layer_{LL}/pos_{a|b}/    # per-concept JSONs
│   └── summary/                           # checkpoint + final CSVs
│       ├── fourier_results.csv            # main screening summary (pending)
│       └── numtok_fourier_results.csv     # number-token summary (with FDR)
├── activations_numtok/                    # number-token position activations (9.2 GB)
│   └── level{N}_layer_{LL}_pos_{a|b}.npy  # 48 files, float16, shape (N, 4096)
└── curated/                               # B.1 curated set (22 MB)
    └── curated_set_v1.json                # 8,264 problems with full labels + matched-pair metadata; sha256 recorded in build log
```

## Setup

```bash
conda create -n geometry python=3.11 -y
conda activate geometry
pip install torch numpy scipy pandas scikit-learn matplotlib seaborn pyyaml \
            transformers accelerate huggingface_hub tqdm joblib umap-learn

# Download model locally
python -c "
from huggingface_hub import snapshot_download
snapshot_download('meta-llama/Meta-Llama-3.1-8B',
                  local_dir='/data/user_data/anshulk/arithmetic-geometry/model')
"
```

## Usage

```bash
# Step 1: L5 screening (A6000 GPU, ~50 minutes — run once, cached)
sbatch run_l5_screen.sh

# Step 2: Main pipeline (A6000 GPU, ~14 minutes)
sbatch run.sh

# Phase B: concept deconfounding (CPU only, <1 minute)
sbatch run_phase_b.sh

# Phase C: concept subspaces (GPU-accelerated, ~3.5 hours)
sbatch run_phase_c.sh

# Phase D: LDA refinement (4×A6000 GPU, ~18 hours for L5)
sbatch run_phase_d.sh

# Phase E: Residual hunting (GPU, ~2 hours)
sbatch run_phase_e.sh

# Phase F/JL: Between-concept angles + JL distance check (A6000 GPU, ~17 hours)
sbatch run_phase_f_jl.sh

# Phase G: Fourier screening (A6000 GPU + CPU, ~8 hours)
sbatch run_phase_g.sh

# Phase G: Number-token Fourier screening (CPU only, ~1-2 hours)
sbatch run_phase_g_numtok.sh

# B.1: Build the curated set (CPU, ~3 minutes; outputs JSON + coverage report + log)
python build_curated_set.py            # refuses to overwrite existing v1
python build_curated_set.py --force    # re-runs (deterministic, byte-identical given seed=42)

# Or run any phase with pilot mode first (L3/layer16 only, ~2 minutes)
python phase_c_subspaces.py --config config.yaml --pilot
python phase_g_numtok_fourier.py --config config.yaml --pilot
```

## Documentation

- **[Data Generation Stage Analysis](docs/datageneration_analysis.md)** — Every design decision, full math walkthroughs, tokenization patterns, carry bound proofs, error classification, carry binning decisions, all results verified against logs
- **[Phase A Analysis](docs/phase_a_analysis.md)** — UMAP/t-SNE embedding methodology, interestingness scoring, CKA matrices, activation norm profiles, priority list for downstream phases
- **[Phase B Analysis](docs/phase_b_analysis.md)** — Label-level correlation diagnostics, the suppression effect discovery, four-category classification system, deconfounding plan, all 2,677 pairs verified
- **[Phase C Analysis](docs/phase_c_analysis.md)** — Conditional covariance + SVD methodology, permutation null validation, eigenvalue spectra, cross-layer alignment, correct/wrong divergence, all 2,844 subspaces documented. Includes first-run vs rerun comparison showing which findings survived the 17x L5 scale-up
- **[Phase F/JL Analysis](docs/phase_f_jl_analysis.md)** — Principal angles between all concept pairs, superposition detection, JL distance preservation across 99 slices with 43.9 billion pairwise distances, variance-vs-distance gap analysis, complete L2-L5 results. Marks the completion of the subspace-finding pipeline
- **[Phase G Analysis](docs/phase_g_analysis.md)** — Fourier screening for periodic structure (circles, helices) within concept subspaces. K&T replication, synthetic pilot validation, full mathematical framework, Run 3 results (500 helix detections, carries dominant), number-token screening design and pilot results, interpretation of position-dependent representations
- **[Curated Set Coverage Report](docs/curated_set_coverage_report.md)** — Step B.1 truth document: full per-position digit/carry coverage tables across L3/L4/L5, matched-pair diagnostics (1,000 + 1,400 pairs, 0 unmatched, mean diffs ≈ 0), 19 documented hard-ceiling gaps with mathematical justification, eight-check verification, reproducibility manifest. 1,589 lines
- **[Next Steps](docs/next_steps.md)** — Part B execution plan: pre-registered thresholds, dependency graph, failure modes, the eleven steps that follow B.1 (within-group PCA, orthogonalization, GPLVM, persistent homology, the two causal experiments, difficulty-matched validation, cross-method table, case-study figure, paper writing)

## What's Next: Part B (depth-of-structure on the curated set)

**Phases A through G are complete. Step B.1 (curated set v1) is complete. Steps B.2 through B.12 are pending.**

Every atomic concept has a clean linear subspace (Phase C/D). The union of 43 concept subspaces captures >98.7% of pairwise distance structure (Phase E/JL). Concepts share dimensions in proportion to their algebraic relationship (Phase F). Composed middle answer digits lack linear subspaces at L5 (Phase C). Phase G revealed the first nonlinear structure: carries sit on generalized helices inside their linear subspaces (500 helix detections / 3,480 cells, 458 floor-saturated, FDR-significant), while operand digits at the `=` position are completely null. The 8,264-problem curated set with 2,400 difficulty-matched pairs (Step B.1) is the input to every remaining Part B step. See [`docs/next_steps.md`](docs/next_steps.md) for the full execution plan and pre-registered thresholds.

- **B.2 Within-group PCA + disconnected-manifold diagnostic**: settle whether centroid analysis is sound for each (concept, value) cell, and whether single-GPLVM is admissible (k-NN connectivity check on centroids). Pre-registered green/yellow/red interpretation of WGCR + Hartigan dip
- **B.3 Orthogonalization control for superposition**: project carry activations onto the orthogonal complement of their algebraic correlates (col_sum, partial products) and re-run Phase G's conjunction test. Quantifies how much of the helix belongs to the carry's own representation vs. inherited from shared subspaces
- **B.4 GPLVM as the primary non-linear manifold method (HIGH PRIORITY)**: exact Bayesian GPLVM with ARD on the curated set. Phase G periods (18, 27, 19, 10) plug in as kernel priors so the comparison becomes a falsification of the periodic hypothesis under the more flexible probabilistic framework, not a search over unknown periods. Comparison of periodic / RBF / periodic+linear via marginal log-likelihood at the K&R "decisive" threshold (≥5 nats ≈ 2.17 log₁₀)
- **B.5 RBF VAE as a scalable surrogate (downgraded)**: shares GPLVM's uncertainty mechanism, so agreement is expected by construction; runs only if GPLVM is infeasible
- **B.6 Persistent homology as primary independent validation (elevated)**: H₀/H₁ persistence diagrams using `gudhi` or `ripser` on the curated centroids. Different math (combinatorial topology) from GPLVM (Bayesian inference); genuine cross-method evidence. Pre-registered 1.5× null-floor threshold with sensitivity sweep at 1.0×, 1.5×, 2.0×
- **B.7 Causal experiment 1 — subspace ablation on carry_1**: project residual stream onto carry_1's orthogonal complement at layer 16 and continue the forward pass. 100-trial random-Grassmannian control + irrelevant-concept control. Compute budget pre-committed: ~1.47M forward passes, ~20 GPU-hours
- **B.8 Causal experiment 2 — helix rotation on the trailing answer digit**: the surgical experiment. Calibrate Δθ_step on the 10 ans_digit_5_msf centroids in the curated set, rotate by one digit-step, measure whether the model's output increments (v+1) mod 10. Pre-registered ≥25% predicted-shift threshold (vs 10% chance). Compute budget: ~1 GPU-hour
- **B.9 Difficulty-matched correct/wrong validation**: re-run Phase G on the 2,400 matched pairs to test whether the correct/wrong helix asymmetry survives controlling for difficulty. Pre-registered Branch 1/2/3 framings ready in advance
- **B.10–B.12**: cross-method consistency table, case-study figure (carry_1 at L5 layer 16), paper writing
