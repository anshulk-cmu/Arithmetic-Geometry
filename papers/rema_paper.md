# Analysis: REMA — A Unified Reasoning Manifold Framework for Interpreting Large Language Models

**Paper:** REMA: A Unified Reasoning Manifold Framework for Interpreting Large Language Model  
**Authors:** Bo Li (Tsinghua/Baidu), Guanzhi Deng, Ronghao Chen, Junrong Yue, Shuo Zhang, Qinghua Zhao, Linqi Song, Lijie Wen  
**Affiliations:** Tsinghua University, City University of Hong Kong, Peking University, BUPT, Beihang University, Baidu Inc.  
**Venue:** Submitted to ICLR 2026, **Withdrawn** (January 26, 2026)  
**arXiv:** 2509.22518 (September 26, 2025)  
**Type:** Empirical / Methods paper  
**Purpose:** Deep understanding — this paper is directly relevant to our geometric interpretability pipeline for Llama 3.1 8B, particularly the manifold hypothesis and geometric deviation analysis  
**Peer-Reviewed:** No — submitted to ICLR 2026 but withdrawn before decision. OpenReview reviews are not publicly accessible for this withdrawn submission.

---

---

# PHASE -1: Paper Classification

This is an **Empirical / Methods** paper. It proposes a framework (REMA) for analyzing LLM reasoning failures by comparing the geometry of internal representations between correct and incorrect reasoning. There are no theorems or proofs — just a hypothesis, a methodology, and experiments. The classification matters because we should evaluate it primarily on experimental rigor, not theoretical depth.

**Reading strategy:** Visual-First Pass 1 (figures are the argument), then standard Pass 2 for methodology critique, then Empirical Pass 3 variant for stress-testing the claims.

---

# PHASE 0: Pre-Reading Context

**Who are the authors?** Mixed group from Chinese universities (Tsinghua, Peking, CityU HK) plus Baidu. First three are listed as equal contribution. No established track record in interpretability that I can find — the senior authors (Zhao, Song, Wen) work in NLP and software engineering. This is relevant: they're applying geometric analysis without deep roots in the manifold learning or interpretability literature.

**Where submitted?** ICLR 2026 — a top venue. The paper was withdrawn, which could mean anything from "reviews were bad" to "authors found an error" to "they want to revise and resubmit." Given it was an ICLR withdrawal, the reviews were likely unfavorable.

**Code available?** No — they say "Our code will be publicly available when the paper is accepted." Since the paper was withdrawn, no code is available. This is a red flag for reproducibility.

**Credibility assessment flags:**
- No released code
- Authors not established in interpretability
- Claims are broad ("unified framework," "new avenues for in-depth understanding")
- The paper uses only instruction-tuned models, not base models
- The "manifold" terminology is used loosely — more on this below

---

---

# PASS 1 — Jigsaw Puzzle (What does the paper do?)

## Q1: What is the problem being solved?

The paper asks: when an LLM gets a reasoning question wrong, can we detect that the internal representations went "off track" and pinpoint *where* (which layer) the derailment happened?

In one sentence: "This paper studies whether incorrect reasoning in LLMs corresponds to geometric deviation of internal representations from a 'correct reasoning manifold,' and whether the layer where deviation begins can be localized."

## Q2: Why is the problem interesting and nontrivial?

Understanding why LLMs fail at reasoning is important. Existing failure analysis methods either (a) require task-specific probes (e.g., hallucination detectors), or (b) need controlled input pairs (e.g., neutral vs. emotional prompts). A general, task-agnostic framework for detecting and localizing reasoning failures from internal representations would be valuable.

In one sentence: "This is nontrivial because existing failure diagnosis methods are task-specific, and a unified geometric framework for all reasoning failures doesn't exist."

**Devil's advocate on importance:** Is "errors deviate from correct representations" actually nontrivial, though? If a model gets a different answer, its internal states will naturally be different. The question is whether REMA tells us anything beyond "wrong answers have different representations than right answers."

## Q3: What is the main claim?

The paper makes three empirical claims:

1. Internal representations of both correct and incorrect reasoning live in low-dimensional subspaces (low intrinsic dimension).
2. Error representations are geometrically farther from correct representations than correct representations are from each other (measured by k-nearest-neighbor distance).
3. The layer where error representations first diverge significantly can be localized, and this divergence point varies by model and task.

In one sentence: "They show that erroneous reasoning representations are measurably farther from the correct reasoning region using kNN distance, with divergence beginning at identifiable layers."

## Visual-First Scan

**Figure 1 (ID and MI across layers):** Intrinsic dimension is 5-30, well below the ambient dimension of 2048-4096. MI between representations and correct answers is higher for correct samples in early layers. These are the most interesting results — they show real structure. But ID being low is expected from the manifold hypothesis and doesn't validate REMA specifically.

**Table 1 (Deviation distances):** Every model-task pair shows error distance > correct internal distance, with large t-statistics. The relative deviation is 20-73%. The Spearman correlation between accuracy and relative deviation (ρ = 0.598) is moderate.

**Figure 2 (UMAP):** Final-layer UMAP shows separation between correct and error. This is the weakest evidence — UMAP is notoriously unreliable for making claims about geometric structure. You can get UMAP to show clusters that don't really exist and miss clusters that do.

**Figure 3 (SVM separability across layers):** Classification accuracy increases with layer depth. This is interesting but expected — of course later layers are more task-specific.

**Figure 4 (Divergence point histograms):** Shows that divergence points are distributed across layers, not concentrated. This is the most novel finding, but the histograms look like they could be noise in some cases.

**Table 2 (Ablation on pooling):** Mean-pooling and attention-weighted pooling perform best. This is a useful methodological finding.

## Pass 1 Verdict

The core observation — that error representations are farther from correct ones — is probably true but potentially trivial. The more interesting question is whether the divergence point localization provides real mechanistic insight. The paper's biggest weakness is the gap between the "manifold" branding and what's actually computed (kNN distances). There's no real manifold learning here.

**Relevance to our work:** Moderate. The geometric perspective aligns with our pipeline, but the methodology is crude compared to what we're building (GP-based probabilistic geometry, conditional covariance, LDA). The paper uses only Euclidean distances in the full ambient space — no subspace identification, no Riemannian geometry, no uncertainty quantification.

**Decision:** Proceed to Pass 2. The claims need scrutiny and the connections to our pipeline need evaluation.

---

---

# PASS 2 — Scuba Dive (How does the paper work?)

## Prerequisite Concepts for a Grade 12 Student

Before diving into the paper's methodology, let me explain every concept from scratch. Assume you know basic algebra, vectors (lists of numbers with direction and length), and what a derivative is (rate of change).

### What is a "representation" or "hidden state"?

When a language model reads text and generates an answer, it doesn't jump straight from input to output. At each of its internal layers (think of them as processing stages), it creates a summary of what it has figured out so far. This summary is a long list of numbers — for Llama 3.1 8B, it's a list of 4096 numbers.

Each of these lists is a point in a very high-dimensional space. Just like a point on a map has 2 coordinates (latitude, longitude), a point in the model's "mental space" has 4096 coordinates.

When the model processes a math problem, the list of numbers at layer 0 is different from layer 10, which is different from layer 32. The representation *changes* as it passes through the layers, like a thought developing step by step.

### What is a "manifold"?

Imagine you live on the surface of the Earth. The Earth is a 2D surface (you only need latitude and longitude to locate yourself) that sits inside 3D space. The Earth's surface is a manifold — a low-dimensional shape that lives inside a higher-dimensional space.

The "manifold hypothesis" in machine learning says that real data, even though it sits in a very high-dimensional space (like 4096 dimensions), actually lies on or near a much lower-dimensional surface. Think of it this way: photos of human faces technically live in a million-dimensional space (one dimension per pixel), but the "space of all realistic faces" is much smaller because faces share common structure (two eyes, a nose, a mouth in roughly the same arrangement).

In this paper, the claim is that when an LLM correctly solves reasoning problems, the internal representations at each layer don't scatter randomly in 4096-dimensional space. Instead, they concentrate on a low-dimensional surface — the "reasoning manifold." The key word is *concentrate*: this is a statistical claim, not a hard geometric claim.

### What is "intrinsic dimension"?

If you have a bunch of data points that sit on a surface in high-dimensional space, the *intrinsic dimension* tells you the actual dimensionality of that surface. If the points lie on a line (even a curved one), the intrinsic dimension is 1. If they lie on a sheet (even a wrinkled one), the intrinsic dimension is 2. And so on.

The Earth analogy: you live in 3D space, but the surface you move on is 2D. The intrinsic dimension of the Earth's surface is 2.

The paper estimates intrinsic dimension using a method called TwoNN (Facco et al., 2017). The idea is beautifully simple:

1. For each data point, find its first nearest neighbor (distance r₁) and second nearest neighbor (distance r₂).
2. Compute the ratio μ = r₂/r₁.
3. If the data lives on a d-dimensional manifold, the ratio μ follows a specific mathematical distribution: P(μ > x) = x^(-d).
4. By fitting this distribution to the observed ratios, you estimate d.

Think of it like this: if you live on a 1D line, your nearest and second-nearest neighbors are almost always right next to each other on the line, so μ ≈ 1 mostly. If you live on a 2D surface, there are more directions neighbors can be in, so the ratio spreads out more. Higher dimension means the ratio distribution spreads even further. The shape of this spread tells you the dimension.

### What is "mutual information"?

Mutual information (MI) measures how much knowing one thing tells you about another thing. If knowing a student's height tells you nothing about their math grade, the MI between height and math grade is zero. If knowing someone's study hours tells you a lot about their exam score, the MI is high.

In the paper, they measure MI between the layer representations and the correct answer. High MI means: if you could read the representation, you'd know a lot about what the right answer is. Low MI means the representation has lost (or hasn't yet developed) information about the answer.

The paper uses the KSG estimator (Kraskov et al., 2004), which is a clever method based on counting nearest neighbors in both the representation space and the answer space. I'll spare the details, but the key point: this is a non-parametric estimator, meaning it makes minimal assumptions about what distributions look like.

### What is "k-nearest-neighbor distance"?

Given a point and a collection of other points, the k-nearest-neighbor (kNN) distance is the average distance to the k closest points in that collection. With k=5 (as used in this paper), you find the 5 closest points and average their distances to your query point.

This is the paper's core metric. For each error sample, they compute: "how far is this error representation from the 5 nearest correct representations?" They compare this to: "how far is each correct representation from the 5 nearest *other* correct representations?"

If error representations consistently have larger kNN distances to the correct set than correct representations have to each other, that's evidence that error representations are "outside" the region where correct representations live.

### What is mean pooling?

During generation, the model produces representations at each step of the output. If the model generates 20 tokens for a math answer, there are 20 representations at each layer. Mean pooling just averages all 20 into one, giving a single representative vector per sample per layer.

Think of it like this: instead of tracking the model's state at every word of its answer, you take the average state across the whole answer. This is a simplification — it throws away the temporal sequence and keeps only the "overall feel" of what the model was doing at that layer.

### What is the Welch t-test?

A statistical test that asks: "Are two groups of numbers drawn from distributions with different means?" If you have a bunch of distances from error samples and a bunch from correct samples, the t-test tells you whether the difference in their averages is "real" (statistically significant) or could just be random noise.

A large t-statistic means the difference is very unlikely to be due to chance. The t-statistics in this paper range from about 5 to 59, which are all extremely large — the differences are unambiguously real in a statistical sense. But statistical significance doesn't mean practical significance or mechanistic insight.

### What is an SVM (Support Vector Machine)?

A classifier that finds the best "wall" (hyperplane) separating two groups of points in high-dimensional space. With an RBF kernel (as used here), it can find curved boundaries, not just flat walls.

The paper trains SVMs to separate correct vs. error representations at each layer. If the SVM can do this accurately, it means the two groups occupy different regions of the representation space.

---

## The Core Methodology, Step by Step

### Step 1: Data Preparation (Section 3.1)

The paper takes standard reasoning benchmarks (GSM8K for arithmetic, MATH for math competition problems, GPQA for science, etc.) and runs models on them in zero-shot mode. For each sample:

1. The model generates an answer.
2. The answer is compared to ground truth by exact string matching.
3. The sample is labeled "correct" or "error."

At each layer l (from 0 to L-1), they extract the hidden state at the last token position during generation. They do this at every generation step, then average across all steps (mean pooling) to get one vector z^l_i per sample per layer.

**Critical question 1: Is mean pooling appropriate?**

Mean pooling throws away temporal structure. A model might reason correctly for 15 steps and then go wrong at step 16. Mean pooling would dilute the error signal from step 16 with the correct signals from steps 1-15. The paper addresses this in the ablation (Table 2) and finds mean pooling works well — but "works well for separability" doesn't mean "is best for understanding the process." The paper's own framing is about *reasoning processes*, but mean pooling discards the process and keeps only the average.

For our pipeline with Llama 3.1 8B on multiplication, this is particularly relevant. The carry digit computations happen at specific generation steps. Mean pooling would blur the representation of "got carry digit 3 right at step 5" with "got everything else right." We should be suspicious of any geometric claims based on mean-pooled representations.

**Critical question 2: Is exact match a good partition criterion?**

A model might follow perfect reasoning but make a minor arithmetic slip at the very end. Its representations through most layers would be indistinguishable from a correct sample. Exact match lumps these "almost correct" samples with samples that were completely confused from the start.

The paper acknowledges this in the limitations (Appendix E) but doesn't address it. A "soft scoring" or "partial correctness" partition would be more informative and is a clear weakness.

### Step 2: Manifold Characterization (Section 3.2)

The paper computes two properties at each layer:

**Intrinsic Dimension (ID):** Using TwoNN, they estimate how many "true" dimensions the data occupies. For Qwen3 (4B) on MATH, the ID ranges from about 6 to 14 across layers (vs. ambient dimension 2048). For Llama3.2 (11B) on SNLI-VE, ID ranges from about 18 to 33 (vs. ambient 4096).

**Mutual Information (MI):** Between layer representations and the correct answer (converted to sentence embeddings). MI is high early and decreases with depth.

**Critical analysis of ID claims:**

The intrinsic dimension being low is real and interesting, but it's not evidence for a "reasoning manifold" specifically. ANY task-specific processing will concentrate representations in subspaces — this is what neural networks *do*. A model encoding "cat" images will have low ID because cat images share common structure, not because there's a "cat manifold" with deep geometric meaning.

More problematic: both correct and error representations have similar ID profiles (Figure 1). If the "reasoning manifold" were a real thing that correct samples lived on and error samples didn't, you'd expect their IDs to be substantially different. Instead, the IDs are remarkably similar (e.g., Figure 1a: both correct and error ID fluctuate between 6 and 14, tracking each other closely). This actually *undermines* the manifold narrative — error representations seem to live in similarly-structured subspaces, just *different* ones.

**Critical analysis of MI claims:**

The MI finding that correct samples have higher MI with the answer in early layers is genuinely interesting. It suggests that from the very beginning, the model's processing of eventually-correct samples already contains more task-relevant information. However:

1. MI is estimated with the KSG estimator with k=5. On small sample sizes (some tasks have only ~100 correct or error samples), this estimator has high variance.
2. The answer is converted to a sentence embedding using all-MiniLM-L6-v2. This conversion is lossy and somewhat arbitrary — different embedding models would give different MI estimates.
3. MI decreasing with depth is expected: later layers are more specialized for the *next-token prediction* format rather than preserving raw answer information. This doesn't mean information is "lost" — it means it's been *transformed* into a format useful for generation, which is harder to extract with a generic MI estimator.

### Step 3: Deviation Distance (Section 3.3)

This is the core method. For each layer l:

1. For each error sample j, compute the average Euclidean distance to its k'=5 nearest neighbors in the correct set. Call this D^l_j (the error's deviation).
2. For each correct sample i, compute the average Euclidean distance to its k'=5 nearest neighbors *within* the correct set (excluding itself). Call this d^l_i (the correct set's internal spread).
3. Average D^l_j across all error samples to get D^l_error. Average d^l_i across all correct samples to get D^l_correct.
4. Run a Welch t-test comparing the two distributions.

If D^l_error > D^l_correct with a large t-statistic, error representations are farther from correct ones than correct ones are from each other.

**Critical analysis — the central weakness:**

This is NOT manifold analysis. This is a simple statistical comparison of kNN distances in ambient space. The paper's title says "manifold framework" and the entire narrative is built around "reasoning manifolds," but the actual computation is: "are error points farther from correct points than correct points are from each other?"

This would be true for ANY binary classification problem where the two classes occupy different regions of space. It's true for cats vs. dogs in image space. It's true for spam vs. non-spam in email embedding space. There is nothing specific to "manifolds" or "reasoning" about this observation.

Let me be concrete about what would constitute real manifold analysis:
- Computing geodesic distances (distances along the manifold surface, not straight-line Euclidean)
- Estimating local tangent spaces and curvature
- Checking that the manifold is connected, smooth, and has consistent topology
- Using manifold-aware distance metrics (like diffusion distances)

The paper uses none of these. Calling kNN Euclidean distance a "manifold framework" is misleading at best.

**Alternative explanation for the core finding:**

Error representations are farther from correct ones. But consider: the model is producing *different outputs* for error vs. correct samples. Different outputs require different hidden states (that's how autoregressive generation works — the hidden state determines the next token). So of course the representations are different. The question is: does this tell us anything beyond "different outputs have different internal states"?

The Spearman correlation (ρ = 0.598) between accuracy and relative deviation is moderate. The paper interprets this as "harder tasks → more deviation." But an equally valid interpretation is "harder tasks → more diverse error types → more scattered error representations." This doesn't require manifold theory.

### Step 4: Divergence Point Localization (Section 3.4)

For each error sample j, they track D^l_j across layers and find the first layer where:

D^l_j > μ^l_correct + α · σ^l_correct

where μ and σ are the mean and standard deviation of correct-sample internal distances at that layer, and α=2 is a threshold factor. The earliest layer satisfying this is the "divergence point."

**Critical analysis:**

This is the most interesting part of the paper. The idea of tracking when error representations first become distinguishable from correct ones, layer by layer, is useful. But several issues:

1. **The threshold α=2 is arbitrary.** The sensitivity analysis (Appendix F.3.2) shows that changing α from 1.0 to 2.0 dramatically changes the divergence point distribution. For MathVista, the peak shifts from early layers (α=1.0) to late layers (α=2.0). This means the "finding" that divergence happens at specific layers is largely determined by the threshold choice, not by the data.

2. **No baseline correction for the layer-by-layer testing.** They're doing multiple comparisons (one per layer) without correcting for multiple testing. With 32-40 layers, a false positive at any layer is quite likely, especially with α=2 (which corresponds to ~95th percentile, meaning ~5% false positive rate per layer).

3. **The divergence point depends on sample size.** With more data, you'd detect smaller deviations earlier. With less data, you'd miss real deviations. The paper doesn't account for this — divergence point distributions would shift if you used different dataset sizes.

4. **No causal validation.** Finding that error representations diverge at layer 15 doesn't mean layer 15 *caused* the error. It means that by layer 15, the error is detectable. The actual cause could be at layer 5, with the signal amplifying through subsequent layers until it crosses the threshold at layer 15. The paper conflates "detectable" with "originating."

### Step 5: Separability Test (Section 3.3)

They train SVM classifiers at each layer to distinguish correct from error representations. The accuracy increases with layer depth, reaching 70-80% in late layers.

**Critical analysis:**

This is the least problematic part of the methodology, but also the least novel. Probing classifiers to distinguish correct/incorrect outputs is a well-known technique. The finding that separability increases with depth is expected (later layers are more committed to the specific output).

The more interesting question: if a linear probe (no RBF kernel) achieved similar accuracy, that would suggest the separation is a simple linear phenomenon. The use of RBF kernel SVMs hides whether the separation is linear or requires nonlinear boundaries. This matters for the "manifold" story — if a linear boundary separates the groups, there's no need to invoke manifold geometry.

---

## What Was the Main Technical Hurdle Before This Paper?

The paper claims the hurdle was: existing failure analysis methods are task-specific (e.g., hallucination detectors) or require controlled input contrasts. A task-agnostic, purely geometric approach was missing.

**Is this framing accurate?** Partially. There's a real gap in having unified frameworks for failure analysis. But the paper oversells its novelty — kNN distance comparisons are not new. Probing classifiers for correct/incorrect distinctions are not new. The specific combination (kNN distance + layer-by-layer thresholding + intrinsic dimension) is somewhat novel, but none of the individual tools are.

## What is the KEY INSIGHT?

The key insight is supposed to be: "All reasoning failures, regardless of type, manifest as geometric deviation from the correct reasoning manifold."

**Is this actually insightful?** This is essentially saying: "wrong answers produce different internal states than right answers." Framing this as "geometric deviation from a manifold" makes it sound more profound than it is. The "manifold" is just the point cloud of correct representations — there's no learned or fitted manifold structure.

The actually useful contribution is the *layer-by-layer tracking* of deviation. The idea that you can watch an error "develop" across layers, rather than just detecting it at the end, has real diagnostic value. But the execution (fixed threshold, no multiple testing correction, no causal validation) undermines it.

## What's Still Open?

1. **Causation vs. correlation.** Does divergence at layer X mean something went wrong at layer X? Or did something go wrong earlier and the signal just became detectable at X?
2. **Actionability.** Even if you know errors diverge at layer 15, what do you do about it? The paper mentions "pulling back" deviating representations as future work, but this is speculative.
3. **Generalization to base models.** All experiments use instruction-tuned models. Base models (like our target, Llama 3.1 8B base) might behave completely differently.
4. **Fine-grained error types.** The paper lumps all errors together. But a "didn't understand the question" error and a "understood but computed wrong" error are fundamentally different. A truly useful framework would distinguish these.
5. **The manifold itself is never characterized.** Beyond ID estimation, the paper tells us nothing about the shape, topology, or geometry of the claimed manifold. Is it convex? Connected? What's its curvature? None of these questions are addressed.

---

## Devil's Advocate Protocol — Three Weakest Points

### Weakness 1: The "Manifold" Is Not a Manifold

This is the paper's biggest intellectual dishonesty. The word "manifold" appears 54 times in the main text, but no manifold is ever learned, fitted, or characterized. The "reasoning manifold" is simply the point cloud of correct representations. Computing kNN distances to a point cloud is NOT manifold analysis — it's nearest-neighbor statistics.

Real manifold analysis would involve:
- Estimating the tangent space at each point (local dimensionality and orientation)
- Computing geodesic distances (paths along the manifold rather than straight lines)
- Estimating curvature (how the manifold bends)
- Checking topological properties (connected components, holes)

Compare this with the Gurnee et al. "When Models Manipulate Manifolds" paper from Anthropic (transformer-circuits.pub, October 2025). That paper finds genuine manifold structure: a helical curve in ~6D subspace representing character counts, with quantified curvature, discovered through feature analysis, and validated through causal interventions. REMA does none of this.

The paper is essentially doing outlier detection (are error representations outliers relative to correct ones?) and calling it "manifold analysis." This is a packaging problem, and likely contributed to the paper's rejection.

### Weakness 2: The Core Finding May Be Trivially True

Error representations should be farther from correct representations than correct representations are from each other, for a simple reason: the model is producing *different outputs*. During autoregressive generation, the hidden state at each layer determines the next token. If the model generates "42" (correct) vs. "57" (incorrect), the hidden states must be different to produce different tokens. Mean-pooling over the generation steps averages this difference, but doesn't eliminate it.

So the finding "D_error > D_correct" may just be saying "different outputs have different internal states," which is tautological for autoregressive models.

To test whether the finding is non-trivial, the paper would need a control: compare two groups of *correct* samples that produce *different* answers (e.g., correct answers to different problems). If these groups also show D_group1→group2 > D_group1→group1, then the REMA finding is just a consequence of output diversity, not reasoning quality.

The paper never performs this control. This is a fatal flaw in the experimental design.

### Weakness 3: No Causal Validation Whatsoever

The paper's claims are purely observational. It observes that error representations are farther from correct ones. But it never tests whether this deviation is *causally related* to the error.

Causal tests would include:
- **Activation patching:** Replace an error sample's representation at the divergence layer with the nearest correct representation. Does the model then produce the correct answer?
- **Steering:** Push a correct sample's representation away from the manifold at a specific layer. Does it start producing errors?
- **Ablation:** Ablate the components responsible for the divergence. What happens?

Without any causal evidence, the "deviation causes the error" narrative is speculation. The deviation might be an *effect* of the error (the model has committed to a wrong path and this shows up in the geometry), not a *cause*.

Compare again with Gurnee et al., who validate their geometric findings with targeted interventions, ablations, and "visual illusions" that hijack specific mechanisms. The methodological gap is enormous.

---

## Connection to Hauberg's "Only Bayes Should Learn a Manifold"

Hauberg's paper (in our project context) argues that deterministic methods for estimating manifold geometry are biased — they systematically underestimate uncertainty and can produce wrong geometric conclusions, especially in regions with sparse data.

REMA's approach is entirely deterministic: Euclidean kNN distances with fixed k=5, no uncertainty quantification. Hauberg would raise several objections:

1. **The kNN distance metric ignores curvature.** If the correct representations live on a curved manifold, Euclidean distance overestimates the "on-manifold" distance between nearby points and underestimates it for far-apart points. This means the deviation metric D^l_j could systematically mischaracterize how far errors are from the manifold.

2. **No uncertainty on the deviation estimate.** The paper reports mean distances and t-statistics, but never asks: "How confident are we in the manifold approximation itself?" With only ~100-500 correct samples in a 2048-4096 dimensional space, the point cloud is *extremely* sparse relative to the ambient dimension. The "manifold" approximation (the point cloud) is noisy, and the distances to it are therefore uncertain.

3. **The O(1/D) variance result doesn't apply here.** Hauberg shows that in high dimensions, the expected Riemannian metric tensor has low variance. But this applies to probabilistic generative models, not to kNN distance computation. The paper can't borrow Hauberg's theoretical grounding because it uses fundamentally different methods.

**Bottom line:** REMA's methodology is exactly the kind of naive deterministic approach that Hauberg's paper warns against. Our GP-based probabilistic framework would provide more reliable geometric characterization than anything in REMA.

---

## Connection to Gurnee et al. "When Models Manipulate Manifolds" (transformer-circuits.pub)

The Gurnee et al. paper from Anthropic is the gold standard for geometric analysis of transformer representations, and the contrast with REMA is instructive.

**What Gurnee et al. do right that REMA doesn't:**

1. **They discover genuine manifold structure.** Character count is represented on a 1D helical curve in ~6D subspace. This is a real manifold with quantifiable curvature, not just a point cloud labeled "manifold."

2. **They use the discrete-continuous duality.** SAE features tile the manifold, providing local coordinates. This connects the geometric view with the feature-based view. REMA never engages with SAEs or feature-based analysis.

3. **They validate causally.** Ablations, patching, and "visual illusions" confirm that the geometric structure is functionally relevant. REMA is purely observational.

4. **They explain the geometry mechanistically.** The helical shape is optimal (proven via an optimization argument). The boundary detection uses QK-rotation of one manifold relative to another. Every geometric observation has a mechanistic explanation. REMA never explains *why* representations deviate — only that they do.

5. **They study a specific computation.** Gurnee et al. pick one task (linebreaking) and fully decompose it. REMA spreads across 7 datasets and 8 models but never deeply understands any of them.

**The lesson for our pipeline:** Depth beats breadth. Our approach of focusing on Llama 3.1 8B base on multiplication, discovering specific subspaces (carry digits, Fourier bases), and validating with interventions is much more in the spirit of Gurnee et al. than REMA. REMA's shotgun approach (many models, many tasks, shallow analysis) produces statistical observations but no mechanistic insight.

---

## Connection to the "Geometry of Reasoning" Paper (Zhou et al., Duke)

Our previous analysis of the Zhou et al. paper raised similar concerns: geometric observations without mechanistic validation, deterministic methods without uncertainty quantification, and framework-heavy papers with limited insight-per-page ratios.

REMA shares several of Zhou et al.'s weaknesses:
- Claims about "reasoning geometry" based on simple statistics
- No engagement with the feature-manifold duality
- No causal interventions
- Overclaiming novelty of relatively straightforward observations

REMA is somewhat better than Zhou et al. in that its core observation (error representations deviate more) is at least cleanly stated and empirically validated, whereas Zhou et al.'s claims about trajectory smoothness were potentially artifactual. But both papers share the same fundamental problem: describing patterns without explaining mechanisms.

---

## Implications for Our Pipeline (Llama 3.1 8B Base, Multiplication)

### What REMA Gets Right That We Should Consider

1. **Layer-by-layer tracking is valuable.** The idea of tracking how representations change across layers, and identifying where they diverge, is a good one. For our multiplication pipeline, we should track how carry-digit representations develop across Llama 3.1 8B's 32 layers and identify where wrong-carry representations first diverge from right-carry ones.

2. **The kNN-distance baseline is simple and useful.** Even though it's not true manifold analysis, the "error deviation vs. correct internal distance" comparison is a clean way to quantify representation drift. We could use this as a sanity check alongside our more sophisticated GP-based methods.

3. **ID estimation is worth doing.** The TwoNN method is fast and gives useful information about effective dimensionality at each layer. For our pipeline, we should compute ID of the multiplication representations at each layer as part of Step 1 (visual scouting).

### What REMA Gets Wrong That We Should Avoid

1. **Don't confuse point-cloud statistics with manifold geometry.** kNN distances in ambient space tell you about clustering, not about manifold structure. Our pipeline should use actual geometric tools: conditional covariance for subspace identification, GP-based Riemannian metrics for curvature, principal angles for superposition detection.

2. **Don't skip causal validation.** If we find that carry-digit error representations diverge at layer 20, we must test this with activation patching: replace the layer-20 representation with the nearest "correct carry" representation and see if the output fixes itself. Observation without intervention is incomplete.

3. **Don't use instruction-tuned models if the goal is mechanistic understanding.** REMA uses only instruction-tuned models (Qwen3 Instruct, Llama 3.2 Vision Instruct, etc.). Instruction tuning adds a thick layer of behavioral alignment on top of the base model's computations, potentially confounding the geometry. Our choice of Llama 3.1 8B *base* is correct — it avoids this confound.

4. **Don't use mean pooling for sequence tasks.** For multiplication, where the carries and digit computations happen at specific positions in the generated sequence, mean pooling would smear the signal. We should analyze specific token positions (e.g., the position where each digit of the product is predicted).

5. **Don't use UMAP as evidence.** UMAP is good for visualization and visual scouting (Step 1 of our pipeline), but should never be presented as evidence for geometric claims. It's a dimensionality reduction tool with known failure modes (creating artificial clusters, destroying real structure). The paper's reliance on UMAP figures is a weak point.

### The Deeper Issue: Manifold ≠ "Point Cloud of Correct Samples"

The central conceptual problem with REMA is that it defines the "reasoning manifold" as the set of correct representations and then asks whether errors are far from this set. This is outlier detection, not manifold analysis.

For our pipeline, the manifold we're looking for is more specific and more rigorous: it's the subspace (or curved surface) in which a specific concept (e.g., "carry digit value") is represented. We find this using conditional covariance analysis (removing the variance explained by other known concepts), not by collecting "correct" samples and calling them a manifold.

The difference is fundamental: REMA's "manifold" is defined by behavioral outcome (correct/incorrect), while our manifolds are defined by semantic content (what concept is being represented). REMA can't distinguish between a model that gets the right answer for the wrong reason (correct by luck) and one that reasons properly, because it only looks at the final answer to partition samples.

---

## The ICLR 2026 Rejection

The paper was submitted to ICLR 2026 and withdrawn (January 26, 2026). The OpenReview page lists the submission but the reviews are not publicly accessible for withdrawn submissions.

Based on the paper's weaknesses, likely reviewer concerns included:

1. **Novelty:** kNN-based outlier detection is not a new framework. The "manifold" framing adds terminology but not substance.
2. **Lack of causal analysis:** ICLR reviewers (particularly in the interpretability track) increasingly expect causal interventions, not just observational statistics.
3. **The core finding may be trivially true:** "Wrong answers have different internal states" needs a non-trivial control experiment that the paper lacks.
4. **Overclaiming:** Calling this a "unified framework" when it's kNN distance + threshold comparison is a stretch.
5. **No code release:** For an empirical paper, code availability is increasingly expected.
6. **Instruction-tuned models only:** Limits the mechanistic interpretability angle.
7. **No comparison with existing probing methods:** Linear probes, CCS, contrast-consistent search, and other methods can also distinguish correct from incorrect internal states. How does REMA compare?

---

---

# PASS 3 — Empirical Analysis (Stress-Testing the Claims)

## Claim 1: "Internal Reasoning States Are Low-Dimensional"

**What the paper shows:** ID estimates of 5-33 at various layers, vs. ambient dimensions of 2048-4096.

**Is this surprising?** No. Nearly every neural network study since 2017 has found low intrinsic dimensionality in hidden representations. This is how neural networks work — they project data into task-relevant subspaces. Pope et al. (2021) showed this for image models. Aghajanyan et al. (2020) showed this for language models. The observation is valid but completely expected and provides no evidence specific to "reasoning manifolds."

**What it would take to be surprising:** If correct and error representations had systematically different IDs (e.g., correct ID << error ID, suggesting correct reasoning is "more constrained"). The paper actually shows the opposite — IDs are similar, which undermines the manifold narrative.

**For Llama 3.1 8B multiplication:** We should expect ID to be low (probably 5-20) for multiplication representations, regardless of correctness. The more interesting question is whether the ID *decreases* at specific layers where the model "figures out" the carry structure.

## Claim 2: "Error Representations Deviate from the Correct Reasoning Manifold"

**What the paper shows:** D_error > D_correct with large t-statistics across all model-task pairs.

**The missing control:** As argued above, this could be a trivial consequence of output diversity. Consider: take all correct samples and split them into two random halves (A and B). Compute D_{A→B} (average kNN distance from group A to group B) and D_{A→A} (internal distance within group A). If D_{A→B} > D_{A→A}, then the REMA finding is just "different subgroups have larger cross-distances than within-distances," which is trivially true for any non-degenerate partition.

Even more telling: the paper's own Appendix F.3.3 reports that for Qwen3 on GPQA, the correct samples have *larger* internal distances than the error samples (D_correct > D_error at every layer). The authors interpret this as "correct reasoning explores more diverse representational space." But it directly contradicts the narrative that errors "deviate from the manifold" — if correct samples are more spread out, then the "manifold" has larger internal distances, and errors being closer together means they're actually in a *more concentrated* region.

This is a self-contradicting finding that the paper tries to spin as interesting. In reality, it shows that the "deviation" story is more nuanced than D_error > D_correct, and the simple kNN framework can't capture this nuance.

## Claim 3: "Divergence Points Can Be Localized"

**What the paper shows:** Histograms of first-layer-of-divergence across error samples.

**The sensitivity problem:** Table 6 in Appendix F.3.2 shows that for MathVista on Qwen2.5-VL (3B), changing α from 1.0 to 2.0 shifts the peak from layers 0-7 (count: 100) to layers 24-31 (count: 60). This means the "divergence point" is not a robust property of the data — it's largely an artifact of the threshold choice.

**The accumulation problem:** The threshold condition D^l_j > μ^l_correct + 2σ^l_correct is checked independently at each layer. But representations at adjacent layers are highly correlated (each layer applies a relatively small update to the residual stream). So if an error representation barely misses the threshold at layer 14, it will likely cross at layer 15. The "divergence point" is then determined by exactly where a gradually accumulating deviation crosses an arbitrary threshold — not by a discrete "something went wrong here" event.

A more meaningful analysis would look at the *rate of change* of deviation across layers (i.e., d(D^l_j)/dl). A sudden increase in deviation rate would genuinely signal that something went wrong at that layer, regardless of where it crosses any fixed threshold.

## Claim 4: "REMA Provides a Unified Framework for Failure Analysis"

**What "unified" means here:** The same kNN metric works across models and tasks. But this "unification" comes at the cost of specificity — the framework tells you that something went wrong at approximately layer X, but nothing about what went wrong, why, or how to fix it.

Compare with existing methods:
- **Probing classifiers** can tell you what specific information is present/absent at each layer.
- **Causal tracing** can identify which components are responsible for a specific output.
- **SAE feature analysis** can identify which human-interpretable features are active.
- **Circuit analysis** can decompose the full computation into understandable steps.

REMA provides a coarser signal than any of these. The claim of "unified framework" means "lowest common denominator" — it works everywhere because it measures very little.

---

## Comparison with the Ablation Study (Table 2)

The ablation on pooling strategies is actually the most methodologically sound part of the paper. The finding that mean-pooling and attention-weighted pooling outperform last-token and max-pooling, with the simpler method (mean-pooling) matching the complex one, is clean and useful.

The fact that all pooling strategies show D_error > D_correct confirms that the core finding is robust to methodological choices. But again, this robustness might indicate triviality rather than importance — if even a bad pooling strategy shows the same result, the signal might be trivially easy to detect.

---

## What the Paper Would Need to Be Convincing

1. **The random-partition control.** Split correct samples randomly and show that the cross-distance is NOT systematically larger than within-distance. This would rule out the "trivially true" explanation.

2. **Causal validation.** Activation patching at the divergence layer: replace error representations with nearest-correct representations and check if outputs improve.

3. **Comparison with probing baselines.** How does the kNN deviation metric compare with a simple linear probe for detecting errors? If a linear probe works just as well, the geometric framework adds no value.

4. **Fine-grained error analysis.** Distinguish different error types (computation errors, comprehension errors, format errors) and show they have different geometric signatures. This would demonstrate that REMA provides information beyond "right vs. wrong."

5. **Base model experiments.** Use at least one base model (like Llama 3.1 8B base) to remove the confound of instruction tuning.

6. **Actual manifold methods.** Use diffusion maps, Isomap, or local PCA to estimate manifold structure, rather than claiming "manifold" while computing only kNN distances.

---

## Key Takeaways

1. **The core observation is real but potentially trivial.** Error representations are indeed farther from correct representations than correct ones are from each other. But this might just be saying "different outputs have different internal states."

2. **The "manifold" branding is misleading.** No manifold is learned, fitted, or characterized. The paper does point-cloud statistics and calls it manifold analysis. This is the kind of packaging that leads to rejection at top venues.

3. **Layer-by-layer tracking is the best idea in the paper.** Watching deviation develop across layers has genuine diagnostic value, even if the current execution (fixed threshold, no multiple testing correction) is rough.

4. **No causal validation means no mechanistic insight.** The paper is purely observational. For interpretability research in 2025-2026, this is increasingly insufficient.

5. **Mean pooling is the wrong aggregation for process-level analysis.** The paper claims to study "reasoning processes" but then averages away the temporal structure. For our pipeline, we should avoid this.

6. **The paper is a negative example for our research.** It shows what happens when you use geometric vocabulary without geometric methods, claim manifold analysis without doing manifold analysis, and do observational studies without causal validation. Our pipeline, with its GP-based probabilistic geometry, concept-specific subspace analysis, and planned validation through interventions, avoids all of REMA's weaknesses.

---

## Research Ideas Generated

1. **The "correct partition" control experiment.** For our multiplication pipeline: split correct-answer problems into two groups (e.g., by answer magnitude) and check if the cross-group kNN distance is also larger than within-group distance. If it is, then deviation-from-correct is a trivial property of output diversity. If it isn't, then there's something genuinely special about the correct-reasoning region.

2. **Rate-of-divergence instead of threshold-crossing.** Instead of finding the first layer where deviation exceeds a threshold, compute d(D^l)/dl — the rate at which deviation grows. Sudden jumps in this rate would indicate genuine divergence events, independent of arbitrary thresholds.

3. **Concept-specific divergence tracking.** For multiplication in Llama 3.1 8B, track divergence separately for different concepts: carry-digit representation accuracy, partial product representation, digit-level embeddings. This would tell us *what specifically* goes wrong at each layer, not just that "something" diverges.

4. **Connecting REMA-style deviation to SAE features.** At the layer where kNN deviation spikes, which SAE features differ most between correct and error samples? This would bridge the statistical observation to mechanistic understanding.

---

## Summary Table

| Aspect | REMA | What We Need for Our Pipeline |
|--------|------|------------------------------|
| Manifold characterization | Point cloud of correct samples | GP-based probabilistic Riemannian geometry |
| Distance metric | kNN Euclidean in ambient space | Geodesic or subspace-projected distances |
| Subspace identification | None | Conditional covariance + rSVD + LDA |
| Error analysis | Binary (correct/error) | Concept-specific (carry value, digit accuracy) |
| Causal validation | None | Activation patching, ablation |
| Uncertainty quantification | None (purely deterministic) | GP posteriors on metric tensors |
| Model type | Instruction-tuned models | Base model (Llama 3.1 8B) |
| Representation extraction | Mean-pooled over generation steps | Position-specific extraction |
| Feature connection | None | SAE features ↔ manifold duality |

**Final verdict:** REMA addresses a real and interesting question (where do reasoning failures originate in the model's layers?) but answers it with methods too crude to provide mechanistic insight. The "manifold" framing is misleading. The paper is a useful negative example: it shows the gap between geometric vocabulary and geometric analysis, and clarifies what we need to do better in our own pipeline.