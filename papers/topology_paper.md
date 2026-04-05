# Analysis: "Learning Geometry and Topology via Multi-Chart Flows"

**Authors:** Hanlin Yu, Søren Hauberg, Marcelo Hartmann, Arto Klami, Georgios Arvanitidis
**Venue:** arXiv preprint (2505.24665), submitted May 30, 2025. Under review. NOT peer-reviewed.
**Type:** Empirical/Methods paper with some theory (one theorem + proof, algorithms, experiments)
**Purpose:** Deep understanding — this paper is from Hauberg's group (co-author on our foundational reference) and directly extends the geometric manifold learning framework we're building on. The multi-chart idea addresses a fundamental limitation we'll face when our digit manifolds have non-trivial topology (circles, tori).

---

# PRE-READING: Who Are These People?

## Author Track Record

**Søren Hauberg** (Technical University of Denmark): This is THE Hauberg from our "Only Bayes Should Learn a Manifold" foundational reference. Senior researcher, ERC grant holder, long publication record in Riemannian geometry for machine learning. His 2018/2019 paper is one of our project's theoretical pillars. Having him as co-author gives this paper significant credibility on the geometric methodology front.

**Georgios Arvanitidis** (Technical University of Denmark): Hauberg's long-time collaborator. His PhD thesis was literally titled "Geometrical Aspects of Manifold Learning." Published at ICLR 2018 ("Latent Space Oddity: on the Curvature of Deep Generative Models"), AISTATS 2019, 2021, 2022, NeurIPS 2025. This is someone who has been thinking about pullback metrics, geodesics in latent spaces, and Riemannian structure of generative models for nearly a decade. The 2018 ICLR paper is actually cited in Hauberg's "Only Bayes" paper and shares the core insight about using Jacobian-based pullback metrics.

**Hanlin Yu** (University of Helsinki): First author. Appears to be a PhD student. Co-authored with Arvanitidis and Hauberg on a NeurIPS 2025 paper about connecting neural models' latent geometries. This suggests he's working at the intersection of Helsinki and DTU groups.

**Marcelo Hartmann and Arto Klami** (University of Helsinki): The Helsinki side of the collaboration. Klami is a known researcher in probabilistic machine learning.

**Bottom line:** This is a strong author team with deep expertise in exactly the right area. The Hauberg + Arvanitidis combination is the gold standard for Riemannian geometry in generative models.

## Credibility Assessment (Non-Peer-Reviewed)

- **Author track record:** Excellent. Multiple ICLR, NeurIPS, AISTATS, ICML publications on this exact topic.
- **Reproducibility signals:** No code link mentioned in the paper. They describe using PyTorch, normflows, Stochman (Hauberg's own library), and open-source datasets. Hyperparameters are specified. But no GitHub repo = a soft red flag for reproducibility.
- **What's NOT disclosed:** No wall-clock training times per experiment (just vague "within several hours"). No analysis of failure modes during training. No sensitivity analysis on number of charts C.
- **Related prior work by same authors:** Kalatzis et al. (2022) — "Density estimation on smooth manifolds with normalizing flows" and the earlier "Multi-chart flows" paper (also Kalatzis, Ye, Pouplin, Wohlert, Hauberg) already proposed the multi-chart idea. This paper appears to be an evolution/extension of that earlier work. Critical question: what's actually new here versus the 2022 paper?

---

# PASS 1 — Jigsaw Puzzle (What does this paper do?)

## Q1: What is the problem being solved?

**The problem in one sentence:** When you try to learn a low-dimensional curved surface (manifold) from high-dimensional data using a normalizing flow, a single flow can only learn surfaces that look like flat Euclidean space everywhere — it cannot correctly represent surfaces with "holes" or "wrapping" (non-trivial topology), like a sphere or a torus.

### Let me explain this from the ground up.

**What is a normalizing flow?** Think of it as a learnable, reversible function. You start with a simple distribution (like a Gaussian blob in some space) and you push it through a series of smooth, invertible transformations until it matches the shape of your data. Because each transformation is invertible, you can compute exact probabilities — you always know how likely any given data point is.

The key property: the function from input to output is a *bijection* (one-to-one and onto). Every point in the simple space maps to exactly one point in the complex space, and vice versa.

**What is the manifold hypothesis?** Real-world high-dimensional data (images, text embeddings, neural network activations) don't actually fill up the full high-dimensional space. Instead, they tend to lie on or near a much lower-dimensional curved surface. A 1000-dimensional vector of pixel values for face images might actually only vary along maybe 50 meaningful dimensions (pose, lighting, expression, etc.). That 50-dimensional surface embedded in 1000-dimensional space is the "data manifold."

In our specific project context: Llama 3.1 8B has 4096-dimensional activations. Our Phase C found that digit representations live in 2-9 dimensional subspaces within that 4096D space. The manifolds we care about (circles for modular structure, possibly tori for multi-digit interactions) are low-dimensional surfaces inside those subspaces.

**What goes wrong with a single flow?** A normalizing flow is a bijection — a smooth, invertible function from a d-dimensional flat space to a d-dimensional surface in the ambient space. By definition, this means the learned surface must be "topologically equivalent" (homeomorphic) to flat Euclidean space. In plain language: you can stretch and bend it, but you can't poke holes in it or wrap it around to meet itself.

A circle is NOT topologically equivalent to a line. A circle wraps around. If you try to parameterize a circle with a single flow mapping from the real line, you inevitably create a "gap" — one point on the circle that doesn't get covered, or two points that get mapped to the same place. Similarly, a sphere is not topologically equivalent to a plane. A torus (donut shape) isn't either.

This is not a practical limitation that better architectures can overcome. It's a *mathematical impossibility*. A single bijection from flat space CANNOT cover a sphere or torus or circle. Period. This is one of the first things you learn in a topology course.

**Why does topology matter for geometry?** If the learned manifold has the wrong topology (e.g., a line instead of a circle), then the geodesics (shortest paths on the surface) will be wrong. Two points that are close on a circle (separated by a short arc) might appear very far apart if the flow has a gap between them — the "geodesic" would have to go all the way around the circle. This means distances are wrong, and any downstream computation that uses geometry (clustering, interpolation, etc.) will be unreliable.

## Q2: Why is this interesting and nontrivial?

**This is nontrivial because:** The single-chart topology limitation is fundamental and cannot be fixed by making the single flow more expressive. You need multiple flows that each cover part of the manifold and overlap in "border regions" — exactly like an atlas of maps covers the Earth (no single flat map can represent the whole sphere without distortion). But "gluing" multiple flows together is hard for three reasons:

1. **Training problem:** How do you train multiple flows jointly so that each one learns a different region of the manifold? You need to assign data points to flows (or share them), optimize all flows together, and ensure they cooperate rather than compete.

2. **Overlap problem:** In classical differential geometry, charts (local coordinate patches) are required to overlap perfectly in border regions. But neural network flows won't extrapolate perfectly — they'll disagree in regions where their training data is sparse. The paper needs to handle this imperfect overlap.

3. **Geodesic computation problem:** To compute shortest paths on the manifold, you need to be able to follow a curve that transitions smoothly between different charts. With imperfect overlaps, this is numerically challenging.

**Previous work couldn't handle this because:** Earlier papers (Brehmer & Cranmer 2020, Caterini et al. 2021) showed how to train single degenerate flows on manifolds. Kalatzis et al. (2022) — from the same research group — proposed multi-chart flows but didn't develop the geometric tools (geodesics, exponential/logarithmic maps) needed to actually use the learned geometry. Other approaches (Schonsheck et al. 2020, Sidheekh et al. 2022) used multiple autoencoders or flows but "lacked a principled way of choosing the correct encoder" and "did not study the geometric and topological implications."

**Devil's advocate question:** Is this problem really important for practical applications? The paper implicitly assumes we care about the *geometry and topology* of the learned manifold, not just density estimation. For pure density estimation, a single flow might be "good enough" even if the topology is wrong — it just fits a density over a slightly wrong surface. The paper needs to justify why correct topology matters. They do this somewhat through the persistence diagram analysis, but the practical use case remains abstract.

## Q3: What is the main claim?

**The main claim:** Using a mixture of normalizing flows (multi-chart flows), trained via an adapted EM algorithm with responsibility-weighted geometry, yields:
1. Better reconstruction of the data manifold (lower reconstruction error)
2. Better density estimation (lower Wasserstein distance of generated samples to test data)
3. Correct estimation of geodesic distances
4. Faithful recovery of the manifold's topology (verified via persistent homology)

The key technical contributions are:
- A training scheme (MLE or EM) for mixtures of degenerate normalizing flows
- Algorithms for computing exponential maps (Algorithm 1) and geodesics/logarithmic maps (Algorithm 2) on manifolds defined by multiple charts
- A probabilistic weighting scheme using EM responsibilities to handle non-overlapping charts

**The mathematical claim (Theorem 1):** The geodesic equation in the latent space induced by the pullback metric takes the specific form: z̈(t)^k = -g^{kl} Σ_{i,j,m} (∂²x_m / ∂z_i ∂z_j)(∂x_m / ∂z_l) ż^i ż^j. This is not a novel result in differential geometry — it's the standard geodesic equation for a pullback metric. The contribution is writing it in a form amenable to automatic differentiation.

---

## Pass 1 Verdict: PROCEED TO PASS 2

**Reasons:**
- Directly from Hauberg's group; extends the theoretical framework we're building on
- Addresses a specific limitation (topology) that will matter for our digit manifold analysis (Fourier structure implies circles/tori)
- The geodesic computation algorithms could be directly useful if we adopt flow-based manifold learning
- The responsibility-weighted geometry idea is relevant to our multi-population (correct vs. wrong) analysis

---

---

# PASS 2 — Scuba Dive (How does the paper work?)

## Prerequisite Concepts Explained for a Grade 12 Student

### What is a "chart" in differential geometry?

Think about a globe of the Earth. You can't flatten the entire globe into a single flat map without tearing or distorting it. (This is why Greenland looks enormous on most world maps — the Mercator projection distorts areas near the poles.)

The solution cartographers use: make multiple maps that each show a small region accurately. A map of North America, a map of Europe, a map of Asia, etc. Each individual map is called a **chart**. It's a flat representation of a small patch of the curved surface.

The critical part: neighboring maps must agree where they overlap. If the North America map and the Europe map both show Iceland, they'd better agree on where Iceland is. This "smooth agreement in overlapping regions" is what mathematicians call a **transition function**.

A collection of charts that covers the entire surface (with proper overlaps) is called an **atlas**. Just like an atlas of the Earth covers the whole globe even though each page only shows a small region.

**The key insight from topology:** Some surfaces NEED multiple charts. You cannot cover a sphere with a single chart (one flat map). You need at least two. A torus (donut) needs at least two as well. A circle needs at least two. The minimum number of charts tells you something deep about the shape's topology.

### What is a "normalizing flow" really doing?

Imagine you have a jar of red paint (a simple, known distribution — like a blob of Gaussian noise in 2D). You want to reshape this paint blob into a complicated pattern (like the distribution of stars in the night sky).

A normalizing flow is like a series of smooth, reversible stirring operations. Each "stir" is a simple transformation — maybe squeeze the paint in one direction, or shift it, or twist it. After many carefully chosen stirs, the red blob matches the star pattern.

The magic: because each stir is reversible, you can always go backwards. Given any point in the star pattern, you can "un-stir" it back to find where it came from in the original blob. This lets you compute exact probabilities.

**Degenerate flows:** What if your data lives on a 2D surface in 3D space (like paint spread on the surface of a ball)? Now the "simple blob" is in 2D, but the "star pattern" is in 3D. The flow maps from 2D to 3D — it's a "degenerate" flow because the output dimension is higher than the input. The paper decomposes this flow as f = h̃ ∘ g, where:
- g handles the 2D-to-2D part (reparameterizing the latent space)
- h̃ handles the 2D-to-3D part (embedding the surface into the ambient space)

### What is a "pullback metric"?

This is the same concept we discussed in our Hauberg analysis. The key equation in this paper is:

G_Z(z) = (∂x/∂z)^T (∂x/∂z) = J^T J

where J is the Jacobian matrix (the matrix of all partial derivatives of the mapping from latent space to ambient space).

In simple terms: the pullback metric tells you "how much does a tiny step in latent space stretch into a big step in the real world?" If the Jacobian stretches a lot in one direction, distances in that direction are large. The metric captures this stretching at every point.

**Why this matters for our project:** When we do GPLVM on Llama 3.1 8B activations, we'll learn a mapping from a low-dimensional latent space into the 4096D activation space. The pullback metric of that mapping will tell us the true Riemannian geometry of the digit representations. This paper shows how to compute geodesics using that pullback metric when the manifold needs multiple charts.

### What is "persistent homology"?

This is a tool from algebraic topology that lets you detect the "shape" of data. The core idea:

1. Start with your data points.
2. Draw a ball of radius r around each point.
3. As r grows from 0 to infinity, the balls overlap and form connected regions.
4. Track when "holes" appear and disappear.

A "hole" that appears at a small radius and persists for a long time as the radius grows is a genuine topological feature. A "hole" that appears briefly and disappears is noise.

**H₀** counts connected components (clusters). **H₁** counts 1-dimensional holes (loops — like the hole in a ring). **H₂** counts 2-dimensional holes (cavities — like the inside of a sphere).

A sphere should have H₀ = 1 (one connected piece), H₁ = 0 (no loops), H₂ = 1 (one cavity). A torus should have H₀ = 1, H₁ = 2 (two independent loops), H₂ = 1.

The persistence diagram plots birth vs. death of features. Points far from the diagonal are significant features; points near the diagonal are noise.

### What is "Expectation Maximization" (EM)?

EM is a classic algorithm for training mixture models. Imagine you have data that comes from a mix of two different sources (say, heights from a population of men and women mixed together). You want to figure out the parameters of each source.

EM alternates between two steps:
- **E-step (Expectation):** Given your current guess of the parameters, compute the "responsibility" of each source for each data point. "How likely is it that this 5'8" person came from the male distribution vs. the female distribution?"
- **M-step (Maximization):** Given the responsibilities, update the parameters of each source to best fit the data points it's responsible for.

You keep alternating until convergence. In this paper, each "source" is a separate normalizing flow (chart), and the responsibilities tell you which chart is "in charge of" which data point.

---

## Q1: What was the main technical hurdle? How does this paper overcome it?

### The Barrier

The fundamental barrier is topological: a single bijection from R^d to a d-dimensional manifold forces the manifold to be homeomorphic to R^d. This rules out spheres, tori, projective spaces, and most interesting manifolds. Previous work (Brehmer & Cranmer 2020, Caterini et al. 2021) essentially ignored this problem — they showed good results on manifolds that happen to be Euclidean (or nearly so), but their approach fundamentally breaks for non-trivial topology.

The secondary barrier is practical: even if you use multiple flows, you need to:
1. Train them to cover different regions without collapse (all flows learning the same region)
2. Handle the borders between charts where flows disagree
3. Compute geometric quantities (geodesics) that traverse multiple chart boundaries

### The Key Insight

**Insight 1 (Training):** Treat the collection of flows as a mixture model. Each data point has a "responsibility" under each chart, computed via Bayes' rule. Train via EM: E-step assigns responsibilities, M-step trains each flow on its responsible data. Add a reconstruction penalty (equation 9) to keep the flows faithful to the manifold geometry.

**Insight 2 (Geometry at boundaries):** Don't rely on a single chart's geometry at any point. Instead, use the responsibility-weighted average over all charts. The manifold point corresponding to an ambient point x' is defined as:

Σ_c r_c(x') h̃_c(h̃_c†(x'))

where r_c(x') is chart c's responsibility for point x', and h̃_c(h̃_c†(x')) is chart c's reconstruction of x'. This is a probabilistic "consensus" among charts.

**Insight 3 (Geodesics across charts):** For exponential maps, use an Euler-like integration where at each step, you compute the geodesic update from EACH responsible chart and take the weighted average (Algorithm 1). For logarithmic maps, parameterize the curve directly in ambient space (not in any chart's latent space), project it onto the manifold via the responsibility-weighted reconstruction, and optimize the energy (Algorithm 2).

### Critical Evaluation of These Insights

**On Insight 1:** The EM approach for mixture of flows is not new. The paper cites Izmailov et al. (2020) and Atanov et al. (2020) as prior work on mixture normalizing flows. The novelty is applying this to *degenerate* flows (where ambient dim > latent dim) combined with the reconstruction penalty. But the paper doesn't prove that EM will converge to the correct solution, or even to a solution with the right topology. The K-Means initialization is a heuristic with no guarantees.

**Devil's advocate on Insight 1:** What prevents all charts from collapsing to the same region? The paper adds a regularization term (equation 33) that penalizes uneven average responsibilities. But this is a soft constraint — it encourages uniform coverage, it doesn't guarantee it. And the threshold on responsibilities (only train on data points where responsibility > threshold) introduces a hard hyperparameter. How sensitive is the method to this threshold? The paper never says.

**On Insight 2:** The responsibility-weighted reconstruction is elegant but raises questions. If two charts disagree significantly about where a point lies on the manifold, averaging their reconstructions gives you a point that might NOT lie on either chart's manifold. The average of two points on a sphere is a point inside the sphere, not on it. The paper doesn't address this — it just assumes the averaging works because the flows are "sufficiently trained."

**Devil's advocate on Insight 2:** The paper defines the learned manifold as the set {Σ_c r_c(x') h̃_c(h̃_c†(x')), ∀x' ∈ R^D}. This is a peculiar definition. In classical differential geometry, a manifold is a topological space with charts and transition functions. Here, the "manifold" is defined as the image of a weighted-averaging operation. Does this actually define a smooth manifold? The paper doesn't prove this. It's entirely possible that the averaging creates cusps or self-intersections, especially in regions where multiple charts have comparable responsibilities.

**On Insight 3:** Algorithm 1 (exponential maps via weighted Euler) is simple but potentially inaccurate. Standard geodesic integration already accumulates errors; averaging over multiple charts introduces additional error. The paper shows in Table 2 that the "Ambient" algorithm (which takes the weighted average to infinity) is sometimes better and sometimes catastrophically worse (10^16 error on Torus-M). The "Euler" method is more stable but still has significant error on some datasets.

**Devil's advocate on Insight 3:** Algorithm 2 (geodesics via ambient curve optimization) parameterizes a curve in the ambient D-dimensional space and optimizes it. But geodesics live on the d-dimensional manifold, not in D-dimensional space. Projecting the ambient curve onto the manifold at each optimization step could introduce artifacts. And the paper doesn't discuss convergence of this optimization — how do you know you've found the geodesic and not just a local minimum of the energy?

## Q2: What is the simplest baseline? By what metric is the new method better?

### The Baselines

- **SINGLE-M:** A single normalizing flow trained sequentially — first optimize h for reconstruction, then optimize g for density estimation. This is the Brehmer & Cranmer (2020) approach.
- **SINGLE:** A single normalizing flow trained jointly on both reconstruction and density.
- **MULT:** The proposed multi-chart flow with EM training.

### The Metrics

1. **Reconstruction error (Recons):** How well can the model project a data point onto the learned manifold and back? Lower is better.
2. **Wasserstein distance (W):** How different are generated samples from real test data? Lower is better.
3. **Exponential map error (Exp maps):** How close is the computed exponential map to the ground truth? Lower is better.
4. **Distance error (Dists):** How close are the computed geodesic distances to the ground truth? Lower is better.

### What the Results Show

**Figure 2 (Sphere and Torus):** MULT wins on all metrics for all four datasets (Sphere-U, Sphere-M, Torus-U, Torus-M). The improvements are dramatic:
- Reconstruction: MULT achieves 10^-6 vs. SINGLE's 10^-4 to 10^-2 (100x to 10,000x better)
- Distances: MULT gets 3.77×10^-4 on Sphere-U vs. SINGLE's 5.22×10^-2 (138x better)
- Wasserstein: MULT is comparable or slightly better

**Table 1 (Mocap1 circular structure):** This is the most compelling result. Five points uniformly placed on the 1-dimensional circular manifold. SINGLE methods show wildly uneven segment lengths (one segment is 2-3x longer than others — the "gap" in the circle). MULT shows nearly uniform segment lengths (11.3-11.8 across all segments, variance 0.0). This directly demonstrates that MULT captures the circular topology while SINGLE methods don't.

**Figure 3 (Persistence diagrams):** For the sphere, MULT correctly identifies H₀ = 1, H₂ = 1 (one connected component, one 2D hole), matching the ground truth. SINGLE methods show spurious or missing features.

### Critical Assessment of the Comparison

**Is the comparison fair?** The paper states "we keep the number of parameters for each method identical or comparable." For sphere/torus: SINGLE uses 12 layers for g and 36 for h. MULT uses 4 charts × (3 layers for g + 9 layers for h). So MULT has 4 × (3+9) = 48 total layers vs. SINGLE's 12+36 = 48 total layers. This is a fair parameter comparison.

**But:** The EM training procedure, K-Means initialization, and warm-up pretraining give MULT extra structural advantages that aren't captured by just counting parameters. SINGLE has to learn a global mapping; MULT divides the problem into subproblems via K-Means and then refines. This is a meaningful advantage beyond just parameter count.

**Devil's advocate:** The paper tests on manifolds where the topology is KNOWN to be non-trivial (sphere, torus, circle). On these specific manifolds, a single chart provably can't work. The comparison is somewhat rigged in MULT's favor for these examples. What about manifolds that ARE diffeomorphic to Euclidean space (like a bowl shape)? Would MULT still help? The paper doesn't test this, which is a gap.

**Another concern:** The number of charts C is set manually (4 for sphere/torus, 2 for Mocap1, 5 for Mocap3). How do you choose C in practice when you don't know the topology? The paper cites Zhang et al. (2023) who can identify intrinsic dimensionality while training, but leaves this as future work. For practical applications, this is a critical missing piece.

## Q3: What's still open and why doesn't their insight apply there?

### Acknowledged Limitations

1. **High curvature:** "Multi-chart flows are more technically involved and can be difficult to train especially for manifolds with high curvature." This is honest. High curvature means charts need to be smaller, which means you need more charts, which makes training harder.

2. **Number of charts:** The paper assumes C is known. In practice, you don't know how many charts you need. Too few = topology still wrong. Too many = wasteful and hard to train.

3. **Dimensionality:** The paper assumes d (intrinsic dimensionality) is known. Real data rarely comes with this information.

### Unacknowledged Limitations (My Additions)

4. **Scalability:** All experiments use d=1, 2, or 3 dimensional manifolds in D=2, 3, or ~100 dimensional ambient spaces. The paper never tests on the scale relevant to modern ML (d=10-50, D=1000-4096). For our Llama 3.1 8B project, we'd need d=1-9 dimensional manifolds in D=4096. Can the Jacobian computation (O(d^3) per step) and geodesic integration scale to this?

5. **No uncertainty quantification:** This is ironic given that Hauberg (a co-author) is the person who argued most forcefully that "Only Bayes should learn a manifold." The entire premise of Hauberg's 2018/2019 paper is that deterministic manifold learning produces biased geometry, and you NEED probabilistic methods (GP posteriors over metric tensors) to get reliable geometry. Yet this paper uses entirely deterministic normalizing flows. The responsibility weighting provides a weak form of "uncertainty" (multiple estimates averaged), but it's not the principled Bayesian approach Hauberg himself advocates.

This is a genuine tension. Hauberg's earlier work says deterministic methods can't reliably learn geometry. This paper uses deterministic methods to learn geometry. The paper never acknowledges or resolves this tension.

6. **No causal validation:** The paper shows that MULT learns manifolds with better topology as measured by persistent homology. But there's no evidence that the learned geometry is *causally meaningful* — i.e., that it actually reflects the true data-generating process rather than an artifact of the model architecture and training procedure.

7. **Geodesic solver instability:** Table 2 shows that the "Ambient" solver produces 10^16 error on Torus-M (essentially NaN). The "Euler" solver produces 1.07 error on Torus-M (order of magnitude too large). Even the best solver has issues on the more complex datasets. For practical use, this instability is concerning.

8. **No comparison with GPLVM or VAE-based approaches:** The paper only compares different flavors of normalizing flows. But the manifold learning landscape includes VAEs (which Hauberg's own work uses), GPLVMs (which we're planning to use), and other approaches. How does multi-chart NF compare to a well-configured GPLVM on these tasks?

## Q4: Does their insight apply to OTHER unconsidered problems?

### Connection to Our Llama 3.1 8B Project

This is the critical question. Let me think through this carefully.

**Our situation:** Phase C found that digit representations in Llama 3.1 8B live in low-dimensional subspaces (2-9D) within the 4096D residual stream. Our next step (Fourier screening) will test whether these representations have circular structure. If digits 0-9 lie on a circle (as Bai et al. found in toy models, and as the Fourier representation hypothesis predicts), then the manifold has non-trivial topology.

**Could we use multi-chart flows?** In principle, yes. We could train a multi-chart normalizing flow to learn the mapping from a low-dimensional latent space to the activation subspace, using the activations as data points. The multi-chart approach would correctly handle the circular topology that a single chart would distort.

**But should we?** There are several reasons to be cautious:

1. **We're analyzing, not generating.** The paper's primary use case is generative modeling — learning a density on the manifold to sample from. We don't need to sample from the digit manifold. We need to CHARACTERIZE its geometry (curvature, topology, metric structure). Multi-chart flows are a more complex tool than we need if simpler methods (GPLVM, Fourier analysis directly on the subspace) can characterize the geometry.

2. **Hauberg's own argument against deterministic methods applies here.** If we use normalizing flows (deterministic) to learn the manifold, Hauberg's 2018/2019 paper says the geometry may be biased. If we use GPLVM (probabilistic), we get uncertainty quantification for free. The multi-chart flow paper doesn't resolve this tension.

3. **We know the topology a priori.** For digit representations, we expect circles (for single digits mod 10) or tori (for multi-digit interactions). We don't need to DISCOVER the topology from data — we have strong theoretical priors from Fourier analysis. Multi-chart flows would be overkill for our case.

4. **Sample sizes.** Our datasets have hundreds to thousands of activation vectors per concept per layer. The paper uses 12,000 training points for sphere/torus and 30,000-50,000 for Mocap. Our per-concept datasets are smaller, especially for L5 correct population (~600 samples if we scale up). Training multiple normalizing flows on this might overfit.

**Where multi-chart flows ARE relevant to us:**

1. **Validating topology:** If we claim digit representations lie on a circle, we could use the paper's persistent homology approach to verify. Train multi-chart and single-chart flows, compare persistence diagrams — if multi-chart shows H₁ = 1 and single-chart doesn't, that's evidence for circular topology. This is a nice validation tool, though not the primary analysis.

2. **Computing geodesics across charts:** If we eventually need geodesic distances on the digit manifold (e.g., for studying how the model interpolates between digits), the multi-chart geodesic algorithms (Algorithms 1-2) would be relevant.

3. **The responsibility-weighting idea for multi-population analysis:** The paper's approach of weighting geometric estimates by responsibilities is conceptually similar to what we might need for our correct/wrong population comparison. Instead of charts, we have populations. The idea of taking responsibility-weighted averages of geometric quantities could transfer.

### Connection to Gurnee et al. ("When Models Manipulate Manifolds")

The Gurnee et al. paper from Anthropic found that character count representations in Claude 3.5 Haiku form 1D manifolds (helical curves) in ~6D subspaces. These helices are curves that wind through space — they have the topology of a circle (or a line, depending on how the boundary is handled).

If you wanted to learn the geometry of these character count manifolds precisely, you'd face exactly the problem this paper addresses. A single normalizing flow from R^1 to R^6 would miss the circular topology. Multi-chart flows with C=2 charts could capture it.

But Gurnee et al. didn't need flows at all — they computed PCA, found the manifold visually, and studied it mechanistically. The multi-chart flow approach is a heavier hammer than what mechanistic interpretability research typically needs.

### Connection to MOLT (Sparse Mixtures of Linear Transforms)

The MOLT paper from Anthropic found that geometric computations (like "rotate a circle by +3") require one MOLT transform but hundreds of transcoder features. This validates the geometric view: the model does geometric operations on manifolds.

Multi-chart flows could be used to LEARN the manifold on which the MOLT transforms operate. The multi-chart structure would naturally capture the topology (circles, tori) that the model manipulates. But this is speculative — nobody has connected these two lines of work yet.

## Q5: What are the caveats and takeaways?

### The Three Weakest Points (Devil's Advocate)

**Weakness 1: No uncertainty quantification, contradicting Hauberg's own prior work.** The single biggest weakness. Hauberg co-authored "Only Bayes Should Learn a Manifold" which argues forcefully that deterministic methods produce biased manifold geometry. This paper uses deterministic methods. The responsibility weighting provides a weak form of ensemble uncertainty, but it's not the GP posteriors over metric tensors that Hauberg himself advocates. The paper never discusses this tension. This is a significant intellectual gap.

**Weakness 2: Experiments only on known, low-dimensional manifolds.** All test manifolds (circle, sphere, torus, triangular meshes) are 1-3 dimensional with known topology. The method is never tested on: (a) high-dimensional manifolds, (b) manifolds with unknown topology, (c) real-world data where ground truth geometry is unavailable. The triangular meshes are somewhat realistic, but the ground truth distances are approximated via graph shortest paths on refined meshes — this approximation is itself questionable.

**Weakness 3: Critical hyperparameters with no guidance.** The number of charts C, the responsibility threshold, the reconstruction weight λ, the choice between MLE and EM, the K-Means initialization, the pretraining duration — all of these are set manually with no principled guidance for new problems. For someone trying to use this method on a new dataset, the paper provides no actionable advice on how to set these hyperparameters. The paper says "we tune λ among [100, 1000, 10000]" and "we tune learning rates among [1e-4, 3e-4, 1e-3]" — that's a 3×3 grid search. Not very illuminating.

### Key Takeaways

1. **The topological limitation of single flows is real and provable.** This is the strongest part of the paper. If your manifold isn't diffeomorphic to Euclidean space, a single flow WILL fail to capture the topology. No amount of architecture improvement fixes this.

2. **Multi-chart flows with EM training can capture correct topology.** The experimental evidence for this is convincing, at least on the tested manifolds. The persistence diagrams are particularly compelling.

3. **Geodesic computation on multi-chart manifolds is hard.** The three solver algorithms (Hard_switch, Euler, Ambient) have different tradeoffs, and none is universally reliable. This is an open problem.

4. **The paper is an incremental advance over Kalatzis et al. (2022).** The earlier paper by the same group proposed multi-chart flows. This paper adds geodesic computation (new), more experiments (new), and the formal connection to classical differential geometry (new). But the core idea of using mixture of degenerate flows is not new.

---

---

# PASS 3 — The Swamp (Deep Technical Dive)

## Theorem 1: The Geodesic Equation

### The Claim

The geodesic equation in the latent space induced by the pullback metric G_Z(z) = J^T J (where J = ∂x/∂z is the Jacobian of the embedding) is:

z̈(t)^k = -g^{kl} Σ_{i,j,m} (∂²x_m / ∂z_i ∂z_j) (∂x_m / ∂z_l) ż^i ż^j

### Step-by-Step Proof Walkthrough (Simplified)

**Step 1: Start with the pullback metric.**

The metric is g_{ij} = Σ_m (∂x_m/∂z_i)(∂x_m/∂z_j). This is just the dot product of the i-th and j-th columns of the Jacobian. In matrix form, G = J^T J.

Think of it this way: each column of J tells you how the m-th ambient coordinate changes as you move in the i-th latent direction. The metric measures how much these directions "overlap" in ambient space.

**Step 2: Compute the Christoffel symbols.**

The Christoffel symbols Γ^k_{ij} are the "correction terms" that tell you how straight lines in latent space curve in the manifold. They're defined as:

Γ^k_{ij} = (1/2) g^{kl} (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij})

where g^{kl} is the inverse of the metric matrix. This is a standard formula from Riemannian geometry.

**Step 3: The key simplification.** The paper computes ∂_l g_{ij} (the derivative of the metric with respect to the l-th coordinate):

∂_l g_{ij} = Σ_m [(∂²x_m / ∂z_i ∂z_l)(∂x_m / ∂z_j) + (∂x_m / ∂z_i)(∂²x_m / ∂z_j ∂z_l)]

This is just the product rule applied to g_{ij} = Σ_m (∂x_m/∂z_i)(∂x_m/∂z_j).

**Step 4: The cancellation.** When you plug this into the Christoffel symbol formula (with the specific combination ∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij}), most terms cancel (equations 19-22 in the appendix), leaving:

(1/2)(∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij}) = Σ_m (∂²x_m / ∂z_i ∂z_j)(∂x_m / ∂z_l)

This cancellation is clean and the algebra checks out. It works because the metric is specifically J^T J — the Euclidean pullback metric. For a general metric, you wouldn't get this simplification.

**Step 5: Plug into the geodesic equation.** The geodesic equation is z̈^k + ż^i ż^j Γ^k_{ij} = 0, which gives:

z̈^k = -g^{kl} Σ_{i,j,m} (∂²x_m / ∂z_i ∂z_j)(∂x_m / ∂z_l) ż^i ż^j

**Verification:** This is indeed the standard result for pullback metrics in embedded manifolds. The paper notes it's similar to the geodesic induced by the Fisher information matrix with Gaussian likelihood — this makes sense because the Fisher information matrix IS a pullback metric (of the statistical manifold into probability space).

### What This Means Practically

To compute a geodesic, at each time step you need:
1. The Jacobian J (D×d matrix) — computed via forward-mode AD
2. The metric G = J^T J (d×d matrix) — cheap matrix multiply
3. The metric inverse G^{-1} (d×d matrix) — O(d^3) inversion
4. The second-order Jacobian (the "Hessian-like" term ∂²x_m/∂z_i∂z_j) contracted with the velocity — computed via two Jacobian-vector products

**For our setting (d ≤ 9, D = 4096):** The d×d operations are trivially cheap. The D×d Jacobian costs d forward-mode AD passes through the flow. The second-order terms cost another d forward-mode AD passes. Total cost per geodesic step: ~2d forward passes through the flow. For d=9, that's 18 forward passes. Feasible but not cheap if you need many geodesic evaluations.

## Algorithm 1: Multi-Chart Exponential Maps

### How It Works

For each time step t from 0 to T-1:
1. Compute the responsibilities r_c for the current point x_t under each chart
2. Filter to only charts above the responsibility threshold
3. For each responsible chart c, compute one geodesic step using chart c's geometry: (x^c_{t+1}, v^c_{t+1}) = Exp^c_{Δt}(x_t, v_t)
4. Average the results: x_{t+1} = Σ_c r_c x^c_{t+1}, v_{t+1} = Σ_c r_c v^c_{t+1}

### Critical Issues

**Averaging positions:** When you average x^c_{t+1} over charts, you get a point that doesn't lie exactly on any chart's manifold. In the next step, you use this averaged point as the starting point — but it might be off the manifold. The algorithm doesn't re-project onto the manifold between steps.

**Averaging velocities:** Even worse, velocities from different charts live in different tangent spaces. Averaging them in ambient space is only meaningful if the tangent spaces are nearly aligned. In regions of high curvature or where charts have very different orientations, this averaging could produce a velocity that points away from the manifold.

**The Δt hyperparameter:** The paper doesn't discuss how to choose T (the number of steps) or Δt = 1/T. Too few steps = large errors from the Euler approximation. Too many steps = accumulated floating-point errors. Table 2 shows significant variance across different solvers, suggesting this is a real practical issue.

## Algorithm 2: Multi-Chart Geodesics and Logarithmic Maps

### How It Works

1. Parameterize a curve γ_ϕ(t) in the ambient space (e.g., a spline)
2. For each optimization iteration:
   a. At each point along the curve, compute responsibility-weighted projection onto the manifold
   b. Compute the energy of the projected curve
   c. Optimize the spline parameters to minimize energy
3. Extract the logarithmic map as the initial velocity of the optimized curve

### Critical Issues

**Ambient parameterization:** The curve lives in D-dimensional ambient space, but the manifold is d-dimensional. This means the optimization has D degrees of freedom per control point, even though only d of them matter. The optimization could "cheat" by going through the interior of the ambient space (off the manifold) if the projection step doesn't perfectly constrain it.

**Non-convexity:** Energy minimization for geodesics is non-convex in general. Multiple local minima exist (e.g., the two geodesics connecting opposite points on a sphere). The paper acknowledges this — Figure 9 shows a case where the solver found the "wrong" geodesic on the torus (the path going the long way around). This is a fundamental limitation, not a bug.

**Computational cost:** Each optimization iteration requires: (a) evaluating the curve at T points, (b) computing responsibilities at each point, (c) projecting via responsibility-weighted reconstruction (each chart does a forward+inverse pass), (d) computing the energy. This is expensive. The paper says some evaluations "may need more than a day" for ill-defined geometries. For large-scale applications, this is prohibitive.

---

## Boundary Exploration: What Happens If We Weaken Assumptions?

### What if the manifold dimension d is unknown?

The paper assumes d is given. If d is wrong, everything breaks: the flows will either waste capacity (d too large) or fail to represent the manifold (d too small). Zhang et al. (2023) proposed a method to identify d during training, but it hasn't been integrated with multi-chart flows. For our project, Phase C gives us estimated dimensionalities, but these are upper bounds (rank of the subspace), not necessarily the true intrinsic dimensionality.

### What if the number of charts C is wrong?

Too few charts: the topology will still be wrong (e.g., 1 chart for a sphere leaves a gap).
Too many charts: each chart has less data, leading to worse individual flow quality. The paper adds a regularization term (equation 33) to prevent charts from becoming too uneven, but this is a band-aid.
There's no principled way to select C in this paper. In classical differential geometry, the minimum number of charts for a manifold is well-defined (e.g., 2 for S^1, 2 for S^2, etc.), but you need to know the manifold first.

### What if the data is noisy?

The paper doesn't explicitly add noise to the manifold data. Real data (including neural network activations) always has noise. If data points don't lie exactly on a d-dimensional manifold but are spread around it, the reconstruction penalty (equation 9) becomes critical. The λ parameter controls the tradeoff between density estimation and reconstruction fidelity. But how do you set λ when you don't know the noise level? The paper tunes it via grid search, which is unsatisfying.

---

## Techniques We Could Borrow

1. **Persistent homology for topology validation:** After we do Fourier screening and find circular structure, we could compute pairwise geodesic distances on the learned manifold and feed them to a persistent homology analysis. If H₁ = 1 (one loop), that confirms circular topology. This is a clean, model-agnostic validation tool.

2. **The pullback metric formulation (equation 5):** If we train any differentiable mapping from a low-dimensional latent space to the 4096D activation space (whether via flows, GPLVM, or even a simple autoencoder), we can compute the pullback metric and use it for geometric analysis. The geodesic equation (Theorem 1) applies to ANY such mapping. This is directly useful for our GPLVM step.

3. **Responsibility-weighted geometric averaging:** The idea of using soft assignments (responsibilities) to combine geometric estimates from multiple models/charts/populations is transferable. In our correct/wrong analysis, we could weight geometric estimates by population membership probabilities rather than hard assignments.

4. **The ambient-space curve parameterization for geodesics (Algorithm 2):** If we need to compute geodesics that cross between different concept subspaces (e.g., a path from one digit's representation to another), parameterizing the curve in the 4096D ambient space and projecting onto the manifold is a practical approach.

---

## Research Ideas Generated

### Idea 1: Persistent Homology Validation of Digit Manifold Topology
**What:** After Fourier screening identifies circular structure in digit representations, compute pairwise geodesic distances (using pullback metric from GPLVM) and run persistent homology. Check if H₁ = 1.
**Why interesting:** Clean topological validation independent of the Fourier analysis.
**Feasibility:** High. Off-the-shelf TDA tools exist (scikit-tda, Ripser). The challenge is getting reliable distance estimates.

### Idea 2: Multi-Chart vs. Single-Chart Geometry as a Diagnostic
**What:** Train both single-chart and multi-chart flows on digit activations. Compare persistence diagrams. If multi-chart reveals topology that single-chart misses, that's evidence for non-trivial manifold structure.
**Why interesting:** It turns the paper's methodology into a diagnostic tool for our interpretability pipeline.
**Feasibility:** Medium. Training normalizing flows on 4096D data is non-trivial. Would need to work in the identified subspaces (2-9D).

### Idea 3: Responsibility-Weighted Correct/Wrong Geometry
**What:** Adapt the EM responsibility mechanism to weight geometric analysis by population membership (correct/wrong). Instead of hard-assigning examples to correct or wrong groups, use soft responsibilities based on the model's internal confidence.
**Why interesting:** Avoids the sharp boundary between correct and wrong populations that Phase C currently imposes.
**Feasibility:** High. The EM framework is well-understood. The key question is what to use as the responsibility model.

---

## Final Assessment: Paper Quality and Relevance

### Overall Grade: B+

**Strengths:**
- Addresses a real mathematical limitation (topology) with the right mathematical tools (multiple charts)
- Clean experimental demonstration of the topology problem and its solution
- Geodesic computation algorithms are novel (for the ML community)
- Strong author team with deep expertise

**Weaknesses:**
- No uncertainty quantification (contradicts Hauberg's own prior work)
- Only low-dimensional, known-topology test cases
- Critical hyperparameters with no principled selection method
- No code release
- Incremental over the group's prior work (Kalatzis et al. 2022)

### Relevance to Our Project: Medium-High

The paper is not directly a method we'll adopt — we're planning GPLVM, not normalizing flows, for manifold characterization. But the insights are valuable:

1. **Topology matters.** If our digit manifolds have circular/toroidal topology, methods that assume Euclidean topology will produce biased geometry. This validates our concern about the limitations of linear methods.

2. **The pullback metric framework applies broadly.** Equation 5 and Theorem 1 work for any differentiable embedding, including GPLVM. We'll use the same mathematical machinery.

3. **Persistent homology is a valid topology diagnostic.** We should add it to our pipeline after Fourier screening.

4. **Multi-chart geometry is hard.** If our manifolds have non-trivial topology, computing geodesics becomes significantly more challenging. This is a practical concern for our pipeline.

5. **The Hauberg UQ tension is unresolved.** Even Hauberg's own group hasn't reconciled "Only Bayes should learn a manifold" with practical normalizing flow methods. This tension is real and affects our choice of methods: it further supports choosing GPLVM (probabilistic) over flows (deterministic) for our geometry characterization.

---

## Comparison with Related Transformer-Circuits.pub Work

### Gurnee et al. ("When Models Manipulate Manifolds")

Gurnee et al. found geometric manifolds (helical curves, tiled by SAE features) in Claude 3.5 Haiku's residual stream. Their manifolds are 1D curves in ~6D subspaces — exactly the setting where topology matters (a circle vs. a line).

This multi-chart flows paper provides the theoretical framework for WHY Gurnee et al.'s finding is important: if the character count manifold is truly circular, then any single-chart representation will miss the topology. The persistent homology validation this paper proposes could be used to confirm whether Gurnee et al.'s manifolds have the circular topology they visually appear to have.

However, Gurnee et al. used PCA and direct inspection — much simpler tools — and discovered rich geometric structure. The multi-chart flows paper's heavy machinery may not be necessary for discovery; it's more useful for rigorous characterization and distance computation on already-discovered manifolds.

### MOLT (Sparse Mixtures of Linear Transforms)

The MOLT paper showed that geometric operations (rotations on circles) are efficiently captured by sparse linear transforms, while transcoders "shatter" the geometry into lookup tables. Multi-chart flows are a different way to respect the same geometric structure — they learn the manifold that MOLT transforms operate on.

The connection is conceptual rather than practical: MOLT describes how computation happens on manifolds; multi-chart flows describe how to learn and characterize the manifolds themselves. Both validate the "geometric view" of transformer computation.

---

## Summary Table: What to Remember from This Paper

| Aspect | Key Point | Relevance to Us |
|--------|-----------|-----------------|
| Core problem | Single flows can't capture non-trivial topology | Directly relevant if digit manifolds are circles/tori |
| Solution | Mixture of flows with EM training | Possible validation tool, not primary method |
| Geodesics | Responsibility-weighted multi-chart algorithms | Useful if we need cross-chart geodesics |
| Topology validation | Persistent homology on learned distances | Should add to our pipeline |
| UQ gap | No uncertainty despite Hauberg being co-author | Supports our choice of GPLVM over flows |
| Scalability | Tested only on d≤3, D≤100 | Our d≤9, D=4096 setting is untested |
| Code | Not released | Cannot easily reproduce |