# Paper Analysis: Riemannian Laplace Approximations for Bayesian Neural Networks

**Authors:** Federico Bergamin, Pablo Moreno-Muñoz, Søren Hauberg, Georgios Arvanitidis  
**Institution:** DTU Compute, Technical University of Denmark  
**Venue:** NeurIPS 2023 (peer-reviewed, main conference track — top-tier ML venue)  
**arXiv:** 2306.07158 (June 2023 preprint, published December 2023)  
**Relevance to our project:** HIGH — Hauberg is a co-author; this paper uses the same geometric machinery (pull-back metrics, exponential maps, GPLVM-adjacent ideas) that our pipeline builds on for activation space analysis

---

## Pre-Reading: Context Check

**Who are these authors?**  
Søren Hauberg is the same researcher behind "Only Bayes Should Learn a Manifold" (2018), which is already a foundational reference in our pipeline. Arvanitidis is a collaborator on that line of work (ICLR 2018 "Latent Space Oddity"). This is the core DTU group that built the probabilistic Riemannian geometry toolkit. Their track record is strong and this is their natural domain.

**Where published?** NeurIPS 2023 — the flagship ML venue. Peer review passed. This is not a borderline or controversial paper; it went through rigorous review.

**Is there follow-up work?** Yes — a 2025 paper "Tubular Riemannian Laplace Approximations" explicitly extends and critiques this work, noting the ODE sampling bottleneck as the key limitation. That the field is already building on it is a positive signal.

**Code released?** The paper says "code will be released upon acceptance" — it was released with the NeurIPS version. Implementation uses `laplace` library + `functorch` + `scipy`.

---

## Phase -1: Paper Classification

**Type:** Empirical/Methods paper with light theory.  
The paper proposes a new method (RIEM-LA), proves one technical lemma (Lemma B.3, simplification of the geodesic ODE), and validates experimentally across regression and classification tasks. There are no asymptotic guarantees or statistical theorems — this is a principled engineering improvement with geometric motivation.

**Reading Strategy:** Visual-First Pass 1 → Standard Pass 2 → Pass 3 on the metric and geodesic ODE. The figures (Figure 3 especially) ARE the argument; the math in Section 3 is the mechanism.

---

## Phase 0: Pre-Reading — Why Are We Reading This?

**Our reading purpose:** Deep understanding. This paper is core to our research for two reasons:

1. **Same geometric toolkit**: The pull-back metric M(θ) = I + ∇L∇Lᵀ is the **same conceptual structure** as the metric tensors we plan to use in GPLVM for characterizing activation manifolds. The paper operationalizes Hauberg's framework in a concrete ML setting.

2. **Justification for probabilistic methods**: The paper's argument for *why* Gaussian approximations fail in curved spaces is exactly the argument for *why* our analysis needs GPLVM rather than PCA/LDA. The Section 3.1 geometry lesson directly supports our "only Bayes should learn a manifold" citation strategy.

---

## PASS 1: Jigsaw — What Does This Paper Do?

### Q1: What Is the Problem?

**The setup in plain English:**

Imagine you've trained a neural network to recognize images. The network has millions of "knobs" (weights, denoted θ) that were tuned during training. The best-performing setting of all these knobs is called θ* (theta-star), or the MAP (Maximum A-Posteriori) estimate — essentially the "most likely" weight configuration given your data.

Now, Bayesian statistics says: don't just use one setting of the knobs. Instead, maintain a **probability distribution** over all possible knob configurations — more probability where the knobs work well, less where they don't. This distribution is called the **posterior** p(θ|D). If you can sample different plausible weight configurations from this distribution, you can make predictions that come with calibrated uncertainty estimates — the network "knows what it doesn't know."

**The problem:** Computing this posterior exactly is mathematically impossible for any modern neural network. The weight space has millions of dimensions and the posterior has a complicated, non-symmetric shape. So we **approximate** it.

**The standard approximation: Laplace Approximation (LA)**

The classic trick (Laplace approximation, first used by Pierre-Simon Laplace in 1774, re-popularized for neural networks by MacKay 1992) is:

> "The posterior is weird and complicated globally. But locally, near the best point θ*, maybe it looks like a smooth hill. A hill is a Gaussian (bell curve). So let's approximate the posterior as a Gaussian centered at θ*."

Mathematically, this means doing a **second-order Taylor expansion** of the log-posterior around θ*:

```
log p(θ|D) ≈ log p(θ*|D) + [gradient term] + ½(θ-θ*)ᵀ H (θ-θ*)
```

The gradient term is zero at θ* (it's a minimum of loss = maximum of posterior). The H is the **Hessian** — a matrix of second derivatives that tells you the curvature of the loss surface. Inverting H gives the covariance of the Gaussian: Σ = H⁻¹.

So the Laplace Approximation says: posterior ≈ N(θ*, H⁻¹).

**Why does this fail?**

The loss landscape of a real neural network is NOT a smooth hill near θ*. It has ridges, valleys, and flat regions that are fundamentally non-Gaussian. Sagun et al. (2016) showed empirically that the Hessian at θ* is often not positive definite — meaning the "hill" has directions where it's not a hill at all but a flat mesa or a valley. When you sample from the Gaussian approximation N(θ*, H⁻¹), many samples land in regions with **high loss** (bad weights), not near the true posterior. A network initialized with those weights performs terribly.

This is the core problem: **the Gaussian approximation puts probability mass where the true posterior has almost none.**

**One sentence summary:**  
"This paper studies Bayesian inference for neural network weights in the setting where the Gaussian Laplace approximation fails because the true posterior is locally non-Gaussian, and proposes a Riemannian geometry-based fix."

---

### Q2: Why Is It Hard / Nontrivial?

The obvious fix is "just use a more flexible distribution." But flexible distributions come with costs:

- **Normalizing flows** (Kristiadi et al. 2022): Require training an additional neural network on top of your already-trained network. Expensive, and you need data to train it.
- **Variational inference** (Graves, Blundell): Requires optimization from scratch with a different objective. Can fail to converge.
- **MCMC / Hamiltonian Monte Carlo** (Neal 1995): Guaranteed to sample from the true posterior eventually, but infeasibly slow for large networks — billions of weight samples needed.
- **Deep ensembles** (Lakshminarayanan 2017): Train multiple full networks. 5× or 10× the compute.

The Laplace approximation's main selling point is **simplicity** — you already trained the network; you just need the Hessian at the end. Any fix that requires major additional computation destroys this advantage.

The **nontrivial question** is: *Can we fix the Gaussian approximation's poor fit to the posterior without abandoning the parametric simplicity of a Gaussian distribution?*

**One sentence summary:**  
"This is nontrivial because the posterior is non-Gaussian in ways that defeat simple parametric fixes, and more flexible alternatives require expensive auxiliary computation that defeats the practical appeal of the Laplace approximation."

---

### Q3: What Is the Main Claim?

**The key insight (explained simply):**

The authors argue: *The problem is not that we use a Gaussian — it's that we use a Gaussian in the wrong space.*

Here's an analogy. Suppose you live on the surface of the Earth and someone tells you: "The nearest city is 500 km away in a straight line." But if you travel in a straight line, you'd go underground! The actual path along the Earth's surface (the **geodesic**) is curved — it goes around the surface, not through it.

Similarly, when we draw "nearby" samples from a Gaussian in weight space, we're measuring distance with a ruler that says "100 km in flat Euclidean space." But the neural network's loss landscape is not flat — it's curved. Moving in some directions barely changes the loss (flat landscape = easy), while moving in other directions spikes the loss immediately (steep landscape = bad). A Gaussian that ignores this curvature treats all directions equally and happily samples into high-loss "underground" regions.

**The fix:**  
Define a **Riemannian metric** (a position-dependent ruler) on weight space that accounts for the loss landscape. Make "distance" larger in directions where the loss increases rapidly. Then:

1. Do the Laplace approximation's Gaussian **in the tangent space** of this curved metric (where the geometry is locally flat)
2. **"Push" the samples back** to the original weight space by following the **geodesic** (the shortest curved path on the loss surface)

Because the geodesic naturally hugs the low-loss valley, samples arrive in good weight regions.

**The specific metric they choose:**

M(θ) = I_K + ∇_θL(θ) · ∇_θL(θ)ᵀ

Where:
- I_K is the K×K identity matrix (the "flat" Euclidean part)
- ∇_θL(θ) is the gradient of the loss — a vector of length K
- ∇_θL(θ) · ∇_θL(θ)ᵀ is the **outer product** — a K×K matrix

This metric is **rank-1 perturbed identity**: it adds the gradient's self-outer-product to the identity. The meaning: in directions of large gradient (high loss slope), distances are stretched. In flat directions (gradient ≈ 0), the metric is approximately identity (unchanged). The metric thus automatically encodes the loss landscape locally.

**The sampling procedure:**

1. Sample a vector v from the standard Laplace Gaussian: v ~ N(0, Σ)
2. Think of v as an initial velocity pointing away from θ*
3. Solve the **geodesic ODE** starting at θ* with initial velocity v
4. The endpoint θ* + [geodesic path] = your actual weight sample

The ODE is:
```
c̈(t) = -∇L(c(t)) · (1 + ∇L(c(t))ᵀ ∇L(c(t)))⁻¹ · ⟨ċ(t), H[L](c(t)) ċ(t)⟩
```
Where c(t) is the curve through weight space, c̈(t) is its acceleration, and ⟨ċ, H ċ⟩ is a curvature term. The ODE is solved numerically (Runge-Kutta 4/5 order via scipy).

**One sentence summary:**  
"They show that a Laplace-style Gaussian in the tangent space of a loss-gradient-derived Riemannian metric, pushed back to weight space via the exponential map (geodesic ODE), produces samples that consistently land in low-loss regions and improves over standard Laplace approximation across regression and classification tasks."

---

### Pass 1 Template Summary

```
PAPER: Riemannian Laplace Approximations for Bayesian Neural Networks
AUTHORS: Bergamin, Moreno-Muñoz, Hauberg, Arvanitidis | VENUE: NeurIPS 2023 | YEAR: 2023
TYPE: Empirical/Methods (proposes new method with geometric motivation)
PURPOSE: Deep understanding — core toolkit and justification for our GPLVM approach
PEER-REVIEWED: Yes — NeurIPS 2023 main track

PROBLEM: Laplace approximation for BNN posterior samples land in high-loss weight regions
WHY HARD: Can't easily use more flexible approximations without losing parametric simplicity
MAIN CLAIM: Riemannian metric M = I + ∇L∇Lᵀ warps weight space so Laplace samples naturally 
            land in low-loss regions; implemented via geodesic ODE (exponential map)
RELEVANCE TO MY WORK: Same DTU group as Hauberg 2018; pull-back metrics are exactly what we 
                      use for GPLVM; supports our probabilistic-methods-only argument

VERDICT: Proceed to Pass 2 — HIGH priority
PRIORITY: High
```

---

## PASS 2: Scuba Dive — How Does It Work?

### Background You Need: Five Concepts Explained Simply

Before diving into the technical details, you need five concepts. Each is built from grade-12 math.

---

**Concept 1: What is a posterior distribution?**

Imagine you have a coin. Before flipping it, your "prior" belief might be that it's fair (50% heads, 50% tails). You flip it 10 times and get 8 heads. Your "posterior" updates that belief — now you think the coin might be biased toward heads.

For a neural network: before seeing data, you have a prior p(θ) over weights — maybe you think weights should be small (regularization). After seeing data D, you update to a posterior p(θ|D). The posterior tells you: "given all the data I've seen, how probable are different weight configurations?"

Bayes' rule: p(θ|D) = p(D|θ) · p(θ) / p(D)

- p(D|θ) = likelihood: how well weights θ explain the data
- p(θ) = prior: your initial belief about weights
- p(D) = normalizing constant (makes it integrate to 1 — this is the intractable part)

For a neural network with millions of weights, computing p(D) requires integrating over an impossibly large space. This is why we approximate.

---

**Concept 2: What is a Taylor expansion (in plain English)?**

A Taylor expansion says: "I can't evaluate a complicated function everywhere, but I can approximate it near a known point using derivatives."

Near a point θ*, any smooth function f(θ) is approximately:

f(θ) ≈ f(θ*) + f'(θ*) · (θ - θ*) + ½ f''(θ*) · (θ - θ*)²

The **first derivative** f'(θ*) tells you the slope. The **second derivative** f''(θ*) tells you the curvature (how quickly the slope is changing — a positive curvature means the function curves up like a bowl; negative means it curves down).

In multiple dimensions: the second derivative becomes the **Hessian matrix** H. The Hessian at θ* tells you: for every pair of weight dimensions, how do they jointly curve the loss? If you invert the Hessian, you get H⁻¹, which is the **covariance** of the Gaussian approximation.

**The problem with this for neural networks:**
The Laplace approximation assumes the loss looks like a parabolic bowl near θ*. But neural network loss surfaces are famously NOT parabolic — they have flat directions (manifolds of equal loss), sharp walls, and multiple local minima. The bowl assumption fails.

---

**Concept 3: What is a Riemannian metric?**

On a flat piece of paper, distance is computed by the Pythagorean theorem: d² = Δx² + Δy². This is the Euclidean (flat) metric.

On the surface of the Earth, the distance between two points isn't measured by a straight line through the Earth — it's measured along the curved surface. The formula for distance on a curved surface at each point depends on that point's local geometry — this is the **Riemannian metric** M(x). It's a matrix that varies from point to point and tells you: "at location x, if I move a small amount in direction v, the true distance is √(vᵀ M(x) v)."

The Riemannian metric is always:
- **Symmetric**: M(x) = M(x)ᵀ
- **Positive definite**: vᵀ M(x) v > 0 for any non-zero v (distances are positive)
- **Smooth**: M changes continuously with x

In this paper, x = θ (the weights), and the metric M(θ) = I + ∇L(θ)∇L(θ)ᵀ. At points where the gradient is zero (minimum of loss = maximum of posterior), M = I (flat Euclidean space — nothing changes). At points where the gradient is large (high loss), M is large in the gradient direction (distances are stretched — the space curves away from high-loss regions).

---

**Concept 4: What is a geodesic and the exponential map?**

A **geodesic** is the shortest path between two points on a curved surface. On a sphere (like Earth), geodesics are great circles (like flight paths).

The **exponential map** Exp_x(v) answers: "Starting at point x, moving in direction v with speed ||v||, where do I end up after walking for 1 unit of time along the geodesic?"

On flat Euclidean space: Exp_x(v) = x + v (trivially — you just add the vector).

On a curved space: Exp_x(v) = the endpoint of the geodesic starting at x with velocity v. Computing this requires solving a differential equation.

The authors use the exponential map as a **sampler**:
1. Draw a velocity v from the Gaussian (standard Laplace sample)
2. Compute Exp_{θ*}(v) — follow the geodesic from θ* in direction v
3. The endpoint is your weight sample

Because geodesics follow low-loss valleys (the curved metric makes high-loss paths longer, so the shortest path hugs the valley), the endpoint naturally lands in a good region.

---

**Concept 5: What is a pull-back metric?**

Suppose you have a function g: A → B that maps space A into space B. The "pull-back metric" of B's geometry onto A asks: "if I measure distances in B, what does that imply about distances in A?"

Formally: if g maps θ (in weight space A = ℝᴷ) to [θ, L(θ)] (in loss-augmented space B = ℝᴷ⁺¹), then the Jacobian of g is:

J_g(θ) = [I_K; ∇_θL(θ)]ᵀ     (dimension K+1 × K)

The inner product in B between two tangent vectors J_g v₁ and J_g v₂ is:

⟨J_g v₁, J_g v₂⟩_B = v₁ᵀ J_gᵀ J_g v₂

So the **pull-back metric** is M(θ) = J_gᵀ J_g = I_K + ∇L(θ) ∇L(θ)ᵀ.

This is the metric in Equation 3 of the paper. It's elegant: you're literally asking "what does flat Euclidean geometry in the (weight, loss) space look like when pulled back to just the weight space?" The answer automatically encodes the loss landscape.

---

### Q2.1: What Was the Technical Hurdle Before This Paper?

**The hurdle:** Standard Laplace approximation generates samples in Euclidean weight space. The Gaussian puts probability mass in all directions equally weighted by the inverse Hessian. But the Hessian at θ* doesn't capture the **global** shape of the loss landscape — it only captures the local curvature. Beyond a tiny neighborhood of θ*, the Gaussian approximation is entirely unaware of the loss surface's shape, and samples wander into high-loss regions.

**Previous fixes and their problems:**

Previous work (Mackay 1992) proposed **linearization**: instead of sampling weights and evaluating f_θ(x), use the linear approximation f_θ*(x) + ⟨J, θ-θ*⟩. This partially helps because linearized predictions are more stable, but it doesn't fix the underlying problem that samples from the Gaussian are still in bad weight regions.

The **key insight that unlocks progress:**

> The problem is in the SPACE where approximation is done, not the DISTRIBUTION being used.

If you use a curved metric that makes high-loss directions "feel" far away, then even a Gaussian (which maximally spreads in "close" directions under the metric) will concentrate in low-loss regions. You don't need a more complex distribution — you need the right notion of distance.

The mathematical move: Do the Laplace approximation in **normal coordinates** of the tangent space at θ* (where the curved metric looks flat), then map back via the exponential map. The Gaussian is still Gaussian in the tangent space; it's just that "nearby" in the tangent space = "low loss" in the original space.

**Critical question:** Is this really a new insight, or just a re-framing? The pull-back metric idea (Tosi et al. 2014, Arvanitidis et al. 2018) has been used in latent variable models before. The novelty is applying it specifically to the **BNN weight posterior** and making it practical via the ODE simplification (Lemma B.3). The metric choice M = I + ∇L∇Lᵀ is specifically tuned for tractability — more complex metrics (e.g., full Fisher information) would be better geometrically but computationally prohibitive.

---

### Q2.2: Detailed Technical Walkthrough

#### Step 1: Define the Loss Surface as a Manifold

Take weight space Θ = ℝᴷ and define g: Θ → ℝᴷ⁺¹ by:

g(θ) = [θ₁, θ₂, ..., θ_K, L(θ)]

This lifts each weight vector to a (K+1)-dimensional point by appending the loss value. The image of g is a K-dimensional surface (manifold) living in ℝᴷ⁺¹ — the loss landscape graph.

The **Jacobian** of g is J_g(θ) = [I_K | ∇_θL(θ)]ᵀ, where I_K is K×K identity and ∇_θL is the K-dimensional gradient vector.

The **pull-back metric** from ℝᴷ⁺¹ onto Θ:

M(θ) = J_g(θ)ᵀ J_g(θ) = I_K + ∇_θL(θ)∇_θL(θ)ᵀ

**Sanity check:** Does this make sense dimensionally? M is K×K (good — it must be square). Is it positive definite? Yes: vᵀMv = vᵀv + (∇L·v)² ≥ 0, and equals zero only if v=0. ✓

**What does this metric do?**

At θ*: gradient is approximately zero (it's a minimum), so M(θ*) ≈ I_K. The space is **locally flat** at the optimum.

Away from θ*: the gradient is large in some directions, so M is large in those directions. Distance in those directions is stretched — you'd have to travel much farther (by the metric's measure) to reach a high-loss region than the Euclidean distance suggests.

#### Step 2: Normal Coordinates — Flattening the Curved Space Locally

At any point on a curved manifold, you can choose local coordinates where the metric looks like the identity at that point (called **normal coordinates**). This is the manifold equivalent of choosing a coordinate system aligned with the principal directions of the curvature.

The transformation from tangential coordinates v to normal coordinates v̄:

v = A(θ*) v̄   with   A(θ*) = M(θ*)^{-1/2}

(The matrix square root of the inverse metric.)

In normal coordinates: ⟨v, M(θ*)v⟩ = ⟨v̄, v̄⟩ — the metric is identity.

**Why does this matter?** Because the Laplace approximation (second-order Taylor expansion) is most natural when the geometry is flat. In normal coordinates, we can do the Taylor expansion exactly as in the standard Laplace case, but now it's curved-geometry-aware.

#### Step 3: Taylor Expand in the Tangent Space

Define h(v̄) = L(Exp_{θ*}(M(θ*)^{-1/2} v̄)) — the loss evaluated at the point reached by following the geodesic from θ* in direction A v̄.

Taylor expand h around v̄ = 0:

ĥ(v̄) ≈ h(0) + ⟨∂_{v̄}h|_{v̄=0}, v̄⟩ + ½⟨v̄, H_{v̄}[h]|_{v̄=0} v̄⟩

The first-order term vanishes because θ* is a loss minimum (gradient = 0). The Hessian term is:

H_{v̄}[h]|_{v̄=0} = A(θ*)ᵀ H_θ[L](θ*) A(θ*)

This gives a Gaussian distribution q̄(v̄) = N(v̄ | 0, Σ) in normal coordinates with:

Σ = (A(θ*)ᵀ H_θ[L](θ*) A(θ*))^{-1}

**Critical observation:** In tangential coordinates v (not normal), the covariance is:

v ~ N(0, A Σ Aᵀ) = N(0, H_θ[L](θ*)^{-1})

**This is exactly the standard Laplace approximation covariance!** The Riemannian approach doesn't change what Gaussian you start from — it changes what you DO with the samples after drawing them. The transformation A(θ*) is symmetric and cancels in the covariance formula.

#### Step 4: Map Samples Back via the Exponential Map

Draw tangent vector v from q_T(v) = N(v | 0, H^{-1}). This is your "velocity." Then compute:

θ_sample = Exp_{θ*}(v) = endpoint of geodesic starting at θ* with velocity v

This is where the geometry actually does work. The geodesic naturally follows the loss surface, keeping you in low-loss regions. The final weight sample θ_sample is a valid weight configuration with good properties.

#### Step 5: The Simplified Geodesic ODE

The general geodesic ODE on a Riemannian manifold is complicated — it involves **Christoffel symbols**, which require third derivatives of the metric. For most metrics, this is intractable.

**The key technical result (Lemma B.3):** For the specific metric M(θ) = I + ∇L∇Lᵀ, the geodesic equation simplifies dramatically. Using the **Sherman-Morrison formula** (a trick for inverting rank-1-updated matrices):

(I + ∇L ∇Lᵀ)^{-1} = I - ∇L∇Lᵀ / (1 + ||∇L||²)

The geodesic ODE becomes:

c̈(t) = -∇L(c(t)) · (1 + ||∇L(c(t))||²)^{-1} · ⟨ċ(t), H[L](c(t)) ċ(t)⟩

**Reading this equation:**
- c̈(t) is the acceleration of the geodesic (how it curves)
- ∇L(c(t)): gradient at current position — points "uphill" in loss
- (1 + ||∇L||²)^{-1}: a normalizing scalar
- ⟨ċ, Hċ⟩: a scalar measuring how the current velocity interacts with the curvature (Hessian-vector product)

The acceleration is always in the **gradient direction** — the geodesic is "pushed back" toward low-loss directions by the loss gradient. This is the geometric mechanism: the curved space makes the geodesic "fall" into the loss valley.

**Practical advantage:** Computing ⟨ċ, Hċ⟩ requires only a **Hessian-vector product**, not the full K×K Hessian matrix. This is computable in O(K) memory using modern automatic differentiation (backward-over-forward). For a network with millions of parameters, storing the full Hessian is impossible, but Hessian-vector products are tractable.

---

### Q2.3: Comparison to Baselines

**Methods compared:**
- MAP: point estimate, no uncertainty
- Vanilla LA: standard Laplace approximation (the broken baseline)
- Lin-LA: linearized Laplace (function linearized around θ*, then Laplace applied)
- RIEM-LA (theirs): Riemannian Laplace
- Lin-RIEM-LA (theirs): Riemannian + linearized

**Key results:**

On banana dataset (2D binary classification):
- Vanilla LA accuracy: 59.5% (with prior optimized) — catastrophically bad
- RIEM-LA accuracy: 87.6% — near-perfect, better than MAP (86.7%)
- NLL: Vanilla LA = 0.678, RIEM-LA = 0.287 — 57% better

On UCI datasets (6 datasets, tabular classification):
- RIEM-LA is the best method on NLL in 5 out of 6 datasets
- RIEM-LA consistently outperforms vanilla LA by large margins (e.g., Vehicle NLL: 1.209 → 0.454)

On MNIST (CNN, 5000 training examples):
- RIEM-LA accuracy: 96.74% vs MAP 95.02% — better than the deterministic baseline!
- RIEM-LA NLL: 0.115 vs vanilla LA: 0.871 — 87% better

On FashionMNIST:
- RIEM-LA accuracy: 83.33% vs MAP 79.88% — again better than MAP
- Consistent NLL improvement

On OOD detection (MNIST trained model vs FashionMNIST/EMNIST/KMNIST):
- RIEM-LA AUROC: 0.911-0.953 vs Lin-LA: 0.854-0.930 — consistently better

**Critical assessment of baselines:**
- Deep ensembles are NOT included. This is the standard strong baseline for uncertainty calibration and is known to outperform all single-model Bayesian approximations. The omission is conspicuous.
- No comparison to SWAG (Stochastic Weight Averaging Gaussian) or other practical BNN methods.
- The CNN tested on MNIST is tiny (two conv layers, four channels each). Real-world networks have millions of parameters. The ODE solver cost scales with network size.

---

### Q2.4: What's Still Open / Limitations

**Computational cost (acknowledged by authors):**
Integrating the geodesic ODE is the central bottleneck. For each sample, you must run a numerical ODE solver — scipy's Runge-Kutta 5(4) with default tolerances. The cost scales linearly with training data (to compute gradients) and requires multiple Hessian-vector products per step. For Llama 3.1 8B (our model) with 8 billion parameters, this would be completely intractable. The paper acknowledges this: "our implementation relies on an off-the-shelf ODE solver and we expect significant improvements can be obtained using a tailor-made numerical integration method."

Follow-up work (2025 Tubular Riemannian Laplace) explicitly identifies this as the key weakness: "bergamin2023riemannian use a Monge-type metric to define an ODE that transports Gaussian samples through curved parameter space... the iterative nature of the solver scales linearly with the number of samples."

**Metric choice is ad-hoc (not acknowledged adequately):**
The metric M = I + ∇L∇Lᵀ is chosen because it makes the ODE tractable (via Sherman-Morrison) and avoids third-order derivatives. It is NOT the Fisher information metric (which would be theoretically better motivated for Bayesian inference). It's also not the GGN (Gauss-Newton) approximation to the Hessian. The authors don't really justify WHY this particular metric is the right one — they justify it on tractability grounds, but tractability ≠ correctness.

**The linearized version underperforms:**
Lin-RIEM-LA often performs worse than vanilla Lin-LA, especially when using the full dataset for ODE integration. The authors explain this as "over-regularization" — the linearized metric is only small near θ*, so samples stay too close to it, reducing diversity. This is a genuine failure mode, not a minor caveat.

**No asymptotic guarantees:**
There's no theorem saying "RIEM-LA converges to the true posterior as we do more samples" or "RIEM-LA is a better approximation than LA in some formal sense." The improvements are purely empirical. A critic could argue that RIEM-LA just happens to work for the specific tasks tested.

**tanh activations only (for the geometry to be smooth):**
The appendix explicitly restricts to smooth activations: "to satisfy the smoothness condition for M we restrict to activation functions as the tanh." For networks with ReLU (the most common activation), the gradient ∇L is not smooth — it's discontinuous at zeros. The metric M = I + ∇L∇Lᵀ would then have discontinuities, violating the Riemannian manifold assumptions. All experiments use tanh. This is a significant restriction for practical deployment.

**Small-scale experiments:**
The largest network is a small CNN on MNIST/FashionMNIST (two conv layers, three FC layers). No experiments on ResNets, VGGs, or modern transformers. The paper is honest about this: "the high dimensionality of the parameter space is one of the main limitations of the ODE solver." But this limits the practical relevance claim.

---

### Q2.5: Devil's Advocate — Three Weakest Points

**Weakness 1: No comparison to deep ensembles.**  
Deep ensembles (Lakshminarayanan et al. 2017) are the gold standard for uncertainty calibration in BNNs and are known to outperform all single-model approximate Bayesian methods. The paper explicitly acknowledges ensembles in the introduction but never compares against them in experiments. This omission significantly weakens the "RIEM-LA is better" claim. Is RIEM-LA beating the right baselines? We don't know.

**Weakness 2: The metric M = I + ∇L∇Lᵀ is rank-1 perturbed identity.**  
This metric only "stretches" space in the gradient direction — a single direction at each point. Real loss landscapes have complicated curvature in many directions simultaneously. The Fisher information matrix, which is the natural metric for Bayesian inference, captures the full second-order curvature structure. The authors' metric is a proxy motivated by tractability, not by principled geometric reasoning. The claim that it "adapts to the shape of the true posterior" is somewhat overstated — it adapts only to the gradient direction, not to the full Hessian structure.

**Weakness 3: The "improvement over vanilla LA" baseline comparison is somewhat unfair.**  
Vanilla LA (direct sampling from N(θ*, H⁻¹)) is known to perform catastrophically in practice, which is why **linearized LA** became the standard. The paper shows large improvements over vanilla LA (which is basically broken) but more modest improvements over Lin-LA (the actual practical baseline). The improvements over Lin-LA are real but smaller, and RIEM-LA's linearized variant sometimes underperforms Lin-LA. A more honest framing would compare primarily against Lin-LA, not vanilla LA.

---

### Q2.6: Connections to Our Research

**Connection 1 (Direct): Pull-Back Metric as the Metric Tensor Concept**

The metric M(θ) = I + ∇L∇Lᵀ is a **pull-back metric** — it measures distances in weight space by reference to a higher-dimensional space (the loss landscape). Our GPLVM pipeline does the same thing conceptually: it fits a metric tensor over the activation space, learned from the data (the concept labels). The difference is that Bergamin et al. define the metric analytically (gradient formula), while GPLVM learns it probabilistically from data.

The key shared principle: **curved space requires curved measurement**. A metric that ignores the structure of the space gives misleading distance estimates, and misleading distances lead to misleading geometry. This is exactly why we reject PCA/SVD as final answers (they use flat Euclidean distance) in favor of GPLVM (which learns the metric).

**Connection 2 (Direct): Justification for the Exponential Map / Geodesic ODE**

Bergamin et al. show that the exponential map can be made computationally practical for the specific metric M = I + ∇L∇Lᵀ via the Sherman-Morrison simplification. Our pipeline uses the exponential map conceptually in the GPLVM framework (Hauberg 2018's argument about "only Bayes should learn a manifold" uses exactly this formalism). The tractability of the ODE in Bergamin et al. suggests that computing geodesics under simple rank-1-updated metrics is tractable — relevant if we ever need to compute geodesic distances in our activation manifolds.

**Connection 3 (Important): Metric = Gradient Outer Product as a Special Case**

Our Phase C uses PCA (eigendecomposition of conditional covariance) to find concept subspaces. The Bergamin metric M(θ) = I + ∇L∇Lᵀ is essentially the **Fisher score function** metric from information geometry — the gradient of log-probability forms the natural metric in exponential families. Our concept covariance matrices are second-order statistics; the paper uses first-order statistics (gradient outer product). Both are trying to capture the structure of a probability model's geometry. The relationship suggests our Phase C subspaces and the Bergamin metric are measuring related but distinct aspects of the geometry.

**Connection 4 (Critical): Limitation for Activation Space Analysis**

The Bergamin framework operates on **weight space** (parameters θ). Our research operates on **activation space** (representations r). The pull-back metric from loss to weight space doesn't directly apply to activation space. We'd need a different derivation to get an analogous metric on activations. This is possible (the Fisher information metric on representations has been studied) but is not what this paper provides.

**Connection 5 (Cautionary): tanh restriction**

The paper requires smooth activations (tanh). Llama 3.1 8B uses SiLU (Sigmoid Linear Unit) activations, which are smooth and differentiable everywhere (unlike ReLU). So the theoretical framework does apply to our model, but SiLU ≠ tanh in terms of gradient behavior — the geometry would look different.

---

### Pass 2 Template Summary

```
PAPER: Riemannian Laplace Approximations for BNNs

TECHNICAL BARRIER: Laplace Gaussian puts probability mass in high-loss weight regions;
                   flexible fixes (flows, MCMC) are too expensive
KEY INSIGHT: Don't change the distribution — change the metric on weight space.
             M = I + ∇L∇Lᵀ stretches space in gradient directions so Gaussian samples
             naturally end up in low-loss valleys via geodesic (exponential map)
COMPARISON: RIEM-LA consistently beats vanilla and linearized LA; no comparison to ensembles
LIMITATIONS: ODE solver cost scales with dataset size and network size; restricts to smooth
             activations (tanh); linearized version unreliable; metric choice ad-hoc;
             no asymptotic guarantees; no comparison to deep ensembles
TRANSFERABILITY: Pull-back metric concept → directly relevant to our GPLVM pipeline;
                 exponential map as sampler → relevant to geodesic distance computation;
                 does NOT directly apply to activation space (weight space paper)

KEY ASSUMPTIONS:
  1. Loss L(θ) is smooth (requires tanh, not ReLU)
  2. θ* is a good MAP estimate (training has converged)
  3. ODE solver is accurate enough (default tolerances may not be tight enough)
  4. Mini-batching the metric is a valid approximation (unclear theoretically)

DEVIL'S ADVOCATE — THREE WEAKEST POINTS:
  1. No comparison to deep ensembles — the actual gold standard baseline is missing
  2. Metric M = I + ∇L∇Lᵀ is rank-1, only captures one direction of curvature per point
  3. Vanilla LA baseline is broken — comparison to Lin-LA shows more modest improvement

OPEN QUESTIONS:
  1. Can this be made tractable for large networks (like Llama 8B) without ODE?
  2. Is there a principled justification for the metric choice beyond tractability?
  3. Does the approach work with ReLU/SiLU networks (non-smooth gradients)?
  4. Can the pull-back metric idea be applied to ACTIVATION space rather than weight space?

CONNECTIONS TO OUR WORK:
  - Hauberg group: direct pipeline of ideas from "Only Bayes Should Learn a Manifold" (2018)
  - Pull-back metric: the same concept we use in GPLVM (different domain: activations vs weights)
  - Exponential map: directly used in GPLVM sampling
  - Critical gap: this paper is about WEIGHT space; our research is about ACTIVATION space
  - Supports our "probabilistic geometry > linear probes" argument
  - Confirms that tanh-style smooth activations are needed; Llama uses SiLU (smooth — okay)

VERDICT: Proceed to Pass 3 on Section 3.1 (metric) and Lemma B.3 (ODE simplification)
```

---

## PASS 3: The Swamp — Deep Technical Verification

### Deep Dive 1: The Metric M(θ) = I + ∇L∇Lᵀ

**Step-by-step derivation from scratch:**

We have the map g: ℝᴷ → ℝᴷ⁺¹:
```
g(θ) = [θ₁, θ₂, ..., θ_K, L(θ)]
```

The Jacobian J_g (how g changes with θ) is a (K+1)×K matrix:
```
J_g(θ) = [1    0    ... 0  ]   ← Row 1: ∂g₁/∂θⱼ = δ₁ⱼ
          [0    1    ... 0  ]   ← Row 2
          [... ...  ... ... ]
          [0    0    ... 1  ]   ← Row K
          [∂L/∂θ₁  ...  ∂L/∂θ_K]  ← Row K+1: gradient of loss
```

More compactly: J_g(θ) = [I_K; ∇_θL(θ)]ᵀ

The pull-back metric is M(θ) = J_g(θ)ᵀ J_g(θ):
```
M(θ) = [I_K | ∇_θL(θ)] · [I_K; ∇_θL(θ)ᵀ]
      = I_K · I_K + ∇_θL(θ) · ∇_θL(θ)ᵀ
      = I_K + ∇_θL(θ)∇_θL(θ)ᵀ
```

This is an identity matrix plus a **rank-1 outer product** matrix. The outer product ∇L∇Lᵀ has rank exactly 1 (it's a vector times a vector-transposed). So M has rank K (full rank), is positive definite, and is invertible. ✓

**What does the inverse look like?**

Using the Sherman-Morrison formula for rank-1 updates of the identity:

(I + uuᵀ)^{-1} = I - uuᵀ / (1 + uᵀu)

With u = ∇L(θ):

M(θ)^{-1} = I_K - ∇L(θ)∇L(θ)ᵀ / (1 + ||∇L(θ)||²)

**Intuitive reading:** The inverse metric is identity minus a small "bleed-out" in the gradient direction, scaled by 1/(1+||∇L||²). In the gradient direction, the inverse metric is smaller — meaning the "dual" space (velocity space) is compressed in that direction. This corresponds to the primal space being stretched.

**Is this a reasonable metric for Bayesian inference?**

The "right" metric for Bayesian inference in parameter space is the **Fisher Information Matrix** (FIM):

F(θ) = E_{y~p(y|x,θ)}[∇_θ log p(y|x,θ) ∇_θ log p(y|x,θ)ᵀ]

The FIM measures the expected curvature of the log-likelihood — it's the natural Riemannian metric on statistical manifolds (Amari's information geometry).

The paper's metric M = I + ∇L∇Lᵀ uses the **actual gradient** at the current point rather than the **expected gradient** (FIM). This is:
- Cheaper to compute (no expectation needed)
- Specific to the current θ (FIM averages over data)
- Not invariant to reparametrization (FIM is — this is a theoretical weakness)

The follow-up paper (Fisher-Laplace, 2024) uses the actual FIM and claims better results, confirming that M = I + ∇L∇Lᵀ is a tractable approximation to a more principled choice.

**Critical question: Why not use the Hessian directly?**

The Hessian H = ∇²L gives the full second-order curvature. Using M = H (or H + αI for numerical stability) would give the standard Laplace approximation's metric. But:

1. The full Hessian requires O(K²) storage — impossible for large networks
2. The Hessian is not always positive definite (Sagun et al. 2016) — not a valid Riemannian metric
3. The GGN (Gauss-Newton) approximation is positive semi-definite but still O(K²) storage

The paper's M = I + ∇L∇Lᵀ requires only O(K) storage (just the gradient vector) — this is the key computational advantage.

---

### Deep Dive 2: Lemma B.3 — The ODE Simplification

**General geodesic ODE (for any Riemannian metric):**

c̈(t) = -½ M⁻¹(c(t)) · [stuff involving ∂M/∂c and the Kronecker product of ċ with itself]

This involves derivatives of the metric M with respect to θ, which for M = I + ∇L∇Lᵀ means **Hessian times Hessian** or **third derivatives of L** in general.

**The simplification for M = I + ∇L∇Lᵀ:**

The paper shows (Appendix B, Lemma B.3) that for this specific metric, the general ODE reduces to:

c̈(t) = -∇L(c(t)) / (1 + ||∇L(c(t))||²) · ⟨ċ(t), H[L](c(t)) ċ(t)⟩

**Step-by-step proof outline (simplified):**

The general ODE has three terms:
1. M⁻¹ term
2. ∂M/∂cᵢ terms (how metric changes with position)
3. Kronecker product c̈ ⊗ c̈ term

For M = I + ∇L∇Lᵀ:
- M⁻¹ = I - ∇L∇Lᵀ/(1+||∇L||²) [Sherman-Morrison]
- ∂M/∂cᵢ = ∂(∇L∇Lᵀ)/∂cᵢ = Hᵢ∇Lᵀ + ∇LHᵢᵀ [product rule, Hᵢ = i-th column of Hessian]

After careful matrix algebra (which the appendix shows step by step), the terms combine to:

c̈(t) = -(∇L / (1 + ||∇L||²)) · ⟨ċ, Hċ⟩

**Key insight:** The only Hessian computation needed is the **scalar** ⟨ċ, Hċ⟩ — a Hessian-vector product evaluated at ċ. This requires no Hessian storage.

**What does this ODE mean physically?**

The acceleration c̈ is always in the gradient direction ∇L. The magnitude of acceleration is:
- Proportional to ||⟨ċ, Hċ⟩|| — how much the Hessian "bends" the current velocity
- Damped by 1/(1+||∇L||²) — near a loss minimum (||∇L||≈0), damping is small; far away (large gradient), damping approaches 1/||∇L||²

**Physical interpretation:** The geodesic is "attracted" toward the gradient direction (which points to high loss) and must be "pulled back" by the curvature term. The net effect is a curved path that navigates the loss landscape.

**Verification: Does this reduce to flat Euclidean at θ*?**

At θ*: ∇L(θ*) ≈ 0. So c̈ ≈ 0. The ODE becomes c̈ = 0, which means c(t) = θ* + t·v (straight line = geodesic in flat space). ✓ This is consistent with M(θ*) ≈ I (flat metric at the optimum).

---

### Deep Dive 3: The Normal Coordinates Calculation

**Why normal coordinates?**

In Riemannian geometry, the **Taylor expansion of a function** around a point looks different depending on which coordinate system you use. In arbitrary coordinates, the second-order term includes **Christoffel symbols** (correction terms for the curvature). In normal coordinates, these Christoffel symbols vanish — the Taylor expansion looks like the flat Euclidean one.

**The calculation:**

In tangential coordinates v: the metric is M(θ*). In normal coordinates v̄: v = A v̄ where A = M(θ*)^{-1/2}.

The loss in normal coordinates: h(v̄) = L(Exp_{θ*}(A v̄))

Taylor expand h around v̄ = 0:
- h(0) = L(θ*) (constant)
- ∂h/∂v̄|_{v̄=0} = Aᵀ ∇_θL(θ*) ≈ 0 (gradient zero at minimum)
- H_{v̄}[h]|_{v̄=0} = Aᵀ H_θ[L](θ*) A (standard Euclidean Hessian conjugated by A)

So: Σ_v̄ = (Aᵀ H_θ[L] A)^{-1} = A^{-1} H_θ[L]^{-1} A^{-ᵀ} = M(θ*)^{1/2} H^{-1} M(θ*)^{1/2}

Converting back to tangential coordinates:
v ~ N(0, A Σ_v̄ Aᵀ) = N(0, M^{-1/2} · M^{1/2} H^{-1} M^{1/2} · M^{-1/2}) = N(0, H^{-1})

**The covariance is identical to the standard Laplace approximation!** This is a crucial observation — it means the starting distribution for RIEM-LA is the SAME as for LA. The only difference is the mapping applied to samples (exponential map vs identity). RIEM-LA takes the same samples as LA and pushes them to better locations via the geodesic.

**Implication for our research:** If we apply a similar normal coordinates trick in activation space (which is what GPLVM does), the covariance structure in tangent space may match standard PCA/covariance estimates. The geometric improvement comes from the mapping back to the data manifold — this is exactly what GPLVM's latent space → data space mapping does.

---

## Synthesis: What This Paper Is Really Saying

The paper's core message, stated plainly:

1. **Neural network loss landscapes are curved.** The Gaussian approximation ignores curvature and generates bad samples.

2. **The fix is to measure distance with a curved ruler** (Riemannian metric) that naturally avoids high-loss regions. The metric M = I + ∇L∇Lᵀ does this efficiently.

3. **Curved sampling = following geodesics.** The exponential map is the correct "move along the curved surface" operation. It requires solving an ODE, which is computationally expensive but tractable for small networks.

4. **The starting Gaussian distribution doesn't change.** You still use the Hessian-based covariance. What changes is how you move from "point in tangent space" to "point in weight space."

5. **Empirically this works.** Samples land in better regions → better predictions → better uncertainty estimates.

---

## Critical Assessment Summary

### What the Paper Gets Right

- **Elegant geometric framing.** The pull-back metric idea is clean, principled, and connects to a rich mathematical tradition.
- **Tractable simplification.** Lemma B.3 is a genuine technical contribution — reducing the geodesic ODE to require only Hessian-vector products rather than full Hessian computation.
- **Consistent empirical improvements.** The results are consistent across multiple datasets and tasks, with proper error bars and multiple random seeds. This is done well.
- **Robustness to prior choice.** The claim that RIEM-LA is less sensitive to the prior precision hyperparameter is convincingly demonstrated and represents a real practical advantage.
- **Honest about limitations.** The paper is relatively forthcoming about the computational cost issue and the failure mode of the linearized version.

### What the Paper Gets Wrong or Omits

- **Missing deep ensembles comparison.** This is the most significant gap. Without this comparison, we don't know if RIEM-LA is competitive with the practical state of the art.
- **Metric choice under-justified.** The gradient outer product metric is presented as "the natural choice" for this loss surface formulation, but it's really the tractable choice. The Fisher information metric (used in follow-up work) is more principled.
- **Scale mismatch between claims and experiments.** The abstract says this works for "Bayesian neural networks" — but all experiments use tiny networks with tanh activations. The practical relevance for modern large networks (which use ReLU/GeLU/SiLU and have many more parameters) is undemonstrated.
- **Mini-batching theory gap.** The paper empirically shows mini-batching works but provides no theoretical justification for why sampling from a mini-batch metric approximates sampling from the full-data metric.
- **The linearized version is often the practical one,** and it sometimes underperforms. The paper buries this finding somewhat.

### How Reliable Are the Results?

**The qualitative story is solid**: curved geometry → better samples → better uncertainty. This is geometrically sound.

**The quantitative improvements** on small models and tasks are well-documented. I would trust the banana/UCI/MNIST results.

**I would NOT extrapolate** to large networks, ReLU networks, or tasks where the training data is large (making ODE evaluation expensive). The paper's experiments are specifically chosen to be where RIEM-LA can run — this is survivorship bias in experimental design.

---

## Relevance to Our Research: Mechanistic Interpretability on Llama 3.1 8B

### What We Can Directly Borrow

**1. The pull-back metric concept (but in activation space, not weight space)**

Our pipeline can define an analogous metric on activation space. Instead of the loss landscape g(θ) = [θ, L(θ)], we could define for each activation r:

g(r) = [r, f(r)]

where f(r) encodes some concept label (e.g., carry value, digit identity). The pull-back metric would then be:

M_activation(r) = I + ∇_r f(r) ∇_r f(r)ᵀ

This would give us a metric that stretches activation space in the direction where the concept label changes most rapidly — exactly what we want for understanding which directions "encode" the concept.

**2. The normal coordinates intuition**

The result that normal coordinates reduce the Taylor expansion to standard form (Σ = H^{-1}) connects to our Phase C finding that concept subspaces have the same covariance structure. In our framework: the "normal coordinates" of the concept manifold may be what PCA is finding. GPLVM then provides the mapping from "flat tangent space" back to "curved concept manifold."

**3. The geodesic as a conceptual tool**

The exponential map in our context would answer: "Starting from a neutral activation (no specific digit encoded), following the geodesic in the direction of 'carry = 3', where do you end up?" This is related to our planned causal intervention experiments — we're essentially asking what the geodesic between "carry = 0" and "carry = k" looks like in activation space.

### Why This Paper Does NOT Directly Solve Our Problem

The paper operates in **weight space** (the space of all possible neural network parameters). We operate in **activation space** (the space of internal representations for specific inputs). These are fundamentally different:

- Weight space: 8 billion dimensions for Llama 3.1 8B
- Activation space: 4096 dimensions at each layer (our actual workspace)

The geometric intuitions transfer, but the formulas need to be re-derived for our setting. The key question — "what is the right metric on 4096-dimensional activation space that encodes which directions carry which concepts?" — is answered differently (via GPLVM, not pull-back from loss).

### The Core Connection to Our Paper's Argument

Our paper's central claim: **The Linear Representation Hypothesis (LRH) finds the "room" (linear subspace) but misses the "furniture arrangement" (non-linear geometry inside the subspace).**

Bergamin et al. make the same argument one level up: **Standard Laplace finds the "room" (Gaussian approximation) but misses the "furniture arrangement" (non-Gaussian shape of the true posterior).**

The solution is the same: equip the space with a curved metric (Riemannian structure) that respects the actual geometry. In our case, the "metric" is what GPLVM learns — the differential geometric structure of the concept manifold.

**Citation strategy:** This paper can be cited as a concrete instance where Riemannian geometry solves a problem that flat-space approximations fail to handle. The argument "Gaussian approximations fail in curved weight spaces, requiring Riemannian methods" is directly analogous to our argument "linear probes fail in curved activation spaces, requiring GPLVM." Cite as: *"The same geometric principle motivates our approach: Bergamin et al. (2023) show that flat-space approximations systematically fail for curved neural network posterior landscapes; we show that flat-space probes (PCA, LDA) systematically fail for curved concept manifolds in activation space."*

---

## Quick Reference Table

| Aspect | Standard Laplace | RIEM-LA (This Paper) | Our GPLVM Approach |
|--------|-----------------|---------------------|-------------------|
| Space | Weight space ℝᴷ | Weight space ℝᴷ (curved) | Activation space ℝ⁴⁰⁹⁶ |
| Metric | Euclidean (Hessian) | M = I + ∇L∇Lᵀ | Learned by GPLVM |
| Distribution | Gaussian in flat space | Gaussian in tangent space | GP posterior |
| Sampling | Direct: θ ~ N(θ*, H⁻¹) | Geodesic ODE from θ* | GPLVM latent → observed |
| Key insight | Curvature of posterior | Loss landscape curvature | Concept manifold curvature |
| Limitation | Ignores curved geometry | Expensive ODE; small networks only | Scalability to many concepts |
| Evaluation | Prediction accuracy | NLL, Brier, ECE | Causal patching, probing |

---

## Reading Queue Generated by This Paper

1. **Arvanitidis et al. (2018), "Latent Space Oddity"** — Pull-back metrics on latent spaces of VAEs; directly relevant to our activation space geometry. **Pass 2.**
2. **Tosi et al. (2014), "Metrics for Probabilistic Geometries"** — Foundation of pull-back metrics for Gaussian processes; closely related to our GPLVM+GP metric tensor approach. **Pass 3.**
3. **Hauberg (2018), "Only Bayes Should Learn a Manifold"** — Already in our pipeline; re-read Section 3 (the central argument about deterministic vs probabilistic manifold learning). **Pass 3 (already partially done).**
4. **Sagun et al. (2016), "Eigenvalues of the Hessian in Deep Learning"** — Empirical evidence that loss landscapes are not bowl-shaped; motivates both this paper and our non-linear geometry findings. **Pass 1.**
5. **Tubular Riemannian Laplace (2025)** — Direct extension that addresses the ODE bottleneck; good to understand where the field went after Bergamin et al. **Pass 2.**
6. **Daxberger et al. (2021), "Laplace Redux"** — The software library used in this paper; practical guide to Laplace approximations. **Pass 1.**

---

*Analysis conducted following the Ramdas three-pass methodology. All empirical claims cross-referenced against the paper's tables and figures. Devil's advocate critiques represent genuine weaknesses, not superficial nitpicking.*

*Document status: Complete. Created March 2026.*