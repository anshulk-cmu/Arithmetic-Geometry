# Analysis: Fourier Circuits in Neural Networks and Transformers
## A Case Study of Modular Arithmetic with Multiple Inputs

**Paper:** Li, Liang, Shi, Song, Zhou (2025)
**arXiv:** 2402.09469v4, March 2025
**Type:** Theory + Empirical (Breakthrough/Theory category)
**Peer-Reviewed:** No (arXiv preprint, not yet published at a venue as of last version)
**Purpose:** Deep understanding. This is core to our research pipeline on Fourier-basis geometric representations in Llama 3.1 8B.

---

# PRE-READING: Who Are We Dealing With?

**Authors:** A team spanning Fuzhou University, University of Hong Kong, University of Wisconsin-Madison, UC Berkeley (Simons Institute), and USC. The key names to notice are Yingyu Liang and Zhao Song, who have extensive publication records in theoretical ML. Zhao Song in particular has dozens of publications in optimization and Fourier analysis. This is a theory-heavy group.

**Venue status:** This is arXiv-only. Four revisions (v4 as of March 2025), which suggests active refinement. No evidence of publication at a top venue yet, though it cites and builds directly on [MEO+24] (Morwani, Edelman, Oncescu, Zhao, Kakade) which appeared at ICLR 2024.

**Red flag check:** No released code repository mentioned in the paper itself, though supplementary material claims code is included. The experiments are on synthetic data (modular addition), which is appropriate for a theory paper. The paper has an extremely long reference list (much of it self-citations from the same group), which is a yellow flag worth watching.

**Connection to our work:** This paper provides the theoretical foundation for WHY neural networks develop Fourier-based representations for modular arithmetic. This is directly relevant to our pipeline's Fourier screening step (Step 2) and connects to Kantamneni & Tegmark's trigonometric addition circuits in pre-trained LLMs. If the margin maximization argument holds, it tells us that Fourier circuits aren't a quirky accident. They're the optimal solution that gradient descent converges to.

---

# PASS 1 — Jigsaw Puzzle (What does the paper do?)

## Q1: What problem is being solved?

The paper studies this question: when you train a neural network on modular addition with k inputs, meaning the task (a₁ + a₂ + ... + aₖ) mod p where p is a prime number, what internal representations does the network learn?

Previous work (Nanda et al. 2023, Morwani et al. 2024) showed that for k=2 (just two inputs), trained networks develop Fourier-based circuits. Each neuron learns to compute cosine functions at specific frequencies, and together these cosines can recover the modular sum. The question this paper tackles is: does the same thing happen when you have MORE inputs? And if so, how many neurons do you need?

**In one sentence:** This paper studies whether networks trained on k-input modular addition (a₁ + ... + aₖ mod p) learn Fourier-based representations, and what the minimum network size is for this to happen.

## Q2: Why is this problem interesting and non-trivial?

Three reasons make this worth studying.

First, modular arithmetic is one of the cleanest testbeds we have for understanding what happens inside neural networks. The ground truth is computable, the structure is mathematical, and any Fourier pattern we find is either there or it isn't. No hand-waving needed.

Second, extending from k=2 to general k is genuinely harder, not just a routine generalization. With k=2, you're multiplying two cosines, which gives you a nice product-to-sum identity. With k inputs, you need to decompose a product of k cosines into sums of cosines, which requires a novel sum-to-product identity (their Lemma E.1). The number of neurons needed grows exponentially: 2^(2k-2) · (p-1), which connects to known hardness results for learning parity functions.

Third, the paper connects to grokking, one of the most puzzling phenomena in deep learning. Grokking is when a model suddenly jumps from terrible generalization to perfect generalization long after it has memorized the training data. Understanding WHY Fourier features emerge (through margin maximization) might explain the dynamics of this sharp transition.

**In one sentence:** Generalizing from 2 to k inputs is non-trivial because it requires new algebraic identities for decomposing k-fold cosine products, the neuron count grows exponentially, and the framework connects margin maximization to grokking dynamics.

## Q3: What is the main claim?

**Theorem 4.1 (Main Result):** For a one-hidden-layer network with polynomial activation of degree k, if the network has at least m ≥ 2^(2k-1) · (p-1)/2 neurons, then the maximum L₂,ₖ₊₁-margin solution has these properties:

1. The maximum margin equals γ* = 2(k!) / [(2k+2)^((k+1)/2) · (p-1) · p^((k-1)/2)]
2. Every neuron uses exactly ONE Fourier frequency ζ, with weights shaped like cosines: uᵢ(aᵢ) = β · cos(θ*ᵢ + 2πζaᵢ/p)
3. Every frequency ζ ∈ {1, ..., (p-1)/2} is used by at least one neuron

**In plain English:** When you train this network on modular addition until it converges (small enough regularization), every neuron specializes in a single "wavelength" and takes the shape of a cosine wave at that wavelength. Collectively, the network covers ALL possible wavelengths. This is the optimal strategy under the margin maximization lens.

**In one sentence:** They prove that the max-margin solution for k-input modular addition consists of neurons that each lock onto a single Fourier frequency (cosine shape), with all frequencies covered, requiring Ω(2^(2k) · p) neurons.

## Relevance to Our Work

Extremely high. This paper:
- Provides theoretical justification for our Fourier screening step (Step 2 of our pipeline)
- Predicts EXACTLY what Fourier structure we should find in modular arithmetic representations
- The single-frequency-per-neuron prediction is testable in Llama 3.1 8B's MLP neurons
- The phase offset constraint (θ*_u1 + ... + θ*_uk = θ*_w) gives a specific geometric signature to test for
- The exponential neuron scaling (2^(2k)) connects to computational hardness of multiplication

**VERDICT:** Proceed to Pass 2 AND Pass 3. This is foundational for understanding why Fourier representations emerge.

---

# PASS 2 — Scuba Dive (How does the paper work?)

## Prerequisite Concepts (Built from Scratch)

Before getting into the technical details, let me build up every concept you need, starting from things a grade-12 student would know.

### What is modular arithmetic?

Regular arithmetic: 7 + 8 = 15.
Modular arithmetic with modulus p=11: 7 + 8 = 15 mod 11 = 4.

Think of it like a clock. A 12-hour clock does arithmetic mod 12. If it's 10 o'clock and 5 hours pass, it's not 15 o'clock. It's 3 o'clock (because 15 mod 12 = 3).

In this paper, p is always a prime number (like 47, 97, 31, etc.). Prime modulus matters because it makes the mathematical group structure clean. Every non-zero element has a multiplicative inverse, which means the Fourier analysis works out nicely.

The task is: given k numbers (a₁, a₂, ..., aₖ), each between 0 and p-1, compute their sum modulo p. For k=2, this is just two-number addition on a clock. For k=4, you're adding four numbers on a clock.

### What is a one-hot encoding?

If you have p possible values (say p=5, so values 0,1,2,3,4), one-hot encoding turns each value into a vector of length p with a single 1:
- 0 → [1, 0, 0, 0, 0]
- 1 → [0, 1, 0, 0, 0]
- 2 → [0, 0, 1, 0, 0]
- etc.

So an input aᵢ becomes a p-dimensional vector xᵢ. The network's weight vector uᵢ ∈ ℝᵖ, when dotted with the one-hot xᵢ, just picks out the aᵢ-th component: u⊤ᵢxᵢ = uᵢ(aᵢ).

This is why the paper can write uᵢ(aᵢ) as both a function (the aᵢ-th component of vector uᵢ) and a dot product. They're the same thing.

### What is the Discrete Fourier Transform (DFT)?

Any function f defined on integers {0, 1, ..., p-1} can be broken down into a sum of "waves" at different frequencies. The DFT does exactly this.

For a vector u ∈ ℝᵖ, its DFT is:

û(j) = Σ_{a=0}^{p-1} u(a) · exp(-2πi · ja/p)

Here, i = √(-1) (the imaginary number), and j is the frequency. The DFT tells you "how much of frequency j is in the signal u."

Think of it like this. You record a musical note. The raw recording is u(a) - the amplitude at each time step. The DFT û(j) tells you the amplitude of each pitch (frequency) in the note. A pure tone has all its energy at one frequency. A complex chord has energy spread across multiple frequencies.

**Plancherel's theorem** says that the total energy is preserved: Σ|u(a)|² = (1/p) · Σ|û(j)|². This is important because the paper's norm constraint on the weights in the original domain translates to a norm constraint in Fourier domain. You can optimize in whichever domain is more convenient.

### What does it mean for a neuron to have a "cosine shape"?

The paper's key finding is that each neuron's weight vector u takes the form:

u(a) = β · cos(θ* + 2πζa/p)

Let me unpack this. Plot u(a) against a (from 0 to p-1). You get a sampled cosine wave. The frequency ζ controls how many complete oscillations the wave makes as a goes from 0 to p-1. The phase θ* shifts the wave left or right. The scalar β controls the amplitude.

Look at Figure 1 in the paper. Each row shows a trained neuron. The left column shows the weight values (red dots) overlaid with the best-fit cosine (blue curve). The right column shows the Fourier power spectrum. Nearly ALL the power is at a single frequency. This is what "single-frequency" or "one-sparse in the Fourier domain" means.

### What is a margin?

In machine learning, the margin measures how confidently a classifier separates different classes. For a data point with correct label y, the margin is:

margin = f(x)[y] - max_{y' ≠ y} f(x)[y']

This is the gap between the score for the correct class and the score for the best wrong class. A positive margin means the classifier gets it right. A LARGER margin means it gets it right with more confidence.

**Why does margin matter?** There's a deep result in optimization theory: when you train certain types of networks with regularization (or with gradient descent for long enough), the solution converges toward the maximum-margin solution. The network doesn't just learn to get the right answer; it learns to get the right answer with as much confidence as possible under a norm constraint.

This is the paper's key leverage point. Instead of analyzing the messy dynamics of gradient descent (which is extremely hard), they analyze the maximum-margin solution (which is a clean optimization problem). Lemma 3.7 from prior work guarantees that the trained network converges to this solution as regularization goes to zero.

### What is a class-weighted margin?

There's a technical problem with the regular margin. The "max" operation in the margin definition makes it non-decomposable. You can't break the network's margin into individual neuron contributions because of that "max" over wrong classes.

The fix is the class-weighted margin. Instead of taking the maximum over wrong classes, you take a WEIGHTED AVERAGE over wrong classes:

g'(θ, x, y) = f(x)[y] - Σ_{y' ≠ y} τ(x,y)[y'] · f(x)[y']

where τ assigns weights to wrong classes that sum to 1.

The paper uses UNIFORM weighting: τ(x,y)[y'] = 1/(p-1) for all wrong classes. This means every wrong answer is penalized equally.

Why does this help? Because expectation (weighted average) is LINEAR, so you CAN decompose the weighted margin into neuron-by-neuron contributions. This reduces the problem from a NETWORK-level optimization to a SINGLE-NEURON optimization, which is much easier to solve.

Under certain conditions (Condition 3.8), the class-weighted margin equals the regular margin. The paper needs to verify this condition holds, which it does.

### What is the polynomial activation function?

Normal neural networks use ReLU or sigmoid activations. This paper uses a polynomial activation of degree k:

ϕ(θᵢ, x) = (u⊤ᵢ,₁x₁ + ... + u⊤ᵢ,ₖxₖ)^k · wᵢ

Why? Two reasons:
1. **Homogeneity:** f(αθ, x) = α^(k+1) · f(θ, x). This property is needed for Lemma 3.7 (the margin convergence result). ReLU is also homogeneous (degree 1), but polynomial of degree k gives homogeneity of degree k+1, which matches the structure of the problem.
2. **Fourier analysis:** The k-th power of a sum of cosines can be decomposed using sum-to-product identities. This is the technical heart of the construction (Lemma E.1). Try doing this with ReLU and you'll see why they chose polynomials.

**CRITICAL NOTE (Devil's advocate):** This is a major limitation. Real networks DON'T use polynomial activations. ReLU, GELU, SiLU are standard. The paper's theory applies to a stylized network, not to practical architectures. The experiments partially address this gap by showing similar patterns in Transformers with softmax attention, but the theory itself does not cover standard activations.

### What is the L₂,ₖ₊₁ norm?

For a network with m neurons, each with parameter vector θᵢ:

‖θ‖₂,ₖ₊₁ = (Σᵢ ‖θᵢ‖₂^(k+1))^(1/(k+1))

This is a "mixed norm." The inner norm (L₂) measures the size of each neuron's parameters. The outer norm (L_{k+1}) combines the neurons. The exponent k+1 matches the homogeneity degree.

Why this particular norm? Because for a (k+1)-homogeneous network, the right regularizer to induce margin maximization is ‖θ‖₂,ₖ₊₁. Using a different norm would give a different "max-margin" solution, possibly one WITHOUT Fourier structure. The choice of norm is not innocent; it partially determines the conclusion.

---

## The Proof Strategy (High-Level Map)

The proof has four main moves. Let me trace each one.

### Move 1: Reduce network-level optimization to single-neuron optimization

**What they do:** Using the class-weighted margin trick and the decomposability of linear expectations, they show that the network's optimal margin can be achieved by combining individually optimal neurons. Lemma C.1 from [MEO+24] provides this reduction.

**The logic:** If each neuron independently maximizes its class-weighted margin, and you can combine neurons with appropriate scaling, then the full network achieves the maximum margin. This only works because (a) the weighted margin is decomposable over neurons (linearity of expectation), and (b) Condition 3.8 ensures the weighted margin equals the true margin.

**Devil's advocate:** This reduction requires uniform class weighting (τ = 1/(p-1)). Other class weightings might lead to different solutions. The paper doesn't explore whether this is the "natural" weighting that gradient descent converges to. In practice, gradient descent doesn't explicitly optimize for any particular class weighting. The assumption that uniform weighting is the right proxy needs justification beyond "it makes the math work."

### Move 2: Solve the single-neuron optimization in Fourier domain

**What they do (Lemma D.8):** They take the single-neuron class-weighted margin objective and transform it to the Fourier domain using the DFT.

The key calculation: after zeroing out terms that vanish due to symmetry (many cross-terms like u₁(a₁)³w(...) average to zero because w has zero mean), the objective reduces to:

maximize: [k! / ((p-1)p^k)] · Σ_{j≠0} ŵ(-j) · Π_{i=1}^k ûᵢ(j)

subject to: ‖û₁‖² + ... + ‖ûₖ‖² + ‖ŵ‖² ≤ p

This is a product of Fourier magnitudes. Using AM-GM inequality (the arithmetic mean is always ≥ the geometric mean), you can show this product is maximized when:
- All Fourier mass is at a SINGLE frequency ζ (one-sparse)
- The magnitudes are EQUAL across all uᵢ and w at that frequency
- The phases satisfy θ_u1 + ... + θ_uk = θ_w

**Devil's advocate:** The AM-GM step is clean but creates a very specific optimum. In practice, trained networks don't achieve exact AM-GM equality. The Fourier mass won't be PERFECTLY concentrated at one frequency. Figure 2(c) shows the max normalized power is "almost 1" for trained neurons, not exactly 1. The gap between "approximately one-sparse" (what training gives you) and "exactly one-sparse" (what the theory proves) is swept under the rug.

### Move 3: Construct the full network from single-neuron solutions

**What they do (Lemma E.3):** They show how to build a network that computes the correct function using neurons from the solution set Ω'*_q.

The key trick is the sum-to-product identity (Lemma E.1):

2^k · k! · Π_{i=1}^k aᵢ = Σ_{c ∈ {-1,+1}^k} (-1)^((k - Σcᵢ)/2) · (Σ_{j=1}^k cⱼaⱼ)^k

This identity lets you express a PRODUCT of k values as a signed sum of k-th POWERS of linear combinations. Since each neuron computes (linear combination)^k · w, you can combine 2^(k-1) neurons (with different sign patterns) to compute a product.

Then, using the cosine expansion:

cos(Σaᵢ - c) = Σ_{b ∈ {0,1}^(k+1)} [product of cos and sin terms with specific sign rules]

This expansion has 2^k terms, each requiring 2^(k-1) neurons. So each frequency ζ needs 2^(2k-1) neurons. Across (p-1)/2 frequencies, the total is 2^(2k-1) · (p-1)/2.

**Devil's advocate:** The construction is valid but extremely wasteful. The 2^(2k-1) neurons per frequency arise from a worst-case combinatorial expansion. In practice, trained networks use FAR fewer neurons. For k=4, p=47, the theory says you need at least 2^7 · 23 = 2944 neurons. They use exactly this number in experiments. But would a network with fewer neurons still learn Fourier features? Almost certainly yes, though possibly not covering all frequencies. The lower bound on neuron count is probably loose.

### Move 4: Prove all frequencies are covered

**What they do (Lemma F.2):** They show that in ANY max-margin solution (not just the one they constructed), every frequency ζ ∈ {1, ..., (p-1)/2} must be used by at least one neuron.

The argument is clever. The network function can be decomposed as f = f₁ + f₂, where f₁ depends only on inputs (not on the output label c) and f₂ = λ · 1[a₁ + ... + aₖ = c] (the indicator function times the margin). The DFT of f₂ is nonzero whenever j₁ = ... = jₖ = -jₖ₊₁ ≠ 0. Since f₁'s DFT is zero for jₖ₊₁ ≠ 0, the total DFT must be positive at these points. But the DFT of each neuron is nonzero only at its own frequency ζ, so every frequency must have at least one neuron.

**Devil's advocate:** This argument proves that all frequencies are covered in the MAX-MARGIN solution. It says nothing about what happens during training. A network might converge to a non-max-margin solution that only uses SOME frequencies and still generalizes perfectly. The connection between max-margin and actual training relies on Lemma 3.7, which requires λ → 0 (regularization goes to zero). In practice, λ is finite and nonzero, so the trained network is an approximation of the max-margin solution, not the exact thing.

---

## What Exactly is the Cosine Identity That Makes This Work?

Let me walk through the key identity for k=3 so you can see the mechanics.

We need to show that cos(a₁ + a₂ + a₃ - c) (which is the Fourier kernel for checking if a₁ + a₂ + a₃ ≡ c mod p) can be computed by neurons of the form (u₁(a₁) + u₂(a₂) + u₃(a₃))³ · w(c).

Step 1: Expand using angle addition:

cos(a₁ + a₂ + a₃ - c) = [8 terms involving products of cos and sin of individual variables]

For example, one term is cos(a₁)cos(a₂)cos(a₃)cos(c), another is -sin(a₁)sin(a₂)cos(a₃)cos(c), and so on.

Step 2: Each such product (e.g., cos(a₁)cos(a₂)cos(a₃)) can be expressed using the sum-to-product identity. For k=3:

2³ · 3! · d₁d₂d₃ = (d₁+d₂+d₃)³ - (d₁+d₂-d₃)³ - (d₁-d₂+d₃)³ - (-d₁+d₂+d₃)³ + (d₁-d₂-d₃)³ + (-d₁+d₂-d₃)³ + (-d₁-d₂+d₃)³ - (-d₁-d₂-d₃)³

Each of those 8 terms is a cube of a sum, which is exactly what one neuron computes (since the activation is a k-th power). By choosing the phase offsets θ* appropriately, you can flip cos to -cos or cos to sin, giving you the ± patterns needed.

Step 3: The w(c) factor comes from the output weight vector, which is also a cosine at the same frequency ζ.

So each of the 8 terms in the cosine expansion needs 4 neurons (from the 8 terms in the sum-to-product identity, but symmetry halves this to 4). That's 8 × 4 = 32 neurons per frequency. For (p-1)/2 frequencies, you need 16(p-1) neurons total. For general k, this becomes 2^(2k-1) · (p-1)/2.

---

## The Experiments

### Experiment 1: One-Hidden-Layer Neural Networks (Section 5.1)

**Setup:** Train a two-layer network (input → hidden → output) with m = 2^(2k-2) · (p-1) neurons on k-sum mod p.

For k=4, p=47: m = 2^6 · 46 = 2944 neurons.

**What they measure:**
- (a) The shape of each neuron's weight vector → should look like a cosine
- (b) The Fourier power spectrum of each neuron → should be one-sparse (all power at one frequency)
- (c) The frequency distribution across neurons → should cover all frequencies roughly uniformly

**Results (Figures 1 and 2):**
- Neurons indeed show clear cosine shapes (Figure 1, left)
- Fourier spectra are strongly one-sparse (Figure 2c)
- All frequencies from 1 to (p-1)/2 are covered with roughly uniform distribution (Figure 2a)

**Devil's advocate on the experiments:**

1. **They chose m exactly equal to the theoretical minimum.** This is a strong experimental choice. What happens with fewer neurons? Do you still get Fourier features but with some frequencies missing? What about many more neurons? The paper doesn't explore this.

2. **They use AdamW, not plain SGD.** The theory is about the regularized loss minimizer as λ→0. AdamW has a different implicit bias than SGD. The theory doesn't cover AdamW at all. The fact that it still works is nice empirically but theoretically unexplained.

3. **They use polynomial activations.** This matches the theory, so of course it works. The real question is whether standard activations (ReLU, GELU) give the same result. They partially address this with the Transformer experiments, but those use softmax, not polynomial activations, so the theory doesn't apply there either.

4. **The initialization and training dynamics are completely ignored.** The theory predicts what the FINAL solution looks like but says nothing about whether gradient descent actually FINDS this solution. Figure 2b shows the initial distribution is spread across all frequencies with no concentration. The transition from random initialization to concentrated one-sparse features is the interesting part, and the paper doesn't explain it.

### Experiment 2: One-Layer Transformers (Section 5.2)

**Setup:** A one-layer Transformer with 160 attention heads, hidden dimension 128, trained on k-sum mod p.

**What they find:** The attention matrices W^KQ (the product of key and query weight matrices) show 2D cosine patterns (Figure 3). The Fourier power spectrum shows energy concentrated at specific 2D frequency pairs.

**Devil's advocate:**

1. **The theory covers one-hidden-layer networks, NOT Transformers.** The Transformer results are purely empirical observations. The paper says "we observe similar computational mechanisms" but provides zero theoretical justification for why Transformers should develop Fourier features.

2. **The connection between attention matrix cosine patterns and the network-level Fourier circuits is unclear.** Having cosine-shaped W^KQ tells you the attention PATTERN is periodic, but it doesn't tell you what COMPUTATION the Transformer is performing. The attention matrix is just one component. The embedding E, the value matrix W^V, the projection W^P, and the MLP (if present) all matter too.

3. **160 attention heads is a lot for such a small problem.** With p=31 and k=4, the dataset has 31⁴ ≈ 924K points but only 31 possible output classes. The Transformer is massively overparameterized for this task.

### Experiment 3: Grokking Under Different k (Section 5.3)

**Setup:** Train Transformers on k=2,3,4,5 with different primes (p=97,31,11,5) chosen to keep dataset sizes roughly equal.

**Finding:** As k increases, grokking weakens. The gap between training convergence and test convergence shrinks. With k=5, the model generalizes almost as fast as it memorizes.

**Devil's advocate:**

1. **Confounding variable: p differs across conditions.** For k=2 they use p=97, for k=5 they use p=5. A modular group of size 5 is trivially small. With only 5 possible values per input and 5 possible outputs, the task is almost trivially learnable regardless of k. The paper claims to control for dataset size, but it doesn't control for the COMPLEXITY of the modular structure.

2. **The "grokking weakens" claim is not cleanly supported by theory.** The paper's theory gives a neuron count requirement of 2^(2k-1) · (p-1)/2, which INCREASES with k. But in the grokking experiments, p DECREASES with k. The interplay between k increasing and p decreasing means the theoretical neuron requirement doesn't monotonically change. The intuitive argument in Section 5.3 about NTK vs. feature learning regimes is hand-wavy.

3. **No error bars on grokking curves.** Actually, the paper says they ran 3 seeds and reported mean and variance in Figure 4, which is decent but minimal.

---

## Connection to Nanda et al. (2023) and Transformer Circuits

The paper by Nanda, Chan, Lieberum, Smith, and Steinhardt ("Progress measures for grokking via mechanistic interpretability") from transformer-circuits.pub (Anthropic's interpretability research thread) is the empirical precursor that this paper aims to explain theoretically.

Nanda et al. found that Transformers trained on modular addition (k=2) develop circuits that:
1. Embed inputs as points on a circle using Fourier components
2. Compute rotations in this circular representation
3. Use the identity: (a₁ + a₂) mod p = argmax_c cos(2πζ(a₁ + a₂ - c)/p)

The Li et al. paper provides theoretical justification for WHY this happens. Nanda et al. showed WHAT the circuit looks like; Li et al. prove that this structure is the maximum-margin solution.

However, there's a subtlety: Nanda et al. studied full Transformers with embedding layers, attention, MLPs, and unembedding. Li et al. prove results about stylized one-hidden-layer networks with polynomial activations. The gap between these two settings is substantial.

The Anthropic "When Models Manipulate Manifolds" paper (Gurnee et al., 2025, also from transformer-circuits.pub) provides another data point. That paper studies character counting in Claude 3 Haiku and finds:
- Circular/helical manifolds for count representations
- Fourier-like ringing in cosine similarity structure
- An analytical explanation: truncated circulant matrix eigendecomposition produces Fourier modes

Together, these three papers paint a consistent picture: Fourier representations emerge spontaneously for cyclic/modular tasks across different architectures and scales. But the THEORETICAL explanation only covers the simplest case (one-hidden-layer, polynomial activation).

---

## Connection to Our Pipeline

### For Step 2 (Fourier Screening)

This paper validates our Fourier screening approach. If modular arithmetic representations converge to single-frequency cosines, then a Fourier power spectrum analysis of Llama 3.1 8B's neuron activations should reveal Fourier structure when the model processes arithmetic tasks. Specifically:

- **What to test:** Extract MLP neuron activations while Llama processes multiplication problems. Compute DFT of the activation-as-a-function-of-input-digit-value. Check if the spectrum is one-sparse.
- **Expected signature:** High normalized power at a single frequency, matching the paper's Figure 2(c).
- **Caveat:** Llama uses GELU activation, not polynomial. The theory doesn't apply. But the empirical evidence from both this paper and Kantamneni & Tegmark suggests Fourier features emerge regardless of activation function.

### For Step 1 (Concept Subspace Discovery)

The phase offset constraint (θ*_u1 + ... + θ*_uk = θ*_w) is a specific geometric signature. If we find neurons in Llama 3.1 8B that have Fourier structure, we can check whether their phase offsets satisfy additive constraints. This would be evidence that the same margin-maximization mechanism drives feature formation in large pre-trained models.

### For the Broader Pipeline

The exponential neuron scaling (2^(2k)) connects to our concern about multiplication being harder than addition. For k=2 (addition), you need O(p) neurons. For k=4 (four-number addition), you need O(16p) neurons. For actual multiplication (which involves products, not just sums), the required representation capacity is likely much larger, which may explain why Llama 3.1 8B struggles with multi-digit multiplication.

---

## Devil's Advocate: Three Weakest Points

### Weakness 1: The Stylized Architecture Gap

The theory applies to one-hidden-layer networks with polynomial activations. Nobody uses this architecture. The experiments on Transformers show similar patterns but have no theoretical backing. The entire theoretical contribution is about a network architecture that doesn't exist in practice.

The paper acknowledges this (sort of) in the Limitations section, but downplays it. They argue that "beginning with a simplistic model setup and achieving a thorough and theoretical understanding... serves as a valuable starting point." That's fair, but it means the THEORETICAL claims should not be interpreted as explaining Transformer behavior. They explain stylized-network behavior, full stop.

### Weakness 2: The Neuron Count Lower Bound Is Probably Very Loose

The paper proves you need at least 2^(2k-1) · (p-1)/2 neurons. For k=4, p=47, that's 2944 neurons. But the construction that achieves this bound is extremely wasteful. It expands cos(sum) into 2^k terms, each requiring 2^(k-1) neurons for the sum-to-product identity. There are likely much more efficient constructions.

In practice, the relevant question for Llama 3.1 8B is: does a 4096-dimensional activation space have enough "effective neurons" (MLP hidden units, attention heads, etc.) to represent Fourier features for the arithmetic it needs to do? The paper's lower bound gives us SOME guidance but is probably orders of magnitude too pessimistic.

### Weakness 3: The Uniform Class Weighting Assumption Is Unjustified

The entire analysis hinges on using uniform class weighting τ(x,y)[y'] = 1/(p-1). This is the assumption that makes the Fourier analysis work out cleanly. But there's no argument for WHY gradient descent should converge to a solution that looks like the uniform-class-weighted max-margin solution.

Different class weightings could lead to different optimal representations. The paper's answer to "why Fourier circuits?" is really "IF you optimize for uniform-class-weighted margin, THEN Fourier circuits." The "if" is doing a lot of heavy lifting.

---

# PASS 3 — The Swamp (Deep Critical Analysis)

## Reconstructing the Core Argument

Let me trace the full proof chain to identify where the real difficulties and insights are.

### The Proof Architecture

```
Lemma 3.7: Training converges to max-margin (from prior work)
    ↓
Condition 3.8: Class-weighted margin = true margin (need to verify)
    ↓
Lemma D.7/D.8: Single-neuron max-weighted-margin solution is Fourier (THE KEY)
    ↓
Lemma E.1: Sum-to-product identity (new algebraic tool)
    ↓
Lemma E.3: Construct network from single-neuron solutions
    ↓
Lemma C.1/C.2: Combined network achieves max margin (from prior work)
    ↓
Lemma F.2: All frequencies must be covered (uniqueness-type result)
    ↓
Theorem 4.1: Main result
```

The original content of THIS paper (vs. what comes from [MEO+24]) is:
- Lemma D.7/D.8 (generalization of single-neuron analysis to k inputs)
- Lemma E.1 (sum-to-product identity for general k)
- Lemma E.3 (construction for general k)
- Lemma F.2 (frequency coverage for general k)

### Deep Dive: The Single-Neuron Optimization (Lemma D.8)

This is the mathematical heart of the paper. Let me trace it carefully.

**Starting point:** Maximize the expected class-weighted margin of a single neuron:

η(0) - E_{δ≠0}[η(δ)]

where η(δ) = E_{a₁,...,aₖ}[(u₁(a₁) + ... + uₖ(aₖ))^k · w(a₁+...+aₖ - δ)]

**Step 1: Kill the cross-terms.**

When you expand (u₁ + ... + uₖ)^k using the multinomial theorem, you get many terms. Most of them vanish when you take the expectation over aᵢ and the difference η(0) - E_δ[η(δ)].

Why do they vanish? Because w has zero mean (E_a[w(a)] = 0 since ŵ(0) = 0 in the max-margin solution). Any term where some aᵢ appears only in w(a₁+...+aₖ-δ) and not in any uⱼ factor will average to zero.

The only surviving term is the fully-mixed term: k! · u₁(a₁)·u₂(a₂)·...·uₖ(aₖ)·w(a₁+...+aₖ-δ). This is because every aᵢ must appear in at least one uⱼ factor (otherwise that variable averages out through w), and the only way all k variables appear when you have k factors of u's is the fully-mixed term.

**Step 2: Transform to Fourier domain.**

Using the DFT, the surviving term becomes:

(k! / ((p-1)p^k)) · Σ_{j≠0} ŵ(-j) · Π_{i=1}^k ûᵢ(j)

The key simplification: the sum is only over NON-ZERO frequencies. The j=0 terms cancel in the η(0) - E_δ[η(δ)] difference (because E_δ[ρ^{jδ}] = 0 for j≠0 but = 1 for j=0).

**Step 3: Use symmetry of real-valued signals.**

Since u₁,...,uₖ,w are real-valued, their DFTs satisfy û(-j) = conjugate(û(j)). This means we can pair up frequencies j and -j:

Σ_{j≠0} = Σ_{j=1}^{(p-1)/2} [term at j + term at -j]
         = Σ_{j=1}^{(p-1)/2} 2·|ŵ(j)|·Π|ûᵢ(j)|·cos(Σθ_{ui}(j) - θ_w(j))

where θ_{ui}(j) is the phase of ûᵢ(j).

**Step 4: Optimize the cosine.**

The cosine is maximized (equals 1) when Σθ_{ui}(j) = θ_w(j). This is the PHASE CONSTRAINT that appears in Theorem 4.1.

**Step 5: Apply AM-GM.**

Now we need to maximize |ŵ(j)| · Π|ûᵢ(j)| subject to Σ|ûᵢ|² + |ŵ|² ≤ p (Plancherel).

AM-GM says the product of (k+1) non-negative numbers with fixed sum of squares is maximized when all numbers are equal. So |û₁(j)| = ... = |ûₖ(j)| = |ŵ(j)| for the optimal j.

Furthermore, the product Π|ûᵢ(j)| · |ŵ(j)| ≤ ((1/(k+1)) · z(j))^((k+1)/2) where z(j) = Σ|ûᵢ(j)|² + |ŵ(j)|².

To maximize Σ_j z(j)^((k+1)/2) subject to Σ_j z(j) ≤ p/2, since (k+1)/2 > 1, the sum is maximized by concentrating all mass at ONE frequency (convexity argument). This gives the ONE-SPARSE result.

**My assessment:** This is a clean, well-executed calculation. The key insight is that the Fourier domain makes the product structure transparent, and AM-GM + convexity force concentration at a single frequency. The proof technique is not groundbreaking (AM-GM and convexity are standard tools), but the application is nice.

**Where I see potential weakness:** The convexity argument that concentrates mass at one frequency relies on the exponent (k+1)/2 > 1. For k=1 (which would mean linear activation), (k+1)/2 = 1 and there's no concentration. This actually makes sense because a linear network can't distinguish frequencies. But it means the result is SPECIFIC to nonlinear activations. What about ReLU, where the "effective degree" is somewhere between 1 and 2? The theory has nothing to say.

### The Sum-to-Product Identity (Lemma E.1)

This is the paper's algebraic contribution. The identity:

2^k · k! · Π_{i=1}^k aᵢ = Σ_{c∈{-1,+1}^k} (-1)^((k-Σcᵢ)/2) · (Σ cⱼaⱼ)^k

**Proof check:** The proof for general k is elegant. Set a₁ = 0. Then the RHS has symmetric cancellations (flipping c₁ pairs up terms that cancel). So a₁ is a factor of RHS. By symmetry, every aᵢ is a factor. Since RHS is degree k and has all aᵢ as factors, RHS = α · Πaᵢ. Setting all aᵢ = 1 gives α = 2^k · k!.

This is correct and slick. The identity is essentially the polarization identity for multilinear forms, generalized to the k-th power setting. It's not new mathematics (it's related to classical results in combinatorics), but the specific form for the cosine decomposition application is useful.

### The Frequency Coverage Argument (Lemma F.2)

This is the subtlest part of the proof. The argument that ALL frequencies must be used relies on decomposing f = f₁ + f₂ where f₂ = λ · 1[Σaᵢ = c].

**The key insight:** The indicator function 1[Σaᵢ = c] has a flat DFT for j₁ = ... = jₖ = -jₖ₊₁ ≠ 0. Since f₁ contributes nothing at these frequencies (its DFT is zero for jₖ₊₁ ≠ 0), the full network's DFT must be positive there. But each neuron's DFT only covers its own frequency ζ. So every frequency must have at least one neuron.

**My concern:** This argument assumes the network achieves a POSITIVE margin (λ > 0). If the network could achieve zero margin on some data points, the decomposition doesn't force positive DFT values at all frequencies. The argument works because the max-margin solution has CONSTANT margin across all data points (a consequence of the uniform structure of modular addition), but this uniformity is special to this particular task. For a non-uniform task, the argument might not go through.

---

## Boundary Exploration: What Happens If We Change Things?

### What if p isn't prime?

The paper assumes p is prime. This ensures Z_p is a field (every nonzero element has a multiplicative inverse) and the DFT is cleanly defined with p distinct frequencies. For composite p, the DFT still works but the structure of the Fourier domain changes. There might be "degenerate" frequencies where the analysis breaks down. The paper doesn't address this at all.

For our pipeline applied to Llama 3.1 8B doing multiplication, the relevant modulus is 10 (for digit extraction via mod 10). 10 = 2 × 5 is NOT prime. Does the Fourier analysis still predict cosine features at the relevant frequencies? Probably yes (the DFT still works), but the theorem doesn't cover it.

### What if we use ReLU instead of polynomial activation?

This is the big open question. ReLU networks have different implicit bias. The max-margin framework still applies (ReLU networks are 1-homogeneous per layer), but the margin norm changes and the decomposition properties of single neurons are completely different.

Empirically, Nanda et al. (2023) find Fourier features in ReLU/GELU Transformers. This suggests the phenomenon is robust to activation function choice. But the theoretical explanation breaks down because:
1. ReLU^k doesn't factor nicely like polynomial^k
2. The sum-to-product identity relies on (sum)^k structure
3. The Fourier domain optimization becomes more complex

### What about multi-layer networks?

The paper analyzes one-hidden-layer networks. Real Transformers are multi-layer. In a multi-layer network:
1. Different layers might specialize in different frequencies
2. The margin analysis becomes much harder (composing homogeneous functions)
3. Intermediate representations might use NON-Fourier bases that get converted to Fourier form later

For Llama 3.1 8B with 32 layers, the single-layer theory is a very rough guide at best.

### What is the relationship between Fourier circuits and grokking?

The paper shows (experimentally, not theoretically) that grokking weakens as k increases. They interpret this through the lens of NTK-to-feature-learning transition: with larger k, the network needs more neurons to achieve the max-margin solution, so it exits the NTK regime sooner and enters the feature-learning regime earlier.

But this interpretation is speculative. The actual grokking dynamics involve:
1. Phase 1: Memorization (overfit training data)
2. Phase 2: Circuit formation (develop Fourier features, overwrite memorized solution)
3. Phase 3: Generalization (Fourier circuit generalizes to test data)

The paper's theory describes the END STATE (Phase 3) but not the DYNAMICS (how you get there). The grokking observations are suggestive but not explained by the theorem.

---

## Research Ideas Generated

1. **Test single-frequency-per-neuron in Llama 3.1 8B:** Extract MLP activations while processing modular addition. Compute DFT of activation-as-function-of-input-value. Check if Fourier spectra are concentrated.

2. **Test phase offset constraints:** If Fourier features exist in Llama 3.1 8B, check whether phase offsets across layers satisfy additive constraints (θ_u1 + ... + θ_uk = θ_w). This would be strong evidence for margin-maximization-driven feature emergence in production models.

3. **Composite modulus extension:** Extend the theory to p = 10 (composite). Does the max-margin solution still use Fourier features? If so, what's different about the frequency coverage?

4. **Relate neuron count to model failure:** If multiplication requires 2^(2k) neurons per frequency, and Llama 3.1 8B has finite capacity, predict the maximum digit count where multiplication should succeed. Compare with actual performance.

5. **GP-based geometric analysis of Fourier manifold:** The single-frequency-per-neuron result means the activation manifold for each neuron is a circle (cos function traces a circle in 2D Fourier space). Apply Hauberg's probabilistic Riemannian metric to characterize this manifold with uncertainty quantification. This is a concrete test case for our Paper 1 framework.

---

## Techniques I Can Borrow

1. **Fourier domain margin analysis:** The trick of converting margin optimization to Fourier domain, where AM-GM forces one-sparsity, could apply to analyzing representations of ANY periodic/cyclic concept in Llama 3.1 8B.

2. **The indicator function DFT trick (Lemma F.2):** The argument that the correct-answer indicator forces ALL frequencies to be covered is transferable. For our pipeline, this means: if we find a network perfectly solving modular addition, we know it MUST have all frequencies represented somewhere.

3. **Sum-to-product identity for probing:** The identity from Lemma E.1 tells us exactly which linear combinations of neuron activations should give us product terms. This provides specific probe directions for our concept subspace analysis.

---

## Final Assessment

### What This Paper Does Well
- Clean, complete theoretical analysis for the specific setting considered
- Good connection between theory and experiments
- The Fourier domain analysis technique is elegant
- Honest about the gap between stylized networks and Transformers

### What This Paper Does Poorly
- Massive gap between theory (polynomial activations, one layer) and practice (ReLU/GELU, multi-layer)
- Uniform class weighting is assumed without justification
- Neuron count lower bound is almost certainly very loose
- Grokking analysis is hand-wavy speculation, not rigorous
- Extensive self-citation inflates the reference list
- The Transformer experiments are described too briefly given how important they are

### Overall Grade
Solid theory paper for the specific setting. The result is correct and the proof is clean. But the practical relevance hinges entirely on the empirical observation that Fourier features appear in architectures the theory doesn't cover. For our pipeline, the most valuable contribution is theoretical validation that Fourier features are OPTIMAL (not just one possible solution), which justifies our Fourier screening step. The weakest part for us is that the theory says nothing about Llama 3.1 8B's architecture, depth, activation function, or training procedure. We should treat the theoretical results as strong motivation and use the experimental methodology (DFT of weight vectors, normalized power spectra) as a template for our own analysis.