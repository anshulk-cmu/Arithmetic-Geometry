# Deep Analysis: "Why Can't Transformers Learn Multiplication?"
## Bai, Pres, Deng, Tan, Shieber, Viégas, Wattenberg, Lee (2025)

**arXiv: 2510.00184v1 | September 30, 2025**

---

# PHASE -1: Paper Classification

**Type:** Empirical / Methods with mechanistic interpretability (reverse-engineering)

This is NOT a pure theory paper. There are no formal theorems or proofs. It is an empirical mechanistic interpretability paper that reverse-engineers a trained model, discovers geometric structure, and then uses those insights to propose a fix. The reading strategy should focus on: experimental methodology, whether the baselines are fair, whether the mechanistic claims are actually well-supported by the evidence, the quality of the ablation (or lack thereof), and reproducibility.

---

# PHASE 0: Pre-Reading Context

**Authors:**
- Xiaoyan Bai (University of Chicago) and Itamar Pres (MIT) — equal contribution, likely graduate students
- Yuntian Deng (University of Waterloo) — the creator of the ICoT method being studied here
- Chenhao Tan (University of Chicago)
- Stuart Shieber (Harvard) — senior NLP/formal languages researcher
- Fernanda Viégas and Martin Wattenberg (Harvard / Google DeepMind) — visualization researchers, known for interpretability-adjacent work
- Andrew Lee (Harvard) — contact author

**Venue:** arXiv preprint, not yet peer-reviewed at time of writing. The author list is strong (Harvard, MIT, Chicago, DeepMind affiliations), but this has not gone through formal peer review.

**Code:** Released at https://github.com/ajyl/icot — good reproducibility signal.

**Why we are reading this:** This paper is directly relevant to our research on geometric representations of arithmetic in transformers. It studies 4×4 digit multiplication, discovers Fourier-basis digit representations and Minkowski sum geometry in attention heads, and identifies why standard training fails. This connects to our pipeline work on concept subspaces in Llama 3.1 8B.

**Critical note on scope:** This paper studies tiny 2-layer, 4-head models trained from scratch. Our target is Llama 3.1 8B (32 layers, 4096 dimensions). The gap between these is enormous. Everything found here needs to be evaluated for whether it would transfer to a production-scale model.

---

# PASS 1 — The Jigsaw Puzzle (What does the paper do?)

## Q1: What is the problem?

The paper asks: why do transformers fail at multi-digit multiplication, even when fine-tuned specifically on this task?

This is a real and well-documented problem. Models like GPT-4 and Llama-3.2 90B still fail at 4×4 digit multiplication. Even explicitly fine-tuning on the task doesn't fix it — Yang et al. (2023) showed a 2B parameter model still plateaus at 95% accuracy. So this isn't just a "small model" problem.

**In one sentence:** The paper studies why standard fine-tuning (SFT) fails at 4×4 digit multiplication by reverse-engineering a model that succeeds via implicit chain-of-thought (ICoT).

## Q2: Why is the problem interesting and non-trivial?

Multiplication is interesting because it requires *long-range dependencies*. To compute the k-th digit of the product, you need to combine partial products from potentially all pairs of input digits, plus propagate carry information from all previous digits. This means computing, say, the 4th output digit requires information from input digits that are 10+ tokens away.

The paper argues this is non-trivial because gradient descent with an autoregressive loss doesn't encourage the model to learn the right long-range dependencies. The model gets stuck in a local optimum where it can predict the easy digits (first, last) but not the middle ones.

**In one sentence:** This is non-trivial because multi-digit multiplication requires accumulating information across many distant token pairs (long-range dependencies), and standard gradient descent fails to learn these dependencies — the model converges to a local optimum.

## Q3: What is the main claim?

The paper makes three interlocking claims:

1. **Mechanism claim:** The ICoT model (which succeeds) encodes long-range dependencies by constructing a binary-tree-like directed acyclic graph through attention patterns, "caching" pairwise partial products at earlier timesteps and "retrieving" them later.

2. **Geometry claim:** Attention heads compute partial products via Minkowski sums of digit embeddings, and digits are represented using Fourier bases, forming a pentagonal prism structure.

3. **Failure diagnosis claim:** Standard fine-tuning fails because the model never learns the long-range dependencies needed for middle digits. An auxiliary loss that supervises the "running sum" (ĉ_k) through a linear probe fixes this, achieving 99% accuracy.

**In one sentence:** The ICoT model succeeds at multiplication by constructing attention trees for caching partial products and using Fourier-based Minkowski sum geometry, while SFT fails because gradient descent with autoregressive loss gets stuck in a local optimum lacking long-range dependencies — fixable with an auxiliary running-sum loss.

## Relevance to Our Research

Extremely high. This paper:
- Studies exactly the arithmetic domain we're targeting (multiplication)
- Discovers Fourier digit representations — which connects to Li et al.'s Fourier circuits and Kantamneni & Tegmark's trigonometric addition work
- Finds geometric structure (Minkowski sums, pentagonal prisms) in activation space — directly relevant to our concept subspace pipeline
- The carry propagation problem is exactly the kind of "long-range dependency" our pipeline needs to detect
- The pentagonal prism structure is a concrete example of the kind of geometric concept representation we aim to characterize with our GP-based framework

**VERDICT:** Proceed to Pass 2 and Pass 3. This is core to our research.

---

# PASS 2 — Scuba Dive (How does the paper work?)

## Prerequisite Concepts Explained

Before digging into the technical details, let me build up the concepts you need.

### What is multi-digit multiplication, really?

When you multiply two 4-digit numbers, say a = a₃a₂a₁a₀ × b = b₃b₂b₁b₀, you need to compute all the "partial products" — every pair aᵢ × bⱼ. For a 4×4 multiplication, that's 16 pairwise products.

The key insight is how these partial products contribute to the answer. The product has 8 digits c₇c₆...c₀. To compute the k-th digit c_k, you need all partial products where i + j = k (these contribute directly), PLUS all the carries from computing digits c₀ through c_{k-1}.

For example, to compute c₃ (the 4th output digit), you need:
- Direct partial products: a₃b₀ + a₂b₁ + a₁b₂ + a₀b₃
- Plus the carry from c₂, which depends on c₁, which depends on c₀

This creates a chain of dependencies. Computing c₅ requires knowing what happened at c₀, c₁, c₂, c₃, and c₄. That's the "long-range dependency" — output digit c₅ depends on input digits that were processed many tokens ago.

The paper introduces an intermediate quantity ĉ_k = s_k + r_{k-1}, where s_k is the sum of partial products for position k, and r_{k-1} is the carry from the previous position. This ĉ_k captures everything you need: c_k = ĉ_k mod 10 (the digit) and r_k = floor(ĉ_k / 10) (the carry to propagate).

### What is ICoT (Implicit Chain of Thought)?

Normally, chain-of-thought (CoT) means writing out intermediate reasoning steps explicitly — like showing your work on a math test. The model generates tokens for each intermediate step before producing the final answer.

ICoT is a training trick: you START by giving the model full chain-of-thought during training, but then gradually REMOVE the CoT tokens epoch by epoch, forcing the model to "internalize" the reasoning into its hidden states rather than outputting it as tokens.

Concretely, at epoch 1 the training data looks like:
```
1338 * 5105 || [full chain of thought tokens] #### 56997714
```
Each subsequent epoch removes some CoT tokens from the LEFT side. By the final epoch, all CoT tokens are gone:
```
1338 * 5105 || #### 56997714
```
The model must now compute the answer without any explicit intermediate steps — it has to do all the intermediate computation in its hidden states.

### What is a Minkowski sum?

A Minkowski sum is a way of combining two sets of points. If you have set A = {a₁, a₂, a₃} and set B = {b₁, b₂, b₃}, their Minkowski sum A ⊕ B = {aᵢ + bⱼ for all pairs i, j}.

Visually, think of it like this: take every point in A, and at each one, place a copy of B. The resulting cloud of points is the Minkowski sum. It creates a "grid-like" structure where you can see clusters-within-clusters.

In this paper, when an attention head attends to two digit tokens aᵢ and bⱼ with weights α and (1-α), the output is αAᵢ + (1-α)Bⱼ, where Aᵢ = W_O W_V E[aᵢ] and Bⱼ = W_O W_V E[bⱼ]. The set of all possible outputs across different digit pairs forms a Minkowski sum of the value-projected digit embeddings.

### What are Fourier bases for digits?

You have 10 digits (0-9). One way to represent them is the obvious one: use a one-hot vector. But there's a more efficient way that also encodes the *relationships* between digits.

The Fourier basis represents each digit n using trigonometric functions:
```
Φ(n) = [1, cos(2πn/10), sin(2πn/10), cos(2πn/5), sin(2πn/5), (-1)^n]
```

This is only 6 numbers per digit, but it captures important structure:
- cos(2πn/10) and sin(2πn/10) place digits around a circle (period 10)
- cos(2πn/5) and sin(2πn/5) place digits around a circle (period 5)
- (-1)^n separates even from odd digits

When you use the period-5 components (k=2), digits 0,2,4,6,8 sit on one regular pentagon and 1,3,5,7,9 on another. Stack these two pentagons using the parity component, and you get a pentagonal prism. That's exactly what the paper finds in the ICoT model's hidden states.

Why is this useful? Because Fourier representations make arithmetic operations into geometric rotations. Adding n to a number becomes a rotation by 2πn/10 on the circle. This is a much simpler operation for a neural network than trying to learn an arbitrary lookup table.

### What is a linear probe?

A linear probe tests whether some information is encoded in a model's hidden states. You take the hidden state vector h (a high-dimensional vector) and train a single linear function w·h to predict some target value.

If the probe works well (low error), it means the information was already present in h in a linearly readable form. If it fails, either the information isn't there, or it's encoded in a nonlinear way that a linear probe can't extract.

In this paper, they probe for ĉ_k (the running sum) at each timestep where the model predicts output digit c_k. High probe accuracy means the model has encoded the full accumulated partial product information at that layer.

---

## The Core Experimental Setup

**Architecture:** 2-layer, 4-head GPT model trained from scratch. This is tiny — 8 attention heads total. No pre-trained knowledge, no confounding factors.

**Task:** 4×4 digit multiplication. Digits written least-significant first (so 1234 becomes "4321"). 80,800 training samples, 1,000 validation, 1,000 test.

**Three models compared:**
1. **SFT (Standard Fine-Tuning):** Trained only on input-output pairs. Achieves ~1% accuracy.
2. **ICoT (Implicit Chain of Thought):** Trained with gradually removed CoT. Achieves 100% accuracy.
3. **Auxiliary Loss model:** Standard fine-tuning plus a linear probe loss on running sums. Achieves 99% accuracy.

**Critical observation about digit ordering:** Digits are written least-significant first. This means a₀ is the ones digit, a₃ is the thousands digit. This is important because it means the "easy" digits (c₀, c₁) only require nearby partial products, while the "hard" middle digits require information from many input positions.

---

## Finding 1: Evidence of Long-Range Dependencies (Section 3.2)

### Logit Attribution Experiment

The authors test whether each model has learned the correct dependencies between input and output digits. They do this by:
1. Taking an input and recording the logits (pre-softmax outputs) for each output digit c_k
2. Swapping one input digit with a random replacement
3. Measuring how much the logits changed

If digit a₂ matters for computing c₅, then changing a₂ should change the logits for c₅.

**Results (Figure 2):**
- **SFT model:** Only nearby digits affect the output. Early input digits have almost no effect on middle output digits. The model hasn't learned that a₀ affects c₄.
- **ICoT model:** The correct dependency pattern emerges. Each output digit c_k is affected by all input digits aᵢ, bⱼ where i + j ≤ k, with the strongest effects from pairs where i + j = k.

### Linear Probe Experiment

They train linear probes to predict ĉ_k (the running sum that captures all long-range dependencies) from the hidden states.

**Results (Figure 3):**
- **SFT model:** Mean Absolute Error (MAE) ranges from 28 to 113. The probes fail badly, especially for middle digits. The running sum information simply isn't in the hidden states.
- **ICoT model:** MAE ranges from 0.56 to 2.00. The probes are nearly perfect. The hidden states contain all the necessary long-range information.

### Devil's Advocate on Finding 1

**Criticism 1: Logit attribution is crude.** Swapping a single digit and measuring logit change is a blunt instrument. It measures *any* kind of sensitivity, not necessarily *correct* sensitivity. The ICoT model could be sensitive to distant digits for wrong reasons (like some kind of artifact).

**Criticism 2: Linear probes prove readability, not usage.** Just because you can linearly decode ĉ_k from the hidden states doesn't mean the model actually uses ĉ_k for computation. This is the classic "probe tax" problem. The information might be there as a side effect without being causally important. A causal intervention (like adding noise to the subspace encoding ĉ_k and checking if the output changes) would be stronger evidence.

**Criticism 3: The SFT model gets 81% digit-level accuracy.** This means it gets most digits right! The 1% full-accuracy figure hides the fact that the model does learn significant structure. It just fails on the hardest digits (c₃-c₆ specifically). The framing makes SFT sound completely incompetent, which isn't quite right.

---

## Finding 2: Attention Trees (Section 3.3)

This is the mechanistic heart of the paper. The claim is that the ICoT model constructs a binary-tree-like computation graph through its attention patterns:

**Layer 1 (Caching):** Each attention head at each timestep attends to exactly TWO tokens — one digit from a and one from b. This computes the pairwise product aᵢbⱼ and "caches" it in the hidden state at that timestep.

**Layer 2 (Retrieval):** When predicting output digit c_k, the second layer's attention heads attend to the timesteps where the relevant partial products were cached in layer 1.

The example in Figure 4 shows how c₂ is computed:
- To compute c₂, you need a₂b₀, a₁b₁, a₀b₂, plus carry from ĉ₁
- At timestep b₀ in layer 1: heads attend to {a₂, b₀} → caches a₂b₀
- At timestep b₂ in layer 1: one head attends to {a₁, b₁}, another to {a₀, b₂}
- At timestep c₀ in layer 1: heads attend to {a₁, b₀} and {a₀, b₁}
- Layer 2 then attends to these cache sites to gather all partial products

This is a clever computational strategy. The model essentially uses earlier token positions as "scratch space" to store intermediate results, then reads them back later.

### Devil's Advocate on Finding 2

**Criticism 1: Attention maps are averaged over 1,000 samples.** This shows the *average* attention pattern, not what happens on any individual example. Individual attention patterns could be much messier. Averaging can create the appearance of clean structure that doesn't exist per-example.

**Criticism 2: The "binary tree" metaphor is loose.** A real binary tree has a strict structure. What we actually see is "attention heads tend to attend to pairs of digit tokens." Calling this a "binary tree" or "DAG" is an interpretive frame imposed by the authors, not something rigorously demonstrated. The attention patterns in Figure 10 (appendix) are quite noisy for some heads.

**Criticism 3: How does the model actually MULTIPLY the two attended digits?** The paper says each head attends to aᵢ and bⱼ, but attention computes a weighted SUM of value vectors, not a product. The output is α·V(aᵢ) + (1-α)·V(bⱼ), which is a weighted average, not a multiplication. The paper hand-waves this by moving to the Minkowski sum analysis (Section 4.1), but never actually shows that the product aᵢ × bⱼ is represented in the resulting vector. This is a significant gap.

**Criticism 4: Only 2 layers.** With only 2 layers, the model has extremely limited depth. The "cache and retrieve" pattern is essentially forced by the architecture — there's no other way to get information from input tokens to output tokens with only 2 layers. In a deeper model (like Llama 3.1 8B with 32 layers), the model has many more options and might not use this pattern at all.

**Criticism 5: What about a₀b₀?** The paper admits in a footnote that "there may be a couple of different ways that a₀b₀ is derived" and that it "plays a relatively minor role in computing c₂." This is concerning — if the mechanism can't fully account for even one partial product, how confident can we be in the overall story?

---

## Finding 3: Minkowski Sum Geometry (Section 4.1)

When an attention head attends to exactly two tokens (aᵢ and bⱼ) with weights α and (1-α), the output is:

```
ATT¹(i,j) = α·Aᵢ + (1-α)·Bⱼ + ε
```

where Aᵢ = W_O W_V E[aᵢ] and Bⱼ = W_O W_V E[bⱼ] are the value-projected embeddings.

Across all digit pairs, the set of outputs forms a subset of the Minkowski sum (αA) ⊕ ((1-α)B).

The paper shows this visually through 3D PCA plots (Figure 5): you see clusters (one per aᵢ value) where each cluster has sub-clusters (one per bⱼ value), and the sub-cluster geometry mirrors the global cluster geometry. This "nested" or "fractal-like" structure is a signature of Minkowski sums.

The covariance analysis (Equation 5) explains why: Σ_ATT = α²Σ_A + (1-α)²Σ_B, and since both covariances share the same eigenvectors (they involve the same weight matrices), PCA picks up the same directions at both scales.

### Devil's Advocate on Finding 3

**Criticism 1: The Minkowski sum result is almost trivially true.** If attention is sparse (attending to exactly 2 tokens), then by definition the output is a weighted sum of two value vectors. The "Minkowski sum" framing sounds fancy, but it's just saying "weighted averages of two things form a grid." This is a mathematical consequence of the architecture, not a discovered insight.

**Criticism 2: The shared eigenvector claim requires ignoring positional encodings.** The paper explicitly says "ignoring position embeddings." But position embeddings are not ignorable — they're how the model knows WHICH digit is which. If the positional encoding significantly perturbs the value vectors, the neat Minkowski sum structure breaks down. How large is this perturbation in practice?

**Criticism 3: The PCA visualization is in 3D.** The actual computation happens in the full model dimension (which they don't specify but appears to be 128-256d based on the 4-head architecture). Showing 3D PCA captures only the top 3 principal components. What's happening in the other dimensions? The clean nested structure in 3D might not capture important computation happening in dimensions 4+.

**Criticism 4: This still doesn't explain multiplication.** A Minkowski sum of aᵢ and bⱼ gives you a representation that encodes BOTH digits, but that's not the same as computing their PRODUCT. How does the MLP (or subsequent attention) extract the actual numerical product from this representation? The paper doesn't address this.

---

## Finding 4: Pentagonal Prism via Fourier Bases (Section 4.2)

The paper's most visually striking finding: digits 0-9 in the ICoT model's final hidden layer form a pentagonal prism when viewed through PCA.

**How this works:**

The model represents each digit n using Fourier basis functions with frequencies k ∈ {0, 1, 2, 5}:
```
Φ(n) = [1, cos(2πn/10), sin(2πn/10), cos(2πn/5), sin(2πn/5), (-1)^n]
```

This is a 6-dimensional representation. The key frequencies:
- k=2 (period 5): cos(2πn/5) and sin(2πn/5) place digits on a pentagon. Because 10/gcd(10,2) = 5, this creates a 5-fold symmetry.
- k=5 (parity): (-1)^n separates even and odd digits.
- Combined: Two pentagons (even and odd) stacked along the parity axis = pentagonal prism.

**Evidence quality:** Table 1 shows R² values of Fourier fits:
- Embeddings: R² = 0.84 (with k=0,1,2,5) → 1.0 (with k=0,1,2,3,4,5)
- MLP weights: R² = 0.95 → 1.0
- Final hidden layer: R² = 0.99 → 1.0

These are strong fits. The 6-frequency Fourier basis explains essentially all the variance, and the 10-frequency basis gives perfect reconstruction (which is expected since 10 basis functions for 10 digits is a complete basis).

### Devil's Advocate on Finding 4

**Criticism 1: This is specific to base-10 digits.** The Fourier basis over 10 elements is naturally well-suited to this problem because 10 has factors 2 and 5. This creates the pentagonal prism specifically because of the number theory of base 10. This finding wouldn't generalize to, say, hexadecimal or arbitrary tokenizations.

**Criticism 2: The connection to Kantamneni & Tegmark (2025) raises priority questions.** Kantamneni & Tegmark already found Fourier bases in LLM digit representations for addition. This paper confirms it for multiplication in a tiny model. Is this a new finding, or a replication?

**Criticism 3: Only 10 points in a 6D space.** Fitting 10 points with 6 basis functions is not very impressive statistically. You have 6 free parameters (the Fourier coefficients) to fit 10 data points. The R² = 0.84 for embeddings with 6 parameters / 10 data points actually isn't that high — you'd expect a reasonable fit from almost any 6-dimensional basis. The R² = 1.0 with 10 basis functions for 10 data points is trivially expected.

**Criticism 4: SFT model shows "no obvious patterns" — but is that actually true?** The paper dismisses the SFT model's hidden states as unstructured (Figure 6, left panel). But the SFT model gets 81% digit-level accuracy. It must have SOME structure. Maybe it uses a different geometric organization that isn't visible in a 3D PCA. The paper doesn't try harder visualization methods (like t-SNE, UMAP) or higher-dimensional analysis on the SFT model.

**Criticism 5: Why these specific frequencies?** The paper takes k ∈ {0, 1, 2, 5} following Kantamneni & Tegmark. But why does the model choose these frequencies? Is there a theoretical reason, or is it an empirical observation? The paper doesn't explain why k=3 and k=4 are absent from the primary representation (even though adding them gives R²=1). This could matter — the model might use k=3 and k=4 for different computational purposes.

---

## Finding 5: Why SFT Fails — Local Optimum (Section 5)

The paper examines per-digit gradient norms and losses during SFT training (Figure 7a):

1. **First learned:** c₀ and c₁ (easiest — require fewest partial products and no/minimal carry chains)
2. **Next learned:** c₇ (the last digit — also relatively easy since it only depends on a₃b₃ plus carry)
3. **Eventually learned:** c₂
4. **Never learned:** c₃ through c₆ — losses plateau permanently

The interpretation: once c₀, c₁, c₇ are learned, their gradients drop to near zero. The remaining gradient signal comes only from c₃-c₆, but the model is stuck. It would need to fundamentally restructure its attention patterns to create the long-range dependencies needed for these digits, but gradient descent can't find a path to this restructuring from the current local optimum.

**Scaling doesn't help:** A 12-layer, 8-head model shows the same pattern (Figure 9, Appendix C). Same ~1% accuracy.

### Devil's Advocate on Finding 5

**Criticism 1: The "local optimum" claim is unproven.** The paper observes that loss plateaus, but that doesn't prove it's a local optimum. It could be a saddle point. It could be that the loss landscape has very flat gradients in the direction needed for improvement. It could be a limitation of the optimizer (Adam vs SGD vs different learning rates). The paper uses a single learning rate (5e-5) and doesn't explore whether different optimization strategies would help.

**Criticism 2: The 12-layer model uses the same hyperparameters.** The paper says the 12-layer model achieves the same low accuracy, but does it use the same learning rate, same optimizer, same data? If so, this is a poorly controlled scaling experiment. Larger models often need different hyperparameters.

**Criticism 3: The training data is tiny.** 80,800 samples for 4×4 multiplication is a small fraction of the 10^8 possible inputs (10,000 × 10,000). With so little data, the model might simply not have enough coverage to learn the general algorithm. The paper doesn't explore whether more data helps SFT.

**Criticism 4: The digit ordering matters.** Digits are written least-significant first. This is a specific choice that affects what's "easy" and "hard." With most-significant-first ordering, the difficulty pattern might be completely different. The paper doesn't discuss this dependency.

---

## Finding 6: The Auxiliary Loss Fix (Section 6)

The key intervention: add a loss that forces the model to predict the running sum ĉ_k through linear probes attached to attention head outputs in the second layer.

The loss is:
```
L = L_LM + λ · L_aux
```
where L_aux is the MSE between the linear probe predictions and the true running sums ĉ_k.

**Results:** 99% accuracy on 4×4 multiplication with the same tiny 2-layer model.

**Learning dynamics (Figure 7b):** The model learns digits outside-in: c₀, c₁, c₇ first, then c₂, c₃, c₄, c₆, and finally c₅ (the hardest middle digit).

The auxiliary-loss model also develops similar attention patterns to ICoT: binary-tree attention in layer 1, plus an interesting "parallelogram" attention pattern in Layer 2 Head 2 that attends to all relevant digits simultaneously.

### Devil's Advocate on Finding 6

**Criticism 1: This is a task-specific hack, not a general solution.** The paper openly acknowledges this. The auxiliary loss requires knowing ĉ_k, which requires knowing the exact algorithmic decomposition of multiplication. You can't apply this to a task where you don't know the intermediate computation. The paper says "we anticipate generic improvements" but doesn't provide any.

**Criticism 2: 99% is not 100%.** ICoT achieves 100% accuracy, but the auxiliary loss only reaches 99%. What fails in that last 1%? Is it specific patterns? The paper doesn't analyze the failure cases.

**Criticism 3: The value of λ is not specified.** The paper doesn't tell us what weight λ they used for the auxiliary loss, or how sensitive the results are to this choice. Is there a wide range of λ that works, or does it need careful tuning?

**Criticism 4: The auxiliary loss requires ground-truth running sums.** Computing ĉ_k requires actually knowing how to multiply the numbers. So the supervision signal already contains the answer decomposed into steps. This is almost as much supervision as full chain-of-thought — you're essentially giving the model the intermediate computation targets. The paper doesn't clearly acknowledge that this is a significant amount of extra supervision.

---

## Cross-Cutting Critical Issues

### Issue 1: Tiny Model, Big Claims

The entire paper studies 2-layer, 4-head models. The paper claims implications for "why Transformers fail at multiplication" generally, but everything is specific to this minimal architecture. The 12-layer, 8-head experiment is a single data point that's not well-controlled. The gap to real models (Llama, GPT-4) is huge.

The paper's title is "Why Can't Transformers Learn Multiplication?" but the honest title would be "Why Can't Tiny 2-Layer Models Learn 4×4 Multiplication via Standard Fine-Tuning?"

### Issue 2: ICoT Confounds

The ICoT model was trained with explicit CoT tokens that were gradually removed. The model has seen the intermediate computation during training. Comparing this to SFT (which never sees intermediates) and concluding that the difference is about "long-range dependencies" ignores the massive difference in supervision signal. Of course the model that was trained on intermediate steps does better. The interesting question is why it RETAINS the correct computation after the CoT tokens are removed, but the paper doesn't deeply address this.

### Issue 3: Missing Ablations

Several important ablations are missing:
- What happens with different digit orderings (most-significant first)?
- What happens with different data sizes?
- What happens with different optimizers (SGD vs Adam)?
- What happens with learning rate schedules?
- How sensitive is ICoT to the rate of CoT token removal?
- Does SFT work better with more layers if you also tune hyperparameters?

### Issue 4: The Geometry Story is Incomplete

The paper finds Fourier bases and Minkowski sums, but never closes the loop to show HOW these geometric structures actually compute multiplication. Finding structure is not the same as explaining computation. The paper shows what the representations LOOK LIKE but not how they FUNCTION to produce the correct output. Specifically:
- How does the MLP use the Fourier representation to compute mod-10 and division-by-10 (for extracting digits and carries)?
- How does the Minkowski sum of two digit embeddings get transformed into their numerical product?
- How is the addition of partial products performed across the attention tree?

### Issue 5: Relationship to Llama 3.1 8B

For our research, the biggest question is: does ANY of this transfer to production models?

Llama 3.1 8B has:
- 32 layers (vs 2), so the "cache and retrieve" attention tree would be completely different
- 4096 dimensions (vs ~256), so the Fourier basis representation exists in a vastly higher-dimensional space
- BPE tokenization that doesn't tokenize individual digits, so the digit-level analysis doesn't directly apply
- Pre-trained knowledge that confounds clean multiplication learning

The Fourier digit representation is the most likely finding to transfer, based on Kantamneni & Tegmark's independent observation in large models. The attention tree mechanism is the least likely to transfer — larger models will use completely different computational strategies with their deeper architectures.

---

## Connection to Transformer Circuits (transformer-circuits.pub)

The "When Models Manipulate Manifolds" paper from Anthropic's interpretability team provides a useful comparison point. That paper studies character counting in Claude 3 Haiku (a much larger model) and finds:

- **Circular/helical manifolds** for representing character counts — analogous to the Fourier-based pentagonal prism here, but in a natural task rather than a synthetic one
- **Dilation** in the feature representations — character count features cover increasingly wide ranges at higher counts, analogous to the Weber-Fechner law in perception
- **Boundary detection features** that interact geometrically with count features
- **Fourier-like ringing** in the cosine similarity structure, analytically explained through truncated circulant matrix eigendecomposition

The key connection: both papers find that neural networks spontaneously develop periodic/Fourier representations for numerical quantities. The Anthropic paper's analytic explanation (truncated Fourier modes of circulant similarity matrices produce ringing) provides a theoretical foundation for WHY Fourier bases emerge, which the Bai et al. paper lacks.

The Anthropic paper also demonstrates that geometric structure matters in much larger models, not just tiny ones — partially addressing our concern about transferability.

The "Sparse Mixtures of Linear Transforms" (MOLT) work from Anthropic is also relevant: they found that transcoders decomposed addition circuits into "lookup table" features (like "6 + 9 = 5 mod 10") that MISSED the geometric structure. This suggests that the choice of interpretability tool matters — SAE features might not capture the continuous geometric representations that this paper (and Kantamneni & Tegmark) find.

---

## Pass 2 Summary Template

```
PAPER: Why Can't Transformers Learn Multiplication?
TYPE: Empirical / Mechanistic Interpretability
PEER-REVIEWED: No (arXiv preprint)

PROBLEM: Transformers fail at 4×4 multiplication even with fine-tuning.
WHY HARD: Requires long-range dependencies across all partial products and carries.
MAIN CLAIM: ICoT succeeds by constructing attention trees for caching partial products
with Fourier-based Minkowski sum geometry; SFT fails by getting stuck in local optima;
auxiliary running-sum loss fixes this.

TECHNICAL BARRIER: Standard gradient descent with autoregressive loss cannot discover
the long-range attention patterns needed for middle output digits.
KEY INSIGHT: Multiplication can be decomposed into a tree of cached partial products,
and the running sum ĉ_k is a sufficient intermediate representation. Supervising this
intermediate breaks the local optimum.

COMPARISON: ICoT 100%, SFT ~1%, Auxiliary loss 99% on 4×4 multiplication.
LIMITATIONS: Tiny model (2 layers), task-specific fix, geometry story incomplete,
no causal intervention, missing ablations, unknown transferability to large models.

DEVIL'S ADVOCATE — THREE WEAKEST POINTS:
1. The paper never shows HOW the geometric structures (Minkowski sums, Fourier bases)
   actually compute multiplication — finding structure ≠ explaining mechanism.
2. The 2-layer architecture forces the "cache and retrieve" pattern; this tells us about
   the minimum viable architecture, not about how real models work.
3. The auxiliary loss requires knowing the complete algorithmic decomposition — it's
   nearly as much supervision as full CoT, undermining the "fix" narrative.

OPEN QUESTIONS:
1. Do larger pre-trained models (like Llama 3.1 8B) use similar geometric structures
   for multiplication, or entirely different approaches?
2. Can the Fourier basis finding be used diagnostically — if we find pentagonal prism
   structure in a model's activations, does that predict multiplication capability?
3. What is the generic version of the auxiliary loss that could help with arbitrary
   long-range dependency tasks?

CONNECTIONS TO OUR RESEARCH:
- Pipeline Step 1 (concept subspace discovery): The Fourier digit representation is exactly
  the kind of structured subspace our pipeline should detect. LDA on carry values should
  find this structure.
- Pipeline Step 2 (Fourier test): This paper confirms Fourier structure in multiplication
  representations, validating our Fourier test as a screening tool.
- Pipeline Step 3 (GP approach): The pentagonal prism is a concrete manifold whose
  curvature and topology we could characterize with Hauberg's probabilistic geometry.
- For Llama 3.1 8B: We should check whether the Fourier representation transfers by
  probing last-token activations for Fourier structure in our multiplication dataset.
```

---

# PASS 3 — The Swamp (Deep Critical Analysis)

## The Multiplication Algorithm in Detail

Let me carefully work through the math to make sure the paper's formulation is correct and complete.

Given a = (a₃, a₂, a₁, a₀) and b = (b₃, b₂, b₁, b₀), the product c = a × b has 8 digits (c₇, ..., c₀).

The partial product sum for position k is:
```
s_k = Σ_{i+j=k} aᵢ × bⱼ
```

For each k:
- s₀ = a₀b₀ (max value: 9×9 = 81)
- s₁ = a₁b₀ + a₀b₁ (max: 2×81 = 162)
- s₂ = a₂b₀ + a₁b₁ + a₀b₂ (max: 3×81 = 243)
- s₃ = a₃b₀ + a₂b₁ + a₁b₂ + a₀b₃ (max: 4×81 = 324)
- s₄ = a₃b₁ + a₂b₂ + a₁b₃ (max: 3×81 = 243)
- s₅ = a₃b₂ + a₂b₃ (max: 2×81 = 162)
- s₆ = a₃b₃ (max: 81)

The running sum:
```
ĉ_k = s_k + r_{k-1}    (where r_{-1} = 0)
c_k = ĉ_k mod 10
r_k = floor(ĉ_k / 10)
```

**What are the ranges of ĉ_k?** This matters for probing:
- ĉ₀ ∈ [0, 81]
- ĉ₁ ∈ [0, 162 + 8] = [0, 170]  (carry r₀ ≤ 8)
- ĉ₂ ∈ [0, 243 + 17] = [0, 260]
- ĉ₃ ∈ [0, 324 + 26] = [0, 350]
- ĉ₄ ∈ [0, 243 + 35] = [0, 278]
- ĉ₅ ∈ [0, 162 + 27] = [0, 189]
- ĉ₆ ∈ [0, 81 + 18] = [0, 99]
- ĉ₇ = r₆ ∈ [0, 9]

These ranges matter because the middle digits have the LARGEST ranges for ĉ_k (up to 350), meaning the model needs the most precision for exactly the digits where it fails. The linear probe accuracy (Figure 3) tracks this: the MAE is largest for ĉ₃ and ĉ₄ in the SFT model.

**Key insight I notice:** The difficulty pattern (middle digits are hardest) is not just about long-range dependencies — it's also about the RANGE of values that need to be represented. ĉ₃ ∈ [0, 350] requires representing 351 distinct values, while ĉ₀ ∈ [0, 81] only requires 82. The paper doesn't disentangle these two factors (long-range dependencies vs. representation precision).

## The Attention Tree Mechanism: Can It Actually Work?

Let me think carefully about whether the described mechanism is computationally sufficient.

**Claim:** Layer 1 heads attend to exactly two tokens (aᵢ, bⱼ) and "cache" the product aᵢbⱼ.

**Problem:** The attention output is α·W_O W_V E[aᵢ] + (1-α)·W_O W_V E[bⱼ]. This is a weighted AVERAGE of two projected embeddings, not their product. How can a weighted average encode a product?

**Possible resolution:** The model doesn't need to compute the EXACT product aᵢ × bⱼ. It needs to produce a representation from which the PRODUCT can be extracted. With Fourier bases, this is actually possible:

If aᵢ is represented as cos(2πaᵢ/10) and bⱼ as cos(2πbⱼ/10), then:
```
cos(A) + cos(B) = 2·cos((A+B)/2)·cos((A-B)/2)
```

The SUM of Fourier representations contains information about the PRODUCT of the frequencies. This is the standard trick in Fourier analysis. The MLP could then extract the product information from these combined signals.

But wait — the attention output is a WEIGHTED sum (with weight α), not a simple sum. Does the weighting matter? Yes, but the paper doesn't analyze how α affects the computation. And the actual product aᵢ × bⱼ is a MULTIPLICATION of integer values, not a frequency interaction. The Fourier trick gives you information about aᵢ + bⱼ and aᵢ - bⱼ (sums and differences), not aᵢ × bⱼ.

**This is a genuine gap in the paper.** The story should be: Fourier representations → some operation in the MLP → actual products. But the paper never shows the MLP computation step. The MLP is treated as a black box.

## The Fourier Basis: Is Pentagonal Prism Optimal?

The paper finds frequencies k ∈ {0, 1, 2, 5} in the primary representation. Let me think about why.

The digit multiplication table (aᵢ × bⱼ for aᵢ, bⱼ ∈ {0,...,9}) has specific structure:
- It's symmetric (aᵢbⱼ = bⱼaᵢ)
- Results range from 0 to 81
- The MOD 10 operation on ĉ_k creates periodicity modulo 10

For the MOD 10 operation, the k=1 frequency (period 10) is the most relevant: cos(2πn/10) and sin(2πn/10) can distinguish all residues mod 10.

For the FLOOR division by 10 (extracting carries), you need to distinguish between different ranges, which the k=2 frequency (period 5) helps with — it separates digits into two groups of 5.

The parity component k=5 separates {0,2,4,6,8} from {1,3,5,7,9}, which is useful because the carry computation depends on whether intermediate sums are even or odd.

**So the specific frequencies are probably not arbitrary** — they correspond to the mathematical structure of base-10 arithmetic. k=1 handles mod 10, k=2 handles the quintic structure, k=5 handles parity. This is elegant but the paper doesn't make this connection explicit.

## Reproducing the Local Optimum Argument

The paper claims that under SFT, the model reaches a local optimum where:
1. c₀, c₁, c₇ are learned (easy digits)
2. c₂ is eventually learned
3. c₃ through c₆ are never learned (loss plateaus)

Let me think about whether this learning order makes mathematical sense.

**c₀ = (a₀b₀) mod 10:** This only requires a₀ and b₀ — the two nearest input digits. Easy.

**c₁ = (a₁b₀ + a₀b₁ + r₀) mod 10:** Requires three inputs: a₀, a₁, b₀, b₁. Still relatively local.

**c₇ = r₆:** The last digit is just the final carry. But computing it requires knowing ALL the partial products, so why is it "easy"? Possible explanation: c₇ has a small range (0-9 as a carry digit), and it's the most significant digit which constrains the answer in a way that might have heuristic shortcuts. Actually, r₆ only depends on a₃b₃ and r₅, and r₅ depends on fewer terms going back. The carry chain propagates but the CONTRIBUTION of each partial product to c₇ diminishes exponentially with distance. So approximate heuristics might work for c₇.

**c₂ learned later:** s₂ = a₂b₀ + a₁b₁ + a₀b₂ requires 3 partial products plus carry. The model eventually finds a way to attend to these pairs.

**c₃-c₆ never learned:** s₃ requires 4 partial products, s₄ requires 3 but with large carries from s₃. The carry chain length is the critical bottleneck. To compute c₅, you need carries from c₀ through c₄, meaning you need ALL the partial products for ALL earlier positions. This is where the long-range dependency becomes critical.

**But why can't gradient descent learn this?** The paper suggests it's a local optimum, but here's an alternative hypothesis: maybe the 2-layer architecture simply doesn't have enough capacity to implement the full carry chain computation without the attention tree trick. The SFT model never discovers the cache-and-retrieve strategy because gradient descent doesn't naturally push toward this specific pattern.

The 12-layer model's failure weakens this "capacity" argument — more layers provide more capacity. But the 12-layer model uses the same learning rate and training setup, so it's not clear if the issue is truly fundamental or just an optimization failure.

## What This Means for Our Pipeline

### For Step 1 (Concept Subspace Discovery)

The Fourier basis finding directly validates our plan to check for Fourier structure in Llama 3.1 8B's multiplication representations. Specifically:

1. **LDA on carries (Phase D):** We encode carries by value (0-8). The paper suggests these carry values should correlate with specific Fourier modes. Our LDA should find directions that align with Fourier basis vectors.

2. **Conditional covariance (Phase C):** Conditioning on specific digit values and examining the covariance should reveal the Minkowski sum structure — fixed aᵢ should show bⱼ-organized sub-clusters.

3. **UMAP visualization (Phase A):** We should see pentagonal prism-like structure in UMAP projections of the final hidden states at multiplication answer positions. But we need to be careful — UMAP can distort this structure.

### For Step 2 (Fourier Test)

This paper strongly validates Fourier testing as a screening tool. If we DON'T find Fourier structure in Llama 3.1 8B's multiplication representations, that could mean either:
- The model doesn't use Fourier representations (uses a different strategy)
- The Fourier structure is present but at a different layer or token position
- Llama uses multi-digit tokens that obscure digit-level structure

### For Step 3 (GP Approach)

The pentagonal prism is a concrete manifold to characterize:
- It's a 2-manifold (the surface of the prism) embedded in 6+ dimensions
- It has interesting curvature properties (flat pentagonal faces, high curvature at edges)
- The cross-layer geometric transformation (from embedding → hidden layer) should show how the manifold evolves

This would be an excellent validation case for our Hauberg-style probabilistic geometry framework: fit GP posterior over the Riemannian metric on the pentagonal prism manifold, compute curvature with uncertainty, and verify against the known analytic Fourier structure.

### For Llama 3.1 8B Specifically

Key differences to account for:
1. **Tokenization:** Llama uses BPE, not character-level tokenization. "1234" might be tokenized as a single token or split unpredictably. We need to design our multiplication prompts carefully to control tokenization.

2. **Pre-training confounds:** Llama has seen multiplication examples in its training data. Its representations will be influenced by ALL of its training, not just multiplication. The concept subspace for multiplication will be embedded in a much higher-dimensional space with many other concepts superimposed.

3. **32 layers vs 2:** The computation will be distributed across many more layers. The "cache and retrieve" pattern could be much more distributed, with different layers handling different partial products. Our layer-by-layer probing needs to map out WHERE in the 32-layer stack each component of the multiplication is computed.

4. **4096 dimensions vs ~256:** The concept subspace for multiplication will occupy a tiny fraction of the full activation space. Our conditional covariance + rSVD pipeline is designed for exactly this scenario — finding low-dimensional concept subspaces in high-dimensional activation spaces.

---

## Key Equations Reference

**Partial product sum:** s_k = Σ_{i+j=k} aᵢbⱼ

**Running sum:** ĉ_k = s_k + r_{k-1}, where r_{k-1} = floor(ĉ_{k-1}/10), r₋₁ = 0

**Output digit:** c_k = ĉ_k mod 10

**Carry:** r_k = floor(ĉ_k / 10)

**Minkowski sum attention output:** ATT¹(i,j) = α·W_O W_V E[aᵢ] + (1-α)·W_O W_V E[bⱼ] + ε

**Covariance decomposition:** Σ_ATT = α²Σ_A + (1-α)²Σ_B

**Fourier basis:** Φ(n) = [1, cos(2πn/10), sin(2πn/10), cos(2πn/5), sin(2πn/5), (-1)^n]

**Auxiliary loss:** L = L_LM + λ · (1/H)Σ_h (1/8)Σᵢ (w_h · ATT²_h(·) - ĉᵢ)²

---

## Final Assessment

### What This Paper Gets Right
1. Clearly identifies a real problem (multiplication failure) and finds a clean experimental setup
2. The logit attribution and probing evidence for long-range dependency failure is convincing
3. The Fourier basis finding is independently validated by other work and likely robust
4. The auxiliary loss experiment is a nice proof-of-concept
5. Good reproducibility: code released, experimental details provided

### What This Paper Gets Wrong or Leaves Incomplete
1. Never explains HOW the geometric structures compute multiplication — just shows they exist
2. The 2-layer model is too small to draw general conclusions about "Transformers"
3. The "local optimum" claim is not rigorously established
4. Missing critical ablations (data size, optimizer, digit ordering, learning rate)
5. The auxiliary loss is nearly as supervised as full CoT, which undermines the "simple fix" narrative
6. No causal interventions — all evidence is correlational

### Overall Grade
Strong empirical work within its limited scope. The insights about Fourier representations and long-range dependency failure are valuable and connect well to the broader mechanistic interpretability literature. But the paper overclaims in its title and framing — this is a study of tiny models, not a general answer to why Transformers fail at multiplication.

For our research, the most valuable takeaways are:
1. Fourier basis representations for digits are robust across models and tasks
2. The running sum ĉ_k is the right intermediate quantity to probe for
3. The carry chain creates a genuine long-range dependency bottleneck
4. Geometric structure (Minkowski sums, pentagonal prisms) exists but its computational role is not yet understood — this is exactly the gap our GP-based framework could fill