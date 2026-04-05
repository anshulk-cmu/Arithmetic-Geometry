# Deep Analysis: "The Geometry of Reasoning: Flowing Logics in Representation Space"
## By Yufa Zhou*, Yixiao Wang*, Xunjian Yin*, Shuyan Zhou, Anru R. Zhang (Duke University)

**Paper:** arXiv:2510.09782v1, October 10, 2025

---

# PASS 1 — The Jigsaw Puzzle (What is this paper about?)

## Paper Classification

**Type:** Theory/Framework paper with empirical validation. It proposes a new geometric framework and then tests it with experiments.

**Authors:** Five researchers from Duke University, three of them marked as equal contribution (*). Anru R. Zhang is a faculty member partially supported by an NSF CAREER grant, which signals established academic credibility. The other authors appear to be graduate students. This is an academic paper, not an industry lab product.

**Venue:** arXiv preprint (not yet peer-reviewed at a top venue). This means the ideas have been written up but have NOT gone through the rigorous review process of a top conference like NeurIPS, ICML, or ICLR. We should read with calibrated skepticism — the ideas might be interesting but haven't been stress-tested by expert reviewers yet.

**Code and Data:** They release both code (GitHub) and a dataset (HuggingFace), which is a positive reproducibility signal.

**Citations of note:** The paper cites Anthropic's Transformer Circuits work [1], the Modell et al. concept manifolds paper [46], and many recent interpretability papers. It positions itself at the intersection of geometric interpretability, formal logic, and reasoning analysis.

---

## Q1: What problem is being solved?

**In the simplest possible language:**

When a large language model (like ChatGPT or Claude) "thinks through" a math problem step by step, what is actually happening inside the model's brain? Specifically: do those internal thinking steps follow some kind of organized path through the model's "mental space," or is it just random noise?

**Slightly more precisely:** The paper asks whether chain-of-thought (CoT) reasoning in LLMs can be understood as smooth, continuous trajectories (called "flows") through the model's high-dimensional representation space, and whether the *logical structure* of the reasoning (as opposed to the *topic* being reasoned about) is what governs the shape of these trajectories.

**One-sentence summary:** "This paper studies whether LLM reasoning can be modeled as smooth geometric flows in representation space, where logical structure — not surface-level semantics — acts as the controller of how these flows move."

---

## Q2: Why is the problem interesting and hard?

**Why it matters:** If reasoning in LLMs really does trace organized geometric paths, this would give us:
- A new way to *understand* what the model is doing when it reasons
- A way to *detect* when reasoning is going wrong (the flow goes in a bad direction)
- A path toward *controlling* reasoning (steer the flow)
- Mathematical tools (velocity, curvature) to *measure* reasoning quality

**Why it's hard:**
1. LLMs operate in extremely high-dimensional spaces (thousands of dimensions), making geometric analysis difficult
2. The output of an LLM is discrete (tokens/words), but geometry requires continuous objects (smooth curves) — bridging this gap is non-trivial
3. Separating "logical structure" from "surface-level topic" is hard because they're deeply entangled in natural language
4. It's unclear whether the geometric properties we observe are truly meaningful or just artifacts of the dimensionality reduction methods we use to visualize them

**One-sentence version:** "This is nontrivial because reasoning is discrete (step by step) while geometry is continuous (smooth curves), and because separating logical structure from semantic content requires carefully controlled experiments."

---

## Q3: What is the main claim?

**The headline claims (there are two):**

1. **LLM reasoning corresponds to smooth flows in representation space.** When a model reasons step by step, and you record the model's internal state at each step, the resulting sequence of points forms a smooth curve (not a random scatter).

2. **Logical structure acts as a local controller of these flows' velocities.** The *topic* of the reasoning (weather, finance, sports) determines *where* the flow is in space (its position), but the *logical structure* (the pattern of "if A then B", "A and B therefore C") determines *how the flow moves* (its velocity and curvature).

**The key empirical finding:** When you take the same logical proof skeleton (like "A implies B, B implies C, therefore A implies C") and dress it up in completely different topics (weather vs. finance) and languages (English vs. German), the velocity and curvature of the resulting reasoning flows are highly similar. But the positions are different. This suggests the model has internalized logical structure as a *dynamical* property of the flow, separate from the semantic content.

**In more mathematical terms:** Let y_t be the model's hidden state at reasoning step t. The position y_t clusters by topic/language. But the velocity Δy_t = y_{t+1} - y_t and the Menger curvature (a measure of how sharply the flow turns) are more similar for flows sharing the same logical skeleton than for flows sharing the same topic.

---

## Relevance to Mechanistic Interpretability Research

This paper is directly relevant to research on how transformers represent concepts geometrically. The Gurnee et al. "When Models Manipulate Manifolds" paper (in the project context) shows how Claude represents scalar quantities like character counts on geometric manifolds (helices, curves). This paper extends the geometric perspective from *static* representations to *dynamic* reasoning trajectories. The connection is: if concepts live on manifolds, then reasoning might be understood as *flows along* those manifolds.

The paper also connects to the Hauberg analysis in the project context — the question of whether you need probabilistic methods to correctly characterize manifold geometry becomes relevant when asking whether the "smooth flow" hypothesis is actually a property of the model or an artifact of the analysis method.

**Verdict:** Proceed to Pass 2 and Pass 3. This paper is directly relevant to understanding geometric representations in transformers, extends the manifold perspective to reasoning dynamics, and its claims — if validated — would have significant implications for interpretability.

---

---

# PASS 2 — Scuba Dive (How does the paper work?)

## Prerequisite Concepts Explained for a Grade 12 Student

Before we can understand this paper, we need to build up some vocabulary. I'll explain everything assuming you know basic algebra, basic calculus (derivatives = rates of change), and some familiarity with vectors (arrows in space with direction and length).

### What is a "representation" or "embedding"?

Think of an LLM like a student reading a sentence. At each stage of processing, the model creates an internal "summary" of what it has understood so far. This summary is a list of numbers — typically thousands of them. For example, after reading "The cat sat on the mat," the model might internally represent this as a list of 3,072 numbers like [0.42, -1.7, 0.003, ...].

This list of numbers is called a **representation** or **embedding**. You can think of it as a point in a very high-dimensional space. Just like a point on a flat map has 2 coordinates (x, y), a point in the model's "mental space" has thousands of coordinates.

**Key insight:** Sentences with similar meanings tend to have similar representations (their points are close together in this high-dimensional space). Sentences with different meanings tend to have points that are far apart.

### What is a "flow" or "trajectory"?

Imagine you're tracking a ball rolling across a table. At each moment, the ball is at a specific position. If you record its position every second, you get a sequence of dots. Connect the dots smoothly, and you get a *trajectory* — the path the ball took.

Now imagine instead of a ball on a table (2D), you have a point moving through a space with 3,072 dimensions. That's what a "flow" or "trajectory" is in this paper: the sequence of representations the model produces as it reasons through a problem, step by step, connected into a smooth curve.

**The critical detail:** The paper uses "context-cumulative" representations. This means at step t, the representation is NOT just the embedding of step t's text, but the embedding of *everything the model has read so far* (the problem + all reasoning steps up to step t). This is like tracking where the ball is after rolling for 1 second, 2 seconds, 3 seconds... not where it was pushed at each second individually.

### What is "velocity" in this context?

In physics, velocity is how fast something is moving and in what direction. If a car moves from position x₁ to position x₂ in one second, its velocity is approximately (x₂ - x₁).

Here, "velocity" means the same thing but in the model's high-dimensional space:

```
velocity at step t = y_{t+1} - y_t
```

where y_t is the representation at step t. This tells you *how much and in what direction* the representation changed from one reasoning step to the next.

### What is "curvature" (specifically Menger curvature)?

Imagine driving a car. On a straight road, the curvature is zero — you're not turning. On a sharp bend, the curvature is high — you're turning a lot. On a gentle curve, the curvature is moderate.

**Menger curvature** is a specific way to measure curvature using three consecutive points. Given three points (like three consecutive positions of a ball), you can find a unique circle that passes through all three. If the points are nearly in a straight line, this circle is very large (large radius = low curvature). If the points make a sharp turn, the circle is small (small radius = high curvature).

Mathematically:
```
Menger curvature = 1 / R
```
where R is the radius of the circle through the three points.

The paper's formula is:
```
curvature at step t = (2 × √(1 - cos²(angle between consecutive steps))) / (distance from y_{t-1} to y_{t+1})
```

This captures *both* how sharply the flow turns *and* how spread out the points are.

**Why Menger curvature and not just the angle?** The paper explicitly discusses this (Remark C.9, Figure 7): if two flows have the same turning angle at a point but different step sizes, cosine similarity would say they're identical, but Menger curvature would correctly distinguish them. Menger curvature captures both angular change AND scale, which is a strength for analyzing reasoning steps of different "sizes."

### What is "formal logic" and "natural deduction"?

Formal logic is the study of the *structure* of arguments, stripped of their content. Consider:

- "If it rains, the ground is wet. It rained. Therefore the ground is wet."
- "If you study, you pass the exam. You studied. Therefore you passed the exam."

These two arguments have different content (weather vs. studying) but identical logical structure:
```
A → B    (If A then B)
A         (A is true)
∴ B       (Therefore B is true)
```

A **natural deduction system** is a formal framework for building valid logical proofs using specific rules. Each logical symbol (→, ∧, ∨, ¬) has introduction and elimination rules. For example, if you know A is true and B is true, you can conclude A ∧ B (A and B) — that's the ∧-introduction rule.

The paper uses these formal logical structures as the "skeleton" of reasoning, and then dresses them up in different topics and languages to see if the model treats the logic separately from the content.

### What is the "Linear Representation Hypothesis" and "Concept Manifold"?

The **Linear Representation Hypothesis** says that concepts in an LLM (like "dog," "happy," "France") correspond to specific directions in the high-dimensional representation space. If you move in the "happiness" direction, representations become happier. If you move in the "France" direction, representations become more France-related.

A **concept manifold** (from the Modell et al. paper [46]) extends this: instead of single directions, some concepts occupy multi-dimensional surfaces (manifolds) in the representation space. For example, "color" might be a 3-dimensional surface, with different colors at different positions on the surface.

The paper here builds on both ideas: it uses the multidimensional linear representation hypothesis (Hypothesis 4.4) as a starting assumption, and then studies how *sequences* of representations (trajectories) behave on these manifolds.

---

## The Core Framework, Step by Step

### Step 1: Defining the Spaces (Section 4)

The paper introduces four key spaces and the maps between them:

1. **Input space X**: The raw text (tokens/sentences). Discrete and finite.

2. **Concept space C**: An abstract "human-level" cognitive space where ideas live. Think of this as the space of all possible thoughts and meanings. It's assumed to be smooth and continuous. (This is a philosophical assumption, inspired by William James's "stream of consciousness.")

3. **Representation space R**: The model's embedding space (R^d, where d might be 1,024 or 3,072). This is where we can actually measure things.

4. **Logic space L**: Two versions:
   - L_form: The space of formal logical structures (abstract proof skeletons)
   - L_rep: The space of representation increments (Δy_t = y_{t+1} - y_t)

The key maps:
- **Γ**: Maps input text to curves in concept space (what the text "means" at a human level)
- **Ψ**: Maps input text to curves in representation space (what the model computes)
- **A = Ψ ∘ Γ⁻¹**: The "canonical alignment" that connects concept curves to representation curves
- **F_C**: Maps concept curves to formal logic (extracts the logical structure of a thought)
- **D_R**: Maps representation curves to increments (extracts the velocity of the flow)

The central question (Figure 1c, the dashed arrow with "?"): Is there a meaningful correspondence between L_form and L_rep? That is, does the model's internal representation of reasoning dynamics (velocities, curvatures) reflect the formal logical structure?

**My critical take on Step 1:**

This is an ambitious framework but several things concern me:

1. **The concept space C is undefined.** The paper says it's "an abstract semantic space that models human-level cognitive structures" (Definition 4.1). But what does that mean mathematically? It's assumed to have "smooth geometric structure" but this is never justified. This is not a mathematical definition — it's a philosophical aspiration wearing mathematical notation.

2. **The map Γ (input to concept space) is never constructed.** Since C is abstract and undefined, Γ is also abstract and undefined. It's a placeholder that sounds rigorous but isn't. The paper never actually uses Γ in any computation — all computations happen in R (representation space). So what work is C and Γ doing? They're giving the paper a philosophical flavor but adding no mathematical content.

3. **The "canonical alignment" A requires Γ to be injective** (one-to-one). This means every different input text must map to a different concept trajectory. But paraphrases — different texts with the same meaning — would violate this! "The cat sat on the mat" and "On the mat, a cat was sitting" have the same meaning but are different inputs. The paper acknowledges this needs to hold "on the domain of interest" but this is a very strong assumption that's never verified.

4. **Why these four spaces?** The paper never argues that this is the *right* decomposition. Why not five spaces? Why not three? The input → concept → representation path is philosophically appealing but could be replaced by simpler constructs.

### Step 2: The Context-Cumulative Flow (Section 4.2, Algorithm 1)

This is the most concrete and useful definition in the paper. Here's exactly what they do:

1. Start with a problem prompt P
2. The model generates reasoning steps x₁, x₂, ..., x_T
3. At each step t, form the full context S_t = (P, x₁, ..., x_t) — everything the model has seen so far
4. Feed S_t through the model and extract the hidden state y_t from the final layer
5. The sequence Y = (y₁, ..., y_T) is the "reasoning trajectory" or "context-cumulative flow"

The crucial detail: They use **mean pooling** over the tokens of each step. That is, for step t, which spans multiple tokens, they average the hidden states of all those tokens. This gives one vector per reasoning step.

**My critical take on Step 2:**

1. **Mean pooling is a significant choice that's never justified.** Why average the hidden states? Why not use the last token (which is what autoregressive models use for prediction)? Why not the first token? Different pooling strategies could give very different trajectories. The paper doesn't show that their results are robust to this choice.

2. **The granularity of "reasoning steps" is ambiguous.** What counts as one "reasoning step"? A sentence? A paragraph? A logical inference? The paper says they "split by steps" using their dataset, where each step is labeled. But for natural CoT reasoning, how do you segment? This ambiguity could significantly affect the trajectories.

3. **Context-cumulative vs. isolated embeddings.** The paper argues (correctly, I think) that context-cumulative embeddings are more informative than isolated step embeddings. They show that isolated embeddings look noisy while cumulative ones look smooth. But this smoothness might be *trivially expected*: since S_t and S_{t+1} differ by only one additional step (a small change to a long context), of course their embeddings are similar! The smoothness might just reflect the *mechanical fact* that adding one step to a long context doesn't change the embedding much. This is not the same as "reasoning is a smooth flow."

### Step 3: The Smooth Trajectory Hypothesis (Hypothesis 4.6)

The paper claims (as a hypothesis, not a theorem) that the discrete sequence of representations y₁, y₂, ..., y_T lies on a smooth (C¹, meaning continuously differentiable) curve in R^d.

To support this, they construct a specific smooth interpolation using a "relaxed prefix mask" (Appendix C.1). The idea: instead of abruptly including token i (mask = 0 or 1), they smoothly ramp up its inclusion using a smooth step function g. Since the neural network Φ is smooth (assuming smooth activations like GELU/SiLU), the composite map is also smooth.

**My critical take on Step 3:**

1. **The smoothness is *constructed*, not *discovered*.** The paper shows that you CAN build a smooth curve through the points by choosing the right interpolation. But this doesn't mean the MODEL'S COMPUTATION is smooth! Any finite set of points can be interpolated by a smooth curve (this is a basic result in approximation theory). The existence of a smooth interpolation tells us almost nothing about whether the model's reasoning is inherently smooth.

2. **The relaxed mask mechanism is artificial.** Real LLMs use hard attention masks (token is either visible or not). The relaxed mask (where tokens are fractionally included) doesn't correspond to any actual computation the model performs. It's a mathematical convenience, not a description of reality.

3. **The ReLU problem is acknowledged but hand-waved away.** The paper notes (Remark C.5) that ReLU activations make the network only piecewise smooth, not globally smooth. They say "this does not affect the manifold-level geometric reasoning" and "we idealize Φ as smooth throughout." But this is exactly the kind of assumption that needs justification, not idealization!

4. **This is the weakest part of the framework.** The paper's central concept — that reasoning IS a smooth flow — is supported by a construction that any set of points admits. The hypothesis is unfalsifiable as stated: you can always find a smooth curve through finitely many points. The question should be: does the manifold hypothesis predict something that a "random points connected by a smooth curve" hypothesis doesn't?

### Step 4: Logic as Differential Constraints (Section 4.3)

This is the conceptual heart of the paper. The key proposition is:

**Proposition 4.10 (Logic as Integrated Thought):** The change in representation from step t to step t+1 equals the integral of the velocity:

```
∫_{s_t}^{s_{t+1}} v(s) ds = y_{t+1} - y_t = Δy_{t+1}
```

This is just the fundamental theorem of calculus applied to the trajectory! The paper then interprets this as: "each representation–logic step is the integration of local semantic velocity."

The central claim: **Logic acts as the controller of semantic velocity** — it governs both the magnitude and direction of how the representation changes at each step.

The prediction: Reasoning flows with the same logical skeleton but different semantic carriers should have:
- **Different positions** (because the topic determines where in representation space you are)
- **Similar velocities** (because the logic determines how the flow moves)
- **Similar curvatures** (because the logic determines how the flow turns)

**My critical take on Step 4:**

1. **Proposition 4.10 is trivially true.** It's literally the fundamental theorem of calculus. There is no content here beyond "the change in position equals the integral of velocity." This is true for ANY smooth curve, regardless of whether logic has anything to do with it. The paper dresses up a tautology in interpretive language ("logic as integrated thought") to make it seem profound. But the math is saying nothing about logic — it's saying something about smooth curves.

2. **The "logic as controller" metaphor is evocative but untested at this point.** The paper says logic "governs" the velocity. But the proposition doesn't establish any causal relationship between logic and velocity. It just defines velocity as the derivative of the trajectory. The actual test of whether logic controls velocity comes in the experiments (Section 6), not from this proposition.

3. **The curvature prediction is more interesting.** The paper argues that flows with the same logical skeleton should have similar curvatures, even across topics and languages. This is because curvature is a second-order property (it depends on how velocity changes), and translations/rotations of a curve (which change position) don't change curvature. This is geometrically sound reasoning, and is the most testable prediction of the framework.

4. **But the argument has a gap.** The paper says that changing the topic might correspond to a translation or rotation of the flow in representation space. If so, curvature would indeed be invariant. But why should changing the topic correspond to a rigid motion (translation + rotation)? What if changing the topic corresponds to a more complex transformation like stretching or warping? Then curvature would change. The paper doesn't justify why the transformation should be rigid.

### Step 5: The Dataset (Section 5)

The paper creates a carefully designed dataset to test their framework:

1. **30 distinct logical structures**, each with 8-16 reasoning steps
2. Each logical structure instantiated across **20 topics** (weather, finance, education, sports, etc.)
3. Each instantiation in **4 languages** (English, Chinese, German, Japanese)
4. Total: 2,430 reasoning sequences
5. Generated using **GPT-5** (!) in a two-stage pipeline:
   - Stage 1: Create abstract logical templates (symbolic form)
   - Stage 2: Rewrite each template in specific topics and languages

Example (from Table 2 in Appendix):
- Abstract logic: [1] A → B, [2] B → C, [3] ∀x(H(x) → J(x)), [4] H(a), [5] A, [6] B (from [1],[5]), [7] C (from [2],[6]), [8] J(a) (from [3],[4]), [9] C ∧ J(a) (from [7],[8])
- Weather: "If moisture converges over the city, then thunderclouds develop..."
- Finance: "If the firm's interest coverage ratio exceeds 3.0x, then the firm is deemed able to meet interest obligations..."

**My critical take on the dataset:**

1. **Using GPT-5 to generate the data introduces significant concerns.** The data was generated by an LLM, and then the experiments test how LLMs process this data. There's a circularity risk: LLM-generated text might have systematic properties (repetitive structures, specific phrasings, predictable patterns) that don't exist in natural reasoning. The paper doesn't acknowledge or address this circularity.

2. **The "logical structure" might not be preserved in the rewriting.** When GPT-5 rewrites an abstract template into "weather" language, does it perfectly preserve the logical structure? Or does the rewriting introduce subtle logical differences? The paper doesn't validate this. There's no human evaluation checking that the rewritten reasoning sequences are logically identical to the templates.

3. **The dataset is small by ML standards.** 2,430 sequences across 30 logic patterns, 20 topics, and 4 languages means roughly 2 sequences per (logic, topic, language) triple. This is very thin. The statistical power of any conclusions drawn from this data is questionable.

4. **The logical structures are simple.** The examples show straightforward propositional logic (A→B, B→C, therefore A→C) and simple universal instantiation (∀x H(x)→J(x), H(a), therefore J(a)). These are the most basic building blocks of logic. Would the findings extend to more complex reasoning patterns? Nested quantifiers? Proof by contradiction? Mathematical induction? The paper doesn't test this.

5. **30 logical structures is not many.** With only 30 different logical skeletons, there are at most 30×30 = 900 pairwise comparisons. How do you know you're not just seeing noise? The statistical analysis (next section) needs to be robust.

6. **Natural language reasoning doesn't look like this.** Real chain-of-thought reasoning is messy, with backtracking, heuristics, analogies, and implicit steps. This dataset has clean, step-by-step proofs with explicit justifications. Testing on this dataset tells us about the model's behavior on formal logical reasoning, not about "reasoning in general."

### Step 6: The Experiments (Section 6)

**Setup:**
- Models: Qwen3 0.6B, 1.7B, 4B, and LLaMA3 8B
- Hidden states extracted from the final transformer layer (before the LM head)
- Mean pooling over tokens for each reasoning step
- Three similarity metrics computed:
  - **Position similarity**: Mean cosine similarity between embedding vectors y_t
  - **Velocity similarity**: Mean cosine similarity between velocity vectors Δy_t
  - **Curvature similarity**: Pearson correlation between Menger curvature sequences

**Results (Table 1):**

| | Position Sim | Velocity Sim | Curvature Sim |
|---|---|---|---|
| **Same Logic** | 0.26-0.44 | 0.15-0.19 | 0.46-0.58 |
| **Same Topic** | 0.30-0.46 | 0.06-0.08 | 0.11-0.13 |
| **Same Language** | 0.74-0.89 | 0.07-0.09 | 0.13-0.17 |

The key finding: Position similarity is dominated by language (0.74-0.89 for same language vs. 0.26-0.44 for same logic). But velocity similarity and curvature similarity are dominated by logic (0.15-0.19 and 0.46-0.58 for same logic vs. much lower for same topic/language).

**Visualization (Figure 2):** Block matrices for 5 logic templates × multiple topics/languages on Qwen3 0.6B:
- Position: Diagonal blocks (= same topic) are high similarity, off-diagonal are mixed
- Velocity: On-diagonal blocks within each logic template are high, off-diagonal are low
- Curvature: Even cleaner separation by logic template; logics B and C happen to be similar

**My critical take on the experiments:**

1. **The absolute numbers for velocity similarity are very low.** The "same logic" velocity similarity is 0.15-0.19. This means that even flows with the *same* logical skeleton have velocity cosine similarity of only ~0.17. Yes, this is higher than the 0.06-0.08 for same topic, but 0.17 is still very low in absolute terms! If logic truly "controls" velocity, shouldn't flows with identical logic have much higher velocity similarity?

   To put this in perspective: a cosine similarity of 0.17 means the velocity vectors are nearly orthogonal (cos 80° ≈ 0.17). The paper's claim is essentially: "two nearly perpendicular vectors are still more similar than two even more nearly perpendicular vectors." This is technically true but doesn't support the strong claim that "logic governs velocity."

2. **The curvature numbers are more convincing but still moderate.** Same-logic curvature correlation of 0.46-0.58 is genuinely above the 0.11-0.17 for same language. A Pearson correlation of 0.53 means that about 28% of the variance in curvature is explained by shared logical structure. That's meaningful but hardly overwhelming. 72% of the variance is *not* explained by logic.

3. **Language dominates position similarity to a concerning degree.** Same-language position similarity is 0.74-0.89, while same-logic is only 0.26-0.44. This means the model's representations are overwhelmingly organized by language, not by logical or topical content. This is expected (language is a massive surface feature), but it means the paper's framework (concept space → representation space alignment) is working in a space where the dominant signal is the least interesting one (language identity).

4. **No statistical significance testing is reported.** With only 30 logical structures, 20 topics, and 4 languages, the sample sizes for computing these means are modest. The paper reports no confidence intervals, no p-values, no effect sizes beyond the raw means. How do we know these differences aren't due to chance? This is a significant omission for a paper claiming empirical validation.

5. **The block matrices (Figure 2) look good but could be misleading.** The curvature similarity matrix shows clean block-diagonal structure for logic templates. But this visualization is on a *selected subset* of the data (5 logic templates on one model). Cherry-picking a clean example to visualize is not the same as showing the pattern holds broadly.

6. **The choice of models is interesting but limited.** They test on Qwen3 (0.6B, 1.7B, 4B) and LLaMA3 (8B). These are relatively small models. The smallest (0.6B) shows the *highest* curvature similarity (0.53), while the 1.7B shows the *lowest* (0.46). This non-monotonic pattern is puzzling — if logic is a fundamental organizing principle, why would a bigger model show *less* logical structure in curvature? The paper doesn't address this.

7. **They don't test any reasoning models.** The models tested are base or instruction-tuned models, not specialized reasoning models (like DeepSeek-R1 or models trained with RLHF on math). If the framework is about "reasoning as flows," testing on models that are specifically good at reasoning would be much more compelling.

8. **The experiment doesn't actually test the "flow" hypothesis.** The paper claims reasoning is a "smooth flow" (Hypothesis 4.6), but the experiment only tests whether velocity and curvature similarities are higher for same-logic than same-topic comparisons. You don't need the smoothness hypothesis for this! The same result could hold for discrete representations without any smooth flow interpretation. The smoothness claim is never actually tested or needed for the results.

### Step 7: Discussion and Broader Claims

The paper makes several broader claims:

- "LLMs are not mere stochastic parrots" because they "acquire [logic] emergently from large-scale data."
- The geometric framework provides "formal tools for studying reasoning phenomenon."
- The flow perspective contrasts with and is superior to the "graph perspective" on reasoning.
- Practical implications include trajectory-level control for steering, alignment, and safety.

**My critical take on the broader claims:**

1. **The "not stochastic parrots" claim is overblown.** The paper shows that models represent logical structure in their hidden states (which is interesting). But this doesn't refute the "stochastic parrot" critique, which is about whether models *understand* what they're computing, not about whether they have structured representations. A lookup table also has structured representations but doesn't "understand" anything.

2. **"Formal tools" is a stretch.** The paper provides definitions and framework, but the definitions (velocity = difference, curvature = Menger curvature) are not novel mathematical contributions. They're applications of existing concepts from differential geometry. The framework is a vocabulary, not a tool.

3. **The contrast with graph perspectives is reasonable.** The paper correctly notes that graph-based views of reasoning (treating CoT as a random walk on a graph) miss the smooth, directed structure they observe. This is a fair point. But the paper doesn't actually show that the flow perspective predicts anything that the graph perspective doesn't.

4. **The practical implications are speculative.** "Trajectory-level control for steering, alignment, and safety" is a wish, not a demonstrated capability. The paper provides no evidence that understanding reasoning flows enables better control or alignment.

---

## Devil's Advocate — The Five Weakest Points

### Weakness 1: The Smoothness Hypothesis is Trivially True and Does No Work

Any finite set of points can be connected by a smooth curve. The paper shows this (Proposition C.4) and calls it evidence for smooth reasoning. But this tells us nothing about the model. The hypothesis is unfalsifiable: for *any* model behavior, you can find a smooth interpolation. The real question — whether the smoothness reflects genuine structure in the model's computation — is never addressed.

### Weakness 2: The Velocity Similarity Numbers Are Barely Above Chance

Same-logic velocity cosine similarity of 0.15-0.19 is technically higher than same-topic (0.06-0.08), but these numbers are all close to zero. In a high-dimensional space (d = 1,024 or more), random vectors have expected cosine similarity near 0. A similarity of 0.17 is barely distinguishable from random. The paper needs to show that 0.17 is meaningfully above the null distribution, not just above the other conditions.

### Weakness 3: The Abstract Concept Space C Does No Mathematical Work

The concept space C is never defined mathematically, never used in computations, and never tested. All actual work happens in R (representation space). The entire concept-space formalism (Definitions 4.1-4.3, the canonical alignment A) could be removed without changing any result. It adds philosophical gravitas but zero mathematical content.

### Weakness 4: The Dataset Has Circular Design

Using GPT-5 to generate reasoning data, then testing LLMs on that data, introduces potential circularity. LLM-generated formal logic might have stylistic regularities that explain the observed similarities better than "the models have internalized logic." The paper needs to show the same results hold for human-written reasoning, or at least for reasoning generated by very different models.

### Weakness 5: Context-Cumulative Smoothness Might Be a Trivial Artifact

Because S_t and S_{t+1} differ by only one reasoning step added to a long context, of course their embeddings are similar. This "smoothness" is a mathematical consequence of how language model embeddings work (most of the context is shared), not evidence of meaningful geometric structure. A random sequence of tokens added step by step would also produce smooth trajectories under this scheme.

---

---

# PASS 3 — The Swamp (Deep Dive into Proofs, Logic, and Connections)

## Proof Walkthrough: Continuity of the Relaxed-Mask Trajectory (Proposition C.4)

This is the paper's main theoretical result — that the representation trajectory is C¹ (continuously differentiable).

### The Construction

**Step 1:** Fix the full token stream U₁:N of length N (the entire reasoning sequence concatenated).

**Step 2:** Introduce a continuous progress parameter s ∈ [0, 1]. At progress s, define a "relaxed mask" m_s(i) = g(sN - i), where g is a smooth step function:
- g(x) = 0 for x ≤ -δ (token fully excluded)
- g(x) = 1 for x ≥ δ (token fully included)
- g is C∞ (infinitely differentiable) everywhere
- δ < 1/2 (the transition zone is narrow)

**Step 3:** The masked input at progress s is z_s(i) = m_s(i) × E(U_i), where E is the token embedding function. Similarly for positional encodings.

**Step 4:** The trajectory is Ψ̃(s) = Φ(z_s, ι_s, M_s), where Φ is the full neural network encoder.

**Step 5:** Since g is C∞, and Φ is smooth (assuming smooth activations), the composition is C¹.

**Step 6:** At sentence boundaries s_t = N_t/N, the mask is exactly 0 or 1 (because δ < 1/2), so the relaxed trajectory passes exactly through the discrete points.

### My Critical Analysis of the Proof

1. **The proof is technically correct** (given the assumptions), but it proves something almost trivial. It says: "if you smooth out the mask, and the neural network is smooth, then the resulting trajectory is smooth." This is just saying "a smooth function of smooth inputs is smooth." The chain rule does the work.

2. **The key assumption — Φ is C¹ — is the elephant in the room.** Modern transformers use:
   - **Layer normalization:** Contains division by standard deviation, which is smooth as long as the standard deviation is nonzero (almost always true in practice)
   - **Softmax attention:** Smooth, no problems
   - **Activation functions:** GELU and SiLU are smooth; ReLU is NOT smooth (it has a kink at 0)
   - **The mask itself:** The hard mask M_s is piecewise constant, not smooth. The proof handles this by noting M_s is *locally* constant (doesn't change near the sentence boundaries), so the trajectory is smooth *at* the sentence boundaries.

3. **The proof has a subtle issue with the mask discontinuity.** M_s(i) = 1{m_s(i)=1} changes discontinuously when m_s(i) crosses the threshold. The proof argues that M_s is locally constant "on neighborhoods that avoid the transition band |sN - i| < δ." This is true at the sentence boundaries, but the trajectory between sentence boundaries passes through the transition band, where the mask *is* changing. The proof doesn't actually show the trajectory is smooth *everywhere* on [0,1] — only that it's smooth on intervals where M_s is fixed, and smooth at the sentence boundaries.

4. **What does this construction buy us?** It shows we can find A smooth curve through the discrete points. But we already knew this — you can always fit a smooth curve through finitely many points (spline interpolation, for example). The specific construction via relaxed masks doesn't tell us anything special about LLMs. It tells us about the mathematical properties of smooth functions of smooth inputs.

5. **The paper says "many alternative constructions are possible" (Remark C.5).** This is exactly the problem! The smooth trajectory is not unique. Different constructions give different curves with different velocities and curvatures *between* the discrete points. So the geometric properties (velocity, curvature) at the discrete points depend on the interpolation scheme, which is arbitrary. The paper only computes these properties at the discrete points (using finite differences), so this non-uniqueness doesn't directly affect the experiments, but it undermines the claim that the "flow" is a well-defined mathematical object.

## Analysis of the Menger Curvature Framework

### The Formula (Proposition C.8)

For three consecutive points y_{t-1}, y_t, y_{t+1}:

```
κ_t = 2√(1 - cos²θ) / ||y_{t+1} - y_{t-1}||
```

where θ is the angle between vectors u = y_t - y_{t-1} and v = y_{t+1} - y_t.

This can be rewritten as:

```
κ_t = 2 sin θ / c
```

where c = ||y_{t+1} - y_{t-1}|| is the distance between the first and last points.

### Critical Analysis

1. **Why Menger curvature and not other curvature measures?** The paper argues (Remark C.9) that Menger curvature is better than cosine similarity because it incorporates both angle AND distance. This is valid. But there are other curvature measures that also do this — for example, the standard Frenet curvature κ = ||γ'' × γ'|| / ||γ'||³ for parametric curves. The paper doesn't compare Menger curvature to these alternatives.

2. **Menger curvature is scale-dependent.** If you scale all embeddings by a factor of 2 (multiply every y_t by 2), the Menger curvature halves (because c doubles but the numerator stays the same). This means curvature comparisons between models of different sizes (which have different embedding scales) are not directly meaningful without normalization. The paper doesn't discuss this.

3. **Three-point curvature is noisy.** Menger curvature uses only three consecutive points. In a high-dimensional space with noise, this can be very noisy. A windowed or smoothed curvature measure might be more robust. The paper doesn't investigate the sensitivity of their results to noise in the embeddings.

4. **The curvature computation requires at least 3 steps.** For reasoning chains with only a few steps, you get very few curvature values, making the Pearson correlation unreliable. With the 8-16 step chains in their dataset, you get 6-14 curvature values per chain. Computing Pearson correlation on 6-14 points is statistically shaky.

## Connections to the Gurnee et al. "When Models Manipulate Manifolds" Paper

The Gurnee et al. paper (from Anthropic, 2025) provides a spectacular case study of geometric representations in transformers. Several connections and contrasts with this paper are worth noting:

### Connection 1: Manifolds as Representation Structures

Both papers start from the premise that transformers represent information on geometric manifolds in their residual streams. Gurnee et al. find that character count is represented on a 1D manifold (helical curve) in a ~6D subspace. This paper hypothesizes that reasoning traces trajectories (curves) on concept manifolds.

**Critical difference:** Gurnee et al. *discover* the manifold from the data (unsupervised features → manifold) and then *validate* it with causal interventions (ablation, patching). This paper *assumes* manifold structure (Hypothesis 4.4, 4.6) and never validates it causally. The Gurnee et al. approach is much more rigorous because they show the manifold is *causally necessary* for the model's behavior, not just present in the representations.

### Connection 2: The "Rippled" Geometry and Curvature

Gurnee et al. show that the character count manifold has "rippled" geometry — a helix with ringing patterns in cosine similarity. They prove this rippling is *optimal* for packing information into low-dimensional spaces. The curvature of the helix has a well-understood mathematical origin (the tradeoff between capacity and distinguishability).

This paper uses Menger curvature to measure the "sharpness" of reasoning turns. But unlike Gurnee et al., where the curvature has a clear functional role (enabling boundary detection via QK rotation), this paper never explains *why* reasoning should have particular curvature patterns. The curvature is measured but not mechanistically explained.

### Connection 3: Discrete Features vs. Continuous Manifolds (The Duality)

Gurnee et al. beautifully demonstrate the duality between discrete features (from sparse autoencoders) and continuous manifolds. The discrete features "tile" the manifold, providing local coordinates. This duality gives two complementary views of the same object.

This paper doesn't engage with this duality at all. It works purely at the continuous level (trajectories, velocities, curvatures) without investigating what the discrete features look like along the reasoning trajectory. A natural question: do sparse autoencoder features along the reasoning flow show the same logical-structure-grouping that the continuous measures show?

### Connection 4: The Role of Attention in Geometric Computation

Gurnee et al. show that specific attention heads perform geometric operations on manifolds — the "boundary heads" rotate (twist) one manifold to align with another. This is a concrete mechanistic explanation of *how* the model computes with geometry.

This paper doesn't investigate which model components create the reasoning flows. It treats the model as a black box that produces hidden states. There's no analysis of which attention heads or MLP layers contribute to the velocity and curvature patterns. This is a significant missed opportunity: understanding *how* the flow is constructed would be much more informative than merely observing that it exists.

## Connection to the Hauberg Paper ("Only Bayes Should Learn a Manifold")

The Hauberg paper (analyzed in the project context) raises a fundamental question: can you trust the geometry of manifolds learned from data? The answer depends on whether you use deterministic or probabilistic methods.

This paper uses purely deterministic methods (PCA, cosine similarity, Pearson correlation) to analyze manifold geometry. Hauberg would argue that these methods may be biased, especially in regions of representation space with sparse data. For reasoning flows, the "sparse data" concern is relevant: some logical structures or topic combinations may have very few examples, and the geometry in those regions may not be reliable.

**The key question Hauberg raises for this paper:** Are the observed curvature similarities between same-logic flows a property of the *true* manifold geometry, or an artifact of the *deterministic* analysis methods? The paper doesn't address this question, and Hauberg's work suggests it should.

## Connection to the Sparse Mixtures of Linear Transforms (MOLT) Paper

The MOLT paper from Anthropic introduces an alternative to transcoders for understanding MLP computation. MOLTs learn sparsely active linear transforms that bridge representations between layers. This is relevant because:

1. MOLTs capture *transformations* (how representations change), which is exactly what this paper's "velocity" measures. A natural question: do MOLT transforms correspond to the velocity vectors of reasoning flows?

2. MOLTs are designed to handle *geometric* computation efficiently (the "plus 3 rotation on a circle" example). If reasoning involves geometric operations on manifolds, MOLTs might be the right tool for understanding those operations.

3. The MOLT paper notes that transcoders "shatter" computation into lookup-table features, while MOLTs capture the underlying geometry more faithfully. This is analogous to the paper's observation that position embeddings (0-order) cluster by topic/language, but velocity (1st-order) reveals logical structure — the "shattering" by topic at the position level is overcome by looking at the transformation (velocity).

## Connection to Activation Oracles

The Activation Oracle paper proposes training LLMs to answer questions about neural activations. A natural application: could an Activation Oracle be trained to identify the logical structure of a reasoning flow from its velocity/curvature signature? This would provide a more flexible and potentially more powerful way to test the paper's claims than the fixed similarity metrics used here.

---

## Section-by-Section Critical Analysis

### Section 1 (Introduction): Grade B
Clear motivation, nice visualization (Figure 1a-b), but oversells the contribution. The claims "first work to formalize and empirically validate such a dynamical perspective" and "quantitative evidence" are stronger than what the paper actually delivers. The visualization in Figure 1 is compelling but only for one example.

### Section 2 (Related Work): Grade B+
Comprehensive coverage of related areas. Well-organized by topic. Correctly positions the work at the intersection of concept geometry, mechanistic interpretability, and reasoning analysis. The claim that "our work employs formal logic not as an end task, but as a tool" is a clear and useful distinction.

### Section 3 (Preliminaries): Grade A-
Clear definitions. Definition 3.2 (Representation Operator) is well-formulated. The Menger curvature introduction is appropriately concise with details in the appendix.

### Section 4 (Core Framework): Grade C+
The weakest theoretical section. The concept space C and the map Γ are never grounded mathematically. Hypothesis 4.6 (smooth trajectory) is unfalsifiable. Proposition 4.10 is a tautology (fundamental theorem of calculus). The framework sounds impressive but lacks mathematical substance. The one genuinely useful contribution is Definition 4.5 (context-cumulative flow) and Algorithm 1.

### Section 5 (Dataset): Grade B
The dataset design is the paper's best methodological contribution. The idea of fixing logical structure while varying topics and languages is clever and well-executed. The use of formal natural deduction is appropriate. However, the dataset's reliance on GPT-5 generation and its small size weaken it.

### Section 6 (Experiments): Grade B-
Results are interesting but not overwhelming. The key finding (velocity and curvature similarity higher for same logic than same topic) is supported but with low absolute numbers. No statistical significance testing. Limited model diversity. The visualization (Figure 2) is effective for the specific example shown.

### Section 7 (Discussion): Grade C
Too brief. The "not stochastic parrots" claim is overblown. The practical implications are entirely speculative. The contrast with graph perspectives is fair but underdeveloped. Missing: honest discussion of limitations, failure cases, and what the framework can NOT do.

### Appendix C (Geometric Foundations): Grade B+
The relaxed-mask construction (C.1) is technically clean, even if the result is trivially expected. The Menger curvature development (C.2) is rigorous and well-presented. The proof of Proposition C.8 is correct and clearly written.

### Appendix D (Data Generation): Grade A-
Transparent disclosure of the exact prompts used for data generation. This is excellent for reproducibility. The example in Table 2 is illuminating and helps the reader understand exactly what the data looks like.

---

## What This Paper Gets Right

1. **The core empirical observation is interesting and likely real:** When you look at first-order differences (velocities) and second-order differences (curvatures) of reasoning trajectories, the logical structure emerges as a more important organizing principle than the surface semantics. This is a genuine and non-obvious finding, even if the absolute effect sizes are modest.

2. **The dataset design is clever:** Holding logic constant while varying topics and languages is a clean experimental design that allows causal attribution. This is the right way to test whether models represent logic separately from semantics.

3. **The use of Menger curvature is well-motivated:** As the paper explains (Remark C.9), Menger curvature captures both angular deviation and scale, which cosine similarity alone cannot. This is a genuinely useful tool for analyzing discrete trajectories in high-dimensional spaces.

4. **The paper is well-written and clearly presented:** The definitions are precise, the figures are informative, and the exposition flows logically. The paper is accessible to researchers from multiple backgrounds.

5. **The connection between differential geometry and reasoning is stimulating:** Even if the mathematical framework has weaknesses, the conceptual idea — that reasoning can be analyzed through the lens of differential geometry, with velocity and curvature as key descriptors — is thought-provoking and may inspire productive future work.

## What This Paper Gets Wrong or Oversells

1. **The theoretical framework is largely hollow.** The concept space C, the maps Γ and A, and Proposition 4.10 add no mathematical content. They create an impression of rigor that the actual results don't support. Stripping away this formalism, the paper's contribution is: "look at velocity and curvature of reasoning trajectories and compare them across conditions." This is a valid empirical contribution but shouldn't be dressed up as "establishing a geometric theory."

2. **The smoothness hypothesis is unfalsifiable and does no work.** Any finite set of points can be connected by a smooth curve. The paper's experiments only use finite differences (Δy_t), which don't require smoothness. The smooth flow interpretation adds no predictive power beyond what the discrete analysis provides.

3. **The absolute effect sizes are weak.** Velocity similarity of 0.17 for same-logic vs. 0.07 for same-topic is a relative difference, but both numbers are close to zero. The paper needs better statistical analysis to show these differences are meaningful.

4. **The paper lacks any mechanistic content.** It describes WHAT happens (flows have logic-correlated curvature) but never WHY (which model components create this pattern). Compare this to Gurnee et al., who not only describe the character count manifold but also show which attention heads construct it and how. This paper stays at the descriptive level.

5. **The conclusions overreach.** Claims like "effectively rediscovering in data the universal logic that took humans two millennia to formalize" and "a hallmark of genuine intelligence" are not supported by showing that velocity similarities are 0.17 instead of 0.07. These are extraordinary claims that require extraordinary evidence.

---

## Key Takeaways

1. **The idea is better than the execution.** Analyzing reasoning as geometric flows and using velocity/curvature as descriptors is a promising research direction. But the specific theoretical framework and experiments in this paper don't live up to the idea's potential.

2. **Velocity and curvature reveal structure that positions don't.** This is the paper's most durable empirical contribution. First-order and second-order differences of representation trajectories contain different information than the representations themselves, and this information is more closely related to logical structure.

3. **The Menger curvature tool is useful.** For anyone studying sequential processes in representation space (not just reasoning), Menger curvature is a well-motivated measure that captures both angular and scale information.

4. **The dataset design is a template for future work.** The approach of holding one factor constant while varying others (logic vs. topic vs. language) is a clean experimental design that could be applied to study many other aspects of LLM representations.

5. **Much more work is needed.** The paper opens a door but doesn't walk through it. The natural next steps would be: (a) mechanistic analysis of which model components create the flows, (b) causal interventions to test whether the flow geometry is necessary for correct reasoning, (c) testing on realistic (non-synthetic) reasoning tasks, (d) connecting to existing interpretability infrastructure (SAEs, attribution graphs, etc.).

---

## Open Questions from This Paper

1. **Is the smoothness real or trivial?** Does the context-cumulative trajectory have genuine smooth structure beyond what you'd expect from the mechanical fact that consecutive prefixes share most of their content?

2. **What creates the curvature?** Which attention heads or MLP layers are responsible for the curvature patterns? Do they correspond to identifiable logical operations?

3. **Would causal interventions validate the framework?** If you intervene on the velocity direction (push the flow in a different direction), does the model's reasoning change in predictable ways?

4. **Does this scale to complex reasoning?** The paper tests on simple propositional logic with 8-16 steps. What about 100-step mathematical proofs? Multi-hop reasoning? Reasoning with uncertainty?

5. **How does this relate to the "manifold manipulation" story?** Gurnee et al. show that transformers manipulate manifolds (rotate, twist, compose). Does reasoning involve specific manifold manipulations, and can we characterize them?

6. **What happens for wrong reasoning?** When the model makes a logical error, does the flow deviate from the expected curvature profile? Could curvature anomalies be used to detect reasoning errors?

7. **Is the velocity similarity really about logic, or about structural features of the text?** The same logical skeleton implies similar sentence structures (premises, intermediate conclusions, final conclusion). Could the velocity similarity be driven by these structural features rather than by genuine logical understanding?

8. **Why do smaller models show higher curvature similarity?** The 0.6B model has curvature similarity of 0.53, while the 1.7B has 0.46. This is opposite to what you'd expect if larger models have better logical understanding. What explains this?

---

*Analysis conducted following Ramdas's "How to Read Research Papers" framework (CMU, Statistics & ML) with practitioner-tested enhancements, applied at full Pass 1-2-3 depth.*