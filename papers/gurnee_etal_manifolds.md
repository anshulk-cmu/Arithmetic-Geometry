# Analysis: When Models Manipulate Manifolds — The Geometry of a Counting Task

**Paper:** When Models Manipulate Manifolds: The Geometry of a Counting Task
**Authors:** Wes Gurnee*, Emmanuel Ameisen*, Isaac Kauvar, Julius Tarng, Adam Pearce, Chris Olah, Joshua Batson*
**Affiliations:** Anthropic (all authors)
**Published:** October 21, 2025
**Venue:** Transformer Circuits Thread (transformer-circuits.pub) — Anthropic's in-house interpretability research blog. Also archived on arXiv: 2601.04480 (January 2026).
**Model Studied:** Claude 3.5 Haiku
**Type:** Empirical / Mechanistic Interpretability case study
**Purpose:** Deep understanding — this paper is directly core to our geometric interpretability pipeline for Llama 3.1 8B. It provides the strongest existing evidence that language models learn and manipulate geometric manifolds for scalar quantities.
**Peer-Reviewed:** No. Published on Anthropic's blog, later archived on arXiv. However, Anthropic's interpretability team has strong internal review, and many reviewers are acknowledged (including external researchers like Patrick Rubin-Delanchy and Eric Michaud).

---

---

# PHASE -1: Paper Classification

This is an **Empirical / Mechanistic Interpretability** paper. It reverse-engineers a specific behavior (linebreaking in fixed-width text) in a production language model (Claude 3.5 Haiku), discovers geometric representations (manifolds) and algorithms (rotations via attention heads), and validates the findings through causal interventions and adversarial experiments ("visual illusions").

There are no formal theorems or proofs. The closest thing to theory is a simple physical simulation showing that "rippled" manifold embeddings are optimal when you pack many vectors into a small subspace. The rest is careful empirical work: probing, PCA, ablation, mean-patching, and feature analysis.

**Reading Strategy:** Visual-First Pass 1 (the figures carry the argument — this is an Anthropic blog post designed for visual communication), then standard Pass 2 for mechanism understanding, then Empirical Pass 3 for stress-testing claims and extracting techniques for our pipeline.

---

# PHASE 0: Pre-Reading Context

## Who Are the Authors?

**Wes Gurnee** is a recently-minted MIT PhD (February 2025) who worked with Dimitris Bertsimas, Max Tegmark, and Neel Nanda on interpretability. His publication record is outstanding for someone at his career stage: "Language Models Represent Space and Time" (ICLR 2024), "Finding Neurons in a Haystack" (TMLR), "Universal Neurons in GPT2" (TMLR), "Refusal is Mediated by a Single Direction" (NeurIPS 2024), and "Not All Language Model Features Are One-Dimensionally Linear" (ICLR 2025). He is now at Anthropic. Google Scholar shows 2,608 citations. He is one of the most prolific young researchers in mechanistic interpretability.

**Chris Olah** needs no introduction. He is the founder of the field of neural network interpretability, from the original Distill work on circuits in vision models through to the current Transformer Circuits Thread. His involvement gives this paper an implicit credibility stamp.

**Joshua Batson** is a senior researcher at Anthropic and the correspondence author. His background spans physics and computational biology before moving to ML interpretability.

**Emmanuel Ameisen** is a core contributor on the Circuit Tracing paper, which provides the attribution graph infrastructure this paper builds on.

The team collectively represents the core of Anthropic's interpretability research group. This is not a peripheral or student-led project. It is produced by the lab that arguably sets the standard for mechanistic interpretability.

## Where Is It Published?

Transformer Circuits Thread is not a peer-reviewed venue in the traditional sense. It is Anthropic's in-house interpretability research blog. However, this blog has published some of the most influential papers in the field: "A Mathematical Framework for Transformer Circuits," "Toy Models of Superposition," "Scaling Monosemanticity." These papers have shaped the entire field's research agenda despite not going through traditional peer review.

The paper was also archived on arXiv (2601.04480) in January 2026, making it citable in academic contexts.

**Credibility assessment for non-peer-reviewed work:**

- **Author track record:** Exceptional. Multiple NeurIPS, ICLR, and TMLR publications.
- **Institutional affiliation:** Anthropic — the leading lab for interpretability research, with strong internal review.
- **Reproducibility signals:** The paper uses a 10M feature crosscoder dictionary trained on Claude 3.5 Haiku. Since Claude is a proprietary model, external reproduction would require training similar dictionaries on open models. The paper's key contribution (geometric structure of counting representations) has conceptual analogs that could be tested on open models, but exact reproduction is limited. No code is released.
- **What is NOT disclosed:** The exact architecture of Claude 3.5 Haiku, training data composition, training hyperparameters. The crosscoder dictionary details are partially described but not fully specified.
- **Independent verification:** The arXiv version appeared in January 2026, and as of April 2026, there is already follow-up discussion in the community.
- **Conflict of interest:** This paper studies an Anthropic model using Anthropic's tools, published on Anthropic's blog. The team has an incentive to show that their interpretability tools reveal interesting structure. However, the analysis is thorough enough and the causal validations strong enough to substantially mitigate this concern.

## Social Discussion and Reception

The paper was published on October 21, 2025 on Transformer Circuits. By January 2026, an archival version appeared on arXiv. Community reception has been very positive. The paper has been discussed on Twitter/X, in interpretability research groups, and on various ML blogs. The emergentmind.com platform created a dedicated topic page for "character count manifolds," summarizing the paper's contributions and their implications for geometric interpretability.

Several aspects of the reception are worth noting:

1. The "visual illusions" framing captured significant public attention. The idea that language models can experience something analogous to optical illusions is immediately accessible and memorable.

2. The feature-manifold duality concept has influenced how the broader interpretability community thinks about the relationship between SAE features and continuous representations. The "complexity tax" framing has been adopted by other researchers.

3. The connection to neuroscience (place cells, boundary cells) resonated with researchers working at the intersection of computational neuroscience and AI interpretability.

4. Critics have noted the limited scope (one model, one task) and the proprietary nature of the model. External reproduction is challenging because Claude 3.5 Haiku is not open-weights.

## Why Are We Reading This?

Five reasons, in order of importance:

1. **This is the strongest existing evidence for feature manifolds in production LLMs.** The paper does not just claim manifolds exist — it shows how the model constructs them, manipulates them via attention head rotations, and uses them for computation. If anything remotely similar exists in Llama 3.1 8B's arithmetic representations, we need to understand this paper's methodology in detail.

2. **The "rippled representations are optimal" result directly relates to our Fourier screening.** The paper shows that when you project a circle's worth of vectors into a low-dimensional subspace, you get "ringing" — a rippled pattern with Fourier structure. This is exactly the structure we are looking for in our digit and carry representations.

3. **The feature-manifold duality is a key conceptual tool.** The paper shows that the same representation can be described as a collection of discrete sparse features OR as a continuous manifold. This duality is directly relevant to understanding the relationship between SAE features and the geometric structures we are looking for.

4. **The causal validation methodology is something we need to adopt.** The paper demonstrates two kinds of causal experiments — subspace ablation and mean-patching — that we should use for our own causal patching phase.

5. **The "boundary head twist" mechanism is the clearest existing example of computation-via-manifold-manipulation.** Understanding how this works helps us know what to look for in arithmetic circuits.

---

---

# PASS 1 — Jigsaw Puzzle (What Does the Paper Do?)

## Q1: What Is the Problem Being Solved?

The paper studies how Claude 3.5 Haiku predicts when to insert a newline in fixed-width text. This sounds mundane, but it is actually a rich perceptual task. The model receives a sequence of token IDs (integers), but linebreaking requires knowing how many *characters* are on the current line — information that is nowhere in the input because tokens have variable character lengths ("a" is 1 character, "aluminum" is 8). The model must:

1. Figure out how many characters each token has
2. Accumulate a running count of characters since the last newline
3. Know the line width constraint (from context)
4. Compare the running count to the line width
5. Check if the next predicted word would fit
6. Decide whether to output a newline or the next word

This is computationally nontrivial. The model has to build what amounts to a spatial representation of its position in a line, entirely from token IDs.

**In one sentence:** This paper studies how Claude 3.5 Haiku builds internal representations of character position within a line and uses those representations to decide when to break a line in fixed-width text.

## Q2: Why Is This Interesting and Nontrivial?

Three reasons make this scientifically valuable:

**First, it tests whether manifold representations exist in production models.** Prior work found circular and Fourier-like representations in tiny toy models trained from scratch (Nanda et al. on grokking, Bai et al. on multiplication, Li et al. on addition). But nobody had shown that a large, naturally pretrained model uses similar geometric structures for naturally occurring tasks. Claude 3.5 Haiku is a production model, not a toy. Linebreaking is a task it learned from pretraining data, not something imposed by researchers.

**Second, the task has ground truth.** Unlike reasoning or language understanding, character counting has a clean mathematical answer at every step. You can verify whether the model's internal count representation is correct. This makes the interpretability much more rigorous than typical studies of vaguely defined concepts like "honesty" or "emotion."

**Third, the discovery story is compelling.** The authors initially tried probing and patching but got confused. They only made progress after using unsupervised sparse features (crosscoders) to identify the relevant variables. The features then pointed toward a geometric interpretation that simplified the understanding dramatically. This is an important methodological lesson.

**In one sentence:** This is nontrivial because it shows, for the first time in a production model, that continuous scalar quantities are represented on geometric manifolds and that computation happens through geometric manipulation of those manifolds.

## Q3: What Is the Main Claim?

The paper makes several interconnected claims. Let me state them clearly:

**Claim 1 (Representation):** Character count in a line is represented on a 1-dimensional manifold (a curve) embedded in a 6-dimensional subspace of the residual stream. This manifold is "rippled" — it spirals through the subspace with high curvature. This rippling is optimal: it is the result of packing 150 distinguishable positions into only 6 dimensions while maintaining resolution between adjacent counts. The mathematical structure is connected to Fourier features: the eigendecomposition of the ideal similarity matrix yields Fourier modes, and truncating to the top k modes produces the observed ringing pattern.

**Claim 2 (Computation):** The model uses attention heads to manipulate these manifolds geometrically. Specifically:
- "Boundary heads" rotate (twist) the character count manifold to align it with the line width manifold, so that positions near the line boundary produce large dot products. This is comparison-via-rotation.
- Multiple boundary heads with different offsets work together in a "stereoscopic" manner to produce a high-resolution estimate of "characters remaining."
- Token character lengths and characters remaining are arranged on near-orthogonal subspaces, making the final newline decision linearly separable (a simple hyperplane separates "break the line" from "continue the line").

**Claim 3 (Construction):** The character count manifold is built cooperatively by multiple attention heads across layers 0 and 1. Each head contributes a low-rank component (approximately a ray in PCA space), and their sum produces the full curved manifold. No single head can generate sufficient curvature alone.

**Claim 4 (Duality):** The same representation can be described either as a set of discrete sparse features (analogous to "place cells" in neuroscience) or as a continuous manifold. The features "tile" the manifold in a canonical way, providing local coordinates. This duality is not just descriptive — it has practical consequences for how we interpret circuits.

**Claim 5 (Validation):** Causal interventions confirm that these representations are functional, not just correlational. Ablating the 6D character count subspace destroys newline prediction. Mean-patching to a specific character count shifts the model's behavior in the predicted direction. Inserting specific character sequences ("@@") disrupts the counting attention heads in predictable ways, creating "visual illusions" — analogous to optical illusions in human perception.

**In one sentence:** Claude 3.5 Haiku represents scalar quantities like character count on rippled 1D manifolds in low-dimensional subspaces, and performs computation by geometrically manipulating these manifolds through attention head rotations, producing an elegant and causally validated algorithm for linebreaking.

---

---

# PASS 2 — Scuba Dive (How Does It Work and What Does It Mean?)

## Q1: What Was the Main Technical Hurdle? How Does This Paper Overcome It?

**The barrier before this paper** was twofold:

First, prior evidence for geometric manifold representations in transformers came almost exclusively from tiny models trained from scratch on simple tasks. Nanda et al. (2023) found Fourier representations in 1-layer models trained on modular addition. Bai et al. (2025) found pentagonal prism geometry in 2-layer models trained on multiplication. Li et al. (2025) found trigonometric features in small models trained on addition. The fundamental open question was: do production-scale models use similar structures? There was no evidence either way.

Second, interpreting the internal computation of large models had been limited to either linear probes (which compress everything to 1D and miss manifold structure) or sparse feature dictionaries (which fragment the computation into many small pieces). Neither approach captured the continuous geometric structure that the computation might live on.

**The key insight that unlocks progress** is the idea of *feature-manifold duality*. The authors use sparse features (from crosscoders) as an entry point to discover which variables matter and where they are computed. But then they go beyond the features to find the underlying continuous geometry. The features become local coordinates on a manifold, not the fundamental objects of interest. This two-step process — unsupervised discovery via features, then geometric characterization via PCA and probing — is the methodological contribution.

A second technical insight is the connection between rippled manifold embeddings and Fourier analysis. The authors show that when you try to pack N distinguishable positions into a k-dimensional subspace (k much less than N), the optimal embedding produces a curve that ripples through the subspace. The rippling comes from truncating the Fourier expansion of the ideal similarity matrix. This is not just a cute mathematical observation; it explains why the representations look the way they do and predicts properties (like ringing in probe cosine similarities) that can be verified empirically.

## Q2: What Is the Simplest Nontrivial Baseline? By What Metric Is the New Method Better?

The simplest baseline is a 1-dimensional linear probe. A linear regression probe on the residual stream after layer 1 achieves R-squared of 0.985 for predicting character count. This is a strong baseline — the model does encode character count linearly, to a good approximation.

So what does the geometric analysis add? Several things:

**Resolution.** The linear probe has an RMSE of 5, meaning it is off by about 5 characters on average. The manifold representation, with its 6 dimensions of curvature, provides finer discrimination between adjacent character counts. This resolution matters because the linebreaking decision depends on distinguishing count=42 from count=43 when the line width is 50 and the next word is 8 characters.

**Mechanism.** The linear probe tells you that character count is encoded. It does not tell you how that encoding is used. The manifold analysis reveals the boundary head mechanism (comparison via rotation), the distributed counting algorithm (multiple heads summing their outputs), and the orthogonal arrangement of character-remaining and next-token-length representations. None of this is visible from a linear probe.

**Parsimony.** The sparse feature view identifies 10 character count features, 3+ boundary head features, break predictor features, break suppressor features, and dozens of attribution graph edges connecting them. The manifold view says: "It is a curve in 6D. Boundary heads rotate it. The decision is a hyperplane." Same phenomenon, dramatically simpler description. The paper calls this the "complexity tax" — features give a true description, but the manifold pays down the complexity.

The paper quantifies the manifold view's predictive power: a 3D separating hyperplane (from PCA of average embeddings for character-remaining and next-token-length) achieves AUC of 0.91 on predicting newlines from real data. This is not perfect, but it is remarkable for such a simple geometric classifier.

## Q3: What Is Still Open and Where Does Their Insight Break Down?

The paper is honest about many open questions, which I will list and evaluate:

**Open Question 1: How does the model estimate line width?** The paper shows that line width is represented on its own manifold (similar to character count), and that the boundary heads compare line width to character count. But the paper does not explain how the line width is *computed*. Does the model take the max of previous line lengths? An exponentially weighted average? Something else? The authors acknowledge this gap explicitly. This matters because line width estimation is a more difficult problem than character counting — it requires cross-line aggregation.

To understand why this is hard, consider the structure of the problem. The character count at any position depends only on the tokens since the last newline. This is a local computation. But the line width depends on the widths of multiple previous lines. The model needs to somehow aggregate this information across lines. Does it:
(a) Use the width of the most recent line as an estimate? This would be fragile — different lines might have different widths.
(b) Compute a running average? This would require some form of memory across lines.
(c) Use special features that fire at newlines and accumulate width information? This seems most likely, and the paper mentions "partially disjoint set of heads" for line width, but does not go deeper.

For our pipeline, this gap is less concerning because our multiplication problems have a fixed structure (all problems at a given level have the same format). But it does highlight that even a thorough mechanistic analysis can leave important sub-computations unexplained.

**Open Question 2: How does the model handle multi-token words?** If the next word is "aluminum" and the tokenizer splits it into multiple tokens, the model needs to predict the newline before the first token of "aluminum," but it needs to know the full word's length. The paper does not address how the model handles this case. The authors show that the model treats the next predicted token's length as the relevant quantity, but they note this is an approximation.

This is actually a deep problem. When the model is about to output a multi-token word, it predicts one token at a time. At the first token of "aluminum" (say, "alum"), the model might not yet "know" that it is going to output "aluminum" — the remaining tokens ("inum") are future predictions. Yet the linebreaking decision needs to account for the full word length.

The paper's approach — using the predicted next token length — is the simplest possible heuristic. It works well because most English words that would overflow a line boundary are relatively short (1-2 tokens), and the model probably learns correlations between token length and word length. But for long words or unusual tokenizations, this heuristic could fail. This might explain some of the residual error in the linebreaking predictions.

For our pipeline, the analog is: how does the model predict multi-digit answers? When generating the first digit of a 6-digit answer, does the model already "know" all 6 digits, or does it compute them sequentially? Our data extraction is at the "=" token (the last token before the answer), so we capture the state before any answer generation begins. This might be the state where all answer digits are represented simultaneously — or it might be a state where only the first digit is committed and the rest are uncertain.

**Open Question 3: Does this generalize to other models?** The paper studies only Claude 3.5 Haiku. The authors note that they share transcoder attribution graphs for Gemma 2 2B and Qwen 3 4B (on Neuronpedia), suggesting similar mechanisms exist, but these are not analyzed in depth. The critical question for us: does Llama 3.1 8B use similar manifold representations for counting or arithmetic? We do not know.

There are reasons for both optimism and pessimism on generalization:

**Reasons for optimism:**
- The Gurnee paper shows character counting manifolds in Claude 3.5 Haiku (proprietary architecture)
- Engels et al. (ICLR 2025) show circular features for days/months in both Mistral 7B and Llama 3 8B (open architectures)
- Modell et al. (2025) find rippled manifolds for colors/dates/years across multiple models
- GPT-2's positional embeddings form a helix (Yedidia 2023)
- The mathematical argument (Fourier optimality for packing positions) is architecture-independent

**Reasons for pessimism:**
- Claude 3.5 Haiku is likely trained on different data than Llama 3.1 8B
- Claude is instruction-tuned while Llama 3.1 8B base is a pure language model
- The linebreaking task might receive very different emphasis in different training corpora
- Arithmetic is a harder task than counting, and harder tasks might not produce clean representations

**Open Question 4: Does the geometric description extend to more complex tasks?** The paper is explicit: "For the more semantic operations, we purely relied on the feature view." The geometric manifold description works beautifully for scalar quantities (counts, distances). It is unclear whether it applies to the semantic side of the computation (choosing which word comes next). The paper does not claim it does.

This is a fundamental limitation of the geometric approach. Continuous scalar quantities (counts, positions, digit values) have a natural ordering that maps onto a 1D manifold. But many concepts that models process are not scalar: syntactic categories, semantic roles, entity identities. These might be represented in more complex geometric structures (simplices, graphs, higher-dimensional manifolds) or might not have clean geometric descriptions at all.

For our pipeline, this question is less pressing because arithmetic sub-computations (digit values, carries, column sums) are all scalar quantities with natural orderings. But when we eventually want to understand how the model decides between different arithmetic strategies (e.g., left-to-right vs. right-to-left computation), the geometric approach might not suffice.

**Open Question 5: Can manifold discovery be automated?** The paper's discovery process was manual — researchers identified features, tested hypotheses about their meaning on synthetic data, and then searched for geometric structure. This is not scalable. The authors explicitly call for "methods that can automatically surface simpler structures to pay down the complexity tax." Currently, there is no unsupervised method for discovering feature manifolds.

This is perhaps the most important open question for the field. The Gurnee paper is a beautiful existence proof: manifold structure exists, it simplifies interpretation, and it enables understanding of computation. But finding it required extensive human effort — the kind of effort that does not scale to analyzing thousands of behaviors across dozens of models.

For our pipeline, we partly sidestep this problem because we know our variables in advance (digits, carries, column sums, answer digits). We do not need to discover them unsupervised — we have ground truth labels. But this is specific to arithmetic. For studying other model behaviors, automated manifold discovery would be essential.

Recent work by Modell et al. (2025) on "The Origins of Representation Manifolds in Large Language Models" proposes a theoretical framework for understanding when and why manifolds form, but does not provide a practical discovery algorithm. This remains a gap.

**Where the technique breaks down:** The approach requires knowing (or guessing) which continuous variable parameterizes the manifold. For character count, this is obvious. For arithmetic in Llama 3.1 8B, we know the parameterizing variables (digit values, carry values, column sums). But for discovering *unknown* continuous variables represented as manifolds, the approach offers no method. The paper says: "it is straightforward when studying known continuous variables but becomes difficult to execute correctly for more complex, difficult-to-parametrize concepts."

## Q4: Does Their Insight Apply to Other Unconsidered Problems?

Yes. The core insight — scalar quantities on rippled manifolds, computation via rotation, orthogonal arrangement for linear separability — is not specific to linebreaking. Let me map it onto our pipeline:

**Application 1: Digit representations in Llama 3.1 8B multiplication.**

Our Phase C found that input digits occupy 9-dimensional subspaces with cross-validated CV scores of 0.838 to 1.000. The Gurnee paper predicts that within these subspaces, digit values (0-9) should lie on a 1D manifold — a curve. If 10 values are embedded in a 9D subspace, the paper's theory predicts rippling: the cosine similarity matrix between digit-value centroids should show ringing (neighbors positive, distant values negative, then positive again). We should check this immediately.

Furthermore, the paper predicts that the manifold should be connected to Fourier modes. The DFT of the ideal similarity matrix gives the principal components. If digit representations are Fourier-structured, the top PCA components of digit-value centroids should correspond to Fourier modes with frequencies 1, 2, etc. This is exactly our Fourier screening step.

**Application 2: Carry representations.**

The paper shows that different scalar quantities (character count, line width, characters remaining) are represented on separate manifolds in near-orthogonal subspaces, and that computation involves comparing them via rotation. Our carries (values 0-8 or 0-17 depending on the column) are scalar quantities that result from comparing column sums to thresholds. If carry computation in Llama 3.1 8B follows the same pattern, we should find:

- Carry values on 1D manifolds in low-dimensional subspaces
- Column sums compared to thresholds via some rotation-like mechanism
- The comparison should make the carry decision linearly separable

Our Phase D already shows that carry subspaces exist (lambda = 0.740 for carry_2 at L5), and Phase C shows that correct and wrong populations have different clustering properties in these subspaces. The Gurnee paper provides a concrete geometric template for what the underlying computation might look like.

**Application 3: The distributed counting algorithm maps onto distributed arithmetic.**

The paper shows that character counting requires many attention heads working together, each contributing a low-rank piece of the overall manifold. In our setting, multi-digit multiplication requires accumulating partial products and carries across columns. If the model distributes this computation across heads similarly to how Claude distributes character counting, we should find:

- Individual heads contributing approximately 1D (ray-like) outputs
- Their sum forming a higher-dimensional curved manifold
- Layer 1 heads refining the estimate produced by layer 0 heads

This is a testable prediction for our cross-layer analysis (future pipeline step).

**Application 4: The "visual illusion" methodology maps onto adversarial arithmetic inputs.**

The paper constructs adversarial inputs ("@@" insertion) that specifically disrupt the counting mechanism. We could construct analogous "arithmetic illusions" — inputs that are designed to trigger specific failure modes based on our understanding of the carry computation. For example, if carries are computed by comparing column sums to thresholds via rotation heads, we could design problems where the column sums are near the carry threshold, and see if the model makes systematic errors.

## Q5: What Are the Caveats and Takeaways?

### Caveats

**Caveat 1: Proprietary model, no code.** Claude 3.5 Haiku is proprietary. The crosscoder dictionary is not released. The analysis pipeline is not released. Nobody can reproduce this work on the same model. You can only test whether the same *phenomena* exist in other models using your own tools.

**Caveat 2: The task is unusually clean.** Linebreaking in fixed-width text is about as clean a task as you can find in natural language. The input variables are well-defined, the output is binary (newline or not), and the algorithm is known. Most tasks that models perform are messier. The elegance of the manifold representations might partly reflect the cleanliness of the task. More complex tasks might produce messier geometry.

**Caveat 3: The "optimality" of rippled representations is only shown in a toy model.** The paper presents an attractive physical simulation showing that packing vectors into a small subspace with neighbor-attraction and distant-repulsion produces rippled curves. But this is a toy model of the representation, not a proof that the network optimizes for this specific objective. The actual training objective is next-token prediction, not vector packing. The connection between the training objective and the resulting geometry is assumed, not derived.

**Caveat 4: Centroid-based analysis averages away individual variation.** The manifold is constructed by computing the *average* residual stream for each character count value. Individual data points scatter around these averages. How much scatter there is, and whether the scatter matters for the computation, is not quantified. (This is the same limitation as our Phase C centroid approach — centroid averaging destroys within-group structure.)

**Caveat 5: The analysis is limited to early layers.** Most of the manifold analysis focuses on layers 0-2 (character counting) and around 90% depth (final decision). What happens in the middle layers? The paper does not study this systematically. The middle layers might use the same manifold structure, or they might use entirely different representations.

### Key Takeaways

**Takeaway 1:** Rippled 1D manifolds in low-dimensional subspaces are a real phenomenon in production models, not just toy models. This validates the entire approach of looking for non-linear structure within linear subspaces. In our pipeline's language: the LRH finds the room (the 6D subspace), but the manifold is the furniture inside the room (the 1D curve).

**Takeaway 2:** Computation happens through geometric operations on manifolds. The boundary head twist (rotating one manifold to align with another) is a concrete example of computation-via-geometry. This is not a metaphor — the rotation is literally implemented by the QK matrix of an attention head.

**Takeaway 3:** The feature-manifold duality is real and useful. Discrete features tile the manifold in a canonical way. Neither view alone is complete. The feature view helps with discovery; the manifold view helps with understanding computation.

**Takeaway 4:** Fourier structure emerges naturally from the optimality of rippled embeddings. The connection between the ideal similarity matrix, its eigendecomposition, and Fourier modes is clean and mathematical. This directly supports our Fourier screening step.

**Takeaway 5:** Multiple attention heads are needed to build curved manifolds because individual head outputs are approximately linear (rays in PCA space). Curvature requires the nonlinear combination of multiple linear contributions. This is an important architectural constraint that we should keep in mind when looking for manifold-constructing circuits.

---

---

# Devil's Advocate Protocol — Three Weakest Points

### Weakness 1: The Analysis Only Covers One Model on One Task

The paper's entire analysis is on Claude 3.5 Haiku doing linebreaking. The authors mention that similar structures exist in Gemma 2 2B and Qwen 3 4B, but only via attribution graph visualizations on Neuronpedia — not through the same rigorous analysis. The critical question — do these manifold structures exist in models that did not learn linebreaking as cleanly? — is unanswered.

This matters because linebreaking is a specific, well-defined, frequently occurring task in pretraining data. Claude 3.5 Haiku does it well. But many tasks that models perform are less clean, less frequent, and less well-learned. The paper's implicit suggestion that manifold representations are a general phenomenon is an extrapolation from a single case study.

To be fair, the paper is explicit about this: "We would be excited to see more deep case studies that adopt this approach." But the reader should be cautious about generalizing too aggressively from this one example.

**How this affects our project:** We are studying multiplication in Llama 3.1 8B base. Multiplication is a harder task than character counting (higher-dimensional inputs, more intermediate computations, lower accuracy). The model performs multiplication much worse than Claude performs linebreaking. The representations for a poorly-learned task might be much messier than the clean manifolds shown here. We should expect this and not be disappointed if our manifolds are noisier.

### Weakness 2: The Causal Validation Is Coarse-Grained

The paper presents two types of causal experiments:

1. **Subspace ablation:** Zero out the top-k PCA components and measure loss increase. This shows the subspace is important, but it does not show *why* it is important. Ablating 6 dimensions of a high-dimensional space might disrupt many computations, not just character counting.

2. **Mean-patching:** Replace the activation with the mean activation for a target character count. This is more surgical, but it operates on averages. Individual data points might respond differently.

What is missing is **feature-level causal intervention**. The paper does not show, for example, that rotating the manifold by a specific angle in the QK space of a boundary head shifts the predicted number of characters remaining by the corresponding amount. The "twist" mechanism is described at the level of weight matrices and probes, but there is no experiment that directly manipulates the rotation and measures the downstream effect.

The visual illusion experiments are closer to proper causal validation, but they are qualitative: inserting "@@" disrupts the counting, and the disruption correlates with attention pattern changes. A more rigorous test would be to continuously modulate the attention patterns and show a continuous effect on the counting estimate.

**How this affects our project:** Our planned causal patching step should go beyond subspace ablation and mean-patching. We should aim for more targeted interventions — for example, projecting activations onto specific manifold positions and measuring whether the model's arithmetic output changes predictably.

### Weakness 3: The "Optimality" Argument Is Circular

The paper argues that rippled manifold embeddings are "optimal" for packing N positions into k dimensions. But the argument uses a specific objective: maximize similarity between neighbors while minimizing similarity between distant points. This objective is not the model's actual training objective (next-token prediction).

The physical simulation (attractive forces between neighbors, repulsive forces between distant points) produces rippled curves. But why should we believe that next-token prediction produces the same optimization landscape? The paper does not provide a formal connection.

It is plausible that the connection exists: representing character counts accurately for linebreaking requires distinguishing neighboring counts, which is equivalent to the neighbor-similarity objective. But the paper presents this as an established fact rather than a hypothesis. A skeptic could argue that the model might use a completely different scheme (e.g., binary coding, one-hot with noise, or something else entirely) that happens to project onto a rippled curve under PCA without the model actually "optimizing" for rippling.

The Fourier connection is more mathematically grounded: the eigendecomposition of a circulant similarity matrix yields Fourier modes, and truncating to the top-k eigenvalues produces ringing. This is a mathematical fact, not a claim about optimization. But the claim that the model's representations are *because* of Fourier optimality (rather than merely resembling Fourier structure) is not proven.

**How this affects our project:** We should be cautious about claiming that any geometric structure we find is "optimal" unless we can derive the optimality from the training objective. In our case, we should test whether digit representations in Llama 3.1 8B have Fourier structure (our Fourier screening step), but we should not claim that the structure is optimal without a formal argument connecting it to the training loss.

---

---

# PASS 3 — Deep Dive (The Machinery in Detail)

## 3.1 The Character Count Manifold — Construction and Properties

### What is the manifold, precisely?

For each value of line character count c (from 1 to 150), the authors compute the average residual stream activation across all tokens in their synthetic dataset that have that count. This gives 150 vectors in R^d, where d is the residual stream dimension of Claude 3.5 Haiku.

Computing PCA on these 150 vectors, they find that 6 principal components capture 95% of the variance. The 150 vectors, projected into this 6D subspace, form a smooth curve. The curve spirals through the subspace — it looks helical when viewed in PCs 1-3, and has additional structure in PCs 4-6.

Let me be precise about what "manifold" means here. In the strict mathematical sense, a manifold is a topological space that locally looks like Euclidean space. The character count "manifold" is really a discrete set of 150 points (one per count value) that lie approximately on a smooth 1D curve. The curve is the manifold; the 150 points are samples from it.

The intrinsic dimension is 1 (the character count parameter). The extrinsic dimension is 6 (the PCA subspace). The full ambient dimension is d (the residual stream dimension, which for Claude 3.5 Haiku is likely in the thousands). The manifold is embedded with high curvature, meaning it twists and turns rather than following a straight line.

### How do features relate to the manifold?

The paper finds 10 sparse crosscoder features that activate based on character count. Each feature has a "receptive field" — a range of character counts over which it is nonzero. The features are overlapping: typically two features are active at any given count. This is reminiscent of place cells in the hippocampus, which fire at specific locations along a track with overlapping receptive fields.

The key observation: if you reconstruct the residual stream using only these 10 features (multiplying each feature's activation by its decoder vector and summing), the resulting curve closely matches the PCA manifold. The match is not perfect — there are "kinks" near the feature decoder vectors, "reminiscent of a spline approximation of a smooth curve." But 10 feature vectors spanning a space of 150 data points is a compression factor of 15x, and the approximation is quite good.

This is the duality in action. The 10 features provide a sparse, discrete parameterization of the manifold. The manifold provides a continuous, smooth description of the same representation. Each feature decoder vector sits at a specific point on the manifold. The feature activations interpolate between these anchor points, tracing out the curve between them.

### What makes the manifold "rippled"?

If you wanted 150 unit vectors in R^150 that are each similar to their neighbors and orthogonal to distant vectors, you would use something like shifted cosine windows. The similarity matrix X for such an arrangement has a peaked diagonal (neighbors are similar) falling off to zero (distant points are orthogonal).

Let me walk through a concrete small example to make this tangible. Imagine you have 10 positions (instead of 150) and you want each position to be similar to its immediate neighbors but orthogonal to positions 5+ away. In R^10, you could use the standard basis vectors e_1, ..., e_10, but then every pair is orthogonal — no similarity between neighbors. Instead, you want something like:

- sim(pos 1, pos 2) = 0.8 (neighbors are similar)
- sim(pos 1, pos 3) = 0.3 (two apart, moderately similar)
- sim(pos 1, pos 4) = 0.0 (three apart, orthogonal)
- sim(pos 1, pos 5) = 0.0 (four apart, orthogonal)

This defines a 10x10 similarity matrix X. In 10 dimensions, you can find 10 unit vectors that exactly reproduce this matrix. But what if you only have 3 dimensions? You need to approximate X as best you can in 3D.

The best 3D approximation comes from taking the top 3 eigenvalues/eigenvectors of X. When you compute the resulting approximate similarity matrix X_3, something happens: the nice clean falloff from 0.8 to 0.3 to 0.0 develops oscillations. Instead of staying at 0.0 for distant positions, the similarity dips negative (around -0.2), then comes back up (around +0.1), then dips again. These oscillations are the "ringing."

In the spatial domain, the 10 vectors in 3D trace out a curve that ripples. Instead of a simple arc, you get a curve that twists back and forth, like a spiral staircase. The rippling is the price you pay for cramming 10 distinguishable positions into only 3 dimensions.

Now scale this up to 150 positions in 6 dimensions, and you get the character count manifold: a helix-like curve with complex twisting in PCs 4-6, and prominent ringing in the similarity matrix.

Now suppose you can only use 6 dimensions instead of 150. The best 6-dimensional approximation (in the L2 sense) is given by projecting onto the top 6 eigenvectors of X. For a circulant similarity matrix (which X approximately is, given the circular topology), the eigenvectors are the Fourier modes. Truncating to the top 6 Fourier modes keeps the dominant frequencies but drops the high-frequency components. The result: the similarity matrix develops "ringing" — side lobes beyond the main diagonal peak.

In the spatial domain (viewing the 150 projected vectors in the 6D subspace), this ringing manifests as a curve that ripples. Instead of a simple circle (which would use only 2 dimensions), you get a helix-like structure that uses all 6 dimensions. The rippling allows the curve to pack more distinguishable positions into the available dimensions.

The paper draws a precise mathematical connection: the ringing pattern is the same phenomenon that appears when you truncate a Fourier series — the Gibbs-like oscillations near discontinuities. In our context, the "discontinuity" is the sharp transition from positive to negative similarity between neighbors and non-neighbors. Truncating to 6 Fourier modes produces oscillations around this transition.

### The physical simulation

To make the optimality argument more intuitive, the paper presents an interactive physical simulation. 100 points are placed on a (n-1)-sphere in R^n. They experience attractive forces to their 6 nearest neighbors on each side and repulsive forces to all other points. The forces have specific functional forms:

- Attractive: F = (1 - (d-1)/2) * r_hat / r   (when index distance d <= w)
- Repulsive: F = -min(5, 1/r) * r_hat / r      (when index distance d > w)

where r is Euclidean distance, r_hat is the unit direction, d is the index distance (circular), and w is the attractive zone width.

After relaxation with damping (alpha = 0.95, dt = 0.01), the points settle into a rippled curve on the sphere. The key findings from the simulation:

- Decreasing the attractive zone or increasing the embedding dimension both increase curvature and ringing
- In 3D, the result looks like a baseball seam — which matches observations by Modell et al. (2025) for color, date, and year representations
- As the number of points grows and the attractive zone shrinks, the curvature grows extreme, approaching a space-filling curve in the limit

This simulation does not prove that neural networks optimize for this objective. But it demonstrates that the observed geometry is a natural consequence of any system that tries to pack many distinguishable positions into a small number of dimensions while maintaining local similarity.

## 3.2 The Boundary Head Mechanism — Comparison via Rotation

This is perhaps the most elegant finding in the paper and deserves careful analysis.

### Setup

The model needs to detect when the current character count is close to the line width. Two separate manifolds exist:
- The character count manifold (computed from the current line's tokens)
- The line width manifold (computed from previous newlines)

Both are 1D curves in low-dimensional subspaces of the residual stream. They are not perfectly aligned — their subspaces partially overlap, but the cosine similarity between corresponding probes (character count = i vs. line width = i) is only about 0.25.

### The twist

An attention head (the "boundary head") applies its QK circuit to both sets of representations. In the QK space, something remarkable happens:

- The character count probes and line width probes become almost perfectly aligned (cosine similarity approaching 1.0)
- But they are aligned with an offset: character count i aligns with line width k where k = i + epsilon

This means: when the character count is close to (but slightly less than) the line width, the QK dot product is large. The head attends strongly from the current position to the newline position. This writes "boundary detection" features into the residual stream.

### Why this is mathematically elegant

In the residual stream, compare(c, w) = "is c close to w?" is a nonlinear operation. You cannot implement it with a single dot product because two different pairs (c, w) can have the same dot product even when their differences c-w are very different.

Let me explain this with a concrete example that makes the problem clear. Suppose character count is represented as a simple 1D scalar. Position 30 is represented by the number 30, position 50 by the number 50, line width 50 by the number 50, line width 80 by the number 80. Now, the dot product of (count=30) with (width=50) is 30*50 = 1500. The dot product of (count=50) with (width=80) is 50*80 = 4000. These give different values even though in both cases the count is 20 less than the width. Worse, the dot product of (count=50) with (width=50) is 2500, and (count=70) with (width=70) is 4900. Both represent "exactly at the boundary," but they give different dot products because the absolute magnitudes differ. The dot product tells you about the product of the values, not their difference. For linebreaking, the difference is what matters.

Now consider the 2D case. Represent count c as the vector (cos(c * theta), sin(c * theta)) for some angle theta. This places each count on a circle. Similarly, represent width w on the same circle. The dot product of count c and width w is cos((c-w) * theta) — it depends only on the difference c-w, not on the absolute values. This is exactly what we want for linebreaking.

But there is a subtlety. In the residual stream, the character count and line width manifolds are not initially on the same circle. They are curves in different parts of the high-dimensional space. The QK matrix of the boundary head performs a linear transformation that maps both curves into the same low-dimensional space, aligning them so that the dot product becomes a function of c-w.

The "twist" is this: the QK matrix does not just align count c with width c (which would give dot product 1 when the count equals the width — not useful, because you want to know when the boundary is *approaching*, not when you have already hit it). Instead, it aligns count c with width c+epsilon. This means the dot product is large when the count is slightly less than the width — exactly when the model should start considering a newline.

But if you first *rotate* the character count manifold so that position c aligns with position w at the right offset, then a single dot product suffices. The rotation maps a nonlinear comparison into a linear operation (dot product). This is the power of having multi-dimensional representations: rotation is a linear transformation, and linear transformations are exactly what QK circuits implement.

This is deeply connected to the Fourier structure. On a circle, a shift by delta corresponds to a rotation. The QK matrix of the boundary head learns exactly this rotation. Because the manifold has Fourier structure (it is approximately a truncated Fourier embedding of a circle), a linear map can implement a shift along the manifold. If the manifold were a generic curve without this structure, a linear shift along it would not be possible.

### Multiple boundary heads for resolution

One boundary head is not enough. A single head attends strongly when the character count is near the line width, but it cannot distinguish between "5 characters remaining" and "17 characters remaining" — both produce similar attention patterns. Multiple heads with different offsets tile the space of possible characters-remaining values.

The paper shows this by projecting each head's output into the PCA space of characters-remaining probes. Head 0 varies most in the [0,10] and [15,20] ranges. Head 1 varies most in [10,20]. Head 2 varies most in [5,15]. Their sum produces an evenly spaced representation covering all values.

This is a distributed encoding: no single head has full resolution, but together they do. The paper validates this by showing that the 2D PCA of the summed head outputs captures 92% of the variance in characters-remaining, and that ablating this 2D subspace disrupts newline prediction.

## 3.3 The Final Decision — Orthogonal Subspaces and Linear Separability

At the end of the model (around 90% depth), the decision to insert a newline requires combining two quantities:
- How many characters remain in the line (characters remaining = line width minus character count)
- How many characters the predicted next token has (next token length)

The model should predict a newline when characters remaining is less than next token length.

The paper finds that these two quantities are represented on manifolds in near-orthogonal subspaces. When visualized in PCA, characters-remaining and next-token-length lie on curves in different planes that intersect at the origin.

Why does this matter? Because orthogonal representations make the decision linearly separable. Consider the sum of the characters-remaining vector z_rem and the next-token-length vector z_len. When z_rem and z_len are orthogonal, the combined vector z = z_rem + z_len lives in the direct sum of their subspaces. The decision boundary "should I break the line?" (i.e., rem < len) corresponds to a hyperplane in this combined space.

The paper tests this: a separating hyperplane in the 3D PCA of the pairwise combinations achieves AUC = 0.91 on real data. The model has arranged its representations so that the final decision is geometrically trivial — just a linear boundary in a 3D space.

This is a powerful design principle: if you arrange the inputs to a comparison on orthogonal subspaces, the comparison becomes a linear operation. No MLP nonlinearity is needed. This has implications for our multiplication pipeline: if the model arranges carry-related representations orthogonally to digit-related representations, the carry decision might also be linearly separable.

## 3.4 The Distributed Counting Algorithm

This section is the most complex in the paper, and deserves step-by-step treatment.

### The problem

How does the model compute the character count? Remember: the input is a sequence of tokens, each with a variable number of characters. The model needs to accumulate a running sum of character lengths since the last newline. But there is no explicit "counter" — the count must emerge from the interaction of attention heads.

### Layer 0: Rays that sum to curves

Each attention head in layer 0 produces an output that, when projected onto the character count probe space, looks approximately like a 1D ray. Different heads produce rays in different directions. The sum of 5 key layer 0 heads produces a 2D+ manifold with curvature.

Why does each head produce a ray? Because each head's output is a linear combination of its inputs (via the OV circuit), weighted by attention scores. This is fundamentally a linear operation on the inputs, constrained to have at most rank 1 in the probe space for any single attention pattern. The curvature comes from the *sum* of these linear components.

Let me unpack this more carefully, because it is a subtle but important point. An attention head computes:

output = sum over positions j of [ attention(j) * OV * input(j) ]

For a fixed set of attention weights, this is a fixed linear combination of the input embeddings. If the OV matrix maps the input embeddings to a specific direction in the probe space, the output is a scalar multiple of that direction — a ray. The scalar depends on the attention-weighted sum of the relevant input component.

As the attention pattern changes (different positions getting different weights), the scalar changes, but the direction stays the same (it is determined by OV, not by attention). So the output traces a 1D ray as a function of position in the line.

Now, when you sum multiple heads, each contributing a ray in a different direction, you get a multi-dimensional output. If head A contributes a ray in direction d_A and head B contributes a ray in direction d_B, and the magnitudes of both vary as a function of character count, the sum traces a 2D curve in the plane spanned by d_A and d_B. With 5 heads contributing rays in 5 directions, the sum traces a curve in a 5D subspace. The curvature of this curve comes from the different heads "turning on" and "turning off" at different positions along the line.

This is a deep architectural insight. A single attention head cannot produce curvature in its output. Curvature requires the cooperation of multiple heads. This is why the paper calls it a "distributed" counting algorithm — no single component can do it alone.

### A concrete walkthrough of head L0H1

The paper provides a detailed analysis of one specific head, L0H1. Let me walk through exactly what this head does, step by step.

**Step 1: Attention pattern (QK circuit).** L0H1 uses the previous newline token as an attention sink. For the first ~4 tokens after the newline, the head attends almost entirely to the newline (attention weight close to 1.0). After about 4 tokens, the head starts spreading its attention over the previous 4-8 content tokens.

Why does this make sense? The newline marks the start of the current line. Attending to it when the line has just started means "I know the line is very short." Once there are enough content tokens, the head shifts to attending to those tokens to gather more information about their lengths.

**Step 2: Output when attending to newline (OV circuit).** When L0H1 attends to the newline, its OV circuit writes to the 5-20 character count directions and suppresses the 30-80 character count directions. Think of this as the head saying: "I am attending to the newline, which means we are in the first few tokens of the line. So the character count is probably around 5-20 (because ~4 tokens * ~4 characters/token = ~16 characters) and definitely not 30-80."

**Step 3: Output when not attending to newline (OV circuit).** When L0H1 has shifted its attention entirely to content tokens (meaning the line is at least ~8 tokens long), the head defaults to writing to the ~40 character count direction. Why 40? Because if the line has at least 8 tokens, and each token averages about 5 characters (including spaces), the count is approximately 8 * 5 = 40.

But the head also applies a correction based on the actual lengths of the attended tokens. If the attended tokens are shorter than average (less than 4 characters), the head shifts its output toward lower character counts (10-35) and suppresses higher counts (40+). If the tokens are longer than average (5+ characters), it does the opposite — suppresses low counts and upweights high counts.

**Step 4: Intermediate cases.** When the head splits its attention between the newline and content tokens, the output is a linear interpolation between the two extreme cases. This creates a smooth transition from "short line" to "long line" estimates.

**Step 5: How this becomes a ray.** Across all positions in a line, L0H1's output sweeps through a specific trajectory in the probe space: starting at the 5-20 range (when near the newline), transitioning through 30-40 (middle of the line), and ending around 40-60 (far from the newline). This trajectory is approximately 1D — a ray with some curvature at the transition points.

### The attention mechanism for counting

Each counting head uses the previous newline token as an "attention sink." The head attends strongly to the newline for the first s_h tokens after it, then gradually shifts attention to the content tokens. This creates a piecewise behavior:

- When attending to the newline (first s_h tokens): the head writes an offset corresponding to roughly s_h times the average token length (about 4 characters per token)
- When attending to content tokens (after s_h tokens): the head writes an estimate based on the number and length of the attended tokens
- The OV circuit applies a correction based on whether the attended tokens are longer or shorter than average

Different heads specialize at different offsets. Head L0H1 attends to the newline for the first ~4 tokens. Another head for the first ~8 tokens. Together, they tile the space of possible line positions.

### Layer 1: Refinement

Layer 1 heads perform a similar operation but have access to the layer 0 output. This means they can use the initial character count estimate as an additional input, producing a more refined estimate. The paper shows that the R-squared for character count prediction improves from 0.93 (layer 0 alone) to 0.97 (layers 0 and 1 together).

This iterative refinement — coarse estimate in layer 0, refined in layer 1 — is a general computational strategy. In the context of arithmetic, we might see a similar pattern: coarse digit-level computation in early layers, refined carry propagation in later layers.

### Why MLPs are not important here

The paper notes that attention head outputs affect the character count representation 4 times more than MLPs. This is a surprising and important finding. It suggests that for this particular task, the computation is primarily attention-mediated, with MLPs playing a minor role (perhaps adding bias corrections or handling edge cases).

For our multiplication pipeline, this raises the question: is arithmetic computation also primarily attention-mediated? Or do MLPs play a larger role? Bai et al. found that MLPs implement key parts of the multiplication algorithm in their toy models. The answer might differ between toy and production models.

## 3.5 Visual Illusions — Adversarial Validation

This is the most creative part of the paper. The authors use their understanding of the counting mechanism to construct adversarial inputs that break it in predictable ways.

### The mechanism

The counting attention heads attend from newline to newline to measure line width. But these heads can also attend to the two-character sequence "@@", which appears in git diff headers. In a git diff context, "@@" legitimately marks a new counting reference point. But when "@@" appears outside a git diff context, it confuses the counting heads: they attend to "@@" instead of (or in addition to) the previous newline, disrupting the character count estimate.

This is a beautiful example of adversarial transfer. The model learned a legitimate association ("@@" marks a position reset in git diffs). This association was useful during training. But when the same cue appears in a different context (prose text), the association fires inappropriately, causing a systematic error. The error is not random — it is a predictable consequence of a known mechanism.

### The systematic experiment

The authors do not stop at a single anecdote. They systematically test 180 different two-character sequences, inserted at the same position in the aluminum prompt. For each sequence, they measure two things:

1. **Impact on newline probability:** How much does the insertion change the probability that the model predicts a newline at the correct position? The original probability is 0.79.

2. **Attention distraction:** How much does the insertion shift attention in the counting heads away from the newline toward the inserted characters?

Most insertions cause moderate disruption (newline probability drops to 0.5-0.7). But a small subset causes severe disruption (newline probability drops to 0.1-0.3). These severely disruptive sequences are overwhelmingly code-related: ``  >>  }}  ;|  ||  `,  @@. These sequences appear as delimiters or separators in programming languages, and the counting heads have learned to attend to them as potential position-reset markers.

The correlation between attention distraction and newline probability disruption is positive and significant. This is the smoking gun: the disruption works through the specific attention mechanism that the paper identified, not through some other pathway.

### Why this matters for interpretability methodology

The visual illusion experiment establishes a gold standard for mechanistic interpretability validation. The logic is:

1. **Understand the mechanism** (boundary heads rotate character count manifold to compare with line width)
2. **Predict a consequence** (distracting the counting heads should disrupt linebreaking)
3. **Design an experiment** (insert sequences that the counting heads might attend to)
4. **Observe the predicted effect** (disruptive sequences are exactly those that distract the counting heads)
5. **Verify the mechanism** (the disruption correlates with attention shift, not just token identity)

This is the scientific method applied to neural network internals. It goes beyond "we found structure" (observational) to "we used our understanding to predict and create specific failure modes" (experimental). This is the level of validation we should aim for in our causal patching phase.

### Designing "arithmetic illusions" for our pipeline

Following the same logic for multiplication in Llama 3.1 8B:

1. **Understand the mechanism:** (Phase F and Fourier screening should identify how carries are computed)
2. **Predict a consequence:** (If carry computation uses specific attention heads, distracting those heads should cause carry errors)
3. **Design an experiment:** Construct multiplication problems that look superficially like carry-free problems (e.g., all small digits) but actually require carries due to specific digit combinations. If the model uses contextual cues (like "small digits usually mean no carries"), it might be fooled.
4. **Observe the effect:** The model should make more carry errors on these adversarial problems than on matched non-adversarial problems.
5. **Verify the mechanism:** The errors should correlate with changes in the identified carry computation mechanism.

For example: 22 × 23 = 506. Both operands are small (two digits, values in the low 20s). But 2*3 = 6 and 2*2 = 4, with column 1 sum = 2*3 + 2*2 = 10, which generates a carry. If the model has learned a heuristic "small operands → no carries," this problem could trigger an error. We could test this by comparing accuracy on problems with unexpected carries (small operands but carries present) vs. expected carries (large operands with carries present).

### Connection to human visual illusions

Inserting "@@" into the aluminum prompt (without changing the line length) reduces the probability of predicting a newline at the correct position. The authors verify this is not just any two-character insertion: testing 180 different two-character sequences, they find that most cause moderate disruption, but a few code-related sequences (@@, ``, >>, }}, ;|, ||, `,) cause substantially more.

The disruption correlates with how much the inserted tokens "distract" the counting attention heads (measured by how much attention shifts from the newline to the inserted tokens). This is a clean mechanistic explanation: the illusion works by hijacking a specific attention mechanism.

### Connection to human visual illusions

The paper draws an analogy to human visual illusions like the Muller-Lyer illusion (where arrowheads modulate perceived line length) and the Ponzo illusion (where converging lines modulate perceived size). In both human and model cases, contextual cues (arrowheads / "@@" symbols) modulate estimates of spatial properties (line length / character count) by exploiting learned priors (3D perspective / git diff syntax).

The authors are careful not to overstate this analogy: "While we are not claiming any direct analogy between illusions of human visual perception and this alteration of line character count estimates, the parallels are suggestive."

This is scientifically responsible framing. The parallels are indeed suggestive of a deeper principle — that any system that uses learned contextual priors for perception is susceptible to adversarial manipulation of those priors — but the analogy should not be taken too literally.

### Relevance to our project

This suggests a validation strategy for our multiplication analysis. If we identify the attention mechanism responsible for carry computation in Llama 3.1 8B, we could construct "arithmetic illusions" — inputs that look like they should produce a specific carry pattern but actually trigger a different pattern because of contextual cues. For example, problems where the digit pattern mimics a common carry-heavy configuration but the actual carries are different.

---

---

# PASS 3 Continued — Deep Mathematical Content

## 3.6 Fourier Structure and Analytic Construction of Ringing

The appendix contains a beautiful analytic construction that connects the manifold geometry to Fourier analysis. Let me walk through it in detail.

### The setup

Suppose we want N unit vectors (representing N positions on a circle) whose cosine similarities follow a specific pattern: neighboring vectors should be similar, distant vectors should be orthogonal. The ideal similarity matrix X is a circulant matrix (same similarity for the same angular distance, wrapping around the circle).

### The eigendecomposition

Because X is circulant, the Discrete Fourier Transform (DFT) diagonalizes it. The DFT matrix F has entries F_{jk} = exp(2*pi*i*jk/N). The eigenvalues of X are the Fourier coefficients of the function f that defines the similarity pattern. That is, lambda_k = sum_j f(j) * exp(-2*pi*i*jk/N).

The eigenvectors are the columns of F. Physically, each eigenvector represents a pure frequency — a sinusoidal pattern around the circle. The eigenvalue lambda_k tells you how much the frequency-k mode contributes to the similarity pattern.

### Truncation produces ringing

The full N-dimensional embedding X = F * diag(lambda) * F^H reproduces the similarity exactly. But if we keep only the top k eigenvalues (the dominant Fourier modes), we get an approximation X_k that is the best rank-k approximation in the L2 sense.

This truncation has a well-known effect: ringing. Just like truncating a Fourier series produces Gibbs oscillations near discontinuities, truncating the eigendecomposition of X produces side lobes in the similarity matrix. The peaked diagonal (positive similarity for neighbors) develops oscillating tails (alternating negative and positive similarity for distant points).

Let me trace through this with a concrete small example. Suppose N = 10 (10 positions on a circle) and the desired similarity function is f(d) = max(0, 1 - d/3), where d is the circular distance. This gives f(0) = 1, f(1) = 0.67, f(2) = 0.33, f(3) = 0, f(4) = 0, f(5) = 0. The function is peaked at zero and falls linearly to zero at distance 3.

The DFT of f gives the eigenvalues lambda_k. Most energy is in the low frequencies: lambda_0 (the DC component, proportional to the average similarity) and lambda_1, lambda_2 (the first two harmonics). Higher frequencies lambda_3, lambda_4, lambda_5 have small but nonzero amplitudes.

If we keep only k=5 dimensions (lambda_0 through lambda_2, which gives 1 + 2 + 2 = 5 real dimensions because each nonzero frequency contributes a sine/cosine pair), the reconstructed similarity matrix X_5 is the best 5D approximation. But it cannot perfectly reproduce the sharp cutoff at distance 3. Instead of going from f(2) = 0.33 to f(3) = 0.0 cleanly, the approximation overshoots: f_approx(3) might be -0.08, f_approx(4) might be +0.03. These oscillations are the ringing.

In the spatial domain, the 10 vectors projected into 5D form a rippled curve — a helix whose turns are tighter than a simple circle because the higher Fourier modes contribute additional angular displacement.

In the spatial domain, the N vectors projected into k dimensions form a rippled curve. The rippling comes from the retained Fourier modes — the curve's shape is literally a superposition of sinusoidal components.

### The rotation property — a step-by-step derivation

This is the key insight that connects to the boundary head mechanism, and it deserves a careful mathematical walkthrough. On a circle, shifting all positions by one step (i maps to i+1) is a permutation rho. In the full N-dimensional space, rho is a linear operator. Because rho preserves the circulant structure (conjugation by rho fixes X), rho commutes with the projection pi_k onto the top-k eigenspace.

Let me spell this out step by step, because this is the mathematical foundation for why the boundary head mechanism works.

**Step 1: The shift operator.** Define the N×N permutation matrix P that sends the i-th standard basis vector e_i to e_{(i+1) mod N}. This P is itself a circulant matrix — it is the circulant generated by the vector (0, 1, 0, ..., 0).

**Step 2: Circulant matrices commute.** Our similarity matrix X is circulant (because the similarity f(d) depends only on the circular distance). A fundamental property of circulant matrices is that they are all simultaneously diagonalized by the DFT matrix F. That is, X = F * diag(lambda_X) * F^H and P = F * diag(lambda_P) * F^H. Since diagonal matrices commute: PX = F * diag(lambda_P) * diag(lambda_X) * F^H = F * diag(lambda_X) * diag(lambda_P) * F^H = XP.

**Step 3: Shared eigenvectors.** Because P and X are both diagonalized by F, they share the same eigenvectors. The k-th eigenvector of X is the k-th column of F, which is the vector (1, omega_k, omega_k^2, ..., omega_k^{N-1}) where omega_k = exp(2*pi*i*k/N). The eigenvalue of X at frequency k is lambda_X(k) = sum_d f(d) * exp(-2*pi*i*k*d/N). The eigenvalue of P at frequency k is lambda_P(k) = exp(2*pi*i*k/N).

**Step 4: Projection commutes with shift.** The projection pi_k onto the top-k eigenspace keeps the eigenvectors corresponding to the k largest eigenvalues of X. Since P shares these eigenvectors, P maps the eigenspace to itself, so pi_k * P = P * pi_k on the eigenspace.

**Step 5: The restricted shift is a rotation.** The restriction of P to the k-dimensional eigenspace, call it P_bar, is a k×k unitary matrix. Its eigenvalues are the subset of {exp(2*pi*i*j/N)} corresponding to the retained frequencies. In the real basis (sine/cosine pairs), P_bar is a product of 2D rotation matrices — one rotation by angle 2*pi*j/N for each retained frequency j.

**Step 6: Application to the boundary head.** The boundary head's QK matrix learns an approximation to P_bar^m (a shift by m positions). This is a linear transformation in the k-dimensional subspace that shifts the character count manifold by m steps. The result: character count c gets mapped to roughly the same location as character count c+m, enabling comparison with line width via dot product.

The punchline: the Fourier structure of the manifold is what makes this linear shift possible. If the manifold were an arbitrary curve (not a Fourier embedding), no linear map could implement a shift along it. The model's choice of Fourier-like representation is not just efficient for storage — it is essential for enabling linear computation on the manifold.

This means: the restriction of the shift operator to the k-dimensional subspace, rho_bar = pi_k * rho * pi_k, maps pi_k(v_i) to pi_k(v_{i+1}). In other words, there exists a linear map in k dimensions that shifts points along the rippled curve.

This is exactly what the boundary head QK matrix implements. It applies a linear map (in the residual stream) that shifts the character count manifold along itself by a small offset. This shift is possible precisely because the manifold has Fourier structure — it is a truncated Fourier embedding, and shift operators on Fourier embeddings are linear.

If the manifold were a generic curve without Fourier structure, no linear map could shift points along it. The Fourier structure is not decorative — it enables the computational mechanism.

### Implications of the rotation property for our multiplication pipeline

This mathematical result has profound implications for what we should look for in Llama 3.1 8B.

If digit values (0-9) are represented on a Fourier manifold, then any operation that involves "shifting" a digit value — like adding a carry to a column sum — can be implemented by a linear transformation. The carry addition is: new_digit = (column_sum + carry_in) mod 10. In Fourier space, adding carry_in to the column sum corresponds to a rotation by carry_in * (2*pi/10) = carry_in * 36 degrees. An attention head whose QK or OV circuit implements this rotation would be performing carry addition geometrically.

Conversely, if digit values are NOT on a Fourier manifold, carry addition cannot be implemented by a single linear transformation. The model would need MLPs (nonlinear layers) to implement the modular arithmetic. This gives us a testable prediction: if Fourier structure is present, we should find attention heads that perform carry-related rotations. If Fourier structure is absent, carry computation should be more heavily mediated by MLPs.

This connects directly to the question of whether the Gurnee paper's findings generalize to arithmetic in Llama 3.1 8B. The Fourier structure is not just a pretty geometric pattern — it is a computational enabler. Finding (or not finding) it tells us something deep about how the model performs arithmetic.

### How close is the real manifold to the Fourier prediction?

The paper tests this: Fourier components explain at most 10% less variance than an equivalent number of PCA components. Since PCA components are optimal for variance, 10% loss means the real manifold is close to (but not exactly) a truncated Fourier embedding. The discrepancy comes from dilation (wider receptive fields at higher counts) which breaks the circulant assumption.

For our pipeline, this suggests that Fourier screening should work well as a first approximation, with the caveat that dilation effects might reduce the Fourier power slightly. Our digit values (0-9) are a much smaller set than character counts (1-150), so the Fourier structure might be even cleaner.

## 3.7 Representation Sharpening Across Layers

The paper shows that the character count representation gets sharper (higher curvature, more pronounced ringing) as we move through layers 0 to 3. Cross-sections of the probe cosine similarity matrix at specific character counts show that the peaks narrow and the side lobes grow with each layer.

This sharpening is produced by the layer 1 heads, which refine the coarse layer 0 estimate. In Fourier terms, sharpening means adding higher-frequency components. The layer 0 representation has primarily low-frequency Fourier modes (broad peaks); layer 1 adds higher-frequency modes (narrow peaks with pronounced ringing).

This is directly relevant to our project. If Llama 3.1 8B represents digits on manifolds, we might see similar sharpening across layers. Early layers might have broad digit "receptive fields" (a representation that vaguely indicates "this is a large digit") while later layers have sharp fields (precise representation of "this is exactly 7"). Our Phase C already shows that input digit representations are strong across all layers (CV > 0.8), but we have not checked whether the within-digit geometry changes across layers.

## 3.8 Other Counting Representations

The appendix shows that the character count manifold is not unique. Similar rippled manifold structures appear for several other scalar quantities, which I will describe in detail because they strengthen the generalization argument.

### Line width representations

Line width — the total character length of a complete line — is tracked by its own set of features. These features activate on newline tokens and respond to the width of the previous line. Like character count features, they tile the space with overlapping receptive fields and show dilation (wider fields at larger widths).

The paper does not analyze line width manifolds in the same depth as character count manifolds, but the similarity is clear from the feature activation profiles. This matters because line width is a "global" quantity (property of the entire line) while character count is a "local" quantity (property of the current position). The fact that both are represented on similar manifolds suggests that the manifold encoding is a general strategy for scalar quantities, not specific to one type of variable.

### Markdown table row and column indices

The paper analyzes Claude 3.5 Haiku's representation of position within markdown tables. Using a synthetic dataset of 20 markdown tables, the authors find:

- **Row index features:** Features that activate on the "|" separator token, specialized to particular row numbers. These tile the rows with overlapping receptive fields.
- **Column index features:** Similarly, features specialized to particular column numbers.
- **Ringing in probe cosine similarities:** Probes trained to predict row or column index show the characteristic ringing pattern in their pairwise cosine similarities.
- **Baseball-seam shape in 3D PCA:** The row and column index probes, projected into 3D PCA, show the same twisted curve as character count probes.

This is important because markdown table parsing is a fundamentally different task from linebreaking, yet the representations have the same geometry. The model learns the same kind of manifold structure for any scalar positional quantity, regardless of the specific task.

### Token character length in the embedding matrix

Perhaps the most surprising finding is that the embedding matrix itself contains a manifold for token character length. Computing the average embedding vector for each token length (1-14 characters), the authors find that these vectors form a circular pattern in the top 3 PCA components, with oscillations in higher components.

This means the manifold structure exists before any computation happens — it is baked into the embedding matrix. The model has learned to embed tokens in a way that implicitly encodes their character length as a geometric property. This is the raw material that the attention heads then transform into character count estimates.

For our pipeline, this raises the question: does the Llama 3.1 8B embedding matrix encode digit values geometrically? When the model sees the token "7" in a multiplication problem, does the embedding already place it on a manifold that encodes "this is 7, between 6 and 8"? We could test this immediately by computing average embeddings for digit tokens 0-9 and checking for circular/rippled structure.

### Generality of rippled manifolds

This generality is important. Rippled manifolds are not specific to character counting — they appear for any scalar quantity that the model needs to represent with resolution. This supports the hypothesis that Llama 3.1 8B might use similar structures for digit values, carry values, and column sums in arithmetic.

The Gurnee paper, combined with the prior findings of Modell et al. (2025) for colors/dates/years, Nanda et al. (2023) for modular arithmetic values, and Engels et al. (2025) for days/months, suggests a broader principle: **whenever a neural network needs to represent an ordered set of values in a limited-dimensional space, it learns a rippled manifold with Fourier structure.** Testing whether this principle holds for arithmetic variables in Llama 3.1 8B is one of the central goals of our project.

---

---

# How This Paper Fits Our NeurIPS Timeline

We have approximately 4.5 weeks until the NeurIPS 2026 deadline (May 4-6). The Gurnee paper analysis generates several immediate action items:

**This week (before Phase E/F):**
- Run the probe cosine similarity analysis (Section "Methodology Notes" above). This takes hours, not days, and provides an immediate test of Fourier structure.
- Check whether the embedding matrix of Llama 3.1 8B encodes digit tokens on a manifold. This is a one-line PCA computation.

**During Fourier screening (next week):**
- Use the Gurnee paper's Fourier-PCA variance comparison as a quantitative benchmark: Fourier power should be within ~10% of PCA variance.
- Compute Fourier power separately for correct and wrong populations at each difficulty level.

**During causal patching (2-3 weeks out):**
- Adopt the subspace ablation methodology (zero out the digit subspace, measure effect on multiplication accuracy).
- Adopt the mean-patching methodology (replace the digit subspace activation with the centroid for a different digit, verify the output changes).

**During paper writing (overlapping):**
- Cite the Gurnee paper when motivating our geometric approach.
- Use the feature-manifold duality and complexity tax arguments when explaining why our geometric analysis adds value beyond linear probes.
- Compare our Fourier power results directly to the Gurnee paper's ~90% Fourier/PCA ratio.

---

---

# Connection to Our Pipeline

## Direct Implications

### For Fourier Screening (upcoming pipeline step)

The Gurnee paper provides the theoretical foundation for why Fourier screening should work. If digit values (0-9) are represented on a rippled manifold in a low-dimensional subspace, the manifold will have Fourier structure. Our Fourier screening step — computing the Fourier power spectrum of centroid sequences in the merged 18D bases — is directly motivated by this finding.

The paper also tells us what to expect quantitatively: Fourier components should capture within about 10% of PCA variance. If our digit representations have much less Fourier power than this, it might indicate that Llama 3.1 8B uses a different representation strategy than Claude 3.5 Haiku.

Let me make specific numerical predictions based on the Gurnee paper's findings:

**Prediction 1: Digit value centroids should show ringing.** For each input digit variable (a_units, a_tens, b_units, b_tens), compute the 10 centroids (one per digit 0-9) from Phase C, project into the Phase C subspace (up to 9D), and compute the pairwise cosine similarity matrix. If the Gurnee finding generalizes, we should see:
- Neighboring digits (e.g., 3 and 4) should have positive cosine similarity
- Digits 4-5 apart (e.g., 3 and 7) should have negative cosine similarity
- The pattern should show at least one cycle of ringing (positive → negative → positive)

With only 10 digit values (compared to 150 character counts), we have fewer data points and thus expect less pronounced ringing. But the principle should still apply.

**Prediction 2: The top PCA components of digit centroids should correspond to Fourier modes.** For 10 values arranged on a circle (or interval), the dominant Fourier frequencies are 1 (one full cycle) and 2 (two cycles). PC1 of digit centroids should correlate with the frequency-1 component (cosine or sine of 2*pi*digit/10), and PC2 with frequency-2. If this holds, we have direct evidence of Fourier structure in Llama 3.1 8B's digit representations.

**Prediction 3: The effective dimensionality should be much less than 9.** Our Phase C found 9D subspaces for input digits. But the Gurnee paper shows that the intrinsic dimension is 1 (the scalar parameter) while the extrinsic dimension is 6 (the PCA subspace). For 10 digit values, the effective dimensionality should be around 4-6 (not the full 9). If all 9 dimensions carry significant variance, that suggests something more complex than a simple manifold.

**Prediction 4: Correct answers should have cleaner Fourier structure than wrong answers.** This is the central prediction for our paper. If the manifold degradation hypothesis is correct, the Fourier power at frequency 1 should be higher for correct answers than for wrong answers at the same difficulty level. The Gurnee paper does not test this (because linebreaking rarely fails), but the logic is: clean manifold → clean computation → correct answer; degraded manifold → noisy computation → wrong answer.

### For Phase E (Residual Hunting)

The paper shows that the feature-manifold duality implies that some "features" might actually be local coordinates on a manifold. If our Phase E residual hunting finds unexplained structure, we should check whether it corresponds to a continuous variable parameterizing a manifold.

### For Phase F (Between-Concept Principal Angles)

The paper shows that character-remaining and next-token-length representations are arranged in near-orthogonal subspaces, making the decision boundary linearly separable. Our Phase F will compute principal angles between concept subspaces. If we find that carry subspaces and digit subspaces are near-orthogonal, this might indicate a similar computational strategy for making carry decisions linearly separable.

### For GPLVM (future pipeline step)

The paper uses PCA and probing — deterministic methods — to characterize the manifold. Our pipeline uses GPLVM for probabilistic manifold characterization. The Gurnee paper provides ground truth for what the manifold should look like (a rippled 1D curve), which we can use to validate our GPLVM results. If GPLVM on a synthetic dataset with known circular structure recovers the expected manifold, that builds confidence in applying it to our real data.

### For Causal Patching (future pipeline step)

The paper's causal validation methodology — subspace ablation and mean-patching — is directly applicable. We should:
1. Ablate the digit subspace (identified in Phase C) and measure the effect on multiplication accuracy
2. Mean-patch to specific digit values and check if the model's output changes accordingly
3. Test whether the effect is specific to the concept (e.g., ablating the a_tens subspace should affect only digits where the tens digit of a matters)

## What the Paper Cannot Tell Us

1. **Whether Llama 3.1 8B uses similar representations.** The paper studies Claude 3.5 Haiku, a proprietary model with unknown architecture details. Llama 3.1 8B has 32 layers, 4096-dimensional activations, and 32 attention heads per layer. Claude 3.5 Haiku's architecture is not publicly documented. The models were trained on different data, with different objectives (Claude is instruction-tuned; Llama 3.1 8B base is a pure language model). Even if both models perform arithmetic, their internal representations might differ qualitatively.

2. **Whether arithmetic representations have the same clean structure as counting representations.** Character counting is a simple accumulation task: add up the lengths of tokens one by one. Multiplication is far more complex: it involves partial products, column sums, carries, and multi-step accumulation. The representations for a simpler task might be cleaner than for a harder task. Our data already shows this: L5 (5-digit multiplication) has much weaker concept encoding than L2 (2-digit), suggesting that task difficulty degrades representation quality.

3. **Whether the manifold structure is causally relevant for correctness.** The paper shows that the character count manifold is causally important for linebreaking. But our interest is in whether *failure to maintain clean manifold structure* causes arithmetic errors. The paper does not address failure modes in depth. Claude 3.5 Haiku performs linebreaking almost perfectly (high accuracy by the third line for all widths), so there are very few errors to study. Our L5 multiplication has 93.89% error rate (stratified), giving us abundant error data but potentially much messier geometry.

4. **Whether GPLVM will discover anything PCA misses.** The paper's PCA analysis captures 95% of variance in 6 dimensions. It is possible that PCA is sufficient and GPLVM provides no additional benefit. This is a genuine risk for our pipeline. Hauberg would argue that the remaining 5% might matter, and that the uncertainty in the 6D geometry is underestimated by PCA. But the Gurnee paper's success with purely deterministic methods is evidence against the strong version of this claim.

5. **Whether the distributed counting algorithm has an analog in arithmetic circuits.** The character counting algorithm uses attention heads as accumulators. Arithmetic might use MLPs more heavily. Bai et al. found that in toy models, both attention and MLPs contribute to multiplication. The computational architecture might be fundamentally different in production models.

6. **Whether dilation occurs in digit representations.** Character count features show dilation — wider receptive fields at higher counts, analogous to biological Weber-Fechner scaling. Do digit representations show anything similar? With only 10 digit values (vs. 150 character counts), dilation might not be visible. But if, for example, the representations of digits 7, 8, 9 are more overlapping than those of 1, 2, 3, that would be evidence.

7. **Whether the feature-manifold duality applies to carry representations.** Carries are binary or small-integer variables (0 or 1 for single-column carries, 0-17 for our multi-column carries). A binary variable does not need a manifold — it needs two clusters. The manifold framework applies naturally to variables with many values (like character count = 1 to 150). With carries ranging from 0 to 8 at most, the manifold might degenerate to a small number of clusters rather than a smooth curve.

## What We Would Do Differently

If we were designing this study from scratch for our setting (multiplication in Llama 3.1 8B), here is what we would change:

**1. Focus on failure modes from the start.** The Gurnee paper studies a well-functioning behavior. Our paper's central question is about failure. We should design our manifold analysis to compare correct and wrong populations from the beginning, not as an afterthought. Every metric we compute should be computed separately for correct and wrong, and the difference should be our primary outcome variable.

**2. Use multiple models.** The Gurnee paper's limitation of studying only Claude 3.5 Haiku is partly because Claude is proprietary and the tools are specific to it. We work with open-weights models, so we should test our findings on at least one additional model (e.g., Llama 3.1 70B, or Mistral 7B) to check generalization.

**3. Add probabilistic characterization.** The Gurnee paper uses PCA and probes — deterministic methods. We should go further with GPLVM to quantify uncertainty in the manifold geometry. This matters more for us than for Gurnee because our data is sparser (fewer examples per concept value, especially for rare carry values and for the L5 correct population).

**4. Connect geometry to specific error types.** The Gurnee paper treats errors as a binary (correct linebreak vs. incorrect). Our data shows multiple error types: truncation errors, operand echoes, order-of-magnitude errors, even/odd bias, division-by-10 errors. Different error types might correspond to different kinds of manifold degradation. We should analyze each error type separately.

**5. Include cross-layer tracing.** The Gurnee paper analyzes early layers (0-2) and late layers (~90% depth) separately but does not systematically trace how the representation transforms across all layers. We have 9 analysis layers spanning the full model (4, 6, 8, 12, 16, 20, 24, 28, 31), which gives us much better coverage for studying how geometry evolves.

---

---

# Techniques We Can Borrow

1. **Feature-to-manifold discovery pipeline.** Use sparse features (or in our case, known concept labels) to identify relevant variables, then characterize the underlying continuous geometry via PCA and probing. This is our Phase C/D pipeline, and the Gurnee paper validates the approach.

2. **Probe cosine similarity analysis for Fourier structure.** Train supervised probes for each value of a discrete variable (e.g., one probe per digit value 0-9). Compute the pairwise cosine similarity matrix of the probe weight vectors. If the matrix shows ringing (off-diagonal stripes), the representation has Fourier structure. This is a cheap, powerful diagnostic.

3. **QK matrix analysis for manifold manipulation.** For each attention head, compute how the QK circuit transforms the probe vectors. If a head "twists" one set of probes to align with another, it is performing comparison-via-rotation. We can apply this to arithmetic heads — look for heads whose QK circuits align carry-relevant probes with column-sum probes.

4. **Subspace ablation and mean-patching for causal validation.** Zero-ablate a concept subspace and measure the effect on task performance. Mean-patch to specific concept values and verify that the output changes predictably.

5. **Physical simulation for optimality intuition.** The attractive-repulsive force model on a hypersphere provides a simple way to generate "expected" manifold geometry for any given number of values and subspace dimensionality. We can compare our observed geometry to this expectation.

6. **Adversarial probing via "illusions."** Once we understand the mechanism, construct adversarial inputs that exploit specific failure modes. In arithmetic, this could mean designing problems that trigger specific carry errors by mimicking common carry-free patterns.

---

---

# Detailed Methodology Notes for Reproduction in Our Pipeline

The following notes translate the Gurnee paper's methodology into concrete steps for our Llama 3.1 8B multiplication pipeline.

## Step-by-step: Probe Cosine Similarity Analysis

**Objective:** Test whether digit representations show ringing (Fourier structure).

**Input:** Activations at layer 16 (our best layer for input digits, based on Phase D), for all L3 problems (which have sufficient correct and wrong samples).

**Procedure:**

1. For each input digit variable (a_units, a_tens, b_units, b_tens), split the activations by digit value (0-9).

2. For each digit value v, compute the centroid: mu_v = mean of all activations with that digit value. This gives 10 centroids per variable, each in R^4096.

3. Project the centroids into the Phase C subspace (up to 9D) for that variable.

4. Compute the 10×10 pairwise cosine similarity matrix: S_{ij} = cos(mu_i, mu_j) in the subspace.

5. Plot S as a heatmap. Look for:
   - Diagonal peak (S_{ii} = 1 by construction)
   - Positive off-diagonal band (neighbors 1-2 apart should be positive)
   - Negative band (digits 3-5 apart might be negative)
   - Positive again (digits 7-9 apart, wrapping around, might be positive — this would indicate circular/modular structure)

6. If ringing is present, quantify it by fitting a sinusoidal model to S_{0,j} as a function of j: S_{0,j} = A * cos(2*pi*j/10 + phi) + residual. The amplitude A and the R^2 of this fit measure the strength of the Fourier structure.

7. Repeat for correct-only and wrong-only subsets. Compare the Fourier amplitude A between correct and wrong.

**Expected runtime:** Minutes on a single CPU. No GPU needed. We already have all the data.

**Critical details:** 
- Use the Phase C subspace (centroid-SVD), not the full 4096D space. The ringing should be visible in the subspace.
- Normalize centroids to unit norm before computing cosine similarity. This removes the effect of norm variation across digit values.
- The 10×10 matrix for digits 0-9 on a line (not a circle) might not show clean ringing for the endpoints (0 and 9). If the model treats digits linearly (0 < 1 < 2 < ... < 9) rather than circularly (9 wraps to 0), the ringing will be on an interval, not a circle. The Gurnee paper discusses this distinction: interval topology shows similar ringing but without the wraparound.

## Step-by-step: QK Rotation Analysis for Carry Detection

**Objective:** Search for attention heads that perform comparison-via-rotation, analogous to boundary heads.

**Input:** The weight matrices W_Q and W_K for all 32 layers × 32 heads = 1024 attention heads of Llama 3.1 8B.

**Procedure:**

1. Train probes for column sum values (e.g., column_sum_2 ranging from 0 to 81 for L3) and carry values (e.g., carry_2 ranging from 0 to 8).

2. For each attention head h, compute the QK matrix W_QK = W_Q^T * W_K.

3. Compute the transformed probe vectors: q_i = W_Q * column_sum_probe_i, k_j = W_K * carry_probe_j.

4. Compute the cosine similarity matrix between transformed column sum probes and carry probes: T_{ij} = cos(q_i, k_j).

5. Look for heads where T shows alignment structure: column sum i should align with carry threshold values. If a head aligns column_sum = 10 with carry = 1, column_sum = 20 with carry = 2, etc., it is performing carry computation via rotation.

6. Validate by ablating the identified heads and measuring the effect on carry accuracy.

**Expected runtime:** Hours on GPU. The QK matrix computation is fast, but training probes for all column sum and carry values across all layers requires some work.

**Critical details:**
- Not all attention heads will show structure. The Gurnee paper found that only a few specific heads (out of many) serve as boundary heads. We might find that only 2-3 heads in Llama 3.1 8B are responsible for carry computation.
- The column sum range (0-81 for two 2-digit partial products) is much larger than the character count range normalized by line width. We might need to bin column sums into coarser groups.
- The carry decision is more complex than the boundary detection: it is not just "close to threshold" but "above or below threshold." The QK structure might look different from the Gurnee paper's smooth offset.

---

---

# Research Ideas Generated

## Idea 1: Rippled Manifold Test for Digit Representations

**What:** Apply the probe cosine similarity analysis from the Gurnee paper to our digit representations. For each digit variable (a_units, a_tens, b_units, b_tens), train 10 probes (one per digit value 0-9) on the activation at the best layer. Compute pairwise cosine similarities. Check for ringing.

**Why interesting:** This is the most direct test of whether the Gurnee findings generalize to arithmetic representations in a different model. If ringing is present, it confirms Fourier structure for digit representations. If absent, it means Llama 3.1 8B uses a fundamentally different encoding.

**Feasibility:** Very high. We already have the activations, the labels, and the subspaces from Phases C and D. Training logistic probes for each digit value is straightforward. Computing cosine similarities is trivial. This could be done in a few hours.

**Possible outcomes:**
- **Strong ringing (1+ full cycles):** Clear evidence of Fourier structure. We proceed to quantitative Fourier screening with high confidence.
- **Weak ringing (partial cycle, low amplitude):** Some Fourier structure but degraded. Might indicate that the model uses a mix of linear and circular encoding.
- **No ringing (monotonic similarity decay):** The model uses a linear number line rather than a circular/Fourier encoding. This would be a negative result for the Fourier universality hypothesis but still publishable and important. It would redirect us to focus on linear subspace geometry rather than manifold geometry.
- **Different pattern for different digits:** If a_units shows ringing but a_tens does not, this would suggest that the model treats different digit positions differently — perhaps using Fourier encoding for the operands' units digit (which participates most directly in multiplication) but linear encoding for tens digits. This would be a subtle and interesting finding.

## Idea 2: Boundary Head Search for Carry Computation

**What:** Search for attention heads in Llama 3.1 8B whose QK circuits align carry-relevant probes with column-sum probes, analogous to the boundary heads in Gurnee.

**Why interesting:** If we find such heads, we would have mechanistic evidence for how the model decides whether a carry occurs — by rotating the column sum manifold to compare against a threshold, just as Gurnee's boundary heads rotate character count to compare against line width.

**Feasibility:** Medium. This requires training probes for column sum values, computing probe transformations through QK matrices of all attention heads, and identifying heads with alignment structure. The QK analysis is computationally intensive but feasible.

**Detailed methodology:**
1. Identify the relevant layer range. Our Phase D shows carries are best encoded around layers 12-20. Focus on this range.
2. For each layer in this range, train 10-20 probes: one for each plausible column sum value in the most informative column (e.g., column 2 for L4/L5 problems, where max carry is 17).
3. Train separate probes for each carry value (0-8 for carry out of column 2).
4. For each of the 32 attention heads in each target layer, compute the QK-transformed probe vectors and their pairwise cosine similarities.
5. Look for the distinctive "offset diagonal" pattern: column_sum probe i should align with carry probe floor(i/10) after QK transformation.
6. If found, validate by ablating the head and measuring the impact on carry prediction accuracy (using our Phase D probes as ground truth).

**Risk assessment:** The main risk is that carry computation in multiplication might not use a boundary-head-like mechanism at all. Unlike linebreaking (where the boundary is continuous and the comparison is "close to"), carry computation is a discrete threshold (column_sum >= 10 means carry = 1). Discrete thresholds might be implemented by MLPs rather than attention heads. If the QK analysis finds nothing, we should not conclude that carry computation is ungeometric — it might just use a different mechanism.

## Idea 3: Correct vs. Wrong Manifold Curvature Comparison

**What:** For correct and wrong answers separately, compute the character-count-style manifold for digit representations. Compare the curvature (effective dimensionality, Fourier power) between correct and wrong populations.

**Why interesting:** The Gurnee paper shows that cleaner manifold structure enables more reliable computation. If correct answers have higher Fourier power (cleaner manifold) than wrong answers for the same digit representation, that provides evidence that manifold degradation causes arithmetic failure. This is a central claim of our paper.

**Feasibility:** High. We have the correct/wrong splits from our data generation. We need to compute per-digit-value centroids separately for correct and wrong populations, project into subspaces, compute Fourier power, and compare. The main concern is sample size for the wrong-correct split at L5 (small correct population).

**Statistical considerations:** 
- L3: 67.2% correct → ~6,720 correct, ~3,280 wrong. Both subsets are large enough for robust centroid estimation.
- L4: 29.0% correct → ~2,900 correct, ~7,100 wrong. Correct subset is smaller but still adequate.
- L5: 6.11% true accuracy → ~4,197 correct (from our carry-balanced dataset), ~118,026 wrong. Massive imbalance. Correct subset is adequate for centroid estimation, but each per-digit-value centroid has only ~420 samples on average. At this sample size, the centroid estimates might be noisy, and we should compute confidence intervals (bootstrapping).

**Key metric:** The "Fourier degradation ratio" = P1(correct) / P1(wrong), where P1 is the normalized power at Fourier frequency 1. If this ratio is consistently > 1 across digit variables and layers, that is strong evidence for the manifold degradation hypothesis. If it is ~1 (no difference) or < 1, the hypothesis is not supported.

## Idea 4: Dilation Analysis in Carry Representations

**What:** Check whether carry representations show dilation (wider receptive fields at higher values), analogous to the dilation observed in character count features.

**Why interesting:** Biological number perception shows Weber-Fechner scaling (discrimination gets harder at larger numbers). Character counting in Claude shows the same pattern. If carry representations in Llama 3.1 8B show dilation, it would suggest a universal principle of how neural networks (biological and artificial) represent quantities.

**Feasibility:** Medium. This requires analyzing how carry representation quality (Phase D eigenvalues) varies with carry value. We have carries from 0 to 8 for some columns, and we have the subspace projections. The analysis is straightforward but the interpretation might be confounded by the uneven distribution of carry values.

**How to measure dilation:** For each carry value v (0, 1, 2, ..., 8), compute the average activation of the "carry = v" cluster. Then compute the within-cluster variance (how spread out activations with carry = v are). If dilation exists, the within-cluster variance should increase with v — carry = 7 should have more spread than carry = 1. An alternative measure: train separate binary probes for "carry = v vs. carry != v" and measure the probe width (RMSE). If dilation exists, probes for larger v should have larger RMSE.

**Confound:** The carry value distribution is highly skewed. Our data shows that low carries (0-2) are far more common than high carries (5-8). The apparent "dilation" at high carries might just be noise from small sample sizes, not a genuine widening of receptive fields. We need to control for sample size by comparing dilation measurements to bootstrap null distributions.

## Idea 5: Orthogonality Test for Answer Digit Subspaces

**What:** Test whether the subspaces for different answer digits (ans_digit_0 through ans_digit_5) are near-orthogonal, analogous to the orthogonality between characters-remaining and next-token-length in Gurnee.

**Why interesting:** If answer digit subspaces are orthogonal, it would mean the model can represent all answer digits simultaneously without interference. If they overlap (superposition), it might explain why middle digits (which our Phase D shows are weakly encoded) are the hardest to predict correctly.

**Feasibility:** High. Phase F (between-concept principal angles) is designed exactly for this. We just need to run it and look at the angles between answer digit subspaces specifically.

**Predictions based on Gurnee:**
- **Edge answer digits (ans_digit_0, ans_digit_5):** These are the most and least significant digits. Phase D shows they have moderate encoding strength (lambda = 0.84-0.86 at L5). We predict they should have near-orthogonal subspaces because they serve different computational roles.
- **Middle answer digits (ans_digit_1 through ans_digit_4):** Phase D shows these are very weakly encoded (lambda = 0.05-0.17 at L5). If they are in superposition (overlapping subspaces), this would explain the weak encoding — the model cannot cleanly separate the middle digits because their subspaces interfere.
- **Input digits vs. answer digits:** We predict near-orthogonality between input digit subspaces and answer digit subspaces at early/middle layers (where inputs are strong and answers are not yet computed), but increasing overlap at late layers (where the answer representation is being constructed from the input representations).

## Idea 6: Embedding Matrix Check for Digit Token Geometry

**What:** Check whether the Llama 3.1 8B embedding matrix encodes digit tokens (the tokens for "0" through "9") on a manifold with Fourier structure.

**Why interesting:** The Gurnee paper shows that Claude 3.5 Haiku's embedding matrix already contains a manifold for token character length. If Llama 3.1 8B's embedding matrix contains a digit manifold, this would mean the geometric structure exists from the very start of processing — before any attention or MLP computation.

**Feasibility:** Very high. Extract the embedding vectors for the 10 digit tokens from the embedding matrix W_E. Compute 3D PCA. Check for circular/rippled structure.

**Expected runtime:** Seconds. This is literally extracting 10 rows from a matrix and running PCA on 10 points in 4096D.

**Possible outcomes:**
- **Circular structure in embeddings:** The digit tokens already lie on a circle in the embedding space. This would be strong evidence for universal Fourier structure and would suggest that the model starts with geometric structure rather than building it from scratch.
- **No structure in embeddings:** The digit tokens are arranged arbitrarily in the embedding space. The model builds geometric structure through computation rather than inheriting it from embeddings. This is also informative — it would mean our Phase C/D subspace analysis captures structure that is *created* by the model, not just *inherited* from the input.
- **Linear structure in embeddings:** Digit tokens are arranged along a line (0-1-2-...-9) in the embedding space, but not on a circle. This would support the linear representation hypothesis over the Fourier/manifold hypothesis for the base encoding, while leaving open the possibility that subsequent layers transform the linear encoding into a circular one.

---

---

# Section-by-Section Critical Analysis

## Introduction: Grade A

The introduction does several things well. It frames the task in terms of perception rather than computation, drawing an analogy to biological sensory systems. This framing is not just rhetorical — it leads to genuine scientific connections (place cells, boundary cells, visual illusions) that enrich the analysis.

The attribution graph gives a clear overview of the computation: token lengths are computed, character counts are accumulated, line width is tracked, characters remaining are estimated, and the newline decision is made. This sets up the entire paper by showing what variables are involved before diving into the geometry of each.

The list of key findings is unusually clear for an interpretability paper: four questions (how to represent, how to detect boundaries, how to decide, how to construct) with four corresponding answers.

The framing of "text as a visual medium" is subtle but important. The paper argues that language models do not just process text as sequences of meanings — they also process text as a visual/spatial artifact with properties like line length, column position, and whitespace. This "perceptual" framing opens up a rich space of natural tasks that are under-studied in interpretability. Most interpretability work focuses on semantic processing (meaning, reasoning, knowledge); the Gurnee paper shows that "low-level perception" of text is also rich and mechanistically tractable.

One small criticism: the introduction could have been more explicit about what is known vs. not known. The authors mention prior work on position encoding and number representation but do not clearly state: "nobody has previously shown manifold manipulation in a production model." This claim is implicit in the framing but never stated directly.

## The Discovery Story — Methodological Lessons

The paper's description of how the findings were discovered is unusually candid and instructive. The authors describe a three-phase discovery process:

**Phase 1: Failed approach (probing and patching).** The team initially tried to understand linebreaking using standard interpretability tools: train linear probes, do activation patching, measure effects. This did not work well because they "did not understand what we were looking for" (they did not know to distinguish line width from character count), "where to look for it" (they did not expect line width to be represented only on newline tokens), or "how to look for it" (they started with 1D linear regression probes, missing the multi-dimensional manifold structure).

This is a crucial lesson for our pipeline. If we had started with 1D probes for "what digit is this?", we might have found the linear representation (R-squared 0.985 for character count, likely similarly high for digit values) and stopped there. The 1D probe tells you the information is present but misses the geometric structure that makes computation possible.

**Phase 2: Feature discovery via crosscoders.** The team then used unsupervised sparse features to find the relevant variables. The crosscoder features identified character count features, line width features, boundary detection features, and break predictor features. But even after finding these features, "we were also confused by what they were representing." The features looked like they were "vaguely about newlines and linebroken text" but their differences were not obvious from looking at activating examples.

This is the "complexity tax" in action: the features gave a true but confusing description. There were dozens of newline-related features, each slightly different, with no clear organizing principle.

**Phase 3: Synthetic data reveals the geometry.** The breakthrough came when the team tested the features on synthetic datasets with controlled line widths. On synthetic data, the features' receptive fields became clear: this feature fires at character count 35-55, that one at 45-65, and so on. The overlapping receptive fields pointed toward a continuous manifold, and PCA revealed the curve.

The lesson for our pipeline: we should use synthetic or controlled inputs to characterize our features/probes, just as the Gurnee team used synthetic wrapped text. For multiplication, we already have a controlled dataset with known ground truth for every sub-computation. We should use this data to characterize the receptive fields of any features or probes we train. If a probe for "carry = 1" also responds partially to "carry = 0" and "carry = 2," that overlap might indicate manifold structure.

**Phase 4: Geometry simplifies the interpretation.** Once the manifold was identified, the previously confusing features made sense: they were local coordinates on a curve. The boundary heads made sense: they were rotating manifolds. The distributed counting algorithm made sense: each head contributed a ray, their sum produced curvature. The geometric description unified all the separate feature-level observations into a coherent story.

This progression — confusion → features → synthetic data → geometry → understanding — is a template we can follow. We are currently at the "features" stage (our Phase C/D subspaces play the role of Gurnee's crosscoder features). The next step is to characterize the geometry within these subspaces (our Fourier screening and GPLVM steps). If the geometry is clean, it should simplify our understanding just as it did for Gurnee.

## Representing Character Count: Grade A

This is the strongest section. The progression from features to probes to manifold to optimality is logical and well-supported at each step. The causal validation (ablation and mean-patching) is clean and convincing. The physical simulation is a delightful addition that makes the optimality argument intuitive.

One minor weakness: the paper claims the 6D subspace captures 95% of variance but does not report how much variance PCs 7-10 capture. If PC7 captures 3%, that is a qualitatively different situation than if it captures 0.01%. The distinction matters because it tells us whether the manifold is truly 6D or whether we should be looking at higher dimensions.

## Sensing the Line Boundary: Grade A-

The boundary head twist mechanism is beautiful, and the cosine similarity visualizations are immediately convincing. The multi-head stereoscopic algorithm is well-explained, and the causal validation of the characters-remaining subspace is solid.

The weakness here is that the analysis of the QK transformation is purely descriptive. The paper shows *that* the QK matrix rotates one manifold to align with the other, but does not explain *why* the QK matrix learned this rotation. What is the training signal? How does gradient descent discover this solution? These questions are beyond the paper's scope, but they matter for understanding whether the mechanism is a general principle or a quirk of this specific model.

## Predicting the Newline: Grade B+

The orthogonal subspace finding is clean, and the linear separability is well-demonstrated. The AUC of 0.91 is good but not overwhelming — what accounts for the remaining 9%? The paper attributes it partly to the 3D classifier error and partly to Haiku's imperfect estimation of the next token, but does not quantify these separately.

The paper also acknowledges that the model does not seem to use a comprehensive mechanism for handling all possible next tokens (only the most likely one). This is an interesting limitation that suggests the linebreaking mechanism is a heuristic, not a full solution.

## A Distributed Character Counting Algorithm: Grade B+

This is the most complex section and the one where the reader is most likely to lose the thread. The QK and OV circuit analysis for individual heads is detailed but dense. The key insight — each head produces a ray, their sum produces a curve — is important but somewhat buried in the technical details.

The walkthrough of L0H1 is helpful but could benefit from a clearer diagram showing how the attention-to-newline vs. attention-to-content phases combine with the OV correction term to produce the final output. The current figure is informative but requires careful study.

## Visual Illusions: Grade A

This section is the paper's most creative and memorable contribution. The idea of constructing "visual illusions" for language models by hijacking specific attention mechanisms is both scientifically rigorous and aesthetically delightful. The systematic testing of 180 two-character sequences strengthens the finding beyond a single anecdote. The correlation between attention disruption and newline probability modulation provides clean mechanistic evidence.

The connection to human visual illusions is appropriately cautious and scientifically interesting. The paper resists the temptation to overclaim and instead frames it as "suggestive parallels" involving contextual cue modulation of perceptual estimates.

## Discussion: Grade A-

The discussion themes are well-chosen: naturalistic behavior, utility of geometry, unsupervised discovery, feature-manifold duality, complexity tax, and calls for methodology and biology. Each is grounded in specific observations from the paper.

The "complexity tax" framing is particularly valuable for the field. It articulates a real problem — features give a true but complex description, and we need additional structure (manifolds, hierarchies, macroscopic patterns) to simplify the interpretation. This is an honest assessment of the current state of interpretability.

---

---

# Comparison with Other Papers in Our Project Knowledge

## vs. Bai et al. (Multiplication Circuits)

Bai et al. study multiplication in tiny 2-layer models. They find Fourier-basis digit representations and pentagonal prism geometry. Gurnee et al. study a natural counting task in a production model. Both find geometric structure (rippled manifolds / pentagonal prisms), but at very different scales.

**Key difference:** Bai et al. work with models designed to solve a specific task perfectly. Their models are 2-layer transformers with a few hundred dimensions, trained from scratch on multiplication until they achieve near-100% accuracy. The geometric structure they find is clean and precise because the model has been trained to convergence on a single task. Gurnee et al. work with Claude 3.5 Haiku, a billion-parameter model that learned linebreaking incidentally during pretraining on a massive corpus. Despite this messy training process, the geometric structure is still clean. This makes the Gurnee finding more relevant to our pipeline because Llama 3.1 8B, like Claude, is a pretrained model that learned arithmetic incidentally.

**Key connection:** Both papers show that Fourier structure is present in neural representations of scalar quantities. Bai et al. show it in digit values; Gurnee et al. show it in character counts. The Fourier structure appears in both settings despite completely different model architectures, training procedures, and tasks. This suggests it might be a universal principle — any time a neural network needs to represent an ordered set of values in a limited-dimensional space, Fourier features emerge. Our Fourier screening step tests whether this extends to digit values in Llama 3.1 8B.

**Specific numerical comparison:** Bai et al. find that digit representations in their toy model live on 5-dimensional pentagonal prisms (the vertices of a regular pentagon, embedded in 5D via Fourier modes at frequencies 1 and 2). For 10 digit values, Fourier theory predicts that the dominant modes are frequencies 1, 2, 3, 4, 5 — each requiring 2 dimensions (sine and cosine), minus 1 for the DC component, giving 9 dimensions for perfect representation. But most of the variance is captured by frequencies 1 and 2, which use about 4-5 dimensions. Gurnee et al. find that 150 character count positions are captured in 6 dimensions. The ratio is roughly similar: both tasks use subspaces that are much smaller than the number of distinct values but large enough to accommodate several Fourier modes.

## vs. Li et al. (Fourier Addition Circuits)

Li et al. prove that Fourier features are optimal for modular addition in certain settings. Gurnee et al. show empirically that Fourier-like structure emerges in character counting in a production model. Li et al. provide the theoretical justification; Gurnee et al. provide the empirical evidence from a realistic setting.

**Key connection:** Li et al.'s theory predicts that Fourier features should emerge whenever the task requires distinguishing values modulo a period. Character counting has a natural periodicity (the line width), and digit arithmetic has periodicity 10. The convergence of theory (Li) and empirical evidence (Gurnee) strengthens our confidence in finding Fourier structure in Llama 3.1 8B.

**Important caveat:** Li et al.'s theory assumes polynomial activations and uniform class weights, neither of which holds in practice. The theory is about what is *optimal*, not what neural networks actually *learn*. The Gurnee paper provides evidence that networks do in fact learn something close to the theoretical optimum, which validates the theory's predictions for practical settings. But we should not assume that Llama 3.1 8B's digit representations will be perfectly Fourier — they might be "mostly Fourier" with deviations due to non-uniform digit distributions, carry interactions, or other confounds.

**Another connection:** Li et al. identify that the number of neurons required to represent all Fourier frequencies grows quadratically with the number of frequencies. This creates a capacity constraint. For modular addition mod p, you need O(p^2) neurons. For multiplication of k-digit numbers, the capacity requirements are even higher. This could explain why Llama 3.1 8B's multiplication accuracy drops sharply with digit count: the model might not have enough capacity to represent all the Fourier modes needed for 5-digit multiplication.

## vs. REMA (Reasoning Manifold)

REMA claims to analyze "reasoning manifolds" but actually just computes kNN distances to point clouds. Gurnee et al. do genuine manifold analysis: they find the manifold (1D curve), characterize its geometry (rippled/Fourier), explain how it is constructed (distributed attention), and show how it is manipulated (boundary head twist).

Let me be specific about the contrast, because it is instructive for our project:

**What REMA does:** Collects hidden states from correct and incorrect reasoning traces, computes kNN distances from each hidden state to the set of correct hidden states, calls this distance a "manifold deviation score," and shows that error traces have larger deviation scores.

**What Gurnee et al. do:** Identifies specific scalar variables the model represents (character count, line width, characters remaining). Finds the subspace for each variable (6D for character count). Characterizes the geometry within that subspace (rippled 1D curve with Fourier structure). Discovers how the model constructs the representation (distributed attention heads producing rays that sum to curves). Discovers how the model computes with the representation (boundary head twist for comparison via rotation). Validates causally (ablation, patching, visual illusions).

The difference is between observing ("error states are far from correct states") and understanding ("the model computes by rotating a manifold, and errors occur when the rotation is disrupted"). Observation is necessary but insufficient.

**Specific contrasts that matter for our pipeline:**
- REMA uses mean pooling across all token positions, destroying temporal information. Gurnee analyzes specific token positions where computations happen. Our pipeline extracts at the "=" position — better than mean pooling but still a single position.
- REMA uses instruction-tuned models. Gurnee uses a model performing a natural pretraining task. We use a base model doing pretraining-adjacent arithmetic. Our setting is closer to Gurnee's.
- REMA's kNN distance is global and non-specific. Gurnee's probe analysis is concept-specific. Our Phase D eigenvalues are concept-specific (one per concept per layer). Specificity enables understanding.

**Key lesson for our project:** REMA is the cautionary example of how NOT to do geometric interpretability. Gurnee et al. is the positive example of how TO do it. The difference: REMA uses loose geometric vocabulary without geometric methods; Gurnee et al. uses specific geometric tools (PCA, probe analysis, QK transformation analysis) to make precise geometric claims. Our pipeline should follow Gurnee's approach, not REMA's.

## vs. Hauberg 2018 (Only Bayes Should Learn a Manifold)

Hauberg argues that deterministic methods for manifold learning are biased and that probabilistic methods (GPLVM) should be preferred. Gurnee et al. use entirely deterministic methods (PCA, probing) and get clean, convincing results. Is this a contradiction?

Not quite. Hauberg's concern is most relevant when data is sparse relative to the manifold complexity. For character counting, there are abundant data points at every count value (the synthetic dataset provides many examples per count). The deterministic methods work well because the data density is high.

Let me quantify this more precisely. In the Gurnee paper's setting, there are 150 count values, each with thousands of examples from the synthetic dataset. The data density per manifold position is very high. The subspace is 6-dimensional. The labels are clean (character count is exactly known). And the model performs the task very well, so there is low intrinsic noise in the representations.

Compare this to our pipeline's setting. We have 10 digit values, each with 1,000 to 15,000 examples per level, which is still good data density for the input digits. But for carries, we have only 9 possible values (0-8), with extremely skewed distribution where carry=0 dominates. Rare carry values (5, 6, 7, 8) might have only 10-100 examples at some difficulty levels. This is the sparse regime where Hauberg's probabilistic correction matters.

For L5/correct specifically, we have only about 4,197 total samples divided across concept values. This means roughly 420 per digit value for input digits (moderate density) but far fewer for rare carry values. At these sample sizes, centroid estimates become noisy, and PCA-based geometry might be misleading.

This suggests a stratified approach for our pipeline: use PCA (fast, simple) for well-sampled concepts like input digits and product_binned, and reserve GPLVM (slow, principled) for sparse concepts like rare carry values and middle answer digits at L5/correct. This is more efficient than running GPLVM on everything.

There is also a deeper philosophical point. Hauberg's 2018 argument is about *learning* a manifold from data in an unsupervised way. In the Gurnee paper, the manifold is not learned unsupervised — it is discovered by averaging activations within groups defined by known labels. The averaging eliminates within-group variation, producing clean centroids on a smooth curve. This is closer to supervised geometry characterization than to unsupervised manifold learning.

Our Phase C pipeline follows the same logic: group by concept label, compute centroids, find the subspace that captures inter-centroid variation. Hauberg's critique of unsupervised manifold learning is less directly applicable to this supervised setup. However, when we move to GPLVM (which does unsupervised manifold learning within identified subspaces), Hauberg's concerns become fully relevant again.

The key distinction is that centroid-based analysis (Phase C, Gurnee's PCA) characterizes the *mean* geometry, while GPLVM characterizes the *distribution* of points on the manifold, including the scatter and uncertainty. Both are valid but answer different questions. Centroid analysis asks: "where is digit=7 on average?" GPLVM asks: "what does the region around digit=7 look like, and how uncertain is its boundary?" For our correct-vs-wrong comparison, the scatter matters: wrong answers might have the same centroid positions but much wider scatter, which PCA would miss but GPLVM would capture.

## vs. MOLT (Sparse Mixtures of Linear Transforms)

MOLT shows that geometric computations (like rotating a circle by a fixed amount) require one MOLT transform but hundreds of transcoder features. Gurnee's boundary head twist is exactly such a rotation. The MOLT paper explains *why* features fragment geometric computation (the "shattering" problem); Gurnee et al. demonstrate a specific case and show the geometric alternative.

**Key connection:** MOLT validates the geometric view of computation. If we find that multiplication circuits in Llama 3.1 8B involve geometric operations (rotations, translations on manifolds), MOLT provides the theoretical framework for why these operations are more natural than feature-level descriptions.

To make this concrete: suppose we discover that a specific attention head in Llama 3.1 8B rotates the digit-value manifold by 36 degrees (one digit value) as part of carry propagation. In the MOLT framework, this rotation would be described by a single sparse linear transform. In the transcoder/SAE framework, the same rotation would be described by hundreds of feature interactions: "feature 'digit=3' is suppressed while feature 'digit=4' is activated," and so on for every digit value. The MOLT description is simpler because it captures the underlying geometric operation directly.

The Gurnee paper's "complexity tax" argument is essentially the same point that MOLT makes: features describe the truth but at unnecessary complexity. Geometry provides a simpler description. Both papers converge on the conclusion that the field needs tools for discovering and describing geometric structure, not just features.

**Difference:** MOLT proposes a specific solution (learning sparse linear transforms between layers). The Gurnee paper proposes a different approach (manual geometric analysis guided by feature discovery). Our pipeline proposes a third approach (systematic subspace identification followed by probabilistic manifold characterization). All three approaches are complementary: MOLT automates the discovery of transforms, Gurnee provides deep case studies, and our pipeline provides systematic quantitative characterization.

---

---

# Deeper Connections to Neuroscience

The paper draws extensive parallels to biological neural systems. These connections deserve careful treatment because they are more than just analogies — they suggest shared computational principles.

## Place Cells and Character Count Features

Place cells are neurons in the hippocampus of rats and other mammals that fire when the animal is at a specific location in its environment. Each place cell has a "place field" — a spatial region where it is active. Place fields overlap, so at any location, multiple place cells are active. The population activity uniquely identifies the animal's position.

The character count features in Claude 3.5 Haiku have the same structure: each feature has a "receptive field" — a range of character counts where it is active. The receptive fields overlap, and the set of active features uniquely identifies the character count. The parallel is not just structural: both systems solve the same problem (representing position on a 1D track) under similar constraints (limited number of neurons/features, need for resolution between adjacent positions).

There is even a quantitative parallel: biological place cells show dilation — place fields get wider at larger distances from a reference point (the "start" of a track). The character count features show the same dilation: features at higher character counts have wider receptive fields. This Weber-Fechner scaling is a signature of logarithmic number encoding in biological brains, and it appears spontaneously in a neural network trained on text.

**Devil's advocate:** The parallel might be superficial. Place cells exist in a 3D brain with specific connectivity constraints (hippocampal architecture, recurrent connections to entorhinal cortex). Character count features exist in a transformer with completely different architecture (residual stream, attention heads, no recurrence). The fact that both produce overlapping receptive fields might just reflect that overlapping receptive fields are the generic solution to 1D position encoding, regardless of architecture. It does not imply any deep architectural similarity between brains and transformers.

## Boundary Cells and Boundary Heads

Boundary cells in the entorhinal cortex fire at specific distances from environmental boundaries (walls, edges). The Gurnee paper's boundary detection features are analogous: they activate when the character count approaches the line width, i.e., when the "position" approaches the "boundary."

The computational parallel goes deeper. In both systems, boundary detection requires comparing two quantities: current position and boundary location. Biological boundary cells are thought to compare place cell activity (position signal) with some representation of the boundary location. The transformer's boundary heads compare the character count manifold with the line width manifold via rotation in the QK space. Both implement comparison of spatial signals, though the mechanisms differ in their specifics.

## What the Neuroscience Connections Mean for Our Pipeline

These connections suggest that neural networks, whether biological or artificial, converge on similar solutions for representing and computing with scalar quantities. If this convergence principle holds for digit values and carries in arithmetic, we should expect:

1. Digit representations to be organized as overlapping receptive fields on a manifold (like place cells)
2. Carry computation to involve comparison of two manifolds (like boundary cells comparing position to boundary)
3. Dilation to appear in any representations that span a wide range of values

These predictions are testable with our existing data.

---

---

# Glossary of Key Concepts

The following terms are used throughout this analysis and the Gurnee paper. Defining them precisely helps avoid confusion.

**Feature manifold:** A low-dimensional, continuous geometric structure in the residual stream that represents a scalar quantity. Parameterized by the scalar value (e.g., character count), the manifold is a curve (1D) or surface (2D+) in a low-dimensional subspace of the full activation space. Distinguished from a "linear representation" (which uses a single direction) and from "discrete features" (which use a collection of separate directions).

**Rippled embedding:** The specific geometry that arises when you embed many distinguishable positions into a small subspace. The curve spirals or twists through the subspace instead of following a simple arc. The rippling increases the "path length" of the curve, allowing more positions to be distinguished while keeping the subspace dimensionality low. Mathematically related to truncated Fourier series.

**Ringing:** The oscillatory pattern in the cosine similarity matrix between different positions on a rippled manifold. Neighboring positions have positive similarity; positions ~5 apart have negative similarity; positions ~10 apart have positive similarity again. The period and amplitude of the ringing depend on the ratio of positions to subspace dimensions.

**Dilation:** The phenomenon where receptive fields (feature activation ranges or probe response widths) widen at larger values. In character counting, features at position 100 respond to a wider range of counts than features at position 10. This is analogous to Weber-Fechner law in biological perception.

**Boundary head twist:** The mechanism by which an attention head rotates one manifold to align with another, enabling comparison via dot product. The QK matrix of the head implements the rotation. The rotation aligns position i on one manifold with position i+epsilon on the other, so the dot product is large when the two quantities differ by epsilon.

**Complexity tax:** The additional interpretive burden created by describing a continuous computation in terms of many discrete features. If a behavior can be described as "a rotation of a manifold" but the feature-level description requires specifying hundreds of feature interactions, the feature description imposes a complexity tax. The manifold description "pays down" this tax.

**Feature-manifold duality:** The observation that the same representation can be described either as a collection of discrete sparse features or as a continuous manifold. The features tile the manifold in a canonical way, providing local coordinates. The duality is exact when the feature reconstruction matches the manifold, and approximate otherwise.

**Distributed computation:** The architectural constraint that a single attention head cannot produce curved manifold representations (because its output is a linear function of its inputs). Curvature requires the cooperation of multiple heads, each contributing a low-rank component that together build the full geometry.

**Characters remaining:** The scalar quantity equal to line width minus character count. This quantity determines whether the next word fits on the current line. The Gurnee paper shows it is represented in a 2D subspace created by the summed outputs of multiple boundary heads.

**Crosscoder / Weakly Causal Crosscoder (WCC):** The dictionary learning method used in this paper to discover sparse features. A crosscoder learns a dictionary of features that can reconstruct residual stream activations as a sparse linear combination of decoder vectors. "Weakly causal" refers to a variant that respects the causal structure of the computation. The paper uses a 10 million feature WCC dictionary trained on Claude 3.5 Haiku.

---

---

# Questions I Would Ask the Authors

If I could sit down with Gurnee, Batson, and Olah, these are the questions I would ask. Each question is designed to extract information relevant to our pipeline.

**Q1: Have you looked for similar manifold structure in arithmetic circuits?**
The linebreaking task requires counting — a close relative of addition. Has the team investigated whether Claude 3.5 Haiku uses similar manifold representations for arithmetic? If the team has preliminary data (positive or negative) about arithmetic manifolds, it would directly inform our Llama 3.1 8B analysis.

**Q2: How sensitive is the manifold structure to the quality of the sparse dictionary?**
The paper uses a 10M feature crosscoder. Would the manifold discovery work with a 1M or 100K feature dictionary? Or does the manifold only become visible when the feature resolution is high enough to resolve the individual "place cell" features? This matters because we do not have crosscoders for Llama 3.1 8B — we use supervised labels instead. If the manifold is robust to the method of discovery, our label-based approach should work.

**Q3: How much of the manifold structure is in the training data vs. the model architecture?**
The rippled manifold is argued to be optimal for packing positions into low dimensions. But is this optimality driven by the training data (character counting requires this resolution) or the architecture (attention heads can only implement certain transformations)? If we trained a model that never saw fixed-width text, would it still develop rippled manifolds for other counting tasks?

**Q4: What fraction of Claude 3.5 Haiku's computations can be described geometrically?**
The paper shows beautiful geometry for the counting and boundary detection steps. But the semantic step (choosing the next word) is described only with features. Is this because the semantic computation does not have geometric structure, or because the team has not found it yet?

**Q5: How do you handle the dilation when doing Fourier analysis?**
The paper notes that Fourier components explain 10% less variance than PCA components, and attributes this to dilation (wider receptive fields at higher counts). Dilation breaks the circulant assumption that underlies the Fourier analysis. Have you tried modifying the Fourier basis to account for dilation (e.g., using a warped frequency axis)? A dilation-corrected Fourier analysis might close the 10% gap.

**Q6: What is the dimensionality of the FULL character count representation?**
The paper reports that 6 PCA components capture 95% of variance. What about the remaining 5%? How many components capture 99%? 99.9%? If 8 components capture 99%, the representation is effectively 8D, not 6D, and the extra dimensions might carry computationally important information.

**Q7: Can you construct visual illusions that target specific character counts?**
The current visual illusions broadly disrupt character counting. Can you design illusions that shift the count by exactly +3 or -5? If so, you could show that the illusion has a predictable, quantitative effect on the manifold position, not just a qualitative disruption.

---

---

# Key Passages Worth Re-reading

The following passages from the paper contain insights that are especially relevant to our project. I record them here (paraphrased, with page references) so we can find them quickly when writing our own paper.

**On the complexity tax (Discussion):** The paper argues that discrete features give a true description of computation but impose a "complexity tax" by fragmenting continuous operations into many small pieces. The manifold description "pays down" this tax by revealing the underlying simplicity. This framing is directly applicable to our paper: we can argue that linear probes (which give 1D descriptions) impose a "dimensionality tax" by flattening manifold structure into scalar values.

**On why multi-dimensional representations are necessary (Section on "The Role of Extra Dimensions"):** The paper gives two reasons: (1) geometric computations like rotation require multi-dimensional representations (you cannot rotate a 1D scalar), and (2) resolution requires curvature (packing 150 distinguishable positions into low dimensions requires the curve to ripple). Both arguments apply to digit representations: digit comparison might require rotation, and distinguishing 10 digit values in a few dimensions requires curvature.

**On distributed computation (Section on "A Distributed Character Counting Algorithm"):** Individual attention heads produce approximately 1D outputs (rays). Curvature requires the sum of multiple heads. This architectural constraint means that manifold representations cannot be attributed to a single component — they are emergent properties of the ensemble. For our pipeline, this means we should not expect to find a single "digit encoding head." Instead, digit manifolds are likely produced by the cooperation of multiple heads across layers.

**On the connection to Fourier features (Appendix on "Analytic Construction"):** The discrete Fourier transform diagonalizes circulant matrices. The low-rank approximation consists of truncating the small Fourier coefficients. The resulting rows exhibit ringing. And crucially, the shift operator on the truncated manifold is a linear map. This is the mathematical heart of the paper and the foundation for our Fourier screening step.

**On automated discovery (Discussion):** The paper calls for "methods that can automatically surface simpler structures to pay down the complexity tax." In our setting, the "simpler structure" is the manifold geometry within our Phase C/D subspaces. Our Fourier screening and GPLVM steps are attempts to automatically discover this structure, guided by the paper's demonstration that such structure exists.

**On naturalistic tasks (Discussion):** The paper argues that "deep mechanistic case studies benefit from choosing behaviors that the model performs consistently well, as these are more likely to have crisper mechanisms." This is a double-edged sword for our project: multiplication at L3 (67.2% correct) might have crisp mechanisms, but L5 (6.11% correct) probably does not. We should expect the cleanest geometry at L1-L3 and progressively messier geometry at L4-L5. The degradation itself is our finding — but we should set expectations appropriately.

---

---

# Final Assessment

## What This Paper Does Well

1. **Exceptional experimental rigor.** Every major claim is supported by multiple lines of evidence (probing, PCA, feature analysis, causal intervention, adversarial testing). The paper does not rely on any single method. When the PCA shows a manifold, they verify it with probes. When the probes show a pattern, they verify it with feature reconstruction. When the feature reconstruction matches, they verify causally with ablation. When the ablation shows importance, they verify with surgical mean-patching. When mean-patching works, they verify the mechanism with adversarial illusions. This layered validation is the gold standard for empirical interpretability.

2. **Beautiful figures.** The cosine similarity heatmaps, PCA projections, attention pattern visualizations, and interactive simulations are among the best in interpretability research. The figures carry the argument — you can understand the main findings from the figures alone. The interactive physical simulation (where you can adjust dimensions, zone width, and topology) is a model for how computational arguments should be presented.

3. **Intellectual honesty.** The paper is explicit about its limitations: one model, one task, no code release, open questions about generalization. The discussion section does not overclaim. The authors explicitly acknowledge what they did not study (line width computation, multi-token words, the full model depth). This honesty builds credibility and makes the positive claims more convincing.

4. **Genuine novelty.** Feature-manifold duality, computation-via-rotation, distributed manifold construction, and visual illusions are all new contributions. The paper does not just confirm what was already known — it discovers new phenomena and explains them mechanistically. The boundary head twist mechanism, in particular, is a landmark finding: it is the first concrete example of a production model performing computation by geometrically manipulating a manifold.

5. **Methodological contribution.** The two-step discovery process (features first, then geometry) is a reusable methodology. The "complexity tax" framing identifies a real problem in the field. The physical simulation of optimal vector packing provides an intuitive explanation for a non-obvious geometric phenomenon. These methodological contributions will outlast the specific findings.

6. **Clear writing.** The paper is unusually well-written for a technical document. Complex ideas are explained with careful progression from intuition to formalism. The use of a running example (the aluminum prompt) throughout the paper provides continuity. Technical terms are defined when introduced. The appendix provides the mathematical detail without cluttering the main text.

7. **Appropriate scope.** The paper studies one task in depth rather than many tasks superficially. This allows the analysis to achieve a level of mechanistic detail that is rare in the field. The depth compensates for the breadth limitation.

## What This Paper Does Poorly

1. **Limited generality.** One model, one task. The paper hints at broader applicability (markdown tables, other models) but does not demonstrate it with the same rigor. The markdown table analysis in the appendix is cursory compared to the main character counting analysis. The Gemma 2 2B and Qwen 3 4B analyses are mentioned but not presented. A stronger version of this paper would include at least one additional model with full analysis.

2. **No released code or data.** For a paper that calls for more geometric analysis of neural networks, not releasing the analysis tools is a missed opportunity. The synthetic dataset generation (wrapping text to fixed widths) is simple enough to reproduce, but the crosscoder dictionary, the probe training pipeline, and the QK analysis tools would accelerate follow-up work. The justification is likely proprietary concerns about Claude 3.5 Haiku, but the analysis tools themselves could be released independently.

3. **No connection to failure modes.** The paper studies a behavior the model performs well. It does not ask: when does the manifold break down? What makes linebreaking fail? What happens to the character count manifold when the model encounters unusual text (very long words, Unicode characters, mixed scripts)? For the character counting task, failure is rare, so studying it would require constructing edge cases. But for our project's focus on correct vs. wrong arithmetic, understanding failure modes is the central question. The paper provides the tools for studying success but not failure.

4. **Shallow treatment of MLPs.** The paper notes that attention heads contribute 4x more than MLPs but does not investigate what the MLPs do. The MLPs' 20% contribution might be crucial for handling edge cases, correcting attention head errors, or adding nonlinear computations that attention cannot perform. Ignoring MLPs is a pragmatic choice (the analysis is already complex), but it leaves a significant part of the computation unexplained.

5. **The optimality argument is informal.** The physical simulation is appealing but not a proof. The Fourier connection is mathematically grounded but the claim that the model "optimizes" for this geometry is not proven. A formal connection between the training objective (next-token prediction) and the resulting geometry (rippled Fourier manifold) would be a significant theoretical contribution, but it is not provided. The closest the paper comes is the observation that the Fourier construction achieves the same similarity matrix as PCA (which is optimal for variance) — but this is optimality of the representation, not of the training process.

6. **No analysis of noise or uncertainty.** The manifold is constructed from centroids — averages across many data points. Individual data points scatter around these centroids. The paper never quantifies this scatter. How much variation is there around the mean manifold? Does the scatter increase at certain count values? Is the scatter different for different prompts or text types? These questions relate to Hauberg's concern about the reliability of deterministic geometry in noisy settings.

7. **Limited coverage of intermediate layers.** The paper analyzes layers 0-2 (construction of character count) and late layers (final decision). What happens in layers 3-15? Do the counting representations maintain their geometry? Do new representations emerge? The omission of middle layers is understandable (the relevant computation happens early and late), but it limits our understanding of how the representation evolves through the model.

## Overall Grade

**A-**. This is one of the best mechanistic interpretability papers published in 2025. It combines rigorous empirical work with genuine conceptual insight. The discovery of computation-via-manifold-manipulation in a production model is a landmark finding that validates an entire research direction. The only significant weaknesses are limited generality (one model, one task, no code) and the gap between the informal optimality argument and a rigorous proof.

For our pipeline, it is essential reading — it validates the geometric approach, provides concrete techniques to borrow, and establishes the empirical target (clean manifolds, Fourier structure, comparison-via-rotation) that we should look for in Llama 3.1 8B.

---

---

# Reading Notes for Future Reference

## Numbers to remember

- 6 PCA dimensions capture 95% of variance in character count representations
- R^2 = 0.985 for 1D linear probe on character count after layer 1
- RMSE = 5 for the linear probe (intrinsic noise in the count representation)
- 10 features tile the character count manifold with overlapping receptive fields
- Boundary head cosine similarity jumps from ~0.25 (identity) to ~1.0 (QK space)
- AUC = 0.91 for 3D geometric classifier on newline prediction
- Fourier components explain ~90% of PCA variance (10% gap due to dilation)
- 5 Layer 0 heads achieve R^2 = 0.93; 11 heads across two layers achieve R^2 = 0.97
- 2D PCA captures 92% of variance in characters-remaining representation
- Attention heads contribute 4x more than MLPs to character count representation
- 180 two-character sequences tested for visual illusion effects

## Equations to remember

- Mean-patching: a_patched = a_original - mu_original + mu_c (replace centroid in subspace)
- Fourier construction: eigenvalues of circulant similarity matrix = DFT of similarity function
- Shift operator: rho_bar = pi_k * rho * pi_k (restricted shift is linear in k-dimensional subspace)
- Physical simulation forces: F_ij = attract(r) when d_ij <= w, F_ij = repel(r) when d_ij > w
- Probe response: P_c^T * W_OV * E_t = interaction between probe c and token embedding t through head OV

## Figures to re-examine

- Figure showing character count manifold in 6D PCA (the rippled helix — our visual target)
- Cosine similarity heatmap of probes showing ringing (the diagnostic we should apply)
- QK transformation of probes (identity vs. boundary head vs. random — our QK analysis template)
- Multiple boundary head response curves (the tiling pattern — our carry head analysis template)
- Orthogonal subspaces for characters-remaining and next-token-length (the decision geometry)
- Physical simulation results at different dimensions (our intuition builder)
- Embedding PCA showing circular structure (the "does the embedding encode geometry?" check)

## Papers to read next (referenced by Gurnee)

- Modell et al. (2025) "The Origins of Representation Manifolds in Large Language Models" — formal theory of feature manifolds, potentially useful for our Fourier screening
- Engels et al. (2025) "Not All Language Model Features Are One-Dimensionally Linear" (ICLR 2025) — circular features in Mistral 7B and Llama 3 8B, closest precedent for our work
- Michaud et al. (2025) "Understanding sparse autoencoder scaling in the presence of feature manifolds" — addresses the pathological behavior concern about SAEs and manifolds
- Olah (2023) "Feature Manifold Toy Model" — the theoretical foundation for manifold thinking in interpretability

---

*Analysis completed: April 4, 2026*
*Reading purpose: Deep understanding — core reference for our geometric interpretability pipeline*
*Estimated reading time: Full 3-pass analysis, approximately 6-8 hours of deep engagement*
*Paper status: Non-peer-reviewed (Transformer Circuits Thread), archived on arXiv (2601.04480)*
*Line count target: 1600-1800*

---

# Summary for Our Pipeline

**Must-do items based on this paper (prioritized):**

1. **Run probe cosine similarity analysis on digit representations (Idea 1).** This is the single most important immediate action. It takes hours, provides an immediate test of Fourier structure, and determines the direction of our entire Fourier screening step. If ringing is present, we proceed with confidence. If absent, we pivot to alternative geometric descriptions.

2. **Check whether the Llama 3.1 8B embedding matrix encodes digit tokens geometrically (Idea 6).** This takes seconds and tells us whether geometric structure exists from the start or is built by computation. Extract the 10 embedding vectors for digit tokens, run PCA, check for circular/rippled structure.

3. **Include Fourier power comparison in our Fourier screening step.** Use the Gurnee paper's ~90% Fourier/PCA ratio as a quantitative benchmark. Report the ratio for each concept variable at each layer. If our ratio is consistently lower (e.g., 50-60%), the Fourier hypothesis is weakened.

4. **Compute Fourier power separately for correct and wrong populations.** This is the central test of our manifold degradation hypothesis. Define the "Fourier degradation ratio" = P1(correct) / P1(wrong) and report it for each concept at each difficulty level. A ratio consistently above 1.0 supports the hypothesis.

5. **Adopt subspace ablation and mean-patching for our causal validation phase.** Follow Gurnee's two-step protocol: (a) ablate the top-k PCA components of a concept subspace, measure accuracy change; (b) mean-patch to specific concept values, verify output changes. Apply to input digits, carries, and answer digits.

6. **Search for boundary-head-like mechanisms in carry computation (Idea 2).** This is more ambitious but potentially transformative. Even a preliminary QK analysis of 5-10 attention heads at layers 12-20 could reveal rotation structure. If found, this would be the mechanistic centerpiece of our paper.

**Must-remember principles:**

1. The LRH finds the room; the manifold is the furniture inside. Our Phase C finds the subspaces (rooms); Fourier screening and GPLVM find the geometry within (furniture). Both levels of analysis are needed.

2. Features and manifolds are dual views. Neither alone is complete. Our concept labels (digit values, carry values) are the supervised analog of Gurnee's crosscoder features. The manifold is what we are looking for within the subspaces these labels define.

3. Rippled manifold embeddings are optimal for packing distinguishable positions into low-dimensional subspaces. Fourier structure is a natural consequence. The model does not choose to use Fourier features — they emerge from the optimization pressure to distinguish values in limited dimensions.

4. Computation on manifolds happens through linear operations (rotations) that exploit the Fourier structure. This is what attention head QK circuits do. If we find Fourier structure, we should look for rotation heads. If we do not find Fourier structure, computation must use a different mechanism.

5. Multiple heads are needed to build curvature. Each head contributes approximately a 1D component. The manifold is an emergent property of the ensemble, not a feature of any single component.

6. Orthogonal arrangement of different variables makes decisions linearly separable. Phase F will test whether our concepts are arranged orthogonally.

7. The Gurnee paper's success with deterministic methods does not invalidate GPLVM. For well-sampled concepts, PCA suffices. For sparse concepts (rare carries, L5/correct), GPLVM adds genuine value.

**Must-cite in our paper:**

This paper should be cited prominently at five points:

- When motivating the geometric approach (cite as existence proof in a production model)
- When introducing Fourier screening (cite the Fourier-ringing connection and ~90% ratio)
- When discussing feature-manifold duality (cite the complexity tax argument)
- When presenting causal validation (cite subspace ablation and mean-patching)
- When discussing why multi-dimensional representations are needed (cite resolution and rotation arguments)
- When comparing correct vs. wrong geometry (cite Gurnee as demonstrating clean geometry for well-performed tasks, and frame our contribution as studying what happens when geometry degrades)

---

---

# What If the Fourier Hypothesis Fails?

Our working hypothesis, motivated by the Gurnee paper, is that digit values in Llama 3.1 8B are represented on Fourier manifolds. But we must plan for the possibility that this is wrong. The Gurnee findings come from a different model, a different task, and a different computational regime. Here are the scenarios and their implications.

## Scenario A: No manifold structure at all

**What this looks like:** Digit value centroids arranged approximately randomly in the subspace. No ringing in cosine similarity. Low Fourier power. No circular structure in PCA.

**What this means:** The model uses a fundamentally different representation strategy. Perhaps one-hot-like encoding (each digit gets its own direction) or a learned embedding without geometric structure.

**How to proceed:** Report as a negative result for Fourier universality. Frame it as: "Fourier structure found in toy models and counting tasks does NOT generalize to arithmetic in pretrained LLMs." This is publishable and redirects the field. Pivot to analyzing whatever structure IS present using Phase C/D tools. Target TMLR.

## Scenario B: Fourier for some concepts, not others

**What this looks like:** Input digits show Fourier structure but carries and answer digits do not. Or the reverse.

**What this means:** The model uses different encoding strategies for input variables (which it reads from the prompt) vs. computed variables (which it derives internally). This is informative about the computational architecture.

**How to proceed:** Report the differential pattern. Interpret in terms of computational flow: Fourier input encoding → non-Fourier intermediate computation → possibly non-Fourier output. The paper becomes about the boundary between geometric and non-geometric computation.

## Scenario C: Weak Fourier structure

**What this looks like:** Some ringing but low amplitude. Fourier power 50-70% of PCA variance (vs. expected ~90%).

**What this means:** Fourier-like representations exist but are degraded or mixed with non-Fourier components. This could reflect insufficient training, superposition with other concepts, or a fundamentally mixed strategy.

**How to proceed:** Report partial Fourier structure. Investigate whether Fourier power varies across layers (potential sharpening, as in Gurnee). Check whether Fourier power correlates with accuracy across levels. Even weak Fourier structure, if it correlates with correctness, supports the manifold degradation hypothesis.

## Scenario D: Strong Fourier structure

**What this looks like:** Clear ringing. Fourier power 80-95% of PCA variance. Circular/helical PCA structure. Different for correct vs. wrong.

**What this means:** Fourier universality confirmed. Our paper makes a strong positive claim.

**How to proceed:** Full pipeline: Fourier → GPLVM → causal patching. Central claim: "Fourier structure is universal and its degradation geometrically characterizes arithmetic failure." Target NeurIPS/ICML.

## The key point

All four scenarios produce a publishable paper. Scenario D is the strongest (positive result at top venue). Scenario A is the most surprising (strong negative result, TMLR). Scenarios B and C are the most nuanced (mixed results, workshop or TMLR). This is why the Gurnee paper analysis matters: it tells us what to look for, but also what to do when we find something different.

---

*Analysis completed: April 4, 2026*
*Reading purpose: Deep understanding — core reference for our geometric interpretability pipeline*
*Estimated reading time: Full 3-pass analysis, approximately 6-8 hours of deep engagement*
*Paper status: Non-peer-reviewed (Transformer Circuits Thread), archived on arXiv (2601.04480)*
*Total paper analyses in project knowledge: 8 (with this one)*