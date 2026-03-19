# Data Generation Stage: Complete Analysis

**Anshul's Geometric Manifold Interpretability Project**
**Carnegie Mellon University, March 2026**

This document records every decision, every number, every piece of math, and every
result from the data generation stage. It is the truth document for this stage.

---

## Table of Contents

1. [Purpose of This Stage](#1-purpose-of-this-stage)
2. [Why Multiplication](#2-why-multiplication)
3. [Why a Difficulty Gradient](#3-why-a-difficulty-gradient)
4. [The Model](#4-the-model)
5. [Tokenization](#5-tokenization)
6. [The Five Difficulty Levels](#6-the-five-difficulty-levels)
7. [The Label System](#7-the-label-system)
8. [Label Verification](#8-label-verification)
9. [The Prompt Format](#9-the-prompt-format)
10. [Activation Extraction](#10-activation-extraction)
11. [Answer Generation](#11-answer-generation)
12. [Accuracy Results](#12-accuracy-results)
13. [Error Classification](#13-error-classification)
14. [Per-Digit Accuracy and the U-Shape](#14-per-digit-accuracy-and-the-u-shape)
15. [Carry Correlation](#15-carry-correlation)
16. [Error Structure Patterns](#16-error-structure-patterns)
17. [Product Magnitude Effects](#17-product-magnitude-effects)
18. [The Code](#18-the-code)
19. [Output Files](#19-output-files)
20. [What This Stage Does NOT Do](#20-what-this-stage-does-not-do)
21. [What This Stage Already Tells Us](#21-what-this-stage-already-tells-us)
22. [Runtime and Reproducibility](#22-runtime-and-reproducibility)
23. [Carry Binning for Phase C/D](#23-carry-binning-for-phase-cd)

---

## 1. Purpose of This Stage

We are studying how Llama 3.1 8B internally represents multiplication. The core
question: when the model gets multiplication wrong, where does the internal math
break down? Does the model fail to represent partial products? Do carry signals
degrade? Does geometric structure (manifolds, subspaces, Fourier periodicity) that
exists for easy problems collapse for hard ones?

To answer these questions we need three things:

1. Multiplication problems with known correct answers and full intermediate labels
2. The model's internal state (activations) while processing each problem
3. The model's actual answers so we can split problems into correct and wrong groups

This stage produces all three. The downstream geometric analysis (UMAP, rSVD, LDA,
Fourier screening, correct/wrong divergence) consumes these artifacts. This stage
does not do any geometric analysis itself.

---

## 2. Why Multiplication

Multiplication is special for interpretability research because we know the complete
ground truth for every intermediate computation. For a problem like 347 x 58, we know
every partial product (7x8=56, 7x5=35, 4x8=32, ...), every column sum, every carry,
every answer digit. No other common task gives this level of ground-truth granularity.

- Language tasks have fuzzy correctness.
- Reasoning tasks have ambiguous intermediate steps.
- Multiplication has exact, verifiable intermediates.

### Prior work this builds on

Bai et al. (2025) studied multiplication in tiny 2-layer models and found
Fourier-basis digit representations and pentagonal prism geometry. Kantamneni and
Tegmark found similar Fourier structure for addition. Gurnee et al. (2025) found
helical manifolds for character counting in Claude 3.5 Haiku. Nobody has
systematically tested whether these structures exist in a large pre-trained model
doing arithmetic. That is the gap this project fills.

---

## 3. Why a Difficulty Gradient

Our original pilot on 3-digit x 3-digit multiplication showed roughly 2% accuracy.
Out of 1,500 problems, about 30 were correct. Thirty correct examples is not enough
for reliable geometric analysis. You cannot estimate centroids, fit subspaces, or
test Fourier structure with 30 data points.

The solution: test multiplication at five difficulty levels.

- Easy levels (1x1 digit) give many correct examples to characterize "what good
  geometry looks like."
- Hard levels (3x3 digit) give many wrong examples to study failure modes.
- Middle levels (2x2 digit) give both.

This gradient lets us track how internal representations change as the model goes
from perfect performance to near-total failure. That transition is where the
interesting science happens.

---

## 4. The Model

### Llama 3.1 8B base

We use `meta-llama/Meta-Llama-3.1-8B`, the base (not instruct, not chat) version.
Downloaded locally to `/data/user_data/anshulk/arithmetic-geometry/model` (30 GB
on disk, confirmed by the SLURM preflight check: `Model found: 30G`).

### Why this specific model

- 32 layers gives enough depth for rich intermediate representations.
- 4096-dimensional activations are large enough to contain interesting subspaces but
  small enough to analyze.
- Open weights means anyone can reproduce our results.
- The model is well-studied in the mechanistic interpretability community.
- It is capable enough at arithmetic to produce a meaningful mix of correct and wrong
  answers across our difficulty levels.

### Why base, not instruct

Instruction tuning (RLHF, safety training) adds confounds. If we found geometric
structure in the instruct model, we would not know whether it came from pre-training
or from the fine-tuning. The base model's representations reflect what the model
learned from pre-training data alone.

### Architecture facts (confirmed against actual model)

| Property | Value |
|----------|-------|
| Transformer layers | 32 (indexed 0 through 31) |
| Hidden dimension | 4096 |
| Attention heads | 32 |
| Inference dtype | bfloat16 (~16 GB VRAM) |
| GPU used | NVIDIA RTX A6000 (49,140 MiB total) |
| Hook path for residual stream | `model.model.layers[l]`, output index `[0]` |
| Load time | 44.7 seconds |
| Vocab size | 128,000 tokens |

The model loads exactly once during the pipeline. The same instance is used for
activation extraction and answer generation.

---

## 5. Tokenization

### How Llama 3.1 8B tokenizes numbers

Llama 3.1 8B uses byte-level BPE (Byte Pair Encoding) tokenization. The tokenizer's
regex pre-tokenization pattern groups up to 3 consecutive digits into a single token.
This means:

- 1-digit numbers (2 through 9): one token
- 2-digit numbers (10 through 99): one token
- 3-digit numbers (100 through 999): one token

Our prompt template is `"{a} * {b} ="`. Every single prompt across all five difficulty
levels tokenizes to exactly 6 tokens. This was verified on 20 samples per level
before any extraction was performed.

From the pipeline log:

```
Level 1: token lengths={6}, last tokens={' ='}
Level 2: token lengths={6}, last tokens={' ='}
Level 3: token lengths={6}, last tokens={' ='}
Level 4: token lengths={6}, last tokens={' ='}
Level 5: token lengths={6}, last tokens={' ='}
Tokenization verification: PASS
```

### Token-level breakdown examples

For a Level 1 prompt `"2 * 2 ="`:

```
Position 0: <BOS>       (token ID 128000, beginning of sequence)
Position 1: "2"          (single-digit number)
Position 2: " *"         (space + asterisk)
Position 3: " "          (space separator)
Position 4: "2"          (single-digit number)
Position 5: " ="         (space + equals sign)
```

For a Level 3 prompt `"66 * 59 ="`:

```
Position 0: <BOS>
Position 1: "66"         (two-digit number, single token)
Position 2: " *"
Position 3: " "
Position 4: "59"         (two-digit number, single token)
Position 5: " ="
```

For a Level 5 prompt `"612 * 383 ="`:

```
Position 0: <BOS>
Position 1: "612"        (three-digit number, single token)
Position 2: " *"
Position 3: " "
Position 4: "383"        (three-digit number, single token)
Position 5: " ="
```

In every case: 6 tokens, last token is `" ="`. The BPE tokenizer absorbs each
operand (1, 2, or 3 digits) into a single token because the regex matches `\d{1,3}`.

### Why tokenization consistency matters

We extract activations at the last token position. Because tokenization is consistent,
"last token" always means the `" ="` token at position 5. If tokenization varied within
a level (some prompts having 5 tokens, others 7), we would need more complex logic to
find the right extraction position.

### Pad token configuration

The Llama 3.1 tokenizer has no default pad token. We set two things in the code:

```python
tokenizer.pad_token = tokenizer.eos_token    # pad_token_id = 128001
tokenizer.padding_side = "left"
```

Left padding ensures that the last token in every batch item is always the content
token (the `" ="` token), not a pad token. This matters because our hook extracts
`hidden[:, -1, :]` (the last sequence position). With left padding, position -1 is
always the `" ="` token regardless of how much padding was added.

### Tokenization verification code

The verification function (`verify_tokenization` in pipeline.py, line 412) takes 20
samples per level, encodes each one, and checks two things:

1. All token lengths within a level form a set of size 1 (meaning they are all equal).
2. The last token decodes to a string containing `"="`.

Both checks passed for all five levels.

---

## 6. The Five Difficulty Levels

### Level 1: 1-digit x 1-digit

| Property | Value |
|----------|-------|
| a range | {2, 3, ..., 9} |
| b range | {2, 3, ..., 9} |
| Product range | 4 to 81 |
| Number of problems | 64 (exhaustive: all 8 x 8 pairs) |
| Unique pairs | 64 (100%) |
| Config flag | `unique_only: true` |

Why 64 and not 4,000: Llama 3.1 8B in eval mode is deterministic. Running `"7 * 8 ="`
multiple times produces bit-identical activations every time. PyTorch's forward pass is
deterministic for the same input on the same GPU (no special flags needed for inference;
Flash Attention's forward pass is always deterministic). Padding to 4,000 with
repetitions wastes compute and creates duplicate rows that would mislead any downstream
variance estimate. Each prompt is run exactly once.

Why exclude 0 and 1: multiplying by 0 always gives 0 (degenerate), multiplying by 1
is identity (degenerate). Neither tests arithmetic computation. Including them would
inflate accuracy without providing information about how the model computes.

### Level 2: 2-digit x 1-digit

| Property | Value |
|----------|-------|
| a range | {10, 11, ..., 99} |
| b range | {2, 3, ..., 9} |
| Product range | 20 to 891 |
| Number of problems | 4,000 (random, seed=42) |
| Unique pairs | 718 out of 4,000 |

This level introduces real carries and multiple partial products. For 34 x 7: the
model must compute 4x7=28, carry the 2, then compute 3x7=21 plus carry 2 = 23.
Two partial products, two columns, one or two carries.

### Level 3: 2-digit x 2-digit

| Property | Value |
|----------|-------|
| a range | {10, 11, ..., 99} |
| b range | {10, 11, ..., 99} |
| Product range | 100 to 9,801 |
| Number of problems | 10,000 (random, seed=42) |
| Unique pairs | 5,733 out of 10,000 (70.8% of full 8,100-pair space) |

Four partial products. Three columns with carries. Column 1 sums two partial products
and receives a carry from column 0, so the running sum can reach 170 (when both
products are 81 and carry in is 8).

**Why 10,000:** With 67.2% accuracy, 10,000 problems yields 6,720 correct answers.
That gives roughly 672 correct per digit value (0 through 9) for any concept label,
well above the 100-per-value floor needed for Phase C subspace identification and LDA
stability. Uniform random sampling suffices because the correct population is large at
this accuracy level.

**Duplicate rate:** 5,733 unique out of 10,000 means 4,267 duplicates. The space is
90 x 90 = 8,100 unique pairs, so 10,000 random samples has a ~43% collision rate by
the birthday problem. This does not break anything — Phase C's centroid method handles
duplicates fine (they just add density to existing clusters, not new clusters) — but
the effective unique coverage is 5,733 / 8,100 = 70.8% of the full space.

### Level 4: 3-digit x 2-digit

| Property | Value |
|----------|-------|
| a range | {100, 101, ..., 999} |
| b range | {10, 11, ..., 99} |
| Product range | 1,000 to 98,901 |
| Number of problems | 10,000 (random, seed=42) |
| Unique pairs | 9,414 out of 10,000 |

Six partial products. Four columns with carries. The carry chain is longer and carry
values are larger.

**Why 10,000:** With 29.0% accuracy, 10,000 problems yields 2,897 correct answers.
That gives roughly 290 correct per digit value, solid for LDA and centroid estimation.
Uniform random sampling is sufficient; the two-phase approach used for L5 is not needed
here.

### Level 5: 3-digit x 3-digit (two-phase approach)

| Property | Value |
|----------|-------|
| a range | {100, 101, ..., 999} |
| b range | {100, 101, ..., 999} |
| Product range | 10,000 to 998,001 |
| Full input space | 900 x 900 = 810,000 possible pairs |
| Selection method | Two-phase: exhaustive screening + carry-balanced selection |
| Selected size | 122,223 problems (all unique) |
| Model accuracy on full space | **6.11%** (49,504 / 810,000) |
| Model accuracy on selected subset | 3.4% (4,197 / 122,223) — lower due to carry oversampling |

Nine partial products. Five columns with carries. The middle columns receive up to
three partial products plus a carry, creating running sums up to 260.

**Why a two-phase approach for L5:** At roughly 6% accuracy, a random sample of
20,000 L5 problems would yield approximately 1,200 correct answers. That sounds
adequate in aggregate, but the distribution across concept values is highly skewed.
Carry values are the binding constraint. Consider carry_0 (the carry out of the
units column): carry_0 = 8 requires a_units = 9 AND b_units = 9 (only 1 of 100
possible digit pairs), so only about 200 of a 20K random sample would even have
carry_0 = 8. At 6% accuracy, that gives roughly 12 correct answers with carry_0 = 8.
Twelve is below the MIN_GROUP_SIZE = 20 threshold needed for Phase C subspace
identification and is far below the 100 needed for reliable LDA. Random sampling
cannot fix this because the skew is intrinsic to the mathematics.

The solution: enumerate all 810,000 possible 3x3 problems, evaluate each one for
correctness, and then deliberately select a carry-balanced subset.

### The L5 two-phase algorithm

**Phase 1: Exhaustive screening.** The script `generate_l5_problems.py` enumerates
all 810,000 (a, b) pairs where a and b range from 100 to 999. For each pair, it
computes all concept labels (carries, partial products, column sums) deterministically
from (a, b) without the model. It then evaluates each pair with greedy decoding to
determine correctness. This took 48.9 minutes of GPU time on an A6000 at
batch_size = 256. Results are cached to `l5_evaluation_cache.npz` (2.4 MB NPZ file
containing 810K entries with a, b, correct flag, and all five carry values).
Re-running the selection with different parameters loads this cache and skips the
expensive evaluation step.

**Phase 1 result:** 49,504 correct out of 810,000 = **6.11% accuracy** on the full
L5 input space. This is the model's true accuracy on uniform 3-digit × 3-digit
multiplication. The 3.4% accuracy on the selected subset is lower because the
selection deliberately oversamples high-carry problems where accuracy is 2-3%.

A critical property: all concept labels are deterministic functions of (a, b).
Carry_0 is determined by a_units and b_units. Carry_1 is determined by a_units,
a_tens, b_units, b_tens, and carry_0. And so on. This means we can compute every
label before ever touching the model, and we know the carry distribution of any
candidate subset without needing to evaluate it.

**Phase 2: Carry-balanced selection.** The selection algorithm stratifies on carry_0
first (the tightest constraint), then carry_1 within each carry_0 group. For each
carry_0 value (0 through 8):

- **Rare groups** (carry_0 values where the total correct count across all 810K is
  at or below CORRECT_CAP = 500): Include ALL problems with that carry_0 value.
  This captures every possible correct answer. For carry_0 = 7 and carry_0 = 8,
  there are approximately 400 correct answers in the entire 810K space. These are
  hard ceilings: no amount of sampling can produce more.

- **Abundant groups** (carry_0 values where correct count exceeds CORRECT_CAP):
  Include CORRECT_CAP = 500 correct problems, selected via stratified sampling on
  carry_1 to maintain carry_1 balance. Then include enough wrong problems to match
  the natural accuracy ratio of that carry_0 group, with a floor of FLOOR_ALL = 500
  total problems. Wrong problems are also stratified-sampled on carry_1.

The target is at least FLOOR_CORRECT = 100 correct samples per carry value, where
the mathematics allows it. For rare groups, a hard ceiling is documented: if only
507 correct answers exist in all 810,000 problems for carry_0 = 7, we cannot conjure
more. The balance report logs every carry value and flags any that fall below 100
correct.

**Actual selection result:** 122,223 problems selected. 4,197 correct (3.4%).
carry_0 values 0-7 each have exactly 500 correct. carry_0 = 8 has 197 correct
(hard ceiling: only 197 correct in all 8,100 carry_0=8 problems in the full space).

### Hard ceilings: what mathematics forbids

The screening identified 25 carry values where the correct population in the entire
810,000-problem space is below 100. These are absolute mathematical limits — no
sampling strategy can produce more correct answers than the model actually gets right:

| Carry variable | Values below ceiling | Worst case |
|---------------|---------------------|------------|
| carry_0 | None — all values have >= 197 correct | carry_0=8: 197 correct in 8,100 problems |
| carry_1 | >= 13 (5 values) | carry_1=17: 5 correct in 81 problems |
| carry_2 | >= 15 (12 values) | carry_2=23-25: 0 correct |
| carry_3 | >= 12 (7 values) | carry_3=16-17: 0 correct |
| carry_4 | = 9 (1 value) | carry_4=9: 16 correct in 5,077 problems |

These ceilings define the scope of the L5 correct-population analysis. For the paper:
"For carry_1 >= 13, carry_2 >= 15, carry_3 >= 12, and carry_4 = 9, fewer than 100
correct examples exist in the entire 810,000-problem input space. These values are
binned with adjacent values for Phase C and LDA (see Section 23)."

### Carry binning for downstream analysis

The binning decisions are documented in Section 23 (Carry Binning for Phase C/D).
The raw labels store exact carry values. Binning is applied only by the analysis code
when grouping problems into concept classes.

### Why hierarchical carry stratification

Concepts in multiplication are deterministically coupled. You cannot balance all 43
concept labels simultaneously because they are functions of each other. For example:

- carry_0 = 8 requires (a_units * b_units) mod 10 to produce carry 8. The only way
  to get carry_0 = 8 is a_units = 9 AND b_units = 9 (since 9 x 9 = 81, carry = 8).
  Oversampling carry_0 = 8 necessarily distorts the a_units and b_units distributions.

- carry_0 = 0 occurs when a_units * b_units < 10. Many digit pairs satisfy this
  (1x1 through 3x3), so oversampling carry_0 = 0 would skew toward small unit digits.

Carry values are chosen as the stratification axis because they are the binding
constraint for the downstream analysis. Phase C uses equal-weight centroids when
estimating concept subspaces, so sample imbalance in other dimensions (like a_units
or b_units) does not affect subspace estimation. What matters is having enough
correct examples per carry value to estimate centroids reliably.

### Why CORRECT_CAP = 500

The CORRECT_CAP = 500 parameter gives a comfortable margin above the floor of 100
correct per concept value:

- Total correct budget: approximately 4,500 (9 carry_0 values x 500 cap)
- This is well above the minimum for LDA stability and Phase C subspace identification
- For rare carry_0 values (7, 8), the hard ceiling of approximately 400 correct in
  all 810K is documented; no cap can exceed the number that exist

### Random sampling and digit coverage

Levels 2 through 4 use `np.random.RandomState(42)` for sampling. Level 5 uses a
deterministic selection from the pre-screened 810K (also seeded with 42 for the
subsampling within abundant groups). The pipeline logs digit coverage to check for
sampling bias. For example, Level 3:

```
Level 3 digit coverage:
  a_leading={6: 452, 7: 438, 1: 416, 8: 485, 9: 458, 4: 423, 3: 447, 5: 422, 2: 459}
  b_leading={5: 444, 9: 459, 8: 464, 1: 428, 6: 456, 7: 417, 2: 454, 3: 445, 4: 433}
```

Leading digits are roughly uniform (each ~440 out of 4,000 in the original run), as
expected from `randint` on a uniform range. The digit coverage heatmaps (saved as
`plots/digit_coverage.png`) visualize this for all five levels.

### Why randint for L2-L4 and not stratified sampling

Using `np.random.randint` with a uniform distribution means some operand pairs are
sampled more than once. Level 2 has only 718 unique pairs out of 4,000 draws (because
the space is 90 x 8 = 720 possible pairs, nearly exhaustive). For L3 and L4 at 10,000
draws, collisions are rarer but still present because the input spaces are larger
(90 x 90 = 8,100 for L3 and 900 x 90 = 81,000 for L4).

A stratified or Latin hypercube design would guarantee uniform coverage, but randint
is simpler and the downstream analysis does not weight by uniqueness. Duplicate
problems produce duplicate activations, which is fine for UMAP and subspace methods
(they just add density to existing clusters, not new clusters).

For L5, random sampling is replaced by the two-phase approach because the correct
population is too sparse and too skewed for random sampling to produce usable
per-carry-value counts.

### Why the two-phase approach is NOT needed for L3 and L4

L3 at 66% accuracy with 10,000 problems gives approximately 6,600 correct. Even the
rarest carry_0 value (carry_0 = 8, requiring a_units = 9 and b_units = 9) occurs in
roughly 1/100 of problems. In 10,000 random draws, roughly 100 problems have
carry_0 = 8, and 66% of those (roughly 66) are correct. This is below 100 but close
enough that LDA is still feasible, and the other carry_0 values each have hundreds
to thousands of correct examples. The cost of screening all 8,100 L3 pairs does not
justify the marginal improvement.

L4 at 28.7% accuracy with 10,000 problems gives approximately 2,870 correct. The
L4 input space is 81,000 pairs, still manageable but not as cheaply enumerable as
L3. The correct count per carry_0 value is sufficient for downstream analysis at
10,000 problems.

---

## 7. The Label System

### Why algorithm-agnostic labels

We do not know what algorithm the model uses internally. It might do something
resembling long multiplication. It might operate in Fourier space. It might do
something completely alien that has no name in human mathematics.

Our labels do not assume any algorithm. They capture mathematical facts about the
product that must be true regardless of how you compute it:

- The partial products of 34 x 57 are 4x7=28, 4x5=20, 3x7=21, 3x5=15. This is
  true whether you call the method "long multiplication," "Urdhva-Tiryagbhyam"
  (the Vedic crosswise method), "lattice multiplication," or anything else.
- The column sums and carries follow from the partial products. They are properties
  of the numbers, not properties of any algorithm.

Our labels test whether the model represents these mathematical quantities. If the
model's activations encode the carry out of column 2, that is a fact about its
representations regardless of how it arrived at that carry.

### Complete walkthrough: 66 x 59 = 3,894

This is the actual first problem of Level 3 in our dataset (prompt: `"66 * 59 ="`).

**Step 1: Digit decomposition**

Break each number into digits by place value. Stored least-significant-first (LSF,
index 0 = units):

```
a = 66  ->  a_digits_lsf = [6, 6]      (units=6, tens=6)
b = 59  ->  b_digits_lsf = [9, 5]      (units=9, tens=5)
```

The code also produces a named decomposition:

```
a_decomposition = {units: 6, tens: 6, num_digits: 2}
b_decomposition = {units: 9, tens: 5, num_digits: 2}
```

LSF ordering is used internally because carry propagation naturally goes from units
to higher places. The code also stores the MSF (left-to-right) version for output
comparison.

**Step 2: All pairwise partial products**

Every pair of one digit from a and one digit from b, multiplied together:

```
a0_x_b0 = 6 x 9 = 54     (a_units x b_units)
a0_x_b1 = 6 x 5 = 30     (a_units x b_tens)
a1_x_b0 = 6 x 9 = 54     (a_tens  x b_units)
a1_x_b1 = 6 x 5 = 30     (a_tens  x b_tens)
```

Four partial products for a 2-digit x 2-digit problem. The naming `a{i}_x_b{j}` uses
LSF indices (i=0 is units). Each partial product is at most 9 x 9 = 81.

**Step 3: Column sums**

Partial products are grouped by output column. Column k receives all products where
the digit positions sum to k (i + j = k):

```
Column 0 (units):    a0_x_b0 = 54                           sum = 54
Column 1 (tens):     a0_x_b1 + a1_x_b0 = 30 + 54 = 84      sum = 84
Column 2 (hundreds): a1_x_b1 = 30                           sum = 30
```

Three columns for a 2x2 problem (n_columns = n_digits_a + n_digits_b - 1 = 2+2-1 = 3).

The code stores which specific products map to each column (`column_products`):

```
column_products = {
    "0": ["a0_x_b0"],
    "1": ["a0_x_b1", "a1_x_b0"],
    "2": ["a1_x_b1"]
}
```

**Step 4: Carry propagation**

Process columns from column 0 (units) upward. At each column, the running sum is the
column sum plus the incoming carry. Output digit = running_sum mod 10. Carry out =
floor(running_sum / 10).

```
Column 0:  running_sum = 54 + 0  = 54   ->  digit = 4,  carry_out = 5
Column 1:  running_sum = 84 + 5  = 89   ->  digit = 9,  carry_out = 8
Column 2:  running_sum = 30 + 8  = 38   ->  digit = 8,  carry_out = 3
(remaining carry: 3 becomes the leading digit)
```

Answer digits LSF: [4, 9, 8, 3] -> MSF: [3, 8, 9, 4] -> 3894. Correct.

Both carries and running sums are stored. Bai et al. showed the running sum is the
key intermediate quantity for the model: its range directly predicts which digits are
hardest.

**Step 5: Per-digit difficulty annotation**

For each answer digit position (in LSF order), the code records:

```
Position 0 (units):     1 partial product,  max_col_sum=81,  carry_chain_length=0
Position 1 (tens):      2 partial products, max_col_sum=162, carry_chain_length=1
Position 2 (hundreds):  1 partial product,  max_col_sum=81,  carry_chain_length=2
Position 3 (thousands): 0 partial products, max_col_sum=0,   carry_chain_length=3
```

Position 3 is the leading digit that comes from the remaining carry (no partial
products contribute directly). Higher positions have longer carry chains. Middle
positions have more products.

### Second walkthrough: 612 x 383 = 234,396

This is the actual first problem of Level 5 (prompt: `"612 * 383 ="`).

**Digit decomposition:**

```
a = 612  ->  a_digits_lsf = [2, 1, 6]     (units=2, tens=1, hundreds=6)
b = 383  ->  b_digits_lsf = [3, 8, 3]     (units=3, tens=8, hundreds=3)
```

**All nine partial products:**

```
a0_x_b0 = 2 x 3 =  6     a0_x_b1 = 2 x 8 = 16     a0_x_b2 = 2 x 3 =  6
a1_x_b0 = 1 x 3 =  3     a1_x_b1 = 1 x 8 =  8     a1_x_b2 = 1 x 3 =  3
a2_x_b0 = 6 x 3 = 18     a2_x_b1 = 6 x 8 = 48     a2_x_b2 = 6 x 3 = 18
```

**Column sums and product mapping:**

```
Col 0 (i+j=0):  a0_x_b0                           =  6
Col 1 (i+j=1):  a0_x_b1 + a1_x_b0                 = 16 + 3  = 19
Col 2 (i+j=2):  a0_x_b2 + a1_x_b1 + a2_x_b0       =  6 + 8 + 18 = 32
Col 3 (i+j=3):  a1_x_b2 + a2_x_b1                 =  3 + 48 = 51
Col 4 (i+j=4):  a2_x_b2                           = 18
```

Five columns for a 3x3 problem (3 + 3 - 1 = 5). Column 2 has three partial products,
the maximum for any column in a 3x3 multiplication.

**Carry propagation:**

```
Col 0:  running_sum =  6 + 0 =  6   ->  digit = 6,  carry = 0
Col 1:  running_sum = 19 + 0 = 19   ->  digit = 9,  carry = 1
Col 2:  running_sum = 32 + 1 = 33   ->  digit = 3,  carry = 3
Col 3:  running_sum = 51 + 3 = 54   ->  digit = 4,  carry = 5
Col 4:  running_sum = 18 + 5 = 23   ->  digit = 3,  carry = 2
(remaining carry: 2 becomes leading digit)
```

Answer digits LSF: [6, 9, 3, 4, 3, 2] -> MSF: [2, 3, 4, 3, 9, 6] -> 234396. Correct.

**Per-digit difficulty for this problem:**

```
Pos 0 (units):          1 product,  max_sum=81,   chain=0
Pos 1 (tens):           2 products, max_sum=162,  chain=1
Pos 2 (hundreds):       3 products, max_sum=243,  chain=2   <-- hardest column
Pos 3 (thousands):      2 products, max_sum=162,  chain=3
Pos 4 (ten-thousands):  1 product,  max_sum=81,   chain=4
Pos 5 (leading):        0 products, max_sum=0,    chain=5
```

Column 2 is the hardest: 3 partial products and a carry chain of length 2. This is
the column where 13.0% digit accuracy was measured at Level 5.

### Simple walkthrough: 2 x 2 = 4

Level 1 (prompt: `"2 * 2 ="`).

```
a_digits_lsf = [2],  b_digits_lsf = [2]
Partial products: a0_x_b0 = 2 x 2 = 4
Column sums: [4]
Running sums: [4]
Carries: [0]
Answer digits MSF: [4]
```

One column, one partial product, zero carries. This is the simplest possible case.

### The label computation code

The `compute_labels` function (pipeline.py, line 176) takes two integers a and b and
returns a dictionary with all the fields shown above. The implementation:

1. Converts a and b to digit lists via `reversed(str(n))` for LSF ordering.
2. Computes all pairwise products with a nested loop over digit indices.
3. Accumulates column sums by the rule `col = i + j`.
4. Propagates carries with a single forward pass through columns.
5. Handles the remaining carry with a while loop (`while carry > 0`).
6. Strips leading zeros (safety check; should not happen for nonzero products).
7. Verifies reconstruction: `sum(d * 10^i for i, d)` must equal `a * b`.

The `decompose_digits` helper (line 166) maps digit positions to human-readable place
names using the `PLACE_NAMES` list: `["units", "tens", "hundreds", "thousands",
"ten_thousands", "hundred_thousands"]`.

---

## 8. Label Verification

Every label set is verified with three independent checks before any model interaction
happens. The verification function is `verify_labels` (pipeline.py, line 308).

### Check 1: Product reconstruction

```python
assert lab["product"] == lab["a"] * lab["b"]
```

The stored product matches a direct multiplication. This catches any corruption in the
label pipeline.

### Check 2: Carry bounds

For each difficulty level, the code computes tight per-column carry bounds using the
`compute_carry_bounds` function (pipeline.py, line 280). The maximum carry at every
column is achieved when all digits are 9. This is provably true: maximizing carry_in
and column_sum never conflict because both are maximized by all-9 inputs.

The bounds are computed dynamically from the digit ranges, not hardcoded:

```python
max_col_sums = [0] * n_cols
for i in range(n_a):
    for j in range(n_b):
        max_col_sums[i + j] += 81    # 9 * 9
# Then propagate: carry = running_sum // 10
```

Every carry in every label set is checked against these bounds:

```python
assert cv <= max_carries[col]
```

### Check 3: Running sum consistency

For each column, the running sum must equal the column sum plus the incoming carry:

```python
expected = lab["column_sums"][col] + carry_in
assert lab["running_sums"][col] == expected
carry_in = expected // 10
```

### Verified carry bounds by level

From the pipeline log:

```
Level 1: max carry bounds = [8]
Level 2: max carry bounds = [8, 8]
Level 3: max carry bounds = [8, 17, 9]
Level 4: max carry bounds = [8, 17, 17, 9]
Level 5: max carry bounds = [8, 17, 26, 18, 9]
```

Understanding these numbers:

- Level 3, column 1: max carry = 17. Column 1 receives 2 products (max 2x81=162) plus
  carry from column 0 (max 8), giving running sum 170, carry out = 17.
- Level 5, column 2: max carry = 26. Column 2 receives 3 products (max 3x81=243) plus
  carry from column 1 (max 17), giving running sum 260, carry out = 26.
- Level 4, column 2: max carry = 17, NOT 26. Level 4 column 2 has only 2 products
  (max 162), plus carry 17 = 179, carry out = 17.

All labels pass all three checks. The exact count depends on the total number of
problems across all levels, which now exceeds the previous 16,064 due to the L3, L4,
and L5 scale-ups.

---

## 9. The Prompt Format

### MSF (most-significant-first)

We write numbers the normal way: `"347 * 582 ="`. This is most-significant-first
(MSF): hundreds digit first, then tens, then units.

Bai et al. used least-significant-first (LSF) for their tiny 2-layer model:
`"743 * 285 ="`. They had a good reason: with LSF output, the model predicts the
easiest digit (units) first, and can build up carry information progressively.

We use MSF because Llama 3.1 8B was pre-trained on internet text that writes numbers
in MSF. Feeding it LSF prompts would test a format it has never seen, which is a
different experiment.

### The MSF output ordering effect

With MSF output, the model must predict the most significant answer digit first. That
digit depends on the overall magnitude and the entire carry chain. But the data shows
a surprise: the model gets the first digit right 99%+ of the time even at Level 5.
The digits it gets wrong are in the middle. This tells us the difficulty is not about
output ordering. It is about the computational structure of each digit position.

We track this with per-digit difficulty labels so downstream analysis can correlate
geometric quality with per-digit difficulty (see Section 14).

### The Vedic math question

We considered whether the labeling scheme assumes "English long multiplication" vs.
"Vedic math" (Urdhva-Tiryagbhyam / crosswise method). The answer: the labels are
algorithm-agnostic. The partial products for 34 x 57 are 4x7, 4x5, 3x7, 3x5
regardless of method. The column sums and carries are mathematical facts about the
numbers. The model may use none of these algorithms internally.

---

## 10. Activation Extraction

### What we extract

At each of 9 target layers, we extract the residual stream activation at the last
token position. The residual stream is the vector that gets passed from one
transformer layer to the next. Each layer reads from it and adds to it. By layer l,
the residual stream contains everything layers 0 through l have contributed.

The last token position is the `" ="` token (position 5). In autoregressive models,
this is the only position that has attended to the entire prompt. The model's state
here is its complete representation of the multiplication problem just before it
starts generating the answer.

Each extracted vector is 4,096 floats (float32). One vector per problem per layer.

### Target layers

```
[4, 6, 8, 12, 16, 20, 24, 28, 31]
```

Nine layers spanning the full depth. The selection rationale:

- Layers 4, 6, 8: early layers where input encoding happens
- Layers 12, 16, 20: middle layers where computation happens
- Layers 24, 28, 31: late layers where output assembly happens

Layer 31 is the final transformer block (indices run 0 through 31; there is no
layer 32). These layers are validated against the model's actual depth at config
load time (pipeline.py, line 47).

### The hook mechanism

We use PyTorch forward hooks registered on `model.model.layers[l]`. When the model
runs a forward pass, each hook captures the layer's output (the residual stream
tensor), takes the last sequence position, detaches it from the computation graph,
converts to float32, and moves it to CPU.

### The closure-over-loop-variable bug (and how we avoid it)

There is a critical Python bug that must be avoided. If you write hooks in a loop:

```python
# WRONG: all hooks capture the same variable 'layer'
for layer in layers:
    def hook_fn(module, input, output):
        captured[layer] = output[0][:, -1, :].detach().cpu()
    model.model.layers[layer].register_forward_hook(hook_fn)
```

Python closures capture the variable `layer` by reference, not by value. When the
hooks fire, `layer` has the value of the last iteration (31). All 9 hooks write to
the same key. You get 9 copies of layer 31's activation and nothing else.

The fix is a factory function (pipeline.py, line 476):

```python
def make_hook(storage, layer_idx):
    def hook_fn(module, input, output):
        hidden = output if isinstance(output, torch.Tensor) else output[0]
        storage[layer_idx] = hidden[:, -1, :].detach().float().cpu()
    return hook_fn
```

Each call to `make_hook` creates a new closure that captures its own `layer_idx`.

The hook also handles a transformers library version difference: in transformers >= 5.x,
`LlamaDecoderLayer` returns a plain tensor; in earlier versions it returns a tuple.
The `isinstance(output, torch.Tensor)` check handles both.

### torch.no_grad()

All extraction runs inside `torch.no_grad()`. Without this, PyTorch builds
computation graphs for backpropagation, which fills GPU memory and causes
out-of-memory errors. We are only doing inference, not training.

### Batch size: 256

The pipeline uses batch_size = 256 for both activation extraction and answer
generation. The A6000 has 48 GB of VRAM. The memory budget:

- Model weights (bfloat16): ~16 GB
- KV cache for batch of 256 x 20 tokens x 32 layers: ~2.7 GB
- Forward pass overhead: ~100 MB
- Total: ~19 GB, leaving ~29 GB headroom

The previous pipeline used batch_size = 32. The 8x increase reduces the number of
batches and eliminates batch-launch overhead as a bottleneck. For L5 with 122,223
problems, this means approximately 478 batches instead of ~3,820 at batch_size=32.

### Checkpoint and resume

Before extracting activations for a given level, the code checks if all 9 layer
files already exist with the correct shape (pipeline.py, line 507). If they do,
that level is skipped entirely. This means if extraction crashes at Level 4 after
completing Levels 1 through 3, restarting skips all completed work.

```python
fpath = act_dir / f"level{lvl}_layer{layer}.npy"
if fpath.exists():
    arr = np.load(fpath, mmap_mode="r")
    if arr.shape != (n, hidden_dim):
        all_exist = False
```

### Post-extraction sanity checks

After extraction, three checks run per level (pipeline.py, line 580):

**Check 1: Distinctness.** The first two problems (which have different prompts)
must produce different activation vectors. If they are identical, the closure bug
is present.

```python
if np.allclose(arr[0], arr[1]):
    logger.error("first two different prompts have identical activations (closure bug?)")
```

**Check 2: Numerical stability.** No NaN or Inf values in any layer.

**Check 3: Norm statistics.** L2 norms are logged for manual inspection.

### Extraction timing (actual)

With batch_size = 256, actual timing from the pipeline run:

- Level 1 (64 problems): 2.8 seconds
- Level 2 (4,000 problems): 4.9 seconds
- Level 3 (10,000 problems): 11.6 seconds
- Level 4 (10,000 problems): 11.4 seconds
- Level 5 (122,223 problems): 142.9 seconds (2.4 minutes)

Total extraction: 173.6 seconds (2.9 minutes) for all levels, dominated by L5.

---

## 11. Answer Generation

### Method

Greedy decoding on all problems across five levels. Settings from config.yaml and the
code (pipeline.py, line 649):

- `do_sample=False` (no randomness, deterministic)
- `max_new_tokens=12` (largest product is 998,001 = 6 digits; 12 tokens gives
  margin for formatting variations)
- `batch_size=256`
- `pad_token_id=tokenizer.eos_token_id`
- Runs under `torch.no_grad()`

For each problem, the model generates text after the prompt. The `parse_number`
function (pipeline.py, line 638) uses a regex to extract the first integer from
the generated text:

```python
m = re.search(r'\d[\d,]*\d|\d', text.strip())
if m:
    return int(m.group().replace(",", ""))
```

This regex handles comma-separated thousands (e.g., `"3,894"` parses as 3894) and
single-digit results. The extracted number is compared to the ground truth product.
Exact match = correct. Everything else = wrong.

### Why greedy

We want the model's single best answer. Sampling would introduce randomness: the
same problem could be correct on one run and wrong on another. Greedy gives the
deterministic "what does the model actually predict" answer.

### What the raw output looks like

The model sometimes generates text beyond just the answer. Examples from the actual
data:

A correct Level 3 answer:
```
Prompt:  "70 * 52 ="
Raw text: " 3640\n70 * 52 = 3640"
Parsed:   3640    (matches ground truth 3640)
```

A wrong Level 3 answer:
```
Prompt:  "66 * 59 ="
Raw text: " 3869\nWhat is 66 * 59?\n"
Parsed:   3869    (ground truth is 3894, off by 25)
```

A Level 2 error (magnitude error):
```
Prompt:  "71 * 9 ="
Raw text: " 63 * 9 = 567\n63 * "
Parsed:   63      (ground truth is 639, model returned a 2-digit number)
```

The model's raw output sometimes includes the original problem restated, follow-up
calculations, or other text. The regex parser ignores all of this and extracts only
the first number.

### L5 screening-pipeline answer match

For Level 5, the pipeline re-generates answers for the selected subset via the same
greedy decoding used in the screening phase. Because greedy decoding is deterministic
(same weights, same prompt, same GPU), the pipeline's `n_correct` must match the
screening's `n_correct`. The post-run validation script checks this explicitly:

```python
if expected == actual:
    print(f'PASS: screening n_correct={expected} matches pipeline n_correct={actual}')
else:
    print(f'MISMATCH: screening n_correct={expected} vs pipeline n_correct={actual}')
```

A mismatch would indicate non-deterministic generation and requires investigation.

### Generation timing (actual)

Answer generation takes approximately 6x longer than activation extraction per level
because it involves autoregressive decoding (generating up to 12 new tokens per
problem) rather than a single forward pass. With batch_size = 256, actual timing
from the pipeline run:

- Level 1 (64 problems): 1.2 seconds
- Level 2 (4,000 problems): 14.3 seconds
- Level 3 (10,000 problems): 35.5 seconds
- Level 4 (10,000 problems): 35.7 seconds
- Level 5 (122,223 problems): 435.9 seconds (7.3 minutes)

Total generation: 8.7 minutes for all levels, dominated by L5.

---

## 12. Accuracy Results

### The accuracy gradient

| Level | Type | Problems | Correct | Wrong | Accuracy |
|-------|------|----------|---------|-------|----------|
| 1     | 1x1  | 64       | 64      | 0     | 100.0%   |
| 2     | 2x1  | 4,000    | 3,993   | 7     | 99.8%    |
| 3     | 2x2  | 10,000   | 6,720   | 3,280 | 67.2%    |
| 4     | 3x2  | 10,000   | 2,897   | 7,103 | 29.0%    |
| 5     | 3x3  | 122,223  | 4,197   | 118,026 | 3.4%*  |
| **Total** | | **146,287** | **17,871** | **128,416** | |

*L5 accuracy on the carry-stratified analysis dataset. The model's true accuracy on
uniform L5 inputs is **6.11%** (49,504 / 810,000), verified by exhaustive screening.
The 3.4% is lower because the dataset deliberately oversamples high-carry problems
where accuracy is 2-3%. When reporting L5 accuracy in the paper, use 6.11%.

The gradient works. Accuracy drops monotonically from 100% to ~6%.

### Level-by-level interpretation

**Level 1 (100.0%):** The model perfectly memorized the multiplication table for
digits 2 through 9. All 64 problems correct. This is the gold standard for correct
geometry.

**Level 2 (99.8%):** Near-perfect. Only 7 errors out of 4,000. The model essentially
memorized 2-digit times 1-digit multiplication. The errors include 5 magnitude errors
and 2 close arithmetic errors — too few wrong answers for meaningful correct/wrong
geometric comparison.

**Level 3 (67.2%):** The sweet spot. 6,720 correct and 3,280 wrong. Both groups are
large enough for full geometric analysis. 6,720 correct examples gives roughly 672
per digit value (0 through 9), well above the 100-per-value floor for Phase C subspace
identification and LDA stability.

**Level 4 (29.0%):** 2,897 correct, 7,103 wrong. Ample correct examples for geometric
analysis (~290 per digit value). The wrong group is large and rich for error pattern
analysis.

**Level 5 (6.11% true / 3.4% dataset):** The failure regime. 4,197 correct examples
in the carry-stratified dataset — a 17x improvement over the previous 239 from random
sampling. carry_0 values 0-7 each have 500 correct. carry_0=8 has 197 (hard ceiling).
The wrong group (118,026) is the richest dataset for failure mode analysis.

### Effective analysis range

Level 2 was easier than expected, making it nearly useless for correct/wrong
comparison. The effective range for geometric analysis is Levels 3 through 5.
Level 1 serves as the perfect-accuracy baseline.

---

## 13. Error Classification

The analysis script (analysis.py, line 125) classifies each wrong answer into one
of four categories based on comparing the predicted number to the ground truth.

### Category definitions

| Category | Rule | Meaning |
|----------|------|---------|
| **garbage** | `predicted is None` | Output was unparseable; no number found |
| **magnitude_error** | `len(str(pred)) != len(str(gt))` | Wrong number of digits |
| **close_arithmetic** | Same digit count AND `abs(pred - gt)/gt < 0.05` | Close but not exact |
| **large_arithmetic** | Same digit count AND `abs(pred - gt)/gt >= 0.05` | Same magnitude, far off |

### Results

```
Level 1:  0 wrong
Level 2:  7 wrong     —  magnitude_error: 5, close_arithmetic: 2
Level 3:  3,280 wrong — close_arithmetic: 3,015, magnitude_error: 257, large_arithmetic: 8
Level 4:  7,103 wrong — close_arithmetic: 6,755, magnitude_error: 348
Level 5:  118,026 wrong — close_arithmetic: 98,001, magnitude_error: 19,998, large_arithmetic: 27
```

### What the categories tell us

**Garbage: zero cases at all levels.** The model always produces a parseable number.
It never outputs nonsense. This means the model has robustly learned the format of
arithmetic answers.

**Magnitude errors** are relatively rare at Levels 3 and 4 (7.8% and 4.9% of errors)
but grow significantly at Level 5 (16.9%). At Level 2, magnitude errors are 71% of
all errors (5 out of 7) because the dominant failure mode is predicting a 2-digit
number when the answer is 3 digits.

**Close arithmetic** dominates at Levels 3 through 5. At Level 4, 95.1% of errors
are close arithmetic (6,755 out of 7,103). The model gets the right number of digits
and comes within 5% of the correct answer but misses specific digits. At Level 5,
83.0% of errors are close arithmetic.

**Large arithmetic** is near-zero (8 cases at Level 3, 0 at Level 4, 27 at Level 5).
The model almost never produces a same-magnitude number that is wildly off.

The picture: when the model fails, it fails narrowly. It gets close to the right
answer but not exactly right. This is structured failure, not random guessing.

---

## 14. Per-Digit Accuracy and the U-Shape

### How per-digit accuracy is computed

For wrong answers where the predicted number has the same digit count as the ground
truth, we compare digit-by-digit in MSF order (position 0 = most significant,
position N-1 = least significant). The code (analysis.py, line 156) iterates over
each character position in the string representation:

```python
for i in range(len(gt_str)):
    total_by_pos[i] += 1
    if gt_str[i] == pred_str[i]:
        correct_by_pos[i] += 1
```

Magnitude-error cases (different digit counts) are excluded from this analysis.

### Per-digit accuracy results

Same-magnitude predictions only (excluding magnitude errors):

```
Level 1: [100.0%, 100.0%]                                     (n=64)
Level 2: [100.0%, 100.0%, 99.9%]                               (n=3,995)
Level 3: [99.2%, 93.6%, 71.7%, 81.7%]                          (n=9,743)
Level 4: [99.3%, 92.6%, 51.8%, 40.1%, 81.7%]                   (n=9,652)
Level 5: [99.2%, 91.8%, 44.3%, 13.0%, 29.3%, 80.2%]            (n=102,225)
```

### The U-shape pattern

High accuracy on the first digit (most significant), a dip in the middle digits, and
a recovery on the last digit (units). This forms a U-shape. It gets more pronounced
as difficulty increases.

At Level 5 (6-digit answers, n=102,225 same-magnitude predictions):

```
Position 0 (hundred-thousands): 99.2%   -- almost always correct
Position 1 (ten-thousands):     91.8%   -- starts to degrade
Position 2 (thousands):         44.3%   -- well above chance but failing
Position 3 (hundreds):          13.0%   -- barely above random for a 10-way choice
Position 4 (tens):              29.3%   -- partial recovery
Position 5 (units):             80.2%   -- strong recovery
```

Position 3 (the fourth digit from the left in a 6-digit number) drops to 13.0%. The
model is barely better than random guessing for this digit. But position 0 (99.2%)
and position 5 (80.2%) remain much better.

### Why the U-shape happens

**First digit (high accuracy):** Depends on the overall magnitude of the product. The
model estimates this well. Getting the leading digit right requires knowing whether
the product is in the 100,000s vs 200,000s, which is a coarse-grained computation.

**Last digit (high accuracy):** The units digit of the product depends only on the
units digits of the operands. Specifically, `product_units = (a_units * b_units) mod 10`.
This is a simple lookup with no carry chain involved. The model handles this well.

**Middle digits (low accuracy):** These digits depend on the most partial products AND
the longest carry chains. For a 6-digit Level 5 product:

- Position 3 in MSF = position 2 in LSF = column 2.
- Column 2 in a 3x3 multiplication receives 3 partial products: `a0_x_b2`,
  `a1_x_b1`, `a2_x_b0`.
- It has a carry chain of length 2 (carries from columns 0 and 1 propagate here).
- The maximum running sum at this column is 260.
- The digit accuracy at this position: 16.2%.

This confirms Bai et al.'s finding in their 2-layer toy model: the middle digits of
the product are the hardest because they sit at the intersection of maximum partial
product count and maximum carry chain length. Our data shows the same pattern in a
32-layer, 8-billion parameter pre-trained model.

### Concrete wrong-answer examples showing the U-shape

From the actual Level 5 data:

```
612 * 383 = 234396    predicted 234354    wrong at positions [4, 5]
638 * 515 = 328570    predicted 328590    wrong at position  [4]
349 * 494 = 172406    predicted 172546    wrong at positions [3, 4]
237 * 897 = 212589    predicted 212329    wrong at positions [3, 4]
```

The errors concentrate in the middle-to-late digit positions. The leading digits are
consistently correct.

From Level 4:

```
781 * 29 = 22649      predicted 22673     wrong at positions [3, 4]
303 * 32 = 9696       predicted 9688      wrong at positions [2, 3]
653 * 77 = 50281      predicted 50321     wrong at positions [2, 3]
```

Same pattern: leading digits correct, middle digits wrong.

---

## 15. Carry Correlation

### Carry count vs accuracy

The analysis script counts how many non-zero carries each problem has and groups
accuracy by that count (analysis.py, line 205).

The accuracy-vs-carry-count plot confirms the monotonic relationship at every level.
At L5, with 122,223 problems, the carry correlation is measured with far more
statistical power than the previous 4K run. The L5 curve now uses thousands of
problems per carry-count bin (versus tens before), making the monotonic decline clean
and unambiguous.

The key finding holds with high confidence: carries are the dominant factor in
difficulty. It is not just "big numbers are hard." It is "numbers whose products
require many carries are hard."

### Carry sum binning

The analysis also bins problems by total carry sum (sum of all carry values, not just
count of non-zero carries) and by maximum single carry value. These provide finer
resolution for the downstream geometric analysis: a problem with carries [1, 1, 1]
is easier than a problem with carries [8, 17, 9], even though both have 3 non-zero
carries.

---

## 16. Error Structure Patterns

The analysis script (analysis.py, line 258) computes several structural properties
of the errors (predicted minus ground truth).

### Even error bias

```
Level 3: even_frac = 93.8%
Level 4: even_frac = 95.0%
Level 5: even_frac = 89.5%
```

90-95% of the error values (predicted - ground_truth) are even numbers. This means
in the vast majority of cases, the model's prediction has the same parity (odd or
even) as the ground truth. The model rarely flips parity. When it makes a mistake,
it tends to be off by an even amount (2, 4, 6, 8, 10, ...).

### Divisibility by 10

```
Level 3: div10_frac = 49.5%
Level 4: div10_frac = 70.0%
Level 5: div10_frac = 67.6%
```

If the error is divisible by 10, it means the model got the units digit exactly right
but made mistakes in higher digits. At Level 4, 70% of wrong answers have the correct
units digit. At Level 5, 67.6%. This is consistent with the per-digit accuracy data
showing high accuracy on the last digit (80-82%).

The div10 fraction increases with difficulty from L3 to L4. Harder problems have more
digits, and the model increasingly gets only the easiest (units) and hardest (middle)
digits right/wrong respectively.

### 10's complement pattern

When the units digit is wrong, what kind of wrong is it?

```
Level 3: units_complement_frac = 47.6%  (of 1,656 units mismatches)
Level 4: units_complement_frac = 39.8%  (of 2,128 units mismatches)
Level 5: units_complement_frac = 34.2%  (of 38,260 units mismatches)
```

When the model gets the units digit wrong, 34-48% of the time the wrong units digit
and the correct units digit sum to 10 (or 0). For example: predicting units digit 8
when the answer is 2 (8 + 2 = 10), or predicting 4 when the answer is 6 (4 + 6 = 10).

This suggests the model sometimes computes the correct intermediate value for the
units column but applies the modular reduction incorrectly. If the units column sum
is 56, the correct digit is 6 (56 mod 10 = 6). A 10's complement error would give
4 (10 - 6 = 4). This pattern is consistent with partial carry propagation errors.

### Underestimation bias

```
Level 3: underestimate_frac = 61.7%
Level 4: underestimate_frac = 56.4%
Level 5: underestimate_frac = 65.8%
```

56-66% of wrong answers underestimate the true product (predicted < ground_truth).
The model slightly prefers smaller numbers. The median error is negative at all levels
(L3: -10, L4: -20, L5: -240).

### Median relative error

```
Level 3: median_rel_error = 0.462%
Level 4: median_rel_error = 0.258%
Level 5: median_rel_error = 0.244%
```

These are remarkably small. At Level 4, the median wrong answer is off by 0.26%.
At Level 5, 0.24%. These are not random guesses; they are precise computations with
small errors in specific digit positions. The model is doing real arithmetic but
making mistakes in the hardest sub-computations.

The median absolute errors tell the same story in raw numbers:

```
Level 3: median_abs_error = 16
Level 4: median_abs_error = 60
Level 5: median_abs_error = 530
```

A median absolute error of 60 on products averaging around 23,000 (Level 4) means
the typical wrong answer is off by about one digit in one position.

---

## 17. Product Magnitude Effects

### Accuracy by product digit count

Within each level, larger products are harder. This is expected: larger products have
more digits, which means more columns and longer carry chains. Magnitude is a proxy
for carry complexity.

The digit-count split also explains some of the per-digit accuracy structure: within
a level, the mix of different-length products means the per-digit analysis aggregates
over products of different lengths. The analysis script handles this by only comparing
same-magnitude predictions.

---

## 18. The Code

### Architecture

Four Python scripts, two shell scripts, and supporting files:

**generate_l5_problems.py** (~565 lines) handles the L5 two-phase dataset generation.
It enumerates all 810,000 3x3 problems, evaluates correctness with the model, and
selects a carry-balanced subset. It imports utility functions (`compute_labels`,
`load_config`, `load_model`, `load_tokenizer`, `parse_number`) from pipeline.py.
It runs independently before the main pipeline, producing a JSON file that pipeline.py
reads.

**pipeline.py** (~934 lines) handles everything that needs the GPU: problem generation,
label computation, verification, tokenization, model loading, activation extraction,
answer generation, and diagnostic plots. For L5, it reads the pre-selected problem set
from the JSON file produced by `generate_l5_problems.py` instead of generating random
problems. The `generate_problems()` function has a branch:

```python
if lc.get("problems_file"):
    # Read pre-selected problems (e.g., L5 two-phase balanced selection)
    pf = Path(lc["problems_file"])
    with open(pf) as f:
        selected = json.load(f)
    a_vals = selected["a"]
    b_vals = selected["b"]
```

**analysis.py** (~777 lines) handles error analysis: it loads the saved answers and
labels (no GPU needed), classifies errors, computes per-digit accuracy, carry
correlation, error structure, and generates 6 analysis plots plus a JSON summary.
It runs in seconds on CPU.

**config.yaml** holds all parameters. Key additions for the new pipeline:

- Per-level `problems_per_level` overrides (L3 and L4 set to 10,000)
- `problems_file` key for L5 pointing to the screening output
- `batch_size: 256` (up from 32)

```yaml
dataset:
  problems_per_level: 4000   # default (L2 uses this)
  levels:
    3:
      problems_per_level: 10000
    4:
      problems_per_level: 10000
    5:
      problems_file: /data/.../l5_screening/l5_selected_problems.json
generation:
  batch_size: 256
```

**run_l5_screen.sh** is a SLURM script that runs `generate_l5_problems.py` with
pre-flight checks and post-screening validation. It requests 3 hours wall time
(actual runtime: 50 minutes).

**run.sh** is a SLURM script that runs `pipeline.py` and `analysis.py` in sequence
with pre-flight checks and post-run validation. It verifies that the L5 screening
output exists before starting (prerequisite check). It requests 6 hours wall time
(actual runtime: 14.3 minutes).

### Execution order

The overall execution has two stages:

**Stage 1: L5 Screening** (run_l5_screen.sh)

```
 Step  | What                                  | GPU? | Estimated Time
-------|---------------------------------------|------|---------------
   1   | Enumerate all 810,000 L5 problems     | No   | ~5s
   2   | Compute carry labels for all 810K     | No   | (part of step 1)
   3   | Evaluate correctness (810K problems)  | YES  | 60-80 min
   4   | Cache evaluation results to .npz      | No   | ~5s
   5   | Run carry-balanced selection           | No   | ~1s
   6   | Print balance report                  | No   | instant
   7   | Identify hard ceilings                | No   | instant
   8   | Save selected problems to JSON        | No   | instant
```

**Stage 2: Main Pipeline** (run.sh)

```
 Step  | What                                          | GPU? | Estimated Time
-------|-----------------------------------------------|------|---------------
   1   | Load and validate config                      | No   | instant
   2   | Initialize logging                            | No   | instant
   3   | Generate problems (L1-L4 random; L5 from file)| No   | < 1s
   4   | Compute labels (~60K+ problems)               | No   | ~2s
   5   | Verify all labels                             | No   | ~1s
   6   | Format prompts                                | No   | instant
   7   | Save labeled datasets as JSON                 | No   | ~5s
   8   | Load tokenizer, verify                        | No   | ~6s
   9   | Load model (bfloat16)                         | YES  | ~45s
  10   | Extract activations (all 5 levels)            | YES  | ~65s
  11   | Post-extraction sanity checks                 | No   | ~5s
  12   | Generate answers (greedy)                     | YES  | ~6-7 min
  13   | Save answers with correctness                 | No   | < 1s
  14   | Generate 3 diagnostic plots                   | No   | ~3s
  15   | Log summary statistics                        | No   | instant
  16   | Run analysis.py                               | No   | ~5s
```

Steps 1 through 8 require no GPU. Step 9 commits GPU memory. This ordering means
that if label generation or tokenization fails, no time is wasted loading the 16 GB
model.

### Execution order (analysis.py)

```
 Step  | What                              | Time
-------|-----------------------------------|-------
   1   | Load config                       | instant
   2   | Load saved answers (5 levels)     | < 1s
   3   | Load saved labels (5 levels)      | < 1s
   4   | Merge into enriched dataset       | < 1s
   5   | Classify errors                   | < 1s
   6   | Compute per-digit accuracy        | < 1s
   7   | Compute carry correlation         | < 1s
   8   | Compute error structure           | < 1s
   9   | Compute input difficulty          | < 1s
  10   | Generate 6 plots                  | ~0.5s
  11   | Save JSON summary                 | instant
  12   | Print report                      | instant
```

### Key design decisions in the code

**Deferred imports.** Torch and transformers are imported only inside the functions
that need them (pipeline.py, lines 455, 495). This means steps 1 through 7 can run
on a CPU-only machine without any GPU libraries installed. Useful for debugging label
computation without submitting a GPU job. The same pattern is used in
`generate_l5_problems.py`, where torch is imported inside `evaluate_correctness()`.

**Matplotlib Agg backend.** Both scripts call `matplotlib.use("Agg")` before
importing pyplot. This avoids requiring an X11 display, which is not available on
compute nodes.

**Rotating file handler.** Logs use `RotatingFileHandler` with a 10 MB max size and
3 backups (pipeline.py) or 50 MB max size and 2 backups (generate_l5_problems.py).
This prevents log files from growing unboundedly across multiple runs.

**Config-driven design.** All parameters live in config.yaml: operand ranges, layer
indices, batch size, paths. Changing from Level 5 = 3x3 to Level 5 = 4x4 requires
editing only the config file, not the code.

**L5 evaluation cache.** The 810K model evaluation is the most expensive single
operation in the project (60-80 minutes of GPU time). The cache file
(`l5_evaluation_cache.npz`) stores all 810K evaluations so that changing selection
parameters (e.g., CORRECT_CAP, FLOOR_ALL) does not require re-evaluating the model.
The `--reselect` flag on `generate_l5_problems.py` triggers this cache-only mode.

**L5 re-import in pipeline.py.** pipeline.py imports the pre-selected L5 problems
from the JSON file via the `problems_file` config key. This keeps the two scripts
loosely coupled: `generate_l5_problems.py` produces a JSON contract (with `a`, `b`,
`n_selected`, `n_correct`, and `metadata` fields), and pipeline.py consumes it.
Changing the selection algorithm requires only re-running the screening script.

---

## 19. Output Files

### On the workspace (lightweight, in git)

```
/home/anshulk/arithmetic-geometry/
    pipeline.py                          # data generation + extraction
    generate_l5_problems.py              # L5 two-phase screening + selection
    analysis.py                          # error analysis (CPU only)
    config.yaml                          # all parameters
    run.sh                               # SLURM job script (main pipeline)
    run_l5_screen.sh                     # SLURM job script (L5 screening)
    docs/
        datageneration_analysis.md       # this document
    labels/
        level_1.json                     # 64 problems (39 KB)
        level_2.json                     # 4,000 problems (2.9 MB)
        level_3.json                     # 10,000 problems (9.1 MB)
        level_4.json                     # 10,000 problems (11.0 MB)
        level_5.json                     # 122,223 problems (160.0 MB)
        analysis_summary.json            # accuracy and error stats (6.4 KB)
    plots/
        accuracy_by_level.png            # bar chart of difficulty gradient
        activation_norm_profile.png      # norm growth across layers
        digit_coverage.png               # sampling coverage heatmaps
        per_digit_accuracy_heatmap.png   # the U-shape visualization
        error_distributions.png          # error histograms
        accuracy_vs_carries.png          # carry correlation lines
        accuracy_vs_magnitude.png        # accuracy by product digits
        error_categories.png             # stacked bar by category
        digit_accuracy_by_carry.png      # carry vs no-carry split
    logs/
        pipeline.log                     # timestamped execution log
        generate_l5_problems.log         # L5 screening log
        slurm-*.out / slurm-*.err        # SLURM stdout/stderr
```

### On the data volume (heavy, not in git)

```
/data/user_data/anshulk/arithmetic-geometry/
    model/                               # Llama 3.1 8B weights (30 GB)
    l5_screening/                        # L5 two-phase outputs
        l5_evaluation_cache.npz          # 810K evaluations (2.4 MB)
        l5_selected_problems.json        # selected 122,223 problems (1.2 MB)
    activations/                         # 45 .npy files (20.09 GB total)
        level1_layer4.npy                # shape (64, 4096), float32
        level1_layer6.npy
        ...
        level3_layer31.npy               # shape (10000, 4096), float32
        level4_layer31.npy               # shape (10000, 4096), float32
        level5_layer31.npy               # shape (122223, 4096), float32
    answers/                             # 20.7 MB total
        level_1.json through level_5.json
        analysis_summary.json
```

### Activation file sizes

Each `.npy` file contains a 2D numpy array of shape `(n_problems, 4096)` in float32.
The file naming convention is `level{N}_layer{L}.npy`.

- Level 1: 9 files, shape (64, 4096), ~1.1 MB each, ~10 MB total
- Level 2: 9 files, shape (4000, 4096), ~63 MB each, ~567 MB total
- Level 3: 9 files, shape (10000, 4096), ~157 MB each, ~1.41 GB total
- Level 4: 9 files, shape (10000, 4096), ~157 MB each, ~1.41 GB total
- Level 5: 9 files, shape (122223, 4096), ~1.9 GB each, ~17.1 GB total
- **Total: 45 files, 20.09 GB**

Row index matches the problem index in the label and answer JSON files. So
`level3_layer16.npy[0]` is the layer-16 residual stream for the first Level 3
problem (`66 * 59 =`).

### Label JSON structure

Each level's JSON file has the following schema:

```json
{
    "level": 3,
    "n_problems": 10000,
    "unique_problems": 8100,
    "level_config": {"a_range": [10, 99], "b_range": [10, 99], "problems_per_level": 10000},
    "problems": [
        {
            "index": 0,
            "prompt": "66 * 59 =",
            "labels": {
                "a": 66,
                "b": 59,
                "product": 3894,
                "a_digits_lsf": [6, 6],
                "b_digits_lsf": [9, 5],
                "a_decomposition": {"units": 6, "tens": 6, "num_digits": 2},
                "b_decomposition": {"units": 9, "tens": 5, "num_digits": 2},
                "partial_products": {
                    "a0_x_b0": 54, "a0_x_b1": 30,
                    "a1_x_b0": 54, "a1_x_b1": 30
                },
                "column_sums": [54, 84, 30],
                "column_products": {
                    "0": ["a0_x_b0"],
                    "1": ["a0_x_b1", "a1_x_b0"],
                    "2": ["a1_x_b1"]
                },
                "carries": [5, 8, 3],
                "running_sums": [54, 89, 38],
                "answer_digits_lsf": [4, 9, 8, 3],
                "answer_digits_msf": [3, 8, 9, 4],
                "digit_difficulty": [
                    {"position_lsf": 0, "num_partial_products": 1,
                     "max_column_sum": 81, "carry_chain_length": 0},
                    {"position_lsf": 1, "num_partial_products": 2,
                     "max_column_sum": 162, "carry_chain_length": 1},
                    {"position_lsf": 2, "num_partial_products": 1,
                     "max_column_sum": 81, "carry_chain_length": 2},
                    {"position_lsf": 3, "num_partial_products": 0,
                     "max_column_sum": 0, "carry_chain_length": 3}
                ]
            }
        },
        ...
    ]
}
```

### Answer JSON structure

```json
{
    "level": 3,
    "n_problems": 10000,
    "n_correct": 6600,
    "accuracy": 0.66,
    "results": [
        {
            "index": 0,
            "a": 66,
            "b": 59,
            "ground_truth": 3894,
            "predicted": 3869,
            "correct": false,
            "raw_text": " 3869\nWhat is 66 * 59?\n"
        },
        ...
    ]
}
```

### L5 selected problems JSON structure

The output of `generate_l5_problems.py`:

```json
{
    "metadata": {
        "total_screened": 810000,
        "total_correct_in_space": 49504,
        "accuracy": 0.06111604938271605,
        "selection_parameters": {
            "correct_cap": 500,
            "floor_all": 500,
            "floor_correct": 100
        },
        "balance_report": {
            "carry_0": {
                "0": {"total": 5152, "correct": 500},
                "1": {"total": 13915, "correct": 500},
                "...": "...",
                "8": {"total": 8100, "correct": 197}
            },
            "carry_1": { "...": "..." },
            "carry_2": { "...": "..." },
            "carry_3": { "...": "..." },
            "carry_4": { "...": "..." },
            "a_units": { "...": "..." },
            "...": "..."
        },
        "hard_ceilings": {
            "carry_1_13": {"total_in_space": 2835, "correct_in_space": 51},
            "carry_1_17": {"total_in_space": 81, "correct_in_space": 5},
            "carry_2_23": {"total_in_space": 56, "correct_in_space": 0},
            "...": "... (25 entries total)"
        }
    },
    "a": [100, 101, ...],
    "b": [105, 103, ...],
    "n_selected": 122223,
    "n_correct": 4197
}
```

---

## 20. What This Stage Does NOT Do

This is a data generation and characterization pipeline. It does not do geometric
analysis. Specifically:

- No UMAP or t-SNE dimensionality reduction
- No Spearman correlations between labels and activations
- No conditional covariance or rSVD for subspace identification
- No LDA (Linear Discriminant Analysis) on carry values
- No Fourier screening for periodic structure
- No correct/wrong geometric comparison (the core paper analysis)
- No Gaussian process fitting or probabilistic geometry
- No cross-layer tracing of representations
- No causal interventions or ablation studies
- No chain-of-thought prompting experiments
- No instruction-tuned model comparison
- No alternative model testing (Mistral, Pythia, etc.)
- No alternative arithmetic tasks (addition, modular arithmetic)
- No LSF prompt format experiments

All of the above consume the artifacts this stage produced. This stage gives them
the data. The science starts with the next script.

---

## 21. What This Stage Already Tells Us

Even before geometric analysis, the error patterns reveal important facts about how
Llama 3.1 8B does multiplication.

### The model is not guessing

Wrong answers are close to correct (median 0.24-0.46% relative error). The median
absolute error at Level 4 is 60 on products averaging around 23,000. This is a
fraction of a percent. The model has learned something real about multiplication; it
just cannot execute it precisely at scale.

### The model fails in specific places

Middle digits are hardest. Carries are the bottleneck. The U-shaped per-digit
accuracy profile matches what Bai et al. found in toy 2-layer models. This is the
first confirmation that the carry-chain bottleneck scales from 2-layer models to
production-scale models.

### The model has systematic biases

- **Even errors (90-95%):** The model almost always preserves the parity of the
  ground truth. Errors tend to be even-valued.
- **Correct units digit (50-70%):** The model frequently gets the easiest digit
  right even when the overall answer is wrong, and this fraction increases with
  difficulty.
- **10's complement errors (34-48%):** When the units digit is wrong, it is often
  the 10's complement of the correct digit, suggesting a carry propagation reversal.
- **Slight underestimation (56-66%):** The model prefers smaller numbers.
- **Zero garbage output (0%):** The model always produces a parseable number.

These are not random noise. They suggest the model has learned real mathematical
structure but breaks down at specific computational bottlenecks.

### The labels enable the next step

Because we know exactly which sub-computation failed for each wrong answer (which
digit, which carry, which column), the downstream geometric analysis can ask targeted
questions: "At which layer does the carry signal for column 2 diverge between correct
and wrong answers?" That specificity is what makes this project different from prior
work that could only say "something diverges" without identifying what.

### The L5 two-phase approach enables carry-specific analysis

The carry-balanced L5 dataset is specifically designed to give Phase C and LDA
enough correct samples per carry value. Without it, a random 20K sample would yield
roughly 12 correct samples with carry_0 = 8, making subspace estimation impossible.
With the two-phase approach, the L5 correct population has 4,197 samples — a 17x
improvement over the 239 from the previous random sample. carry_0 values 0-7 each
have exactly 500 correct. carry_0 = 8 has 197 (the hard mathematical ceiling).
Rare carry values that cannot reach 100 correct individually are binned with adjacent
values (see Section 23).

---

## 22. Runtime and Reproducibility

### Execution environment

| Property | Value |
|----------|-------|
| GPU | NVIDIA RTX A6000, 49,140 MiB VRAM |
| CPU cores | 8 |
| RAM | 64 GB |
| Python | 3.11, conda environment "geometry" |
| PyTorch | 2.x (with bfloat16 support) |
| Transformers | 5.x (LlamaDecoderLayer returns tensor) |

### Actual timing

**Stage 1: L5 Screening** (`run_l5_screen.sh`, SLURM job 6645003)

```
Enumerate + label 810K problems:       1.6 seconds
Load model:                            42.8 seconds
Evaluate 810K problems (GPU):          48.9 minutes
Cache results:                         2.2 seconds
Run selection algorithm:               < 1 second
Total:                                 50 minutes (3,027 seconds wall time)
```

**Stage 2: Main Pipeline** (`run.sh`)

```
Config + problems + labels + save:     19 seconds (L5 label computation: 4.3s)
Load tokenizer + verify:               11 seconds
Load model:                            47.9 seconds
Extract activations (all levels):      173.6 seconds (2.9 minutes)
Post-extraction checks:                32 seconds
Generate answers (all levels):         522.6 seconds (8.7 minutes)
Save answers + plots:                  2.4 seconds
Total:                                 14.3 minutes
```

**Combined total: 64.3 minutes**, dominated by the L5 screening phase. The main
pipeline itself takes 14.3 minutes.

If the L5 evaluation cache already exists (from a previous screening run), Stage 1
completes in under 10 seconds via the `--reselect` flag.

### Reproducibility

Random seed: 42. All problem generation uses this seed via `np.random.RandomState(42)`.
The same seed produces the same problems and labels. The L5 selection algorithm also
uses seed 42 for stratified subsampling within abundant carry_0 groups.

The model was downloaded locally. The config points to the local path
(`/data/user_data/anshulk/arithmetic-geometry/model`), not the HuggingFace hub.
The pipeline runs without internet access and is not affected by model version
changes on HuggingFace.

Greedy decoding (`do_sample=False`) is deterministic. Given the same model weights,
the same prompts, and the same GPU, the activations and answers will be identical.
This is verified for L5 by checking that the pipeline's `n_correct` matches the
screening's `n_correct` (see Section 11).

### GPU memory budget

| Component | Memory |
|-----------|--------|
| Model weights (bfloat16) | ~16 GB |
| KV cache (batch=256, 20 tokens, 32 layers) | ~2.7 GB |
| Forward pass overhead | ~100 MB |
| **Total during inference** | **~19 GB** |
| **Headroom on A6000 (48 GB)** | **~29 GB** |

The 29 GB headroom is more than sufficient. No memory pressure is expected even with
the larger L5 dataset, since batch size (not dataset size) determines peak memory.

---

## Appendix A: Partial Product Counts by Level

| Level | Type | Columns | Products per column (LSF order) | Total products |
|-------|------|---------|-------------------------------|----------------|
| 1     | 1x1  | 1       | [1]                           | 1              |
| 2     | 2x1  | 2       | [1, 1]                        | 2              |
| 3     | 2x2  | 3       | [1, 2, 1]                     | 4              |
| 4     | 3x2  | 4       | [1, 2, 2, 1]                  | 6              |
| 5     | 3x3  | 5       | [1, 2, 3, 2, 1]               | 9              |

The product-per-column pattern is always symmetric and peaks at the middle. For 3x3
it is [1, 2, 3, 2, 1] with the peak at column 2 (3 products). This is the column
where accuracy drops to 16.2%.

## Appendix B: Maximum Running Sums by Level

| Level | Column 0 | Column 1 | Column 2 | Column 3 | Column 4 |
|-------|----------|----------|----------|----------|----------|
| 1     | 81       | -        | -        | -        | -        |
| 2     | 81       | 89       | -        | -        | -        |
| 3     | 81       | 170      | 98       | -        | -        |
| 4     | 81       | 170      | 179      | 98       | -        |
| 5     | 81       | 170      | 260      | 179      | 98       |

Column 0 is always 81 (max single product). The middle columns of harder levels
have both more products and larger incoming carries.

## Appendix C: Complete Error Category Breakdown

| Level | Total wrong | close_arithmetic | magnitude_error | large_arithmetic | garbage |
|-------|-------------|-----------------|-----------------|------------------|---------|
| 1     | 0           | 0               | 0               | 0                | 0       |
| 2     | 7           | 2 (29%)         | 5 (71%)         | 0                | 0       |
| 3     | 3,280       | 3,015 (91.9%)   | 257 (7.8%)      | 8 (0.2%)         | 0       |
| 4     | 7,103       | 6,755 (95.1%)   | 348 (4.9%)      | 0                | 0       |
| 5     | 118,026     | 98,001 (83.0%)  | 19,998 (16.9%)  | 27 (0.02%)       | 0       |

## Appendix D: Complete Norm Statistics (All 9 Layers, All 5 Levels)

From pipeline.log debug output:

**Level 1 (64 problems):**

| Layer | Min  | Mean | Max  |
|-------|------|------|------|
| 4     | 3.7  | 3.8  | 3.8  |
| 6     | 5.2  | 5.3  | 5.5  |
| 8     | 6.2  | 6.5  | 6.6  |
| 12    | 7.8  | 8.0  | 8.2  |
| 16    | 10.2 | 10.6 | 11.3 |
| 20    | 14.9 | 16.0 | 16.9 |
| 24    | 22.0 | 23.9 | 25.8 |
| 28    | 33.1 | 35.4 | 37.8 |
| 31    | 68.5 | 74.8 | 78.0 |

**Level 2 (4,000 problems):**

| Layer | Min  | Mean | Max  |
|-------|------|------|------|
| 4     | 3.6  | 3.7  | 3.8  |
| 6     | 5.0  | 5.1  | 5.3  |
| 8     | 5.9  | 6.3  | 6.5  |
| 12    | 7.3  | 8.1  | 8.4  |
| 16    | 10.2 | 11.1 | 11.7 |
| 20    | 15.6 | 17.1 | 18.2 |
| 24    | 22.4 | 25.6 | 27.2 |
| 28    | 33.8 | 37.4 | 39.4 |
| 31    | 69.7 | 76.1 | 81.6 |

**Level 3 (10,000 problems):**

| Layer | Min  | Mean | Max  |
|-------|------|------|------|
| 4     | 3.6  | 3.7  | 3.8  |
| 6     | 4.9  | 5.1  | 5.3  |
| 8     | 5.9  | 6.3  | 6.7  |
| 12    | 7.7  | 8.4  | 8.8  |
| 16    | 9.8  | 11.0 | 12.0 |
| 20    | 14.1 | 16.6 | 18.0 |
| 24    | 20.4 | 24.9 | 27.7 |
| 28    | 32.1 | 36.7 | 40.1 |
| 31    | 71.2 | 77.5 | 85.1 |

**Level 4 (10,000 problems):**

| Layer | Min  | Mean | Max  |
|-------|------|------|------|
| 4     | 3.5  | 3.6  | 3.8  |
| 6     | 4.8  | 5.0  | 5.4  |
| 8     | 5.9  | 6.5  | 7.3  |
| 12    | 7.8  | 8.4  | 9.1  |
| 16    | 9.0  | 10.7 | 12.0 |
| 20    | 13.0 | 16.0 | 17.9 |
| 24    | 19.2 | 23.9 | 27.4 |
| 28    | 30.8 | 35.9 | 39.7 |
| 31    | 66.3 | 76.6 | 87.2 |

**Level 5 (122,223 problems):**

| Layer | Min  | Mean | Max  |
|-------|------|------|------|
| 4     | 3.4  | 3.6  | 3.7  |
| 6     | 4.6  | 5.1  | 5.5  |
| 8     | 5.9  | 6.6  | 7.4  |
| 12    | 7.7  | 8.3  | 9.6  |
| 16    | 8.8  | 10.4 | 12.2 |
| 20    | 12.8 | 15.4 | 17.8 |
| 24    | 18.9 | 23.2 | 27.5 |
| 28    | 29.5 | 35.0 | 40.5 |
| 31    | 55.6 | 74.4 | 89.6 |

Key trend: at early layers (4-8), all levels have similar norms. At late layers
(28-31), harder levels show wider norm spreads (Level 5 range at layer 31: 34.0
vs Level 1 range: 9.5). The model's internal variability increases with problem
difficulty, especially in the final layers where output representations are being
assembled.

## Appendix E: L5 Carry Distribution in the Full 810K Space

The carry_0 distribution across all 810,000 L5 problems is highly non-uniform.
Carry_0 is determined entirely by (a_units, b_units):

| carry_0 value | Number of digit pairs (a_units, b_units) | Count in 810K | Approximate fraction |
|---------------|------------------------------------------|---------------|---------------------|
| 0             | 14 pairs (e.g., 1x1, 1x2, ..., 3x3)     | ~113,400      | 14.0%               |
| 1             | 12 pairs                                 | ~97,200       | 12.0%               |
| 2             | 14 pairs                                 | ~113,400      | 14.0%               |
| 3             | 14 pairs                                 | ~113,400      | 14.0%               |
| 4             | 13 pairs                                 | ~105,300      | 13.0%               |
| 5             | 10 pairs                                 | ~81,000       | 10.0%               |
| 6             | 9 pairs                                  | ~72,900       | 9.0%                |
| 7             | 4 pairs (7x9, 8x9, 9x8, 9x7... etc)     | ~32,400       | 4.0%                |
| 8             | 1 pair (9x9 only)                        | ~8,100        | 1.0%                |

(Exact counts depend on digit pair enumeration; the table shows the pattern.)

The actual screening result confirmed these patterns. carry_0 = 8 has 8,100 total
problems and 197 correct answers (2.4% accuracy) in all 810K. carry_0 = 0 has
approximately 340,000 total problems and 33,019 correct (9.7% accuracy). The
two-phase approach includes ALL 8,100 carry_0 = 8 problems to capture every
available correct answer.

---

## 23. Carry Binning for Phase C/D

### Why binning is needed

The downstream analysis (Phase C subspace identification, Phase D LDA) groups
problems by concept value and computes centroids or within-class scatter matrices.
For this to work, each group needs at least 100 correct samples. The L5 correct
population has hard ceilings where this is impossible for individual carry values
(see Section 6, Hard Ceilings table).

### The binning decision

For each carry variable, values where the correct population is too sparse are binned
into a single "high" group. This preserves the data (the "all" population has thousands
of problems per value) while respecting statistical limits on the correct population.

| Carry | Individual values | Binned tail | Correct in tail | Total in tail |
|-------|------------------|-------------|-----------------|---------------|
| carry_0 | {0,1,2,3,4,5,6,7,8} | None needed | 197 (carry_0=8) | 8,100 |
| carry_1 | {0,1,...,11} | ">=12" | 199 | 7,974 |
| carry_2 | {0,1,...,12} | ">=13" | 216 | 22,588 |
| carry_3 | {0,1,...,8} | ">=9" | 148 | 17,157 |
| carry_4 | {0,1,...,4} | ">=5" | 159 | 23,392 |

This gives:
- **carry_0**: 9 classes (unchanged — carry_0=8 has 197 correct, above 100 floor)
- **carry_1**: 13 classes (12 individual + 1 binned)
- **carry_2**: 14 classes (13 individual + 1 binned)
- **carry_3**: 10 classes (9 individual + 1 binned)
- **carry_4**: 6 classes (5 individual + 1 binned)

Every class in this scheme has >= 100 correct samples, except carry_0=8 at 197.

### Why Option B (bin) over Option A (exclude) or Option C (flag)

**Option A (exclude tail values)** wastes data. The "all" population has thousands
of problems for these carry values — plenty for centroid estimation. Excluding them
biases the analysis away from the most interesting extreme-carry regime.

**Option C (keep all values, flag below-floor)** is dishonest. Computing a subspace
from 5 correct samples (carry_1=17) has no statistical power. The permutation test
with 5 samples cannot detect real signal. Better not to compute it.

**Option B (bin tail into one group)** preserves the data while respecting statistical
limits. The centroid for "carry_1 >= 12" captures "what does extreme carry_1 look
like" — which is the scientifically interesting question. The difference between
carry_1=14 and carry_1=16 is less interesting than the difference between carry_1=3
and carry_1 >= 12.

### Where binning is applied

Raw labels (in level_5.json) store exact carry values. Binning is applied only by
the Phase C/D analysis code when grouping problems into concept classes. This keeps
the data lossless and allows changing binning boundaries later without regenerating
anything.

### Binning applies only to L5

L3 and L4 do not need binning. At L3 (67.2% accuracy, 10,000 problems), even the
rarest carry values have sufficient correct samples for individual analysis. At L4
(29.0% accuracy, 10,000 problems), the carry ranges are smaller (max carry_0 = 8,
max carry_1 = 17) but accuracy is high enough that individual values are viable.

---

*End of document. All sections reflect the final pipeline run (March 18, 2026,
SLURM jobs 6645003 + subsequent). L5 model accuracy on the full input space is
6.11% (49,504 / 810,000). The 3.4% figure is the accuracy on the carry-stratified
analysis dataset. Use 6.11% when reporting L5 accuracy in the paper.*
