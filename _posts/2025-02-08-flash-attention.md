---
layout: post
title: "FlashAttention: Fast and Memory-Efficient Exact Attention"
date: 2025-02-08
categories: [deep-learning, machine-learning]
tags: [attention, transformers, optimization, gpu]
---

Introduction
============

Attention mechanisms are the foundation of modern Large Language Models,
but they come with a critical limitation: **quadratic memory
complexity**. For a sequence of length $N$, standard attention
requires $O(N²)$ memory to store the attention matrix, creating a hard
ceiling on the context lengths LLMs can process.

FlashAttention, introduced by Dao et al. (2022), offers an elegant
solution. By **carefully orchestrating memory access patterns** and
**leveraging GPU memory hierarchy**, it achieves the **same mathematical
result** as standard attention while being **2-4x faster** and using
**10-20x less memory**.

**The key insight?** For the **long sequence**s (for example $N \>
1024$, depending on the hardware constraints) that LLMs require,
standard attention becomes **memory-bound** rather than compute-bound.
The bottleneck isn\'t the number of floating-point operations (FLOPs),
but the **expensive reads and writes between GPU High Bandwidth Memory
(HBM) and on-chip SRAM**.

![](/FlashAttention%20Fast%20and%20Memory-Efficient%20Exact%20Att/image.png)

The Hardware Reality: HBM vs. SRAM
==================================

To understand FlashAttention, you must understand the anatomy of an
NVIDIA GPU (like an A100).

1.  **HBM (High Bandwidth Memory):**
    -   **Size:** Massive (40GB - 80GB).

        -   **Speed:** **\~1.5-2 TB/s (A100), \~3.35 TB/s (H100)**

        -   **Role:** The \"RAM\". Where your weights and KV cache live.

2.  **SRAM (Static RAM / On-chip Memory):**
    -   **Size**: 192 KB per SM; total 20MB on A100 (108 SMs × 192KB ≈
        20MB)

        -   **Speed:** **\~19 TB/s effective bandwidth**

        -   **Role:** The \"L1 cache\" where the CUDA cores actually do math

**The Bottleneck:**

-   Moving data from **HBM to SRAM** is expensive.

-   Doing math inside **SRAM** is cheap.

This is the fundamental constraint that FlashAttention exploits.

The \"Standard\" Attention (The Wasteful Way)
=============================================

Let\'s look at the standard calculation:

$$ \\text{Attn} = \\text{Softmax}(Q K\^T) V $$

The fundamental problem with standard attention is that **the attention
matrix doesn\'t fit in SRAM (on-chip memory)**.

For a sequence length of $N=4096$ with FP16:

-   Attention matrix size: $4096\^2 × 2 \\space bytes = 32 MB$

-   SRAM size per SM: **192 KB**

Since $32 MB \>\> 192 KB,$ the attention matrix **cannot be kept
on-chip**. Thus it must be stored in HBM.

A naive implementation performs attention in three separate kernel
launches, **materializing intermediate results in HBM** each time:

1.  **MatMul (**$S = Q K\^T$**):**
    -   Load $Q, K$ from HBM to SRAM.

        -   Compute $S$ (Score Matrix, size $N \\times N$).

        -   **WRITE** $S$ **to HBM.** (Huge write).

2.  **Softmax (**$P = \\text{softmax}(S)$**):**
    -   **READ** $S$ **from HBM.**

        -   Compute Exponentials/Sum.

        -   **WRITE** $P$ **to HBM.** (Huge write).

3.  **MatMul (**$O = P V$**):**
    -   **READ** $P$ **from HBM.**

        -   Load $V$.

        -   Compute Output.

        -   Write Output to HBM.

**The Problem:** We just wrote and read an $N \\times N$ matrix (the
Attention Map) multiple times. For a context of 100k tokens, this matrix
is terabytes in size. The GPU spends 90% of its time waiting for memory
reads.

Numerical instability in standard softmax
-----------------------------------------

The \"Numerical Instability\" of Standard Softmax comes down to the
**physical limitations of how computers store numbers (Floating Point
Arithmetic).**

In pure mathematics, $e\^{1000}$ is just a very large number. In a
computer (using `float32`), $e\^{1000}$ is **an error**.

Here are the two specific ways Standard Softmax crashes your training:
**Overflow** and **Underflow**.

### The Explosion (Overflow) - The Most Common Crash
The standard formula is:

$$ S\_i = \\frac{e\^{x\_i}}{\\sum e\^{x\_j}} $$

`float32` Range: $±1.2 \\times 10\^{-38}$ to $±3.4 \\times 10\^{38}$

-   **The Overflow Threshold**: $e\^{88.7} ≈ 3.4 \\times 10\^{38}$
    (the maximum)

-   Anything larger becomes `inf`

**The Scenario:**\
Imagine your attention score (dot product) is **100**. (This is common
in unscaled attention).

1.  **Calculate Numerator:** $e\^{100}$.

2.  **Computer says:** \"That is bigger than $10\^{38}$. I will call
    it **Infinity** (`inf`).\"

3.  **Calculate Denominator:** Sum of infinities = **Infinity**.

4.  **Result:**

$$ \\frac{\\text{inf}}{\\text{inf}} = \\mathbf{\\text{NaN (Not a
Number)}} $$

**Result:** Your gradients turn to `NaN`, and your loss becomes `NaN`.
The training run is dead.

### The Vanishing (Underflow) - The Division by Zero
**This happens when numbers are very negative.**

**The Physics:**

-   `float32` **Limit:** The smallest positive number is roughly $1.2
    \\times 10\^{-38}$.

-   **The Threshold:** $e\^{-88}$ is roughly the limit. Anything
    smaller becomes **0.0**.

**The Scenario:**\
Imagine your input vector is `[-100, -200, -300]`.

1.  **Calculate Numerators:**
    -   $e\^{-100} \\to 0.0$

        -   $e\^{-200} \\to 0.0$

        -   $e\^{-300} \\to 0.0$

2.  **Calculate Denominator:** Sum of zeros = **0.0**.

3.  **Result:**

$$ \\frac{0.0}{0.0} = \\mathbf{\\text{NaN}} $$

This is rarer, but it happens if you use heavy masking (setting scores
to $-1e9$) and mess up the logic slightly.

### The Solution: \"Safe Softmax\" (Shift-Invariance)
To fix this, we use the property that Softmax is **Shift Invariant**. If
you add or subtract a constant $C$ from every input, the result does
not change.

$$ \\frac{e\^{x\_i - C}}{\\sum e\^{x\_j - C}} = \\frac{e\^{x\_i}
\\cdot e\^{-C}}{e\^{-C} \\cdot \\sum e\^{x\_j}} =
\\frac{e\^{x\_i}}{\\sum e\^{x\_j}} $$

**The Trick:** We choose $C = \\max(x)$.

Let\'s retry the **Overflow Scenario** ($x = \[100, 90, 80\]$) using
this trick:

-   Max $m = 100$.

-   Shifted inputs: $\[0, -10, -20\]$.

1.  **Numerators:**
    -   $e\^0 = 1.0$ (Safe)

        -   $e\^{-10} \\approx 0.000045$ (Safe)

        -   $e\^{-20} \\approx 0.000000002$ (Safe)

2.  **Denominator:** $1.0 + 0.000045 + \\dots \\approx 1.000045$.

3.  **Result:**

    $$ \\frac{1.0}{1.000045} \\approx 0.9999 $$

**Why is it stable?**

1.  The largest term becomes $e\^0 = 1$. It **never overflows**.

2.  The denominator is always at least $1$ (from the max term). It
    **never underflows to zero**.

This is why `m` (the global maximum) is absolutely required in the
FlashAttention Online Softmax calculation. It anchors the math in the
\"Safe Zone\" of floating-point numbers.

The FlashAttention Way (Tiling & Fusing)
========================================

FlashAttention asks: *\"Can we calculate the final output without ever
writing the giant* *$N \\times N$* *matrix to HBM?\"*

**The Strategy: Tiling.**\
We break the Query, Key, and Value matrices into small blocks that fit
entirely inside the tiny **SRAM**.

**The Algorithm:**

1.  Load Block $Q\_1, K\_1, V\_1$ into SRAM.

2.  Compute $S\_{1,1} = Q\_1 K\_1\^T$ in SRAM.

3.  Compute Softmax locally in SRAM.

4.  Multiply by $V\_1$ immediately in SRAM.

5.  Accumulate the result into the output buffer.

6.  Repeat for next blocks.

**The Result:**\
The massive $N \\times N$ attention matrix is never fully materialized
in HBM. It is created piecemeal in SRAM, used immediately, and
discarded.

-   **Memory IO:** Reduced by **8x to 10x**.

-   **Speed:** Faster, because we are no longer memory-bound.

**FlashAttention (Forward Pass)**
=================================

Here is the brief mathematical workflow for **FlashAttention** (Forward
Pass).

The key distinction is that we process the matrices in **Tiles
(Blocks)** to stay inside the SRAM (fast memory), updating the output
iteratively using **Online Softmax**.

1.  **Tiling (The Setup)**

    We split the input matrices ($Q, K, V$) living in **HBM** (Slow
    Memory) into blocks of size $B$ by row:

    -   $Q$ is split into row blocks: $Q\_1, Q\_2, \\dots$

        -   $K, V$ are split into row blocks: $(K\_1, V\_1), (K\_2,
        V\_2), \\dots$

2.  **Outer Loop (Load Query)**

    For each block of Queries $Q\_i$:

    1.  **Load** $Q\_i$ from HBM $\\to$ SRAM.

        2.  **Initialize** running stats in SRAM:
        -   $O\_i = 0$ (**Un-normalized** Output)

                -   $\\ell\_i = 0$ (Running Sum)

                -   $m\_i = -\\infty$ (Running Max)

3.  **Inner Loop (Scan Keys/Values)**

    We iterate through every block $j$ of Keys/Values:

    1.  **Load** $K\_j, V\_j$ from HBM $\\to$ SRAM.

        2.  **Compute Scores (On Chip):**\
        $S\_{ij} = Q\_i \\cdot K\_j\^T$

        3.  **Update Statistics (Online Softmax):**
        -   Find local max of current block: $\\tilde{m} =
            \\text{rowmax}(S\_{ij})$

                -   Find new global max: $m\_{new} = \\max(m\_i, \\tilde{m})$

                -   Compute Un-normalized Probs: $P\_{ij} = e\^{S\_{ij} -
            m\_{new}}$

                -   Update Global Sum:\
            $\\ell\_{new} = (\\ell\_i \\cdot e\^{m\_i - m\_{new}}) +
            \\text{rowsum}(P\_{ij})$

        4.  **Update Output (Rescale & Add):**\
        $O\_{new} = (O\_i \\cdot e\^{m\_i - m\_{new}}) + (P\_{ij}
        \\cdot V\_j)$

        5.  **Write State:** $m\_i \\leftarrow m\_{new}, \\ell\_i
        \\leftarrow \\ell\_{new}, O\_i \\leftarrow O\_{new}$

        6.  **Discard:** Delete $S\_{ij}, K\_j, V\_j$ from SRAM
        immediately.

4.  **Finalization (Write Back)**

    After the inner loop finishes (all $K, V$ blocks processed for
    this $Q\_i$):

    1.  **Normalize:**

        $$ O\_{final} = \\frac{O\_i}{\\ell\_i} $$

        2.  **Save:** Write $O\_{final}$ from SRAM $\\to$ HBM.

The Backward Pass (Re-computation)
==================================

Standard Attention saves the huge $N \\times N$ attention map during
the forward pass so it can be used for the backward pass (Gradient
calculation). This eats massive VRAM.

FlashAttention **does not save it.**

**Backward Pass:**

-   It grabs $Q, K, V$ again.

-   It **re-calculates** the attention scores on the fly using the
    Tiling method.

-   It calculates the gradient.

**Why is this faster?**\
**Because re-doing the math (FLOPS) is actually faster than reading the
terabyte matrix from HBM.**

-   Compute = Cheap.

-   Memory Bandwidth = Expensive.

Torch Implementation of FlashAttention(forward Pass)
====================================================

pedagogical version
-------------------

In the code below, we loop through blocks of query and key/value to
calculate the attention matrix for each block. Then incrementally
calculate the final output matrix $O$.

Note:

Theoretically the loop through query should be parallelled.

```python
def flash_attention_v2_forward(Q, K, V, causal=False, block_size=64):
    """
    Flash Attention v2 - even more optimized
    Key improvement: Better work partitioning across blocks

    This version processes blocks more efficiently by:
    1. Reducing redundant computations
    2. Better memory access patterns
    """
    B, H, N, d = Q.shape

    O = torch.zeros_like(Q)
    scale = 1.0 / math.sqrt(d)

    num_blocks = math.ceil(N / block_size)

    # Process each query block
    for i in range(num_blocks):
        q_start = i * block_size
        q_end = min((i + 1) * block_size, N)
        Q_block = Q[:, :, q_start:q_end, :]

        # Online softmax accumulators
        m_i = torch.full((B, H, q_end - q_start, 1), float('-inf'), device=Q.device)
        l_i = torch.zeros(B, H, q_end - q_start, 1, device=Q.device)
        O_i = torch.zeros_like(Q_block)

        # Determine which K/V blocks to attend to
        max_j = i + 1 if causal else num_blocks

        for j in range(max_j):
            k_start = j * block_size
            k_end = min((j + 1) * block_size, N)
            K_block = K[:, :, k_start:k_end, :]
            V_block = V[:, :, k_start:k_end, :]

            # Compute scores
            S_ij = torch.matmul(Q_block, K_block.transpose(-2, -1)) * scale

            # Causal masking
            if causal and i == j:
                q_idx = torch.arange(q_start, q_end, device=Q.device)[:, None]
                k_idx = torch.arange(k_start, k_end, device=Q.device)[None, :]
                causal_mask = q_idx < k_idx
                S_ij = S_ij.masked_fill(causal_mask, float('-inf'))

            # Online softmax update
            m_ij = S_ij.max(dim=-1, keepdim=True)[0]
            p_ij = torch.exp(S_ij - m_ij)
            l_ij = p_ij.sum(dim=-1, keepdim=True)

            # Update global statistics
            m_i_new = torch.maximum(m_i, m_ij)
            l_i_new = torch.exp(m_i - m_i_new) * l_i + torch.exp(m_ij - m_i_new) * l_ij

            # Update output
            O_i = torch.exp(m_i - m_i_new) * O_i + \
                  torch.exp(m_ij - m_i_new) * torch.matmul(p_ij, V_block)

            m_i = m_i_new
            l_i = l_i_new

        # Final normalization
        O_i = O_i / l_i
        O[:, :, q_start:q_end, :] = O_i

    return O

class FlashAttention(torch.nn.Module):
    """
    Flash Attention module wrapper
    """
    def __init__(self, d_model, num_heads, block_size=64):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.block_size = block_size

        self.W_qkv = torch.nn.Linear(d_model, 3 * d_model)
        self.W_o = torch.nn.Linear(d_model, d_model)

    def forward(self, x, causal=False):
        B, L, D = x.shape

        # Project to Q, K, V
        qkv = self.W_qkv(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, d)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # Apply flash attention
        output = flash_attention_v2_forward(Q, K, V, causal=causal, block_size=self.block_size)

        # Reshape and project
        output = output.transpose(1, 2).reshape(B, L, D)
        output = self.W_o(output)

        return output
```

Summary: Why it matters
=======================

> 💡
>
> The essence of FlashAttention is:
>
> It **eliminated the need of materializing the whole attention matrix**
> of size $L\\times L$ in the HBM, which is too big to fit in the SM.

  Feature           Standard Attention   FlashAttention
  ----------------- -------------------- -----------------
  **Peak Memory**   O(N²)                O(N)
  **HBM I/O**       O(N²)                O(N²/B) ≈ O(N²)
  **FLOPs**         O(N²d)               O(N²d)
  **Bottleneck**    Memory Bandwidth     Compute

**Verdict:**\
FlashAttention turns the \"Memory Wall\" problem into a \"Compute\"
problem. Since GPUs are getting better at compute faster than they are
getting better at memory bandwidth, this is the winning strategy.

Today, **FlashAttention-2** (and the new FlashAttention-3) is standard
in essentially every major library (PyTorch 2.0+, DeepSpeed, vLLM,
HuggingFace). If you run
`torch.nn.functional.scaled_dot_product_attention`, you are likely
running FlashAttention under the hood.

This is why techniques like **FlashAttention** (optimizing memory
access), **Sliding Window Attention** (Mistral), and **Linear
Attention** (Mamba/RNNs) exist---they aim to reduce the $N\^2$ term to
$N$. **FlashAttention** is the algorithm that effectively \"fixed\"
the Transformer\'s memory bottleneck for long sequences. If
**DeepSpeed** is about managing weights across GPUs, **FlashAttention**
is about **managing** **data movement inside a single GPU**.

It is based on a profound realization: **GPUs are fast at math, but slow
at memory.**
