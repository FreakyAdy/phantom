# PHANTOM Deep Research Report: Can 5 tok/s Become 14 tok/s?

**Date**: 2026-09-17  
**Scope**: Engineering/research investigation into maximizing LLM inference speed on low-end consumer hardware  
**Repository**: [PHANTOM](https://github.com/FreakyAdy/phantom)  
**Standard**: Systems research, not marketing

---

## EXECUTIVE VERDICT

**Can 5 tok/s realistically become 14 tok/s on the same low-end hardware?**

### Dense 32B Models: **NO — NOT ON THE SAME DENSE MODEL**

For a **dense 32B model** running at **~3 tok/s** on an RTX 4050 (6 GB VRAM) + 24 GB DDR5:
- The **physical bandwidth ceiling** is ~3.4 tok/s (48 GB/s DDR5 ÷ 14 GB RAM-resident weights)
- **14 tok/s requires reading ~14 GB of weights in 71 ms**, which demands **~197 GB/s** of memory bandwidth
- The reference hardware provides **48 GB/s DDR5** for RAM-resident layers
- **This is a 4.1× gap that no software optimization can close for a dense model**

### MoE / Sparse Models: **YES — CONDITIONALLY ACHIEVABLE**

For a **30B MoE model** (like Qwen3-30B-A3B, 3.3B active parameters):
- PHANTOM already achieves **12.95 tok/s** on the same hardware
- With speculative decoding + kernel optimization, **14–18 tok/s is plausible**
- The active parameter footprint (~1.8 GB Q4) fits mostly within the bandwidth envelope

### The Honest Answer

The question "Can 5 tok/s become 14 tok/s?" is **answerable only when you specify the model architecture**:

| Model Type | Current | Target 14 tok/s | Verdict |
|---|:---:|:---:|---|
| Dense 32B (Qwen2.5-Coder-32B) | 2.88 tok/s | 14 tok/s | **IMPOSSIBLE** — 4.9× beyond DDR5 bandwidth wall |
| Dense 32B + Speculative | 2.88 tok/s | ~5–7 tok/s | **PARTIALLY** — spec. decoding helps but RAM bandwidth still dominates |
| MoE 30B (Qwen3-30B-A3B) | 12.95 tok/s | 14 tok/s | **YES** — achievable with optimization |
| MoE 30B + Speculative | 12.95 tok/s | ~18–24 tok/s | **LIKELY** — most active params fit in GPU cache |
| Dense 14B (Q4) | ~5–6 tok/s | 14 tok/s | **POSSIBLY** — with spec. decoding + kernel fusion |
| Dense 8B (Q4) | ~15–30 tok/s | 14 tok/s | **ALREADY THERE** — fits in VRAM |

> [!IMPORTANT]
> **The 5→14 question conflates two distinct problems**: (1) fitting the model in memory, and (2) reading the model fast enough. PHANTOM solves (1) excellently. Solving (2) for dense models requires either reducing bytes-per-token (sparsity, MoE, pruning) or changing the model architecture. No runtime optimization can make copper traces carry 4× more data.

---

## 1. PHANTOM CURRENT ARCHITECTURE AUDIT

### 1.1 What PHANTOM Actually Does

**Implemented & Verified** (with measured evidence):
- **Tiered memory placement**: GPU VRAM → Host RAM → NVMe SSD layer partitioning
- **In-place CPU SIMD evaluation**: RAM-resident layers evaluated by CPU (AVX2/AVX-512), avoiding PCIe weight streaming (ADR-006)
- **GGUF loader**: Native Q4_K_M weight loading without HuggingFace dequantization explosion
- **Capacity planner** (`phantom plan`): Predicts memory tier splits and tok/s within ±2.4% error
- **Wraith LSTM predictor**: CPU-side layer transition predictor (~0.44ms latency, +9.9% throughput when NVMe paging active)
- **Neural Cache**: KV state autoencoder (128→16 dim), 8× KV memory reduction, 100% needle recall at 32K context
- **MoE expert routing**: Sparse execution for MoE models, 9.93× FLOP reduction on Qwen3-30B-A3B
- **Phantom Pages**: NVMe tile paging with double-buffered prefetching
- **Chronos**: Multi-model context switching (~80ms for in-RAM models)

**Experimental / Partially Implemented**:
- Spectral Quantization (DCT FP8): Measured on synthetic tensors, 0.42 PPL delta claimed but not on production model runs
- CUDA kernels exist ([`kernels/`](file:///c:/Work/Projects/Solution%20is%20all%20You%20need/kernels)) but the Rust engine's `generate()` function returns a **placeholder string** — the actual CUDA execution path is **not wired up**
- Custom attention kernels ([`flash_attn_v3.cu`](file:///c:/Work/Projects/Solution%20is%20all%20You%20need/kernels/attention/flash_attn_v3.cu), [`gqa_kernel.cu`](file:///c:/Work/Projects/Solution%20is%20all%20You%20need/kernels/attention/gqa_kernel.cu)) exist as code but are not integrated into the runtime

**Not Implemented** (declared as future work):
- Speculative decoding
- Streaming chunked prefill
- Non-NVIDIA backends (Metal, ROCm)
- The entire Rust core engine inference path (all `generate()`, `serve()`, `calibrate()` calls are **commented out**)

### 1.2 Critical Audit Finding: The Runtime Architecture Gap

> [!CAUTION]
> **The Rust core engine ([`core/src/engine/mod.rs`](file:///c:/Work/Projects/Solution%20is%20all%20You%20need/core/src/engine/mod.rs)) is a skeleton.** The `generate()` method returns `"Generated response for: {prompt} (Simulated)"`. The actual inference path runs through Python using PyTorch/HuggingFace transformers with PHANTOM's custom GGUF loader and CPU SIMD evaluation. The Rust core and CUDA kernels are **architectural scaffolding, not production code**.

This means PHANTOM's current measured performance (2.88–12.95 tok/s) comes from:
1. PyTorch + HuggingFace transformers' own CUDA/CPU execution
2. PHANTOM's custom GGUF loader that keeps weights in compact format
3. PHANTOM's layer placement logic (GPU vs CPU)
4. Standard PyTorch attention and MLP kernels (not PHANTOM's custom CUDA kernels)

### 1.3 True Bottleneck Analysis

For a **dense 32B model** (Qwen2.5-Coder-32B Q4_K_M, ~18.5 GB weights):

| Phase | Bottleneck Type | Time per Token | Fraction |
|---|---|:---:|:---:|
| GPU VRAM layers (0–13, ~4.56 GB) | GPU memory bandwidth | ~27 ms | 8% |
| PCIe activation transfer (10 KB) | PCIe latency | ~1.3 μs | <0.01% |
| CPU RAM layers (14–63, ~14 GB) | **DDR5 bandwidth** | **~290 ms** | **84%** |
| RMSNorm, RoPE, sampling | CPU compute | ~15 ms | 4% |
| KV cache read/write | CPU memory bandwidth | ~10 ms | 3% |
| Python/scheduler overhead | Latency | ~5 ms | 1% |
| **Total** | | **~347 ms** | **~2.88 tok/s** |

**Conclusion**: **84% of time is spent reading 14 GB of weights from DDR5 RAM at ~48 GB/s.** This is a **pure memory bandwidth bottleneck** that no kernel fusion, GPU optimization, or scheduling improvement can address for dense models.

For a **MoE 30B model** (Qwen3-30B-A3B, 3.3B active):

| Phase | Bottleneck Type | Time per Token | Fraction |
|---|---|:---:|:---:|
| Router computation | GPU compute | ~47 μs | <0.1% |
| Active expert weights (~1.8 GB) | GPU/RAM bandwidth | ~37 ms | 48% |
| Attention (with GQA) | GPU compute | ~15 ms | 19% |
| Inactive expert skip | Zero cost | 0 ms | 0% |
| KV cache | Memory bandwidth | ~5 ms | 6% |
| Overhead | Latency/scheduling | ~20 ms | 26% |
| **Total** | | **~77 ms** | **~12.95 tok/s** |

**Key insight**: MoE models are 3–5× faster not because of any runtime magic but because they read **~7.8× fewer bytes per token**.

---

## 2. PHYSICAL PERFORMANCE MODEL

### 2.1 Fundamental Equation

For autoregressive single-token decode:

```
T_token = max(T_compute, T_memory) + T_overhead

where:
  T_memory = M_vram/BW_vram + M_ram/BW_ram + M_nvme/BW_nvme
  T_compute = 2 × A × P_active / FLOPS_peak
  T_overhead = T_pcie + T_sync + T_scheduler + T_sampling + T_dequant
```

**Variables**:
- `M_vram`, `M_ram`, `M_nvme`: Weight bytes in each tier
- `BW_*`: Effective bandwidth of each tier
- `A`: Activations per token (1 for dense, expert_count/total_experts for MoE)
- `P_active`: Active parameters
- `FLOPS_peak`: Peak compute throughput

### 2.2 Bandwidth Ceiling Derivation

```
tok/s_max = BW_effective / bytes_per_token

For Q4_K_M: ~4.5 bits/param → bytes_per_token ≈ P_active × 0.5625 bytes

Dense 32B Q4:  bytes_per_token ≈ 32.76B × 0.5625 ≈ 18.4 GB
MoE 30B Q4:   bytes_per_token ≈ 3.3B × 0.5625  ≈ 1.86 GB (active only)
Dense 14B Q4:  bytes_per_token ≈ 14B × 0.5625   ≈ 7.9 GB
Dense 8B Q4:   bytes_per_token ≈ 8B × 0.5625    ≈ 4.5 GB
```

### 2.3 Theoretical Maximum tok/s by Hardware

| Hardware | Bandwidth | Dense 32B Q4 | Dense 14B Q4 | Dense 8B Q4 | MoE 30B Q4 |
|---|:---:|:---:|:---:|:---:|:---:|
| RTX 4050 VRAM (192 GB/s) | 165 GB/s eff. | N/A (doesn't fit) | N/A | 36.7 tok/s | 88.7 tok/s |
| DDR5 Dual-Ch (51.2 GB/s) | 48 GB/s eff. | **2.6 tok/s** | **6.1 tok/s** | 10.7 tok/s | 25.8 tok/s |
| DDR4 Dual-Ch (38.4 GB/s) | 32 GB/s eff. | **1.7 tok/s** | **4.1 tok/s** | 7.1 tok/s | 17.2 tok/s |
| PCIe 4.0 x8 (16 GB/s) | 7.8 GB/s eff. | 0.42 tok/s | 0.99 tok/s | 1.7 tok/s | 4.2 tok/s |
| PCIe 3.0 x16 (16 GB/s) | 12.8 GB/s eff. | 0.70 tok/s | 1.6 tok/s | 2.8 tok/s | 6.9 tok/s |
| NVMe Gen4 (7 GB/s) | 1.8 GB/s eff. | 0.10 tok/s | 0.23 tok/s | 0.40 tok/s | 0.97 tok/s |

### 2.4 When the Model Breaks

The simple `tok/s = BW / bytes_per_token` model breaks when:

1. **Computation dominates** (batch size > 1, long-context prefill, FP32 on weak GPUs)
2. **Latency dominates** (many small kernel launches, Python overhead, synchronization)
3. **Cache effects** create non-linear performance cliffs (working set exceeds L3/LLC)
4. **Thermal throttling** reduces sustained bandwidth below peak
5. **Multi-tier evaluation** involves sequential dependencies (GPU must wait for CPU)
6. **Dequantization overhead** becomes significant with very aggressive quantization (Q2, Q3)

---

## 3. THE "5 → 14 TOK/S" ANALYSIS

### 3.1 Mathematical Derivation

**Starting point**: 5 tok/s → T_token = 200 ms  
**Target**: 14 tok/s → T_token = 71.4 ms  
**Required speedup**: 2.8×

For this speedup, we need to reduce **effective bytes read per token by 2.8×**, OR find **2.8× more bandwidth**, OR use **2.8× fewer sequential token evaluations per output token**.

### 3.2 Strategy Space

| Strategy | Mechanism | Potential Multiplier | Applicability |
|---|---|:---:|---|
| **Q4→Q3 quantization** | 25% fewer bytes | 1.25× | Dense models |
| **Q4→Q2 quantization** | 50% fewer bytes | Up to 1.5× (with dequant overhead) | Quality risk |
| **Speculative decoding** | Verify N tokens at once | 1.5–3× (theory) | All models |
| **MoE architecture** | Only read active experts | 3–10× | Requires MoE model |
| **Kernel fusion** | Reduce overhead | 1.05–1.15× | All models |
| **KV cache quant** | Reduce KV bandwidth | 1.02–1.1× (short ctx) | Long-context only |
| **Dynamic sparsity** | Skip inactive neurons | 1.2–2× (theory) | Requires model support |
| **Hot-weight cache** | Keep hot layers in GPU | 1.1–1.5× | Partial benefit |
| **Early exit** | Skip later layers | 1.2–1.5× | Quality risk |
| **Model reformulation** | Offline pruning/distillation | 1.3–2× | Requires offline work |

### 3.3 Can We Stack These?

**Optimistic stacking for a dense 14B model (starting at ~5 tok/s)**:

```
Step 1: Baseline                          5.0 tok/s
Step 2: Q4→Q3 quantization (-25% bytes)   6.2 tok/s  [Estimated, ~0.5 PPL delta]
Step 3: Speculative decoding (2× draft)   ~9.3 tok/s  [Estimated, 70% acceptance]
Step 4: Kernel fusion + overhead cut       ~10.0 tok/s [Estimated, 1.07× improvement]
Step 5: Hot-weight GPU residency           ~11.0 tok/s [Estimated, critical layers in VRAM]
         ──────────────────────────────────
         REALISTIC CEILING                ~11 tok/s
         
Step 6: Dynamic sparsity (aggressive)     ~13 tok/s  [SPECULATIVE, requires model support]
Step 7: + More aggressive quant (Q2.5)    ~14 tok/s  [SPECULATIVE, quality unknown]
```

**Verdict**: For a **14B dense model**, 14 tok/s is at the very edge of theoretical possibility with aggressive combined optimization. For a **32B dense model**, it remains **physically impossible**.

---

## 4. SPECULATIVE DECODING: DEEP ANALYSIS

### 4.1 The Math of Speculative Decoding for Bandwidth-Bound Workloads

**Key insight that many analyses miss**: Speculative decoding on consumer hardware with CPU/GPU split is **fundamentally different** from speculative decoding on a single GPU.

**Standard spec. decoding assumption** (single GPU):
```
Speedup ≈ 1 / (1 - α + α/γ + c_draft/(γ × c_target))

where:
  α = acceptance rate (~0.6–0.8)
  γ = draft sequence length
  c_draft = draft model cost
  c_target = target model verification cost for γ tokens
```

**PHANTOM's hybrid scenario** (CPU+GPU):
```
Draft model: runs in GPU VRAM at ~80 tok/s (Qwen2.5-0.5B, ~0.3 GB)
Target model: runs in CPU+GPU at ~3 tok/s (32B, 14 GB in RAM)
Draft latency: ~12.5 ms/token
Verification: Must read 14 GB from DDR5 for γ tokens at once

Key: Verification of γ tokens costs ≈ T_read_weights + γ × T_attention
     NOT: γ × T_read_weights (weights read once, shared across verified tokens)
```

This is the **critical opportunity**: verification amortizes weight reads.

### 4.2 Concrete Calculation

**Draft**: Qwen2.5-0.5B at 80 tok/s, γ=8 draft tokens
- Draft time: 8 × 12.5 ms = 100 ms
- Draft VRAM: ~0.3 GB (fits alongside target's GPU layers)

**Verification**: Read 14 GB RAM weights once, verify 8 tokens
- Weight read time: 14 GB / 48 GB/s = 292 ms (same as 1 token!)
- Extra compute for 8 tokens vs 1: ~15 ms (attention is cheap for batch=8 vs batch=1)
- Verification time: ~307 ms

**Acceptance**: At α=0.7, expected accepted tokens = ~4.7 per draft

**Effective throughput**:
```
Total time per cycle = 100 ms (draft) + 307 ms (verify) = 407 ms
Accepted tokens per cycle = 4.7 + 1 (guaranteed) = 5.7
Effective tok/s = 5.7 / 0.407 = 14.0 tok/s
```

> [!WARNING]
> **This calculation assumes the draft and verification phases are SEQUENTIAL (not overlapped), the verification cost is dominated by a SINGLE weight read (not γ reads), and acceptance rate is 0.7.** Actual implementation details could significantly alter this:
> - If verification requires γ separate weight reads → speed drops to ~4.5 tok/s (no improvement)
> - If acceptance rate drops to 0.5 → ~9.5 tok/s
> - If draft overhead forces GPU layer eviction → negative impact

### 4.3 The Critical Implementation Question

**Does the target model's verification batch γ tokens together such that weights are read once?**

In llama.cpp: **YES** — verification passes all γ tokens through the model in a single forward pass (like a mini-prefill), reading weights once.

In PHANTOM's current architecture: **UNCLEAR** — the hybrid CPU/GPU split may need to handle batch-8 verification differently than single-token decode. CPU SIMD kernels designed for GEMV (matrix-vector) must be replaced with GEMM (matrix-matrix) kernels for batch verification to actually amortize weight reads.

> [!IMPORTANT]
> **This is the single most important technical detail for PHANTOM's speculative decoding strategy.** If CPU SIMD kernels cannot efficiently handle batch-8 GEMM, the entire speculative decoding benefit evaporates.

### 4.4 Speculative Decoding for MoE Models

For Qwen3-30B-A3B (already 12.95 tok/s):
- Active params per token: ~1.8 GB Q4
- Draft model: Qwen2.5-0.5B at 80 tok/s
- Verification of γ=8: read ~1.8 GB once + attention for 8 tokens
- Verification time: 1.8 GB / 48 GB/s ≈ 37.5 ms + 15 ms compute = 52.5 ms
- Draft time: 100 ms
- Cycle time: 152.5 ms, accepted ~5.7 tokens
- Effective: 5.7/0.1525 = **37.4 tok/s** (theoretical)

But this assumes no expert routing overhead for batched verification. Realistic: **18–25 tok/s**.

---

## 5. QUANTIZATION AS SPEED, NOT JUST MEMORY

### 5.1 The Non-Monotonic Relationship

| Quantization | Bits/Param | Bytes for 32B | tok/s (DDR5 48 GB/s) | Quality (PPL) | Notes |
|---|:---:|:---:|:---:|:---:|---|
| FP16 | 16 | 65.5 GB | 0.73 | Baseline | Won't fit in RAM |
| Q8_0 | 8 | 32.8 GB | 1.46 | +0.05 | Barely fits |
| Q6_K | 6.5 | 21.3 GB | 2.25 | +0.10 | |
| Q5_K_M | 5.5 | 18.0 GB | 2.67 | +0.20 | |
| **Q4_K_M** | **4.5** | **14.7 GB** | **3.27** | **+0.40** | **Current PHANTOM** |
| Q3_K_M | 3.4 | 11.1 GB | 4.32 | +0.80 | Noticeable quality loss |
| Q2_K | 2.6 | 8.5 GB | 5.65 | +2.50 | Significant quality loss |
| IQ2_XXS | 2.1 | 6.9 GB | 6.96 | +5.00+ | Often unusable |

### 5.2 The Dequantization Overhead Trap

Moving to Q3_K_M or lower introduces **irregular memory access patterns** and **more complex dequantization arithmetic**:

- Q4_K_M: 32 values in 18 bytes, simple shift/mask
- Q3_K_M: 32 values in 14 bytes, more complex bit-packing
- Q2_K: 32 values in ~10.5 bytes, significant dequantization overhead

**llama.cpp benchmarks show**: Going from Q4_K_M to Q3_K_M can give 0–20% throughput improvement, but going to Q2_K can actually be **slower** on some hardware due to dequantization overhead exceeding bandwidth savings.

### 5.3 Recommendation for PHANTOM

**Q3_K_M** is the sweet spot for bandwidth-constrained inference:
- Saves ~25% bandwidth vs Q4_K_M
- Quality delta is typically +0.4–0.8 PPL (acceptable for most use cases)
- Dequantization overhead is manageable
- **Expected speedup: 1.15–1.25× (Estimated)**

---

## 6. MEMORY BANDWIDTH ESCAPE STRATEGIES

### 6.1 Weight Reuse / Cache Residency

**Idea**: Keep frequently accessed weight blocks in GPU L2 cache or CPU LLC.

**Reality check**:
- RTX 4050 has ~16 MB L2 cache. A single transformer layer is ~280 MB (32B Q4). **0.06 layers fit in GPU L2.**
- CPU L3 cache is typically 18–30 MB. **Same problem.**
- Cache-based weight reuse is effective only for **very small models** or **individual attention heads**

### 6.2 Hot/Cold Weight Separation (PowerInfer-Style)

**Idea**: Profile neuron activation frequency. Keep "hot" neurons (frequently activated) in GPU VRAM, compute "cold" neurons (rarely activated) on CPU.

**For dense models**: Activation sparsity exists (~30–50% ReLU zeros in some models) but modern models use **SiLU/SwiGLU** activations which are **not sparse** (no exact zeros). PowerInfer's 11× speedup was measured on models with **ReLU activations** (OPT, Falcon) where 90%+ neurons can be skipped.

**For SwiGLU models (Llama, Qwen, etc.)**: Hot/cold separation provides minimal benefit because there's no natural sparsity to exploit without model modification.

**Verdict**: Hot/cold is **not directly applicable** to modern dense transformer architectures without architectural changes. However, it is **very effective for MoE models** where expert routing creates natural sparsity.

### 6.3 Weight Streaming Compression

**Idea**: Store weights in a compressed format in RAM and decompress on-the-fly during inference.

**Analysis**:
- Q4 weights have very low entropy (~3.8 bits/param after quantization)
- LZ4 compression of Q4 weights typically achieves 1.1–1.3× compression
- Decompression throughput: LZ4 decompresses at ~5–8 GB/s per core
- **Net bandwidth gain**: minimal, because decompression CPU cost offsets the bandwidth savings

**PHANTOM's spectral quantization (DCT)** is an interesting attempt at this, but:
- DCT encoding is lossy (0.42 PPL delta)
- IDCT reconstruction adds compute overhead
- The claimed 2× compression may not translate to 2× throughput gain

### 6.4 Predictive Tensor Residency

**Idea**: Predict which tensors will be needed and pre-position them.

**For dense models**: This is useless — every layer is needed, in order, every token. The access pattern is perfectly predictable and perfectly sequential.

**For MoE models**: Expert prediction 1–2 tokens ahead could enable prefetching. PHANTOM's Wraith predictor currently predicts **layer transitions** (useful for NVMe paging) but not **expert routing** (which would be valuable for MoE prefetching).

---

## 7. PIPELINING AND OVERLAP OPPORTUNITIES

### 7.1 Current State: Sequential Execution

```
Token N:
  [GPU layers 0-13: 27ms] → [PCIe: 1.3μs] → [CPU layers 14-63: 290ms] → [Sample: 5ms]
  Total: ~322ms
```

### 7.2 Potential Overlap Architecture

```
Token N:
  [GPU layers 0-13: 27ms]────────────────────────────────────────────────
                          [PCIe]→[CPU layers 14-63: 290ms]→[Sample: 5ms]
  
  But: CPU and GPU can work concurrently!

Token N+1 (speculative):
  GPU drafts token N+1 while CPU evaluates token N's layers
```

**Theoretical improvement from pipelining alone** (without speculative decoding):
- Overlap GPU and CPU: save ~27ms → ~295ms → 3.39 tok/s (from 2.88)
- **Only 1.18× improvement** — because CPU dominates (84% of time)

### 7.3 Double-Buffered Speculative Pipeline

```
Cycle 1:
  GPU: [Draft 8 tokens: 100ms]
  CPU: [Idle or completing previous verification]

Cycle 2:
  GPU: [Attention for verification: 15ms]
  CPU: [Read weights, compute MLP for 8-token batch: 305ms]
  
Total: ~405ms for ~5.7 accepted tokens = ~14 tok/s
```

**This only works if**:
1. CPU GEMM (not GEMV) kernels exist for batch-8 evaluation
2. GPU can compute attention for batch-8 during CPU MLP computation
3. Draft model fits in remaining VRAM alongside target model's GPU layers

---

## 8. ROOFLINE ANALYSIS

### 8.1 Single-Token Decode

```
Arithmetic Intensity = FLOPs per byte loaded

For Dense 32B Q4_K_M single-token decode:
  FLOPs = 2 × 32.76B = 65.52 GFLOPs
  Bytes = 18.4 GB = 18,400 MB
  AI = 65.52 / 18400 = 0.00356 FLOPs/byte

RTX 4050 compute ceiling: ~5,400 GFLOPs (FP16)
RTX 4050 memory ceiling:  192 GB/s → needs AI > 28.1 to be compute-bound
DDR5 memory ceiling:      48 GB/s → needs AI > 112.5 to be compute-bound
```

**Result**: At AI = 0.0036, single-token decode is **~7,800× below the compute roofline**. The GPU is operating at **<0.02% utilization** during the decode phase. This workload is **catastrophically memory-bound**.

### 8.2 Batch-8 Verification (Speculative)

```
FLOPs = 2 × 32.76B × 8 = 524 GFLOPs
Bytes = 18.4 GB (same — weights read once)
AI = 524 / 18400 = 0.0285 FLOPs/byte
```

**Result**: Still deeply memory-bound, but **8× better AI**. The GPU moves from ~0.02% to ~0.16% compute utilization. Still far from efficient, but weight reads are amortized.

### 8.3 What Would Make Decode Compute-Bound?

```
Need AI > 28.1 for GPU compute-bound:
  Batch size needed: 28.1 / 0.00356 = ~7,900 tokens
  → Only possible in high-throughput serving (not interactive)

Need AI > 112.5 for DDR5 compute-bound:
  Batch size needed: 112.5 / 0.00356 = ~31,600 tokens
```

**Conclusion**: Interactive single-user inference will **never be compute-bound** on consumer hardware. The optimization target is **always memory bandwidth**.

---

## 9. EXISTING TECHNIQUES COMPARISON

| Technique | Speed Potential | Memory Reduction | Complexity | Risk | Best For |
|---|:---:|:---:|:---:|:---:|---|
| Speculative decoding (EAGLE-3) | 2–3× | None | Medium | Low | All models, single-user |
| MTP (Multi-Token Prediction) | 1.5–2.5× | Minimal | Low | Low | Models with MTP heads |
| N-gram speculation | 1.2–1.5× | None | Low | None | Repetitive/coding tasks |
| Q4→Q3 quantization | 1.15–1.25× | 25% | Low | Medium | Bandwidth-limited |
| KV cache quantization (INT8) | 1.02–1.1× | 50% KV | Low | Low | Long-context only |
| Hot-neuron caching (PowerInfer) | 2–11× | None | High | Medium | ReLU-activation models |
| Dynamic sparsity | 1.2–2× | Proportional | High | High | Specially trained models |
| LayerSkip / early exit | 1.5–2.2× | None | Medium | Medium | Fine-tuned models |
| Kernel fusion (dequant+GEMV) | 1.05–1.15× | None | Medium | Low | All |
| Persistent CUDA kernels | 1.02–1.05× | None | High | Low | GPU-resident models |
| Low-rank factorization | 1.3–2× | 30–50% | Medium | Medium | Offline transform |
| Expert prefetching (MoE) | 1.1–1.5× | None | Medium | Low | MoE models |

---

## 10. BEST KNOWN EXISTING APPROACHES

### 10.1 llama.cpp with Speculative Decoding

- **Status**: Production-ready, supports draft-model and EAGLE-3
- **Performance**: 1.4–3× speedup for low-batch decode
- **Relevance**: llama.cpp on the same hardware (RTX 4050, 6GB) with Qwen2.5-32B Q4_K_M and partial offload would get ~1–3 tok/s decode, comparable to PHANTOM
- **With speculation**: Could reach ~3–6 tok/s (estimated) on dense 32B
- **Limitation**: Same DDR5 bandwidth wall applies

### 10.2 PowerInfer

- **Status**: Specialized for activation-sparse models
- **Performance**: 11× on OPT/Falcon with ReLU; much less on SwiGLU models
- **Relevance**: Limited — modern models (Llama, Qwen) don't exhibit the extreme activation sparsity PowerInfer exploits

### 10.3 ExLlamaV2

- **Status**: GPU-only, requires full model in VRAM
- **Performance**: Fastest single-GPU inference
- **Relevance**: Cannot run 32B on 6GB VRAM — irrelevant for PHANTOM's use case

### 10.4 MLC LLM

- **Status**: Cross-platform (Metal, Vulkan, CUDA)
- **Performance**: Good for mobile/edge deployment
- **Relevance**: Similar bandwidth constraints on shared-memory architectures

---

## 11. TEN NOVEL ARCHITECTURAL CONCEPTS

### Idea 1: Batched Speculative Verification with CPU GEMM Kernels

- **Mechanism**: Replace CPU SIMD GEMV (1 token) with GEMM (8 tokens) during speculative verification, reading weights once for 8 tokens
- **Hardware**: CPU AVX-512 + GPU CUDA
- **Memory behavior**: Same total weight reads, amortized over 8 tokens
- **Compute behavior**: 8× more compute per weight read
- **Expected bottleneck**: CPU GEMM efficiency for small batch sizes
- **Why it might work**: Directly addresses the bandwidth wall by batching
- **Why it might fail**: Small-batch GEMM on CPU may not be 8× faster than 8× GEMV
- **Implementation difficulty**: Medium (need custom AVX-512 GEMM kernels)
- **Expected improvement**: **2–3× (Estimated)**
- **Physically plausible**: YES

### Idea 2: Expert-Aware Double-Buffer Prefetching for MoE

- **Mechanism**: Predict next token's expert routing using router logits from current token, prefetch expert weights to GPU while current experts execute
- **Hardware**: GPU VRAM + CPU RAM + async DMA
- **Memory behavior**: Overlaps expert weight loading with computation
- **Expected bottleneck**: Router prediction accuracy across tokens
- **Why it might work**: MoE router decisions often correlate between adjacent tokens
- **Why it might fail**: Mispredictions waste bandwidth
- **Implementation difficulty**: Medium
- **Expected improvement**: **1.3–1.8× on MoE models (Estimated)**
- **Physically plausible**: YES

### Idea 3: Hybrid Draft Architecture (CPU Draft + GPU Verify)

- **Mechanism**: Run a tiny draft model (0.5B) entirely on CPU using a separate thread, while GPU handles verification of the full model's VRAM layers
- **Hardware**: Multi-core CPU + GPU
- **Memory behavior**: Draft uses CPU cache; no VRAM consumed
- **Expected bottleneck**: CPU contention between draft and target model evaluation
- **Why it might work**: Utilizes idle CPU cores during GPU verification
- **Why it might fail**: CPU contention with target model's RAM layers
- **Implementation difficulty**: High
- **Expected improvement**: **1.5–2× (Estimated)**
- **Physically plausible**: YES

### Idea 4: Progressive Quantization Ladder

- **Mechanism**: Compress weights to Q2 for initial draft-quality evaluation, then selectively re-evaluate "uncertain" tokens at Q4 precision
- **Hardware**: CPU + GPU
- **Memory behavior**: 2× less bandwidth for "easy" tokens
- **Expected bottleneck**: Identifying which tokens need Q4 re-evaluation
- **Why it might work**: Many tokens (especially common words) are "easy" and tolerate Q2
- **Why it might fail**: Q2 quality may be too degraded; re-evaluation overhead
- **Implementation difficulty**: High
- **Expected improvement**: **1.3–1.8× (Speculative)**
- **Physically plausible**: UNCERTAIN

### Idea 5: Activation-Sparsified SwiGLU via Magnitude Gating

- **Mechanism**: Apply a learned magnitude threshold to SwiGLU activations, forcing small activations to zero, enabling sparse computation
- **Hardware**: CPU SIMD with sparse GEMV kernels
- **Memory behavior**: Only load weight rows corresponding to non-zero activations
- **Expected bottleneck**: Threshold calibration to maintain quality
- **Why it might work**: SwiGLU outputs do exhibit heavy-tailed distributions
- **Why it might fail**: Unlike ReLU, SwiGLU values near zero still carry information
- **Implementation difficulty**: Medium (offline calibration + sparse kernels)
- **Expected improvement**: **1.2–1.5× (Speculative)**
- **Physically plausible**: UNCERTAIN

### Idea 6: Persistent Layer Residency with VRAM Micro-Tiling

- **Mechanism**: Instead of whole-layer GPU residency, keep the **most bandwidth-critical sub-blocks** (attention projections, layer norms) of ALL layers in GPU VRAM, run only MLP blocks on CPU
- **Hardware**: GPU VRAM (6 GB) + CPU RAM
- **Memory behavior**: ~2 GB attention matrices in VRAM, ~12 GB MLP in RAM
- **Expected bottleneck**: PCIe activation transfer for each attention block
- **Why it might work**: Attention has lower arithmetic intensity than MLP; MLP dominates weight volume
- **Why it might fail**: Activation transfers may exceed PCIe budget; complex scheduling
- **Implementation difficulty**: High
- **Expected improvement**: **1.1–1.3× (Speculative)**
- **Physically plausible**: YES

### Idea 7: Learned Prefix-Caching for Speculative Drafts

- **Mechanism**: Cache hidden states from previous tokens' forward passes to warm-start speculative draft predictions, avoiding redundant computation
- **Hardware**: CPU RAM cache
- **Memory behavior**: ~100 MB cache for hidden state prefix
- **Expected bottleneck**: Cache invalidation when predictions are wrong
- **Why it might work**: Transformer hidden states are highly correlated between adjacent tokens
- **Why it might fail**: Stale cached states may corrupt predictions
- **Implementation difficulty**: Medium
- **Expected improvement**: **1.1–1.3× draft speedup (Estimated)**
- **Physically plausible**: YES

### Idea 8: NVMe-as-L4-Cache with Predictive Paging

- **Mechanism**: Use NVMe SSD as a 4th memory tier with OS-level predictive page fault handling, pre-staging cold layers before they're needed
- **Hardware**: NVMe Gen4 SSD
- **Memory behavior**: Reduces cold-start penalty for models exceeding RAM
- **Expected bottleneck**: NVMe bandwidth (1.4–1.8 GB/s)
- **Why it might work**: Double-buffered prefetching can hide NVMe latency
- **Why it might fail**: NVMe bandwidth is still 25× slower than DDR5
- **Implementation difficulty**: Medium (already partially implemented in PHANTOM)
- **Expected improvement**: **1.5–2× for NVMe-paged layers only (Measured: 0.39 vs ~0.12 tok/s)**
- **Physically plausible**: YES (limited)

### Idea 9: Multi-Model Ensemble Drafting

- **Mechanism**: Use multiple tiny models (0.1B, 0.5B, 1B) as an ensemble draft system, where each model specializes in different output patterns (code, prose, reasoning)
- **Hardware**: CPU + GPU
- **Memory behavior**: ~1 GB total for ensemble
- **Expected bottleneck**: Ensemble overhead and token selection logic
- **Why it might work**: Higher acceptance rate than single draft model
- **Why it might fail**: Ensemble coordination overhead may exceed gains
- **Implementation difficulty**: High
- **Expected improvement**: **1.1–1.5× over single-draft speculation (Speculative)**
- **Physically plausible**: YES

### Idea 10: Runtime Weight Decomposition (Low-Rank + Residual)

- **Mechanism**: Offline decompose each weight matrix W = L × R + ε (low-rank approximation + residual). During fast decode, evaluate only L × R (reading ~30% of parameters). Periodically evaluate full W for "difficult" tokens.
- **Hardware**: CPU + GPU
- **Memory behavior**: ~30% bandwidth for easy tokens, 100% for hard tokens
- **Expected bottleneck**: Determining which tokens need full evaluation; quality of low-rank approximation
- **Why it might work**: Weight matrices in trained LLMs are approximately low-rank
- **Why it might fail**: Rank-128 approximation of 4096×4096 matrices still requires ~2× compute; residual evaluation schedule
- **Implementation difficulty**: High (offline decomposition + runtime decision system)
- **Expected improvement**: **1.3–2× (Speculative)**
- **Physically plausible**: UNCERTAIN

---

## 12. TOP 5 IDEAS FOR PHANTOM

### #1: Batched Speculative Decoding with CPU GEMM Verification

**Exact Mechanism**: 
1. Keep Qwen2.5-0.5B (0.3 GB) in GPU VRAM as draft model
2. Generate γ=8 draft tokens at 80 tok/s (100ms)
3. Run all 8 tokens through target model's GPU layers in a single batched forward pass
4. Transfer 8 activation vectors (80 KB) to CPU
5. Run CPU SIMD GEMM (not GEMV) to evaluate 8 tokens against RAM-resident layers in a single weight read
6. Accept ~5.7 tokens (α=0.7), reject rest
7. Effective: ~14 tok/s for dense 32B

**Expected Benefit**: 2–3× throughput improvement  
**Implementation Effort**: 3–4 weeks (CPU GEMM kernels, draft model integration, tree attention)  
**Failure Modes**: CPU GEMM for batch=8 may not be significantly faster than 8×GEMV; acceptance rate may be lower than 0.7; draft model may not fit in remaining VRAM  
**Benchmark**: Measure tok/s with and without speculation on Qwen2.5-32B + Qwen2.5-0.5B draft  
**Success Threshold**: >6 tok/s on dense 32B  

### #2: MoE Expert Prefetching with Router Prediction

**Exact Mechanism**:
1. During current token's expert evaluation, examine router logits
2. Predict which experts will be needed for token N+1
3. Asynchronously DMA predicted expert weights from RAM to GPU VRAM
4. When token N+1 arrives, experts are already in VRAM → GPU-speed evaluation

**Expected Benefit**: 1.3–1.8× on MoE models (16–24 tok/s for Qwen3-30B-A3B)  
**Implementation Effort**: 2 weeks  
**Failure Modes**: Mispredicted experts waste DMA bandwidth; expert weights may be too large for VRAM cache  
**Benchmark**: Expert prediction accuracy and tok/s with/without prefetching  
**Success Threshold**: >15 tok/s on Qwen3-30B-A3B  

### #3: Aggressive Quantization (Q3_K_M) with Quality-Gated Re-evaluation

**Exact Mechanism**:
1. Store weights in Q3_K_M (25% smaller than Q4_K_M)
2. Run primary inference at Q3
3. Monitor output token confidence (logit margin)
4. For low-confidence tokens, re-evaluate with Q4 weights (loaded on demand)

**Expected Benefit**: 1.15–1.25× throughput with minimal quality degradation  
**Implementation Effort**: 1 week (use existing llama.cpp Q3 support)  
**Failure Modes**: Frequent low-confidence tokens negate bandwidth savings  
**Benchmark**: Measure throughput and quality metrics (PPL, accuracy) vs Q4 baseline  
**Success Threshold**: >3.3 tok/s on dense 32B with <0.5 PPL delta  

### #4: Hybrid GPU Attention + CPU MLP Architecture

**Exact Mechanism**:
1. Keep ALL attention projection matrices (Q, K, V, O) in GPU VRAM (~2 GB for 32B)
2. Keep ALL MLP matrices (gate, up, down) in CPU RAM (~12 GB)
3. Each token: GPU computes attention → transfer hidden state → CPU computes MLP → transfer back
4. Pipeline: GPU attention for token N+1 overlaps with CPU MLP for token N

**Expected Benefit**: Better GPU utilization, ~1.2× throughput from pipelining  
**Implementation Effort**: 4 weeks (custom layer-split execution, pipelining logic)  
**Failure Modes**: PCIe round-trips per layer (not per model) increase transfer overhead; attention may not be faster on GPU than CPU for batch=1  
**Benchmark**: Per-layer latency profiling and end-to-end tok/s  
**Success Threshold**: >3.5 tok/s on dense 32B  

### #5: llama.cpp Backend Integration

**Exact Mechanism**:
1. Replace PHANTOM's Python/PyTorch inference path with llama.cpp as the execution backend
2. Maintain PHANTOM's capacity planner, memory management, and user interface
3. Leverage llama.cpp's highly optimized CUDA + CPU kernels, speculative decoding, MTP support, and continuous community optimization
4. Add PHANTOM's Neural Cache and Wraith predictor as post-processing layers

**Expected Benefit**: Immediate access to state-of-the-art kernel optimization, speculative decoding, and quantization support  
**Implementation Effort**: 4–6 weeks  
**Failure Modes**: Loss of control over low-level optimization; integration complexity  
**Benchmark**: Compare PHANTOM+llama.cpp vs standalone llama.cpp on same models  
**Success Threshold**: Feature parity with at least 10% improvement from PHANTOM's memory management  

---

## 13. PROPOSED PHANTOM v2 ARCHITECTURE

```
                    ┌─────────────────────────┐
                    │     PHANTOM CLI/API      │
                    │  (plan, run, trace)      │
                    └──────────┬──────────────┘
                               │
                    ┌──────────▼──────────────┐
                    │    Speculative Engine    │
                    │  ┌──────┐  ┌──────────┐ │
                    │  │Draft │  │Acceptance │ │
                    │  │Model │  │ Verifier  │ │
                    │  │(GPU) │  │(GPU+CPU)  │ │
                    │  └──────┘  └──────────┘ │
                    └──────────┬──────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼────────┐ ┌────▼─────┐ ┌────────▼────────┐
    │   GPU Engine      │ │  Router  │ │   CPU Engine     │
    │ ┌──────────────┐  │ │ (MoE)   │ │ ┌──────────────┐ │
    │ │ VRAM Layers  │  │ │ Expert  │ │ │ RAM Layers   │ │
    │ │ Flash Attn   │  │ │ Predict │ │ │ SIMD GEMM    │ │
    │ │ Fused Kernels│  │ │ Prefetch│ │ │ AVX-512      │ │
    │ └──────────────┘  │ └─────────┘ │ └──────────────┘ │
    └─────────┬────────┘              └────────┬────────┘
              │                                │
    ┌─────────▼────────────────────────────────▼─────────┐
    │              Unified Memory Manager                 │
    │  ┌──────┐  ┌──────┐  ┌──────┐  ┌───────────────┐  │
    │  │ VRAM │  │ RAM  │  │ NVMe │  │ Neural Cache  │  │
    │  │ 6 GB │  │24 GB │  │ SSD  │  │ (KV 8× comp.) │  │
    │  └──────┘  └──────┘  └──────┘  └───────────────┘  │
    └────────────────────────────────────────────────────┘
```

Key changes from v1:
1. **Speculative Engine** as first-class component (not bolted on)
2. **CPU GEMM kernels** replacing GEMV for batch verification
3. **MoE Router Prediction** integrated with prefetching
4. **llama.cpp backend option** for kernel execution
5. **Unified Memory Manager** coordinating all tiers with speculative awareness

---

## 14. THE 5 → 14 TOK/S PATH

### For a Dense 32B Model (starting at 2.88 tok/s)

```
2.88 tok/s  [Measured — Current PHANTOM, DDR5 bandwidth wall]
    │
    ▼ Q4→Q3 quantization: -25% bytes, +0.4 PPL
3.5 tok/s   [Estimated — 1.21× improvement]
    │
    ▼ Speculative decoding (γ=8, α=0.7): batch verification with CPU GEMM
~7.0 tok/s  [Estimated — 2× improvement from batched weight reads]
    │
    ▼ Kernel fusion + overhead reduction: fused dequant-GEMM, reduced scheduling
~7.5 tok/s  [Estimated — 1.07× improvement]
    │
    ▼ Hybrid GPU-attention pipelining: overlap GPU attention with CPU MLP
~8.0 tok/s  [Estimated — 1.07× from pipelining]
    ═══════════════════════════════════════════════
    REALISTIC CEILING FOR DENSE 32B ON 6GB + 24GB DDR5
    ═══════════════════════════════════════════════
~8 tok/s    [Estimated — 2.8× over baseline]
```

### For an MoE 30B Model (starting at 12.95 tok/s)

```
12.95 tok/s [Measured — Current PHANTOM, Qwen3-30B-A3B]
    │
    ▼ Expert prefetching + VRAM caching
~15 tok/s   [Estimated — 1.16× from hiding transfer latency]
    │
    ▼ Speculative decoding (GPU-resident draft)
~22 tok/s   [Estimated — 1.5× from amortized expert loads]
    │
    ▼ Kernel optimization + fused expert execution
~24 tok/s   [Estimated — matches Cloud T4 result]
    ═══════════════════════════════════════════════
    REALISTIC CEILING FOR MOE 30B ON 6GB + 24GB DDR5
    ═══════════════════════════════════════════════
~24 tok/s   [Estimated — 1.85× over baseline]
```

### For a Dense 14B Model (starting at ~5 tok/s)

```
~5.0 tok/s  [Estimated — 14B Q4 on DDR5, ~7.9 GB weights]
    │
    ▼ Q4→Q3 quantization
~6.2 tok/s  [Estimated]
    │
    ▼ Speculative decoding (γ=8, α=0.7)
~11 tok/s   [Estimated — weight reads amortized over 8 tokens]
    │
    ▼ Hot layers in VRAM (more of 14B fits in 6GB)
~13 tok/s   [Estimated — ~3 GB in VRAM, ~4 GB in RAM]
    │
    ▼ Kernel fusion + overhead
~14 tok/s   [Estimated — AT THE EDGE]
    ═══════════════════════════════════════════════
    THEORETICAL CEILING FOR DENSE 14B ON 6GB + 24GB DDR5
    ═══════════════════════════════════════════════
~14 tok/s   [Estimated — 2.8× over baseline, TIGHT]
```

> [!IMPORTANT]  
> The **14B dense model** is the correct target for "5 → 14 tok/s." A 32B dense model cannot physically reach 14 tok/s on this hardware. The question should be reframed: **Which is the best model that can reach 14 tok/s?**

---

## 15. HARDWARE-SPECIFIC STRATEGY

### 4 GB GPU (GTX 1650, RTX 3050 4GB)

- **Max useful model**: 7B dense Q4 (~4 GB, fits mostly in VRAM) → 15–25 tok/s
- **With CPU offload**: 14B Q4 → ~3–4 tok/s (DDR4 limited)
- **MoE**: Qwen3-30B-A3B → ~6–10 tok/s (DDR4 bandwidth limits expert loading)
- **Recommendation**: Focus on 7B dense or MoE models with aggressive quantization

### 6 GB GPU (RTX 3050 6GB, RTX 4050)

- **Max useful model**: 8B dense Q4 (~4.5 GB) → 20–35 tok/s fully in VRAM
- **With CPU offload**: 32B Q4 → 2.5–3.5 tok/s; 14B Q4 → 5–6 tok/s
- **MoE**: Qwen3-30B-A3B → 12–15 tok/s (measured)
- **With speculation**: 14B → ~11–14 tok/s; 32B → ~5–8 tok/s
- **Recommendation**: MoE models for speed; dense 32B for quality; speculative decoding is the highest-ROI optimization

### 8 GB GPU (RTX 4060)

- **Max useful model**: 14B dense Q4 (~7.9 GB) → fits mostly in VRAM → 15–25 tok/s
- **With CPU offload**: 32B Q4 → 3–5 tok/s (more layers in fast VRAM)
- **MoE**: Qwen3-30B-A3B → ~20+ tok/s (more experts cached in VRAM)
- **With speculation**: 32B → ~7–12 tok/s
- **Recommendation**: Sweet spot for PHANTOM — 14B dense fits well, 32B with speculation is viable

### 12 GB GPU (RTX 4070)

- **Max useful model**: 32B Q4 (~18.5 GB, ~12 GB in VRAM) → 5–8 tok/s
- **With speculation**: 32B → ~10–16 tok/s (most layers in VRAM, fast verification)
- **Recommendation**: Dense 32B with speculative decoding crosses interactive threshold

### CPU-Only (DDR4 32 GB)

- **32B Q4**: ~1.7 tok/s (DDR4 ~32 GB/s)
- **14B Q4**: ~4 tok/s
- **8B Q4**: ~7 tok/s
- **Recommendation**: Use 8B models for interactive use; 14B for batch processing

---

## 16. THE IMPOSSIBLE ZONE

PHANTOM **cannot** accelerate beyond physics:

1. **Dense 32B on 6GB VRAM + DDR5**: Cannot exceed ~8 tok/s even with perfect speculative decoding (DDR5 bandwidth wall)
2. **Dense 70B on 6GB + 24GB RAM**: Cannot exceed ~0.5 tok/s (NVMe bandwidth wall, 15 GB in NVMe)
3. **Any model on DDR4**: Immediately 30–40% slower than DDR5 (32 vs 48 GB/s)
4. **Any model on PCIe Gen3 x4**: Weight streaming ceiling of ~3.5 GB/s
5. **CPU-only without AVX-512**: Significant performance penalty (~40% slower than AVX-512)
6. **GPU compute**: Will **never** be the bottleneck for single-user interactive inference on consumer hardware. Optimizing GPU utilization is solving the wrong problem.

---

## 17. NEGATIVE RESULTS

### Things That Sound Good But Don't Work

1. **Increasing GPU offload layers** beyond the VRAM-fits-comfortably point: Forces evictions, causes OOM, or creates PCIe thrashing. llama.cpp community reports confirm that `-ngl` values beyond the sweet spot cause **performance cliffs**.

2. **Weight compression (LZ4/zstd) for streaming**: Decompression CPU cost ≈ bandwidth saved. Net gain: ~0–5%. Not worth the complexity.

3. **PowerInfer-style neuron sparsity on SwiGLU models**: SwiGLU doesn't produce natural zeros. Activation sparsity of SwiGLU is typically 5–15%, not the 85–95% PowerInfer needs.

4. **KV cache quantization for short contexts**: At context ≤ 4K, KV cache is ~100 MB for 32B models. Quantizing saves 50 MB — negligible vs 18.5 GB model weights. Only matters at 16K+ context.

5. **GPU persistent kernels for offloaded models**: The GPU completes its VRAM layers in ~27ms. The CPU takes ~290ms. Making the GPU faster is optimizing 8% of the workload.

6. **Dynamic depth / early exit on pre-trained models**: Without fine-tuning, early-exit tokens are mostly garbage. LayerSkip requires training with layer dropout — not applicable to existing models.

7. **CUDA Graphs for hybrid CPU/GPU**: CUDA Graphs capture GPU-only computation. They cannot capture CPU SIMD evaluation, which is the bottleneck.

---

## 18. EXPERIMENTAL ROADMAP

### Phase 1: Cheap Experiments (1–2 weeks)

| Experiment | Hypothesis | Setup | Measurement | Expected Result |
|---|---|---|---|---|
| Q3_K_M throughput | 25% less bytes → 20%+ faster | Download Q3 GGUF, run through PHANTOM | tok/s, PPL | 3.5 tok/s, +0.4 PPL |
| Draft model VRAM fit | 0.5B draft fits alongside 32B GPU layers | Load both, measure remaining VRAM | VRAM usage | 4.86 GB / 6.0 GB |
| CPU GEMM vs GEMV | Batch-8 GEMM is faster than 8×GEMV | Benchmark numpy/blas GEMM(8×5120, 5120×14336) vs 8×GEMV | ms/operation | GEMM: 35ms vs GEMV: 290ms |
| Expert routing correlation | Adjacent tokens route to similar experts | Analyze Qwen3-30B-A3B routing logs | Expert hit rate | >60% overlap |

### Phase 2: Kernel Experiments (2–4 weeks)

| Experiment | Hypothesis | Setup | Measurement | Expected Result |
|---|---|---|---|---|
| Fused dequant-GEMM CPU kernel | Fusing dequant with GEMM reduces CPU overhead | Implement AVX-512 fused Q4 dequant + FP16 GEMM | tok/s vs baseline | 5–10% improvement |
| Batched verification prototype | 8-token verification reads weights once | Modify PHANTOM's CPU path for batch evaluation | tok/s with speculation | 5–7 tok/s |
| Expert prefetch DMA | Async expert DMA overlaps with execution | Implement cuda memcpy async for expert weights | MoE tok/s | 15+ tok/s |

### Phase 3: Runtime Architecture Experiments (4–8 weeks)

| Experiment | Hypothesis | Setup | Measurement | Expected Result |
|---|---|---|---|---|
| Full speculative pipeline | Draft + verify + accept loop | End-to-end speculative decoding implementation | tok/s, acceptance rate | 7+ tok/s dense 32B |
| llama.cpp backend integration | Production kernels outperform Python/PyTorch path | Replace PHANTOM inference with llama.cpp server + PHANTOM frontend | tok/s, features | Comparable speed + PHANTOM UX |
| Hybrid attention-MLP split | GPU attention + CPU MLP pipeline | Custom layer-split execution engine | tok/s | 3.5+ tok/s |

### Phase 4: Novel Research (8–16 weeks)

| Experiment | Hypothesis | Setup | Measurement | Expected Result |
|---|---|---|---|---|
| Low-rank weight decomposition | SVD rank-128 captures 90%+ of weight energy | Offline SVD of Q4 weight matrices | PPL degradation, speed | 1.3× speed, +0.3 PPL |
| Adaptive precision routing | Easy tokens at Q2, hard tokens at Q4 | Confidence-gated dual-precision pipeline | average tok/s, quality | 1.3× speed, <0.2 PPL delta |
| SwiGLU magnitude gating | Thresholded SwiGLU creates 30%+ sparsity | Calibrate thresholds on validation set | sparsity, PPL | 30% sparsity, +0.5 PPL |

---

## 19. THE PHANTOM BREAKTHROUGH

### Core Idea

**Batched Speculative Verification is the single most promising direction for PHANTOM.**

The key insight is that PHANTOM's hybrid CPU/GPU architecture creates a unique opportunity: the CPU RAM bandwidth bottleneck can be **amortized across multiple tokens** by replacing sequential GEMV with batched GEMM during speculative verification.

### Why It Works

1. **Single-token decode reads 14 GB from DDR5 for ONE token** (290 ms)
2. **Batch-8 verification reads 14 GB from DDR5 for EIGHT tokens** (~305 ms)
3. **Draft model generates 8 tokens in 100 ms** (runs in GPU VRAM)
4. **Net result**: 5.7 accepted tokens in 405 ms = **14 tok/s** (vs 2.88 tok/s baseline)

### Why Existing Runtimes Don't Already Completely Solve This

- **llama.cpp** supports speculative decoding, but its CPU+GPU split path may not optimize GEMM for small batches on CPU
- **vLLM/TensorRT-LLM** don't support CPU offloading at all
- **ExLlamaV2** is GPU-only
- **PowerInfer** focuses on neuron sparsity, not speculative batching

PHANTOM's **in-place CPU SIMD evaluation** architecture is uniquely positioned to exploit batch verification because it already has the machinery for CPU-side weight evaluation — it just needs GEMM kernels instead of GEMV.

### Expected Bottleneck

CPU GEMM efficiency for batch=8 on Q4-dequantized weights. If OpenBLAS/MKL GEMM for (8×5120)×(5120×14336) takes ~305 ms (same as 1× GEMV), the full 2.8× speedup materializes. If GEMM overhead causes it to take 500 ms, speedup drops to ~1.7×.

### Prototype Design

```python
# Phase 1: Draft
draft_tokens = draft_model.generate(k=8)  # 100ms on GPU

# Phase 2: Verify (CPU GEMM path)
activations = gpu_layers(draft_tokens)     # batch=8, GPU VRAM layers, ~30ms
cpu_output = cpu_gemm_evaluate(            # batch=8, CPU RAM layers, ~305ms
    activations,                           # [8, hidden_dim]
    ram_layers                             # read once, compute for all 8
)
logits = lm_head(cpu_output)               # [8, vocab_size]

# Phase 3: Accept/Reject
accepted = verify_tokens(draft_tokens, logits)  # ~5.7 accepted tokens
```

### First Experiment

**Benchmark CPU GEMM throughput**:
```bash
# Measure: Time for GEMM(8×5120, 5120×14336) with Q4 dequant
# vs: 8× GEMV(5120, 5120×14336) with Q4 dequant
# Hardware: Intel Core i7 (reference machine)
# Expected: GEMM ≈ GEMV (bandwidth-bound, same total bytes)
# If GEMM ≈ GEMV → speculative decoding gives ~2.8× speedup
```

### Success Threshold

**>6 tok/s on Qwen2.5-Coder-32B on RTX 4050 + 24GB DDR5** with verified output quality (100% greedy agreement with baseline).

### Failure Threshold

**<4 tok/s** with speculation enabled (indicating that overhead exceeds amortization benefit).

### Publishability

**YES** — if the batched CPU GEMM verification technique demonstrates a measurable speedup on real hardware with real models, this would be a publishable contribution to the edge inference literature. The key novelty is the formal analysis of when GEMM batch verification is profitable in a hybrid CPU/GPU split architecture, with experimental validation.

### Implementable by Small Open-Source Team?

**YES** — the core implementation requires:
1. CPU GEMM kernel for Q4-dequantized batch evaluation (~2 weeks)
2. Draft model integration and tree-attention verification (~2 weeks)  
3. Accept/reject loop and KV cache management (~1 week)

Total: ~5 weeks for a competent systems programmer.

---

## 20. FINAL SUMMARY TABLE

| Question | Answer |
|---|---|
| Can 5→14 tok/s on dense 32B? | **NO** — physics prevents it (DDR5 wall) |
| Can 3→8 tok/s on dense 32B? | **YES** — via speculative decoding |
| Can 5→14 tok/s on dense 14B? | **POSSIBLY** — with aggressive combined optimization |
| Can 13→24 tok/s on MoE 30B? | **LIKELY** — with expert prefetching + speculation |
| Is PHANTOM's architecture sound? | **YES for memory tiering; NO for kernel execution** (skeleton engine) |
| Should PHANTOM build its own CUDA kernels? | **NO** — integrate llama.cpp or use PyTorch's optimized kernels |
| What's the single biggest win? | **Speculative decoding with batched CPU GEMM verification** |
| Is 14 tok/s the right target? | **DEPENDS on model** — 14B dense or MoE 30B, not 32B dense |
| Is this publishable research? | **YES** — batched CPU GEMM verification in hybrid architectures |

---

*Report generated 2026-09-17. All estimates labeled [Estimated] or [Speculative] require experimental validation before being treated as facts. All measurements labeled [Measured] correspond to existing PHANTOM benchmark data.*
