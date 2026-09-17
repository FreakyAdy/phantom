## Executive Verdict

**ONLY UNDER CERTAIN CONDITIONS:** turning a genuine **5 tok/s** target into **~14 tok/s** on unchanged low-end hardware is plausible only by reducing *target-model passes per emitted token*—principally lossless speculative decoding—and, for suitable models, reducing *active weight bytes* through model-native sparsity or MoE routing. It is not plausible for a fully dense, RAM-resident model merely through paging, prefetching, kernel fusion, or scheduling.

The immediate, uncomfortable finding is that [PHANTOM](https://github.com/FreakyAdy/phantom) does **not currently implement the claimed end-to-end tiered inference engine**. Its Rust `generate()` returns a simulated response; its Python `run` path delegates actual inference to a local Ollama server when available, otherwise tries Hugging Face Transformers. Its master benchmark suite contains synthetic values and hard-coded “measured” throughputs. Therefore, its reported 2.88, 3.63, and 12.95 tok/s results are **not presently reproducible evidence that PHANTOM itself achieved them**.

For a dense model at 5 tok/s whose weights are read from DDR5 every token, a 2.8× raw decode speedup means cutting effective target-side bytes from roughly $W$ to $\leq 0.36W$, or making each target pass validate about 2.8 final tokens on average. No prefetcher can do that on its own: prefetch can hide latency, not create bandwidth.

## PHANTOM Current Architecture Audit

### What is actually implemented

| Subsystem | Repository evidence | Status |
|---|---|---|
| CLI, model registry, GGUF parsing, hardware detection, model-size planner | Python implementation exists | **Implemented tooling** |
| Direct local generation | `cmd_run()` tries Ollama, then Transformers `generate()` | **Implemented as delegation, not PHANTOM inference** |
| Rust “core engine” | `generate()` is pseudocode and returns `Generated response … (Simulated)` | **Skeleton / not an inference engine** |
| CUDA files for sparse MLP, DCT, GQA, attention, KV encode/decode | CUDA source exists, but no demonstrated binding into the live Python generation path | **Experimental / disconnected** |
| VRAM/RAM/NVMe tier manager | Rust allocation and bookkeeping structures exist | **Partial infrastructure** |
| Wraith predictor | Python implementation and synthetic benchmark | **Experimental; no validated end-to-end benefit** |
| Neural KV autoencoder | Code and synthetic tests exist | **Experimental; quality evidence inadequate** |
| Chronos scheduler | Pointer/swap bookkeeping | **Scheduler prototype, not demonstrated generation scheduling** |
| `phantom benchmark` | Prints fixed benchmark rows | **Claim presentation, not benchmarking** |

### Claim audit

The most consequential discrepancies are:

1. **The advertised benchmark suite is not an end-to-end inference benchmark.**  
   `benchmarks/run_all.py` literally constructs throughput samples such as `2.88 + 0.05*sin(i)`, inserts fixed perplexity values, and labels them “measured.” The planner validation similarly embeds target numbers as constants.

2. **The “real audit” script measures Ollama, not PHANTOM.**  
   It posts to `http://127.0.0.1:11434/api/generate`, which is Ollama’s API. That can be a useful external baseline, but it does not validate PHANTOM kernels, layer placement, paging, quantization, or scheduling.

3. **The Rust core declares missing CUDA FFI and missing weight loading.**  
   Its own comments say CUDA context initialization and loading are omitted; its generation loop is pseudocode.

4. **Sparse MLP code is not production-safe yet.**  
   The shown sparse CUDA path allocates and frees intermediates per invocation and creates/destroys a cuBLAS handle in the dense fallback. Either behavior is catastrophic in a token-by-token decoder. It also assumes a reliable active-neuron mask, yet PHANTOM does not establish a trained predictor, accuracy/recall trade-off, or end-to-end numerical parity for that mechanism.

5. **“Layer-transition prediction” is conceptually weak for a normal dense Transformer.**  
   Every token traverses every layer in the same order. Predicting that layer 15 follows layer 14 is not useful information; a static double-buffered schedule already knows this perfectly. A predictor can matter for *conditional units*—experts, blocks, retrieval pages, or sparse neurons—not for fixed sequential layer order.

### Likely per-token timeline today

For a purported 32B Q4 hybrid workload on a 6 GB GPU plus DDR5 system, the intended execution is:

```text
Token t
  GPU-resident early layers
  → device/host synchronization
  → ~10 KB activation transfer to CPU
  → CPU evaluates RAM-resident layers while streaming their weights from DDR5
  → possible host-side sampling
  → Token t+1
```

The activation transfer itself is not the issue: a 10 KB transfer across even a modest PCIe link is microseconds to low tens of microseconds. The problem is the serial CPU-side reading and evaluating of many gigabytes of dense weights, followed by synchronization at each CPU/GPU boundary.

For an actual dense 32B Q4 target with about 14 GB of weights in RAM, an optimistic bandwidth-only time is:

$$
T_{\text{RAM weights}} \approx \frac{14\text{ GB}}{48\text{ GB/s}}=292\text{ ms}.
$$

That implies 3.4 tok/s before CPU kernel inefficiency, attention, synchronization, and sampling. This is broadly consistent with the repository’s claimed 2.88 tok/s number—but that agreement validates a rough physical calculation, **not PHANTOM’s claimed measurement**.

### Strongest ideas in the repository

* Avoiding repeated full-weight PCIe transfers is correct.
* Capacity planning before loading is useful.
* Separating VRAM-resident work from RAM-resident work is preferable to naïvely swapping whole layers across PCIe every token.
* MoE active-parameter accounting is essential.
* KV-cache compression may reclaim capacity at long contexts, but it is not automatically a decode-speed technique.

### Components to stop treating as validated

* Wraith’s claimed 9.9% throughput lift.
* Spectral/DCT quantization’s stated perplexity delta.
* “100% retrieval recall” from neural KV compression.
* The claimed end-to-end throughput figures.
* The claim that 32B dense execution is numerically verified by PHANTOM itself.

## Physical Performance Model

For one final token, use a resource model rather than one universal bandwidth formula:

$$
T_{\text{token}} =
T_{\text{weights}}
+T_{\text{GEMV/GEMM}}
+T_{\text{dequant}}
+T_{\text{attention}}
+T_{\text{KV}}
+T_{\text{PCIe}}
+T_{\text{sync}}
+T_{\text{launch}}
+T_{\text{CPU/runtime}}
+T_{\text{sampling}}.
$$

For a model partitioned across tiers:

$$
T_{\text{weights}}
\gtrsim
\sum_{d\in \{\text{VRAM,RAM,NVMe}\}}
\frac{W_{d,\text{active}}}{B_{d,\text{effective}}}.
$$

This is a lower bound when accesses are serial. It becomes:

$$
T_{\text{weights}} \gtrsim
\max_d
\left(\frac{W_{d,\text{active}}}{B_{d,\text{effective}}}\right)
$$

only if the work and transfers are genuinely overlapped without dependency stalls.

For a dense model with $P$ parameters stored at $q$ bits/parameter:

$$
W \approx P\frac{q}{8} + W_{\text{metadata}}.
$$

At batch size one, decoding usually has low arithmetic intensity:

$$
I_{\text{decode}}
=
\frac{\text{FLOPs}}{\text{weight bytes}+\text{KV bytes}}
\approx \frac{2P_{\text{active}}}{P_{\text{active}}q/8}
=
\frac{16}{q}
\quad \text{FLOP/byte},
$$

before quantization metadata and non-matmul overhead. Q4 gives a rough ceiling near 4 FLOP/byte. This is far below the compute-to-bandwidth ratio of modern GPUs, so single-stream dense decode is usually weight-bandwidth-bound.

For MoE or dynamically sparse models:

$$
W_{\text{active}} =
W_{\text{attention}}
+W_{\text{router}}
+\sum_{e \in \text{selected experts}} W_e.
$$

A 30B-total model with 3B active parameters can behave closer to a 3B model *for per-token work*, provided the runtime avoids loading inactive experts.

For attention at context length $L$:

$$
T_{\text{KV}} \propto
\frac{L \cdot n_{\text{KV heads}}\cdot d_{\text{head}}\cdot b_{\text{KV}}}
{B_{\text{KV tier}}}.
$$

Grouped-query attention and multi-query attention reduce $n_{\text{KV heads}}$. At short contexts, weights dominate. At long contexts, KV reads can dominate—even if the model weights are fully resident.

[FlashAttention](https://www.alphaxiv.org/abs/2205.14135) demonstrates the important general principle: reading fewer bytes can beat algorithms with fewer FLOPs. It tiles attention so that large intermediate attention matrices are not materialized in slow GPU memory, accepting recomputation to reduce I/O. [IO-Aware Attention](https://www.alphaxiv.org/abs/2205.14135v2?page=2)

### When the simple bandwidth equation breaks

`tok/s ≈ bandwidth / bytes-per-token` fails when:

* the device is compute-bound because verification uses multi-token GEMM;
* PCIe copies or CPU/GPU barriers serialize otherwise independent work;
* dequantization is instruction-throughput-limited;
* low occupancy or tiny kernels dominate;
* KV cache has moved the workload from weight-bound to attention-bound;
* sparse operations lose coalescing and become latency-bound;
* page faults or NVMe reads are on the critical path;
* thermal or power limits reduce sustained clocks.

## Existing Techniques

| Technique | Speed Potential | Memory Reduction | Complexity | Risk |
|---|---:|---:|---:|---|
| Full GPU offload + tuned Q4/Q5 kernels | High if model fits | Moderate | Medium | Low |
| Fused dequant + decode GEMV | 1.1–1.5× utilization | None | High | Low |
| CUDA Graphs / persistent allocations | 1.05–1.2× | None | Medium | Low |
| Flash/paged decode attention | Small at short context; high at long context | Workspace reduction | High | Low |
| KV quantization | Mostly capacity; speed only if KV-bound | 2–8× typical target range | Medium | Accuracy risk |
| CPU SIMD kernel tuning | 1.1–1.6× | None | High | Low |
| PCIe weight streaming | Usually harmful for dense decode | Enables fit only | Medium | Very high |
| MoE expert routing + residency | 2–10× model-dependent | Large effective reduction | High | Medium |
| Activation/neuron sparsity | 1.2–5× if predictor is accurate | Effective reduction | Very high | High |
| Speculative decoding | 1.5–4× practical; higher in favorable cases | No target compression | High | Acceptance-dependent |
| Offline pruning/distillation | Potentially large | Large | Very high | Quality risk |
| Dynamic depth / early exit | Potentially large average gain | Effective reduction | Research | Quality/calibration risk |

The strongest evidence for active computation is [PowerInfer](https://www.alphaxiv.org/abs/2312.12456), which exploits skewed neuron-activation locality: hot neurons are preloaded on GPU, cold selected neurons run on CPU, and the system uses neuron-aware operators plus an offline placement policy. [PowerInfer Design](https://www.alphaxiv.org/abs/2312.12456v2?page=2)

But its applicability is not automatic. Its evaluation includes models with ReLU/ReGLU or measured dynamic sparsity, and sparse decoding needs a predictor and irregular kernels good enough to beat dense execution. The paper reports that generic CPU sparse operators did not beat dense computation until sparsity exceeded 87%; specialized operators were needed. [Sparse Operator Limits](https://www.alphaxiv.org/abs/2312.12456v2?page=12)

## Best Known Existing Approaches

| System / method | What it really offers | PHANTOM relevance |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Mature CPU/GPU GGUF runtime, quantized kernels, draft and EAGLE-family support | Baseline to beat, not a library to hand-wave past |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | Strong NVIDIA full-GPU kernels, graph optimization, speculative paths | Best reference when model fits VRAM |
| [ExLlamaV2](https://github.com/turboderp-org/exllamav2) | Low-bit NVIDIA decoding optimized for consumer GPUs | Relevant Q4/Q6 full-GPU baseline |
| [MLC LLM](https://github.com/mlc-ai/mlc-llm) | Compiler-driven edge backends including non-NVIDIA targets | Useful portability reference |
| [vLLM](https://github.com/vllm-project/vllm) / [SGLang](https://github.com/sgl-project/sglang) | Serving-oriented scheduling, paged KV, speculative serving | Valuable architecture reference, less ideal for a single weak GPU |
| [PowerInfer](https://www.alphaxiv.org/abs/2312.12456) | Model-specific activation locality, hot/cold CPU–GPU neuron placement | The closest serious precedent for PHANTOM’s differentiator |
| [EAGLE-3](https://www.alphaxiv.org/abs/2503.01840) | Lossless feature/token-assisted speculative decoding with trained drafting | Highest-leverage direction for dense targets |
| [FlashAttention](https://www.alphaxiv.org/abs/2205.14135) | I/O-aware exact attention kernels | Mandatory at contexts where KV attention matters |

[PowerInfer](https://www.alphaxiv.org/abs/2312.12456) reported up to 11.69× over its llama.cpp baseline, but on a 24 GB RTX 4090 and an 11 GB RTX 2080 Ti system with materially more VRAM than a 4–6 GB GPU; that result cannot be transferred directly to an RTX 3050-class device. Its lower-end system showed smaller gains precisely because less VRAM limited hot-neuron placement. [Evaluation Limits](https://www.alphaxiv.org/abs/2312.12456v2?page=10)

## New Ideas

| Idea | Mechanism / hardware behavior | Expected gain | Why it may fail | Difficulty / required benchmark |
|---|---|---:|---|---|
| 1. Speculation-first tiering | GPU drafts while CPU/RAM target verifies token blocks | 1.5–3× | Draft competes for VRAM or verifier cannot batch | High; measure cycle timeline and accepted final tokens |
| 2. Expert lookahead cache | Predict top-$k$ MoE experts one layer ahead; prefetch only those blocks | 1.1–2× MoE | Prediction misses trigger stalls; experts may be too large | High; routing recall vs stall rate |
| 3. Hot-block residual cache | Keep common quantized MLP blocks in VRAM; stream residual/cold blocks | 1.1–1.5× if locality exists | Dense weights are used every token; no locality exists | High; tensor-block reuse trace |
| 4. Sparse-plus-dense MLP decomposition | Offline $W \approx W_{\text{low-rank}}+W_{\text{block sparse residual}}$; always execute low rank, selectively execute residual | 1.2–2× | Residual may be too large or quality collapses | Research; perplexity plus downstream eval |
| 5. Verification-aware quantization | Use a faster low-bit target representation for speculative verification, exact/full target only on disagreement | Potentially 1.3–2× | “Exact” output becomes difficult; correction may erase gains | Research; distributional equivalence test |
| 6. Token-class adaptive draft depth | Increase draft tree only where confidence predicts long acceptance | 1.1–1.4× over fixed draft length | Predictor overhead or bad calibration | Medium; acceptance-conditioned throughput |
| 7. KV tiered page cache | Keep recent KV plus attention-important pages in VRAM; CPU/NVMe only for provably cold pages | Long-context only | Incorrect eviction damages quality; PCIe dominates | High; long-context recall and decode p95 |
| 8. Residency-aware MoE training | Train/finetune routers to prefer experts already resident without materially changing loss | 1.1–1.5× | Expert imbalance and degraded quality | Research; quality vs cache-hit Pareto curve |
| 9. Cross-layer fused CPU decode | Fuse RMSNorm, quantized matvec, RoPE/activation where dependencies permit; persistent thread pool | 1.1–1.4× | Little gain if DRAM is already saturated | High; hardware counters required |
| 10. Low-rank draft hidden-state predictor | Predict inexpensive intermediate features and use them only to seed a lossless token draft | 1.2–2× | Training burden; draft quality insufficient | Research; acceptance and latency |
| 11. Prompt-repetition/n-gram fast path | Exact prompt lookup proposals before neural drafting | Workload-dependent | No gain in novel prose/reasoning | Low; report hit rate separately |
| 12. Heterogeneous verifier split | GPU verifies attention/hot MLP, CPU verifies cold sparse MLP concurrently | 1.1–1.8× | Fine-grained synchronization can dominate | Very high; Nsight Systems trace |

These are hypotheses, not benchmark claims.

## Top 5 Ideas for PHANTOM

### 1. Build a real lossless speculative decoding pipeline

**Mechanism:** reserve GPU VRAM for a small draft model; run the large target in RAM/CPU or hybrid tiers; draft $k$ tokens, then verify the block with the target in a batch. Use strict target acceptance so final output distribution is preserved.

**Expected benefit:** the only credible route from 5 to 14 tok/s for a dense target without changing hardware.

**Failure modes:** draft memory crowding out target hot weights; draft consumes the same CPU bandwidth as target; target verification does not reuse weights efficiently; acceptance falls on code, reasoning, or high-temperature sampling.

**Benchmark:** report target-only tok/s, draft tok/s, draft time, verification time, accepted final tokens/cycle, acceptance by position, rejected-token rate, target-equivalent tok/s, and TTFT.

[EAGLE-3](https://www.alphaxiv.org/abs/2503.01840) reported 3.0–6.5× speedups in its evaluated settings and a mean accepted length around 5.9 for a 70B target at greedy decoding; those are favorable trained-draft results, not a guarantee for a RAM-tiered consumer machine. [EAGLE-3 Results](https://www.alphaxiv.org/abs/2503.01840v3?page=7)

### 2. Port or integrate a proven PowerInfer-like sparse backend

**Mechanism:** profile a *specific compatible model*, train activation predictors, establish conservative masks, and use a production-grade sparse operator. Place hot neurons/blocks in VRAM and cold active units in RAM/CPU.

**Expected benefit:** potentially transformative for models with real, exploitable activation sparsity; negligible for models whose SwiGLU activations are insufficiently sparse or whose sparse kernel is slower than dense Q4 GEMV.

**Failure mode:** no evidence yet that PHANTOM’s claimed “61.2% routed” masks exist, are accurate, or preserve output quality.

**Benchmark:** token-level predictor recall/precision, PPL, task accuracy, active bytes/token, GPU/CPU time, synchronization time, and dense-vs-sparse crossover by layer.

### 3. Replace the runtime with one execution authority

**Mechanism:** use either a llama.cpp-derived engine or build a unified C++/CUDA/Rust backend with real FFI. Do not run a planner in Python, dispatch generation to Ollama, and call that PHANTOM execution.

**Expected benefit:** first makes all other optimization results interpretable; likely 10–40% opportunity from eliminating allocation, Python overhead, redundant synchronization, and generic Transformers execution.

**Failure mode:** engineering scope.

**Benchmark:** Nsight Systems timeline covering every CPU thread, CUDA stream, allocation, memcpy, kernel, and synchronization event.

### 4. Long-context KV path: GQA-aware paged KV plus quantization

**Mechanism:** start with native paged KV and proven 8-bit or 4-bit KV cache; use Flash-style decode attention. Consider learned KV compression only after quality and latency tests.

**Expected benefit:** little at 1–4K context where weights dominate; potentially substantial at 32K+ when KV reads dominate.

**Failure mode:** an autoencoder adds decoding FLOPs and can become slower than direct low-bit KV reads.

**Benchmark:** context sweep from 1K to 128K, with decode tok/s, attention-kernel time, KV bytes read/token, retrieval accuracy, and perplexity.

### 5. Make residency conditional only where computation is conditional

**Mechanism:** static layer residency for dense blocks; predicted residency only for experts, sparse MLP blocks, and long-context KV pages.

**Expected benefit:** makes prefetch useful rather than decorative.

**Failure mode:** attempting per-token “hot weights” for a dense Transformer cannot help: every dense matrix is required for every token.

**Benchmark:** transfer bytes requested, transfer bytes consumed, prefetch precision, eviction regret, critical-path stall time.

## Proposed PHANTOM v2 Architecture

```text
                         Offline compiler/profiler
             model weights + calibration + kernel autotuning
                                      ↓
Model manifest → Execution planner → Static layer placement
                                      ↓
                 Conditional-work profiler / predictor trainer
                                      ↓
Request
  ↓
Tokenizer + exact prompt-cache / n-gram proposer
  ↓
GPU draft engine ───────────────┐
  ↓ proposed tree/block          │
  └─────────────────────────────┼──→ Block verifier scheduler
                                ↓
          ┌──────────────── Unified execution runtime ────────────────┐
          │ CUDA graphs • persistent allocations • pinned host buffers │
          │ CPU thread affinity • async streams • event-based deps     │
          └────────────────────────────────────────────────────────────┘
                 ↓                          ↓
       GPU: draft, hot experts,       CPU: RAM dense/sparse
       hot target blocks, KV          target blocks, cold experts
                 ↓                          ↓
                 └──────── explicit merge / verifier result ──────────┘
                                      ↓
                         exact accept/reject + sampler
                                      ↓
                     VRAM cache ← RAM cache ← NVMe staging
```

The important design choice: **NVMe is capacity staging, never a normal decode tier.** If required weights must be read from SSD once per final-token verification pass, interactive throughput is gone. Compression may reduce the bytes, but it must be measured including decompression.

## 5 → 14 tok/s Path

Assume the original 5 tok/s means a 200 ms dense target decode, predominantly RAM weight streaming.

### What will not reach 14

| Change | Credible effect | Why it cannot close the gap alone |
|---|---:|---|
| Better static layer prefetch | 0–10% | Dense layer order is deterministic already |
| Fused CPU kernels | 10–40% | DRAM traffic remains nearly unchanged |
| CUDA Graphs | 0–20% | Kernel/launch overhead is not 64% of a 200 ms decode |
| Q4 to Q3 | workload-specific | Could reduce bytes, but may lose quality or slow dequant |
| KV compression at short context | near zero | Weights, not KV, dominate |
| PCIe streaming | negative | PCIe is slower than VRAM and often slower than host DRAM locality |

### Resource-accounted speculative path

Let:

* baseline target time $T_1=200$ ms;
* draft rate = 50 tok/s, so an 8-token draft costs $T_D=160$ ms;
* independent per-token acceptance probability $p=0.70$;
* speculative block size $k=8$.

The expected final tokens per cycle, including the target correction/bonus behavior, is approximately:

$$
E[N] \approx 1+p+p^2+\dots+p^8
= \frac{1-p^9}{1-p}
\approx 3.20.
$$

If draft and target verification serialize, even an unusually efficient 210 ms target verification gives:

$$
\text{tok/s} \approx \frac{3.20}{0.160+0.210}=8.6.
$$

So **“50 tok/s draft + 70% acceptance” does not automatically equal 14 tok/s.**

To hit 14 tok/s:

$$
T_{\text{cycle}} \leq \frac{3.20}{14}=229\text{ ms}.
$$

That requires one of these:

1. **Overlap:** draft runs largely concurrently with target verification on genuinely separate bottleneck resources. Then cycle time can approach $\max(160, 210)=210$ ms, yielding roughly 15.2 target-equivalent tok/s.
2. **Faster draft:** an 8-token draft in $\leq 100$ ms plus a 125 ms verifier cycle gives 14.2 tok/s, but now verification itself needs a 1.6× improvement.
3. **Higher accepted length:** at $E[N]\approx4.2$, a 300 ms cycle produces 14 tok/s. That needs materially higher acceptance, not merely a 70% first-token rate.

Therefore the honest path is:

```text
5.0 tok/s  Measured target-only baseline
5.5–6.5    Estimated: real execution consolidation + tuned CPU/GPU kernels
6–8        Estimated: lower-bit format only if accuracy and dequant benchmark pass
8–11       Estimated: ordinary draft speculation, serial resource use
12–16      Estimated: high-acceptance trained speculation with overlapped GPU draft
            and RAM/CPU verifier, or model-specific sparse active-weight execution
14+        Plausible only after confirming target verification remains near one
            target pass per accepted block and draft does not contend for the
            target’s bottleneck
```

The path is **not additive**. A “1.2× kernel gain” and a “2× speculation gain” multiply only if they reduce independent bottlenecks. If both consume the same DRAM bandwidth, they do not.

## Hardware-Specific Strategy

| Hardware tier | Best target strategy | 14 tok/s verdict for a dense 5 tok/s workload |
|---|---|---|
| 4 GB GPU | Put only draft/KV or a small target fully in VRAM; CPU target otherwise | **Usually no**; use MoE/sparsity or accept lower speed |
| 6 GB GPU | Reserve 0.5–1.5 GB for draft; target hot blocks/KV in remainder; CPU/RAM verifier | **Possible** with high acceptance and real overlap |
| 8 GB GPU | More target hot blocks plus Q4 draft; use paged KV | **Plausible** for 7–14B and sparse/MoE targets |
| 12 GB GPU | Prefer entire 7–14B target in VRAM; optimized native runtime likely beats hybrid | **Often yes**, but PHANTOM has less unique value |
| RTX 3050 4 GB | Bandwidth and VRAM constrained; avoid target PCIe swapping | **Dense 14 tok/s unlikely** |
| RTX 3050 6 GB / RTX 4050 Laptop 6 GB | Draft-target heterogeneous split is viable | **Conditional** |
| RTX 4060 Laptop 8 GB | Full-GPU 7–8B quantized baseline first; speculate only if it wins | **Likely for mid-size targets** |
| GTX 1650 | No tensor cores and relatively weak bandwidth | **Unlikely for dense oversized targets** |
| AMD equivalent | Use ROCm/Vulkan/MLC or llama.cpp paths; do not promise CUDA behavior transfers | **Hardware- and backend-dependent** |
| Apple Silicon low-memory | Unified memory removes PCIe boundary, but memory bandwidth is shared | **Possible for compact models; oversized dense models remain bandwidth-bound** |
| Intel iGPU / Ryzen APU | Unified memory helps communication, but shared DDR bandwidth is low | **Usually no for large dense targets** |
| CPU-only DDR4 | AVX2/AVX-512, NUMA pinning, Q4/Q5 kernels | **No unless active work is radically reduced** |
| CPU-only DDR5 | Same, with more bandwidth | **Possible only with MoE/sparsity/speculation; not dense raw decode alone** |

## Impossible Zone

PHANTOM cannot realistically accelerate these cases past physical limits:

1. **A dense model whose required Q4 weights are streamed from NVMe every target pass.**  
   A 14 GB stream at 2 GB/s takes about 7 seconds before compute. Async I/O hides latency only if another stage is at least as long; it does not create 14 GB/s.

2. **A dense model where every weight is active and already saturates RAM bandwidth.**  
   Without reducing precision, number of target passes, or active weights, 3× speedup is impossible.

3. **GPU offload that requires repeated full-weight PCIe swapping.**  
   A PCIe Gen3/Gen4 link is not a substitute for VRAM. Moving tens of gigabytes of weights each token turns the interconnect into the limiting memory tier.

4. **Sparse execution without sufficient structured, predictable sparsity.**  
   Skipping 40% of irregular rows can be slower than dense Tensor Core/GEMV code. Sparsity only helps if it saves more traffic and compute than indices, divergence, gathers, and synchronization cost.

5. **“Lossless” learned compression that requires expensive reconstruction of most weights every token.**  
   If decode/decompression costs more than the bandwidth saved, it loses. Compression ratio is not a speed result.

6. **Speculation with a slow draft, low acceptance, or serial verifier contention.**  
   Rejected drafts are not output tokens. Reported draft rate is not final-token rate.

## Experimental Roadmap

### Phase 1 — Establish reality cheaply

| Hypothesis | Setup | Measurement | Success interpretation |
|---|---|---|---|
| PHANTOM currently executes its own target path | Add a runtime provenance flag and kernel counters | Every generated token must identify backend, kernels, and bytes moved | Remove all Ollama-derived PHANTOM claims |
| Decode bottleneck is known | 3B, 8B, 14B, 32B; Q8–Q3; 1K, 8K, 32K contexts | Nsight Systems + CPU perf counters + DRAM/PCIe counters | Produce per-token critical-path breakdown |
| Q3 is actually faster than Q4 | Same model and prompt suite | tok/s, PPL, task score, dequant time | Adopt only if quality threshold and sustained gain pass |
| GPU split helps | Sweep all layer splits | TTFT, tok/s, PCIe bytes, GPU/CPU utilization | Identify cliffs rather than assuming monotonic offload |

**Protocol:** fixed model revision and checksum; 100 generated tokens after warm-up; 20 prompts across chat, code, math, and summarization; report p50/p95; isolate prefill from decode; record total accepted final tokens only.

### Phase 2 — Kernel experiments

| Hypothesis | Setup | Measurement | Expected result |
|---|---|---|---|
| Current kernels leave utilization unused | Compare llama.cpp / CUTLASS / cuBLASLt / custom Q4 GEMV | kernel time, bandwidth achieved, occupancy | Identify 10%+ real kernel headroom |
| Fusion matters | Fuse dequant + matvec + bias/RMSNorm where legal | HBM/VRAM bytes and token latency | Win only if memory traffic falls |
| KV becomes dominant at long context | Flash/paged decode attention versus baseline | attention time and KV bytes/token | Context-dependent crossover point |
| Sparse MLP has a crossover | Dense vs specialized sparse kernels across actual masks | end-to-end tok/s and PPL | Reject sparsity if it loses at real mask density |

### Phase 3 — Runtime architecture experiments

| Hypothesis | Setup | Measurement | Interpretation |
|---|---|---|---|
| Speculation reaches 14 | 0.5–3B draft, 8–32B target, $k=2,4,8,12$ | accepted length, cycle time, final tok/s | Go forward only if final-token throughput rises |
| Heterogeneous overlap is real | Draft on GPU, verifier on CPU/RAM; then reverse | stream timelines, overlap fraction, contention | Require >70% useful overlap |
| Expert prefetch helps | MoE target with routing trace | expert-cache hit rate and stall time | Require net speedup after misses |
| Dynamic residency helps only conditional work | Dense vs MoE vs sparse MLP traces | bytes prefetched but unused | Disable for dense sequential layers |

### Phase 4 — Novel research

| Hypothesis | Setup | Measurement | Go/no-go |
|---|---|---|---|
| Residency-aware routing preserves quality | Fine-tune an MoE router with cache-locality regularization | task quality, expert entropy, cache hits | Continue only if quality loss is negligible |
| Low-rank-plus-residual execution helps | Offline factorize selected MLPs and calibrate residual selection | PPL, downstream scores, active bytes | Require a Pareto improvement |
| Learned sparse blocks can replace neuron masks | Block-level predictor with conservative fallback | false-negative impact and speed | Strict accuracy gate |
| Hidden-state draft improves acceptance | Train a small draft on target traces | accepted length and actual wall time | Must beat conventional draft at equal VRAM |

## THE PHANTOM BREAKTHROUGH

**The single most promising direction is a real heterogeneous, lossless speculative-verification runtime that combines a GPU-resident high-acceptance draft with a RAM-resident large target verifier—and uses conditional residency only for MoE experts or demonstrably sparse blocks.**

1. **Core idea**  
   Replace “make one dense token faster” with “make one expensive target pass certify multiple final tokens.” Use the GPU for drafting, CPU/RAM for target verification, and overlap them.

2. **Why it works**  
   Dense autoregressive decode rereads the same target weights each token. Block verification increases arithmetic intensity and can amortize a target weight pass over several accepted final tokens.

3. **Why existing runtimes do not completely solve it**  
   General runtimes support speculative decoding, but a low-VRAM hybrid target has unusual constraints: the draft competes for scarce VRAM, the verifier may be DRAM-bound, CPU/GPU overlap may be defeated by synchronization, and target-specific draft training is needed for high acceptance.

4. **Expected bottleneck**  
   RAM bandwidth during verification, followed by draft/target resource contention and low acceptance on difficult distributions.

5. **Prototype design**  
   Start from a proven inference core rather than the current PHANTOM skeleton. Add:
   * a 0.5–1.5B Q4 GPU draft;
   * target block verification at $k=4$ and $k=8$;
   * pinned event-driven host/device buffers;
   * static dense-layer placement;
   * optional MoE expert cache and routing prefetch;
   * strict exact acceptance accounting.

6. **First experiment**  
   On one 6 GB laptop GPU plus DDR5 machine, benchmark one 14B dense model and one MoE model against llama.cpp using the exact same quantization, context, sampler, prompt suite, and output length. Measure whether $k=4$ speculation produces at least 2 accepted final tokens per cycle and whether useful GPU/CPU overlap exceeds 60%.

7. **Success threshold**  
   A reproducible **$\geq1.8\times$ final-token decode speedup** over the same target-only backend, with unchanged target output distribution under strict speculative acceptance, and with full p50/p95, energy, and thermal reporting.

8. **Failure threshold**  
   Stop pursuing the design for a model/hardware pair if accepted final tokens per cycle are below 1.7, draft plus verification serializes, or target-equivalent throughput is less than 1.2× baseline after all overhead.

9. **Publishable research?**  
   **Yes**, if it demonstrates a new measured scheduling/residency policy that materially improves lossless speculative decoding under a strict low-VRAM, host-RAM-tiered target constraint—and compares fairly against llama.cpp, [PowerInfer](https://www.alphaxiv.org/abs/2312.12456), and mature speculative baselines.

10. **Implementable by a small open-source team?**  
   **Yes for the speculative runtime and benchmark suite; probably not for a general-purpose sparse-neuron compiler plus cross-vendor production backend at the same time.** Start by making one backend real, one hardware target measurable, and one 5-to-14 claim either reproducible or decisively rejected.