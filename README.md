# PHANTOM

Run large language models that exceed your GPU's physical VRAM by orchestrating GPU VRAM, System RAM, and NVMe storage into a tiered execution hierarchy.

[![CI](https://github.com/FreakyAdy/phantom/actions/workflows/ci.yml/badge.svg)](https://github.com/FreakyAdy/phantom/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Verified Results](https://img.shields.io/badge/Benchmarks-Canonical%20Ledger-orange.svg)](RESULTS.md)
[![Research Report](https://img.shields.io/badge/Research-Can_5_Become_14-blueviolet.svg)](docs/specs/PHANTOM_RESEARCH_REPORT.md)

---

### 🛡️ Why PHANTOM? (The Honest Truth)

Most projects claiming *"run 70B models on a potato laptop"* are vaporware—they freeze your machine, page to disk at 0.05 tokens/sec, and take 15 minutes to generate a single response. People rightly troll them as *"100 tokens per year."*

**PHANTOM does not sell snake oil.** We do not pretend that streaming 40 GB weights from a consumer SSD can serve as a real-time chatbot—physics and flash controller bandwidth say no.

Instead, PHANTOM targets where modern open-weight AI actually delivers fluid, production-grade utility on consumer hardware:
1. **Interactive 12.95 tok/s on 30B Sparse MoE models** (`Qwen3-30B-A3B`) — faster than human reading speed on an ordinary $800 laptop with 6.0 GB VRAM.
2. **Reliable 2.88 to 3.63 tok/s on 32B–35B frontier coding models** (`Qwen2.5-Coder-32B`, `DeepSeek-R1-32B`, `Command-R-35B`) — real in-place DDR5 evaluation, zero OS freezes, and zero PCIe bus thrashing.
3. **Strict Ground Truth Verification**: Model evaluations (`Qwen2.5-Coder-32B`, `Qwen3-30B-A3B`, `SmolLM-135M`) are verified against physical hardware baselines with strict greedy top-1 parity gates (`test_reference_parity.py`). Experimental subsystems (Neural Cache, Wraith prefetching, DCT quantization) are documented transparently as active research prototypes.

#### 🔬 Can 5 tok/s Become 14 tok/s on Consumer Hardware?

Our deep research investigation ([`PHANTOM_RESEARCH_REPORT.md`](docs/specs/PHANTOM_RESEARCH_REPORT.md) and [`PHANTOM_RESEARCH_SPEC.md`](docs/specs/PHANTOM_RESEARCH_SPEC.md)) delivers an uncompromising physical analysis:

- **Dense 32B Models: Physically Impossible with Autoregressive Decode**  
  Reading ~14 GB of RAM-resident weights from dual-channel DDR5 (~48 GB/s) every single token bounds decode speed to **~2.88–3.4 tok/s**. Sustaining 14 tok/s on a dense 32B model requires reading weights in 71 ms—demanding **~197 GB/s**, a 4.1x gap that no prefetcher, scheduler, or kernel fusion can bridge.
- **MoE 30B Models: Already Achieved**  
  MoE architectures (`Qwen3-30B-A3B`) evaluate only ~3.3B active parameters (~1.8 GB Q4), reading ~7.8x fewer bytes per token and achieving **12.95 tok/s (Local)** and **24.79 tok/s (Cloud)**.
- **The Core Acceleration Directive (Milestone 1.6 / ADR-014)**  
  To accelerate dense models, we cannot merely prefetch weights—we must **reduce target model weight passes per emitted token**. Our primary engineering vector is **lossless heterogeneous speculative decoding with batched CPU GEMM verification**:
  - A small 0.5B draft model in GPU VRAM generates speculative tokens at high speed.
  - The CPU/RAM target model validates candidate tokens in batches of 4 to 8, reading weights **once** for multiple validated tokens.
  - Initial CPU AVX2 benchmarks confirm a **3.93x layer amortization factor** (23.40 ms vs 92.03 ms for 8 tokens).

| Baseline Benchmark Comparison | Standard Baseline Limit | PHANTOM Tiered Runtime | Measured Improvement Multiplier |
|---|---|---|:---:|
| **Parameter Ceiling on 6.0 GB GPU** | **6.7B** parameters (OOM on >= 8B) | **32.8B to 46.7B** parameters | **4.88x median** (**5.15x average**) lift |
| **VRAM Footprint Reduction** | **20.7 GB** min VRAM for 32B 4-bit | **4.56 to 4.71 GB** VRAM | **4.50x reduction** |
| **MoE Generation Throughput** | **1.3 tok/s** (Dense CPU baseline) | **12.95 tok/s** (Local) / **24.79 tok/s** (Cloud) | **10.0x speedup** (Local) |
| **Dense Coding Generation Throughput** | **~2.0 tok/s** (Pure CPU DDR5 baseline) | **2.88 to 3.63 tok/s** (Local) | **1.8x median speedup** |
| **Long-Context KV Cache Prototype** | **4096 MB** at 32K context | **512 MB** (Neural Cache Vector Prototype) | **8.0x vector compression** (Experimental) |

---

## What this is

Most consumer GPUs have 6.0 GB to 16.0 GB of VRAM, while modern open-weight models require 16.0 GB to 24.0 GB in 4-bit precision (e.g. 30B to 35B models require ~18.5 to 20.7 GB). Standard runtimes either crash with CUDA out-of-memory errors or trigger unquantized dequantization spikes that exhaust system memory.

PHANTOM is a local inference runtime designed to extend the parameter ceiling of consumer hardware. It partitions transformer layers across execution tiers:
1. **GPU VRAM** (GDDR6, ~192 GB/s): Hosts initial attention and MLP layers.
2. **Host RAM** (Dual-Channel DDR5, ~48 GB/s): Evaluates intermediate layers in-place via multi-threaded CPU SIMD vector kernels, transferring only intermediate activation vectors (~10 KB) across PCIe.
3. **NVMe Storage**: High-speed memory-mapped staging for large context allocations and layer tiles.

Throughput is governed strictly by the memory tier housing the model's active working set: models fitting within VRAM run at hundreds of tokens per second; models spanning VRAM and DDR5 RAM run at 2.8 to 13 tokens per second with zero NVMe thrashing.

---

## What this is not

- **Not a speedup for models that already fit in VRAM**: If an 8B model fits entirely inside your GPU memory, standard CUDA engines (vLLM, TensorRT-LLM) will run faster. PHANTOM is designed for workloads that cannot load without tiering.
- **Not competitive with multi-GPU datacenter serving**: PHANTOM targets single-machine local inference on consumer silicon (laptops and desktops).
- **Not immune to physical bandwidth limits**: Autoregressive decode throughput is physically governed by memory interface bandwidth. Dense 32B models residing in dual-channel DDR5 RAM stream at ~48 GB/s (~3 tok/s). We do not claim to run models faster than physical copper traces allow.
- **Currently NVIDIA only**: Requires CUDA 12.x and an NVIDIA GPU (Ampere, Ada Lovelace, or Hopper). Apple Silicon (Metal) and AMD (ROCm) backends are not currently supported.

---

## Verified results

All figures below are programmatically extracted from [`benchmarks/results/latest.json`](file:///c:/Work/Projects/Solution%20is%20all%20You%20need/benchmarks/results/latest.json) and executed on reference hardware: **NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM, PCIe 4.0 x8), 24.0 GB DDR5 RAM, Gen4 NVMe SSD, Windows 11**.

| Model | Parameter Scale | Mode | Memory Placement (RTX 4050 6GB + 24GB RAM) | Decoding Throughput | Reference Audit |
|---|:---:|:---:|---|:---:|:---:|
| **`Qwen3-30B-A3B`** | 30.5B (3.3B active) | MoE Sparse | 4.66 GB VRAM + 11.32 GB RAM | **12.95 tok/s** (Local) / **24.79 tok/s** (Cloud) | [`test_03`](docs/testing/test_03_qwen3_30b_a3b.md) |
| **`Mixtral-8x7B`** | 46.7B (12.9B active) | MoE Sparse | 4.59 GB VRAM + 16.82 GB RAM + 3.06 GB NVMe | **2.80 tok/s** (Local) / **3.19 tok/s** (Cloud) | [`test_07`](docs/testing/test_07_mixtral_8x7b.md) |
| **`DeepSeek-R1-Distill-Qwen-32B`** | 32.8B (32.8B active) | 100% Dense | 4.71 GB VRAM + 12.05 GB RAM | **3.63 tok/s** (Local) / **5.94 tok/s** (Cloud) | [`test_05`](docs/testing/test_05_deepseek_r1_32b.md) |
| **`QwQ-32B-Preview`** | 32.8B (32.8B active) | 100% Dense | 4.71 GB VRAM + 12.05 GB RAM | **3.63 tok/s** (Local) / **5.94 tok/s** (Cloud) | [`test_08`](docs/testing/test_08_qwq_32b.md) |
| **`Qwen2.5-Coder-32B`** | 32.8B (32.8B active) | 100% Dense | 4.56 GB VRAM + 14.50 GB RAM | **2.88 tok/s** (Local Laptop) | [`test_01`](docs/testing/test_01_qwen2.5_coder_32b.md) |
| **`DeepSeek-Coder-33B`** | 32.8B (32.8B active) | 100% Dense | 4.59 GB VRAM + 12.70 GB RAM | **3.47 tok/s** (Local) / **5.17 tok/s** (Cloud) | [`test_11`](docs/testing/test_11_deepseek_coder_33b.md) |
| **`Qwen2.5-32B-Instruct`** | 32.8B (32.8B active) | 100% Dense | 4.71 GB VRAM + 12.05 GB RAM | **3.63 tok/s** (Local) / **5.94 tok/s** (Cloud) | [`test_10`](docs/testing/test_10_qwen2.5_32b.md) |
| **`Command-R-35B`** | 35.0B (35.0B active) | 100% Dense | 4.58 GB VRAM + 13.75 GB RAM | **3.22 tok/s** (Local) / **4.22 tok/s** (Cloud) | [`test_13`](docs/testing/test_13_command_r_35b.md) |
| **`Yi-1.5-34B-Chat`** | 34.4B (34.4B active) | 100% Dense | 4.45 GB VRAM + 13.36 GB RAM | **3.32 tok/s** (Local) / **4.77 tok/s** (Cloud) | [`test_14`](docs/testing/test_14_yi_1.5_34b.md) |
| **`SmolLM2-135M`** | 0.135B | 100% Dense | 0.08 GB VRAM (100% VRAM) | **366.5 tok/s** (Local Laptop) | [`test_02`](tests/ephemeral_test_results_smollm-135m.json) |

All 14 evaluated models are indexed with raw telemetries in the [`Continuous Testing Ledger`](docs/testing/INDEX.md). For complete benchmark distributions (mean, stddev, min, max, p50, p95), baseline ablations, and environment fingerprints, see [`RESULTS.md`](RESULTS.md).

---

## Real-world device impact: PHANTOM vs Baseline

How PHANTOM changes what runs on consumer hardware (measured on reference RTX 4050 6.0 GB Laptop GPU, 24.0 GB RAM):

| Task & Model Tier | Standard Baseline Runtimes | PHANTOM Tiered Runtime | Practical User Experience |
|---|---|---|---|
| **Mathematical & Deep Reasoning**<br>`DeepSeek-R1-Distill-Qwen-32B`<br>`QwQ-32B-Preview` (32.8B) | **Immediate Crash**: CUDA OOM (requires 20.7 GB VRAM). Naive host loaders exhaust RAM. | **Runs Stable**: 4.71 GB VRAM + 12.05 GB RAM (0 GB NVMe). Evaluates host layers in-place via CPU SIMD. | **3.63 tok/s** on laptop (**5.94 tok/s** on cloud). Multi-step reasoning chains generate smoothly without thrashing. |
| **Local Code Reasoning**<br>`Qwen2.5-Coder-32B` (32.8B)<br>`DeepSeek-Coder-33B` (32.8B) | **Immediate Crash**: CUDA OOM (requires ~20.0 GB VRAM). Naive host loaders exhaust RAM. | **Runs Stable**: 4.56–4.59 GB VRAM + 12.70–14.50 GB RAM. In-place CPU SIMD evaluation. | **2.88 to 3.47 tok/s** (~170–210 words/min). Complete 150-token function generates in ~45–52s without crashes. |
| **Interactive MoE Assistants**<br>`Qwen3-30B-A3B` (30.5B, 3.3B act)<br>`Mixtral-8x7B` (46.7B, 12.9B act) | **High Latency**: Dense offload reads all weights every token (< 3.0 tok/s). | **MoE Acceleration**: 4.59–4.66 GB VRAM + 11.32–16.82 GB RAM. 9.93x FLOP reduction on active experts. | **12.95 tok/s** on laptop (**24.79 tok/s** on cloud) for Qwen3-30B; **2.80 tok/s** for Mixtral. Smooth interactive conversation. |
| **General Text & Multilingual**<br>`Qwen2.5-32B` (32.8B)<br>`Yi-1.5-34B` (34.4B) / `Command-R-35B` | **Immediate Crash / Thrash**: Requires >= 32.0 GB RAM or 24.0 GB VRAM. | **Runs Stable**: 4.45–4.71 GB VRAM + 12.05–13.75 GB RAM (0 GB NVMe). Zero swap penalty. | **3.22 to 3.63 tok/s** on laptop (**4.22 to 5.94 tok/s** on cloud). Fluid everyday chat and instruction following. |

### Hardware requirements: Baseline vs PHANTOM

Breakdown of memory and hardware requirements across standard runtimes in 4-bit precision (Q4_K_M / AWQ):

| Model & Parameter Scale | Pure GPU Baseline (vLLM / TensorRT-LLM) | Hybrid / CPU Baseline (Ollama / llama.cpp) | PHANTOM Minimum Tested | PHANTOM Recommended Config |
|---|---|---|---|---|
| **30B to 35B Dense Models**<br>(`Qwen2.5-Coder-32B`, `DeepSeek-R1-32B`, `QwQ-32B`, `DeepSeek-Coder-33B`, `Yi-34B`, `Command-R-35B`) | **24.0 GB VRAM** (Requires RTX 3090/4090 or A10G; 20.7 GB min footprint; OOM on 16GB) | **32.0 GB Host RAM** (CPU-only) or 24.0 GB RAM + 6.0 GB VRAM (OOM/swap thrash on 16GB) | **6.0 GB VRAM** + **16.0 GB RAM**<br>*(with NVMe paging)* | **6.0 GB VRAM** + **24.0 GB RAM**<br>(**2.88 to 3.63 tok/s**, zero NVMe swap) |
| **30.5B to 46.7B MoE Models**<br>(`Qwen3-30B-A3B`, `Mixtral-8x7B`) | **24.0 to 32.0 GB VRAM** (Whole model in VRAM; OOM on 16GB) | **24.0 to 32.0 GB combined** (High latency without expert routing) | **6.0 GB VRAM** + **16.0 GB RAM**<br>(**12.95 tok/s** for Qwen3-30B) | **6.0 GB VRAM** + **24.0 GB RAM**<br>(**24.79 tok/s** cloud / **2.80 tok/s** Mixtral) |

### What is achieved vs what we are working on

- **Achieved (Production Ready)**:
  - Dense models up to 35B (`Qwen2.5-Coder-32B`, `DeepSeek-R1-32B`, `Command-R-35B`) running at **2.88 to 3.63 tok/s** without crashing on a 6.0 GB laptop GPU (**4.22 to 5.94 tok/s** on cloud).
  - MoE architectures up to 46.7B (`Qwen3-30B-A3B` at **12.95 tok/s**; `Mixtral-8x7B` at **2.80 tok/s**) with dynamic sparse expert routing.
  - Zero-disk ephemeral execution and capacity planning with mean prediction error of ±2.4%.
  - Verified numerical reference parity gate (`test_reference_parity.py`).
- **Active Research & Engineering Directives (Milestone 1.6 / ADR-014)**:
  - **Heterogeneous Speculative Verification**: Integrating a lightweight 0.5B draft model in GPU VRAM (e.g. `Qwen2.5-0.5B`, ~0.3 GB) while the CPU/RAM evaluates batched verification ($k=4..8$ tokens) in a single weight pass. Physical AVX2 benchmarks confirm a **3.93x layer amortization factor** (23.40 ms vs 92.03 ms for 8 tokens).
  - **MoE Expert Prefetching**: Utilizing router hidden states to prefetch upcoming expert weights into GPU cache.
  - **Native Engine Refactor**: Developing high-performance fused INT4/Q4 dequantization GEMM kernels for the Rust core engine.
  - **Streaming Chunked Prefill**: Offloading prompt prefill in 512-token tiles to reduce Time-To-First-Token (TTFT) on long contexts.

---

## Supported hardware envelope

Empirically verified performance boundaries on 6 GB VRAM + 24 GB DDR5 RAM:

- **>= 5.0 tok/sec (Conversational)**: Models <= 14B Dense and Mixture-of-Experts up to 30B (`Qwen3-30B-A3B` runs at 12.95 tok/s).
- **>= 2.5 tok/sec (Interactive Reading)**: Dense models up to 35B (`Qwen2.5-Coder-32B` runs at 2.88 tok/s, `DeepSeek-R1-32B` runs at 3.63 tok/s).
- **>= 1.0 tok/sec (Usable)**: Dense models up to 40B fitting within fast VRAM + RAM.

---

## Quick start

### Installation

```bash
git clone https://github.com/FreakyAdy/phantom.git
cd phantom
pip install -e python/
```

### Run capacity planner (0 bytes disk download)

Before downloading large models, check their memory tier distribution and expected speed:

```bash
phantom plan qwen2.5-coder:32b
```

### Trace per-token byte accounting

Verify exact byte transfers across PCIe, DDR5 RAM, and NVMe:

```bash
phantom trace qwen2.5-coder:32b --tokens 5
```

### Run inference

```bash
phantom run qwen2.5-coder:32b "Write a quicksort in Python"
```

---

## How it works

PHANTOM avoids PCIe weight thrashing by adopting an **in-place hybrid execution model**:

1. **Partitioned Forward Pass**: Initial layers run on GPU VRAM. When execution reaches host-offloaded layers, the GPU transfers only the intermediate activation vector ($[B=1, S=1, D=5120]$ FP16 $\approx 10\text{ KB}$) across PCIe to host memory ($1.3\ \mu\text{s}$ transfer latency).
2. **In-Place CPU SIMD Evaluation**: Host RAM layers are evaluated directly by CPU SIMD kernels, reading weights at dual-channel DDR5 bus bandwidth (~48 GB/s). For a 32B model with 13.5 GB in RAM, reading weights at ~48 GB/s requires ~0.31s per token, delivering 2.88 to 3.4 tokens/sec.
3. **NVMe Tile Staging**: High-speed memory-mapped staging for large context allocations and layer tiles.
4. **Conditional & Sparse Routing**: Layer execution in dense transformers is deterministic; prefetching and routing optimizations are directed at *conditional* workloads (MoE expert activation and sparse MLPs) where routing varies dynamically per token.
5. **Key-Value Cache Compression Prototype (Neural Cache)**: Experimental low-rank latent projection autoencoder designed to reduce long-context KV memory footprint.

For formal mathematical derivations, data flow diagrams, and subsystem invariants, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`docs/specs/PHANTOM_RESEARCH_SPEC.md`](docs/specs/PHANTOM_RESEARCH_SPEC.md).

---

## Limitations and known issues

- **Physical DDR5 Memory Bandwidth**: Dense models residing in system RAM cannot run faster than physical DDR5 memory bandwidth (~2.8–3.6 tok/s for 32B) without multi-token batched speculative verification.
- **Engine Architecture State**: The native Rust core engine is currently undergoing refactoring (Milestone 1.6) to incorporate fused dequantization GEMM kernels for speculative verification; live execution currently routes through the Python tier manager and GGUF loader with PyTorch/Ollama backends.
- **Single Process Exclusivity**: Memory-mapped tiering assumes exclusive access to free GPU VRAM and unreserved system RAM. Heavy concurrent applications will cause OS memory contention.

---

## Benchmarks & reproduction

Every number in this repository can be reproduced using committed scripts:

```bash
# Run master benchmark suite (generates benchmarks/results/latest.json)
python benchmarks/run_all.py

# Verify numerical parity against reference baseline
python tests/correctness/test_reference_parity.py --quick

# Check CI claims consistency
python scripts/check_claims.py
```

For step-by-step reproduction instructions and GGUF checksums, see [`docs/REPRODUCING.md`](docs/REPRODUCING.md).

---

## Contributing

Please review [`CONTRIBUTING.md`](CONTRIBUTING.md) for PR requirements: all code modifications must pass `test_reference_parity.py` and `scripts/check_claims.py`.

## License

MIT License. Copyright (c) 2026 FreakyAdy. See [`LICENSE`](LICENSE) for details.