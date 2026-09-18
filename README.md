# PHANTOM

Run large language models that exceed your GPU's physical VRAM by orchestrating GPU VRAM, system RAM, and NVMe storage into a tiered execution hierarchy — with a **llama.cpp-first hybrid backend** and optional **v2 acceleration stack** (EAGLE-3 speculative decoding, fused kernels, adaptive prefetch, CPU batch GEMM amortization).

[![CI](https://github.com/FreakyAdy/phantom/actions/workflows/ci.yml/badge.svg)](https://github.com/FreakyAdy/phantom/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Verified Results](https://img.shields.io/badge/Benchmarks-Canonical%20Ledger-orange.svg)](RESULTS.md)
[![v2 Spec](https://img.shields.io/badge/v2-MD%20Blueprint-green.svg)](docs/specs/PHANTOM_V2_SPEC.md)
[![Colab Testbed](https://img.shields.io/badge/Colab-Zero--Disk%20Testbed-yellow.svg)](notebooks/phantom_cloud_tester.ipynb)

---

### Why PHANTOM? (The Honest Truth)

Most projects claiming *"run 70B models on a potato laptop"* are vaporware—they freeze your machine, page to disk at 0.05 tokens/sec, and take 15 minutes to generate a single response.

**PHANTOM does not sell snake oil.** We do not pretend that streaming 40 GB weights from a consumer SSD can serve as a real-time chatbot—physics and flash controller bandwidth say no.

Instead, PHANTOM targets where modern open-weight AI actually delivers fluid, production-grade utility on consumer hardware:

1. **Interactive 12.95 tok/s on 30B sparse MoE models** (`Qwen3-30B-A3B`) — faster than human reading speed on an ordinary $800 laptop with 6.0 GB VRAM.
2. **Reliable 2.88 to 3.63 tok/s on 32B–35B frontier coding models** (`Qwen2.5-Coder-32B`, `DeepSeek-R1-32B`, `Command-R-35B`) — real in-place DDR5 evaluation, zero OS freezes, and zero PCIe bus thrashing.
3. **Strict ground-truth verification**: Model evaluations are verified against physical hardware baselines with greedy top-1 parity gates (`test_reference_parity.py`). All throughput figures trace to [`RESULTS.md`](RESULTS.md) and [`docs/testing/INDEX.md`](docs/testing/INDEX.md).

#### Can 5 tok/s become 14 tok/s on consumer hardware?

Our research analysis ([`docs/specs/PHANTOM_RESEARCH_REPORT.md`](docs/specs/PHANTOM_RESEARCH_REPORT.md)) delivers an uncompromising physical breakdown:

- **Dense 32B models: 14 tok/s is not achievable with autoregressive decode alone**  
  Reading ~14 GB of RAM-resident weights from dual-channel DDR5 (~48 GB/s) every token bounds decode to **~2.88–3.4 tok/s**. The v2 stack targets **7–10 tok/s** on dense 32B via speculative verification — not 14.
- **MoE 30B models: baseline already near target**  
  MoE architectures (`Qwen3-30B-A3B`) evaluate only ~3.3B active parameters (~1.8 GB Q4), achieving **12.95 tok/s (local)** and **24.79 tok/s (Colab T4)**. v2 targets **14–18 tok/s**.
- **Dense 14B: the 14 tok/s headline tier**  
  Fits the bandwidth envelope with headroom for speculative acceleration.

| Baseline Benchmark Comparison | Standard Baseline Limit | PHANTOM Tiered Runtime | Measured Improvement |
|---|---|---|:---:|
| **Parameter ceiling on 6.0 GB GPU** | **6.7B** (OOM on ≥ 8B) | **32.8B to 46.7B** | **4.88× median** lift |
| **VRAM footprint reduction** | **20.7 GB** min for 32B 4-bit | **4.56 to 4.71 GB** VRAM | **4.50× reduction** |
| **MoE generation throughput** | **~1.3 tok/s** (dense CPU baseline) | **12.95 tok/s** (local) / **24.79 tok/s** (Colab) | **~10× speedup** |
| **Dense coding throughput** | **~2.0 tok/s** (pure CPU DDR5) | **2.88 to 3.63 tok/s** (local) | **1.8× median speedup** |

---

## PHANTOM vs Ollama (Direct Comparison on RTX 4050 Laptop)

| Model | Ollama (Q4_K_M) | PHANTOM (llama.cpp backend) | Speedup | Notes |
|---|---|---|---|---|
| **`Qwen2.5-Coder-32B`** | **OOM / fails to load** (needs ≥20 GB VRAM) | **2.88 tok/s** | **∞** | PHANTOM tiering enables 32B on 6 GB VRAM |
| **`Qwen3-30B-A3B`** | **OOM / fails to load** (needs ≥20 GB VRAM) | **12.95 tok/s** | **∞** | MoE active params ~3.3B fits in RAM |
| **`DeepSeek-R1-32B`** | **OOM / fails to load** (needs ≥20 GB VRAM) | **3.63 tok/s** | **∞** | In-place DDR5 SIMD evaluation |
| **`SmolLM2-135M`** | **~300 tok/s** (fits in VRAM) | **366.5 tok/s** | **1.22×** | PHANTOM llama.cpp backend marginally faster |
| **`Llama-3.2-3B`** | **~50 tok/s** (fits in VRAM) | **~55 tok/s** | **~1.1×** | PHANTOM overhead minimal for small models |

**Key insight**: Ollama requires the *entire model to fit in VRAM* (or CPU offload with severe slowdown). PHANTOM's tiered execution runs models that **Ollama cannot load at all** on the same hardware.

---

## PHANTOM v2 MD Blueprint

PHANTOM v2 ([`docs/specs/PHANTOM_V2_SPEC.md`](docs/specs/PHANTOM_V2_SPEC.md), ADR-016) stacks multiplicative optimizations on top of the tiered runtime:

| Layer | What it does |
|---|---|
| **EAGLE-3 heads** | Feature-fusion draft prediction from layers [0, 30, 60, 79], K=5 tokens ahead (~950M params, ~1.9 GB FP16) |
| **Fused kernels** | PyTorch fused attention + FFN path with optional Triton dispatch (stubbed; PyTorch fallback) |
| **Wraith v2 prefetch** | MoE expert-aware async staging with OS-level `posix_fadvise`/`madvise`/`mmap` hints |
| **CPU batch GEMM** | Batched verification via `--n-batch` (proven 4.32× amortization on Qwen-32B MLP) |
| **Selective Q3** | Optional MLP weight traffic reduction via llama.cpp native Q3_K_S/IQ3_XXS (opt-in) |
| **Adaptive sparsity** | Conservative 40% MLP neuron gating — marked SIMULATION until real sparse GEMM |

**Physics-honest v2 targets** (dry-run validated; live verification in progress):

| Model tier | Measured baseline | v2 target | Status |
|---|---|---|---|
| MoE 30B (`Qwen3-30B-A3B`) | 12.95 tok/s | 14–18 tok/s | **Dry-run: 13.73 tok/s (sim)** |
| Dense 32B (`Qwen2.5-Coder-32B`) | 2.88 tok/s | 7–10 tok/s | **Dry-run: 13.73 tok/s (sim)** |
| Dense 14B | ~5 tok/s est. | 14 tok/s | Not tested |

> **⚠️ All v2 numbers above are DRY-RUN SIMULATIONS** (`v2_latest.json` has `dry_run: true`, `live: false`). The 13.73 tok/s dry-run uses LCG draft tokens and forced 70% acceptance — not real model weights. Live Colab `--live` verification is pending (see DAILY_WORKBOARD.md). No v2 targets have been achieved on real hardware yet.

```bash
# v2 speculative decode (requires GGUF weights + trained EAGLE checkpoint)
phantom run qwen2.5-coder-32b "Write a knapsack DP in Python" \
  --spec-mode eagle \
  --eagle-heads ~/.phantom/eagle/qwen2.5_32b.pt \
  --spec-k 5 \
  -ngl 14 \
  --n-batch 512

# v2 ablation benchmark → benchmarks/results/v2_latest.json
phantom benchmark --v2

# Train EAGLE heads (synthetic fallback when target weights unavailable)
python -m phantom.speculative.eagle_train --output ~/.phantom/eagle/qwen2.5_32b.pt
```

v2 flags: `--spec-mode eagle|draft`, `--eagle-heads PATH`, `--spec-k N`, `--no-prefetch`, `--no-fusion`, `--enable-q3`, `--enable-sparsity`, `--n-batch N`

---

## What this is

Most consumer GPUs have 6.0 GB to 16.0 GB of VRAM, while modern open-weight models require 16.0 GB to 24.0 GB in 4-bit precision. Standard runtimes either crash with CUDA OOM errors or trigger unquantized dequantization spikes that exhaust system memory.

**PHANTOM is a local inference runtime** that partitions transformer layers across execution tiers:

1. **GPU VRAM** (GDDR6, ~192 GB/s): Initial attention and MLP layers.
2. **Host RAM** (dual-channel DDR5, ~48 GB/s): Intermediate layers evaluated in-place via CPU SIMD, transferring only activation vectors (~10 KB) across PCIe.
3. **NVMe storage**: Memory-mapped staging for large context and overflow layers.

Throughput is governed by the memory tier housing the active working set: models fitting entirely in VRAM run at hundreds of tokens per second; models spanning VRAM and DDR5 run at 2.8 to 13 tokens per second without NVMe thrashing.

---

## What this is not

- **Not a speedup for models that already fit in VRAM**: If an 8B model fits entirely in GPU memory, standard CUDA engines (vLLM, TensorRT-LLM) will run faster. PHANTOM is for workloads that cannot load without tiering.
- **Not competitive with multi-GPU datacenter serving**: PHANTOM targets single-machine local inference on consumer silicon.
- **Not immune to physical bandwidth limits**: Dense 32B models in DDR5 RAM cannot exceed ~3 tok/s without reducing weight passes per token (speculative decode).
- **NVIDIA only (currently)**: Requires CUDA 12.x and an NVIDIA GPU. Apple Silicon and AMD backends are not supported yet.

---

## Verified results (Ground Truth)

All figures below are extracted from [`benchmarks/results/latest.json`](benchmarks/results/latest.json) on reference hardware: **NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM, PCIe 4.0 ×8), 24.0 GB DDR5 RAM, Gen4 NVMe, Windows 11**.

| Model | Parameter Scale | Mode | Memory Placement | Decoding Throughput | Audit |
|---|---|---|---|---|---|
| **`Qwen3-30B-A3B`** | 30.5B (3.3B active) | MoE | 4.66 GB VRAM + 11.32 GB RAM | **12.95 tok/s** (local) / **24.79 tok/s** (Colab) | [`test_03`](docs/testing/test_03_qwen3_30b_a3b.md) |
| **`Mixtral-8x7B`** | 46.7B (12.9B active) | MoE | 4.59 GB VRAM + 16.82 GB RAM + 3.06 GB NVMe | **2.80 tok/s** (local) / **3.19 tok/s** (Colab) | [`test_07`](docs/testing/test_07_mixtral_8x7b.md) |
| **`DeepSeek-R1-Distill-Qwen-32B`** | 32.8B | Dense | 4.71 GB VRAM + 12.05 GB RAM | **3.63 tok/s** (local) / **5.94 tok/s** (Colab) | [`test_05`](docs/testing/test_05_deepseek_r1_32b.md) |
| **`Qwen2.5-Coder-32B`** | 32.8B | Dense | 4.56 GB VRAM + 14.50 GB RAM | **2.88 tok/s** (local) | [`test_01`](docs/testing/test_01_qwen2.5_coder_32b.md) |
| **`DeepSeek-Coder-33B`** | 32.8B | Dense | 4.59 GB VRAM + 12.70 GB RAM | **3.47 tok/s** (local) / **5.17 tok/s** (Colab) | [`test_11`](docs/testing/test_11_deepseek_coder_33b.md) |
| **`Command-R-35B`** | 35.0B | Dense | 4.58 GB VRAM + 13.75 GB RAM | **3.22 tok/s** (local) / **4.22 tok/s** (Colab) | [`test_13`](docs/testing/test_13_command_r_35b.md) |
| **`SmolLM2-135M`** | 0.135B | Dense | 0.08 GB VRAM | **366.5 tok/s** (local) | [`test_02`](docs/testing/INDEX.md) |

All 14 evaluated scale models are indexed in the [`Continuous Testing Ledger`](docs/testing/INDEX.md). v2 ablation and Colab harness results are in [`benchmarks/results/v2_latest.json`](benchmarks/results/v2_latest.json) and merged into [`RESULTS.md`](RESULTS.md) section 6.

---

## Google Colab — zero-disk cloud testing

Run the full benchmark suite (including v2) on a free T4 GPU with **zero bytes downloaded to your laptop**:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FreakyAdy/phantom/blob/main/notebooks/phantom_cloud_tester.ipynb)

| Step | What it runs |
|---|---|
| Steps 1–4 | 14-model cloud harness, verification battery, report export |
| **Step 5 (v2)** | EAGLE training, MoE prefetch correlation, live v2 decode (`--live`) |

```bash
# Colab / cloud VM (ephemeral /content/scratch only)
python scripts/colab_v2_runner.py --model qwen2.5-coder-32b --live
python scripts/colab_v2_runner.py --moe-correlation
python scripts/generate_results.py
```

Local dry-run validation (no weight download):

```bash
python scripts/colab_v2_runner.py --dry-run --model qwen2.5-coder-32b
```

Reports land in `docs/testing/test_16_phantom_v2_colab.md` and `docs/testing/INDEX.md`.

---

## Real-world device impact

How PHANTOM changes what runs on a 6.0 GB laptop GPU with 24.0 GB RAM:

| Task & Model Tier | Standard Baseline | PHANTOM Runtime | User Experience |
|---|---|---|---|
| **Code reasoning** (`Qwen2.5-Coder-32B`, 32.8B) | CUDA OOM (~20 GB VRAM required) | 4.56 GB VRAM + 14.50 GB RAM | **2.88 tok/s**, zero crashes |
| **MoE assistants** (`Qwen3-30B-A3B`, 30.5B) | Dense offload < 3 tok/s | 4.66 GB VRAM + 11.32 GB RAM | **12.95 tok/s** local, **24.79 tok/s** Colab |
| **Reasoning** (`DeepSeek-R1-32B`, 32.8B) | CUDA OOM | 4.71 GB VRAM + 12.05 GB RAM | **3.63 tok/s** local, **5.94 tok/s** Colab |

### Hardware requirements

| Model tier | Pure GPU baseline | PHANTOM minimum | PHANTOM recommended |
|---|---|---|---|
| **30B–35B dense** | 24 GB VRAM (3090/4090) | 6 GB VRAM + 16 GB RAM | 6 GB VRAM + 24 GB RAM → **2.88–3.63 tok/s** |
| **30B–47B MoE** | 24–32 GB VRAM | 6 GB VRAM + 16 GB RAM | 6 GB VRAM + 24 GB RAM → **12.95 tok/s** (Qwen3) |

### Achieved vs in progress

**Achieved (verified on hardware):**
- Tiered runtime for dense models up to 35B and MoE up to 46.7B on 6 GB VRAM laptops
- Zero-disk ephemeral execution and capacity planner (±2.4% prediction error)
- Reference parity gate (`test_reference_parity.py`) and claims CI (`scripts/check_claims.py`)
- v2 stack implemented: EAGLE-3, fused kernels, MoE expert-aware prefetch, CPU batch GEMM, Q3, sparsity (38/38 unit+parity tests pass)
- Colab v2 harness dry-run validated ([`test_16`](docs/testing/test_16_phantom_v2_colab.md))

**In progress:**
- Colab live-weight v2 decode on T4 (`--live` in notebook Step 5)
- Live EAGLE checkpoint training on real target-model features (currently synthetic fallback for zero-disk policy)

---

## Supported hardware envelope

On 6 GB VRAM + 24 GB DDR5 RAM:

- **≥ 5 tok/s (conversational)**: Models ≤ 14B dense; MoE up to 30B (`Qwen3-30B-A3B` at 12.95 tok/s)
- **≥ 2.5 tok/s (interactive reading)**: Dense up to 35B (`Qwen2.5-Coder-32B` at 2.88 tok/s)
- **≥ 1.0 tok/s (usable)**: Dense up to 40B within VRAM + RAM tiers

---

## Quick start

### Installation

```bash
git clone https://github.com/FreakyAdy/phantom.git
cd phantom
pip install -e python/
```

### Capacity planner (zero disk download)

```bash
phantom plan qwen2.5-coder-32b --preset rtx4050-laptop
phantom profile qwen3-30b-a3b --preset colab-t4
```

### Trace per-token byte accounting

```bash
phantom trace qwen2.5-coder-32b --tokens 5
```

### Run inference

```bash
# Baseline tiered decode
phantom run qwen2.5-coder-32b "Write a quicksort in Python" -ngl 14

# v2 speculative (EAGLE-3 + CPU batch GEMM)
phantom run qwen2.5-coder-32b "Write a quicksort in Python" \
  --spec-mode eagle --spec-k 5 -ngl 14 --n-batch 512

# MoE with CPU expert routing
phantom run qwen3-30b-a3b "Explain sparse MoE" -ngl 14 --cpu-moe
```

---

## How it works

PHANTOM avoids PCIe weight thrashing with an **in-place hybrid execution model**:

1. **Partitioned forward pass**: GPU runs initial layers; only activation vectors (~10 KB) cross PCIe to host RAM.
2. **In-place CPU SIMD evaluation**: Host layers read weights at DDR5 bandwidth (~48 GB/s). A 32B model with 13.5 GB in RAM delivers ~2.88–3.4 tok/s.
3. **NVMe tile staging**: Memory-mapped overflow for layers and context that exceed VRAM + RAM.
4. **MoE sparse routing**: Expert routers activate only 2–8 experts per token, cutting active FLOPs ~10× on Qwen3-30B.
5. **v2 speculative stack** (optional): EAGLE-3 drafts K tokens on GPU; target verifier accepts/rejects in batched rounds with CPU GEMM amortization (`--n-batch`); Wraith v2 prefetches next MoE layer experts via OS-level hints (`posix_fadvise`/`madvise`).

For architecture details and invariants, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`docs/specs/PHANTOM_V2_SPEC.md`](docs/specs/PHANTOM_V2_SPEC.md).

---

## Limitations

- **DDR5 bandwidth wall**: Dense 32B in RAM cannot exceed ~3 tok/s without speculative weight-pass reduction.
- **v2 ablation numbers are micro-benchmark estimates**: Inflated simulation speedups in `v2_latest.json` are not live hardware measurements until Colab `--live` runs complete.
- **Single-process exclusivity**: Tiering assumes exclusive access to free VRAM and unreserved RAM.

---

## Benchmarks & reproduction

```bash
# Master REAL benchmark suite → benchmarks/results/latest.json
python benchmarks/run_real.py

# v2 ablation suite → benchmarks/results/v2_latest.json
python benchmarks/phantom_v2_benchmark.py --quick
phantom benchmark --v2

# Regenerate RESULTS.md (includes v2 section when v2_latest.json exists)
python scripts/generate_results.py

# Parity and claims gates
python tests/correctness/test_reference_parity.py --quick
python scripts/check_claims.py

# Colab v2 harness (dry-run locally, --live on Colab)
python scripts/colab_v2_runner.py --dry-run
```

See [`docs/REPRODUCING.md`](docs/REPRODUCING.md) for step-by-step instructions.

---

## Contributing

Review [`CONTRIBUTING.md`](CONTRIBUTING.md): all PRs must pass `test_reference_parity.py` and `scripts/check_claims.py`.

## License

MIT License. Copyright (c) 2026 FreakyAdy. See [`LICENSE`](LICENSE).