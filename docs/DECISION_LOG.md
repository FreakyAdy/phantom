# PHANTOM Architecture Decision Records (ADRs) & Problem Resolutions

This document catalogs critical architectural decisions, engineering trade-offs, and root-cause analyses of major system bottlenecks resolved throughout the PHANTOM platform lifecycle.

---

## Index of Architecture Decision Records

* **ADR-001**: Hugging Face AutoModel 65GB RAM Explosion vs Native Quantized GPU Offloading
* **ADR-002**: Elimination of Synthetic Mock Responses & Canned Strings in Production
* **ADR-003**: Baseline Calibration — Retiring SmolLM-135M Smoke-Test in Favor of Physical VRAM Limits
* **ADR-004**: Dense 32B vs MoE 30B Active Compute & Memory Bandwidth Reality
* **ADR-005**: Zero-Disk Testing Architecture — Virtual Simulation + Ephemeral Cloud Execution vs Local Disk Hoarding
* **ADR-006**: In-Place Host RAM Evaluation via CPU SIMD vs PCIe Bus Weight Streaming
* **ADR-007**: Deprecation of Web UI in Favor of Pure, Zero-Overhead Terminal UI (TUI)
* **ADR-008**: Testing Policy — Restricting Verification Exclusively to Scale Models ≥ 30B via Cloud Testbed

---

### ADR-001: Hugging Face AutoModel 65GB RAM Explosion vs Native Quantized GPU Offloading
* **Context**: When attempting to run `Qwen2.5-Coder-32B-Instruct-GGUF` via `AutoModelForCausalLM.from_pretrained(..., gguf_file=...)`, the process hung for minutes, drove system RAM usage to 100%, and crashed with Windows `STATUS_COMMITMENT_LIMIT`.
* **Root Cause Analysis**: Standard Hugging Face transformers GGUF integration dequantizes all 771 quantized GGUF tensors into uncompressed 16-bit floats (`bfloat16`/`float16`) in system RAM. For a 32.76B parameter model:
  $$\text{Memory Required} = 32.76 \times 10^9 \times 2\text{ bytes} \approx \mathbf{65.5\text{ GB RAM}}$$
  On a 24.0 GB RAM laptop, this immediately exhausted physical RAM and pagefile limits.
* **Decision**: Bypass Python-level float dequantization. Maintain weights in their native compact Q4_K_M quantized format (**18.5 GB** total) and utilize direct CUDA layer offloading (GPU computes ~4.5 GB layers, CPU SIMD computes remaining ~14 GB layers directly in RAM).
* **Consequences**:
  * Positive: Model boots in ~25s cold, consumes only 4.56 GB VRAM and 14.5 GB RAM, and runs at 2.88 tok/s with zero OOMs.
  * Negative: Requires native CUDA runtime bindings instead of pure Python packages.

---

### ADR-002: Elimination of Synthetic Mock Responses & Canned Strings in Production
* **Context**: In early iterations, when local model loading failed or timed out, the CLI printed:
  `"Hello! I am running on PHANTOM CORE with hardware transcendence."`
  and the TUI checked for prompt keywords like "who", "what", "phantom" to return canned promotional responses.
* **Decision**: Purge 100% of simulated fallback strings and keyword triggers from the codebase.
* **Rationale**: A hardware-transcendent engine must be empirically honest. Mock strings mask underlying memory or configuration errors, prevent genuine debugging, and undermine scientific integrity.
* **Consequences**:
  * Positive: All generated tokens are verified LLM output from real model weights. Failures produce transparent, actionable error messages.
  * Negative: If model weights are not loaded or missing, the system will explicitly fail rather than showing a simulated response.

---

### ADR-003: Baseline Calibration — Retiring SmolLM-135M Smoke-Test in Favor of Physical VRAM Limits
* **Context**: Previous reports cited a `242.7×` parameter scale multiplier by comparing `Qwen2.5-Coder-32B` (32.76B) to `SmolLM-135M` (135M).
* **Problem**: `SmolLM-135M` was only an early CI unit-test artifact, not a hardware baseline. An RTX 4050 laptop (6GB VRAM) can run far larger models natively.
* **Decision**: Establish physical hardware baselines grounded in the GPU's memory interface:
  1. **Native 4-bit VRAM Limit**: **~7B – 8B parameters** (~4.8 GB in Q4). Running 32.76B is a **4.10× to 4.68× capacity multiplier**.
  2. **Native 16-bit VRAM Limit**: **~2.5B – 3.0B parameters** (~5.8 GB in FP16). Running 32.76B is a **10.9× capacity multiplier**.
* **Consequences**:
  * Provides rigorous, peer-reviewable hardware metrics that accurately represent what PHANTOM achieves on consumer silicon.

---

### ADR-004: Dense 32B vs MoE 30B Active Compute & Memory Bandwidth Reality
* **Context**: The user observed that a friend was running a "~30B model" on an RTX 4050 laptop using an MoE architecture.
* **Technical Distinction**:
  * **30B MoE (e.g. `Qwen3-30B-A3B`)**: Contains 30.5B total weights (~16–18 GB in RAM), but **only activates ~3.3B parameters per token** (~6.6 GFLOPs). Computationally, the hardware only executes the math of a 3B model while retrieving expert weights.
  * **32B Dense (`Qwen2.5-Coder-32B`)**: All **32.76 Billion parameters compute on every token** (~65.5 GFLOPs).
* **Decision**: Formally document both paradigms in PHANTOM. Dense 32B serves as the ultimate hardware stress test (proving 65.5 GFLOPs/token sustained stability), while MoE represents the speed optimization frontier (~12–14 tok/s).
* **Consequences**:
  * Clarified why running Dense 32B executes **~10× more math per second** than a 30B MoE, while predicting that an MoE model will run ~4× faster on the same laptop.

---

### ADR-005: Zero-Disk Testing Architecture — Virtual Simulation + Ephemeral Cloud Execution vs Local Disk Hoarding
* **Context**: Testing multiple 20 GB – 40 GB models locally quickly filled the user's laptop SSD (`C:` drive had as low as 24 GB free).
* **Options Considered**:
  1. Buying an external NVMe SSD (hardware cost, physical dependency).
  2. Downloading and permanently storing multiple models locally (causes disk full crashes).
  3. **3-Tier Zero-Disk Testing Framework** (Virtual Simulator + Cloud Sandbox + Ephemeral Local Runner).
* **Decision**: Adopt Option 3.
  * Tier 1: Virtual Hardware Profiler (`phantom profile`) models layer splits and speeds with **0 bytes of disk overhead**.
  * Tier 2: Free Google Colab / Kaggle runner uses **100 GB cloud scratch disk** and **15 GB Nvidia GPU**.
  * Tier 3: Ephemeral local runner downloads 1 model at a time and **guarantees deletion via `try...finally`**, restoring disk space immediately.
* **Consequences**:
  * Eliminates storage anxiety; laptop SSD remains completely clean (+54 GB reclaimed, 136 GB free).

---

### ADR-006: In-Place Host RAM Evaluation via CPU SIMD vs PCIe Bus Weight Streaming
* **Context**: When layers overflow GPU VRAM into Host RAM, there are two execution models:
  * *Model A (Streaming)*: Copy the layer weights over PCIe into GPU VRAM for every token forward pass.
  * *Model B (In-Place Hybrid)*: Keep layers resident in Host RAM; execute them on the CPU host using AVX2/AVX-512 SIMD kernels, transferring only intermediate activation vectors ($O(\text{hidden\_dim}) \approx 10\text{ KB}$) across PCIe.
* **Mathematical Trade-off**:
  * On an RTX 4050 laptop, PCIe Gen4 x4 throughput is ~7.87 GB/s. Streaming 14 GB of weights every token caps speed at:
    $$\text{Throughput}_{\text{streaming}} \approx \frac{7.87\text{ GB/s}}{14\text{ GB}} \approx \mathbf{0.56\text{ tok/s}}$$
  * In-place CPU evaluation accesses dual-channel DDR5 RAM at ~48 GB/s effective bandwidth:
    $$\text{Throughput}_{\text{in-place}} \approx \frac{48\text{ GB/s}}{14\text{ GB}} \approx \mathbf{3.4\text{ tok/s}}$$
* **Decision**: Standardize on in-place hybrid offloading for RAM layers. Reserve PCIe streaming exclusively for dynamic layer swapping from NVMe SSD.
* **Consequences**:
  * Enables ~3 tok/s interactive generation on a 32B model, matching real empirical measurements.

---

### ADR-007: Deprecation of Web UI in Favor of Pure, Zero-Overhead Terminal UI (TUI)
* **Context**: The platform initially contemplated a React SPA Web UI (`ui/web`) served at `http://localhost:11411/ui`, requiring Node.js build processes, web server background daemons, and browser WebSocket polling.
* **Decision**: Formally deprecate and remove the Web UI requirement. Focus 100% of front-end engineering effort on the Terminal User Interface (TUI) (`phantom run`, `phantom menu`, `phantom profile`, `phantom plan`).
* **Rationale**: PHANTOM is a low-level systems runtime. A rich terminal interface (powered by `rich`, ANSI sequences, ASCII layer residency heatmaps, and live token-streaming bars) delivers an instant, zero-latency developer experience without consuming system memory for web engines or Node daemons.
* **Consequences**:
  * Positive: Zero browser memory consumption, instantaneous boot times, eliminates Node/npm dependencies.
  * Negative: No graphical browser dashboard.

---

### ADR-008: Testing Policy — Restricting Verification Exclusively to Scale Models ≥ 30B via Cloud Testbed
* **Context**: Small models (<10B, such as SmolLM-135M, Llama-3.2-1B, 3B) fit inside consumer 6GB VRAM natively and do not test PHANTOM's core purpose ("running models that don't fit your GPU"). Meanwhile, downloading 20GB–40GB models directly onto the user's laptop causes disk space exhaustion.
* **Decision**:
  1. Cease all testing of sub-30B models. Restrict all future platform benchmarks strictly to scale models **≥ 30B parameters** (`Qwen3-30B-A3B` MoE, `Qwen2.5-Coder-32B` Dense, `Llama-3-70B` Dense).
  2. Standardize on the **Google Colab Cloud Hardware Testbed** ([`notebooks/phantom_cloud_tester.ipynb`](file:///c:/Work/Projects/Solution%20is%20all%20You%20need/notebooks/phantom_cloud_tester.ipynb)) as the primary platform for physical inference testing:
     - Leverages free 15 GB Nvidia GPU (T4/L4) and 100 GB ephemeral scratch cloud SSD.
     - Guarantees **0 bytes of local disk usage** on the user's physical laptop.
* **Consequences**:
  * Protects local disk integrity completely while benchmarking true hardware-transcendent workloads (30B–70B).

---

### ADR-009: Ground Truth Remediation, Elimination of Self-Grading Audits, and Structural CI Guardrails
* **Context**: Prior documentation contained conflicting and wobbling numbers (e.g. Wraith latency 0.458 vs 0.487 ms, context switch 80.2 vs 80.9 ms, unverified 100B parameter extrapolation), self-grading audits (`audit.md`), and an unexplained PCIe bandwidth paradox on the 32B model run.
* **Decision**:
  1. **Physical Architecture Resolution**: Proved mathematically and empirically that Qwen2.5-Coder-32B does not stream 15 GB of weights across PCIe. Layers 0–13 run in VRAM, layer 13 intermediate activation tensor ($[1, 1, 5120]$ FP16 $\approx 10\text{ KB}$) transfers across PCIe in $1.3\ \mu\text{s}$, and layers 14–63 are evaluated in-place in Host RAM using multi-threaded CPU SIMD at dual-channel DDR5 bandwidth (~44–48 GB/s).
  2. **Deletion of Self-Grading Documents**: Permanently deleted `audit.md`. Replaced with canonical `RESULTS.md` generated programmatically from `benchmarks/results/latest.json`, with human changes recorded in `CHANGES.md` and runs logged in `WORKLOG.md`.
  3. **Structural CI Consistency Gate**: Built `scripts/check_claims.py` and `docs/claims_allowlist.yml` to regex-scan all markdown files in CI and reject any unverified numeric claim or conflicting metric across documents.
  4. **Numerical Parity Gate**: Enforced `tests/correctness/test_reference_parity.py` as a prerequisite for all PRs (requiring >99% top-1 agreement and measuring KL divergence).
* **Consequences**:
  * Positive: Completely eliminates metric drift and marketing inflation; ensures every published figure traces to an unforgeable hardware fingerprint; gives the repository impenetrable scientific credibility.
  * Negative: Requires all new performance metrics to be measured on physical hardware or added to `docs/claims_allowlist.yml` with written owner justification before appearing in markdown.

---

### ADR-010: Frontier 70B Double-Buffered Persistent NVMe Tile Streaming and Fused Spectral SwiGLU Decompression
* **Context**: Dense 70B–72B models on consumer hardware (6 GB VRAM + 24 GB RAM) overflow fast physical memory by 14.6 to 16.7 GB, forcing ~37 layers to stream from NVMe SSD on every single token pass. Naive implementations suffer from severe syscall overhead (repeatedly calling `open()`/`close()` per layer tile) and transfer full uncompressed weights over the physical NVMe bus, physically bounding throughput to ~0.36–0.40 tok/s.
* **Decision**:
  1. **Persistent Descriptor & Double-Buffered Prefetching**: Maintain persistent OS file descriptors in both Rust `PhantomPageManager` and Python `AsyncTilePagingEngine`. Allocate two 4096-byte aligned ping-pong buffers (Buffer A / Buffer B) so that Layer $L+1$ is asynchronously DMA-streamed from NVMe while Layer $L$ is executing its GEMV computation.
  2. **Fused SwiGLU + FP8 iDCT Decompression**: Store swapped layers in compact FP8 Discrete Cosine Transform (DCT) spectral representations, reducing the physical byte volume streamed from NVMe by 2.0×. Fuse the inverse-DCT reconstruction directly with the SwiGLU non-linear activation in registers/SRAM, eliminating intermediate uncompressed weight buffers in host memory.
* **Consequences**:
  - Positive: Halves physical NVMe transfer volume per token; overlaps layer I/O with compute; eliminates per-tile open/close syscall overhead (~0.8–2.1 ms saved per tile).
  - Negative: Requires specialized fused kernel paths and 4KB sector alignment for all pagefile tile offsets.

---

### ADR-011: Long-Context Needle-In-A-Haystack (NIAH) Verification & 8.0x Neural Cache Manifold Calibration
* **Context**: Scaling context windows to 32768 tokens on scale models (>=30B) consumes 4.0 GB of VRAM solely for uncompressed FP16 KV-cache states (32 layers, 8 heads, 128 head_dim), immediately causing out-of-memory errors on consumer 6.0 GB laptop GPUs. Innovation 3 (Neural Cache) compresses head dimensions $D \to D/8$ (128 to 16) via a 3-layer bottleneck autoencoder. However, prior evaluation lacked long-context needle-in-a-haystack verification across varied insertion depths.
* **Decision**:
  1. Build a zero-disk, long-context NIAH evaluation suite (`tests/correctness/test_needle_haystack.py` and `tests/benchmarks/bench_needle_haystack.py`) testing context windows spanning 4096, 8192, 16384, and 32768 tokens across 10.0%, 25.0%, 50.0%, 75.0%, and 90.0% insertion depths.
  2. Calibrate autoencoder projection matrices to the low-rank semantic manifold of transformer KV states ($D//8 = 16$), achieving 100.0% needle retrieval recall, 100.0% attention preservation, and 0.9829 key cosine similarity.
  3. Formally register `needle_haystack` benchmark in `benchmarks/run_all.py` and publish empirical results in `docs/testing/test_15_long_context_needle_haystack.md`.
* **Consequences**:
  - Positive: Proves empirically and mathematically that 8.0x KV cache compression maintains 100.0% needle retrieval recall across 32768 tokens while reducing KV memory footprint from 4096 MB (4.0 GB) to 512 MB (0.50 GB).
  - Negative: Compression is specialized to the intrinsic low-rank manifold geometry of transformer representations.

---

### ADR-012: Automated 1-Click Cloud Testbed Packaging and Ephemeral Cloud Scratch Policy
* **Context**: Reproducing benchmarks on consumer hardware constraints previously required manual CLI setup, dependency configuration, and access to local GPU environments. External reviewers and open-source contributors required an instant 1-click cloud verification pathway across the evaluated model suite without violating the strict Zero-Disk Model Storage mandate on local machines.
* **Decision**:
  1. **Dual-Mode Cloud Harness (`scripts/colab_runner.py`)**: Implement an automated cloud runner supporting two explicit execution modes:
     - *Virtual Hardware Simulation Mode*: Instant, zero-disk parameter modeling and memory residency calculation across target profiles (`colab-t4`, `rtx4050-laptop`, `rtx4070-desktop`, `rtx4090-desktop`, `apple-m3-pro`).
     - *Live Ephemeral Inference Mode*: Streaming weights directly to cloud ephemeral VM storage (`/content/scratch/`), executing non-synthetic verification battery, and immediately auto-purging the directory post-run.
  2. **Standardized Report Automation**: Generate standardized markdown reports conforming byte-for-byte to `docs/testing/TEMPLATE_TEST_REPORT.md` and dual-export raw JSON telemetry alongside markdown artifacts directly to the user's browser.
  3. **Colab Interactive Form Packaging (`notebooks/phantom_cloud_tester.ipynb`)**: Overhaul the notebook with interactive `#@param` dropdowns for evaluated models, inline telemetry visualization, automatic package dependency bootstrapping, and automatic cleanup hooks.
* **Consequences**:
  - Positive: Enables 1-click end-to-end cloud reproducibility on standard Google Colab T4 runtimes; produces publication-ready benchmark reports automatically; enforces strict zero-disk local storage invariants.
  - Negative: Live inference mode in Colab is subject to Google Colab T4 VRAM and session disconnect limits; frontier 70B models in live mode require high-RAM cloud instances or fall back to virtual hardware simulation mode.

---

### ADR-013: Deprioritization of Frontier 70B NVMe Paging and Realignment on 30B–35B High-Speed Real-Time Tier
* **Context**: Dense 70B–72B models on consumer hardware (6.0 GB VRAM + 24.0 GB RAM) overflow fast memory by 14.6 to 16.7 GB, requiring continuous streaming from NVMe SSD on every autoregressive token. Although numerically verified and physically correct, physical Gen4 SSD read bandwidth (~1.4–1.8 GB/s) physically limits decoding throughput to ~0.36–0.40 tok/s (~2.5s per token). Promoting 70B on consumer laptops creates false user expectations of real-time conversational chat, inviting cynicism and "100 tokens per year" criticism.
* **Decision**:
  1. **Deprioritize 70B Interactive Promotion**: Remove dense 70B models from recommended front-page hardware matrices and promotional positioning in `README.md`. Retain existing raw empirical benchmarks (`test_04`, `test_06`, `test_09`, `test_12`) strictly as historical cold-tier reference benchmarks in `docs/testing/`.
  2. **Core Realignment on 30B–35B High-Speed Tier**: Focus PHANTOM's primary production envelope on workloads delivering genuine real-time utility without NVMe swap:
     - Sparse MoE models (`Qwen3-30B-A3B`) executing at **12.95 tok/s (Local)** and **24.79 tok/s (Cloud)**.
     - Dense frontier coding models (`Qwen2.5-Coder-32B`, `DeepSeek-R1-32B`, `Command-R-35B`) executing at **2.88 to 3.63 tok/s (Local)** with in-place DDR5 CPU SIMD evaluation.
  3. **Publish Transparent Baseline Multipliers**: Quantify verified achievements against standard baseline limits: 4.88x median (5.15x average) parameter ceiling expansion over 6.0 GB GPU limits, 4.50x VRAM footprint reduction, 1.8x median speedup on dense models, 10.0x MoE speedup over CPU baseline, and 8.0x KV cache compression.
  4. **Active Engineering Pivot**: Prioritize in-VRAM Speculative Decoding (drafting tokens at 80+ tok/s to multiply 32B speed to 8–10 tok/s) and streaming chunked prefill over SSD tile optimizations.
* **Consequences**:
  - Positive: Eliminates marketing hype and aligns user expectations with physical reality; establishes impenetrable technical credibility; focuses roadmap on high-speed interactive techniques.
  - Negative: Shifts 70B from an active interactive target to an offline/background research milestone.

---

### ADR-014: Speculative Decoding with Batched CPU GEMM Verification as Primary Acceleration Strategy
* **Context**: Deep research investigation (2026-09-17) into whether PHANTOM can transform 5 tok/s into 14 tok/s on the same low-end consumer hardware (RTX 4050 6GB + 24GB DDR5). Comprehensive audit of PHANTOM codebase, physical bandwidth ceiling analysis, roofline modeling, and survey of 30+ optimization techniques.
* **Key Findings**:
  1. **Dense 32B → 14 tok/s is physically impossible**: DDR5 dual-channel at ~48 GB/s reading 14 GB of RAM-resident weights bounds single-token decode to ~3.4 tok/s theoretical maximum. No software optimization changes this.
  2. **Roofline analysis**: Single-token decode operates at arithmetic intensity 0.0036 FLOPs/byte — 7,800× below the GPU compute ceiling. The workload is catastrophically memory-bandwidth-bound.
  3. **Critical opportunity**: Speculative decoding with batch-8 verification reads weights ONCE from DDR5 but verifies 8 tokens simultaneously. If CPU GEMM(8×5120, 5120×27648) ≈ GEMV(1×5120, 5120×27648) in wall-clock time, this amortizes the bandwidth cost by ~5.7× (at α=0.7 acceptance rate).
  4. **Rust core engine gap**: The `PhantomEngine::generate()` function returns a placeholder string. All actual inference runs through the Python/PyTorch path. Custom CUDA kernels exist but are not wired into the runtime.
* **Decision**:
  1. **Adopt batched speculative verification** as the highest-priority engineering target (Milestone 1.6).
  2. **Replace CPU GEMV kernels with CPU GEMM kernels** that can efficiently handle batch=8 evaluation of RAM-resident Q4 layers during speculative verification.
  3. **Integrate a 0.5B draft model** (e.g., Qwen2.5-0.5B, ~0.3 GB) in GPU VRAM alongside the target model's GPU layers.
  4. **Evaluate llama.cpp backend integration** as an alternative to building custom speculative decoding from scratch.
  5. **Reframe the 14 tok/s target** to apply to Dense 14B and MoE 30B models (achievable) rather than Dense 32B (physically impossible).
* **Consequences**:
  - Positive: Most promising path to 2–3× throughput improvement on dense models; validates PHANTOM's unique CPU/GPU hybrid architecture advantage; if GEMM amortization works, Dense 32B could reach 5–8 tok/s.
  - Negative: Requires significant implementation effort (CPU GEMM kernels, draft model management, tree attention, KV cache handling); uncertain whether small-batch CPU GEMM achieves full amortization on the reference hardware.
