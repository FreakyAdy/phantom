# Subsystem Innovations & Research Architecture of PHANTOM

PHANTOM organizes its subsystems into verified architectural mechanisms and active experimental prototypes:

---

### Core Architectural Mechanisms (Verified)

#### 1. In-Place CPU SIMD Evaluation
- **Concept**: Evaluates host-resident layers directly in system DDR5 RAM via multi-threaded CPU SIMD vector kernels.
- **Impact**: Bypasses PCIe weight streaming bottlenecks entirely, transferring only intermediate activations (~10 KB) across the bus.

#### 2. Adaptive MoE Sparse Routing
- **Concept**: Dynamic sparse expert routing for Mixture-of-Experts architectures (`Qwen3-30B-A3B`, `Mixtral-8x7B`).
- **Impact**: Evaluates only active expert parameters (~3.3B for Qwen3-30B), achieving **12.95 tok/s (Local)** and **24.79 tok/s (Cloud)** with a 9.93x FLOP reduction over dense execution.

#### 3. Batched Speculative Verification (Milestone 1.6 / ADR-014)
- **Concept**: Heterogeneous speculative execution co-locating a fast 0.5B draft model in GPU VRAM with a batched target verifier on CPU/RAM.
- **Impact**: Amortizes DDR5 memory reads across multiple draft tokens, achieving an empirical **3.93x layer amortization factor** (23.40 ms vs 92.03 ms for 8 tokens on CPU AVX2).

#### 4. Phantom Pages (Memory Tier Manager)
- **Concept**: Orchestrates 3-tier memory residency across GPU VRAM, System RAM, and NVMe storage.
- **Impact**: Enables 30B–35B models to execute within consumer hardware envelopes with zero NVMe thrashing.

---

### Experimental Research Prototypes (In Active Development)

#### 5. Wraith Micro-Predictor (Conditional Routing)
- **Concept**: Online micro-predictor evaluated for *conditional* layer transitions (e.g. MoE expert routing and dynamic sparsity), where activation paths vary per token.

#### 6. Spectral Quantization (DCT FP8)
- **Concept**: 2D Discrete Cosine Transform compression concentrating weight energy into low-frequency coefficients.
- **Status**: Algorithmic prototype demonstrated on 2D test matrices; model-level perplexity retention is undergoing calibration.

#### 7. Neural Cache (KV Compression)
- **Concept**: Vector autoencoder compressing high-dimensional KV attention representations into low-dimensional latent embeddings.
- **Status**: Tensor prototype achieving 8.0x vector compression on synthetic embeddings; live multi-needle retrieval preservation is under active research.

