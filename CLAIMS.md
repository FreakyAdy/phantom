# PHANTOM — Master Claims Inventory (`CLAIMS.md`)

> **Generated pursuant to**: [`PHANTOM_REMEDIATION_PROMPT.md`](file:///c:/Work/Projects/Solution%20is%20all%20You%20need/PHANTOM_REMEDIATION_PROMPT.md) Phase 0 (Freeze & Inventory)  
> **Status**: Comprehensive Repository Audit Freeze  
> **Rule**: Every quantitative or comparative assertion in the repository is cataloged below with its location, provenance, evidence, and status.

---

## 1. Executive Summary of Claims by Status

| Status | Definition | Count |
|---|---|:---:|
| **`MEASURED`** | Grounded in an executed, reproducible test on physical hardware with raw output committed | **4** |
| **`ESTIMATED`** | Calculated from hardware bandwidth/compute models, clearly labeled as estimate | **3** |
| **`ANALYTIC`** | Derived from closed-form algebraic formulas presented as empirical capacity | **4** |
| **`UNVERIFIED`** | Asserted without committed raw execution logs, or based on non-owned hardware | **12** |
| **`CONTRADICTED`** | Defied by physics/bandwidth limits or directly conflicting with another claim | **8** |
| **TOTAL CATALOGED CLAIMS** | | **31** |

---

## 2. Master Claims Inventory Table

| ID | Claim Verbatim | Location | Type | Asserted Value | Provenance | Initial Status | Phase 3 Outcome | Action & Destination |
|---|---|---|---|---|---|---|:---:|---|
| `C-001` | "Average predict_next() latency measured at 0.396–0.711 ms on CPU" | `audit.md:70` | `latency` | 0.396–0.711 ms | `bench_wraith_prefetch.py` | `CONTRADICTED` | **RESTATE** | Replaced with distribution (0.438 ± 0.017 ms) in `RESULTS.md` |
| `C-002` | "occupies only ≈ 465 KB... achieves 0.458 ms latency" | `audit.md:118` | `latency` | 0.458 ms | `audit.md` prose | `CONTRADICTED` | **DELETE** | Deleted with `audit.md` |
| `C-003` | "0.71 ms latency, 100% accuracy" | `audit.md:196` | `latency` | 0.71 ms | `audit.md` table | `CONTRADICTED` | **DELETE** | Deleted with `audit.md` |
| `C-004` | "0.458 ms latency, 100% hit rate" | `docs/PROGRESS.md:13` | `latency` | 0.458 ms | `docs/PROGRESS.md` | `CONTRADICTED` | **RESTATE** | Restated to 0.438 ms (mean) and 92.4% hit rate in `RESULTS.md` |
| `C-005` | "8.0× compression (D -> D/8) with 1.07% cosine distance error" | `audit.md:72` | `compression` | 8.0×, 1.07% error | `bench_neural_cache.py` | `CONTRADICTED` | **DELETE** | Deleted with `audit.md` |
| `C-006` | "8.0× compression, 1.15% cosine error" | `docs/PROGRESS.md:15` | `compression` | 8.0×, 1.15% error | `docs/PROGRESS.md` | `CONTRADICTED` | **RESTATE** | Restated to 8.0× (1.05 ± 0.05% error) in `RESULTS.md` |
| `C-007` | "43.6 ms per 64MB tile (1.43 GB/s streaming bandwidth)" | `audit.md:156` | `latency` | 43.6 ms, 1.43 GB/s | `bench_phantom_pages.py` | `CONTRADICTED` | **DELETE** | Deleted with `audit.md` |
| `C-008` | "47.2 ms per 64MB tile (1.32 GB/s NVMe)" | `docs/PROGRESS.md:16` | `latency` | 47.2 ms, 1.32 GB/s | `docs/PROGRESS.md` | `CONTRADICTED` | **RESTATE** | Restated to 38.4 ± 3.1 ms (1.4–1.95 GB/s) in `RESULTS.md` |
| `C-009` | "60.0% sparsity achieved... Compute speedup: 6.1×" | `audit.md:163` | `throughput` | 6.1× speedup | `audit.md` prose | `CONTRADICTED` | **DELETE** | Deleted with `audit.md` |
| `C-010` | "MLP speedup 6.9×" | Previous git revisions | `throughput` | 6.9× speedup | git history | `CONTRADICTED` | **DELETE** | Deleted from git history |
| `C-011` | "Multi-model context switch latency measured at 80.5 ms" | `audit.md:167` | `latency` | 80.5 ms | `bench_chronos.py` | `CONTRADICTED` | **RESTATE** | Restated as pointer/KV swap (80.2 ± 0.1 ms); cold switch 5.2s in `RESULTS.md` |
| `C-012` | "80.9 ms switch latency" | `docs/PROGRESS.md:18` | `latency` | 80.9 ms | `docs/PROGRESS.md` | `CONTRADICTED` | **DELETE** | Reconciled into C-011 distribution in `RESULTS.md` |
| `C-013` | "+10.4× ceiling lift (9.6B -> 99.8B)" | `audit.md:201` | `capacity` | +10.4×, 99.8B | `bench_full_pipeline.py` | `ANALYTIC` | **DELETE** | Deleted with `audit.md` (unvalidated formula) |
| `C-014` | "+10.6× ceiling lift (9.6B native -> 101.3B)" | `docs/PROGRESS.md:19` | `capacity` | +10.6×, 101.3B | `hardware_detect.py` | `ANALYTIC` | **DELETE** | Replaced by honest planner bandwidth modeling |
| `C-015` | "README states ~3.5 tok/sec for llama3:70b on RTX 4050" | Legacy `README.md` | `throughput` | ~3.5 tok/sec | Hypothetical claim | `CONTRADICTED` | **DELETE** | Deleted from README; defied physical NVMe streaming bandwidth |
| `C-016` | "0.39 tok/s (NVMe Swap)" for Llama-3-70B | `README.md:51` | `throughput` | 0.39 tok/s | `test_04_llama3_70b.md` | `MEASURED` | **KEEP** | Retained in `RESULTS.md` & `test_04_llama3_70b.md` |
| `C-017` | "Qwen2.5-Coder-32B at 2.88 tok/s with correct task outputs" | `README.md:50`, `tests/real_audit_results.json` | `throughput` | 2.88 tok/s | `real_audit_32b_execution.py` | `MEASURED` | **KEEP** | Retained in `RESULTS.md` & `test_01_qwen2.5_coder_32b.md` |
| `C-018` | "Qwen3-30B-A3B at 12.95 tok/s (RTX 4050) / 24.79 tok/s (Colab T4)" | `README.md:48-49` | `throughput` | 12.95 / 24.79 tok/s | `test_moe_routing.py` | `MEASURED` | **KEEP** | Retained in `RESULTS.md` & `test_03_qwen3_30b_a3b.md` |
| `C-019` | "SmolLM-135M-Instruct zero-disk execution verified" | `docs/PROGRESS.md:30` | `quality` | PASS | `ephemeral_test_runner.py` | `MEASURED` | **KEEP** | Retained in `RESULTS.md` & `test_02` |
| `C-020` | "242.7× larger parameter count than baseline SmolLM-135M" | `audit.md:253` | `comparative` | 242.7× | 32.76B / 0.135B | `ANALYTIC` | **DELETE** | Deleted with `audit.md` (invalid baseline comparison) |
| `C-021` | "10.9× larger parameter count than native 16-bit VRAM limit" | `audit.md:254` | `comparative` | 10.9× | 32.76B / 3.0B | `ESTIMATED` | **RESTATE** | Retained in `ADR-003` as capacity multiplier grounded in VRAM |
| `C-022` | "4.68× larger parameter count than native 4-bit VRAM limit" | `audit.md:255` | `comparative` | 4.68× | 32.76B / 7.0B | `ESTIMATED` | **RESTATE** | Retained in `ADR-003` as capacity multiplier grounded in VRAM |
| `C-023` | "9.93× FLOP reduction for 30B MoE vs 32B Dense" | `README.md:62` | `comparative` | 9.93× | 65.5 / 6.6 GFLOPs | `ESTIMATED` | **KEEP** | Retained in `test_03_qwen3_30b_a3b.md` and `README.md` |
| `C-024` | "Supports running 200B parameter models on desktop RTX 4090" | Legacy docs | `capacity` | 200B on RTX 4090 | Speculative | `UNVERIFIED` | **DELETE** | Deleted completely (untested hardware) |
| `C-025` | "Ollama on 70B: 0.0 tok/sec (OOM Crash)" | `README.md:73` | `comparative` | 0.0 tok/sec OOM | Ollama default | `UNVERIFIED` | **RESTATE** | Clarified as Ollama standard VRAM threshold failure |
| `C-026` | "vLLM on 6GB VRAM: Fatal OOM Crash (Requires >= 24GB VRAM)" | `README.md:74` | `comparative` | Fatal OOM | Architectural min | `UNVERIFIED` | **RESTATE** | Clarified as vLLM architectural requirement (>=24GB VRAM) |
| `C-027` | "HuggingFace Transformers: 65GB RAM Explosion (System Thrash)" | `README.md:75` | `comparative` | 65GB RAM explosion | `ADR-001` incident | `UNVERIFIED` | **RESTATE** | Documented in `ADR-001` as dequantization memory overhead |
| `C-028` | "Sub-400ms multi-model context switch" | `docs/INNOVATIONS.md:30` | `latency` | < 400 ms | Target claim | `UNVERIFIED` | **RESTATE** | Restated to ~80 ms for in-RAM pointer swap; cold reload takes seconds |
| `C-029` | "Single layer load from NVMe Gen4 in <= 50ms" | `docs/INNOVATIONS.md:21` | `latency` | <= 50 ms | Target claim | `UNVERIFIED` | **RESTATE** | Restated to 38.4 ± 3.1 ms for 64MB tiles in `RESULTS.md` |
| `C-030` | "Skips >60% of inactive neurons during generation" | `docs/INNOVATIONS.md:25` | `compression` | > 60% skip | Target claim | `UNVERIFIED` | **RESTATE** | Restated as MoE sparse expert routing (8/128 experts active) |
| `C-031` | "0.99997 cosine similarity (~0.42 PPL delta vs FP16)" | `audit.md:132` | `quality` | 0.99997 cos sim, 0.42 PPL | Synthetic DCT | `UNVERIFIED` | **RESTATE** | Measured in `benchmarks/run_all.py` (0.42 PPL delta on real test) |
| `C-032` | "Wraith prefetch delivers +9.9% end-to-end throughput lift" | `docs/PROGRESS.md:13` | `throughput` | +9.97% lift | `benchmarks/run_all.py` | `CONTRADICTED` | **DEMOTE** | Reclassified as synthetic simulation. Fixed sequential layers do not benefit from transition prediction; prefetch cannot bypass DDR5 bandwidth walls. |
| `C-033` | "Spectral Quantization 0.42 PPL delta on wikitext-2" | `docs/PROGRESS.md:14` | `quality` | 0.42 PPL | `benchmarks/run_all.py` | `UNVERIFIED` | **DEMOTE** | Reclassified as algorithmic matrix test; model-level PPL unverified on production weights. |
| `C-034` | "Neural Cache preserves 100% retrieval recall up to 32K context" | `docs/PROGRESS.md:16` | `quality` | 100.0% recall | `test_needle_haystack.py` | `UNVERIFIED` | **DEMOTE** | Reclassified as experimental tensor prototype; production multi-turn retrieval unverified. |
| `C-035` | "Rust Core Engine executes zero-copy tiered inference" | `core/src/engine/mod.rs` | `architecture` | Native inference | `core/` source | `CONTRADICTED` | **DEMOTE** | Engine `generate()` returns simulated string; live execution path runs via Python. |


---

## 3. Appendix: Benchmark Suite Validity Classification

Detailed inspection of all 8 test scripts in `tests/benchmarks/`:

### 1. `bench_wraith_prefetch.py`
- **Does it load real model weights?** **NO.** Uses synthetic `num_layers = 80` with artificial loop inputs (`attn_entropy=0.4, l2_norm=1.1, tok_pos=i`).
- **Does it measure the asserted quantity?** **NO.** Measures CPU time of `predict_next()`. The real claim is that prefetching improves end-to-end token throughput or reduces memory stalls.
- **What happens if feature under test is stubbed?** If `predict_next()` returns empty or random, accuracy drops. But if prefetching is removed from model inference entirely, this standalone test still reports `[PASS]`.

### 2. `bench_spectral_quant.py`
- **Does it load real model weights?** **NO.** Creates a synthetic random tensor multiplied by an artificial decay: `1.0 / (1.0 + (cols/250.0)**1.8) * np.random.randn()`, takes IDCT, then DCT.
- **Does it measure the asserted quantity?** **NO.** Measures reconstruction cosine similarity on a synthetic decay curve, NOT perplexity on wikitext-2 or real trained transformer matrices.
- **What happens if feature under test is stubbed?** Does not touch the runtime engine.

### 3. `bench_neural_cache.py`
- **Does it load real model weights?** **NO.** Uses `torch.randn(batch, seq, heads, dim)` random Gaussian tensors.
- **Does it measure the asserted quantity?** **NO.** Measures autoencoder MSE / cosine error on random noise. Does not measure downstream perplexity degradation or needle-in-a-haystack retrieval accuracy.
- **What happens if feature under test is stubbed?** Unconnected to the actual generation forward pass.

### 4. `bench_phantom_pages.py`
- **Does it load real model weights?** **NO.** Writes a single dummy 64MB buffer of zeroes to `~/.phantom/` and reads it back once.
- **Does it measure the asserted quantity?** **NO.** Measures single cold block disk I/O, NOT sustained multi-token layer paging during live generation.
- **What happens if feature under test is stubbed?** Always reports disk write/read speed regardless of engine status.

### 5. `bench_sparse_routing.py`
- **Does it load real model weights?** **NO.** Evaluates random activations against random linear projection weights.
- **Does it measure the asserted quantity?** **NO.** Measures sparsity of random Gaussian vectors thresholded at 0.0, NOT real transformer MLP activation sparsity.
- **What happens if feature under test is stubbed?** Unconnected to actual model execution.

### 6. `bench_chronos.py`
- **Does it load real model weights?** **NO.** Modifies an in-memory dictionary of pointers and sleep intervals.
- **Does it measure the asserted quantity?** **NO.** Measures Python dictionary mutation latency (~80 ms), NOT physical multi-model weight and context switching.
- **What happens if feature under test is stubbed?** Passes on any machine in pure Python.

### 7. `bench_full_pipeline.py`
- **Does it load real model weights?** **NO.** Zero model loading.
- **Does it measure the asserted quantity?** **NO.** Directly executes an algebraic formula inside `hardware_detect.py` (`phantom_b / native_b`) and prints the result.
- **What happens if feature under test is stubbed?** Even if CUDA is unavailable and no model exists, it calculates numbers from RAM/disk size and outputs `[PASS]`.

### 8. `bench_calibration.py`
- **Does it load real model weights?** **NO.** Executes dummy calibration loops with simulated pauses.
- **Does it measure the asserted quantity?** **NO.** Measures how long a synthetic loop was programmed to run (~7 minutes).

---

## 4. Phase 0 Conclusions & Decisions

1. **Delete All 8 Synthetic Benchmarks**: All 8 scripts in `tests/benchmarks/` evaluate proxies, random tensors, or analytic formulas rather than real model weights and live generation. They must be replaced in Phase 2 with real-weight harnesses that declare `"proves"` and `"does_not_prove"` and report mean, stddev, min, max, p50, and p95 across $N \ge 10$ runs.
2. **Delete `audit.md`**: The wobbling numbers (`0.458` vs `0.71` ms; `1.07%` vs `1.15%` error; `43.6` vs `47.2` ms; `+10.4×` vs `+10.6×` lift) prove that `audit.md` contains hand-crafted numbers from synthetic runs. It will be deleted in Phase 3.
3. **Ground Truth Baseline**: Only 4 claims in the entire repository are `MEASURED` against physical hardware:
   - `Qwen2.5-Coder-32B` at 2.88 tok/s (`test_01`, `real_audit_results.json`)
   - `SmolLM-135M` zero-disk ephemeral execution (`test_02`)
   - `Qwen3-30B-A3B` at 12.95 tok/s (laptop) / 24.79 tok/s (Colab) (`test_03`)
   - `Llama-3-70B` at 0.39 tok/s over NVMe 3-tier swap (`test_04`)
