## [Unreleased] — 2026-09-18 (Truth-First Remediation & llama.cpp-First Backend)

### Fixed — Fabrication Layer Purge (ADR-019)
* `tests/ephemeral_test_runner.py`: Replaced `passed=True` placeholder with real llama.cpp inference via `create_engine_for_model`; fixed Qwen3-30B-A3B proxy reference to use correct repo (`bartowski/Qwen3-30B-A3B-Instruct-GGUF`).
* `scripts/colab_runner.py`: Removed unconditional `status: "PASS"`; added real llama.cpp inference path; dry-run mode explicitly labeled SIMULATED; tasks now report actual model output and pass/fail based on target verification.
* `benchmarks/run_all.py`: Replaced sine-wave/hardcoded benchmarks with real measurements:
  - `llama_cpp_inference` (SmolLM2-135M real inference)
  - `cpu_gemm_amortization` (proven 4.32x batched GEMM vs GEMV)
  - `nvme_tile_io` (real NVMe tile I/O with double-buffered prefetch)
  - `memory_bandwidth` (real DDR5 streaming copy)
  Removed broken `test_needle_haystack` import; `planner_validation` now derived from measured data.
* `tests/correctness/test_reference_parity.py`: Graceful skip if llama-cpp-python unavailable; validates engine consistency on real model (SmolLM2-135M); true logit parity gate using HF FP32 reference (future).
* **Fabricated artifacts deleted**: `test_03_qwen3_30b_a3b_cloud.md`, `cloud_results_qwen3-30b-a3b_20260917_191551Z.json` (dry-run published as "Colab Live VERIFIED" with Qwen2.5-Coder-32B proxy).

### Fixed — Correctness Bugs (ADR-019 follow-up)
* `python/phantom/loader/gguf_loader.py`: Implemented CPU dequantization for Q5_1, Q5_K, Q2_K, Q3_K, IQ*; replaced silent zero-fallback with `NotImplementedError` for unsupported formats.
* `python/phantom/speculative/eagle_heads.py`: Fixed 64-layer KeyError (h79 missing) by gracefully handling missing fusion layer features.
* `python/phantom/speculative/engine.py`: Fixed first-round KV missing prefix context (EAGLE feature capture now returns `past_key_values`); removed fabricated accounting (`bytes//10`, `x0.85`, `torch.randn` prefetch, hardcoded 70% sim acceptance, LCG drafts); `baseline_tok_per_sec` now 0.0 (must be set from real measurement).
* `python/phantom/kernels/dispatch.py`: Fixed arg-order bug (`use_triton` -> `scale_q`/`scale_w1`); Triton branches now clearly marked as stubs.
* `python/phantom/sparsity/adaptive_gate.py`: Marked as SIMULATION -- `gate_precision` now 0.0 (unvalidated), `mlp_speedup_estimate` 1.0 (no real sparse GEMM); `apply_sparse_ffn` executes dense matmuls.
* `python/phantom/quant/selective_q3.py`: Marked as SIMULATION -- `quantize_tensor_q3` returns full-precision tensor; PPL delta formula is placeholder.
* `python/phantom/prefetch/wraith_v2.py`: Marked as SIMULATION -- LSTM predictor replaced with MoE expert tracker; `prefetch_layer_async` now uses OS-level `posix_fadvise`/`madvise`/`mmap` touch for real I/O hints.

### Added — llama.cpp-First Hybrid Backend (ADR-017)
* `python/phantom/runtime/llamacpp_backend.py`: New `LlamaCppEngine` with real metrics (tok/s, TTFT, VRAM, weight bytes); native speculative decoding via `draft_model` parameter; GGUF mmap tiering; real byte accounting.
* `python/phantom/runtime/__init__.py`: Exports `LlamaCppEngine`, `LlamaCppMetrics`, `create_engine_for_model`, `resolve_model_path`.
* `python/phantom/phantom_cli.py`: Rewired `cmd_run` to prefer llama.cpp backend; added `--n-batch` flag for CPU GEMM amortization tuning; speculative decoding via llama.cpp native draft model.
* **API rewire**: `python/phantom/api/gateway.py`, `ollama_compat.py`, `openai_compat.py` now use real llama.cpp backend; removed fake `/generate`, `/chat`, `/ps` canned strings; implemented real `/api/show`, `/api/ps`.

### Added — Real Measurement & Gates (ADR-018)
* `benchmarks/run_real.py`: Authoritative real benchmark suite (llama.cpp inference, GEMM amortization, NVMe I/O, memory bandwidth) -> `benchmarks/results/latest.json`.
* **CI gate**: Updated `.github/workflows/ci.yml` with real parity gate + real benchmark gate (scheduled); added `llama-cpp-python` dependency.
* `scripts/check_claims.py`: Removed auto-approved magic numbers; validates only against real `latest.json`.

### Added — MoE Expert-Aware Prefetch (ADR-021)
* `python/phantom/prefetch/wraith_v2.py`: New `MoEExpertTracker` (architecture-aware expert activation tracking); `OSLevelPrefetcher` (real `posix_fadvise`/`madvise`/`mmap` touch); `MoEExpertAwarePrefetcher` (expert-aware, real I/O hints); backwards-compatible `WraithV2Predictor` and `AdaptivePrefetchScheduler`.
* `python/phantom/prefetch/moe_correlation.py`: Updated to use `MoEExpertTracker` instead of LSTM predictor.

### Added — CPU Batch GEMM Amortization (ADR-020)
* `python/phantom/phantom_cli.py`: Added `--n-batch` flag (default 512) for CPU GEMM amortization tuning; passed to `create_engine_for_model`.
* `python/phantom/runtime/llamacpp_backend.py`: `n_batch` parameter exposed in `LlamaCppEngine`.
* `benchmarks/run_real.py`: Includes `cpu_gemm_amortization` benchmark (validates 4.32x amortization).

### Added — ADRs & Governance
* **ADR-017**: llama.cpp-First Hybrid Backend Adoption
* **ADR-018**: Measurement-Only Publishing (Dry-Run Never Labeled Live)
* **ADR-019**: Fabrication Layer Purge
* **ADR-020**: CPU Batch GEMM Amortization via n_batch Tuning
* **ADR-021**: MoE Expert-Aware Streaming (Real Wraith v2)
* **Updated `docs/DECISION_LOG.md`** with ADR-017 through ADR-021.

### Changed — Documentation Reconciliation (Phase 5)
* `docs/DECISION_LOG.md`: Added ADR-017 through ADR-021.
* `docs/CHANGELOG.md`: This entry.
* `docs/DAILY_WORKBOARD.md`: Updated session summary with audit findings and remediation queue.
* `docs/claims_allowlist.yml`: Ready for cleanup (remove fabricated 4.2 tok/s family).
* **Stale docs flagged**: `PHANTOMFILE.md`, `PLUGINS.md`, `CONCEPT_MAP.md`, `toolextensio.md` still reference purged subsystems (Neural Cache, Spectral Quantization) -- require follow-up.