# PHANTOM — Daily Mission Workboard & Session Protocol

> **Daily Operation Rule**: Whenever the user says *"start with today"*, the agent immediately reads this workboard, aligns with today's prioritized agenda, executes the **Build $\to$ Debug $\to$ Test $\to$ Document** loop, updates this board, and prepares the next session queue.

---

## 🎯 Active Session Workboard: Today

- **Session Date**: September 18, 2026
- **Session Objective**: PHANTOM v2 MD Blueprint full implementation ([`docs/specs/PHANTOM_V2_SPEC.md`](docs/specs/PHANTOM_V2_SPEC.md), ADR-016).
- **Hardware Profile**: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM) | 24 GB DDR5 System RAM | Zero Local Model Storage Policy.

### 📋 Today's Action Checklist

| Status | Task ID | Domain | Description | Artifact / Target |
|:---:|:---:|:---:|:---|:---|
| ✅ | `RES-DEEP` | Research | Deep research into physical memory bandwidth walls (197 GB/s vs 48 GB/s) and speculative decoding mathematical proof | [`docs/specs/PHANTOM_RESEARCH_REPORT.md`](file:///c:/Work/Projects/Solution%20is%20all%20You%20need/docs/specs/PHANTOM_RESEARCH_REPORT.md) |
| ✅ | `RES-EXP1` | Benchmark | Empirical AVX2 CPU GEMM(8) vs GEMV benchmark: proved 3.93× layer amortization factor | `scratch/bench_gemm_vs_gemv.py` |
| ✅ | `HYG-CLEAN`| Hygiene | Relocate root markdown specs to `docs/specs/` and purge accidental directories | `docs/specs/` |
| ✅ | `DOC-ALIGN`| Truthfulness | Demote synthetic benchmark claims across `benchmarks/run_all.py`, `README.md`, `PROGRESS.md`, and docs | `README.md` & `docs/` |
| ✅ | `PURGE-DEAD`| Refactoring | Execute Phase 1: Purge dead/synthetic subsystems (`wraith_lstm`, `spectral`, `neural_cache`, `core/`, `kernels/`) | Cleaned repo root & subpackages |
| ✅ | `SPEC-CORE` | Runtime | Execute Phase 2: Build `python/phantom/speculative/` runtime (`draft_runner`, `target_verifier`, `acceptance`, `engine`) | `python/phantom/speculative/` |
| ✅ | `SPEC-BENCH`| Benchmarks | Execute Phase 3: Build real speculative physical benchmarks with byte accounting | `benchmarks/speculative_benchmark.py` |
| ✅ | `SPEC-OFFLOAD`| Runtime & CLI | Implement `-ngl`, `--spec-draft`, and `--cpu-moe` hardware offloading with memory bandwidth microbenchmark | `phantom_cli.py`, `benchmarks/bench_memory_bandwidth.py` |
| ✅ | `V2-SPEC` | Docs | Canonical v2 spec + ADR-016; root MD pointers | `docs/specs/PHANTOM_V2_SPEC.md` |
| ✅ | `V2-EAGLE` | Runtime | EAGLE-3 heads + training pipeline | `python/phantom/speculative/eagle_heads.py` |
| ✅ | `V2-KERNEL` | Kernels | Fused attention + FFN + dispatch layer | `kernels/`, `python/phantom/kernels/` |
| ✅ | `V2-WRAITH` | Prefetch | Wraith v2 adaptive prefetch scheduler | `python/phantom/prefetch/wraith_v2.py` |
| ✅ | `V2-QUANT` | Quant/Sparsity | Selective Q3 + 40% adaptive sparsity | `python/phantom/quant/`, `python/phantom/sparsity/` |
| ✅ | `V2-CLI` | Integration | Live model loader + v2 CLI flags | `model_loader.py`, `phantom_cli.py` |
| ✅ | `V2-BENCH` | Benchmarks | v2 ablation + cross-hardware suite | `benchmarks/phantom_v2_benchmark.py` |
| ✅ | `V2-TEST` | Quality | 41 tests PASS + parity gates | `tests/unit/test_eagle_heads.py`, etc. |
| ✅ | `V2-COLAB` | Cloud | Colab v2 harness + MoE correlation + notebook Step 5 | `scripts/colab_v2_runner.py`, `test_16` |
| ✅ | `P0-REALPATH` | Runtime | Fix `cmd_run` AttributeError; real llama.cpp decode path; metrics (`ram_used_gb`, draft_model) | `phantom_cli.py`, `llamacpp_backend.py` |
| ✅ | `P0-SKIP-GUARD` | Benchmarks | Graceful `SKIPPED_LLAMA_CPP_NOT_INSTALLED` before weight download | `run_real.py`, `ephemeral_test_runner.py`, `test_runtime_metrics.py` |
| ✅ | `P1-PURGE-V2` | Refactoring | Delete model-free `phantom_v2_benchmark.py` + `v2_latest.json`; purge 2.5×/0.72 fabrication; honest dry-run/live-failure | `colab_v2_runner.py`, `generate_results.py`, `RESULTS.md` |
| ✅ | `P1-REWIRE` | Integration | `benchmark --v2` → subprocess `run_real.py --focus spec-decode` (fixed module resolution); EMPTY `spec_decode` → `SKIPPED` in `latest.json` | `phantom_cli.py`, `latest.json` |
| ✅ | `P1-DOCS` | Truthfulness | ADR-022/023; `test_16` VOIDED; `test_03` 24.79 removed; README Ollama-vs-PHANTOM table + no v2 achieved | `docs/`, `README.md` |
| ✅ | `P1-NOTEBOOK` | Cloud | Notebook Step 1 llama-cpp-python CUDA build; Step 5 real spec-decode sweep (EAGLE runner removed) | `notebooks/phantom_cloud_tester.ipynb` |
| ⬜ | `V2-COLAB-LIVE` | Cloud | **Live spec-decode run on Colab T4** (commit+push first, then notebook Step 5 LIVE): `qwen2.5-coder-32b` draft `qwen2.5-0.5b`, `--ngl ∈ {0,14,24}` × `--n-batch ∈ {256,512,1024}` | Pending user Colab session + results ingestion |


---

## 🔄 The Daily 5-Step Operational Loop

Whenever we sit down to work on PHANTOM, we follow this strict, reproducible loop:

```mermaid
flowchart LR
    A["1. Standup<br/>('start with today')"] --> B["2. Build<br/>(Surgical Implementation)"]
    B --> C["3. Debug & Test<br/>(Regression + Zero-Disk)"]
    C --> D["4. Auto-Log<br/>(Ledger, Changelog, Progress)"]
    D --> E["5. Handoff & Push<br/>(Commit & Next Queue)"]
```

### Step 1: Standup & Orientation (`"start with today"`)
1. Open [`docs/DAILY_WORKBOARD.md`](file:///c:/Work/Projects/Solution%20is%20all%20You%20need/docs/DAILY_WORKBOARD.md).
2. Check the **Next Session Queue** from the previous session.
3. Verify hardware constraints and active git branch (`git status`).
4. Lock in the primary objective for the day.

### Step 2: Surgical Build
1. Reference the architectural blueprint in [`docs/CONCEPT_MAP.md`](file:///c:/Work/Projects/Solution%20is%20all%20You%20need/docs/CONCEPT_MAP.md).
2. Follow simplicity principles: touch only what must be touched, zero unnecessary abstractions.
3. If making architectural trade-offs, log an entry in [`docs/DECISION_LOG.md`](file:///c:/Work/Projects/Solution%20is%20all%20You%20need/docs/DECISION_LOG.md).

### Step 3: Debug & Test (Zero-Disk First)
1. Run local unit tests: `python -m pytest tests/unit`
2. Run master platform audit: `python tests/audit_suite.py`
3. If running model inference, use the **Ephemeral Zero-Disk Runner**:
   ```bash
   python tests/ephemeral_test_runner.py --model smollm-135m --prompt "Explain MoE"
   ```
   *Never store raw weights on local disk.*

### Step 4: Auto-Log & Record
1. The test runner will automatically append results into [`docs/testing/INDEX.md`](file:///c:/Work/Projects/Solution%20is%20all%20You%20need/docs/testing/INDEX.md).
2. Document user-facing changes and internal fixes in [`docs/CHANGELOG.md`](file:///c:/Work/Projects/Solution%20is%20all%20You%20need/docs/CHANGELOG.md).
3. Update subsystem completion percentages in [`docs/PROGRESS.md`](file:///c:/Work/Projects/Solution%20is%20all%20You%20need/docs/PROGRESS.md).

### Step 5: Handoff & Git Push
1. Mark completed checklist items as `✅`.
2. Populate the **Next Session Queue** below with the exact items to tackle tomorrow.
3. Commit with semantic messages (`feat:`, `fix:`, `docs:`, `test:`) and push to GitHub.

---

## 📌 Next Session Queue (Tomorrow)

When starting the next session, here is our queued roadmap:

- [ ] **Colab live spec-decode run (v2 targets)**: After commit+push, open [`notebooks/phantom_cloud_tester.ipynb`](notebooks/phantom_cloud_tester.ipynb) Step 5, select **Live Cloud Inference**, run `qwen2.5-coder-32b` draft `qwen2.5-0.5b` with `--ngl ∈ {0,14,24}` × `--n-batch ∈ {256,512,1024}` sweeps; export `latest.json` (Step 4) and ingest.
- [ ] **Close the gap to v2 targets**: Baselines (2.88 / 12.95 / 3.63 tok/s) → targets (7-10 / 14-18 / 14). Levers to measure: `--no-mmap`, Q3_K_S, stronger drafts (qwen2.5-1.5b), batch tuning; declare achievement only with real `spec_decode` rows in `latest.json`.
- [ ] **Evaluate llama.cpp backend integration** for GGUF decode path (ADR-017 follow-up; native draft verified).
- [ ] **Residual §10 invariant hygiene**: replace hardcoded constants still shipping (`phantom_tui.py:2711-2716` 4.2 tok/s, `model_manager.py:110`, `claims_allowlist.yml`, stale docs `PHANTOMFILE.md`/`PLUGINS.md`/`CONCEPT_MAP.md`/`toolextensio.md`) with `latest.json` reads.
- [ ] **Resolve doc contradictions**: SmolLM 1000 vs 366.5 tok/s; NVMe 1.43 vs 1.75 vs ~4.5 GB/s; CUDA 12.6 vs 13.3; dense-32B ceiling vs 3.63 tok/s publish.
- [ ] **Fix or remove Triton stubs**: `kernels/attention/fused_attention.py` and `kernels/ffn/fused_ffn.py` call PyTorch fallback in all branches; `kernels/dispatch.py` arg-order bug (`use_triton`→`scale_q`/`scale_w1`).
- [ ] **Verify/delete** leftover `changelog_entry.md` temp file at repo root.
- [ ] **Regenerate `RESULTS.md`** from clean commit (fingerprint `b451e11b (DIRTY)`).
- [ ] **Auto-tune n_batch** in `benchmarks/run_real.py` for optimal CPU GEMM amortization per model.

---

## 🗂️ Project Documentation Directory Map

| Document | File Path | Primary Function |
|:---|:---|:---|
| **Daily Workboard** | [`docs/DAILY_WORKBOARD.md`](file:///c:/Work/Projects/Solution%20is%20all%20You%20need/docs/DAILY_WORKBOARD.md) | **Daily standup, session checklist, and tomorrow queue** |
| **Concept Map** | [`docs/CONCEPT_MAP.md`](file:///c:/Work/Projects/Solution%20is%20all%20You%20need/docs/CONCEPT_MAP.md) | North Star, architecture phases, and 4 evolutionary branches |
| **Progress Dashboard** | [`docs/PROGRESS.md`](file:///c:/Work/Projects/Solution%20is%20all%20You%20need/docs/PROGRESS.md) | Subsystem readiness matrix, tested models, and scorecards |
| **Changelog** | [`docs/CHANGELOG.md`](file:///c:/Work/Projects/Solution%20is%20all%20You%20need/docs/CHANGELOG.md) | Detailed version history, commit records, and kernel fixes |
| **Decision Log (ADR)** | [`docs/DECISION_LOG.md`](file:///c:/Work/Projects/Solution%20is%20all%20You%20need/docs/DECISION_LOG.md) | Architectural trade-offs, bug root causes, and design choices |
| **Testing Ledger** | [`docs/testing/INDEX.md`](file:///c:/Work/Projects/Solution%20is%20all%20You%20need/docs/testing/INDEX.md) | Auto-updated master test registry and hardware throughput matrix |
| **Test Report Template**| [`docs/testing/TEMPLATE_TEST_REPORT.md`](file:///c:/Work/Projects/Solution%20is%20all%20You%20need/docs/testing/TEMPLATE_TEST_REPORT.md) | Standardized report template for new model benchmarks |
| **Specifications Hub** | [`docs/specs/`](file:///c:/Work/Projects/Solution%20is%20all%20You%20need/docs/specs/) | Master prompts, engineering blueprints, and platform specs |

---

## 📜 Historical Session Archive

| Session Date | Lead Tasks | Key Milestones Achieved |
|:---|:---|:---|
| **2026-09-14** | Audit & Verification | Eliminated all mock files; executed real non-synthetic test of Qwen2.5-Coder-32B; proved 4.88× parameter ceiling lift (6GB VRAM); verified pure-CPU SIMD fallback; implemented zero-disk ephemeral streaming harness. |
| **2026-09-15** | Repository Structure | Restructured documentation system: established Concept Map, Progress Dashboard, Changelog, ADR Decision Log, auto-updating Testing Ledger, and Daily Workboard. |
| **2026-09-15** | Ground Truth Remediation | Completed PHANTOM Ground Truth Remediation Brief across Phases 0–7: solved 32B PCIe paradox (10 KB activation copy + in-place DDR5 SIMD), built atomic byte counter & hardware fingerprinting, rewrote benchmark suite (N>=10, ablations, `latest.json`), deleted `audit.md`, generated canonical `RESULTS.md`, updated `README.md` & `ARCHITECTURE.md`, enforced 100% PASS `scripts/check_claims.py` CI gate, authored `REPRODUCING.md`, `CONTRIBUTING.md`, `SECURITY.md`. |
| **2026-09-18** | v2 Colab Harness | Built `scripts/colab_v2_runner.py`, MoE prefetch correlation module, notebook Step 5, merged v2 into `RESULTS.md`; dry-run PASS (`test_16`). |
| **2026-09-18** | Full-Repo Analysis (read-only) | Audited all 166 source/doc files (~28k lines). Confirmed real hardware layer (Ollama 32B audit, bandwidth/GEMM/NVMe measurements, acceptance + live draft/verify). Flagged fabricated-verdict layer: `ephemeral_test_runner` & `colab_runner` unconditional PASS, `run_all.py` sine-wave + broken `test_needle_haystack` import, dry-run published as "Colab Live VERIFIED", EAGLE on synthetic data, Triton stub kernels, hardcoded 4.2 tok/s metric family. Recorded in TODAY.md queue. |
| **2026-09-18** | Truth-First Remediation & llama.cpp-First Backend | **Phase 0-4 Complete**: Purged fabrication layer (ephemeral_test_runner, colab_runner, run_all.py, test_reference_parity); adopted llama.cpp-first hybrid backend (ADR-017); implemented real measurement gates (run_real.py, CI); built MoE expert-aware prefetch (ADR-021) with OS-level I/O hints; exposed n_batch for CPU GEMM amortization (ADR-020); added ADR-017 through ADR-021; all 38 unit tests PASS; parity gate PASS. |
| **2026-09-18/19** | Physics-Honest v2 — Real Spec-Decode Path (Phases 0-1) | Fixed `cmd_run` AttributeError (real llama.cpp path now executes); llama-cpp draft metrics + graceful SKIPPED guard; `run_real.py --focus spec-decode` + sweeps CLI; deleted model-free `phantom_v2_benchmark.py` + fabricated `v2_latest.json`; `colab_v2_runner.py` no longer fabricates (ADR-022/023); `benchmark --v2` rewired; RESULTS.md section 6 honest SKIPPED; `test_16` VOIDED + `test_03` 24.79 removed; notebook Step 1/5 updated; **43/43 unit tests PASS**; parity PASS; end-to-end CLI→SKIPPED→latest.json verified. |
