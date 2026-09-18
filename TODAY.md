# PHANTOM — Today's Daily Workboard

> **Quick Access Pointer**: For the full daily operational loop, see [`docs/DAILY_WORKBOARD.md`](docs/DAILY_WORKBOARD.md)

---

### Quick Session Summary (Today: September 18-19, 2026)

- **Session Focus**: Physics-honest v2 targets — llama.cpp native draft spec-decode (ADR-022/023). All fabricated v2 artifacts purged; real measurement path wired and verified.
- **Phase 0 — Real-path enablement (done)**:
  - [x] Fixed `phantom_cli.py` `cmd_run` AttributeError (`engine.metrics` → `engine.get_metrics()`) that sent every run to the simulated stack
  - [x] `llamacpp_backend.py`: `ram_used_gb` real measurement, `draft_model`/`draft_model_id` metrics, `draft_model_id` engine param
  - [x] Graceful `SKIPPED_LLAMA_CPP_NOT_INSTALLED` in `run_real.py` + `ephemeral_test_runner.py` (never downloads ~20 GB weights when backend absent)
  - [x] `run_real.py --focus spec-decode`: baseline vs draft real benchmark, sweeps (`--ngl`, `--n-batch`), new CLI args
  - [x] `tests/unit/test_runtime_metrics.py` (5 tests); **43/43 unit tests PASS**; parity gate PASS
- **Phase 1 — Fabrication purge (done)**:
  - [x] Deleted `benchmarks/phantom_v2_benchmark.py` and `benchmarks/results/v2_latest.json`
  - [x] `colab_v2_runner.py`: removed 2.5× multiplier / 0.72 acceptance; dry-run → `SIMULATED_DISABLED_MEASUREMENT` (tok/s=None), live failure → `live_measurement_failed`
  - [x] `phantom_cli.py benchmark --v2` rewired to subprocess `run_real.py --focus spec-decode` (fixed ModuleNotFoundError via `__file__`-based repo root)
  - [x] `generate_results.py` v2 section renders only real spec-decode from `latest.json`; `RESULTS.md` section 6 = honest `SKIPPED_LLAMA_CPP_NOT_INSTALLED`
  - [x] README: "PHANTOM vs Ollama (Direct Comparison on RTX 4050)" table + v2-targets section flagged DRY-RUN SIMULATIONS / not achieved
  - [x] Docs: ADR-022 + ADR-023 in DECISION_LOG; `test_16` VOIDED + INDEX row updated; `test_03` 24.79 cloud figure removed
  - [x] Notebook Step 1 builds llama-cpp-python (CUDA); Step 5 replaced EAGLE runner with `run_real.py --focus spec-decode`
  - [x] Verified end-to-end: `benchmark qwen2.5-coder-32b --v2` → subprocess → graceful SKIPPED → `latest.json` + archive written
- **Current Focus / Next Queue**:
  - [ ] **Colab live spec-decode run**: commit+push (ask user first), then notebook Step 5 LIVE on T4: `qwen2.5-coder-32b`, draft `qwen2.5-0.5b`, sweeps `--ngl ∈ {0,14,24}` / `--n-batch ∈ {256,512,1024}`; export `latest.json` via Step 4
  - [ ] Ingest real numbers → close gap to targets (levers: Q3_K_S, stronger drafts, `--no-mmap`, batch tuning), then declare achievement in RESULTS/README
  - [ ] Residual hygiene from audit: `phantom_tui.py:2711-2716`, `model_manager.py:110`, `claims_allowlist.yml`, stale docs `PHANTOMFILE.md`/`PLUGINS.md`/`CONCEPT_MAP.md`/`toolextensio.md`
  - [ ] Delete leftover `changelog_entry.md` temp file (verify)
- **Full-Repo Analysis Session (2026-09-18, read-only audit)**: Audited all 166 files (~28k lines). Identified fabricated layer (dry-run stamped VERIFIED, LCG EAGLE training, sine-wave benchmarks, hardcoded 4.2 tok/s family, Triton stubs) — Phase 0/1 above surgically removed the v2 subset; full cleanup continues.
