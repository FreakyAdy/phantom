# PHANTOM — Daily Mission Workboard & Session Protocol

> **Daily Operation Rule**: Whenever the user says *"start with today"*, the agent immediately reads this workboard, aligns with today's prioritized agenda, executes the **Build $\to$ Debug $\to$ Test $\to$ Document** loop, updates this board, and prepares the next session queue.

---

## 🎯 Active Session Workboard: Today

- **Session Date**: September 17, 2026
- **Session Objective**: Deep Research Investigation & Speculative Runtime Pivot ([`docs/specs/PHANTOM_RESEARCH_REPORT.md`](file:///c:/Work/Projects/Solution%20is%20all%20You%20need/docs/specs/PHANTOM_RESEARCH_REPORT.md) & [`docs/specs/PHANTOM_RESEARCH_SPEC.md`](file:///c:/Work/Projects/Solution%20is%20all%20You%20need/docs/specs/PHANTOM_RESEARCH_SPEC.md)).
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

- [ ] **Configure Self-Hosted GPU Runner for Automated Nightly CI**:
  Connect RTX 4050 runner to GitHub Actions with label `self-hosted-gpu` to execute nightly runs of `benchmarks/run_all.py` and commit fresh `latest.json` archives.
- [ ] **Implement Linux Direct `io_uring` Kernel**:
  Port `phantom_pages` NVMe tile loader from multi-threaded pread to Linux asynchronous `io_uring` SQE/CQE ring buffer for lower latency tile dispatch.
- [ ] **Ultra-Long Multi-Needle Stress Testing (>64K Tokens)**:
  Extend NIAH harness to 64K and 128K context tokens with multi-needle associative retrieval to characterize upper bounds of 8.0x Neural Cache capacity.

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
