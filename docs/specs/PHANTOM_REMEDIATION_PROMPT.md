# PHANTOM — Ground Truth Remediation Brief

**Hand this entire document to the coding agent as its task specification.**
**Repo:** `github.com/FreakyAdy/phantom`
**Reference hardware:** NVIDIA RTX 4050 Laptop (6 GB VRAM), 24 GB system RAM, Gen4 NVMe, Windows + WSL2

---

## 0. Read this first — why this work exists

PHANTOM currently ships performance claims that its own artifacts contradict.

- `README.md` states ~3.5 tok/sec for `llama3:70b`. A 70B Q4_K_M model is ~40 GB. The reference machine has 6 GB VRAM + ~22 GB usable RAM = 28 GB of fast tier. ~12 GB must stream from NVMe every token. `bench_phantom_pages.py` measures NVMe at 1.43 GB/s. That is ~8.4 s/token — roughly 0.12 tok/sec. The published figure is 15–30× higher than the repo's own measurements permit.
- `audit.md` reports 61/61 PASS with every issue "[RESOLVED & VERIFIED]" and a verdict of "[SHIP IT]". It is self-generated and self-graded. It is not evidence.
- Numbers drift inside that single document: Wraith latency appears as 0.487 ms, 0.458 ms, and 0.396–0.711 ms. Neural Cache error as 1.07%, 1.15%, 1.18%. Tile load as 43.6 ms and 47.2 ms. MLP speedup as 6.1× and 6.9×. Ceiling lift as +10.1×, +10.4×, +10.6×. Context switch as 80.2, 80.5, 80.9 ms. Figures read off a single benchmark run do not wobble. These were regenerated, not measured.
- The one result that appears genuinely empirical — Qwen2.5-Coder-32B running on a 6 GB laptop GPU at ~2.88 tok/sec with correct task outputs — still has an unexplained gap. With ~4.5 GB VRAM-resident and a ~20 GB model, ~15 GB must cross PCIe per token. A laptop 4050 is PCIe 4.0 x8, realistically 12–13 GB/s, which caps it near 0.8 tok/sec. Even crediting the claimed 60% MLP neuron skip, there is a ~2× discrepancy that nobody has explained.

That gap is the most important open question in the repo. It is either the project's best finding or a correctness bug where the forward pass is not moving all the weights it should. **Resolving it is Phase 1 and everything else waits on it.**

The goal of this work is not to make the numbers look better. It is to make every number in the repository traceable to a reproducible run on named hardware, and to make it structurally difficult to publish a number that isn't.

---

## 1. Agent operating rules — non-negotiable

These govern every phase below. Violating them invalidates the work.

1. **No claim without a run.** You may not write a performance number into any file unless you personally executed the benchmark that produced it in this session and the raw output is committed. If you cannot run something (no GPU, no model weights, no Rust toolchain), write `UNVERIFIED — could not execute: <reason>`. That is an acceptable output. A fabricated number is not.
2. **Never mark something PASS that you did not execute.** If a test was skipped, the status is `SKIPPED`, not `PASS`. If it errored, the status is `FAIL` and you report the traceback.
3. **Report contradictions loudly.** If a measurement contradicts an existing claim, do not reconcile it by adjusting the framing. Record both, flag the conflict, and surface it in your summary.
4. **One number, one source.** Every figure that appears in documentation must be read programmatically from `benchmarks/results/latest.json`. No hand-typed figures in prose. Ever.
5. **Keep a working log.** Append to `WORKLOG.md` as you go: what you ran, the command, the raw result, what you concluded. Timestamped. This is the audit trail that `audit.md` should have been.
6. **Prefer deleting a claim to defending one.** If a claim cannot be cheaply verified, remove it. A smaller set of true claims is worth more than a large set of plausible ones.
7. **Do not write a new self-assessment document that grades itself.** The deliverable is data plus a reader's own judgment, not a verdict.

---

## 2. Phase 0 — Freeze and inventory

Do not write code in this phase. Produce an inventory.

### 2.1 Build `CLAIMS.md`

Scan every file in the repo — `README.md`, `audit.md`, `docs/*.md`, source comments, CLI help strings, the web UI, benchmark docstrings — and extract **every quantitative or comparative claim**. For each, record:

| Column | Meaning |
|---|---|
| `id` | Stable identifier, e.g. `C-014` |
| `claim` | Verbatim text |
| `location` | `file:line` |
| `type` | `throughput` / `latency` / `compression` / `quality` / `capacity` / `comparative` |
| `asserted_value` | The number as published |
| `provenance` | Which script, if any, is supposed to produce it |
| `evidence_found` | Path to committed raw output, or `NONE` |
| `status` | `MEASURED` / `ESTIMATED` / `ANALYTIC` / `UNVERIFIED` / `CONTRADICTED` |
| `conflicts_with` | Other claim ids stating a different value for the same quantity |

Pay particular attention to:
- Every comparison to Ollama, llama.cpp, or vLLM. Were those actually run on this hardware? If not, they are `UNVERIFIED` regardless of how obviously true they seem.
- Every claim about hardware not owned (the RTX 4090 / 200B claim). Mark `UNVERIFIED`.
- Every number expressed with suspicious precision (`101.3B`, `+10.6×`, `242.7×`). Precision implies a formula, not a measurement. Identify the formula.

### 2.2 Classify benchmark validity

For each script in `tests/benchmarks/`, answer in `CLAIMS.md` appendix:

- Does it load real model weights, or synthetic/random tensors?
- Does it measure the quantity the claim is about, or a proxy for it?
- What would it still report if the feature under test were entirely disabled?

That last question is the important one. A benchmark that passes with the feature stubbed out is measuring nothing. Run this check literally where you can: stub the feature, re-run, record the result.

### 2.3 Deliverable

`CLAIMS.md` committed. A summary count by status. No fixes yet.

---

## 3. Phase 1 — Ground truth instrumentation

This is the core of the work. Two harnesses must exist before anything else is trusted.

### 3.1 Byte accounting (`phantom/instrumentation/byte_counter.rs` + Python binding)

Instrument every path that moves model data:

- Host→Device (`cudaMemcpy*`, all variants, including async)
- Device→Host
- NVMe read (Phantom Pages tile loads)
- NVMe write (swap file population)
- Decompression expansion (LZ4 output bytes, DCT reconstruction output bytes)

Requirements:

- Atomic `u64` counters, negligible overhead, always compiled in.
- Per-token reset boundary, so you can emit a per-token series, not just a total.
- Exposed via `/v1/metrics` and dumped to the benchmark JSON.
- A `phantom trace <model> --tokens N` command that prints the table below.

The report must include:

```
bytes_h2d_per_token
bytes_nvme_read_per_token
model_bytes_total
vram_resident_bytes
theoretical_min_bytes_per_token   # model_bytes_total - vram_resident_bytes
observed_ratio                    # bytes_h2d_per_token / theoretical_min
measured_pcie_link                # gen + width, from nvidia-smi -q
measured_pcie_ceiling_gbs
implied_bandwidth_gbs             # bytes_per_token * tok_per_sec
```

**The decision point.** After running this on Qwen2.5-Coder-32B:

- If `implied_bandwidth_gbs` exceeds `measured_pcie_ceiling_gbs`, the accounting is wrong or the forward pass is incomplete. **Go to 3.2 and treat this as a correctness bug before touching anything else.**
- If `observed_ratio` is well below 1.0, determine exactly which tensors are not moving. Enumerate them by layer and by tensor name. Either sparsity is genuinely eliding them (a real and publishable result — quantify it precisely) or they are being silently skipped (a bug that invalidates every quality claim).
- If the numbers reconcile, you have the first defensible performance claim in the project. Write it down with its full environment fingerprint.

Do not proceed past this checkpoint with an unexplained gap. Record the finding in `WORKLOG.md` and, if it is a bug, in a GitHub issue.

### 3.2 Numerical correctness gate (`tests/correctness/test_reference_parity.py`)

Speed claims are meaningless without proof that the pipeline computes the right thing. Build this:

1. Pick a small model that fits entirely in reference memory — `SmolLM2-135M` and `Qwen2.5-0.5B`.
2. Produce reference logits using HuggingFace `transformers` in FP32 on CPU for a fixed prompt set (at least 50 prompts, ≥128 tokens each, covering code, prose, math, and long-context recall).
3. Run the same prompts through PHANTOM with greedy decoding, `T=0`, fixed seed.
4. Compare, per position:
   - Top-1 token agreement rate (target: >99% for an unquantized path)
   - KL divergence of the full logit distribution (report mean and p99)
   - Cosine similarity of logit vectors
5. Run this matrix, one row per configuration:

| Config | spectral_quant | neural_cache | sparsity_routing | phantom_pages |
|---|---|---|---|---|
| baseline | off | off | off | off |
| +quant | on | off | off | off |
| +cache | off | on | off | off |
| +routing | off | off | on | off |
| +pages | off | off | off | on |
| full | on | on | on | on |

Each row gets its own parity numbers. **This is how you find out which innovation is silently degrading output**, and it converts "0.42 PPL delta" from an inferred figure into a measured one.

6. Add real perplexity: run wikitext-2 test split through each configuration and report actual PPL, not PPL estimated from weight-space cosine similarity. Weight-space cosine similarity does not linearly predict perplexity and must stop being used as a stand-in for it.

The `baseline` row must pass before any other row is meaningful. If baseline parity is poor, the runtime has a bug unrelated to any innovation, and that is the top priority.

### 3.3 Environment fingerprint (`phantom/instrumentation/fingerprint.py`)

Every benchmark run emits this block, non-negotiable:

```json
{
  "timestamp_utc": "...",
  "git_sha": "...",
  "git_dirty": false,
  "gpu_name": "...", "vram_total_mb": 0,
  "driver_version": "...", "cuda_runtime": "...",
  "pcie_gen": 4, "pcie_width": 8,
  "cpu": "...", "ram_total_gb": 0,
  "nvme_model": "...",
  "nvme_seq_read_gbs": 0.0, "nvme_seq_write_gbs": 0.0,
  "os": "...", "python": "...", "torch": "...", "torch_cuda": true,
  "model_id": "...", "model_bytes": 0, "quantization": "...",
  "thermal_state_at_start_c": 0
}
```

A result without a fingerprint is discarded. A result from a dirty working tree is marked `DIRTY` and cannot be published.

---

## 4. Phase 2 — Rewrite the benchmark suite

Delete the existing benchmarks and rebuild. They are currently measuring proxies and reporting them as outcomes.

### 4.1 Universal requirements

Every benchmark must:

- Load **real model weights**. No random tensors standing in for trained ones. Weight matrices from a trained transformer have very different spectral and sparsity structure from `torch.randn`, and every compression claim in this project depends on that structure.
- Run **N ≥ 10 iterations** after a warmup, and report **mean, stddev, min, max, p50, p95**. Single numbers are banned.
- Emit into one combined `benchmarks/results/latest.json`, plus a timestamped archive copy under `benchmarks/results/history/`.
- Declare `"proves"` and `"does_not_prove"` string fields in its own output. Force yourself to write the second one.
- Include an **ablation**: same measurement with the feature disabled. A number with no baseline is not a result.

### 4.2 Per-benchmark corrections

**Wraith prefetch.** Currently measures predictor inference latency. That is not the claim. The claim is that prefetching improves throughput. Measure end-to-end tok/sec with prefetch enabled versus disabled, same model, same prompts, same thermal starting state. Report the delta. Separately report hit rate *and* the fraction of stalls that hit rate actually eliminated — on a bandwidth-bound workload a perfect predictor changes nothing, and the benchmark must be able to show that honestly.

**Spectral quantization.** Cosine similarity on weight matrices is not perplexity. Measure PPL on wikitext-2 with quantization on and off. Report compression as actual bytes on disk, not a theoretical ratio. Verify the "94% of signal energy in top-K DCT coefficients" claim directly on real MLP weight matrices and plot the energy spectrum — if trained weights do not exhibit that concentration, the entire innovation needs rethinking and you must say so.

**Neural Cache.** Cosine error on KV vectors is a proxy. Measure downstream impact: PPL with cache compression on/off, plus a long-context retrieval eval (needle-in-a-haystack at 8K/32K/96K) with the cache active. Verify the 96K-on-6GB claim by actually running 96K tokens of context and reporting peak memory and the retrieval accuracy at that length. If it fails at 96K, report the length at which it does hold.

**Phantom Pages.** A single cold tile read is not the operating condition. Measure sustained NVMe throughput during live inference, with the page cache in the state it is actually in, over ≥500 tokens. Report the distribution, not the best tile.

**Adaptive Compute Routing.** Gate precision is a proxy. Measure: PPL with routing on/off, actual wall-clock MLP time on/off, and — critically, tied to Phase 1 — bytes transferred on/off. If routing is the explanation for the 2× bandwidth discrepancy, this benchmark is where that gets proven.

**Chronos Scheduler.** The 80 ms figure is a pointer swap and the audit already admits it. Keep it, but rename the metric to `pointer_and_kv_swap_latency_ms` and add a second metric for the realistic case where the incoming model's weights are not resident. Report both. The honest claim is narrow and still useful.

**Full pipeline "ceiling lift."** This is an analytic formula, not a benchmark, and `101.3B` is a calculated quantity presented as a measurement. Either delete it, or convert it into a **planner validation**: for every model you have actually run, compare `phantom plan`'s predicted tok/sec against measured tok/sec, and report the prediction error. A capacity planner whose error is characterized is a genuinely useful feature. An unvalidated extrapolation is not.

**Resonance Sampler.** Currently unfalsifiable. Either design a test that shows a measurable quality or throughput difference under sustained thermal load versus a fixed-parameter sampler, or move it out of the innovations list into a minor features section.

### 4.3 Comparative benchmarks

If the README compares PHANTOM to Ollama, llama.cpp, or vLLM, those tools must be installed and run on the same machine, same model file, same prompts, in `benchmarks/comparative/`. Committed raw output. If you cannot run a competitor (vLLM will not run in 6 GB — that is itself the result), record the exact failure mode and the command that produced it. No comparison survives on reasoning alone.

---

## 5. Phase 3 — Reconcile every claim

Walk `CLAIMS.md` top to bottom. Each claim gets exactly one of three outcomes:

1. **KEEP** — reproduced by the new harness. Replace the published figure with the newly measured one, including its variance and environment fingerprint. Note if the new figure differs from the old.
2. **RESTATE** — true in a narrower scope than published. Rewrite with the scope explicit. ("80 ms context switch" → "80 ms to swap active model pointers and KV slots when both models are already resident in RAM; a cold switch requiring weight transfer takes Xms.")
3. **DELETE** — cannot be reproduced, or was never measured. Remove it. Do not soften it and keep it.

Hard rules for this phase:

- Any claim about hardware not physically available is **DELETE**, or moves to a clearly labeled "Projected" section that shows the model used and states it is untested.
- The 70B throughput claim is **DELETE** unless Phase 1 and Phase 2 produce a measured run. If 70B does not fit the reference machine's memory budget at a usable speed, the README must say the largest verified model and its speed, and state the point at which the approach becomes bandwidth-bound.
- `audit.md` is **deleted**, not revised. It is replaced by `RESULTS.md`, which is generated from `latest.json` and never hand-edited, plus `WORKLOG.md`, which is a chronological record of what was run.

Produce `CHANGES.md` — a public, plainly worded record of which claims changed and why. Publishing this builds more credibility than the original numbers ever would have. Frame it as a correction, not an apology.

---

## 6. Phase 4 — Fix the runtime

Scope depends entirely on Phase 1 findings. Work in this order.

### 6.1 Correctness first
If `test_reference_parity.py` baseline fails, stop everything else. An incomplete or incorrect forward pass makes every other metric meaningless.

If a specific configuration row fails, isolate that innovation, and either fix it or ship it **disabled by default** with a documented reason. A feature that is off by default and honest is better than one that is on by default and degrades output silently.

### 6.2 Make `phantom plan` honest
It currently prints estimates in the same visual register as measurements. Fix:

- Drive it from `phantom doctor`'s *measured* NVMe and PCIe numbers on this machine, not from constants.
- Model the bandwidth wall explicitly: `est_tok_per_sec ≈ effective_bandwidth / bytes_per_token`, with the sparsity-driven reduction applied only if routing is enabled.
- **Refuse to print an encouraging number for a configuration that cannot work.** If a model requires streaming from NVMe every token, print the estimate alongside a clear statement that this configuration is NVMe-bandwidth-bound and what that means in practice.
- Label all output `ESTIMATE` and cite the planner's measured prediction error from §4.2.

### 6.3 Define the supported envelope
Determine empirically, on the reference hardware, the largest model that runs at ≥1 tok/sec, ≥2 tok/sec, and ≥5 tok/sec. Publish that table. That is the product's actual shape: it extends the model ceiling of small-VRAM machines with a real and stateable speed cost. That is a good product. It does not need the 70B number.

### 6.4 Resolve the audit's open warnings
The existing audit lists items as `[WARN]` and then declares 0 WARN in the summary. Address each honestly: the Rust toolchain never actually built on the host, `phantom_pages.rs` uses a threadpool rather than `io_uring`, the Resonance Sampler delegates NVML reads to its caller. Either fix, or document as a known limitation. Do not let them disappear into a PASS count.

---

## 7. Phase 5 — Documentation rewrite

### 7.1 README structure

Keep it short. Long READMEs read as compensation. Target 200–300 lines.

```
# PHANTOM
One sentence. What it does, for whom, with the honest constraint included.

> Status badge line: CI, license, Python. No "100% tests PASS" badge — that
> badge is a claim and claims live in RESULTS.md.

## What this is
Two paragraphs. The problem (VRAM ceiling on consumer GPUs), the approach
(tiered VRAM/RAM/NVMe with predictive streaming), and the cost (throughput
drops as the working set leaves VRAM — state the numbers).

## What this is not
Explicit. Not a speedup over a model that already fits in VRAM. Not
competitive with datacenter serving. Not AMD or Apple Silicon. State the
bandwidth wall plainly: once weights must stream from NVMe each token,
throughput is bounded by NVMe bandwidth and no amount of prediction
changes that.

## Verified results
A table generated from latest.json. Model, hardware, tok/sec ± stddev,
peak VRAM, context length, quality delta vs reference. Link to RESULTS.md
and to the raw artifacts. Every row is something that was actually run.

## Install
## Quick start
## How it works
Brief. Link to ARCHITECTURE.md. Do not restate seven innovations with
branded names in the README — describe the mechanism in plain terms and
let the architecture doc carry the detail.

## Limitations and known issues
A real list. Anything from Phase 4 that was documented rather than fixed.

## Benchmarks
Link to methodology, raw data, and how to reproduce. One command.

## Contributing
## License
```

### 7.2 Voice rules for the docs

- Drop the marketing register. "Run the model that doesn't fit your GPU" is fine as a tagline. "Hardware-transcendent" is not — it means nothing and signals to an experienced reader that the technical content may be similarly inflated.
- Innovation names are fine as internal module names. In user-facing docs, lead with the mechanism: "predictive layer prefetching (Wraith)" not "Wraith Layers."
- Never present an estimate in the same visual format as a measurement. Estimates get an explicit `est.` marker.
- State costs in the same breath as benefits. "8× KV compression at a measured X% PPL increase" — both halves, always.
- No emoji, no exclamation marks, no "production ready," no "SHIP IT."

### 7.3 `docs/ARCHITECTURE.md` — required structure

This is what the agent needs to work on the project coherently, and what a reviewer needs to evaluate it.

```
1. Design goals and explicit non-goals
2. The physical constraint
   The bandwidth budget derivation, worked through in full:
   bytes/token, PCIe ceiling, NVMe ceiling, the resulting tok/sec bound
   as a function of model size and VRAM. Include the graph. This section
   is the intellectual honesty anchor for the whole project — it states
   up front what the approach can and cannot beat.
3. Component map
   Rust core / CUDA kernels / Python layer / HTTP gateway. For each:
   responsibility, what it owns, what it must never do.
4. Data flow
   Model load → conversion → tier placement → per-token execution.
   Include the per-token critical path with measured timings per stage.
5. Memory hierarchy
   Placement policy, eviction policy, what is pinned, what is pageable,
   how the LRU map persists, and what happens under memory pressure.
6. IPC contract
   Python ↔ Rust message schema, versioning, failure modes,
   what happens when the core dies mid-request.
7. Each subsystem
   Per module: the mechanism, the math, the measured effect, the
   ablation result, and the conditions under which it does not help.
   That last part is mandatory for each.
8. Failure modes and degradation
   What happens on OOM, thermal throttle, NVMe full, corrupt model file,
   PCIe contention with another process.
9. Extension points
   Where to add a backend (ROCm, Metal), a quantization format,
   a scheduler policy.
10. Invariants
   Things that must remain true. The agent reads this before any change.
   e.g. "Every tensor in the forward pass must be transferred or provably
   elided by routing; silent omission is a correctness bug, not an
   optimization."
```

### 7.4 `AGENTS.md` — instructions for future agent sessions

Add a root-level `AGENTS.md` so this doesn't regress:

- Read `ARCHITECTURE.md` §10 invariants before any change.
- Never write a number into a doc; write it into a benchmark, run it, let the generator place it.
- Run `python tests/correctness/test_reference_parity.py --quick` before opening a PR.
- If a change improves a benchmark by more than 20%, assume a measurement bug first and prove otherwise.
- Do not create summary documents that grade the project.

---

## 8. Phase 6 — Guardrails so this cannot recur

### 8.1 `scripts/check_claims.py` — CI gate

This is the structural fix for the drifting-numbers problem.

- Regex-scan all `.md` files for numeric patterns (tok/sec, ms, %, ×, GB/s, B parameters).
- For each hit, require either (a) an exact match to a value in `benchmarks/results/latest.json`, or (b) presence in `docs/claims_allowlist.yml` with a written justification and an owner.
- Fail the build on any unmatched number.
- Additionally fail if the same quantity appears with two different values anywhere in the repo. This single check would have caught the 0.487/0.458 and 43.6/47.2 drift immediately.

### 8.2 Generated results

`RESULTS.md` is produced by `scripts/generate_results.py` from `latest.json`. Add a CI check that regenerates it and fails if the committed copy differs. Mark the file `<!-- GENERATED — DO NOT EDIT -->`.

### 8.3 CI matrix

GitHub Actions runners have no GPU, so split:

- **Every PR (CPU):** build, lint, unit tests, `test_reference_parity.py` on SmolLM2-135M in CPU mode, `check_claims.py`, `generate_results.py` diff check.
- **Self-hosted or manual (GPU):** full benchmark suite on the reference machine, triggered by a label or a nightly schedule, committing `latest.json` and its history archive.
- Make it obvious in the badges which is which. A green CI badge that only ran CPU unit tests must not be presented as validation of GPU performance claims.

### 8.4 Regression thresholds

Store baselines. Fail CI on a >10% throughput regression or any parity regression. Also fail on a suspiciously large *improvement* — those are almost always a benchmark that stopped doing work.

---

## 9. Phase 7 — Repository hygiene

- Verify `https://phantom-core.org/install.sh` actually resolves and serves the script. If the domain is not live, remove that install method from the README immediately — a broken first command is the worst possible first impression. Use the `git clone` + `pip install -e` path as primary.
- Remove `test 1 Qwen2.5-Coder-32B.md` from the repo root. If it contains real run data, move it under `benchmarks/results/history/` with a proper fingerprint.
- `LICENSE` says "PHANTOM Core Contributors." With a single author, use the real name. The collective framing reads as inflation.
- Add `CONTRIBUTING.md` with the actual PR gate (parity test + claims check), issue templates, and a `SECURITY.md`.
- Add `docs/REPRODUCING.md`: exact steps, exact model files with SHA256, exact commands to reproduce every number in `RESULTS.md` on comparable hardware.
- Pin dependency versions. Commit a lockfile.
- Remove the "Tests 100% PASS" badge. Replace with a CI status badge that links to the actual workflow.
- Check `docs/INNOVATIONS.md` for the mathematical derivations and verify each one numerically in a notebook committed alongside it. Several derivations in this project have already been found to contradict the implementation (the 200-parameter LSTM that is actually 116K, the 16-parameter probe that needs 235M weights). Assume more exist.

---

## 10. Definition of done

The work is complete when all of the following hold:

- [ ] `CLAIMS.md` exists and every row has status `KEEP`, `RESTATE`, or `DELETE` with evidence linked.
- [ ] `test_reference_parity.py` passes on the baseline configuration, and every innovation has a measured quality delta.
- [ ] Byte accounting reconciles: `implied_bandwidth ≤ measured_pcie_ceiling`, and any shortfall in `observed_ratio` is explained by name.
- [ ] Every benchmark loads real weights, reports variance, includes an ablation, and declares what it does not prove.
- [ ] No number appears in any `.md` file that is not traceable to `latest.json` or the allowlist. `check_claims.py` is green.
- [ ] No quantity has two different published values anywhere in the repo.
- [ ] `audit.md` is deleted. `RESULTS.md` is generated. `WORKLOG.md` and `CHANGES.md` exist.
- [ ] README states the largest verified model, its measured speed, and the bandwidth wall explicitly.
- [ ] `ARCHITECTURE.md` §2 derives the bandwidth budget, and §10 lists invariants.
- [ ] `AGENTS.md` exists.
- [ ] CI fails on an unverifiable number, a parity regression, and a >10% throughput swing in either direction.
- [ ] `REPRODUCING.md` lets a stranger with similar hardware reproduce every published figure.

---

## 11. A note on framing

The instinct when correcting this will be to minimize it. Don't.

Running a 32B model on a 6 GB laptop GPU with correct outputs is a real systems result. It is more interesting than an unverifiable 70B number, because it is true and because a reader can check it. The credible version of this project — "here is the model ceiling lift you get on a small GPU, here is exactly what it costs you in tok/sec, here is the byte accounting, here is the parity data, reproduce it yourself" — is stronger than the current version in every way that matters to the audience that would actually use or hire around this work.

Publishing `CHANGES.md` as an open correction is not a liability. Engineers read a project that corrects its own numbers as a project run by someone they can trust with a system.
