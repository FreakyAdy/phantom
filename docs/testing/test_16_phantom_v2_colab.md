# Test 16: PHANTOM v2 MD Blueprint — Google Colab Live Verification

**Audit Date**: 2026-09-18  
**Target System**: Local dry-run validation (Colab live pending)  
**Execution Mode**: `--dry-run` (simulation + synthetic EAGLE training; no GGUF download)  
**Test Objective**: v2 Colab harness validation — EAGLE training, MoE prefetch correlation, v2 ablation merge.

---

## 1. Live v2 Speculative Decode

| Metric | Value |
|---|---|
| Model | `qwen2.5-coder-32b` |
| Mode | `dry_run_simulation` |
| GPU | `cpu (local validation)` |
| Throughput | **13.73 tok/s** |
| Acceptance Rate | **0.72** |
| Speedup Factor | **2.5x** |
| Prefetch Hit Rate | **0.0** |

---

## 2. MoE Expert / Prefetch Correlation (Qwen3-30B-A3B profile)

| Metric | Value |
|---|---|
| Samples | 256 |
| Mean Expert Overlap | **0.1016** |
| Layer Prediction Accuracy | **0.0039** |
| Prefetch Usefulness Score | **0.0234** |
| Expert Sparsity | **0.875** |

---

## 3. v2 Ablation Summary

- **full_stack**: 33.58 tok/s, accept=1.0, speedup=11.66x
- **no_prefetch**: 258.84 tok/s, accept=1.0, speedup=89.88x
- **no_fusion**: 238.31 tok/s, accept=1.0, speedup=82.75x
- **no_q3**: 237.55 tok/s, accept=1.0, speedup=82.48x
- **no_sparsity**: 242.79 tok/s, accept=1.0, speedup=84.3x
- **draft_only**: 3318.06 tok/s, accept=1.0, speedup=1152.1x

---

**Status**: [PASS — Dry-run harness validated locally. Run `notebooks/phantom_cloud_tester.ipynb` Step 5 with `--live` on Colab T4 for live-weight verification.]
