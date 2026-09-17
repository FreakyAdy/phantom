# PHANTOM v2 Canonical Specification

**Status:** Active engineering directive  
**Supersedes:** ADR-015 purge scope (see ADR-016)  
**Sources:** [`PHANTOM_Research_Analysis.md`](../../PHANTOM_Research_Analysis.md), [`PHANTOM_Quick_Reference.md`](../../PHANTOM_Quick_Reference.md), [`PHANTOM_v2_Implementation_Blueprint.md`](../../PHANTOM_v2_Implementation_Blueprint.md)

---

## Executive Summary

PHANTOM v2 maximizes consumer-GPU token throughput via a **multiplicative optimization stack**:

1. **EAGLE-3 speculative decoding** — feature-fusion heads, K=5, 70–75% acceptance
2. **Triton/PyTorch kernel fusion** — fused attention + FFN
3. **Wraith v2 adaptive prefetch** — overlap weight staging with verification
4. **Selective Q3 MLP quantization** — 15–20% weight traffic reduction
5. **Conservative 40% adaptive sparsity** — MLP neuron gating

---

## Physics-Honest Targets

| Model tier | Measured baseline | v2 target |
|---|---|---|
| MoE 30B (Qwen3-30B-A3B) | 12.95 tok/s | **14–18 tok/s** |
| Dense 32B (Qwen2.5-Coder-32B) | 2.88 tok/s | **7–10 tok/s** |
| Dense 14B | ~5 tok/s est. | **14 tok/s** |
| Dense 70B (NVMe tier) | 0.39 tok/s | Not primary (ADR-013) |

Dense 32B → 14 tok/s is physically impossible on DDR5 (48 GB/s). The 14 tok/s headline applies to MoE 30B and dense 14B tiers.

---

## Architecture

```
Input → Embedding → Layers 0–79
                      ↓ feature capture at [0, 30, 60, 79]
              ┌───────┴───────┐
         LM Head          EAGLE-3 Heads (K=5)
              └───────┬───────┘
              Parallel Verification → Lossless Acceptance → Output
Wraith v2 prefetch stages next-layer weights during verification
Fused kernels: dequant+GEMM+RoPE+softmax (attn), dequant+GEMM+GELU (FFN)
```

---

## Implementation Map

| Component | Path |
|---|---|
| EAGLE-3 heads | `python/phantom/speculative/eagle_heads.py` |
| EAGLE training | `python/phantom/speculative/eagle_train.py` |
| Speculative engine | `python/phantom/speculative/engine.py` |
| Model loader | `python/phantom/speculative/model_loader.py` |
| Fused attention | `kernels/attention/fused_attention.py` |
| Fused FFN | `kernels/ffn/fused_ffn.py` |
| Kernel dispatch | `python/phantom/kernels/dispatch.py` |
| Wraith v2 prefetch | `python/phantom/prefetch/wraith_v2.py` |
| Selective Q3 | `python/phantom/quant/selective_q3.py` |
| Adaptive sparsity | `python/phantom/sparsity/adaptive_gate.py` |
| v2 benchmarks | `benchmarks/phantom_v2_benchmark.py` |

---

## CLI Usage

```bash
phantom run qwen2.5-coder-32b "Explain MoE" \
  --spec-mode eagle \
  --eagle-heads ~/.phantom/eagle/qwen2.5_32b.pt \
  --spec-k 5 \
  -ngl 14 \
  --no-prefetch   # ablation only
  --no-fusion     # ablation only
```

**Flags:** `--spec-mode eagle|draft`, `--eagle-heads PATH`, `--spec-k N`, `--no-prefetch`, `--no-fusion`, `--enable-q3`, `--enable-sparsity`

---

## Success Criteria

- MoE 30B ≥ 14 tok/s on RTX 4050
- Dense 32B ≥ 7 tok/s with full stack
- Acceptance rate α ≥ 70%
- PPL delta < 0.2; token agreement > 99.5%
- All metrics from `benchmarks/results/v2_latest.json` → `RESULTS.md`

---

## References

- ADR-016: Reinstatement of MD Blueprint Stack
- [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md) §10 invariants
- EAGLE-3: Li et al., arXiv:2503.01840
