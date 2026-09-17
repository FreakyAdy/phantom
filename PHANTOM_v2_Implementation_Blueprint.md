# PHANTOM v2 Implementation Blueprint

> **Canonical spec:** See [`docs/specs/PHANTOM_V2_SPEC.md`](docs/specs/PHANTOM_V2_SPEC.md)
>
> This blueprint has been implemented in the codebase. See implementation map in the canonical spec.

## Phase Status

| Phase | Component | Status |
|---|---|---|
| 1 | EAGLE-3 heads + training | Implemented |
| 2 | Kernel fusion (Triton/PyTorch) | Implemented |
| 2b | Selective Q3 + adaptive sparsity | Implemented |
| 3 | Wraith v2 prefetch | Implemented |
| 4 | CLI live integration | Implemented |
| 5 | Cross-hardware benchmarks | Implemented |

## Key Files

- `python/phantom/speculative/eagle_heads.py`
- `python/phantom/speculative/eagle_train.py`
- `python/phantom/speculative/model_loader.py`
- `python/phantom/prefetch/wraith_v2.py`
- `python/phantom/kernels/dispatch.py`
- `kernels/attention/fused_attention.py`
- `kernels/ffn/fused_ffn.py`
- `benchmarks/phantom_v2_benchmark.py`

See [`docs/specs/PHANTOM_V2_SPEC.md`](docs/specs/PHANTOM_V2_SPEC.md) for architecture and success criteria.
