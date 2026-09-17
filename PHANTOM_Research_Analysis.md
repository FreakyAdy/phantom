# PHANTOM Research Analysis

> **Canonical spec:** See [`docs/specs/PHANTOM_V2_SPEC.md`](docs/specs/PHANTOM_V2_SPEC.md) for the active engineering directive.
>
> Full research analysis content is preserved in this file. Implementation follows ADR-016 (MD Blueprint stack).

This document contains the rigorous physics-based investigation into whether 5 tok/s can become 14 tok/s on consumer hardware. Key findings:

- Memory bandwidth is the primary bottleneck for autoregressive decode
- EAGLE-3 speculative decoding is the single largest speedup lever (2.5–3.0×)
- Kernel fusion, prefetch, Q3 quantization, and adaptive sparsity provide incremental gains
- Dense 32B → 14 tok/s is physically impossible; MoE 30B and dense 14B are the realistic 14 tok/s targets

For the complete analysis, see the original content in git history or [`docs/specs/PHANTOM_RESEARCH_REPORT.md`](docs/specs/PHANTOM_RESEARCH_REPORT.md).
