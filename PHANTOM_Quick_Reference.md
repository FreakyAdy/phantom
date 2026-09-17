# PHANTOM Quick Reference

> **Canonical spec:** See [`docs/specs/PHANTOM_V2_SPEC.md`](docs/specs/PHANTOM_V2_SPEC.md)

## Speed Roadmap

| Phase | Optimization | Expected gain |
|---|---|---|
| Baseline | Current PHANTOM | model-dependent |
| Phase 1 | Kernel fusion | +6–12% |
| Phase 2 | Q3 MLP + sparsity 40% | +20–25% |
| Phase 3 | EAGLE-3 speculation | +2.5–3.0× |

## Hardware Targets (RTX 4050)

- MoE 30B: 14–18 tok/s
- Dense 32B: 7–10 tok/s
- Dense 14B: 14 tok/s

## CLI Quick Start

```bash
phantom run qwen3-30b-a3b "Hello" --spec-mode eagle -ngl 14 --spec-k 5
phantom benchmark --v2 --quick
```

See [`docs/specs/PHANTOM_V2_SPEC.md`](docs/specs/PHANTOM_V2_SPEC.md) for full details.
