#!/usr/bin/env python3
"""
PHANTOM v2 — Cross-Hardware Benchmark Suite
=============================================
Measures EAGLE-3 + kernel fusion + prefetch stack with ablation support.
Outputs benchmarks/results/v2_latest.json for RESULTS.md generation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from phantom.speculative.draft_runner import DraftRunner
from phantom.speculative.eagle_heads import EagleDrafter, EagleHeads
from phantom.speculative.engine import SpeculativeEngine
from phantom.speculative.target_verifier import TargetVerifier
from phantom.prefetch.wraith_v2 import AdaptivePrefetchScheduler
from phantom.kernels.dispatch import KernelDispatch
from phantom.quant.selective_q3 import SelectiveQ3Quantizer
from phantom.sparsity.adaptive_gate import AdaptiveSparsityGate


HARDWARE_PROFILES = {
    "rtx4050": {"vram_gb": 6, "baseline_tok_s": 2.88, "moe_baseline_tok_s": 12.95},
    "rtx3050": {"vram_gb": 4, "baseline_tok_s": 2.5, "moe_baseline_tok_s": 11.0},
    "rtx4060": {"vram_gb": 8, "baseline_tok_s": 3.2, "moe_baseline_tok_s": 14.0},
}

MODEL_PROFILES = {
    "dense_32b": {"baseline_tok_s": 2.88, "target_tok_s": 7.0},
    "moe_30b": {"baseline_tok_s": 12.95, "target_tok_s": 14.0},
    "dense_14b": {"baseline_tok_s": 5.0, "target_tok_s": 14.0},
}


def _build_engine(
    spec_mode: str = "eagle",
    prefetch: bool = True,
    fusion: bool = True,
    q3: bool = False,
    sparsity: bool = False,
    k: int = 5,
) -> SpeculativeEngine:
    verifier = TargetVerifier(num_layers=64, gpu_layers=14, ram_layers=50)
    if spec_mode == "eagle":
        drafter = EagleDrafter(k=k)
    else:
        drafter = DraftRunner()

    return SpeculativeEngine(
        draft_runner=drafter,
        target_verifier=verifier,
        spec_k=k,
        spec_mode=spec_mode,
        prefetch_scheduler=AdaptivePrefetchScheduler(enabled=prefetch),
        kernel_dispatch=KernelDispatch(fusion_enabled=fusion),
        q3_enabled=q3,
        sparsity_enabled=sparsity,
    )


def benchmark_ablation(
    tokens_to_generate: int = 32,
    k: int = 5,
) -> Dict[str, Any]:
    """Run ablation: disable each innovation individually."""
    prompt = "PHANTOM v2 benchmark: autoregressive decode on consumer hardware"
    configs = {
        "full_stack": dict(spec_mode="eagle", prefetch=True, fusion=True, q3=True, sparsity=True),
        "no_prefetch": dict(spec_mode="eagle", prefetch=False, fusion=True, q3=False, sparsity=False),
        "no_fusion": dict(spec_mode="eagle", prefetch=True, fusion=False, q3=False, sparsity=False),
        "no_q3": dict(spec_mode="eagle", prefetch=True, fusion=True, q3=False, sparsity=False),
        "no_sparsity": dict(spec_mode="eagle", prefetch=True, fusion=True, q3=True, sparsity=False),
        "draft_only": dict(spec_mode="draft", prefetch=False, fusion=False, q3=False, sparsity=False),
    }

    results: Dict[str, Any] = {}
    print("\n" + "=" * 76)
    print("  PHANTOM v2 ABLATION BENCHMARK")
    print("=" * 76)
    print(f"{'CONFIG':<16} | {'TOK/S':<10} | {'ACCEPT %':<10} | {'SPEEDUP':<10} | {'PREFETCH %':<12}")
    print("-" * 76)

    for name, cfg in configs.items():
        engine = _build_engine(k=k, **cfg)
        _, metrics = engine.generate(prompt=prompt, max_new_tokens=tokens_to_generate, k=k)
        results[name] = {
            "tokens_per_second": round(metrics.tokens_per_second, 2),
            "acceptance_rate": round(metrics.mean_acceptance_rate, 3),
            "speedup_factor": round(metrics.speedup_factor, 2),
            "prefetch_hit_rate": round(metrics.prefetch_hit_rate, 3),
            "fusion_calls": metrics.fusion_calls,
            "spec_mode": metrics.spec_mode,
        }
        print(
            f"{name:<16} | {metrics.tokens_per_second:>8.2f} | "
            f"{metrics.mean_acceptance_rate * 100:>8.1f}% | "
            f"{metrics.speedup_factor:>8.2f}x | "
            f"{metrics.prefetch_hit_rate * 100:>10.1f}%"
        )

    return results


def benchmark_hardware_profiles(
    tokens_to_generate: int = 32,
    k: int = 5,
) -> Dict[str, Any]:
    """Simulate benchmarks across hardware profiles."""
    results: Dict[str, Any] = {}
    prompt = "Cross-hardware validation benchmark"

    print("\n" + "=" * 76)
    print("  PHANTOM v2 CROSS-HARDWARE PROFILES")
    print("=" * 76)

    for hw_name, hw_spec in HARDWARE_PROFILES.items():
        engine = _build_engine(spec_mode="eagle", prefetch=True, fusion=True, k=k)
        _, metrics = engine.generate(prompt=prompt, max_new_tokens=tokens_to_generate, k=k)
        projected_tok_s = metrics.speedup_factor * hw_spec["baseline_tok_s"]
        results[hw_name] = {
            "vram_gb": hw_spec["vram_gb"],
            "baseline_tok_s": hw_spec["baseline_tok_s"],
            "projected_tok_s": round(projected_tok_s, 2),
            "speedup_factor": round(metrics.speedup_factor, 2),
            "acceptance_rate": round(metrics.mean_acceptance_rate, 3),
            "meets_moe_14_target": projected_tok_s >= 14.0 if hw_name == "rtx4050" else projected_tok_s >= hw_spec.get("moe_baseline_tok_s", 14),
        }
        print(f"  {hw_name}: baseline={hw_spec['baseline_tok_s']} tok/s -> projected={projected_tok_s:.2f} tok/s ({metrics.speedup_factor:.2f}x)")

    return results


def benchmark_subsystem_micro() -> Dict[str, Any]:
    """Micro-benchmarks for v2 subsystems."""
    results: Dict[str, Any] = {}

    eagle = EagleHeads(hidden_dim=512, vocab_size=1000, k=5)
    results["eagle_param_count"] = eagle.param_count
    results["eagle_memory_mb_fp16"] = round(eagle.memory_mb_fp16, 2)

    q3 = SelectiveQ3Quantizer()
    sd = {"model.layers.0.mlp.gate_proj.weight": torch.randn(128, 512)}
    q3_report = q3.apply_to_state_dict(sd)
    results["q3_traffic_reduction_pct"] = round(q3_report.weight_traffic_reduction_pct, 1)

    gate = AdaptiveSparsityGate(hidden_dim=512, intermediate_dim=1024)
    hidden = torch.randn(2, 512)
    _, sparsity_report = gate.forward(hidden)
    results["sparsity_fraction"] = round(sparsity_report.sparsity_fraction, 3)

    dispatch = KernelDispatch(fusion_enabled=True)
    x = torch.randn(4, 128)
    w = torch.randn(128, 128)
    _ = dispatch.ffn(x, w, w)
    results["fusion_calls"] = dispatch.fusion_calls

    prefetch = AdaptivePrefetchScheduler(enabled=True)
    prefetch.schedule_for_spec_round(
        hidden_state=torch.randn(512), current_layer=0, position=100,
        draft_tokens=[1, 2, 3, 4, 5], vram_layers=set(range(14)),
    )
    results["prefetch_requests"] = prefetch.metrics.total_prefetch_requests

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="PHANTOM v2 Benchmark Suite")
    parser.add_argument("--quick", action="store_true", help="Fast CI mode")
    parser.add_argument("--out", default="benchmarks/results/v2_latest.json")
    args = parser.parse_args()

    tokens = 16 if args.quick else 32
    k = 3 if args.quick else 5

    print("\n" + "=" * 76)
    print("  PHANTOM v2 MD BLUEPRINT BENCHMARK SUITE")
    print("=" * 76)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "quick_mode": args.quick,
        "cpu_threads": torch.get_num_threads(),
        "subsystem_micro": benchmark_subsystem_micro(),
        "ablation": benchmark_ablation(tokens_to_generate=tokens, k=k),
        "hardware_profiles": benchmark_hardware_profiles(tokens_to_generate=tokens, k=k),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Results saved to: {out_path}")
    print("=" * 76 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
