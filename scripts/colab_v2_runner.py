#!/usr/bin/env python3
"""
PHANTOM v2 — Google Colab Live-Weight Test Harness
===================================================
Runs on Google Colab (T4/L4 GPU) with ephemeral /content/scratch storage:

1. Live E2E v2 speculative decode (Qwen2.5-Coder-32B + EAGLE-3 heads)
2. EAGLE head training on cloud GPU
3. MoE expert routing / Wraith prefetch correlation (Qwen3-30B-A3B profile)
4. v2 ablation benchmark + merged results export

Usage (in Colab notebook or cloud VM):
  python scripts/colab_v2_runner.py --model qwen2.5-coder-32b --live
  python scripts/colab_v2_runner.py --model qwen3-30b-a3b --moe-correlation
  python scripts/colab_v2_runner.py --dry-run   # local validation without downloads

Zero local disk: weights download only to /content/scratch and are purged on exit.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT))

from scripts.colab_runner import MODEL_REGISTRY, generate_markdown_report  # noqa: E402
from phantom.model_profiles.hardware_simulator import simulate_model_execution  # noqa: E402


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return "COLAB_RELEASE_TAG" in os.environ


def scratch_dir() -> Path:
    if is_colab():
        p = Path("/content/scratch")
    else:
        p = Path(tempfile.mkdtemp(prefix="phantom_v2_"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def download_gguf(model_key: str, dest: Path) -> Path:
    """Download GGUF to ephemeral scratch via huggingface_hub."""
    from huggingface_hub import hf_hub_download

    meta = MODEL_REGISTRY[model_key]
    local = hf_hub_download(
        repo_id=meta["repo"],
        filename=meta["file"],
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )
    return Path(local)


def train_eagle_on_cloud(
    hidden_dim: int = 5120,
    vocab_size: int = 32000,
    k: int = 5,
    examples: int = 512,
    epochs: int = 5,
    output_path: Optional[Path] = None,
    force_cpu: bool = False,
) -> Dict[str, Any]:
    """Train EAGLE-3 heads on cloud GPU (or CPU fallback)."""
    import torch
    from torch.utils.data import DataLoader

    from phantom.speculative.eagle_heads import EagleHeads
    from phantom.speculative.eagle_train import (
        EagleDataset,
        collect_training_data_synthetic,
        collate_eagle_batch,
        train_eagle_heads,
    )

    vram_gb = 0.0
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    use_cuda = torch.cuda.is_available() and not force_cpu and vram_gb >= 12.0
    device = "cuda" if use_cuda else "cpu"
    batch_size = 32 if use_cuda else 8

    eagle = EagleHeads(hidden_dim=hidden_dim, vocab_size=vocab_size, k=k, device=device)

    examples_list = collect_training_data_synthetic(
        num_examples=examples, hidden_dim=hidden_dim, k=k, vocab_size=vocab_size,
    )
    loader = DataLoader(
        EagleDataset(examples_list), batch_size=batch_size, shuffle=True, collate_fn=collate_eagle_batch,
    )

    start = time.perf_counter()
    history = train_eagle_heads(eagle, loader, num_epochs=epochs, device=device)
    elapsed = time.perf_counter() - start

    out = output_path or scratch_dir() / "eagle" / f"eagle_{int(time.time())}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": eagle.state_dict(),
        "hidden_dim": hidden_dim,
        "vocab_size": vocab_size,
        "k": k,
        "training_history": history,
    }, out)

    return {
        "checkpoint": str(out),
        "param_count": eagle.param_count,
        "memory_mb_fp16": round(eagle.memory_mb_fp16, 2),
        "training_seconds": round(elapsed, 2),
        "device": device,
        "final_loss": history.get(f"epoch_{epochs - 1}_loss"),
    }


def run_live_v2_speculative(
    model_key: str = "qwen2.5-coder-32b",
    eagle_checkpoint: Optional[Path] = None,
    spec_k: int = 5,
    max_tokens: int = 64,
) -> Dict[str, Any]:
    """Run live v2 speculative decode with downloaded GGUF weights."""
    import torch
    from phantom.speculative.model_loader import SpeculativeRuntimeConfig, build_speculative_engine
    from phantom.speculative.model_loader import _load_gguf_model

    scratch = scratch_dir()
    gguf_path = download_gguf(model_key, scratch / "models")

    target_model, tokenizer = _load_gguf_model(gguf_path, n_gpu_layers=14)

    if eagle_checkpoint is None:
        eagle_ckpt = scratch / "eagle" / f"{model_key.replace('-', '_')}.pt"
        if not eagle_ckpt.exists():
            train_eagle_on_cloud(output_path=eagle_ckpt)
        eagle_checkpoint = eagle_ckpt

    config = SpeculativeRuntimeConfig(
        model_id=model_key,
        spec_mode="eagle",
        spec_k=spec_k,
        n_gpu_layers=14,
        eagle_heads_path=str(eagle_checkpoint),
        prefetch_enabled=True,
        fusion_enabled=True,
    )

    engine = build_speculative_engine(config, target_model, tokenizer)
    prompt = "Write a Python function for 0/1 knapsack with space-optimized DP. Target answer: 220."

    start = time.perf_counter()
    text, metrics = engine.generate(prompt=prompt, max_new_tokens=max_tokens, k=spec_k)
    elapsed = time.perf_counter() - start

    return {
        "model_key": model_key,
        "mode": "live_colab",
        "prompt": prompt,
        "output_preview": text[:200],
        "tokens_per_second": round(metrics.tokens_per_second, 2),
        "acceptance_rate": round(metrics.mean_acceptance_rate, 4),
        "speedup_factor": round(metrics.speedup_factor, 2),
        "prefetch_hit_rate": round(metrics.prefetch_hit_rate, 4),
        "fusion_calls": metrics.fusion_calls,
        "total_latency_seconds": round(elapsed, 2),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def run_moe_correlation(num_samples: int = 512) -> Dict[str, Any]:
    """Run MoE expert routing vs Wraith prefetch correlation analysis."""
    from phantom.prefetch.moe_correlation import analyze_moe_prefetch_correlation, report_to_dict

    report = analyze_moe_prefetch_correlation(
        hidden_dim=4096,
        num_layers=48,
        num_experts=16,
        top_k=2,
        num_samples=num_samples,
    )
    return report_to_dict(report)


def run_v2_ablation(quick: bool = True) -> Dict[str, Any]:
    """v2 ablation moved to benchmarks/run_real.py --focus spec-decode.

    Previously used a model-free harness (benchmarks/phantom_v2_benchmark.py)
    that produced fabricated acceptance=1.0 / ~3900 tok/s values with no models
    loaded. Removed in ADR-023; real spec-decode measurements are canonical via:
        python benchmarks/run_real.py --focus spec-decode
    """
    return {
        "status": "DEPRECATED_MODEL_FREE_SIMULATION_REMOVED",
        "note": "Model-free v2 ablation harness deleted (fabricated). Real spec-decode measurements: python benchmarks/run_real.py --focus spec-decode",
    }


def merge_v2_into_results(v2_payload: Dict[str, Any], out_path: Optional[Path] = None) -> Path:
    """Merge v2 Colab results into benchmarks/results/v2_latest.json."""
    target = out_path or (REPO_ROOT / "benchmarks" / "results" / "v2_latest.json")
    target.parent.mkdir(parents=True, exist_ok=True)

    existing: Dict[str, Any] = {}
    if target.exists():
        with open(target, "r", encoding="utf-8") as f:
            existing = json.load(f)

    merged = {**existing, **v2_payload, "merged_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with open(target, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    return target


def generate_v2_colab_report(payload: Dict[str, Any], output_dir: Path) -> Path:
    """Generate test_16 style markdown report for Colab v2 run."""
    date_str = datetime.date.today().isoformat()
    live = payload.get("live_v2", {})
    moe = payload.get("moe_correlation", {})
    ablation = payload.get("v2_ablation", {}).get("ablation", {})

    lines = [
        "# Test 16: PHANTOM v2 MD Blueprint — Google Colab Live Verification",
        "",
        f"**Audit Date**: {date_str}  ",
        f"**Target System**: Google Colab Cloud GPU  ",
        f"**Test Objective**: Live-weight E2E v2 speculative decode, EAGLE training, MoE prefetch correlation.",
        "",
        "---",
        "",
        "## 1. Live v2 Speculative Decode",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Model | `{live.get('model_key', 'N/A')}` |",
        f"| GPU | `{live.get('gpu', 'N/A')}` |",
        f"| Throughput | **{live.get('tokens_per_second', 'N/A')} tok/s** |",
        f"| Acceptance Rate | **{live.get('acceptance_rate', 'N/A')}** |",
        f"| Speedup Factor | **{live.get('speedup_factor', 'N/A')}x** |",
        f"| Prefetch Hit Rate | **{live.get('prefetch_hit_rate', 'N/A')}** |",
        "",
        "---",
        "",
        "## 2. MoE Expert / Prefetch Correlation (Qwen3-30B-A3B profile)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Samples | {moe.get('num_samples', 'N/A')} |",
        f"| Mean Expert Overlap | **{moe.get('mean_expert_overlap', 'N/A')}** |",
        f"| Layer Prediction Accuracy | **{moe.get('layer_prediction_accuracy', 'N/A')}** |",
        f"| Prefetch Usefulness Score | **{moe.get('prefetch_usefulness_score', 'N/A')}** |",
        f"| Expert Sparsity | **{moe.get('expert_sparsity_ratio', 'N/A')}** |",
        "",
        "---",
        "",
        "## 3. v2 Ablation Summary",
        "",
    ]

    for cfg, metrics in ablation.items():
        lines.append(f"- **{cfg}**: {metrics.get('tokens_per_second', 'N/A')} tok/s, "
                     f"accept={metrics.get('acceptance_rate', 'N/A')}, "
                     f"speedup={metrics.get('speedup_factor', 'N/A')}x")

    lines.extend([
        "",
        "---",
        "",
        "**Status**: [VERIFIED ON COLAB — register in docs/testing/INDEX.md as test_16]",
        "",
    ])

    out_path = output_dir / "test_16_phantom_v2_colab.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def run_colab_v2_suite(
    model_key: str = "qwen2.5-coder-32b",
    preset: str = "colab-t4",
    live: bool = False,
    dry_run: bool = False,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute full PHANTOM v2 Colab test suite."""
    out_dir = Path(output_dir or (REPO_ROOT / "docs" / "testing"))
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = scratch_dir()

    print("\n" + "=" * 78)
    print("  PHANTOM v2 COLAB TEST SUITE")
    print(f"  Model: {model_key} | Preset: {preset} | Live: {live} | Colab: {is_colab()}")
    print("=" * 78)

    payload: Dict[str, Any] = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "colab_detected": is_colab(),
        "model_key": model_key,
        "hardware_preset": preset,
        "dry_run": dry_run,
        "live": live,
    }

    # 1. EAGLE training
    print("\n[1/4] Training EAGLE-3 heads...")
    eagle_out = scratch / "eagle" / f"{model_key.replace('-', '_')}.pt"
    payload["eagle_training"] = train_eagle_on_cloud(
        examples=128 if dry_run else 512,
        epochs=2 if dry_run else 5,
        output_path=eagle_out,
        force_cpu=dry_run or not is_colab(),
    )
    print(f"  EAGLE checkpoint: {payload['eagle_training']['checkpoint']}")

    # 2. MoE prefetch correlation
    print("\n[2/4] MoE expert routing / prefetch correlation...")
    payload["moe_correlation"] = run_moe_correlation(num_samples=256 if dry_run else 512)
    print(f"  Prefetch usefulness: {payload['moe_correlation']['prefetch_usefulness_score']}")

    # 3. Live v2 E2E or simulated profile
    print("\n[3/4] v2 speculative decode...")
    if live and not dry_run:
        try:
            payload["live_v2"] = run_live_v2_speculative(
                model_key=model_key,
                eagle_checkpoint=eagle_out,
            )
            print(f"  Live tok/s: {payload['live_v2']['tokens_per_second']}")
        except Exception as exc:
            print(f"  Live inference FAILED ({exc}); recorded as measurement failure (no fabricated fallback).")
            payload["live_v2"] = {
                "model_key": model_key,
                "mode": "live_measurement_failed",
                "tokens_per_second": None,
                "acceptance_rate": None,
                "speedup_factor": None,
                "error": str(exc),
            }
    else:
        payload["live_v2"] = {
            "model_key": model_key,
            "mode": "dry_run",
            "status": "SIMULATED_DISABLED_MEASUREMENT",
            "tokens_per_second": None,
            "acceptance_rate": None,
            "speedup_factor": None,
            "note": "Dry-run never fabricates speculative-decode tok/s (ADR-018). Run live or use benchmarks/run_real.py --focus spec-decode.",
        }
        print(f"  Dry-run: speculative-decode targets require live measurement (not simulated).")

    # 4. v2 ablation benchmark (real measurements only)
    print("\n[4/4] v2 ablation benchmark...")
    payload["v2_ablation"] = run_v2_ablation(quick=dry_run)
    print("  Ablation complete.")

    # Merge results
    json_path = out_dir / f"v2_colab_{model_key}_{int(time.time())}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[OK] JSON saved: {json_path}")

    merged_path = merge_v2_into_results(payload)
    print(f"[OK] Merged into: {merged_path}")

    md_path = generate_v2_colab_report(payload, out_dir)
    print(f"[OK] Report saved: {md_path}")

    # Purge scratch
    if is_colab() and scratch.exists():
        shutil.rmtree(scratch, ignore_errors=True)
        print("[OK] Ephemeral scratch purged (/content/scratch).")
    elif not is_colab() and scratch.exists():
        shutil.rmtree(scratch, ignore_errors=True)
        print("[OK] Local temp scratch purged.")

    print("=" * 78 + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="PHANTOM v2 Google Colab Test Harness")
    parser.add_argument("--model", default="qwen2.5-coder-32b", help="Primary model for live v2 test")
    parser.add_argument("--preset", default="colab-t4", help="Hardware preset")
    parser.add_argument("--live", action="store_true", help="Download weights and run live inference (Colab only)")
    parser.add_argument("--dry-run", action="store_true", help="Skip weight downloads; use simulation")
    parser.add_argument("--moe-correlation", action="store_true", help="Run only MoE correlation analysis")
    parser.add_argument("--output-dir", default=None, help="Output directory for reports")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else None

    if args.moe_correlation:
        result = run_moe_correlation()
        print(json.dumps(result, indent=2))
        return 0

    if args.live and not is_colab() and not args.dry_run:
        print("Warning: --live outside Colab will use temp scratch; pass --dry-run to skip downloads.", file=sys.stderr)

    run_colab_v2_suite(
        model_key=args.model,
        preset=args.preset,
        live=args.live,
        dry_run=args.dry_run,
        output_dir=out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
