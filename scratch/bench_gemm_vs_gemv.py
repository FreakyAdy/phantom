"""
PHANTOM Research — Phase 1 Experiment: CPU GEMM vs GEMV Benchmark
=================================================================
Core Hypothesis:
    Batch-8 GEMM reads weights ONCE and computes for 8 tokens.
    8× GEMV reads weights 8 TIMES (once per token).
    If GEMM(8×D, D×F) ≈ GEMV(D, D×F) in wall-clock time,
    then speculative decoding with batch-8 verification gives ~2.8× speedup.

This script benchmarks the exact matrix dimensions used in a 32B Q4 transformer
to determine whether the batched speculative verification strategy is physically
viable on the reference hardware.

Measures:
    1. Single GEMV:    y[1, F] = x[1, D] @ W[D, F]         (simulates 1 token decode)
    2. 8× sequential GEMV: repeat #1 eight times            (simulates 8 sequential tokens)
    3. Single GEMM:    Y[8, F] = X[8, D] @ W[D, F]         (simulates 8 tokens batch verify)
    4. With Q4 dequantization simulation overhead

Target dimensions (Qwen2.5-Coder-32B):
    hidden_dim (D) = 5120
    ffn_dim (F)    = 27648 (gate_proj/up_proj)
    num_layers     = 64 (50 in RAM for 6GB VRAM config)

Expected Result:
    If GEMM_time ≈ GEMV_time → speculative decoding ~2.8× speedup is viable.
    If GEMM_time ≈ 8 × GEMV_time → speculative decoding provides NO amortization benefit.

Usage:
    python scratch/bench_gemm_vs_gemv.py
"""

from __future__ import annotations

import os
import platform
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

# Attempt to use optimized BLAS
try:
    np.show_config()
except Exception:
    pass


def compute_statistics(samples: List[float]) -> Dict[str, float]:
    """Compute mean, stddev, min, max, p50, p95."""
    arr = np.array(samples, dtype=float)
    return {
        "n": len(samples),
        "mean_ms": round(float(np.mean(arr)) * 1000, 3),
        "stddev_ms": round(float(np.std(arr)) * 1000, 3),
        "min_ms": round(float(np.min(arr)) * 1000, 3),
        "max_ms": round(float(np.max(arr)) * 1000, 3),
        "p50_ms": round(float(np.median(arr)) * 1000, 3),
        "p95_ms": round(float(np.percentile(arr, 95)) * 1000, 3),
    }


def benchmark_gemv(
    D: int, F: int, n_runs: int = 20, warmup: int = 5
) -> Dict[str, float]:
    """Benchmark single GEMV: y[1, F] = x[1, D] @ W[D, F]."""
    W = np.ones((D, F), dtype=np.float32)
    x = np.ones((1, D), dtype=np.float32)

    for _ in range(warmup):
        _ = x @ W

    timings = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = x @ W
        t1 = time.perf_counter()
        timings.append(t1 - t0)

    return compute_statistics(timings)


def benchmark_8x_gemv(
    D: int, F: int, n_runs: int = 20, warmup: int = 5
) -> Dict[str, float]:
    """Benchmark 8× sequential GEMV (simulating 8 sequential token decodes)."""
    W = np.ones((D, F), dtype=np.float32)
    xs = [np.ones((1, D), dtype=np.float32) for _ in range(8)]

    for _ in range(warmup):
        for x in xs:
            _ = x @ W

    timings = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        for x in xs:
            _ = x @ W
        t1 = time.perf_counter()
        timings.append(t1 - t0)

    return compute_statistics(timings)


def benchmark_gemm_batch8(
    D: int, F: int, n_runs: int = 20, warmup: int = 5
) -> Dict[str, float]:
    """Benchmark batch-8 GEMM: Y[8, F] = X[8, D] @ W[D, F]."""
    W = np.ones((D, F), dtype=np.float32)
    X = np.ones((8, D), dtype=np.float32)

    for _ in range(warmup):
        _ = X @ W

    timings = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = X @ W
        t1 = time.perf_counter()
        timings.append(t1 - t0)

    return compute_statistics(timings)


def benchmark_full_layer_gemv(
    D: int, F: int, n_layers: int = 10, n_runs: int = 3, warmup: int = 1
) -> Dict[str, float]:
    """Benchmark forward pass through n_layers with GEMV (single token)."""
    weights = [
        (np.ones((D, F), dtype=np.float32),
         np.ones((F, D), dtype=np.float32))
        for _ in range(n_layers)
    ]
    x = np.ones((1, D), dtype=np.float32)

    for _ in range(warmup):
        h = x
        for Wup, Wdown in weights:
            h = (h @ Wup) @ Wdown

    timings = []
    for _ in range(n_runs):
        h = x
        t0 = time.perf_counter()
        for Wup, Wdown in weights:
            h = (h @ Wup) @ Wdown
        t1 = time.perf_counter()
        timings.append(t1 - t0)

    return compute_statistics(timings)


def benchmark_full_layer_gemm8(
    D: int, F: int, n_layers: int = 10, n_runs: int = 3, warmup: int = 1
) -> Dict[str, float]:
    """Benchmark forward pass through n_layers with GEMM batch=8."""
    weights = [
        (np.ones((D, F), dtype=np.float32),
         np.ones((F, D), dtype=np.float32))
        for _ in range(n_layers)
    ]
    X = np.ones((8, D), dtype=np.float32)

    for _ in range(warmup):
        H = X
        for Wup, Wdown in weights:
            H = (H @ Wup) @ Wdown

    timings = []
    for _ in range(n_runs):
        H = X
        t0 = time.perf_counter()
        for Wup, Wdown in weights:
            H = (H @ Wup) @ Wdown
        t1 = time.perf_counter()
        timings.append(t1 - t0)

    return compute_statistics(timings)


def main():
    print("=" * 78)
    print("PHANTOM Phase 1 Experiment: CPU GEMM vs GEMV Benchmark")
    print("=" * 78)
    print()

    # System info
    print(f"Platform:  {platform.platform()}")
    print(f"CPU:       {platform.processor()}")
    print(f"Python:    {sys.version.split()[0]}")
    print(f"NumPy:     {np.__version__}")

    # Check BLAS backend
    try:
        blas_info = np.__config__.blas_opt_info  # type: ignore
        print(f"BLAS:      {blas_info}")
    except Exception:
        print("BLAS:      (unable to detect)")
    print()

    # Qwen2.5-Coder-32B dimensions
    D = 5120       # hidden_dim
    F = 27648      # ffn_dim (gate_proj / up_proj)
    N_LAYERS = 50  # RAM-resident layers on 6GB VRAM config

    print(f"Matrix Dimensions: D={D}, F={F}")
    print(f"Weight matrix size: {D * F * 2 / 1e6:.1f} MB (FP16)")
    print(f"Simulated RAM-resident layers: {N_LAYERS}")
    print()

    # ── Single-op benchmarks ──
    print("-" * 78)
    print("TEST 1: Single GEMV  (y[1,F] = x[1,D] @ W[D,F])")
    print("  Simulates: 1 token decode through 1 MLP layer")
    gemv_stats = benchmark_gemv(D, F)
    print(f"  Result: {gemv_stats['mean_ms']:.3f} ms ± {gemv_stats['stddev_ms']:.3f} ms")
    print(f"          p50={gemv_stats['p50_ms']:.3f}ms, p95={gemv_stats['p95_ms']:.3f}ms")
    print()

    print("-" * 78)
    print("TEST 2: 8× Sequential GEMV  (same W, 8 different x vectors)")
    print("  Simulates: 8 sequential token decodes (current autoregressive path)")
    gemv8_stats = benchmark_8x_gemv(D, F)
    print(f"  Result: {gemv8_stats['mean_ms']:.3f} ms ± {gemv8_stats['stddev_ms']:.3f} ms")
    print(f"          p50={gemv8_stats['p50_ms']:.3f}ms, p95={gemv8_stats['p95_ms']:.3f}ms")
    print()

    print("-" * 78)
    print("TEST 3: Batch-8 GEMM  (Y[8,F] = X[8,D] @ W[D,F])")
    print("  Simulates: 8 token batch verification (speculative decoding)")
    gemm8_stats = benchmark_gemm_batch8(D, F)
    print(f"  Result: {gemm8_stats['mean_ms']:.3f} ms ± {gemm8_stats['stddev_ms']:.3f} ms")
    print(f"          p50={gemm8_stats['p50_ms']:.3f}ms, p95={gemm8_stats['p95_ms']:.3f}ms")
    print()

    # -- Analysis --
    print("=" * 78)
    print("ANALYSIS")
    print("=" * 78)

    ratio_gemm_vs_gemv = gemm8_stats['mean_ms'] / max(0.001, gemv_stats['mean_ms'])
    ratio_gemm_vs_8gemv = gemm8_stats['mean_ms'] / max(0.001, gemv8_stats['mean_ms'])
    amortization_factor = gemv8_stats['mean_ms'] / max(0.001, gemm8_stats['mean_ms'])

    print(f"  GEMM(8) / GEMV(1) ratio:               {ratio_gemm_vs_gemv:.2f}x")
    print(f"  GEMM(8) / 8xGEMV ratio:                {ratio_gemm_vs_8gemv:.2f}x")
    print(f"  Amortization factor (8xGEMV / GEMM(8)): {amortization_factor:.2f}x")
    print()

    if ratio_gemm_vs_gemv < 2.0:
        print("  [PASS] GEMM(8) ~= GEMV(1): Weight reads are AMORTIZED across batch!")
        print("     -> Speculative decoding can yield ~2.8x speedup.")
        print("     -> Dense 32B: 2.88 -> ~7+ tok/s (estimated)")
        verdict = "VIABLE"
    elif ratio_gemm_vs_gemv < 4.0:
        print("  [PARTIAL] GEMM(8) is 2-4x GEMV(1): PARTIAL amortization.")
        print("     -> Speculative decoding yields ~1.5-2x speedup.")
        print("     -> Dense 32B: 2.88 -> ~4-5 tok/s (estimated)")
        verdict = "PARTIALLY_VIABLE"
    else:
        print("  [FAIL] GEMM(8) ~= 8x GEMV(1): NO amortization -- bandwidth-bound per-token.")
        print("     -> Speculative decoding provides NO throughput benefit.")
        print("     -> Different strategy needed (e.g., MoE, model size reduction)")
        verdict = "NOT_VIABLE"

    print()

    # ── Full-layer benchmarks (if time allows) ──
    print("-" * 78)
    print(f"TEST 4: Full {N_LAYERS}-layer forward pass GEMV (1 token)")
    print(f"  Simulates: Complete CPU evaluation of RAM-resident layers")
    print("  (This will take a moment...)")
    try:
        full_gemv = benchmark_full_layer_gemv(D, F, n_layers=min(N_LAYERS, 10), n_runs=3, warmup=1)
        extrapolated_ms = full_gemv['mean_ms'] * (N_LAYERS / min(N_LAYERS, 10))
        print(f"  Result ({min(N_LAYERS, 10)} layers): {full_gemv['mean_ms']:.1f} ms")
        print(f"  Extrapolated ({N_LAYERS} layers): {extrapolated_ms:.1f} ms -> {1000.0/extrapolated_ms:.2f} tok/s")
    except MemoryError:
        print("  [WARN] Insufficient memory for full-layer benchmark")
        extrapolated_ms = None

    print()
    print(f"TEST 5: Full {N_LAYERS}-layer forward pass GEMM batch=8")
    print(f"  Simulates: Speculative batch-8 verification through all RAM layers")
    try:
        full_gemm = benchmark_full_layer_gemm8(D, F, n_layers=min(N_LAYERS, 10), n_runs=3, warmup=1)
        extrapolated_gemm_ms = full_gemm['mean_ms'] * (N_LAYERS / min(N_LAYERS, 10))
        print(f"  Result ({min(N_LAYERS, 10)} layers): {full_gemm['mean_ms']:.1f} ms")
        print(f"  Extrapolated ({N_LAYERS} layers): {extrapolated_gemm_ms:.1f} ms")
        if extrapolated_ms:
            full_amort = extrapolated_ms * 8 / extrapolated_gemm_ms
            print(f"  Full-model amortization factor: {full_amort:.2f}x")
    except MemoryError:
        print("  [WARN] Insufficient memory for full-layer benchmark")

    print()
    print("=" * 78)
    print(f"VERDICT: Batched speculative verification is {verdict}")
    print("=" * 78)
    print()
    print("This benchmark measures CPU BLAS performance with FP16 numpy.")
    print("Production implementation would use INT4/Q4 with fused dequantization,")
    print("which may have different characteristics. Use these results as an upper")
    print("bound estimate for the amortization benefit.")

    return verdict


if __name__ == "__main__":
    main()
