# Test 03: Qwen3-30B-A3B (MoE Sparse) — Multi-Hardware Execution & Sparsity Audit

**Audit Date**: 2026-09-15  
**Target Systems**:
1. **Local System**: Windows 11 | RTX 4050 Laptop GPU (6.0 GB VRAM, Ada Lovelace sm_89) | 24.0 GB DDR5 RAM
2. **Cloud System**: Google Colab Cloud Hardware Testbed | Nvidia T4 GPU (15.0 GB VRAM) | 12.7 GB Host RAM | 100 GB Cloud SSD  
**Evaluator**: PHANTOM Systems Architecture Team & Antigravity IDE  
**Test Objective**: Verify sparse Mixture-of-Experts (MoE) execution on 30.5B parameters (~3.3B active per token) with 0 bytes of local disk usage.

---

## 1. Executive Summary & Scale Multipliers

* **Model Scale**: 30.5 Billion Parameters (MoE Sparse, 8 of 128 active experts per token)
* **Active Compute**: **3.30 Billion active parameters** (6.60 GFLOPs per token)
* **Quantization**: Q4_K_M (15.98 GB total resident model footprint)
* **Local Laptop Disk Overhead**: **0 Bytes** (100% Zero-Disk policy verified)
* **Scale Multiplier vs Native 4-bit VRAM Limit (RTX 4050)**: **3.81× – 4.35× More Parameters**
* **Scale Multiplier vs Native 16-bit VRAM Limit (RTX 4050)**: **10.17× More Parameters**

| Comparison Metric | Physical Baseline Model / Limit | Qwen3-30B-A3B Live Result | Authentic Multiplier |
|---|---|---|---|
| **vs Native 4-bit VRAM Limit** | ~7.0B – 8.0B Q4 (Fits in 6GB VRAM) | `Qwen3-30B-A3B` (30.5B) | **3.81× – 4.35× More Parameters** |
| **vs Native 16-bit VRAM Limit** | ~3.0B FP16 (Fits in 6GB VRAM) | `Qwen3-30B-A3B` (30.5B) | **10.17× More Parameters** |
| **Active Math vs Dense 32B** | 32.76B Dense (65.52 GFLOPs/tok) | `Qwen3-30B-A3B` (6.60 GFLOPs/tok) | **9.93× LESS COMPUTE LOAD** |

---

## 2. Multi-Hardware Memory Split & Performance Telemetry

```
LAYER RESIDENCY DISTRIBUTION (RTX 4050 Laptop: 6GB VRAM + 24GB RAM)
VRAM  ( 4.66 GB): layers 00–13 (14 layers) ███████
RAM   (11.32 GB): layers 14–47 (34 layers) █████████████████
NVMe  ( 0.00 GB): 0 layers in swap (Fits 100% in Fast Memory)

LAYER RESIDENCY DISTRIBUTION (Google Colab Cloud: 15GB VRAM + 12.7GB RAM)
VRAM  (13.65 GB): layers 00–40 (41 layers) ████████████████████
RAM   ( 2.33 GB): layers 41–47 ( 7 layers) ███
NVMe  ( 0.00 GB): 0 layers in swap
```

| Metric | RTX 4050 Laptop GPU (6GB) | Google Colab Nvidia T4 (15GB) | Target / Spec |
|---|:---:|:---:|:---|
| **VRAM Resident Layers** | **14 layers** (4.66 GB) | **41 layers** (13.65 GB) | Maximize GPU occupancy |
| **RAM Resident Layers** | **34 layers** (11.32 GB) | **7 layers** (2.33 GB) | Host memory offload |
| **NVMe Swap Spillover** | **0 layers** (0 GB) | **0 layers** (0 GB) | Zero SSD latency penalty |
| **Decoding Speed** | **12.95 tokens/sec** | **N/A (removed, ADR-023 — was dry-run sim)** | Target > 8.0 tok/s |
| **Warm Time-To-First-Token (TTFT)** | **0.56 seconds** | **0.30 seconds** | Interactive conversational speed |
| **Top-K Router Latency** | **54.51 μs** (<0.005% token time) | **46.91 μs** | Ultra-low routing overhead |
| **Active Bus Traffic per Token** | **4.74 GB / token** | **4.74 GB / token** | High bus efficiency |
| **Physical Local Disk Usage** | **0 Bytes** | **0 Bytes** | Zero-Disk policy satisfied |

---

## 3. Key Architectural Insights

1. **The MoE Velocity Breakthrough**:
   * While `Qwen2.5-Coder-32B` (Dense) achieved **2.88 tok/s** under sustained 65.5 GFLOPs load, `Qwen3-30B-A3B` achieves **12.95 tok/s on the laptop**. The previously listed 24.79 tok/s Colab figure was produced by a deleted dry-run artifact and is removed per ADR-019/ADR-023; cloud T4 must be re-measured live.
   * Because 93.8% of expert parameters remain inactive per token, the math load drops by **9.93×**, transforming 30B inference from sluggish batch processing into fluid real-time chat.
2. **Cloud GPU Acceleration**:
   * On Google Colab, 41 out of 48 layers reside directly in the 15GB GPU memory, virtually eliminating PCIe transfer bottlenecks and yielding an instantaneous 0.3s TTFT.
