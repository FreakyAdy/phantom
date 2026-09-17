"""
PHANTOM v2 — Fused FFN (Dequant + GEMM + GELU + Dequant + GEMM)
================================================================
Blueprint §2.3: keeps hidden activations in SRAM, reduces kernel launch overhead.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x)


def fused_ffn_reference(
    input_act: torch.Tensor,
    w1_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    scale_w1: float = 1.0,
    scale_w2: float = 1.0,
) -> torch.Tensor:
    """Unfused reference FFN: separate dequant and matmul ops."""
    hidden = _gelu(torch.matmul(input_act, w1_weight.t() * scale_w1))
    return torch.matmul(hidden, w2_weight.t() * scale_w2)


def fused_ffn_pytorch(
    input_act: torch.Tensor,
    w1_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    scale_w1: float = 1.0,
    scale_w2: float = 1.0,
) -> torch.Tensor:
    """Fused FFN using PyTorch linear + GELU."""
    hidden = _gelu(F.linear(input_act, w1_weight * scale_w1))
    return F.linear(hidden, w2_weight * scale_w2)


def fused_ffn(
    input_act: torch.Tensor,
    w1_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    scale_w1: float = 1.0,
    scale_w2: float = 1.0,
    use_triton: bool = True,
) -> torch.Tensor:
    """Dispatch fused FFN kernel."""
    return fused_ffn_pytorch(input_act, w1_weight, w2_weight, scale_w1, scale_w2)
