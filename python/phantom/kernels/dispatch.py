"""
PHANTOM v2 — Kernel Fusion Dispatch Layer
=========================================
Routes attention and FFN ops through fused kernels with PyTorch fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

# Ensure repo root is on path for kernels/ package
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from kernels.attention.fused_attention import fused_attention, fused_attention_reference
    from kernels.ffn.fused_ffn import fused_ffn, fused_ffn_reference
    _KERNELS_AVAILABLE = True
except ImportError:
    _KERNELS_AVAILABLE = False


class KernelDispatch:
    """Configurable fused kernel router."""

    def __init__(self, fusion_enabled: bool = True, use_triton: bool = True):
        self.fusion_enabled = fusion_enabled and _KERNELS_AVAILABLE
        self.use_triton = use_triton
        self.fusion_calls = 0
        self.fallback_calls = 0

    def attention(
        self,
        input_act: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
        rope_freqs: Optional[torch.Tensor] = None,
        scale_q: float = 1.0,
        scale_k: float = 1.0,
        scale_v: float = 1.0,
    ) -> torch.Tensor:
        if self.fusion_enabled:
            self.fusion_calls += 1
            return fused_attention(
                input_act, q_weight, k_weight, v_weight, rope_freqs,
                scale_q=scale_q, scale_k=scale_k, scale_v=scale_v,
                use_triton=self.use_triton,
            )
        self.fallback_calls += 1
        return fused_attention_reference(
            input_act, q_weight, k_weight, v_weight, rope_freqs,
            scale_q=scale_q, scale_k=scale_k, scale_v=scale_v,
        )

    def ffn(
        self,
        input_act: torch.Tensor,
        w1_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        scale_w1: float = 1.0,
        scale_w2: float = 1.0,
    ) -> torch.Tensor:
        if self.fusion_enabled:
            self.fusion_calls += 1
            return fused_ffn(input_act, w1_weight, w2_weight, scale_w1=scale_w1, scale_w2=scale_w2, use_triton=self.use_triton)
        self.fallback_calls += 1
        return fused_ffn_reference(input_act, w1_weight, w2_weight, scale_w1=scale_w1, scale_w2=scale_w2)

    @property
    def stats(self) -> dict:
        total = self.fusion_calls + self.fallback_calls
        return {
            "fusion_enabled": self.fusion_enabled,
            "fusion_calls": self.fusion_calls,
            "fallback_calls": self.fallback_calls,
            "fusion_ratio": self.fusion_calls / max(1, total),
        }


_default_dispatch: Optional[KernelDispatch] = None


def get_kernel_dispatch(fusion_enabled: bool = True) -> KernelDispatch:
    global _default_dispatch
    if _default_dispatch is None or _default_dispatch.fusion_enabled != fusion_enabled:
        _default_dispatch = KernelDispatch(fusion_enabled=fusion_enabled)
    return _default_dispatch
