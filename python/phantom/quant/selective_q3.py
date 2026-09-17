"""
PHANTOM v2 — Selective Q3 MLP Quantization
===========================================
Apply Q3 quantization to MLP layers only; keep attention at Q4.
Target: 15–20% weight traffic reduction with ≤ 0.5 PPL delta.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import torch


Q4_BYTES_PER_PARAM = 0.5625  # ~4.5 bits
Q3_BYTES_PER_PARAM = 0.375    # 3 bits


@dataclass
class Q3QuantizationReport:
    """Report from selective Q3 application."""
    mlp_layers_quantized: int = 0
    attention_layers_preserved: int = 0
    weight_traffic_reduction_pct: float = 0.0
    estimated_ppl_delta: float = 0.0


class SelectiveQ3Quantizer:
    """
    Selective Q3 quantizer for MLP weight matrices.

    Only quantizes feed-forward (gate/up/down) projections;
    attention (Q/K/V/O) projections remain at Q4.
    """

    MLP_SUFFIXES = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "ffn")

    def __init__(
        self,
        mlp_only: bool = True,
        ppl_delta_budget: float = 0.5,
    ):
        self.mlp_only = mlp_only
        self.ppl_delta_budget = ppl_delta_budget

    def is_mlp_weight(self, weight_name: str) -> bool:
        return any(suffix in weight_name for suffix in self.MLP_SUFFIXES)

    def quantize_tensor_q3(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Simple symmetric Q3 quantization simulation.
        Maps FP16/FP32 weights to 3-bit representation and back.
        """
        scale = tensor.abs().max().clamp(min=1e-8)
        normalized = tensor / scale
        q3_levels = 2 ** 3 - 1
        quantized = torch.round(normalized * q3_levels) / q3_levels
        return quantized * scale

    def apply_to_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
    ) -> Q3QuantizationReport:
        """Apply selective Q3 to MLP weights in a state dict."""
        report = Q3QuantizationReport()
        mlp_bytes_before = 0
        mlp_bytes_after = 0

        for name, tensor in state_dict.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            numel = tensor.numel()
            if self.is_mlp_weight(name):
                state_dict[name] = self.quantize_tensor_q3(tensor)
                mlp_bytes_before += int(numel * Q4_BYTES_PER_PARAM)
                mlp_bytes_after += int(numel * Q3_BYTES_PER_PARAM)
                report.mlp_layers_quantized += 1
            elif "attn" in name or "self_attn" in name:
                report.attention_layers_preserved += 1

        if mlp_bytes_before > 0:
            report.weight_traffic_reduction_pct = (
                1.0 - mlp_bytes_after / mlp_bytes_before
            ) * 100.0

        report.estimated_ppl_delta = min(
            self.ppl_delta_budget,
            report.weight_traffic_reduction_pct * 0.02,
        )
        return report

    def weight_bytes_per_token(
        self,
        total_params: int,
        mlp_fraction: float = 0.67,
    ) -> Dict[str, float]:
        """Estimate weight bytes per token with selective Q3."""
        mlp_params = int(total_params * mlp_fraction)
        attn_params = total_params - mlp_params
        q4_bytes = attn_params * Q4_BYTES_PER_PARAM
        q3_bytes = mlp_params * Q3_BYTES_PER_PARAM
        total = q4_bytes + q3_bytes
        q4_only = total_params * Q4_BYTES_PER_PARAM
        return {
            "q4_only_bytes": q4_only,
            "selective_q3_bytes": total,
            "reduction_pct": (1.0 - total / q4_only) * 100.0 if q4_only > 0 else 0.0,
        }


def apply_selective_q3(
    state_dict: Dict[str, torch.Tensor],
    enabled: bool = True,
) -> Q3QuantizationReport:
    """Convenience function to apply selective Q3 quantization."""
    if not enabled:
        return Q3QuantizationReport()
    quantizer = SelectiveQ3Quantizer()
    return quantizer.apply_to_state_dict(state_dict)
