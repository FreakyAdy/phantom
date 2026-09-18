"""
PHANTOM v2 — Adaptive Sparsity Gates (RESEARCH PLACEHOLDER)
===========================================================
Per-MLP sigmoid gate predicting sparse GEMM indices.

⚠️ RESEARCH PLACEHOLDER: This module simulates sparsity gating but does NOT implement
true sparse GEMM kernels. The mask is computed but dense matmuls are still executed.
Real compute savings require custom sparse kernels (e.g., CUTLASS, cuSPARSE, or Triton).

Gate precision must be ≥ 85% before enabling in production (validated on real data).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SparsityReport:
    """Telemetry from adaptive sparsity execution (simulated)."""
    sparsity_fraction: float = 0.0
    neurons_skipped: int = 0
    neurons_total: int = 0
    gate_precision: float = 0.0
    mlp_speedup_estimate: float = 1.0  # Placeholder; no real sparse GEMM


class AdaptiveSparsityGate(nn.Module):
    """
    Per-MLP sigmoid gate predicting active neurons.

    Outputs binary mask over intermediate_dim neurons; inactive neurons
    would be skipped during GEMM (sparse path).

    ⚠️ This is a SIMULATION ONLY. The mask is computed but dense F.linear
    matmuls are still executed in apply_sparse_ffn() because true sparse
    GEMM kernels are not implemented in this codebase.
    """

    def __init__(
        self,
        hidden_dim: int = 5120,
        intermediate_dim: int = 27648,
        sparsity_target: float = 0.40,
        min_gate_precision: float = 0.85,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.sparsity_target = sparsity_target
        self.min_gate_precision = min_gate_precision
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, intermediate_dim),
        )

    def forward(
        self,
        hidden_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, SparsityReport]:
        """
        Args:
            hidden_state: [batch, hidden_dim] or [batch, seq, hidden_dim]

        Returns:
            active_mask: [batch, intermediate_dim] bool mask
            report: SparsityReport telemetry
        """
        if hidden_state.dim() == 3:
            hidden_state = hidden_state[:, -1, :]

        gate_logits = self.gate(hidden_state)
        gate_probs = torch.sigmoid(gate_logits)

        threshold = torch.quantile(gate_probs, self.sparsity_target, dim=-1, keepdim=True)
        active_mask = gate_probs >= threshold

        neurons_total = active_mask.numel()
        neurons_skipped = int((~active_mask).sum().item())
        sparsity_frac = neurons_skipped / max(1, neurons_total)

        # Gate precision MUST be computed on real validation data, not hardcoded
        # Set to 0.0 here to indicate "unvalidated" — production code must
        # override with measured precision before enabling sparsity
        report = SparsityReport(
            sparsity_fraction=sparsity_frac,
            neurons_skipped=neurons_skipped,
            neurons_total=neurons_total,
            gate_precision=0.0,  # UNVALIDATED — requires real data measurement
            mlp_speedup_estimate=1.0,  # No real sparse GEMM → no speedup
        )
        return active_mask, report

    def apply_sparse_ffn(
        self,
        hidden_state: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
    ) -> Tuple[torch.Tensor, SparsityReport]:
        """Apply sparse FFN using gate mask (SIMULATION - dense matmuls executed)."""
        active_mask, report = self.forward(hidden_state)

        # ⚠️ SIMULATION: Always executes dense matmuls.
        # Real sparse GEMM would skip inactive neurons at kernel level.
        if report.gate_precision < self.min_gate_precision or report.gate_precision == 0.0:
            hidden = F.gelu(F.linear(hidden_state, w1))
            return F.linear(hidden, w2), report

        hidden = F.gelu(F.linear(hidden_state, w1))
        if active_mask.any():
            mask_f = active_mask.float()
            hidden = hidden * mask_f
        return F.linear(hidden, w2), report
