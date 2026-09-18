"""
PHANTOM v2 — MoE Expert Routing / Expert-Aware Prefetch Correlation
=====================================================================
Analyzes correlation between expert routing patterns and prefetch scheduling
for sparse MoE models using the real MoEExpertTracker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from phantom.prefetch.wraith_v2 import MoEExpertTracker


@dataclass
class MoEPrefetchCorrelationReport:
    """Correlation metrics between expert routing and prefetch predictions."""
    num_samples: int = 0
    mean_expert_overlap: float = 0.0
    layer_prediction_accuracy: float = 0.0
    prefetch_usefulness_score: float = 0.0
    expert_sparsity_ratio: float = 0.0
    recommended_prefetch_layers: List[int] = None

    def __post_init__(self):
        if self.recommended_prefetch_layers is None:
            self.recommended_prefetch_layers = []


class MoETopKRouter(nn.Module):
    """Minimal MoE router for correlation analysis."""

    def __init__(self, hidden_dim: int = 4096, num_experts: int = 16, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(x)
        top_logits, top_indices = torch.topk(logits, self.top_k, dim=-1)
        top_weights = F.softmax(top_logits, dim=-1)
        return top_weights, top_indices


def analyze_moe_prefetch_correlation(
    model_id: str = "qwen3-30b-a3b",
    hidden_dim: int = 4096,
    num_layers: int = 48,
    num_experts: int = 128,
    top_k: int = 8,
    num_samples: int = 256,
    seed: int = 42,
) -> MoEPrefetchCorrelationReport:
    """
    Simulate MoE routing decisions and measure correlation with expert-aware
    prefetch tracking for the specified model architecture.

    Returns correlation report for prefetch tuning on MoE models like Qwen3-30B-A3B.
    """
    torch.manual_seed(seed)
    
    # Use the real expert tracker for the model
    tracker = MoEExpertTracker(model_id)
    
    router = MoETopKRouter(hidden_dim=hidden_dim, num_experts=num_experts, top_k=top_k)

    expert_hits: Dict[int, int] = {i: 0 for i in range(num_experts)}
    layer_hits = 0
    expert_overlaps: List[float] = []
    prefetch_scores: List[float] = []

    with torch.no_grad():
        for sample_idx in range(num_samples):
            hidden = torch.randn(hidden_dim)
            layer_id = sample_idx % num_layers
            position = sample_idx * 4

            _, expert_indices = router(hidden.unsqueeze(0).unsqueeze(0))
            active_experts = set(expert_indices.flatten().tolist())
            for e in active_experts:
                expert_hits[e] = expert_hits.get(e, 0) + 1

            # Use the tracker's get_active_experts as the "prefetch prediction"
            predicted_experts = tracker.get_active_experts(layer_id, position)
            next_layer = min(num_layers - 1, layer_id + 1)

            # Check if predicted layer matches next layer (for MoE layers)
            if tracker.is_moe_layer(layer_id):
                # Prefetch predicts next layer will be an MoE layer
                predicted_next_is_moe = tracker.is_moe_layer(next_layer)
                if predicted_next_is_moe:
                    layer_hits += 1

            # Check expert overlap between predicted experts and active experts
            predicted_expert_set = set(predicted_experts)
            overlap = len(predicted_expert_set & active_experts) / max(1, len(active_experts))
            expert_overlaps.append(overlap)

    total_expert_activations = sum(expert_hits.values())
    sparsity = 1.0 - (top_k / num_experts)

    hot_experts = sorted(expert_hits.items(), key=lambda x: x[1], reverse=True)[:4]
    # Map hot experts to their likely layers
    recommended_layers = [(e * (num_layers // num_experts)) % num_layers for e, _ in hot_experts]

    return MoEPrefetchCorrelationReport(
        num_samples=num_samples,
        mean_expert_overlap=float(sum(expert_overlaps) / max(1, len(expert_overlaps))),
        layer_prediction_accuracy=layer_hits / max(1, num_samples),
        prefetch_usefulness_score=0.0,  # Would need real prefetch tracking
        expert_sparsity_ratio=sparsity,
        recommended_prefetch_layers=recommended_layers,
    )


def report_to_dict(report: MoEPrefetchCorrelationReport) -> Dict:
    return {
        "num_samples": report.num_samples,
        "mean_expert_overlap": round(report.mean_expert_overlap, 4),
        "layer_prediction_accuracy": round(report.layer_prediction_accuracy, 4),
        "prefetch_usefulness_score": round(report.prefetch_usefulness_score, 4),
        "expert_sparsity_ratio": round(report.expert_sparsity_ratio, 4),
        "recommended_prefetch_layers": report.recommended_prefetch_layers,
    }