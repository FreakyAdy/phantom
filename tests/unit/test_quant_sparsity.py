"""Unit tests for PHANTOM v2 Q3 quantization and adaptive sparsity."""

from __future__ import annotations

import torch

from phantom.quant.selective_q3 import SelectiveQ3Quantizer, apply_selective_q3
from phantom.sparsity.adaptive_gate import AdaptiveSparsityGate


def test_selective_q3_mlp_only():
    quantizer = SelectiveQ3Quantizer()
    sd = {
        "model.layers.0.mlp.gate_proj.weight": torch.randn(256, 512),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(512, 512),
    }
    report = quantizer.apply_to_state_dict(sd)
    assert report.mlp_layers_quantized == 1
    assert report.attention_layers_preserved == 1
    assert report.weight_traffic_reduction_pct > 0


def test_apply_selective_q3_disabled():
    sd = {"w": torch.randn(10, 10)}
    original = sd["w"].clone()
    report = apply_selective_q3(sd, enabled=False)
    assert report.mlp_layers_quantized == 0
    assert torch.equal(sd["w"], original)


def test_adaptive_sparsity_gate():
    gate = AdaptiveSparsityGate(hidden_dim=128, intermediate_dim=512, sparsity_target=0.4)
    hidden = torch.randn(4, 128)
    mask, report = gate.forward(hidden)
    assert mask.shape == (4, 512)
    assert 0.0 <= report.sparsity_fraction <= 1.0
    # Gate precision is 0.0 (unvalidated) - requires real measurement on validation data
    assert report.gate_precision == 0.0
    # Speedup estimate is 1.0 (no real sparse GEMM)
    assert report.mlp_speedup_estimate == 1.0


def test_sparse_ffn_execution():
    gate = AdaptiveSparsityGate(hidden_dim=64, intermediate_dim=128)
    hidden = torch.randn(2, 64)
    w1 = torch.randn(128, 64)
    w2 = torch.randn(64, 128)
    out, report = gate.apply_sparse_ffn(hidden, w1, w2)
    assert out.shape == (2, 64)
    assert report.neurons_total > 0
