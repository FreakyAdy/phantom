"""Unit tests for PHANTOM v2 fused kernels."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from kernels.attention.fused_attention import fused_attention, fused_attention_reference
from kernels.ffn.fused_ffn import fused_ffn, fused_ffn_reference
from phantom.kernels.dispatch import KernelDispatch


def test_fused_attention_parity():
    batch, seq, hidden = 2, 8, 64
    x = torch.randn(batch, seq, hidden)
    q_w = torch.randn(hidden, hidden)
    k_w = torch.randn(hidden, hidden)
    v_w = torch.randn(hidden, hidden)

    ref = fused_attention_reference(x, q_w, k_w, v_w)
    fused = fused_attention(x, q_w, k_w, v_w, use_triton=False)

    assert ref.shape == fused.shape
    assert torch.allclose(ref, fused, atol=1e-4, rtol=1e-3)


def test_fused_ffn_parity():
    batch, seq, hidden, ffn = 2, 4, 64, 128
    x = torch.randn(batch, seq, hidden)
    w1 = torch.randn(ffn, hidden)
    w2 = torch.randn(hidden, ffn)

    ref = fused_ffn_reference(x, w1, w2)
    fused = fused_ffn(x, w1, w2, use_triton=False)

    assert ref.shape == fused.shape
    assert torch.allclose(ref, fused, atol=1e-4, rtol=1e-3)


def test_kernel_dispatch_fusion_enabled():
    dispatch = KernelDispatch(fusion_enabled=True)
    x = torch.randn(2, 64)
    w = torch.randn(64, 64)
    out = dispatch.ffn(x, w, w)
    assert out.shape == (2, 64)
    assert dispatch.fusion_calls >= 1


def test_kernel_dispatch_fusion_disabled():
    dispatch = KernelDispatch(fusion_enabled=False)
    x = torch.randn(2, 64)
    w = torch.randn(64, 64)
    out = dispatch.ffn(x, w, w)
    assert out.shape == (2, 64)
    assert dispatch.fallback_calls >= 1
