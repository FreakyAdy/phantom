"""Unit tests for PHANTOM v2 EAGLE-3 heads."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from phantom.speculative.eagle_heads import DEFAULT_FUSION_LAYERS, EagleDrafter, EagleHeads
from phantom.speculative.eagle_train import (
    EagleDataset,
    EagleTrainingExample,
    collect_training_data_synthetic,
    train_eagle_heads,
)
from torch.utils.data import DataLoader


def _make_features(hidden_dim: int = 128) -> dict:
    return {f"h{layer}": torch.randn(1, hidden_dim) for layer in DEFAULT_FUSION_LAYERS}


def test_eagle_heads_forward():
    eagle = EagleHeads(hidden_dim=128, vocab_size=1000, k=5)
    features = _make_features(128)
    logits = eagle(features)
    assert logits.shape == (5, 1, 1000) or logits.shape == (5, 1000)


def test_eagle_heads_param_count():
    eagle = EagleHeads(hidden_dim=512, vocab_size=32000, k=5)
    assert eagle.param_count > 0
    assert eagle.memory_mb_fp16 < 200  # should be ~50 MB at full scale


def test_eagle_drafter_generate():
    drafter = EagleDrafter(hidden_dim=128, vocab_size=1000, k=5)
    drafter.set_features(_make_features(128))
    prefix = torch.tensor([[1, 2, 3]])
    tokens, probs, _, ms = drafter.generate_draft(prefix, k=5)
    assert len(tokens) == 5
    assert probs is not None
    assert probs.shape[0] == 5
    assert ms >= 0


def test_eagle_checkpoint_roundtrip():
    eagle = EagleHeads(hidden_dim=128, vocab_size=1000, k=3)
    drafter = EagleDrafter(eagle_heads=eagle, k=3)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "eagle.pt"
        drafter.save_checkpoint(path)

        drafter2 = EagleDrafter(hidden_dim=128, vocab_size=1000, k=3, checkpoint_path=path)
        drafter2.set_features(_make_features(128))
        tokens, _, _, _ = drafter2.generate_draft(torch.tensor([[42]]), k=3)
        assert len(tokens) == 3


def test_eagle_training_loop():
    examples = collect_training_data_synthetic(num_examples=32, hidden_dim=64, k=3, vocab_size=500)
    dataset = EagleDataset(examples)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    eagle = EagleHeads(hidden_dim=64, vocab_size=500, k=3)
    history = train_eagle_heads(eagle, loader, num_epochs=2, device="cpu")
    assert "epoch_0_loss" in history
    assert history["epoch_0_loss"] > 0
