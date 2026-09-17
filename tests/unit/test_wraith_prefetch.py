"""Unit tests for PHANTOM v2 Wraith prefetch."""

from __future__ import annotations

import time

import torch

from phantom.prefetch.wraith_v2 import AdaptivePrefetchScheduler, WraithV2Predictor


def test_wraith_predictor_output():
    predictor = WraithV2Predictor(hidden_dim=128, num_layers=32)
    hidden = torch.randn(128)
    layer_probs, urgency, confidence = predictor(hidden, layer_id=5, position=100)
    assert layer_probs.shape == (32,)
    assert 0.0 <= urgency <= 1.0
    assert 0.0 <= confidence <= 1.0


def test_prefetch_scheduler_hit_tracking():
    scheduler = AdaptivePrefetchScheduler(num_layers=32, confidence_threshold=0.0, enabled=True)
    scheduler.prefetch_layer_async(layer=10)
    time.sleep(0.05)
    hit = scheduler.check_staged(10)
    assert hit is True
    assert scheduler.metrics.total_prefetch_hits >= 1


def test_prefetch_should_not_trigger_for_vram_layers():
    scheduler = AdaptivePrefetchScheduler(num_layers=32, confidence_threshold=0.5, enabled=True)
    should = scheduler.should_prefetch(next_layer=5, confidence=0.9, vram_layers={5, 6, 7})
    assert should is False


def test_prefetch_disabled():
    scheduler = AdaptivePrefetchScheduler(enabled=False)
    scheduler.schedule_for_spec_round(
        hidden_state=torch.randn(512), current_layer=0, position=0,
        draft_tokens=[1, 2], vram_layers=set(),
    )
    assert scheduler.metrics.total_prefetch_requests == 0
