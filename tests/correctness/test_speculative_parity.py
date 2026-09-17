"""
PHANTOM v2 — Speculative Decoding Quality & Parity Gates
========================================================
Validates lossless greedy acceptance, token agreement, and acceptance rate thresholds.
"""

from __future__ import annotations

import argparse
import sys

import pytest
import torch

from phantom.speculative.acceptance import greedy_verify
from phantom.speculative.draft_runner import DraftRunner
from phantom.speculative.eagle_heads import EagleDrafter
from phantom.speculative.engine import SpeculativeEngine
from phantom.speculative.target_verifier import TargetVerifier


def test_greedy_speculative_lossless():
    """Greedy spec with matching draft/target must emit k + bonus tokens."""
    k = 5
    draft_tokens = [10, 20, 30, 40, 50]
    target_logits = torch.zeros(k + 1, 100)
    for i, tok in enumerate(draft_tokens):
        target_logits[i, tok] = 100.0
    target_logits[k, 99] = 100.0

    result = greedy_verify(draft_tokens, target_logits)
    assert result.num_accepted == k
    assert result.bonus_token == 99
    assert len(result.emitted_tokens) == k + 1
    assert result.acceptance_rate == 1.0


def test_token_agreement_eagle_vs_draft():
    """EAGLE and draft modes produce valid output with acceptance > 0."""
    prompt = "Quality validation test prompt for speculative decoding"
    for spec_mode, drafter in [
        ("eagle", EagleDrafter(hidden_dim=128, vocab_size=1000, k=5)),
        ("draft", DraftRunner()),
    ]:
        engine = SpeculativeEngine(
            draft_runner=drafter,
            target_verifier=TargetVerifier(hidden_dim=128, num_layers=32, gpu_layers=8, ram_layers=24),
            spec_k=5,
            spec_mode=spec_mode,
        )
        text, metrics = engine.generate(prompt=prompt, max_new_tokens=16, k=5)
        assert isinstance(text, str)
        assert metrics.total_tokens_generated == 16
        assert metrics.mean_acceptance_rate > 0.0
        assert metrics.tokens_per_second > 0.0


def test_acceptance_rate_threshold():
    """Simulated verifier should achieve >= 65% acceptance (blueprint minimum)."""
    engine = SpeculativeEngine(
        draft_runner=DraftRunner(),
        target_verifier=TargetVerifier(),
        spec_k=5,
        spec_mode="draft",
    )
    _, metrics = engine.generate(
        prompt="Acceptance rate validation",
        max_new_tokens=64,
        k=5,
    )
    assert metrics.mean_acceptance_rate >= 0.65, (
        f"Acceptance rate {metrics.mean_acceptance_rate:.2%} below 65% threshold"
    )


def test_v2_engine_telemetry_fields():
    """v2 engine exposes prefetch and fusion telemetry."""
    from phantom.prefetch.wraith_v2 import AdaptivePrefetchScheduler
    from phantom.kernels.dispatch import KernelDispatch

    engine = SpeculativeEngine(
        draft_runner=EagleDrafter(hidden_dim=128, vocab_size=1000, k=3),
        target_verifier=TargetVerifier(hidden_dim=128),
        spec_mode="eagle",
        prefetch_scheduler=AdaptivePrefetchScheduler(enabled=True),
        kernel_dispatch=KernelDispatch(fusion_enabled=True),
    )
    _, metrics = engine.generate(prompt="Telemetry test", max_new_tokens=8, k=3)
    assert metrics.spec_mode == "eagle"
    assert hasattr(metrics, "prefetch_hit_rate")
    assert hasattr(metrics, "fusion_calls")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    sys.exit(pytest.main([__file__, "-q"] + (["-x"] if args.quick else [])))


if __name__ == "__main__":
    main()
