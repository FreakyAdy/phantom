"""
PHANTOM SPECULATIVE — Unit & Correctness Test Suite
===================================================
Rigorous verification of the Heterogeneous Speculative Verification Subsystem:
- Greedy lossless acceptance and correction logic
- Speculative sampling distribution fidelity
- KV cache commit and rollback invariants
- End-to-end SpeculativeEngine generation and telemetry accounting
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from phantom.speculative.acceptance import (
    AcceptanceResult,
    SpeculativeAcceptor,
    greedy_verify,
    speculative_sample_verify,
)
from phantom.speculative.draft_runner import DraftRunner
from phantom.speculative.engine import SpeculativeEngine, SpeculativeMetrics
from phantom.speculative.kv_cache import SpeculativeKVCache
from phantom.speculative.target_verifier import TargetVerifier


def test_greedy_verify_perfect_match():
    """When all draft tokens match target argmax, emit all k + bonus token."""
    vocab_size = 100
    k = 4
    draft_tokens = [10, 20, 30, 40]

    # Construct target logits where argmax matches draft tokens
    target_logits = torch.zeros(k + 1, vocab_size)
    for i, tok in enumerate(draft_tokens):
        target_logits[i, tok] = 10.0

    # Bonus token at position k
    bonus_token = 99
    target_logits[k, bonus_token] = 10.0

    res = greedy_verify(draft_tokens, target_logits)

    assert res.num_accepted == k
    assert res.accepted_tokens == draft_tokens
    assert res.bonus_token == bonus_token
    assert res.correction_token is None
    assert res.divergence_index is None
    assert res.emitted_tokens == [10, 20, 30, 40, 99]
    assert res.acceptance_rate == 1.0


def test_greedy_verify_partial_mismatch():
    """When draft token at index 2 mismatches, accept 0..1, emit correction 2, discard 3."""
    vocab_size = 100
    k = 4
    draft_tokens = [10, 20, 30, 40]

    target_logits = torch.zeros(k + 1, vocab_size)
    target_logits[0, 10] = 10.0  # Match
    target_logits[1, 20] = 10.0  # Match
    target_logits[2, 77] = 10.0  # Target wanted 77, not 30! Mismatch!
    target_logits[3, 40] = 10.0

    res = greedy_verify(draft_tokens, target_logits)

    assert res.num_accepted == 2
    assert res.accepted_tokens == [10, 20]
    assert res.correction_token == 77
    assert res.divergence_index == 2
    assert res.bonus_token is None
    assert res.emitted_tokens == [10, 20, 77]
    assert res.acceptance_rate == 0.5


def test_greedy_verify_immediate_mismatch():
    """When the very first draft token mismatches, num_accepted is 0, emits correction."""
    vocab_size = 100
    k = 3
    draft_tokens = [10, 20, 30]

    target_logits = torch.zeros(k + 1, vocab_size)
    target_logits[0, 99] = 10.0  # Immediate mismatch at index 0

    res = greedy_verify(draft_tokens, target_logits)

    assert res.num_accepted == 0
    assert res.accepted_tokens == []
    assert res.correction_token == 99
    assert res.divergence_index == 0
    assert res.emitted_tokens == [99]
    assert res.acceptance_rate == 0.0


def test_speculative_sample_verify_distribution():
    """Verify that speculative sampling preserves the target distribution."""
    torch.manual_seed(42)
    vocab_size = 5

    # Define distinct target and draft probability distributions
    target_probs_vec = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1])
    draft_probs_vec = torch.tensor([0.3, 0.3, 0.1, 0.1, 0.2])

    target_logits = torch.log(target_probs_vec + 1e-8).unsqueeze(0).repeat(2, 1)
    draft_probs = draft_probs_vec.unsqueeze(0)

    counts = torch.zeros(vocab_size)
    num_trials = 1000

    for _ in range(num_trials):
        # Sample draft token from draft distribution
        draft_tok = int(torch.multinomial(draft_probs_vec, num_samples=1).item())
        res = speculative_sample_verify(
            draft_tokens=[draft_tok],
            draft_probs=draft_probs,
            target_logits=target_logits,
            temperature=1.0,
        )
        # First emitted token is either the accepted draft token or target correction
        first_token = res.emitted_tokens[0]
        counts[first_token] += 1

    empirical_dist = counts / counts.sum()
    # Cosine similarity between empirical distribution and target distribution
    cos_sim = F.cosine_similarity(empirical_dist.unsqueeze(0), target_probs_vec.unsqueeze(0)).item()
    assert cos_sim > 0.98, f"Empirical dist {empirical_dist} deviated from target {target_probs_vec}"


def test_speculative_kv_cache():
    """Test speculative KV cache initialization, tracking, and truncation rollback."""
    cache = SpeculativeKVCache()
    cache.initialize(prefix_len=10)

    assert cache.current_len == 10
    cache.commit(num_new_tokens=4)
    assert cache.current_len == 14

    # Create dummy tuple KV cache: 2 layers, shape [1, 8, 14, 64]
    k1 = torch.randn(1, 8, 14, 64)
    v1 = torch.randn(1, 8, 14, 64)
    k2 = torch.randn(1, 8, 14, 64)
    v2 = torch.randn(1, 8, 14, 64)
    cache.target_past_kv = ((k1, v1), (k2, v2))

    # Rollback to length 12
    cache.rollback(target_len=12)
    assert cache.current_len == 12
    assert cache.target_past_kv[0][0].shape == (1, 8, 12, 64)
    assert cache.target_past_kv[0][1].shape == (1, 8, 12, 64)


def test_speculative_engine_generate():
    """End-to-end integration test of SpeculativeEngine generation."""
    draft_runner = DraftRunner()
    target_verifier = TargetVerifier()
    engine = SpeculativeEngine(
        draft_runner=draft_runner,
        target_verifier=target_verifier,
        spec_k=5,
        temperature=0.0,
    )

    prompt = "Antigravity physics engine"
    output_text, metrics = engine.generate(prompt=prompt, max_new_tokens=32, k=5)

    assert isinstance(output_text, str)
    assert len(output_text) > 0
    assert metrics.total_tokens_generated == 32
    assert metrics.total_draft_steps > 0
    assert metrics.total_draft_tokens > 0
    assert metrics.total_accepted_tokens > 0
    assert metrics.mean_acceptance_rate > 0.0
    assert metrics.tokens_per_second > 0.0
    assert metrics.speedup_factor > 0.0
    assert metrics.total_weight_bytes_read > 0
    assert metrics.bytes_per_accepted_token_mb > 0.0


def test_speculative_engine_cpu_moe():
    """Verify that cpu_moe enables a 10x reduction in RAM weight bytes read."""
    draft_runner = DraftRunner()
    target_dense = TargetVerifier(num_layers=64, gpu_layers=14, ram_layers=50, cpu_moe=False)
    target_moe = TargetVerifier(num_layers=64, gpu_layers=14, ram_layers=50, cpu_moe=True)

    dense_engine = SpeculativeEngine(
        draft_runner=draft_runner,
        target_verifier=target_dense,
        spec_k=5,
        cpu_moe=False,
    )
    moe_engine = SpeculativeEngine(
        draft_runner=draft_runner,
        target_verifier=target_moe,
        spec_k=5,
        cpu_moe=True,
    )

    prompt = "MoE routing bandwidth test"
    _, dense_metrics = dense_engine.generate(prompt=prompt, max_new_tokens=16, k=5)
    _, moe_metrics = moe_engine.generate(prompt=prompt, max_new_tokens=16, k=5)

    assert dense_metrics.total_weight_bytes_read > 0
    assert moe_metrics.total_weight_bytes_read > 0
    # The bytes read for MoE must be approximately 1/10th of dense
    ratio = dense_metrics.total_weight_bytes_read / float(moe_metrics.total_weight_bytes_read)
    assert 9.0 <= ratio <= 11.0, f"Expected ~10x reduction, got ratio {ratio}"


def test_cli_argument_parsing():
    """Verify CLI parser options for v2 speculative flags."""
    from phantom.phantom_cli import build_parser

    parser = build_parser()

    args = parser.parse_args(["run", "qwen2.5-32b"])
    assert args.n_gpu_layers == 0
    assert args.spec_draft is None
    assert args.spec_k == 5
    assert args.cpu_moe is False
    assert args.spec_mode is None
    assert args.no_prefetch is False
    assert args.no_fusion is False

    args_custom = parser.parse_args([
        "run",
        "qwen2.5-32b",
        "-ngl", "16",
        "--spec-mode", "eagle",
        "--eagle-heads", "~/.phantom/eagle/test.pt",
        "--spec-k", "4",
        "--no-prefetch",
        "--enable-q3",
        "--cpu-moe",
    ])
    assert args_custom.n_gpu_layers == 16
    assert args_custom.spec_mode == "eagle"
    assert args_custom.eagle_heads == "~/.phantom/eagle/test.pt"
    assert args_custom.spec_k == 4
    assert args_custom.no_prefetch is True
    assert args_custom.enable_q3 is True
    assert args_custom.cpu_moe is True
