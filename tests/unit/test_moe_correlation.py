"""Unit tests for MoE expert routing / Wraith prefetch correlation."""

from phantom.prefetch.moe_correlation import (
    analyze_moe_prefetch_correlation,
    report_to_dict,
)


def test_moe_correlation_returns_valid_metrics():
    report = analyze_moe_prefetch_correlation(num_samples=64, seed=0)
    d = report_to_dict(report)

    assert report.num_samples == 64
    assert 0.0 <= report.mean_expert_overlap <= 1.0
    assert 0.0 <= report.layer_prediction_accuracy <= 1.0
    assert report.expert_sparsity_ratio > 0.0
    assert len(report.recommended_prefetch_layers) > 0
    assert "prefetch_usefulness_score" in d
