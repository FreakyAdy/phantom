"""Structural smoke tests for the real llama.cpp runtime metrics surface.

These tests do not import llama_cpp and never download weights; they verify
the metrics plumbing that the CLI and benchmark harness depend on.
"""

from __future__ import annotations

from phantom.runtime import LlamaCppMetrics, LlamaCppEngine
from phantom.runtime.llamacpp_backend import _measure_ram_gb


def test_metrics_defaults_include_ram_and_draft():
    m = LlamaCppMetrics()
    assert m.ram_used_gb == 0.0
    assert m.total_weight_bytes == 0
    assert m.draft_model is False
    assert m.draft_model_id is None


def test_metrics_exported_from_runtime():
    assert LlamaCppMetrics.__name__ == "LlamaCppMetrics"


def test_engine_accepts_draft_params_without_llama():
    engine = LlamaCppEngine(model_path="x.gguf", draft_model_id="qwen2.5-0.5b")
    assert engine.draft_model_id == "qwen2.5-0.5b"
    assert engine.get_metrics().draft_model is False
    assert engine.get_metrics().draft_model_id is None


def test_engine_get_metrics_is_callable_api_from_cli_path():
    engine = LlamaCppEngine(model_path="x.gguf", n_gpu_layers=14)
    assert hasattr(engine, "get_metrics")
    m = engine.get_metrics()
    assert m.n_gpu_layers == 0 and m.n_total_layers == 0


def test_measure_ram_gb_returns_float():
    value = _measure_ram_gb()
    assert isinstance(value, float)
    assert value >= 0.0