"""
PHANTOM RUNTIME — Python Package
================================
Heterogeneous Lossless Speculative Verification Engine.
"Break the memory bandwidth wall via batched speculative verification."
"""

__version__ = "1.0.0"
__author__ = "FreakyAdy"

from phantom.loader import (
    ModelFormat,
    ModelMeta,
    TensorMeta,
    detect_format,
    load_model_meta,
    stream_layers,
    GGUFLoader,
)

from phantom.speculative import (
    SpeculativeEngine,
    SpeculativeMetrics,
    DraftRunner,
    TargetVerifier,
    SpeculativeAcceptor,
    SpeculativeKVCache,
    AcceptanceResult,
    greedy_verify,
    speculative_sample_verify,
)

__all__ = [
    # Model loading
    "ModelFormat",
    "ModelMeta",
    "TensorMeta",
    "detect_format",
    "load_model_meta",
    "stream_layers",
    "GGUFLoader",
    # Speculative runtime
    "SpeculativeEngine",
    "SpeculativeMetrics",
    "DraftRunner",
    "TargetVerifier",
    "SpeculativeAcceptor",
    "SpeculativeKVCache",
    "AcceptanceResult",
    "greedy_verify",
    "speculative_sample_verify",
]