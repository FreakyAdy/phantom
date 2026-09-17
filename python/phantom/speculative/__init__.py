"""
PHANTOM SPECULATIVE — Heterogeneous Speculative Verification Runtime
=====================================================================
Lossless speculative inference engine designed to break the DDR5 memory
bandwidth ceiling on consumer hardware via batched CPU/RAM GEMM verification.

Core Architecture:
- DraftRunner: GPU VRAM resident draft model generating k candidate tokens (<10ms/tok)
- TargetVerifier: Heterogeneous target model evaluating candidate sequences in a single batched pass
- SpeculativeAcceptor: Mathematically lossless greedy & speculative sampling rejection logic
- SpeculativeKVCache: Dynamic KV cache with fast rewind/rollback on token rejection
- SpeculativeEngine: Orchestrator coordinating draft, verify, accept, and token streaming
"""

from phantom.speculative.acceptance import (
    AcceptanceResult,
    SpeculativeAcceptor,
    greedy_verify,
    speculative_sample_verify,
)
from phantom.speculative.kv_cache import SpeculativeKVCache
from phantom.speculative.draft_runner import DraftRunner
from phantom.speculative.target_verifier import TargetVerifier
from phantom.speculative.engine import SpeculativeEngine, SpeculativeMetrics

__all__ = [
    "AcceptanceResult",
    "SpeculativeAcceptor",
    "greedy_verify",
    "speculative_sample_verify",
    "SpeculativeKVCache",
    "DraftRunner",
    "TargetVerifier",
    "SpeculativeEngine",
    "SpeculativeMetrics",
]
