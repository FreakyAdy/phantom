"""
PHANTOM SPECULATIVE — Speculative KV Cache Manager
===================================================
Manages key-value cache states for both draft and target models during
speculative generation with zero-cost rewind/rollback on token rejection.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch


class SpeculativeKVCache:
    """
    Manages sequence-length tracking and cache rollback for speculative decoding.

    Supports:
    - Dynamic sequence length tracking (committed length vs speculative length)
    - Fast rollback of torch past_key_values tuples to the divergence point
    - Synchronization between draft model cache and target model cache
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.committed_len: int = 0
        self.speculative_len: int = 0
        self.target_past_kv: Optional[Tuple[Any, ...]] = None
        self.draft_past_kv: Optional[Tuple[Any, ...]] = None

    def initialize(self, prefix_len: int):
        """Initialize cache tracking for a new prompt."""
        self.committed_len = prefix_len
        self.speculative_len = prefix_len
        self.target_past_kv = None
        self.draft_past_kv = None

    @property
    def current_len(self) -> int:
        return self.committed_len

    def commit(self, num_new_tokens: int):
        """Commit accepted tokens into the permanent cache length."""
        self.committed_len += num_new_tokens
        self.speculative_len = self.committed_len

    def rollback(self, target_len: int):
        """
        Rewind cache tensors to target_len after a rejection.

        Args:
            target_len: The new total sequence length after accepting prefix + correction.
        """
        if self.target_past_kv is not None:
            self.target_past_kv = self._truncate_past_kv(self.target_past_kv, target_len)
        if self.draft_past_kv is not None:
            self.draft_past_kv = self._truncate_past_kv(self.draft_past_kv, target_len)
        self.committed_len = target_len
        self.speculative_len = target_len

    @staticmethod
    def _truncate_past_kv(past_kv: Tuple[Any, ...], target_len: int) -> Tuple[Any, ...]:
        """Truncate KV tensors along the sequence length dimension."""
        if past_kv is None:
            return None

        # Check for DynamicCache from transformers
        if hasattr(past_kv, "crop"):
            try:
                past_kv.crop(target_len)
                return past_kv
            except Exception:
                pass

        if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
            # Transformers DynamicCache style
            for idx in range(len(past_kv.key_cache)):
                if past_kv.key_cache[idx] is not None and past_kv.key_cache[idx].shape[-2] > target_len:
                    past_kv.key_cache[idx] = past_kv.key_cache[idx][..., :target_len, :]
                if past_kv.value_cache[idx] is not None and past_kv.value_cache[idx].shape[-2] > target_len:
                    past_kv.value_cache[idx] = past_kv.value_cache[idx][..., :target_len, :]
            return past_kv

        # Standard tuple of (key, value) pairs per layer
        truncated_layers = []
        for layer in past_kv:
            if isinstance(layer, (tuple, list)) and len(layer) >= 2:
                k, v = layer[0], layer[1]
                # Shape is typically [batch, num_heads, seq_len, head_dim]
                # seq_len is index -2
                if k is not None and k.shape[-2] > target_len:
                    k = k[..., :target_len, :]
                if v is not None and v.shape[-2] > target_len:
                    v = v[..., :target_len, :]
                truncated_layers.append((k, v) + tuple(layer[2:]))
            else:
                truncated_layers.append(layer)

        return tuple(truncated_layers)
