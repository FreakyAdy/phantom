"""
PHANTOM SPECULATIVE — Lossless Acceptance & Rejection Engine
============================================================
Implements provably distribution-preserving token verification algorithms:
1. Greedy Verification (Top-1 Match + Bonus Token)
2. Speculative Sampling Verification (Leviathan et al. / Chen et al. 2023)

Guarantees 0.00% statistical drift from the target model's output distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class AcceptanceResult:
    """Outcome of verifying k speculative candidate tokens."""
    accepted_tokens: List[int]
    emitted_tokens: List[int]       # accepted_tokens + [correction_token or bonus_token]
    num_accepted: int               # 0 <= num_accepted <= k
    num_drafted: int                # k
    bonus_token: Optional[int] = None
    correction_token: Optional[int] = None
    divergence_index: Optional[int] = None  # index in draft_tokens where mismatch occurred
    acceptance_rate: float = 0.0

    def __post_init__(self):
        if self.num_drafted > 0:
            self.acceptance_rate = self.num_accepted / float(self.num_drafted)
        else:
            self.acceptance_rate = 0.0


def greedy_verify(
    draft_tokens: List[int],
    target_logits: torch.Tensor,
) -> AcceptanceResult:
    """
    Verify draft tokens greedily against target model logits.

    Args:
        draft_tokens: List of k candidate token IDs proposed by draft model.
        target_logits: Target model logits of shape [k + 1, vocab_size] or [k, vocab_size].
                       target_logits[i] corresponds to predicting the token at position i.

    Returns:
        AcceptanceResult with exact token sequence and acceptance count.
    """
    k = len(draft_tokens)
    if k == 0:
        # No draft tokens, emit argmax of first position
        bonus = int(torch.argmax(target_logits[0], dim=-1).item())
        return AcceptanceResult(
            accepted_tokens=[],
            emitted_tokens=[bonus],
            num_accepted=0,
            num_drafted=0,
            bonus_token=bonus,
        )

    # Compute greedy predictions from target logits
    target_preds = torch.argmax(target_logits[:k], dim=-1).tolist()

    accepted: List[int] = []
    mismatch_idx: Optional[int] = None
    correction: Optional[int] = None

    for i in range(k):
        target_token = target_preds[i]
        draft_token = draft_tokens[i]

        if draft_token == target_token:
            accepted.append(draft_token)
        else:
            mismatch_idx = i
            correction = target_token
            break

    if mismatch_idx is not None:
        # Draft diverged at mismatch_idx: emit accepted prefix + target correction token
        emitted = accepted + [correction]
        return AcceptanceResult(
            accepted_tokens=accepted,
            emitted_tokens=emitted,
            num_accepted=len(accepted),
            num_drafted=k,
            correction_token=correction,
            divergence_index=mismatch_idx,
        )
    else:
        # All k draft tokens accepted! Sample bonus token from position k (if available)
        bonus = None
        if target_logits.shape[0] > k:
            bonus = int(torch.argmax(target_logits[k], dim=-1).item())
            emitted = accepted + [bonus]
        else:
            emitted = accepted

        return AcceptanceResult(
            accepted_tokens=accepted,
            emitted_tokens=emitted,
            num_accepted=k,
            num_drafted=k,
            bonus_token=bonus,
            divergence_index=None,
        )


def speculative_sample_verify(
    draft_tokens: List[int],
    draft_probs: torch.Tensor,
    target_logits: torch.Tensor,
    temperature: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> AcceptanceResult:
    """
    Verify draft tokens using stochastic speculative rejection sampling.

    Matches target probability distribution exactly with zero statistical loss.
    (Leviathan et al., 2023; Chen et al., 2023)

    Args:
        draft_tokens: List of k candidate token IDs.
        draft_probs: Draft model probability tensor of shape [k, vocab_size].
        target_logits: Target model logits of shape [k + 1, vocab_size].
        temperature: Sampling temperature (> 0.0).
        generator: Optional torch random generator.

    Returns:
        AcceptanceResult with mathematically unbiased emitted token sequence.
    """
    k = len(draft_tokens)
    if temperature <= 1e-4:
        return greedy_verify(draft_tokens, target_logits)

    target_probs = F.softmax(target_logits / temperature, dim=-1)

    accepted: List[int] = []
    mismatch_idx: Optional[int] = None
    correction: Optional[int] = None

    for i in range(k):
        tok = draft_tokens[i]
        p_t = target_probs[i, tok].item()
        q_d = draft_probs[i, tok].item()

        # Acceptance threshold: min(1.0, P(x) / Q(x))
        if q_d <= 1e-12:
            ratio = 1.0 if p_t > 0 else 0.0
        else:
            ratio = min(1.0, p_t / q_d)

        # Uniform random draw
        u = torch.rand(1, generator=generator).item()
        if u < ratio:
            accepted.append(tok)
        else:
            # Rejection: sample correction token from normalized positive difference (P - Q)+
            mismatch_idx = i
            diff = torch.clamp(target_probs[i] - draft_probs[i], min=0.0)
            diff_sum = diff.sum()
            if diff_sum > 1e-8:
                norm_diff = diff / diff_sum
                correction = int(torch.multinomial(norm_diff, num_samples=1, generator=generator).item())
            else:
                correction = int(torch.multinomial(target_probs[i], num_samples=1, generator=generator).item())
            break

    if mismatch_idx is not None:
        emitted = accepted + [correction]
        return AcceptanceResult(
            accepted_tokens=accepted,
            emitted_tokens=emitted,
            num_accepted=len(accepted),
            num_drafted=k,
            correction_token=correction,
            divergence_index=mismatch_idx,
        )
    else:
        # All k accepted, sample bonus token from target_probs[k]
        bonus = None
        if target_probs.shape[0] > k:
            bonus = int(torch.multinomial(target_probs[k], num_samples=1, generator=generator).item())
            emitted = accepted + [bonus]
        else:
            emitted = accepted

        return AcceptanceResult(
            accepted_tokens=accepted,
            emitted_tokens=emitted,
            num_accepted=k,
            num_drafted=k,
            bonus_token=bonus,
            divergence_index=None,
        )


class SpeculativeAcceptor:
    """Configurable acceptance engine supporting greedy and stochastic modes."""

    def __init__(
        self,
        temperature: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ):
        self.temperature = temperature
        self.generator = generator

    def verify(
        self,
        draft_tokens: List[int],
        target_logits: torch.Tensor,
        draft_probs: Optional[torch.Tensor] = None,
    ) -> AcceptanceResult:
        if self.temperature <= 1e-4 or draft_probs is None:
            return greedy_verify(draft_tokens, target_logits)
        return speculative_sample_verify(
            draft_tokens=draft_tokens,
            draft_probs=draft_probs,
            target_logits=target_logits,
            temperature=self.temperature,
            generator=self.generator,
        )
