"""
PHANTOM v2 — Speculative Runtime Model Loader
==============================================
Loads target model + EAGLE heads + optional draft model for live inference.
Zero simulation fallback when weights are available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import torch

from phantom.speculative.draft_runner import DraftRunner
from phantom.speculative.eagle_heads import EagleDrafter, EagleHeads
from phantom.speculative.engine import SpeculativeEngine
from phantom.speculative.target_verifier import TargetVerifier

logger = logging.getLogger(__name__)


@dataclass
class SpeculativeRuntimeConfig:
    """Configuration for PHANTOM v2 speculative runtime."""
    model_id: str
    spec_mode: str = "eagle"  # eagle | draft
    spec_k: int = 5
    n_gpu_layers: int = 14
    eagle_heads_path: Optional[str] = None
    draft_model_id: Optional[str] = None
    prefetch_enabled: bool = True
    fusion_enabled: bool = True
    q3_enabled: bool = False
    sparsity_enabled: bool = False
    cpu_moe: bool = False
    temperature: float = 0.0
    hidden_dim: int = 5120
    vocab_size: int = 32000
    num_layers: int = 64


def _load_gguf_model(
    gguf_path: Path,
    n_gpu_layers: int = 0,
) -> Tuple[Any, Any]:
    """Load GGUF model and tokenizer via HuggingFace transformers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from phantom.loader import patch_transformers_gguf_gpu
        patch_transformers_gguf_gpu()
    except ImportError:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_kwargs = {"low_cpu_mem_usage": True}
    if device == "cuda":
        load_kwargs["torch_dtype"] = torch.bfloat16
    try:
        import accelerate  # noqa: F401
        load_kwargs["device_map"] = "auto"
    except ImportError:
        pass

    tokenizer = AutoTokenizer.from_pretrained(str(gguf_path.parent), gguf_file=gguf_path.name)
    model = AutoModelForCausalLM.from_pretrained(
        str(gguf_path.parent), gguf_file=gguf_path.name, **load_kwargs,
    )
    if "device_map" not in load_kwargs:
        model.to(device)
    model.eval()
    return model, tokenizer


def find_gguf_path(model_id: str, find_fn: Optional[Callable[[str], Optional[Path]]] = None) -> Optional[Path]:
    """Locate GGUF file for model_id."""
    if find_fn is not None:
        return find_fn(model_id)

    candidates = [
        Path(model_id),
        Path.home() / ".phantom" / "models" / model_id / f"{model_id}.gguf",
        Path.home() / ".cache" / "phantom" / "models" / model_id,
    ]
    for c in candidates:
        if c.is_file() and c.suffix == ".gguf":
            return c
        if c.is_dir():
            ggufs = list(c.glob("*.gguf"))
            if ggufs:
                return ggufs[0]
    return None


def load_eagle_drafter(
    config: SpeculativeRuntimeConfig,
    tokenizer: Any = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> EagleDrafter:
    """Load or initialize EAGLE-3 drafter."""
    checkpoint = config.eagle_heads_path
    if checkpoint:
        ckpt_path = Path(checkpoint.replace("~", str(Path.home())))
        if ckpt_path.exists():
            return EagleDrafter(
                tokenizer=tokenizer,
                hidden_dim=config.hidden_dim,
                vocab_size=config.vocab_size,
                k=config.spec_k,
                device=device,
                checkpoint_path=ckpt_path,
            )

    default_ckpt = Path.home() / ".phantom" / "eagle" / f"{config.model_id.replace(':', '_')}.pt"
    if default_ckpt.exists():
        return EagleDrafter(
            tokenizer=tokenizer,
            hidden_dim=config.hidden_dim,
            vocab_size=config.vocab_size,
            k=config.spec_k,
            device=device,
            checkpoint_path=default_ckpt,
        )

    logger.warning("No EAGLE checkpoint found; using randomly initialized heads")
    return EagleDrafter(
        tokenizer=tokenizer,
        hidden_dim=config.hidden_dim,
        vocab_size=config.vocab_size,
        k=config.spec_k,
        device=device,
    )


def build_speculative_engine(
    config: SpeculativeRuntimeConfig,
    target_model: Any,
    tokenizer: Any,
    find_gguf_fn: Optional[Callable[[str], Optional[Path]]] = None,
) -> SpeculativeEngine:
    """
    Build fully configured SpeculativeEngine with live weights.

    Raises RuntimeError if target_model is None (no simulation in production path).
    """
    if target_model is None:
        raise RuntimeError(
            f"Target model '{config.model_id}' weights could not be loaded. "
            "Verify model is installed with 'phantom list' or 'phantom pull'."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_layers = min(config.n_gpu_layers, config.num_layers)
    ram_layers = max(0, config.num_layers - gpu_layers)

    verifier = TargetVerifier(
        model=target_model,
        tokenizer=tokenizer,
        device=device,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        gpu_layers=gpu_layers,
        ram_layers=ram_layers,
        cpu_moe=config.cpu_moe,
    )

    if config.spec_mode == "draft":
        draft_model = None
        draft_tokenizer = tokenizer
        if config.draft_model_id:
            draft_path = find_gguf_path(config.draft_model_id, find_gguf_fn)
            if draft_path:
                draft_model, draft_tokenizer = _load_gguf_model(draft_path, n_gpu_layers=999)
        drafter = DraftRunner(
            model=draft_model,
            tokenizer=draft_tokenizer,
            device=device,
            model_name=config.draft_model_id or "qwen2.5-0.5b",
        )
    else:
        drafter = load_eagle_drafter(config, tokenizer, device)

    from phantom.prefetch.wraith_v2 import AdaptivePrefetchScheduler
    from phantom.kernels.dispatch import KernelDispatch

    prefetch = AdaptivePrefetchScheduler(
        num_layers=config.num_layers,
        enabled=config.prefetch_enabled,
    )
    kernels = KernelDispatch(fusion_enabled=config.fusion_enabled)

    engine = SpeculativeEngine(
        draft_runner=drafter,
        target_verifier=verifier,
        spec_k=config.spec_k,
        temperature=config.temperature,
        cpu_moe=config.cpu_moe,
        spec_mode=config.spec_mode,
        prefetch_scheduler=prefetch,
        kernel_dispatch=kernels,
        q3_enabled=config.q3_enabled,
        sparsity_enabled=config.sparsity_enabled,
    )
    return engine


def load_speculative_pair(
    config: SpeculativeRuntimeConfig,
    find_gguf_fn: Optional[Callable[[str], Optional[Path]]] = None,
) -> SpeculativeEngine:
    """Load target model and build speculative engine."""
    gguf_path = find_gguf_path(config.model_id, find_gguf_fn)
    if gguf_path is None:
        raise RuntimeError(f"GGUF not found for model '{config.model_id}'")

    target_model, tokenizer = _load_gguf_model(gguf_path, config.n_gpu_layers)
    return build_speculative_engine(config, target_model, tokenizer, find_gguf_fn)
