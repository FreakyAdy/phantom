from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


@dataclass
class LlamaCppMetrics:
    total_tokens_generated: int = 0
    tokens_per_second: float = 0.0
    ttft_ms: float = 0.0
    prompt_tokens: int = 0
    vram_used_gb: float = 0.0
    ram_used_gb: float = 0.0
    total_weight_bytes: int = 0
    n_gpu_layers: int = 0
    n_total_layers: int = 0
    draft_model: bool = False
    draft_model_id: str | None = None


def _measure_ram_gb() -> float:
    """Best-effort resident set size of the current process (GB)."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024**3)
    except Exception:
        return 0.0


class LlamaCppEngine:
    def __init__(
        self,
        model_path: str | Path,
        n_gpu_layers: int = 0,
        n_ctx: int = 4096,
        n_batch: int = 512,
        n_threads: int | None = None,
        verbose: bool = False,
        draft_model_path: str | Path | None = None,
        draft_model_id: str | None = None,
    ):
        self.model_path = Path(model_path)
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.n_threads = n_threads or os.cpu_count()
        self.verbose = verbose
        self.draft_model_path = Path(draft_model_path) if draft_model_path else None
        self.draft_model_id = draft_model_id

        self._llama: Any = None
        self._metrics = LlamaCppMetrics()

    def load(self) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise RuntimeError("llama-cpp-python not installed. Run: pip install llama-cpp-python") from e

        logger.info(f"Loading model: {self.model_path} (n_gpu_layers={self.n_gpu_layers})")
        start = time.perf_counter()

        draft_model = None
        if self.draft_model_path:
            logger.info(f"Loading draft model: {self.draft_model_path}")
            draft_model = Llama(
                model_path=str(self.draft_model_path),
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                n_batch=self.n_batch,
                n_threads=self.n_threads,
                verbose=self.verbose,
                use_mmap=True,
            )

        self._llama = Llama(
            model_path=str(self.model_path),
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            n_threads=self.n_threads,
            verbose=self.verbose,
            use_mmap=True,
            use_mlock=False,
            draft_model=draft_model,
        )

        load_time = time.perf_counter() - start
        logger.info(f"Model loaded in {load_time:.2f}s" + (" with draft model" if draft_model else ""))

        self._compute_weight_bytes()

        self._metrics.draft_model = self.draft_model_path is not None
        self._metrics.draft_model_id = self.draft_model_id or (
            self.draft_model_path.name if self.draft_model_path else None
        )

    def _compute_weight_bytes(self) -> None:
        if not self._llama:
            return
        try:
            n_total = self._llama.n_layers()
            self._metrics.n_total_layers = n_total
            self._metrics.n_gpu_layers = min(self.n_gpu_layers, n_total)

            total_bytes = 0
            for i in range(n_total):
                total_bytes += self._llama.get_layer_size(i)
            self._metrics.total_weight_bytes = total_bytes
        except Exception:
            pass

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: list[str] | None = None,
        stream: bool = False,
    ) -> Iterator[str] | str:
        if not self._llama:
            raise RuntimeError("Model not loaded. Call load() first.")

        self._metrics = LlamaCppMetrics()

        gen_start = time.perf_counter()
        first_token_time: float | None = None
        token_count = 0

        # llama.cpp handles speculative decoding internally when draft_model is set
        if hasattr(self._llama, 'draft_model') and self._llama.draft_model is not None:
            logger.info("Using llama.cpp native speculative decoding with draft model")

        if stream:
            return self._generate_stream(
                prompt, max_tokens, temperature, top_p, top_k,
                repeat_penalty, stop, gen_start, first_token_time, token_count
            )
        else:
            return self._generate_batch(
                prompt, max_tokens, temperature, top_p, top_k,
                repeat_penalty, stop, gen_start
            )

    def _generate_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        stop: list[str] | None,
        gen_start: float,
        first_token_time: float | None,
        token_count: int,
    ) -> Iterator[str]:
        stream = self._llama.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop or [],
            stream=True,
        )

        for chunk in stream:
            if first_token_time is None:
                first_token_time = time.perf_counter()
                self._metrics.ttft_ms = (first_token_time - gen_start) * 1000

            text = chunk["choices"][0]["text"]
            if text:
                token_count += 1
                yield text

        gen_time = time.perf_counter() - gen_start
        self._finalize_metrics(token_count, gen_time)

    def _generate_batch(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        stop: list[str] | None,
        gen_start: float,
    ) -> str:
        result = self._llama.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop or [],
            stream=False,
        )

        text = result["choices"][0]["text"]
        token_count = result["usage"]["completion_tokens"]
        prompt_tokens = result["usage"]["prompt_tokens"]

        gen_time = time.perf_counter() - gen_start
        self._finalize_metrics(token_count, gen_time, prompt_tokens)

        return text

    def _finalize_metrics(
        self,
        token_count: int,
        gen_time: float,
        prompt_tokens: int = 0,
    ) -> None:
        self._metrics.total_tokens_generated = token_count
        self._metrics.prompt_tokens = prompt_tokens
        self._metrics.tokens_per_second = token_count / gen_time if gen_time > 0 else 0.0

        try:
            import torch
            if torch.cuda.is_available():
                self._metrics.vram_used_gb = torch.cuda.memory_allocated() / (1024**3)
        except Exception:
            pass

        self._metrics.ram_used_gb = _measure_ram_gb()

    def get_metrics(self) -> LlamaCppMetrics:
        return self._metrics

    def unload(self) -> None:
        if self._llama:
            del self._llama
            self._llama = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def resolve_model_path(model_id: str, cache_dir: Path | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    cache = cache_dir or Path.home() / ".phantom" / "models"
    cache.mkdir(parents=True, exist_ok=True)

    if Path(model_id).exists():
        return Path(model_id)

    if ":" in model_id:
        repo, quant = model_id.split(":", 1)
        repo = repo.strip()
        quant = quant.strip()
    else:
        # Default quant for known models
        repo = model_id
        quant = "Q4_K_M"

    if repo == "qwen2.5-coder-32b" or "qwen2.5-coder" in repo.lower():
        return hf_hub_download(
            repo_id="bartowski/Qwen2.5-Coder-32B-Instruct-GGUF",
            filename=f"Qwen2.5-Coder-32B-Instruct-{quant.upper()}.gguf",
            cache_dir=cache,
        )
    elif repo == "llama3" or "llama-3" in repo.lower():
        if "70b" in repo.lower():
            return hf_hub_download(
                repo_id="bartowski/Meta-Llama-3-70B-Instruct-GGUF",
                filename=f"Meta-Llama-3-70B-Instruct-{quant.upper()}.gguf",
                cache_dir=cache,
            )
    elif repo == "qwen3" or "qwen-3" in repo.lower():
        if "30b" in repo.lower() or "a3b" in repo.lower():
            return hf_hub_download(
                repo_id="bartowski/Qwen3-30B-A3B-Instruct-GGUF",
                filename=f"Qwen3-30B-A3B-Instruct-{quant.upper()}.gguf",
                cache_dir=cache,
            )
    elif repo == "smollm" or "smollm-135m" in repo.lower():
        return hf_hub_download(
            repo_id="unsloth/SmolLM2-135M-Instruct-GGUF",
            filename=f"SmolLM2-135M-Instruct-{quant.upper()}.gguf",
            cache_dir=cache,
        )
    elif repo == "qwen2.5" or "qwen2.5-0.5b" in repo.lower():
        return hf_hub_download(
            repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            filename=f"qwen2.5-0.5b-instruct-{quant.upper()}.gguf",
            cache_dir=cache,
        )

    return hf_hub_download(repo_id=model_id, filename=f"{model_id}-{quant.upper()}.gguf", cache_dir=cache)


def create_engine_for_model(
    model_id: str,
    n_gpu_layers: int = 0,
    n_ctx: int = 4096,
    n_batch: int = 512,
    speculative: bool = False,
    draft_model_id: str | None = None,
    cache_dir: Path | None = None,
) -> LlamaCppEngine:
    model_path = resolve_model_path(model_id, cache_dir)

    draft_path = None
    if speculative and draft_model_id:
        draft_path = resolve_model_path(draft_model_id, cache_dir)

    return LlamaCppEngine(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        n_batch=n_batch,
        draft_model_path=draft_path,
        draft_model_id=draft_model_id,
    )