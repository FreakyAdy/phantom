"""
PHANTOM PLATFORM — Ollama Drop-in Compatibility Router
======================================================
Provides exact Ollama API endpoints (/api/generate, /api/chat, /api/tags, etc.)
allowing Open WebUI, Continue.dev, Cursor, and LangChain to switch seamlessly.
Now uses real llama.cpp backend for inference.
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from phantom.registry import ModelManager
from phantom.runtime import create_engine_for_model

ollama_router = APIRouter(prefix="/api", tags=["ollama_compat"])

# Global engine cache
_engine_cache: Dict[str, Any] = {}


class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    stream: bool = True
    options: Optional[Dict[str, Any]] = None


class OllamaChatMessage(BaseModel):
    role: str
    content: str


class OllamaChatRequest(BaseModel):
    model: str
    messages: List[OllamaChatMessage]
    stream: bool = True
    options: Optional[Dict[str, Any]] = None


class OllamaPullRequest(BaseModel):
    name: str
    insecure: bool = False
    stream: bool = True


class OllamaDeleteRequest(BaseModel):
    name: str


class OllamaShowRequest(BaseModel):
    name: str


def _get_engine(model_id: str):
    """Get or create llama.cpp engine for model."""
    if model_id not in _engine_cache:
        try:
            engine = create_engine_for_model(
                model_id=model_id,
                n_gpu_layers=14,  # default for 6GB VRAM
                n_ctx=4096,
                n_batch=512,
            )
            engine.load()
            _engine_cache[model_id] = engine
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Failed to load model {model_id}: {e}")
    return _engine_cache[model_id]


@ollama_router.get("/tags")
async def get_tags():
    """List local models in Ollama format."""
    mgr = ModelManager()
    models_info = mgr.list(format="json")

    ollama_models = []
    for m in models_info:
        ollama_models.append({
            "name": m["id"],
            "model": m["id"],
            "modified_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z"),
            "size": m.get("size_bytes", 0),
            "digest": f"sha256:{m['id']}",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": m.get("name", "unknown"),
                "families": [m.get("name", "unknown")],
                "parameter_size": m.get("parameters", "unknown"),
                "quantization_level": m.get("quantization", "Q4_K_M"),
            },
        })

    if not ollama_models:
        return {"models": []}

    return {"models": ollama_models}


@ollama_router.get("/ps")
async def get_running_models():
    """Running models in Ollama format."""
    running = []
    for model_id, engine in _engine_cache.items():
        metrics = engine.get_metrics()
        running.append({
            "name": model_id,
            "model": model_id,
            "size": metrics.total_weight_bytes,
            "digest": f"sha256:{model_id}",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": model_id,
                "families": [model_id],
                "parameter_size": f"{metrics.n_total_layers // 2}B",
                "quantization_level": "Q4_K_M",
            },
            "expires_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 3600)),
            "size_vram": int(metrics.vram_used_gb * 1024**3),
        })
    return {"models": running}


@ollama_router.post("/generate")
async def generate(req: OllamaGenerateRequest):
    """Ollama generate endpoint with streaming support using real llama.cpp inference."""
    engine = _get_engine(req.model)

    temperature = req.options.get("temperature", 0.7) if req.options else 0.7
    top_p = req.options.get("top_p", 0.95) if req.options else 0.95
    top_k = req.options.get("top_k", 40) if req.options else 40
    repeat_penalty = req.options.get("repeat_penalty", 1.1) if req.options else 1.1

    if not req.stream:
        response_text = ""
        for token in engine.generate(
            prompt=req.prompt,
            max_tokens=256,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stream=True,
        ):
            if isinstance(token, str):
                response_text += token

        metrics = engine.get_metrics()
        return {
            "model": req.model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z"),
            "response": response_text,
            "done": True,
            "total_duration": int(metrics.tokens_per_second * 1_000_000_000) if metrics.tokens_per_second > 0 else 0,
            "load_duration": 0,
            "prompt_eval_count": len(req.prompt.split()),
            "eval_count": metrics.total_tokens_generated,
            "eval_duration": int(metrics.tokens_per_second * 1_000_000_000) if metrics.tokens_per_second > 0 else 0,
        }

    async def _stream_gen() -> AsyncGenerator[str, None]:
        start_time = time.time()
        token_count = 0

        for token in engine.generate(
            prompt=req.prompt,
            max_tokens=256,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stream=True,
        ):
            if isinstance(token, str) and token:
                token_count += 1
                chunk = {
                    "model": req.model,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z"),
                    "response": token,
                    "done": False,
                }
                yield json.dumps(chunk) + "\n"

        metrics = engine.get_metrics()
        final_chunk = {
            "model": req.model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z"),
            "response": "",
            "done": True,
            "total_duration": int((time.time() - start_time) * 1_000_000_000),
            "eval_count": token_count,
            "eval_duration": int((time.time() - start_time) * 1_000_000_000),
        }
        yield json.dumps(final_chunk) + "\n"

    return StreamingResponse(_stream_gen(), media_type="application/x-ndjson")


@ollama_router.post("/chat")
async def chat(req: OllamaChatRequest):
    """Ollama chat endpoint with streaming support using real llama.cpp inference."""
    engine = _get_engine(req.model)

    # Convert messages to prompt
    user_msg = req.messages[-1].content if req.messages else ""
    system_msg = next((m.content for m in req.messages if m.role == "system"), "")
    prompt = f"{system_msg}\n\nUser: {user_msg}\nAssistant:" if system_msg else f"User: {user_msg}\nAssistant:"

    temperature = req.options.get("temperature", 0.7) if req.options else 0.7
    top_p = req.options.get("top_p", 0.95) if req.options else 0.95
    top_k = req.options.get("top_k", 40) if req.options else 40
    repeat_penalty = req.options.get("repeat_penalty", 1.1) if req.options else 1.1

    if not req.stream:
        response_text = ""
        for token in engine.generate(
            prompt=prompt,
            max_tokens=256,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stream=True,
        ):
            if isinstance(token, str):
                response_text += token

        metrics = engine.get_metrics()
        return {
            "model": req.model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z"),
            "message": {
                "role": "assistant",
                "content": response_text,
            },
            "done": True,
            "total_duration": int(metrics.tokens_per_second * 1_000_000_000) if metrics.tokens_per_second > 0 else 0,
            "prompt_eval_count": len(prompt.split()),
            "eval_count": metrics.total_tokens_generated,
        }

    async def _stream_chat() -> AsyncGenerator[str, None]:
        start_time = time.time()
        token_count = 0

        for token in engine.generate(
            prompt=prompt,
            max_tokens=256,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stream=True,
        ):
            if isinstance(token, str) and token:
                token_count += 1
                chunk = {
                    "model": req.model,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z"),
                    "message": {"role": "assistant", "content": token},
                    "done": False,
                }
                yield json.dumps(chunk) + "\n"

        metrics = engine.get_metrics()
        final_chunk = {
            "model": req.model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z"),
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "total_duration": int((time.time() - start_time) * 1_000_000_000),
            "eval_count": token_count,
        }
        yield json.dumps(final_chunk) + "\n"

    return StreamingResponse(_stream_chat(), media_type="application/x-ndjson")


@ollama_router.post("/show")
async def show(req: OllamaShowRequest):
    mgr = ModelManager()
    try:
        details = mgr.show(req.name)
        return {
            "license": "MIT",
            "modelfile": f"FROM {details.id}\nSYSTEM You are running on PHANTOM CORE\n",
            "parameters": "temperature 0.7\ntop_p 0.9\n",
            "template": "{{ .System }}\nUser: {{ .Prompt }}\nAssistant: ",
            "details": {
                "format": "gguf",
                "family": details.manifest.get("model_family", "llama"),
                "parameter_size": details.manifest.get("parameters", "unknown"),
                "quantization_level": details.manifest.get("quantization", "Q4_K_M"),
            },
        }
    except Exception:
        return {
            "modelfile": f"FROM {req.name}\n",
            "details": {"format": "gguf", "family": "llama", "parameter_size": "unknown"},
        }


@ollama_router.post("/pull")
async def pull(req: OllamaPullRequest):
    """Pull model via llama.cpp backend (downloads GGUF from HF)."""
    async def _pull_stream():
        try:
            engine = _get_engine(req.name)
            metrics = engine.get_metrics()
            yield json.dumps({"status": f"pulling {req.name}"}) + "\n"
            yield json.dumps({"status": f"downloaded {metrics.total_weight_bytes / (1024**3):.2f} GB"}) + "\n"
            yield json.dumps({"status": "success"}) + "\n"
        except Exception as e:
            yield json.dumps({"status": f"error: {e}"}) + "\n"

    return StreamingResponse(_pull_stream(), media_type="application/x-ndjson")


@ollama_router.delete("/delete")
async def delete(req: OllamaDeleteRequest):
    mgr = ModelManager()
    try:
        mgr.rm(req.name, force=True)
        if req.name in _engine_cache:
            _engine_cache[req.name].unload()
            del _engine_cache[req.name]
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))