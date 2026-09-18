"""
PHANTOM CORE — OpenAI-Compatible REST API Server
==================================================
Full OpenAI-compatible API server for PHANTOM CORE inference engine.

Implements:
    POST /v1/chat/completions    — Chat completions (streaming + non-streaming)
    POST /v1/completions         — Text completions
    GET  /v1/models              — List available models
    GET  /v1/health              — Health check
    GET  /v1/metrics             — PHANTOM CORE performance metrics (extension)
    WebSocket /v1/stream         — Real-time token streaming

Now uses llama.cpp backend for real inference.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import structlog
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from phantom.runtime import create_engine_for_model

logger = structlog.get_logger(__name__)

# Global engine cache
_engine_cache: Dict[str, Any] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Request/Response models (OpenAI-compatible)
# ─────────────────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str = Field(..., description="Role: 'system', 'user', or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="phantom-core-model")
    messages: List[Message]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1)
    max_tokens: int = Field(default=2048, ge=1)
    stream: bool = Field(default=False)
    repetition_penalty: float = Field(default=1.1, ge=1.0)
    stop: Optional[Union[str, List[str]]] = None


class CompletionRequest(BaseModel):
    model: str = Field(default="phantom-core-model")
    prompt: Union[str, List[str]]
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = Field(default=False)
    stop: Optional[Union[str, List[str]]] = None


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str = "stop"


class ChatCompletionChunkDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]


class PhantomMetrics(BaseModel):
    """PHANTOM CORE extension — live performance metrics."""
    vram_mb: float
    vram_total_mb: float
    ram_mb: float
    nvme_mb: float
    layer_residency: Dict[str, List[int]]
    wraith_accuracy_pct: float
    kv_compression_ratio: float
    active_sparsity_pct: float
    tok_per_sec: float
    thermal_state: str
    throttle_active: bool
    uptime_sec: float
    total_requests: int
    active_requests: int


# ─────────────────────────────────────────────────────────────────────────────
# Engine management
# ─────────────────────────────────────────────────────────────────────────────

def _get_engine(model_id: str):
    """Get or create llama.cpp engine for model."""
    if model_id not in _engine_cache:
        try:
            engine = create_engine_for_model(
                model_id=model_id,
                n_gpu_layers=14,
                n_ctx=4096,
                n_batch=512,
            )
            engine.load()
            _engine_cache[model_id] = engine
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Failed to load model {model_id}: {e}")
    return _engine_cache[model_id]


def _format_chat_prompt(messages: List[Message]) -> str:
    """Format chat messages into a single prompt string."""
    parts = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"<|system|>\n{msg.content} ")
        elif msg.role == "user":
            parts.append(f"<|user|>\n{msg.content} ")
        elif msg.role == "assistant":
            parts.append(f"<|assistant|>\n{msg.content} ")
    parts.append("<|assistant|>")
    return "\n".join(parts)


def _estimate_tokens(text: str) -> int:
    """Quick token count estimate (4 chars ≈ 1 token)."""
    return max(1, len(text) // 4)


# ─────────────────────────────────────────────────────────────────────────────
# Application state
# ─────────────────────────────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.start_time = time.time()
        self.total_requests = 0
        self.active_requests = 0
        self.loaded_models: List[str] = []
        self.request_semaphore = asyncio.Semaphore(
            int(os.environ.get("PHANTOM_MAX_CONCURRENT", "1"))
        )


app_state = AppState()


# ─────────────────────────────────────────────────────────────────────────────
# App lifespan
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    logger.info("phantom_api_starting")
    yield  # App running
    # Cleanup engines
    for engine in _engine_cache.values():
        try:
            engine.unload()
        except Exception:
            pass
    _engine_cache.clear()
    logger.info("phantom_api_shutdown")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PHANTOM CORE API",
    description="OpenAI-compatible inference API for the PHANTOM CORE engine.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/v1/health")
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - app_state.start_time
    return {
        "status": "ok",
        "engine": "PHANTOM CORE (llama.cpp backend)",
        "uptime_sec": round(uptime, 1),
        "active_requests": app_state.active_requests,
        "loaded_models": list(_engine_cache.keys()),
    }


@app.get("/v1/models")
async def list_models():
    """List available models. OpenAI-compatible response."""
    models = list(_engine_cache.keys()) or ["phantom-core-model"]
    return {
        "object": "list",
        "data": [
            {
                "id": m,
                "object": "model",
                "created": int(app_state.start_time),
                "owned_by": "phantom-core",
                "permission": [],
                "root": m,
                "parent": None,
            }
            for m in models
        ],
    }


@app.get("/v1/metrics", response_model=PhantomMetrics)
async def get_metrics():
    """
    PHANTOM CORE extension endpoint — returns live engine performance metrics.
    """
    # Aggregate metrics from all loaded engines
    total_vram = 0.0
    total_ram = 0.0
    total_weight = 0
    total_tok_sec = 0.0
    model_count = len(_engine_cache)

    for engine in _engine_cache.values():
        metrics = engine.get_metrics()
        total_vram += metrics.vram_used_gb * 1024
        total_ram += metrics.ram_used_gb * 1024
        total_weight += metrics.total_weight_bytes
        total_tok_sec += metrics.tokens_per_second

    uptime = time.time() - app_state.start_time

    return PhantomMetrics(
        vram_mb=total_vram,
        vram_total_mb=total_vram,
        ram_mb=total_ram,
        nvme_mb=0.0,
        layer_residency={"vram": [], "ram": [], "nvme": []},
        wraith_accuracy_pct=0.0,
        kv_compression_ratio=1.0,
        active_sparsity_pct=0.0,
        tok_per_sec=round(total_tok_sec, 2),
        thermal_state="nominal",
        throttle_active=False,
        uptime_sec=round(uptime, 1),
        total_requests=app_state.total_requests,
        active_requests=app_state.active_requests,
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.

    Supports both streaming (stream=true) and non-streaming responses.
    Uses llama.cpp backend for real inference.
    """
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    prompt = _format_chat_prompt(request.messages)
    prompt_tokens = _estimate_tokens(prompt)

    engine = _get_engine(request.model)

    app_state.total_requests += 1

    temperature = request.temperature
    top_p = request.top_p
    top_k = request.top_k
    repetition_penalty = request.repetition_penalty
    stop_sequences = ([request.stop] if isinstance(request.stop, str) else request.stop) or []

    if request.stream:
        async def generate_stream() -> AsyncGenerator[bytes, None]:
            app_state.active_requests += 1
            completion_tokens = 0
            start_time = time.time()
            try:
                async with app_state.request_semaphore:
                    for token in engine.generate(
                        prompt=prompt,
                        max_tokens=request.max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repeat_penalty=repetition_penalty,
                        stop=stop_sequences,
                        stream=True,
                    ):
                        if isinstance(token, str) and token:
                            completion_tokens += 1
                            chunk = ChatCompletionChunk(
                                id=request_id,
                                created=created,
                                model=request.model,
                                choices=[
                                    ChatCompletionChunkChoice(
                                        index=0,
                                        delta=ChatCompletionChunkDelta(content=token),
                                        finish_reason=None,
                                    )
                                ],
                            )
                            yield f"data: {chunk.model_dump_json()}\n\n".encode()

                # Final chunk with finish_reason
                final_chunk = ChatCompletionChunk(
                    id=request_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(),
                            finish_reason="stop",
                        )
                    ],
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n".encode()
                yield b"data: [DONE]\n\n"

            except Exception as e:
                logger.error("stream_error", request_id=request_id, error=str(e))
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n".encode()
            finally:
                app_state.active_requests -= 1

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    else:
        # Non-streaming
        app_state.active_requests += 1
        start_time = time.time()
        try:
            async with app_state.request_semaphore:
                response_text = ""
                for token in engine.generate(
                    prompt=prompt,
                    max_tokens=request.max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repetition_penalty,
                    stop=stop_sequences,
                    stream=True,
                ):
                    if isinstance(token, str):
                        response_text += token
        finally:
            app_state.active_requests -= 1

        metrics = engine.get_metrics()
        completion_tokens = metrics.total_tokens_generated

        return ChatCompletionResponse(
            id=request_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """
    OpenAI-compatible text completions endpoint.

    Accepts a raw prompt string or list of prompts.
    """
    request_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    prompt_text = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
    prompt_tokens = _estimate_tokens(prompt_text)

    engine = _get_engine(request.model)

    app_state.total_requests += 1
    app_state.active_requests += 1
    start_time = time.time()
    try:
        async with app_state.request_semaphore:
            response_text = ""
            for token in engine.generate(
                prompt=prompt_text,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stream=True,
            ):
                if isinstance(token, str):
                    response_text += token
    finally:
        app_state.active_requests -= 1

    metrics = engine.get_metrics()
    completion_tokens = metrics.total_tokens_generated

    return {
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "text": response_text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.websocket("/v1/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time token streaming.

    Client sends: {"prompt": "...", "max_tokens": 512, "temperature": 0.7}
    Server streams: {"token": "...", "done": false}
    Server closes: {"token": "", "done": true}
    """
    await websocket.accept()
    logger.info("websocket_connected")

    try:
        while True:
            data = await websocket.receive_json()
            prompt = data.get("prompt", "")
            model = data.get("model", "phantom-core-model")
            if not prompt:
                await websocket.send_json({"error": "No prompt provided"})
                continue

            engine = _get_engine(model)

            app_state.total_requests += 1
            app_state.active_requests += 1
            try:
                async with app_state.request_semaphore:
                    for token in engine.generate(
                        prompt=prompt,
                        max_tokens=data.get("max_tokens", 512),
                        temperature=data.get("temperature", 0.7),
                        top_p=data.get("top_p", 0.9),
                        stream=True,
                    ):
                        if isinstance(token, str):
                            await websocket.send_json({"token": token, "done": False})

                await websocket.send_json({"token": "", "done": True})

            except WebSocketDisconnect:
                break
            except Exception as e:
                await websocket.send_json({"error": str(e), "done": True})
            finally:
                app_state.active_requests -= 1

    except WebSocketDisconnect:
        logger.info("websocket_disconnected")
    except Exception as e:
        logger.error("websocket_error", error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Exception handlers
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled_exception", error=str(exc), path=str(request.url))
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": type(exc).__name__,
                "code": 500,
            }
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(
    host: str = "0.0.0.0",
    port: int = 8080,
    workers: int = 1,
    log_level: str = "info",
) -> None:
    """
    Start the PHANTOM CORE API server.

    Args:
        host:      Bind address (default 0.0.0.0).
        port:      Listen port (default 8080).
        workers:   Number of worker processes.
        log_level: Uvicorn log level.
    """
    import uvicorn

    print("\n╔══════════════════════════════════════════════════╗")
    print("║   PHANTOM CORE API Server — Run the Unreachable  ║")
    print(f"║   Listening on http://{host}:{port}              ║")
    print("╚══════════════════════════════════════════════════╝\n")

    uvicorn.run(
        "phantom.api.openai_compat:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        access_log=True,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PHANTOM CORE API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    main(host=args.host, port=args.port, log_level=args.log_level)