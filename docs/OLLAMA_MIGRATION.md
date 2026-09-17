# PHANTOM — Ollama Migration Guide

> *"Ollama offloads layers greedily or runs CPU-only. PHANTOM orchestrates tiered memory (GPU VRAM + Host DDR5) to execute 30B–35B models without crashing on consumer hardware."*

## Drop-in API Compatibility

PHANTOM provides endpoint compatibility with Ollama via its local API Gateway (`http://localhost:11411`):

| Ollama Endpoint | PHANTOM Support | Notes |
|---|---|---|
| `POST /api/generate` | ✅ Full | Text completion with streaming |
| `POST /api/chat` | ✅ Full | Chat completion with streaming |
| `GET /api/tags` | ✅ Full | Lists installed and available models |
| `POST /api/pull` | ✅ Full | Pulls and stages models locally |
| `DELETE /api/delete` | ✅ Full | Removes model from local registry |
| `POST /api/show` | ✅ Full | Returns model profile and memory allocation |
| `GET /api/ps` | ✅ Full | Lists active models and memory tier distribution |

## Switching Open WebUI

Change one environment variable:
```bash
# Before (Ollama)
OLLAMA_BASE_URL=http://localhost:11434

# After (PHANTOM)
OLLAMA_BASE_URL=http://localhost:11411
```

## Switching Continue.dev

In `~/.continue/config.json`:
```json
{
  "models": [{
    "title": "PHANTOM — qwen2.5-coder:32b",
    "provider": "ollama",
    "model": "qwen2.5-coder:32b",
    "apiBase": "http://localhost:11411"
  }]
}
```

## Importing Ollama GGUF Models

```bash
# Locate existing GGUF blob in Ollama cache
ls ~/.ollama/models/blobs/

# Convert directly into PHANTOM
phantom convert ~/.ollama/models/blobs/<sha256> --output ~/.phantom/models/qwen2.5-coder-32b/
```

## Modelfile to Phantomfile Line-by-Line Migration

PHANTOM's `Phantomfile` engine is backward compatible with Ollama's `Modelfile`. You can use any existing `Modelfile` directly or take advantage of PHANTOM-specific runtime directives:

| Ollama `Modelfile` Directive | PHANTOM `Phantomfile` Directive | Purpose |
|---|---|---|
| `FROM llama3:8b` | `FROM qwen2.5-coder:32b` | Base model (PHANTOM tiers 30B–35B models across VRAM and RAM) |
| `SYSTEM """..."""` | `SYSTEM """..."""` | Model persona and instructions |
| `TEMPLATE """..."""` | `TEMPLATE """..."""` | Custom chat prompt templating |
| `PARAMETER temperature 0.7` | `PARAMETER temperature 0.7` | Standard inference hyperparameter |
| `PARAMETER stop "<|eot_id|>"`| `PARAMETER stop "<|eot_id|>"` | Stop sequence |
| *(Not supported in Ollama)* | `PHANTOM_PARAM moe_expert_routing on` | Sparse expert routing (evaluates only active experts) |
| *(Not supported in Ollama)* | `PHANTOM_PARAM speculative_verify on` | Batched verification amortizing DDR5 weight reads |
| *(Not supported in Ollama)* | `PLUGIN rag-connector` | Built-in local vector retrieval middleware |
| *(Not supported in Ollama)* | `PLUGIN tool-router` | OpenAI function calling & MCP tool servers |

**Migration Command:**
```bash
# Build from an existing Ollama Modelfile directly:
phantom create my-assistant -f Modelfile

# Or build from a Phantomfile:
phantom create coder-32b -f Phantomfile
```

## What PHANTOM Enables Beyond Standard Ollama Offload

1. **30B–35B Frontier Models on 6.0 GB VRAM**: Expands the parameter ceiling 4.88x beyond the 6.7B pure GPU baseline limit without triggering CUDA OOM crashes or OS page thrashing.
2. **MoE Sparse Throughput**: Runs 30.5B MoE architectures (`Qwen3-30B-A3B`) at **12.95 tok/s (Local)** and **24.79 tok/s (Cloud)** by computing only the active parameters per token.
3. **In-Place CPU SIMD Evaluation**: Avoids saturated PCIe bus transfers by evaluating host-resident layers directly in DDR5 RAM via multi-threaded vector kernels.
4. **Zero-Disk Capacity Planning**: Checks hardware memory distribution and estimated throughput via `phantom plan` without downloading gigabytes of weights.
5. **Batched Speculative Verification (Milestone 1.6)**: Amortizes DDR5 weight passes across multiple draft tokens, unlocking 2.4x–3.9x layer acceleration.


