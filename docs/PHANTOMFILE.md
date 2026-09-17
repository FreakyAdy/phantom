# Phantomfile Language Specification

The **Phantomfile** is a declarative configuration file format for building custom model personas on PHANTOM. It is backward-compatible with Ollama `Modelfile`.

## Syntax Directives

### `FROM <model>`
Specifies the base model reference.
```dockerfile
FROM qwen2.5-coder:32b
```

### `SYSTEM <text>`
Specifies the system prompt. Supports single-line or triple-quoted multi-line strings:
```dockerfile
SYSTEM """
You are a senior systems architect specializing in high-performance computing.
Always provide detailed technical answers with code examples.
"""
```

### `PARAMETER <key> <value>`
Configures standard inference parameters:
```dockerfile
PARAMETER temperature     0.7
PARAMETER top_p           0.9
PARAMETER top_k           40
PARAMETER repeat_penalty  1.1
PARAMETER context_window  32768
PARAMETER stop            "<|eot_id|>"
PARAMETER stop            "</s>"
```

### `PHANTOM_PARAM <key> <value>`
Configures PHANTOM CORE innovations:
```dockerfile
PHANTOM_PARAM sparsity_routing  on      # Enable/disable dynamic neuron skipping
PHANTOM_PARAM kv_compression    on      # Enable/disable Neural Cache KV compression
PHANTOM_PARAM spectral_quant    on      # Enable/disable Spectral Quantization
PHANTOM_PARAM safe_mode         off     # Lossless mode (disables lossy compression)
PHANTOM_PARAM tier_preference   vram    # Prefer vram | ram | nvme
PHANTOM_PARAM max_vram_mb       4096    # Hard VRAM cap
```

### `PLUGIN <name>`
Enables built-in or custom middleware plugins:
```dockerfile
PLUGIN rag-connector
PLUGIN tool-router
PLUGIN context-cache
```

### `MESSAGE <role> <content>`
Pre-loads conversation history:
```dockerfile
MESSAGE user      "Hello!"
MESSAGE assistant "Greetings! How can I assist you today?"
```
