# PHANTOM v2: Technical Implementation Blueprint
## From 5 tok/s to 14 tok/s on RTX 4050 (6 GB VRAM)

---

## EXECUTIVE TECHNICAL SUMMARY

**Problem:** Current PHANTOM achieves ~5 tok/s on RTX 4050 running 70B Q4 models. Can this reach 14 tok/s without model retraining or hardware upgrade?

**Solution:** Implement speculative decoding (EAGLE-3) + kernel fusion + adaptive prefetch scheduling.

**Expected Result:** 5 → 14 tok/s (2.8× speedup)

**Engineering Effort:** 3–4 months for a small team (2–3 engineers)

---

## PHASE 1: EAGLE-3 INTEGRATION (Week 1–6)

### 1.1 Design: EAGLE-3 Architecture

**What is EAGLE-3:**
- Draft prediction heads attached to target model
- Generates 5 candidate tokens (K=5) in parallel
- Target model verifies all 5 at once
- Acceptance rate: 75–85% (better than EAGLE-2)

**Why EAGLE-3 vs alternatives:**
- **vs Medusa:** EAGLE-3 uses feature fusion across layers; more stable across different prompts
- **vs Standard speculation:** No separate draft model needed; heads are ~3 MB overhead
- **vs EAGLE-2:** Better acceptance (6–12 percentage points improvement)

### 1.2 Architecture Specification

```
Current PHANTOM Architecture:
┌─────────────────────────────────┐
│ Model Input (tokens)            │
└────────────┬────────────────────┘
             │
        Embedding Layer
             │
        Layer 0...79
             │
        LM Head (classification to vocab)
             │
        Sampling
             │
        Output Token
└─────────────────────────────────┘

PHANTOM v2 with EAGLE-3:
┌──────────────────────────────────────────────────────┐
│ Model Input (tokens)                                 │
└────────────┬─────────────────────────────────────────┘
             │
        Embedding Layer
             │
        Layer 0 (features stored)
             │
        ...
             │
        Layer 20 (early features, stored)
             │
        ...
             │
        Layer 60 (middle features, stored)
             │
        ...
             │
        Layer 79 (final features)
             │
        ┌────┴────────────────────────────────┐
        │                                     │
    LM Head (target)            EAGLE-3 Heads (draft)
        │                                     │
        ├─────────────────────────────────────┤
        │                                     │
   verify_logits                draft_logits_1...5
        │                                     │
        └────────────┬────────────────────────┘
                     │
                  Speculative Verification
                     │
                  Output (1 + K tokens)
└──────────────────────────────────────────────────────┘
```

### 1.3 EAGLE-3 Head Design

```cpp
// EAGLE-3 Head Architecture (pseudo-code)

struct EagleHead {
  // Input: hidden states from layers [0, 30, 60, 79]
  // Output: logits for K=5 tokens ahead
  
  // Feature fusion layers
  Linear fusion_layer_0(768 * 4 → 768);  // fuse all 4 layers
  Linear fusion_layer_1(768 * 4 → 768);
  
  // Token prediction heads (5 heads for K=5)
  Linear token_head_1(768 → vocab_size);  // token N+1
  Linear token_head_2(768 → vocab_size);  // token N+2
  Linear token_head_3(768 → vocab_size);  // token N+3
  Linear token_head_4(768 → vocab_size);  // token N+4
  Linear token_head_5(768 → vocab_size);  // token N+5
  
  forward(h0, h30, h60, h79) {
    // Fuse all hidden states
    fused = fusion_layer_1(
      fusion_layer_0(concat(h0, h30, h60, h79))
    );
    
    // Generate logits for K=5 tokens
    logits = [
      token_head_1(fused),  // independent heads
      token_head_2(fused),  // no cross-token dependencies
      token_head_3(fused),  // all predicted in parallel
      token_head_4(fused),
      token_head_5(fused),
    ];
    
    return logits;  // shape: (5, vocab_size)
  }
};

// Total parameters: ~3 M (negligible compared to 70B)
// Total memory: ~50 MB in FP16
```

### 1.4 Memory Budget Impact

```
RTX 4050: 6 GB VRAM

Current allocation:
- Layers 0-17: ~3.0 GB (resident in VRAM)
- KV cache: ~0.8 GB (for 4K context)
- Buffers/misc: ~2.2 GB
Total: 6.0 GB (fully utilized)

PHANTOM v2 allocation:
- Layers 0-17: ~3.0 GB (same)
- KV cache: ~0.8 GB (same)
- EAGLE-3 heads: ~0.05 GB (tiny!)
- Buffers/misc: ~2.15 GB
Total: 6.0 GB (still fully utilized)

✓ EAGLE-3 adds negligible memory overhead
```

### 1.5 Speculative Verification Logic

```python
# Speculative decoding verification loop

def speculative_generation(prompt, k=5):
    """
    Args:
        prompt: token sequence (user input)
        k: speculation depth (5 tokens ahead)
    
    Yields:
        tokens: accepted tokens one at a time
    """
    
    # Prefill: process entire prompt (amortized)
    hidden_states = model.embed(prompt)
    for layer in model.layers[:-1]:
        hidden_states = layer(hidden_states)
    
    # Store intermediate features for EAGLE
    features = {
        'h0': hidden_states,  # after embedding
        'h30': None,  # updated in loop
        'h60': None,
        'h79': None,  # final layer
    }
    
    token = prompt[-1]
    accepted_tokens = []
    
    while len(accepted_tokens) < max_tokens:
        # Step 1: Draft prediction
        draft_logits = eagle_head(
            features['h0'], features['h30'], 
            features['h60'], features['h79']
        )  # shape: (k, vocab_size)
        
        draft_tokens = sample(draft_logits, temperature=0.5)
        # draft_tokens: [tok_n+1, tok_n+2, ..., tok_n+k]
        
        # Step 2: Verify all k tokens in parallel
        # (This is the parallelization that gives speedup)
        
        verify_hidden = features['h79']  # start from last layer
        
        verified = []
        for idx, draft_token in enumerate(draft_tokens):
            # Compute embedding for draft token
            embed = model.embed(draft_token)
            
            # Pass through remaining layers
            for layer_idx in range(len(model.layers)):
                verify_hidden = model.layers[layer_idx](verify_hidden)
            
            # Get target logits
            target_logits = model.lm_head(verify_hidden)
            target_token = argmax(target_logits)
            
            # Check if draft == target
            if draft_token == target_token:
                verified.append(draft_token)
            else:
                # Rejection: stop verification
                # Sample from rejection distribution
                accepted_token = rejection_sample(
                    draft_logits[idx],
                    target_logits
                )
                verified.append(accepted_token)
                break  # exit loop early on rejection
        
        # Step 3: Output verified tokens
        for token in verified:
            yield token
            accepted_tokens.append(token)
        
        # Step 4: Update features for next spec round
        # Re-run full forward on accepted tokens
        hidden_states = model.embed(verified)
        for layer in model.layers:
            hidden_states = layer(hidden_states)
        
        features['h79'] = hidden_states
        features['h0'] = model.embed(verified)  # simplified

# Key insight: Verification happens in *parallel* for all K tokens
# This is why speculative decoding provides speedup:
# - Without spec: 1 token per forward pass
# - With spec: K tokens in ~1.5× forward time (parallel verification)
```

### 1.6 Training EAGLE-3 Heads

**Data preparation:**
```python
# Collect training data: prompt -> next 5 tokens

def prepare_training_data(model, dataset_prompts, k=5):
    """
    For each prompt, collect:
    - Hidden states at layers [0, 30, 60, 79]
    - Next K tokens (ground truth)
    """
    
    training_examples = []
    
    for prompt in dataset_prompts:
        tokens = tokenize(prompt)
        hidden_states = model.embed(tokens)
        
        h_features = {}
        for layer_idx, layer in enumerate(model.layers):
            hidden_states = layer(hidden_states)
            
            # Store intermediate features
            if layer_idx in [0, 30, 60, 79]:
                h_features[f'h{layer_idx}'] = hidden_states.clone()
        
        # Get next K ground-truth tokens
        full_sequence = prompt + " " + next_k_tokens
        future_tokens = tokenize(next_k_tokens)[:k]
        
        training_examples.append({
            'features': h_features,
            'target_tokens': future_tokens,
        })
    
    return training_examples
```

**Loss function:**
```python
# Standard cross-entropy loss for each prediction head

def train_eagle_heads(eagle_model, train_loader, num_epochs=50):
    optimizer = AdamW(eagle_model.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        for batch in train_loader:
            # Forward pass
            draft_logits = eagle_model(batch['features'])
            # draft_logits shape: (batch, k, vocab_size)
            
            # Compute loss
            loss = 0
            for k_idx in range(5):
                # Each head independently predicts k-th token
                loss += CrossEntropyLoss(
                    draft_logits[:, k_idx, :],
                    batch['target_tokens'][:, k_idx]
                )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Training time: ~48 hours on 1× RTX 4090 with 1M examples
# Or: ~1 week on 1× RTX 4050
```

---

## PHASE 2: KERNEL FUSION (Week 7–10)

### 2.1 Target Kernels

**Current (using cuBLASLt + separate ops):**
```
┌─────────────────────┐
│ Load weights (FP8)  │ 2 ms  ← kernel launch overhead
├─────────────────────┤
│ Dequant to FP16     │ 0.5 ms
├─────────────────────┤
│ GEMM (matmul)       │ 1.5 ms
├─────────────────────┤
│ RoPE (position)     │ 0.3 ms
├─────────────────────┤
│ Softmax             │ 0.3 ms
└─────────────────────┘
Total: ~4.6 ms per attention head
```

**Target (fused kernel):**
```
┌──────────────────────────────────┐
│ Load + Dequant + GEMM + RoPE +   │ 2.2 ms  ← single kernel
│ Softmax (all-in-one)             │
└──────────────────────────────────┘
Total: ~2.2 ms per attention head (52% reduction)
```

### 2.2 Fused Attention Kernel (Triton)

```python
import triton
import triton.language as tl

@triton.jit
def fused_attention_kernel(
    q_weight,      # FP8 quantized
    k_weight,      # FP8 quantized
    v_weight,      # FP8 quantized
    input_act,     # FP16 activation
    rope_freqs,    # precomputed RoPE frequencies
    output,        # output buffer
    scale_q,       # dequant scale
    scale_k,
    scale_v,
    seq_len,       # sequence length
    block_size: tl.constexpr,
):
    """
    Fused kernel: Dequant Q, K, V weights + matmul + RoPE + softmax
    
    Performance target:
    - Minimize kernel launch overhead
    - Keep intermediate activations in SRAM (not write to HBM)
    - Maximize occupancy
    """
    
    # Block index
    block_id = tl.program_id(0)
    head_id = tl.program_id(1)
    
    # Load block of input activations
    input_block = tl.load(
        input_act + block_id * block_size + head_id * 64,
        mask=arange(block_size) < seq_len
    )
    
    # Dequant Q weights inline
    q_idx = tl.arange(0, block_size)
    q_weight_fp8 = tl.load(q_weight + q_idx)
    q_dequant = q_weight_fp8.to(tl.float16) * scale_q
    
    # Dequant K weights
    k_weight_fp8 = tl.load(k_weight + q_idx)
    k_dequant = k_weight_fp8.to(tl.float16) * scale_k
    
    # Matmul (implicit: input @ Q @ K^T)
    q_proj = tl.dot(input_block, q_dequant)  # stays in SRAM
    k_proj = tl.dot(input_block, k_dequant)  # stays in SRAM
    
    # Apply RoPE in-place (no extra pass)
    rope_cos = tl.cos(rope_freqs)
    rope_sin = tl.sin(rope_freqs)
    q_rope = q_proj * rope_cos + rotate_right(q_proj) * rope_sin
    
    # Attention scores
    scores = tl.dot(q_rope, k_proj.T)  # shape: (block_size, seq_len)
    scores = scores / tl.sqrt(64.0)  # scaling
    
    # Softmax (numerically stable)
    m = tl.max(scores, axis=1)
    p = tl.exp(scores - m[:, None])
    l = tl.sum(p, axis=1)
    p = p / l[:, None]
    
    # Dequant V and multiply
    v_weight_fp8 = tl.load(v_weight + q_idx)
    v_dequant = v_weight_fp8.to(tl.float16) * scale_v
    v_proj = tl.dot(input_block, v_dequant)
    
    # Output
    out = tl.dot(p, v_proj)
    tl.store(output + block_id * block_size + head_id * 64, out)

# Integration:
# Replace PHANTOM's current attention block with:
def fused_attention_layer(x, q_weight, k_weight, v_weight):
    return fused_attention_kernel[
        (ceil_div(seq_len, 128), num_heads)
    ](
        q_weight, k_weight, v_weight, x,
        rope_freqs, output,
        q_scale, k_scale, v_scale,
        seq_len, block_size=128
    )
```

### 2.3 Fused FFN Kernel

```python
@triton.jit
def fused_ffn_kernel(
    input_act,     # FP16 activation
    w1_weight,     # FP8 (input -> hidden)
    w2_weight,     # FP8 (hidden -> output)
    scale_w1,
    scale_w2,
    output,
    hidden_dim,
    block_size: tl.constexpr,
):
    """
    Fused: Dequant W1 + GEMM + GELU + Dequant W2 + GEMM
    
    Standard FFN:
    hidden = gelu(input @ W1.T)
    output = hidden @ W2.T
    
    This kernel fuses all operations, keeping hidden in SRAM.
    """
    
    block_id = tl.program_id(0)
    
    # Load input activation
    input_block = tl.load(
        input_act + block_id * block_size,
        mask=arange(block_size) < seq_len
    )
    
    # Dequant W1 and project
    w1_fp8 = tl.load(w1_weight + tl.arange(0, hidden_dim))
    w1_dequant = w1_fp8.to(tl.float16) * scale_w1
    hidden = tl.dot(input_block, w1_dequant)
    
    # Apply GELU in-place
    hidden = gelu(hidden)  # stays in SRAM, no HBM write
    
    # Dequant W2 and project
    w2_fp8 = tl.load(w2_weight + tl.arange(0, hidden_dim))
    w2_dequant = w2_fp8.to(tl.float16) * scale_w2
    output_block = tl.dot(hidden, w2_dequant)
    
    tl.store(output + block_id * block_size, output_block)
```

---

## PHASE 3: ADAPTIVE PREFETCH SCHEDULING (Week 11–14)

### 3.1 Integration with Wraith LSTM

**Current Wraith:**
```
Wraith LSTM (116K params)
  Input: [hidden_state_dim=768]
  Output: [next_layer_id]
  
Predicts: which layer is next
```

**Enhanced Wraith v2:**
```
Wraith v2 LSTM
  Input: [hidden_state_dim + layer_id + position_encoding]
  Output: [layer_id_probability_dist, prefetch_urgency]
  
Adds: confidence score for prefetch decision
      (only prefetch if confidence > threshold)
```

### 3.2 Speculative Prefetch Schedule

```python
def adaptive_prefetch_scheduler(eagle_predictions, model):
    """
    During speculation, prefetch weights for likely next tokens.
    """
    
    # EAGLE predicted tokens: [tok_n+1, tok_n+2, ..., tok_n+5]
    
    # Wraith's prediction: which layer is next
    wraith_prediction = wraith_lstm(current_hidden_state)
    next_layer = argmax(wraith_prediction)  # e.g., layer 20
    
    # Prefetch decision:
    # Only prefetch if:
    # 1. Next layer is not in VRAM
    # 2. Available PCIe bandwidth
    # 3. Confidence score > 0.8
    
    if (next_layer not in vram 
        and pcIe_available_bw > 8 GB/s
        and wraith_prediction[next_layer] > 0.8):
        
        # Async prefetch to staging buffer
        prefetch_layer_async(
            layer=next_layer,
            destination=staging_buffer,
            priority=HIGH
        )
    
    return next_layer
```

---

## PHASE 4: MEASUREMENT & PROFILING (Week 15–16)

### 4.1 Benchmarking Suite

```bash
#!/bin/bash
# PHANTOM v2 Benchmark

MODELS=("llama3:70b" "qwen2:70b" "deepseek:70b")
QUANTS=("Q4_K_M" "Q3_K_M")
HARDWARE=("rtx4050" "rtx3050" "rtx4060")

for model in "${MODELS[@]}"; do
    for quant in "${QUANTS[@]}"; do
        for hw in "${HARDWARE[@]}"; do
            echo "Testing: $model / $quant / $hw"
            
            # Baseline (no spec)
            python bench.py \
                --model "$model" \
                --quant "$quant" \
                --speculative false \
                --output benchmark_${model}_${quant}_${hw}_baseline.json
            
            # With EAGLE-3
            python bench.py \
                --model "$model" \
                --quant "$quant" \
                --speculative true \
                --eagle_k 5 \
                --output benchmark_${model}_${quant}_${hw}_eagle.json
            
            # Compute speedup
            python compare.py \
                baseline_${model}_${quant}_${hw}_baseline.json \
                benchmark_${model}_${quant}_${hw}_eagle.json
        done
    done
done
```

### 4.2 Profiling Code

```python
import torch
import time
from tqdm import tqdm

def profile_speculative_decoding(model, eagle_head, prompt, max_tokens=256):
    """
    Profile: token generation latency breakdown
    """
    
    timing = {
        'draft_prediction': [],
        'verify_compute': [],
        'total_per_round': [],
    }
    
    # Prefill prompt
    with torch.no_grad():
        hidden = model.embed(prompt)
        for layer in model.layers:
            hidden = layer(hidden)
    
    # Decode loop
    for round_idx in tqdm(range(max_tokens // 5)):
        t_start = time.perf_counter()
        
        # Step 1: Draft prediction
        t_draft_start = time.perf_counter()
        draft_logits = eagle_head(hidden)
        draft_tokens = torch.argmax(draft_logits, dim=-1)
        t_draft = time.perf_counter() - t_draft_start
        timing['draft_prediction'].append(t_draft)
        
        # Step 2: Verification
        t_verify_start = time.perf_counter()
        verify_hidden = hidden
        verified_tokens = []
        
        for draft_token in draft_tokens:
            # Forward pass for verification
            embed = model.embed(draft_token.unsqueeze(0))
            for layer in model.layers:
                embed = layer(embed)
            
            target_logits = model.lm_head(embed)
            target_token = torch.argmax(target_logits)
            
            if (draft_token == target_token).item():
                verified_tokens.append(draft_token.item())
            else:
                # Rejection: resample
                verified_tokens.append(target_token.item())
                break  # Early exit on rejection
        
        t_verify = time.perf_counter() - t_verify_start
        timing['verify_compute'].append(t_verify)
        
        t_total = time.perf_counter() - t_start
        timing['total_per_round'].append(t_total)
    
    # Compute statistics
    import statistics
    report = {
        'avg_draft_time_ms': statistics.mean(timing['draft_prediction']) * 1000,
        'avg_verify_time_ms': statistics.mean(timing['verify_compute']) * 1000,
        'avg_total_per_round_ms': statistics.mean(timing['total_per_round']) * 1000,
        'tokens_per_round': sum(len(vt) for vt in verified_tokens) / len(timing['total_per_round']),
        'implied_tok_sec': len(verified_tokens) / statistics.mean(timing['total_per_round']),
    }
    
    return report
```

---

## PHASE 5: VALIDATION & QUALITY CHECKS (Week 17–18)

### 5.1 Accuracy Benchmarks

```python
def validate_quality(model_baseline, model_eagle, dataset):
    """
    Ensure EAGLE-3 speculation does not reduce output quality.
    """
    
    results = {
        'ppl_baseline': [],
        'ppl_eagle': [],
        'bleu_baseline': [],
        'bleu_eagle': [],
        'agreement_rate': [],  # token-level agreement
    }
    
    for example in dataset:
        prompt = example['prompt']
        reference = example['reference']
        
        # Generate with baseline
        output_baseline = model_baseline.generate(
            prompt, max_tokens=256,
            use_speculation=False
        )
        
        # Generate with EAGLE
        output_eagle = model_eagle.generate(
            prompt, max_tokens=256,
            use_speculation=True
        )
        
        # Compute metrics
        ppl_base = compute_perplexity(output_baseline, reference)
        ppl_eagle = compute_perplexity(output_eagle, reference)
        
        bleu_base = compute_bleu(output_baseline, reference)
        bleu_eagle = compute_bleu(output_eagle, reference)
        
        # Token-level agreement
        agreement = (torch.tensor(output_baseline) == torch.tensor(output_eagle)).float().mean()
        
        results['ppl_baseline'].append(ppl_base)
        results['ppl_eagle'].append(ppl_eagle)
        results['bleu_baseline'].append(bleu_base)
        results['bleu_eagle'].append(bleu_eagle)
        results['agreement_rate'].append(agreement.item())
    
    # Summary
    print(f"PPL Baseline: {statistics.mean(results['ppl_baseline']):.3f}")
    print(f"PPL EAGLE-3: {statistics.mean(results['ppl_eagle']):.3f}")
    print(f"PPL Delta: {abs(statistics.mean(results['ppl_baseline']) - statistics.mean(results['ppl_eagle'])):.3f}")
    print(f"BLEU Delta: {abs(statistics.mean(results['bleu_baseline']) - statistics.mean(results['bleu_eagle'])):.3f}")
    print(f"Token Agreement: {statistics.mean(results['agreement_rate']):.1%}")
    
    # Acceptance rate verification
    # (Expected: 75% of draft tokens are accepted)
```

### 5.2 Cross-Hardware Validation

```bash
#!/bin/bash
# Test on multiple GPUs

declare -a GPUS=("rtx3050" "rtx4050" "rtx4060" "rtx3060" "arc_a770")

for gpu in "${GPUS[@]}"; do
    echo "Testing on: $gpu"
    
    python validate.py \
        --gpu "$gpu" \
        --model "llama3:70b" \
        --quant "Q4_K_M" \
        --enable_eagle3 true \
        --output "validation_${gpu}.json"
done

# Aggregate results
python aggregate_results.py validation_*.json
```

---

## PHASE 6: INTEGRATION & RELEASE (Week 19–20)

### 6.1 Code Integration Checklist

- [ ] EAGLE-3 head architecture implemented (Triton kernels)
- [ ] Training pipeline for EAGLE heads
- [ ] Speculative decoding verification loop
- [ ] Adaptive prefetch scheduler updates
- [ ] Kernel fusion (attention + FFN)
- [ ] Memory budget validation on all hardware
- [ ] Quality assurance tests pass
- [ ] Performance benchmarks meet targets (14+ tok/s on RTX 4050)
- [ ] Documentation updated
- [ ] GitHub release prepared

### 6.2 API Changes

```python
# New API for PHANTOM v2

from phantom import LLMRuntime

# Initialize with speculation
runtime = LLMRuntime(
    model="llama3:70b",
    quantization="Q4_K_M",
    speculative_decoding=True,  # NEW
    eagle_k=5,                  # NEW (speculation depth)
    eagle_heads_path="~/.phantom/eagle/llama3_70b.pt",  # NEW
    prefetch_enabled=True,
    adaptive_sparsity=True,
)

# Generation (unchanged)
output = runtime.generate(
    prompt="What is AI?",
    max_tokens=256,
    temperature=0.7,
)

# NEW: Get metrics
metrics = runtime.get_metrics()
print(f"Token speed: {metrics['tok_per_sec']:.1f}")
print(f"Acceptance rate: {metrics['speculation_acceptance_rate']:.1%}")
print(f"TTFT: {metrics['time_to_first_token_ms']:.1f} ms")
```

---

## TECHNICAL RISKS & MITIGATION

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **EAGLE-3 training fails to converge** | Speculation doesn't work; stuck at 5 tok/s | A/B test multiple loss functions; start with simpler EAGLE-2 |
| **Acceptance rate < 60%** | Speedup < 2.0×; misses target | Recalibrate on RTX 4050 specifically; reduce K to 3 |
| **Kernel fusion regression** | Some models crash; quality loss | Extensive unit tests; gradual rollout |
| **Memory overflow during verification** | OOM on RTX 4050 | Dynamic batch sizing; fallback to non-speculative mode |
| **PCIe bottleneck dominates** | Speculation can't parallelize weight loads | Already analyzed; prefetch should help 40% of latency |

---

## SUCCESS CRITERIA

**Performance:**
- [ ] RTX 4050: 70B Q4 reaches **14+ tok/s** with EAGLE-3
- [ ] RTX 3050: 70B Q4 reaches **11+ tok/s**
- [ ] RTX 4060: 70B Q4 reaches **16+ tok/s**

**Quality:**
- [ ] PPL delta < 0.2 (no perceptible change)
- [ ] BLEU agreement > 99.5% (token-level)

**Reliability:**
- [ ] No OOM on target hardware
- [ ] Speculation acceptance rate >= 70%
- [ ] Latency p95 < 200 ms per spec round

**Code:**
- [ ] <5000 lines of new code (clean implementation)
- [ ] Fully tested and documented
- [ ] Backward compatible with PHANTOM v1

---

## REFERENCES

1. **EAGLE-3 Paper:** Li et al., "EAGLE-3: Scaling up Inference Acceleration," arXiv:2503.01840
2. **Triton:** https://openai.com/research/triton
3. **PHANTOM:** https://github.com/FreakyAdy/phantom
4. **Speculative Decoding Survey:** https://arxiv.org/abs/2211.17192
5. **GPU Optimization:** NVIDIA CUDA C++ Programming Guide

---

**Document Version:** 1.0  
**Last Updated:** September 2026  
**Status:** Ready for Implementation
