# Ch13 Benchmark Pair Validity Review Report

**Date:** 2025-03-17  
**Scope:** All baseline_*.py / optimized_*.py pairs in `/home/cfregly/ai-performance-engineering/code/ch13/`  
**Book Chapter 13:** Profiling, Tuning, and Scaling PyTorch

---

## Summary

| Status | Count |
|--------|-------|
| PASS   | 23    |
| FLAG   | 4     |

---

## Detailed Findings

### PASS (No Issues)

| Pair | Notes |
|------|-------|
| **arithmetic_intensity** | Same M/K/N=2048, FP32. Baseline: chunked matmul (low AI); Optimized: full fused matmul (high AI). Docstrings match. |
| **attention_standard** | Same batch=2, seq=8192, heads=8, head_dim=128. Baseline: manual matmul+softmax; Optimized: Flash SDP. |
| **attention_sliding_window** | Same SlidingWindowConfig (batch=2, seq=2048, heads=16, window=256). Baseline: dense masked; Optimized: FlexAttention. |
| **autograd_standard** | Same batch=16, hidden=1024. Baseline: eager autograd; Optimized: CUDA graph replay. Model/optimizer state restored after capture. |
| **bandwidth_naive** | Same size=4096², passes=16. Baseline: stride=8191 (uncoalesced); Optimized: stride=1 (coalesced). Same CUDA extension. |
| **dataloader_default** | Same dataset_size=1000, batch=64, preprocess_steps=8. Baseline: num_workers=0, pin_memory=False; Optimized: num_workers=4, pin_memory=True, prefetch_factor=4. |
| **fp4_perchannel** | Same batch=1024, hidden=8192. Baseline: naive per-tensor FP4; Optimized: TE NVFP4 per-channel. signature_equivalence_group used. |
| **fp8_perchannel** | Same batch=32, seq=512, in/out=4096. Baseline: per-tensor FP8; Optimized: per-channel FP8 via torch._scaled_mm. |
| **fp8_static** | Same batch=32, seq=512, dim=4096. Baseline: dynamic amax overhead (frozen scales for output parity); Optimized: static scales. Intentional design per docstring. |
| **fp8_pad_inner** | Same batch=4096, input_dim=8200, hidden=8192. Baseline: FP32; Optimized: torchao FP8 with pad_inner_dim. |
| **fp8_pad_inner_matmul** | Same m=8192, k=8200, n=8192. Baseline: FP32 matmul; Optimized: FP8 via torchao pad_inner_dim. |
| **fp8_te** | Same batch=256, hidden=4096. Baseline: TE FP16; Optimized: TE FP8 + CUDA graph. |
| **long_context_attention** | Same batch=1, seq=12288, heads=4. Baseline: explicit matmul+softmax+causal mask; Optimized: Flash SDP is_causal=True. |
| **precisionfp8** | Same batch=8192, hidden=8192. Baseline: FP32; Optimized: torchao FP8. |
| **precisionfp8_pad_inner** | Same dims. Baseline: FP32; Optimized: FP8 pad_inner. |
| **precisionmixed** | Same batch=512, hidden=2048, micro_steps=4. Baseline: FP32; Optimized: BF16 autocast. |
| **quantization** | Same batch=8192, in/hidden/out=4096. Baseline: FP32; Optimized: INT8 torch._int_mm + torch.compile. |
| **torchao_quantization** | Same batch=8192, in/hidden/out=4096. Baseline: FP32; Optimized: torchao INT8 + torch.compile. |
| **training_speed** | Subclass pattern. Same config. Baseline: eager BF16; Optimized: CUDA graph replay. |
| **training_standard** | Same hidden=1024, layers=24, seq=1024, batch=8. Baseline: no checkpoint; Optimized: gradient checkpoint every 8 layers. Memory-goal benchmark. |
| **warp_specialization_training** | Same rows=cols=4096, BF16. Baseline: eager _epilogue_chain; Optimized: torch.compile. |
| **context_parallel_multigpu** | Same ContextParallelConfig. Baseline: all_gather_attention; Optimized: ring_attention. Harness runs torchrun; benchmark_fn is single-GPU sim for verification. |
| **expert_parallel_multigpu** | Same ExpertParallelConfig. Baseline: impl="list"; Optimized: impl="single". Harness benchmark_fn is single-GPU sim; real workload in torchrun. |

---

### FLAG (Issues Found)

#### 1. **baseline_kv_cache_naive / optimized_kv_cache_naive** — Workload asymmetry

**Issue:** The baseline processes tokens **one at a time** (decode-style: `for pos in range(seq_len)` with `token = x[:, pos:pos+1, :]`), while the optimized processes in **blocks** (`for pos in range(0, seq_len, self.block_size)` with `token_block = x[:, pos:pos + self.block_size, :]`). The optimized does fewer layer forward passes per sequence.

- Baseline: `seq_len` iterations × `num_layers` layers (token-by-token).
- Optimized: `ceil(seq_len/block_size)` iterations × `num_layers` layers (block-by-block).

**Output capture:** Baseline uses `token` (last token); optimized uses `hidden[:, -1:, :]` (last token of last block). For block_size=128 and seq_len=256, baseline does 256 token steps; optimized does 2 block steps. The workloads are **not equivalent**.

**Recommendation:** Either (a) make optimized process token-by-token with the same loop structure but paged allocation, or (b) document that this pair compares "naive token-by-token" vs "block-prefill style" and accept different workload semantics.

---

#### 2. **baseline_matmul_pytorch / optimized_matmul_pytorch** — Precision asymmetry

**Issue:** Baseline uses **FP32** for matmul; optimized uses **FP16**. The optimized converts to FP32 for verification (`self.C.float()`), but the timed workload is FP16. The docstring says "CUTLASS GEMM optimization" and "torch.compile for kernel fusion" — the optimization is compile + precision, not just compile.

**Recommendation:** Either (a) make baseline FP16 for apples-to-apples (compile-only comparison), or (b) explicitly document that the pair compares FP32 eager vs FP16 compiled, and ensure tolerance (0.5, 5.0) is appropriate for FP32 vs FP16.

---

#### 3. **baseline_memory_profiling / optimized_memory_profiling** — Model + precision asymmetry

**Issue:** Baseline uses **FP32** `SimpleModel` (no checkpoint); optimized uses **BF16** `OptimizedModel` with **gradient checkpointing**. Two differences: (1) precision (FP32 vs BF16), (2) architecture (standard vs checkpointed). The optimization is memory (checkpoint + CUDA graph), but precision also changes.

**Recommendation:** This is a memory-goal benchmark; speed may regress. The pair is valid for memory comparison. Consider aligning precision (both FP32 or both BF16) so the only variable is checkpointing, if that matches the narrative.

---

#### 4. **baseline_regional_compile / optimized_regional_compile** — Docstring vs implementation mismatch

**Issue:** The baseline module docstring says "Full-block torch.compile baseline" and "Compiles the entire Transformer block per-sequence bucket." The code does **not** use `torch.compile` at all — it runs **FP32 eager** (`self.model(x)` with `dtype=torch.float32`). The optimized uses BF16 + regional compile (MLP only).

**Recommendation:** Update the baseline docstring to state it is "FP32 eager baseline" (no compile), or add full-block torch.compile to the baseline so it matches the docstring. The current comparison is eager FP32 vs regional-compile BF16, not full-block compile vs regional compile.

---

---

### Additional Notes

- **optimized_kv_cache_naive_pool**: Pairs with baseline_kv_cache_naive. Same token-by-token loop, same SimpleAttentionLayer. Only difference is KV cache impl (naive dict+torch.cat vs pooled prealloc). **PASS**.
- **SyntheticDataset**: Baseline uses `base = torch.randn(...); self.data = base`; optimized uses `self.data = torch.randn(...)`. Both use the same `randn` pattern; equivalent.
- **Sliding window setup**: `_compiled_flex(self.q, self.k, self.v)` in setup is warmup, not precomputation. Output comes from benchmark_fn.

---

## Checklist Summary

| Criterion | Pairs Checked |
|-----------|---------------|
| Workloads equivalent (sizes, dtypes, iterations) | 27 |
| Optimization matches docstring | 27 |
| No asymmetries (extra work in baseline, less in optimized) | 27 |
| setup() does not precompute timed work | 27 |

---

## Recommendations

1. **KV cache naive vs paged**: Align processing style (token-by-token vs block) or document the difference.
2. **Matmul PyTorch**: Clarify or align precision (FP32 vs FP16).
3. **Memory profiling**: Consider aligning precision for a cleaner memory-only comparison.
4. **Regional compile**: Fix baseline docstring or add full-block compile to match the narrative.
