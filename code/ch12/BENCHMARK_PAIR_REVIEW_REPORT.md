# Ch12 Benchmark Pair Validity Review

**Chapter 12**: Dynamic Scheduling, CUDA Graphs, and Device-Initiated Kernel Orchestration

**Review date**: 2025-03-17

---

## Summary

| Pair | Status | Notes |
|------|--------|------|
| work_queue | PASS | |
| cuda_graphs | PASS | |
| graph_bandwidth | PASS | |
| kernel_launches | **FLAG** | Graph replay compounds work across harness iterations |
| kernel_fusion | PASS | |
| uneven_static | PASS | |
| uneven_partition | PASS | |
| dynamic_parallelism_host | PASS | |
| dynamic_parallelism_device | PASS | |
| cuda_graphs_conditional | PASS | |
| cuda_graphs_conditional_enhanced | PASS | |
| graph_conditional_runtime | PASS | |
| nvfp4_mlp | PASS | |

---

## Detailed Findings

### 1. baseline_work_queue.py / optimized_work_queue.py — **PASS**

- **Workload**: N=1M elements, iterations=5, dtype=float32. Identical.
- **Optimization**: Static work distribution (baseline) vs dynamic atomic work queue (optimized). Matches docstrings.
- **Setup**: Both run a single warmup iteration, then reset data with same seed. No precomputation of timed work.
- **Minor**: Optimized has stray comment "Enable cuDNN benchmarking" (copy-paste); work_queue does not use cuDNN. Cosmetic only.

---

### 2. baseline_cuda_graphs.py / optimized_cuda_graphs.py — **PASS**

- **Workload**: N=1024, iterations=32000. Identical.
- **Optimization**: Separate kernel launches (baseline) vs CUDA graph replay (optimized). Matches docstrings.
- **Setup**: Baseline warms with 1 iteration; optimized warms with full graph_replay. Both reset data before timed runs. Graph capture happens in setup, not in timed path.
- **Config**: Optimized adds nsys_timeout_seconds, nsys_preset_override for profiling. No workload asymmetry.

---

### 3. baseline_graph_bandwidth.py / optimized_graph_bandwidth.py — **PASS**

- **Workload**: N=8192, iterations=64_000. Identical.
- **Optimization**: Separate copy kernel launches vs graph-captured copy. Matches docstrings.
- **Setup**: Both dry-run once; optimized runs graph_kernel (capture + run). Data not reassigned after capture. No precomputation of timed output.
- **Minor**: Optimized has try/except around extension load for clearer error reporting. No workload impact.

---

### 4. baseline_kernel_launches.py / optimized_kernel_launches.py — **FLAG**

**Issue**: Optimized CUDA graph replay does not reset input between harness iterations, causing work compounding.

- **Baseline**: Each `benchmark_fn()` call runs `many_small_ops_regular(self.x.clone(), 1000)` — same input each time, same output.
- **Optimized**: Graph replays in-place on `x_capture`. Each harness iteration runs `graph.replay()`, which modifies `x_capture`. The next replay uses the previous result as input, so work compounds (e.g., 50 harness iterations → effectively 50×1000 iterations).
- **Impact**: Output diverges from baseline; verification would fail. Workload is not equivalent.
- **Fix**: Reset `x_capture` to initial state before each replay, e.g.:
  ```python
  def replay():
      self.x_capture.copy_(self.x_template)
      self.graph.replay()
      return self.x_capture
  ```

**Additional asymmetry**: Baseline `iterations=30`, optimized `iterations=50`. Harness iteration counts differ; acceptable if workload-per-call is fixed first.

---

### 5. baseline_kernel_fusion.py / optimized_kernel_fusion.py — **PASS**

- **Workload**: N=16M, iterations=10. Identical.
- **Optimization**: Separate kernels (add, multiply, sqrt) vs fused kernel. Matches docstrings. Same math in both (add+1, *2, sqrt).
- **Setup**: Both dry-run once, then reset data. No precomputation of timed output.
- **Note**: `optimized_kernel_fusion_llm_*.py` variants are alternative optimized implementations (persistent buffer, stream-friendly setup, etc.). Canonical pair is baseline_kernel_fusion / optimized_kernel_fusion.

---

### 6. baseline_uneven_static.py / optimized_uneven_static.py — **PASS**

- **Workload**: workload_params identical (workload=1, dtype=float32, batch_size=1). Binaries use hardcoded elems, warmup, iters.
- **Optimization**: Static partition (baseline) vs device work stealing (optimized). Wrappers include baseline_uneven_partition.cu / optimized_uneven_partition.cu.
- **Setup**: CudaBinaryBenchmark; no Python setup precomputation.

---

### 7. baseline_uneven_partition.py / optimized_uneven_partition.py — **PASS**

- **Workload**: warmup=1, iters=10, dtype=float32, batch_size=1. Identical.
- **Optimization**: Host-driven static partitions vs device work stealing (atomic next_segment). Same elems, same math (v*v + 0.5*v).
- **Setup**: CudaBinaryBenchmark; binaries handle setup internally.

---

### 8. baseline_dynamic_parallelism_host.py / optimized_dynamic_parallelism_host.py — **PASS**

- **Workload**: workload=1, dtype=float32, batch_size=1. Identical.
- **Optimization**: Host vs device-initiated launch pattern. Wraps CUDA binaries.
- **Setup**: CudaBinaryBenchmark.

---

### 9. baseline_dynamic_parallelism_device.py / optimized_dynamic_parallelism_device.py — **PASS**

- **Workload**: batch_size=262144, elements=262144, dtype=float32. Same N and math.
- **Optimization**: Baseline launches many small child grids (seg=32); optimized launches one large child grid. Same total work (N elements, same fuse_op math). segment_size in workload_params (32 vs 256) reflects launch granularity; binaries hardcode behavior.
- **Setup**: CudaBinaryBenchmark.

---

### 10. baseline_cuda_graphs_conditional.py / optimized_cuda_graphs_conditional.py — **PASS**

- **Workload**: N=1, KERNEL_ITERS=1024, ITERS=5000, dtype=float32, batch_size=1. Identical.
- **Optimization**: Conditional graph nodes (IF/WHILE) in CUDA binaries.
- **Setup**: CudaBinaryBenchmark.

---

### 11. baseline_cuda_graphs_conditional_enhanced.py / optimized_cuda_graphs_conditional_enhanced.py — **PASS**

- **Workload**: N=1, ITERS=5000, dtype=float32, batch_size=1. Identical. (No KERNEL_ITERS in enhanced pair.)
- **Optimization**: Enhanced conditional graph implementation.
- **Setup**: CudaBinaryBenchmark.

---

### 12. baseline_graph_conditional_runtime.py / optimized_graph_conditional_runtime.py — **PASS**

- **Workload**: batch_size=16, seq_len=64, hidden_dim=256, num_loops=256. Identical.
- **Optimization**: Fresh kernel launches (baseline) vs CUDA graph replay (optimized). Same _compute_ops (mul, add, relu).
- **Setup**: Baseline runs _compute_ops once then restores initial_state to match optimized capture semantics. Optimized captures graph in setup. No precomputation of timed output.
- **Verification**: Both use same _verify_input and output tolerance.

---

### 13. baseline_nvfp4_mlp.py / optimized_nvfp4_mlp.py — **PASS**

- **Workload**: batch_size=512, d_model=8192, d_ff=32768, num_layers=2, iterations=20, warmup=10. Identical config.
- **Optimization**: BF16 MLP (baseline, use_nvfp4=False) vs NVFP4 MLP (optimized, use_nvfp4=True). Same NVFP4MLPBenchmark, same workload.
- **Setup**: Delegated to core NVFP4MLPBenchmark.

---

## kernel_fusion LLM Variants (Not Canonical Pairs)

The following are alternative optimized implementations, not separate baseline/optimized pairs:

- `optimized_kernel_fusion_llm_reuse_static_tensor_and_simplify_setup.py`
- `optimized_kernel_fusion_llm_persistent_buffer_and_stream_friendly_setup.py`
- `optimized_kernel_fusion_llm_dedicated_stream_and_prefetch_for_blackwell.py`

These compare against the same `baseline_kernel_fusion.py` and use the same fused_kernel extension. Workload (N=16M, iterations=10) is equivalent. No issues found.

---

## Recommendations

1. **kernel_launches (FLAG)**: Fix optimized to reset `x_capture` before each graph replay so each harness iteration performs the same work as the baseline.
2. **work_queue**: Remove or correct the misleading "Enable cuDNN benchmarking" comment in optimized_work_queue.py setup.
