# Chapter 11 Benchmark Pair Validity Review

**Review date:** 2025-03-17  
**Scope:** All baseline_*/optimized_* pairs in `/home/cfregly/ai-performance-engineering/code/ch11/`  
**Chapter 11 topic:** Inter-Kernel Pipelining, Synchronization, and CUDA Stream-Ordered Memory Allocations

---

## Summary

| Pair | Result | Notes |
|------|--------|-------|
| baseline_streams / optimized_streams | PASS | |
| baseline_stream_ordered / optimized_stream_ordered | PASS | |
| baseline_tensor_cores_streams / optimized_tensor_cores_streams | PASS | |
| baseline_adaptive_streams / optimized_adaptive_streams | FLAG | Docstring mismatch |
| baseline_gemm_streams / optimized_gemm_streams | FLAG | Workload is not GEMM |
| baseline_stream_ordered_kv_cache / optimized_stream_ordered_kv_cache | PASS | |
| baseline_distributed_streams / optimized_distributed_streams | PASS | |
| baseline_warp_specialized_multistream / optimized_warp_specialized_multistream | PASS | |
| baseline_warp_specialization_multistream / optimized_warp_specialization_multistream | PASS | |
| baseline_warp_specialized_two_pipelines_multistream / optimized_warp_specialized_two_pipelines_multistream | FLAG | Extension load asymmetry |
| baseline_warp_specialized_two_pipelines_driver / optimized_warp_specialized_two_pipelines_driver | PASS | |

---

## Detailed Findings

### 1. baseline_streams.py / optimized_streams.py — **PASS**

- **Workload equivalence:** Both use N=5_000_000, num_chunks=20, same _compute() (3 passes of sin/cos/tanh/sigmoid). Identical.
- **Optimization vs code:** Docstring describes pipelined H2D overlapping with compute. Code implements double-buffered pipelining with stream_h2d and stream_compute; compute waits for transfer via wait_stream. Matches.
- **Asymmetries:** None. Baseline uses blocking copy_ (sequential); optimized uses non_blocking=True with streams.
- **Setup precomputation:** Both allocate host/device buffers and create data in setup(); no precomputation of benchmark_fn results.
- **Verification:** Both capture same indices (0, mid, last) for verification payload.

---

### 2. baseline_stream_ordered.py / optimized_stream_ordered.py — **PASS**

- **Workload equivalence:** Both use elements=4096, inner_iterations=500, profile_inner_iterations=8, num_streams=8. Identical.
- **Optimization vs code:** Baseline uses cudaMalloc/cudaFree (run_standard_allocator_capture); optimized uses cudaMallocAsync/cudaFreeAsync (run_stream_ordered_allocator_capture). Matches docstrings.
- **Asymmetries:** None. Same iteration counts, same elements, same NUM_STREAMS in extension.
- **Setup precomputation:** Both warm the extension with a small run (1024, 1) in setup(); this is JIT/warmup, not result precomputation. benchmark_fn() runs the full workload.
- **Verification:** Both capture output from the extension; tolerance (0,0) for exact match.

---

### 3. baseline_tensor_cores_streams.py / optimized_tensor_cores_streams.py — **PASS**

- **Workload equivalence:** Both use num_segments=24, matrix_dim=768, same dtype (bf16 or fp16). Same total work: 24 GEMMs on 768×768 matrices.
- **Optimization vs code:** Baseline synchronizes after each segment (single stream); optimized uses 6 streams and overlaps H2D/GEMM/D2H without per-segment sync. Matches.
- **Asymmetries:** None. Optimized uses 6 device slots (one per stream) vs 1 in baseline; same total compute.
- **Setup precomputation:** Both allocate host tensors and device slots in setup(); no result precomputation.
- **Verification:** Both capture host_output (24×768) with same tolerance (1e-2, 1e-2).

---

### 4. baseline_adaptive_streams.py / optimized_adaptive_streams.py — **FLAG**

- **Workload equivalence:** Both use StridedStreamBaseline/ConcurrentStreamOptimized defaults: num_elements=24_000_000, num_segments=16. Baseline uses 1 stream; optimized uses 2 streams (default). Equivalent workload.
- **Optimization vs code:** Docstring says "adaptive scheduling" for optimized, but code uses fixed round-robin stream assignment (`streams[idx % num_streams]`). No adaptive logic. **Docstring mismatch.**
- **Asymmetries:** None in workload.
- **Setup precomputation:** None.
- **Recommendation:** Update docstring to "concurrent multi-stream execution" or "round-robin stream assignment" to match implementation, or add adaptive scheduling logic.

---

### 5. baseline_gemm_streams.py / optimized_gemm_streams.py — **FLAG**

- **Workload equivalence:** Both inherit from StridedStreamBaseline/ConcurrentStreamOptimized with defaults. Same N, segments, compute.
- **Optimization vs code:** Names suggest GEMM, but `stream_overlap_base.py` implements copy + element-wise ops: `mul_(2.0)`, `add_(1.0)`, `mul_(1.1)`, `add_(0.5)`. **No GEMM.** The workload is H2D copy + element-wise ops + D2H copy.
- **Asymmetries:** None in workload.
- **Setup precomputation:** None.
- **Recommendation:** Rename to reflect actual workload (e.g., baseline_copy_compute_streams) or change workload to use actual GEMM (e.g., torch.matmul) to match the name.

---

### 6. baseline_stream_ordered_kv_cache.py / optimized_stream_ordered_kv_cache.py — **PASS**

- **Workload equivalence:** Both use num_elements=18_000_000, num_segments=8. Baseline uses 1 stream; optimized uses 2 streams. Same total work.
- **Optimization vs code:** Baseline is sequential; optimized overlaps across streams. Matches docstrings.
- **Asymmetries:** None.
- **Setup precomputation:** None.
- **Note:** "KV cache" in the name is conceptual; the base workload is copy+element-wise ops, not literal KV cache updates. Acceptable for Chapter 11 stream overlap focus.

---

### 7. baseline_distributed_streams.py / optimized_distributed_streams.py — **PASS**

- **Workload equivalence:** Both use defaults: 24M elements, 16 segments. Baseline: 1 stream; optimized: 2 streams. Equivalent workload.
- **Optimization vs code:** Baseline is single-stream; optimized uses multi-stream overlap. "Distributed" refers to the pattern, not multi-GPU. Matches.
- **Asymmetries:** None.
- **Setup precomputation:** None.

---

### 8. baseline_warp_specialized_multistream.py / optimized_warp_specialized_multistream.py — **PASS**

- **Workload equivalence:** Both CudaBinaryBenchmark wrappers use identical workload_params: TILE=32, THREADS=96, batches=4096, dtype=float32.
- **Optimization vs code:** Optimization is in the CUDA binaries; Python wrappers only configure the same workload. Fair comparison.
- **Asymmetries:** None.
- **Setup precomputation:** N/A (CudaBinaryBenchmark runs subprocess).
- **Verification:** Uses VERIFY_CHECKSUM from CUDA binary stdout when built with -DVERIFY=1.

---

### 9. baseline_warp_specialization_multistream.py / optimized_warp_specialization_multistream.py — **PASS**

- **Workload equivalence:** Both use num_elements=96_000_000, num_segments=64. Baseline: 1 stream; optimized: 2 streams. Equivalent workload.
- **Optimization vs code:** Baseline is sequential; optimized overlaps across 2 streams. Matches.
- **Asymmetries:** None.
- **Setup precomputation:** None.

---

### 10. baseline_warp_specialized_two_pipelines_multistream.py / optimized_warp_specialized_two_pipelines_multistream.py — **FLAG**

- **Workload equivalence:** Both use tiles=8192, tile_elems=1024. Baseline uses num_streams=1; optimized uses num_streams=4. Same total elements (8,388,608). Intentional difference in stream count for overlap demonstration.
- **Optimization vs code:** Baseline: single-stream; optimized: 4-stream overlap with CUDA::pipeline. Matches.
- **Asymmetries:**
  - **Extension loading:** Baseline loads extension in `setup()`; optimized loads in `__init__` (`self.ext = _load_optimized_extension()`). Optimized may fail at discovery if extension fails; baseline fails at setup. Optimized also calls `ensure_dsmem_supported` in __init__ vs baseline in setup.
- **Setup precomputation:** None. benchmark_fn() is not affected.
- **Recommendation:** Move optimized's extension load to setup() for consistency and to avoid discovery-time failures.

---

### 11. baseline_warp_specialized_two_pipelines_driver.py / optimized_warp_specialized_two_pipelines_driver.py — **PASS**

- **Workload equivalence:** Both CudaBinaryBenchmark wrappers use identical workload_params: tiles=128, tile_elems=1024, num_streams=2, dtype=float32.
- **Optimization vs code:** Optimization is in the CUDA binaries; Python wrappers configure the same workload.
- **Asymmetries:** None.
- **Setup precomputation:** N/A.

---

## Recommendations Summary

1. **baseline_adaptive_streams / optimized_adaptive_streams:** Align docstring with implementation (concurrent streams, not adaptive scheduling).
2. **baseline_gemm_streams / optimized_gemm_streams:** Either rename to reflect copy+element-wise workload or change workload to actual GEMM.
3. **baseline_warp_specialized_two_pipelines_multistream / optimized_warp_specialized_two_pipelines_multistream:** Load extension in setup() for optimized to match baseline and avoid discovery-time failures.
