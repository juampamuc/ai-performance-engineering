2026-03-20T04:09:00Z

Completed Work
- Added a self-contained lab under `labs/memory_bandwidth_patterns/`.
- Added a local CUDA extension with five measurable kernels:
  - `copy_scalar`
  - `copy_vectorized`
  - `copy_async_double_buffered`
  - `transpose_naive`
  - `transpose_tiled`
- Added harness-discoverable baseline and optimized modules for the transpose pair.
- Added a standalone comparison runner with JSON artifact output and clock locking.
- Added fail-fast gating for the async milestone on non-`sm80+` GPUs.
- Added README documentation tied to runnable commands instead of broad slide claims.

Validation
- Syntax/import validation:
  - `python -m py_compile labs/memory_bandwidth_patterns/*.py`
- Standalone runner smoke:
  - `python labs/memory_bandwidth_patterns/compare_bandwidth_patterns.py --rows 1024 --cols 1024 --warmup 2 --iterations 5 --json-out /tmp/memory_bandwidth_patterns_smoke.json`
  - Result highlights:
    - `copy_vectorized` = `1.614x` vs `copy_scalar`
    - `copy_async_double_buffered` = `1.666x` vs `copy_scalar`
    - `transpose_tiled` = `1.758x` vs `transpose_naive`
- Larger standalone measurement:
  - `python labs/memory_bandwidth_patterns/compare_bandwidth_patterns.py --rows 4096 --cols 8192 --warmup 5 --iterations 10 --json-out /tmp/memory_bandwidth_patterns_4096x8192.json`
  - Result highlights:
    - `copy_vectorized`: `4620.041 GB/s`
    - `copy_async_double_buffered`: `3858.073 GB/s`
    - `transpose_naive`: `846.547 GB/s`
    - `transpose_tiled`: `3539.049 GB/s`
    - Observation: the async copy milestone is real and verified, but it was slower than the vectorized copy on this larger shape. The README intentionally does not oversell it.
- Repo harness validation:
  - `python -m cli.aisp bench run --targets labs/memory_bandwidth_patterns:bandwidth_patterns --profile none --iterations 5 --warmup 2 --timeout-seconds 900 --single-gpu --target-extra-arg 'labs/memory_bandwidth_patterns:bandwidth_patterns=--rows 1024 --cols 1024'`
  - Result highlights:
    - baseline `0.03134 ms`
    - optimized `0.02682 ms`
    - speedup `1.169x`
    - verification passed
    - app clocks recorded by harness: `1965/3996 MHz`

Risks
- Harness validation reported a virtualized environment notice. The result is useful for development but should be treated as non-canonical for publication-grade claims.
- No shared test files were added because the ownership boundary for this task was the new lab directory only.
