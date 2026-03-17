"""Optimized: Adaptive-bitwidth quantized KV cache refresh.

Chapter 19: Blackwell-Native Precision Operations

The optimized version keeps the same refresh cadence and tensor footprint as
the baseline, but replaces the FP32 cache refresh with quantized refresh:
- INT8 for early tokens (highest precision)
- INT6 for middle tokens
- INT4 for late tokens (minimal memory)

This reduces memory traffic while preserving the full-cache update pattern.
"""

from __future__ import annotations

from ch19.baseline_dynamic_quantized_cache import (  # noqa: E402
    _DynamicQuantizedCacheBenchmark,
)


class OptimizedDynamicQuantizedCacheBenchmark(_DynamicQuantizedCacheBenchmark):
    """Optimized: adaptive-bitwidth quantized refresh over the same KV cache.

    The benchmark intentionally keeps the same number of steps and cache shape as
    the baseline FP32 refresh. The optimization is algorithmic: refresh
    pre-quantized cache pages with fewer bytes per step while keeping the same
    logical cache footprint and cadence.
    """

    def __init__(self) -> None:
        schedule = [8] * 12 + [6] * 8 + [4] * 12
        # use_fp32_baseline=False means we use quantization (default)
        super().__init__(schedule_bits=schedule, use_fp32_baseline=False)


def get_benchmark():
    return OptimizedDynamicQuantizedCacheBenchmark()
