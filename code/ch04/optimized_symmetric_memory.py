"""Optimized symmetric memory benchmark via the strict multi-GPU wrapper."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch04.optimized_symmetric_memory_multigpu import OptimizedSymmetricMemoryMultiGPU


def get_benchmark() -> BaseBenchmark:
    return OptimizedSymmetricMemoryMultiGPU()

