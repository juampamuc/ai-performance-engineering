"""Optimized NVSHMEM pipeline parallel benchmark via the strict multi-GPU wrapper."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch04.optimized_nvshmem_pipeline_parallel_multigpu import (
    OptimizedNVSHMEMPipelineParallelMultiGPU,
)


def get_benchmark() -> BaseBenchmark:
    return OptimizedNVSHMEMPipelineParallelMultiGPU()

