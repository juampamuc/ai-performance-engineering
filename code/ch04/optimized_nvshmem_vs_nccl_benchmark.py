"""Optimized NVSHMEM vs NCCL benchmark via the strict multi-GPU wrapper."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch04.optimized_nvshmem_vs_nccl_benchmark_multigpu import (
    OptimizedNVSHMEMVsNCCLBenchmarkMultiGPU,
)


def get_benchmark() -> BaseBenchmark:
    return OptimizedNVSHMEMVsNCCLBenchmarkMultiGPU()

