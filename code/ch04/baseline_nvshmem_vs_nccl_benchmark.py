"""Baseline NVSHMEM vs NCCL benchmark via the strict multi-GPU wrapper."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch04.baseline_nvshmem_vs_nccl_benchmark_multigpu import NVSHMEMVsNCCLBenchmarkMultiGPU


def get_benchmark() -> BaseBenchmark:
    return NVSHMEMVsNCCLBenchmarkMultiGPU()

