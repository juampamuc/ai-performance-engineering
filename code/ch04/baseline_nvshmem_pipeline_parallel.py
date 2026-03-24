"""Baseline NVSHMEM pipeline parallel benchmark via the strict multi-GPU wrapper."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch04.baseline_nvshmem_pipeline_parallel_multigpu import (
    NVSHMEMPipelineParallelMultiGPU,
)


def get_benchmark() -> BaseBenchmark:
    return NVSHMEMPipelineParallelMultiGPU()

