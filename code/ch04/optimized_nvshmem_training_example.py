"""Optimized NVSHMEM training example benchmark wrapper."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark
from ch04.optimized_nvshmem_training_example_multigpu import OptimizedNVSHMEMTrainingExampleMultiGPU


def get_benchmark() -> BaseBenchmark:
    return OptimizedNVSHMEMTrainingExampleMultiGPU()

