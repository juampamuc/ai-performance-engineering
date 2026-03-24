"""Optimized NVSHMEM training patterns benchmark wrapper."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark
from ch04.optimized_nvshmem_training_patterns_multigpu import OptimizedNVSHMEMTrainingPatternsMultiGPU


def get_benchmark() -> BaseBenchmark:
    return OptimizedNVSHMEMTrainingPatternsMultiGPU()

