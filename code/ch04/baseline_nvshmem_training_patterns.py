"""Baseline NVSHMEM training patterns benchmark wrapper."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark
from ch04.baseline_nvshmem_training_patterns_multigpu import NVSHMEMTrainingPatternsMultiGPU


def get_benchmark() -> BaseBenchmark:
    return NVSHMEMTrainingPatternsMultiGPU()

