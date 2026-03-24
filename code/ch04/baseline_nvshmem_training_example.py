"""Baseline NVSHMEM training example benchmark wrapper."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark
from ch04.baseline_nvshmem_training_example_multigpu import NVSHMEMTrainingExampleMultiGPU


def get_benchmark() -> BaseBenchmark:
    return NVSHMEMTrainingExampleMultiGPU()

