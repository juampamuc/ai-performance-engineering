"""Optimized bandwidth benchmark suite via the strict multi-GPU wrapper."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch04.optimized_bandwidth_benchmark_suite_multigpu import OptimizedBandwidthSuiteMultiGPU


def get_benchmark() -> BaseBenchmark:
    return OptimizedBandwidthSuiteMultiGPU()

