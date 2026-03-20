"""Baseline benchmark for the memory-bandwidth-patterns lab."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark
from labs.memory_bandwidth_patterns.bandwidth_patterns_common import (
    BaselineBandwidthPatternsBenchmark,
)


def get_benchmark() -> BaseBenchmark:
    return BaselineBandwidthPatternsBenchmark()
