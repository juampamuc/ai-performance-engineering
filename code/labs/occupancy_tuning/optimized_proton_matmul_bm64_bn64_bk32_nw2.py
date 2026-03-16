#!/usr/bin/env python3
"""Optimized: Triton matmul with a 64x64x32 low-warp schedule."""

from __future__ import annotations

from labs.occupancy_tuning.triton_matmul_schedules import (
    LATENCY_FRIENDLY_SCHEDULE,
    TritonMatmulProtonBenchmark,
)
from core.harness.benchmark_harness import BaseBenchmark


class OptimizedProtonMatmulLargeTile(TritonMatmulProtonBenchmark):
    """Optimized Triton matmul with a low-warp latency-friendly tile.

    Block config: 64x64x32, 2 warps
    Benefit: higher occupancy and lower per-launch overhead on smaller problems.
    """

    def __init__(self, size: int = 8192):
        super().__init__(
            schedule=LATENCY_FRIENDLY_SCHEDULE,
            size=size,
            iterations=10,
            warmup=5,
        )

def get_benchmark() -> BaseBenchmark:
    return OptimizedProtonMatmulLargeTile()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
