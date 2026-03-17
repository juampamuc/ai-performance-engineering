#!/usr/bin/env python3
"""Optimized: Triton matmul with a wide-N 64x256x32 schedule."""

from __future__ import annotations

from labs.occupancy_tuning.triton_matmul_schedules import (
    TritonMatmulProtonBenchmark,
    WIDE_N_LATENCY_SCHEDULE,
)
from core.harness.benchmark_harness import BaseBenchmark


class OptimizedProtonMatmul(TritonMatmulProtonBenchmark):
    """Optimized Triton matmul with a wide-N schedule.

    Block config: 64x256x32, 8 warps
    Benefit: more work per launch along N without switching to the larger 128x tile family.
    """

    def __init__(self, size: int = 8192):
        super().__init__(
            schedule=WIDE_N_LATENCY_SCHEDULE,
            size=size,
            iterations=10,
            warmup=5,
        )

def get_benchmark() -> BaseBenchmark:
    return OptimizedProtonMatmul()


