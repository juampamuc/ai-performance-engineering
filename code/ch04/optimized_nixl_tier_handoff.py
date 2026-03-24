"""Optimized packed tier handoff analogue for the Chapter 4 NIXL story."""

from __future__ import annotations

from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark

from labs.nccl_nixl_nvshmem.comm_stack_common import TierHandoffBenchmark


def get_benchmark() -> BaseBenchmark:
    bench = TierHandoffBenchmark(optimized=True)
    return attach_benchmark_metadata(bench, __file__)
