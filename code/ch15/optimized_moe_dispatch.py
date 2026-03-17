#!/usr/bin/env python3
"""Uniform routing plus active-expert dispatch for the `moe_dispatch` target."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch15.moe_routing_benchmark_common import SharedExpertMoEBenchmarkBase


class OptimizedMoEDispatchBenchmark(SharedExpertMoEBenchmarkBase):
    route_mode = "uniform"
    dispatch_mode = "active_experts"
    nvtx_label = "optimized_moe_dispatch"


def get_benchmark() -> BaseBenchmark:
    return OptimizedMoEDispatchBenchmark()
