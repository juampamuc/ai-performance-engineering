#!/usr/bin/env python3
"""Uniform routing plus naive mask-scan dispatch for the `moe_dispatch` target."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch15.moe_routing_benchmark_common import SharedExpertMoEBenchmarkBase


class BaselineMoEDispatchBenchmark(SharedExpertMoEBenchmarkBase):
    route_mode = "uniform"
    dispatch_mode = "mask_scan"
    nvtx_label = "baseline_moe_dispatch"


def get_benchmark() -> BaseBenchmark:
    return BaselineMoEDispatchBenchmark()
