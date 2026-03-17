#!/usr/bin/env python3
"""Uniform-routing baseline for the `moe_routing_topology_aware` target."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch15.moe_routing_benchmark_common import SharedExpertMoEBenchmarkBase


class BaselineMoERoutingTopologyAwareBenchmark(SharedExpertMoEBenchmarkBase):
    route_mode = "uniform"
    dispatch_mode = "mask_scan"
    nvtx_label = "baseline_moe_routing_topology_aware"


def get_benchmark() -> BaseBenchmark:
    return BaselineMoERoutingTopologyAwareBenchmark()
