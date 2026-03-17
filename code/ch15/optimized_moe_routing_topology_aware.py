#!/usr/bin/env python3
"""Topology-aware routing with the same dispatch path for the routing target."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch15.moe_routing_benchmark_common import SharedExpertMoEBenchmarkBase


class OptimizedMoERoutingTopologyAwareBenchmark(SharedExpertMoEBenchmarkBase):
    route_mode = "topology_aware"
    dispatch_mode = "mask_scan"
    nvtx_label = "optimized_moe_routing_topology_aware"


def get_benchmark() -> BaseBenchmark:
    return OptimizedMoERoutingTopologyAwareBenchmark()
