#!/usr/bin/env python3
"""Optimized MoE: CUDA graphs.

Pairs with: baseline_moe.py

This wrapper must stay workload-equivalent with the baseline benchmark. Use the
MoEJourneyBenchmark implementation (Level 6) to keep parameter_count, inputs,
and verification semantics consistent across levels.
"""
from labs.moe_optimization_journey.level5_cudagraphs import Level6CUDAGraphs


def get_benchmark() -> Level6CUDAGraphs:
    return Level6CUDAGraphs()


__all__ = ["Level6CUDAGraphs", "get_benchmark"]
