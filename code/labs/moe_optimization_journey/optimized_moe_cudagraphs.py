#!/usr/bin/env python3
"""Optimized MoE: Level 6 (CUDA Graphs)."""

from labs.moe_optimization_journey.level5_cudagraphs import Level6CUDAGraphs


def get_benchmark() -> Level6CUDAGraphs:
    return Level6CUDAGraphs()


__all__ = ["Level6CUDAGraphs", "get_benchmark"]

