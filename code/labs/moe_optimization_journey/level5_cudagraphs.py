#!/usr/bin/env python3
"""Level 6: CUDA Graphs.

ADDS: Capture kernel sequence for replay with minimal overhead.

CUDA Graphs eliminate:
- Kernel launch latency
- CPU-GPU synchronization overhead
- Python interpreter overhead

Note: Requires static shapes for graph capture.

Cumulative: batched + fused + mem_efficient + grouped + BMM fusion + cuda_graphs
"""
from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level6CUDAGraphs(MoEJourneyBenchmark):
    """Level 6: + CUDA Graphs."""

    LEVEL = 6

def get_benchmark() -> Level6CUDAGraphs:
    return Level6CUDAGraphs()


if __name__ == "__main__":
    run_level(6)

