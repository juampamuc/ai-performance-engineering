#!/usr/bin/env python3
"""Level 5: BMM fusion on the shared journey model.

ADDS: Vectorized scatter plus a single batched-matmul expert path.

Cumulative: batched + fused + mem_efficient + grouped + BMM fusion
"""

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level5Graphs(MoEJourneyBenchmark):
    """Level 5 shared BMM-fusion benchmark."""

    LEVEL = 5

def get_benchmark() -> Level5Graphs:
    return Level5Graphs()


