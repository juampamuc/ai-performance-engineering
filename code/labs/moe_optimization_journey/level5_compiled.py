#!/usr/bin/env python3
"""Level 5: BMM fusion on the shared journey model.

ADDS: Vectorized scatter plus a single batched-matmul expert path.

Cumulative: batched + fused + mem_efficient + grouped + BMM fusion
"""

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level5Compiled(MoEJourneyBenchmark):
    """Level 5 shared BMM-fusion benchmark."""

    LEVEL = 5

def get_benchmark() -> Level5Compiled:
    return Level5Compiled()


if __name__ == "__main__":
    run_level(5)
