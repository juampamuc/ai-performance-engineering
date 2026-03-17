#!/usr/bin/env python3
"""Level 4: Grouped expert routing on the shared journey model.

ADDS: Sort tokens by expert and execute grouped expert work.

Cumulative: batched + fused + mem_efficient + grouped
"""

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level4Graphs(MoEJourneyBenchmark):
    """Level 4 shared grouped-routing benchmark."""

    LEVEL = 4

def get_benchmark() -> Level4Graphs:
    return Level4Graphs()


