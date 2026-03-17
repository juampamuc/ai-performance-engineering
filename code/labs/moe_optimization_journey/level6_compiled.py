#!/usr/bin/env python3
"""Level 6: CUDA graphs on the shared MoE journey model.

ADDS: expert-path CUDA graph capture/replay on top of the BMM-fused path.

This keeps the manual optimization journey cumulative:
- batched routing
- fused SiLU*up
- memory reuse
- grouped routing
- BMM-fused experts
- CUDA graph replay for the hot expert path
"""

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level6Compiled(MoEJourneyBenchmark):
    """Level 6: + CUDA graph replay on the MoE expert path."""

    LEVEL = 6

def get_benchmark() -> Level6Compiled:
    return Level6Compiled()


