"""MoE Optimization Journey: From Naive to Production-Speed.

This lab takes you on a journey from a deliberately slow MoE implementation
to production-quality performance by applying optimization techniques from
the AI Performance Engineering book.

Each level builds on the previous shared model, so the story stays cumulative.

Levels:
    0. Naive - Sequential experts, Python loops (baseline)
    1. Batched - Parallel expert execution
    2. Fused - SiLU*up fusion
    3. MemEfficient - Buffer reuse on the shared model
    4. Grouped - Sort by expert and run grouped expert work
    5. BMM Fusion - Vectorized scatter plus a single BMM expert path
    6. CUDA Graphs - Capture and replay the fused Level 5 path
    7. Compiled - torch.compile on top of the graph-friendly model

Usage with bench CLI:
    # Run all levels
    python -m cli.aisp bench run --targets labs/moe_optimization_journey
    
    # Run specific level
    python -m cli.aisp bench run --targets labs/moe_optimization_journey/level0_naive
    
    # Compare levels
    python -m cli.aisp bench compare labs/moe_optimization_journey/level0_naive labs/moe_optimization_journey/level7_compiled
"""

from labs.moe_optimization_journey.moe_config import MoEConfig, get_config, CONFIGS

__all__ = [
    "MoEConfig",
    "get_config", 
    "CONFIGS",
]
