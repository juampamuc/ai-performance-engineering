#!/usr/bin/env python3
"""Optimized MoE: Level 7 (torch.compile on the graph-friendly path)."""

from labs.moe_optimization_journey.level7_compiled import Level7Compiled, run_level


def get_benchmark() -> Level7Compiled:
    return Level7Compiled()


__all__ = ["Level7Compiled", "get_benchmark"]
