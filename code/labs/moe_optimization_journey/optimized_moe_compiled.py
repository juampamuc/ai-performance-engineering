#!/usr/bin/env python3
"""Optimized MoE with torch.compile enabled (Level 7)."""

from labs.moe_optimization_journey.level7_compiled import Level7Compiled, run_level


def get_benchmark():
    return Level7Compiled()


