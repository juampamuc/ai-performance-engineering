"""Optimized wrapper for the strict overlap-enabled DDP benchmark.

This public target now requires a real torchrun launch with >=2 GPUs. It no
longer publishes a single-GPU overlap simulation.
"""

from __future__ import annotations

from ch04.ddp_overlap import OptimizedOverlapDdpBenchmark


def get_benchmark() -> OptimizedOverlapDdpBenchmark:
    """Factory used by the harness."""
    return OptimizedOverlapDdpBenchmark()
