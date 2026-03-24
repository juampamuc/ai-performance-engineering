"""Baseline wrapper for the strict no-overlap DDP benchmark.

This public target now requires a real torchrun launch with >=2 GPUs. It no
longer publishes substitute single-GPU collectives under the DDP overlap name.
"""

from __future__ import annotations

from ch04.ddp_no_overlap import BaselineNoOverlapBenchmark


def get_benchmark() -> BaselineNoOverlapBenchmark:
    """Factory used by the harness."""
    return BaselineNoOverlapBenchmark()
