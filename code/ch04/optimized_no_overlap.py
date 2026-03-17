"""Optimized wrapper for the overlap-enabled DDP demo.

This matches the chapter narrative and expectations by exposing the
overlapped training path as `optimized_no_overlap.py`, reusing the
implementation from `ddp_overlap.py`. On single-GPU hosts this remains a
simulation of overlap using a host-buffer round-trip stand-in; the real
multi-GPU collective benchmark lives in the `*_multigpu.py` variants.
"""

from __future__ import annotations

from ch04.ddp_overlap import OptimizedOverlapDdpBenchmark


def get_benchmark() -> OptimizedOverlapDdpBenchmark:
    """Factory used by the harness."""
    return OptimizedOverlapDdpBenchmark()

