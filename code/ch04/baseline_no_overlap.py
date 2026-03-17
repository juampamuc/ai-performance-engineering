"""Baseline wrapper for the no-overlap DDP demo.

This module exists to match the chapter text and expectations file.  It
simply re-exports the BaselineNoOverlapBenchmark defined in
`ddp_no_overlap.py`, which performs a single-GPU simulation of the
no-overlap pattern. The host-buffer round-trip is a stand-in for
all-reduce latency; the real collective version lives in the
`*_multigpu.py` benchmarks.
"""

from __future__ import annotations

from ch04.ddp_no_overlap import BaselineNoOverlapBenchmark


def get_benchmark() -> BaselineNoOverlapBenchmark:
    """Factory used by the harness."""
    return BaselineNoOverlapBenchmark()

