"""Optimized: capture once, patch memcpy-node params, then replay."""

from __future__ import annotations

from core.benchmark.wrapper_utils import attach_benchmark_metadata
from labs.parameterized_cuda_graphs.parameterized_cuda_graphs_common import (
    ParameterizedGraphConfig,
    ParameterizedGraphReplayBenchmark,
)


def get_benchmark() -> ParameterizedGraphReplayBenchmark:
    return attach_benchmark_metadata(
        ParameterizedGraphReplayBenchmark(ParameterizedGraphConfig()),
        __file__,
    )
