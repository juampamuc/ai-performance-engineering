"""Baseline: recapture the graph whenever request bindings change."""

from __future__ import annotations

from core.benchmark.wrapper_utils import attach_benchmark_metadata
from labs.parameterized_cuda_graphs.parameterized_cuda_graphs_common import (
    ParameterizedGraphConfig,
    ParameterizedGraphRecaptureBenchmark,
)


def get_benchmark() -> ParameterizedGraphRecaptureBenchmark:
    return attach_benchmark_metadata(
        ParameterizedGraphRecaptureBenchmark(ParameterizedGraphConfig()),
        __file__,
    )
