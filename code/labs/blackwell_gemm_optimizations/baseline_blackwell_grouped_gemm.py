"""Baseline grouped-GEMM benchmark for the Blackwell optimization lab."""

from __future__ import annotations

from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark
from labs.blackwell_gemm_optimizations.blackwell_grouped_gemm_common import (
    BlackwellGroupedGemmBenchmark,
)


def get_benchmark() -> BaseBenchmark:
    bench = BlackwellGroupedGemmBenchmark(
        variant="baseline",
        label="baseline_blackwell_grouped_gemm",
    )
    return attach_benchmark_metadata(bench, __file__)
