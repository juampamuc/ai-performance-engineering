"""Step-3 grouped-GEMM benchmark with autotune, fusion, and swizzle."""

from __future__ import annotations

from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark
from labs.blackwell_gemm_optimizations.blackwell_grouped_gemm_common import (
    BlackwellGroupedGemmBenchmark,
)


def get_benchmark() -> BaseBenchmark:
    bench = BlackwellGroupedGemmBenchmark(
        variant="full_stack",
        label="optimized_blackwell_grouped_gemm_full_stack",
    )
    return attach_benchmark_metadata(bench, __file__)
