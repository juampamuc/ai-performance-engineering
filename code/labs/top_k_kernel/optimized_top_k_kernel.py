"""Triton block-score variant for the Top-K selection kernel lab."""

from __future__ import annotations

from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark
from labs.top_k_kernel.top_k_kernel_common import TopKKernelBenchmark


def get_benchmark() -> BaseBenchmark:
    bench = TopKKernelBenchmark(
        backend="triton",
        label="optimized_top_k_kernel_triton",
    )
    return attach_benchmark_metadata(bench, __file__)
