"""Optimized cache-aware disaggregated inference benchmark."""

from __future__ import annotations

from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark

from labs.cache_aware_disagg_inference.cache_aware_disagg_common import CacheAwareDisaggBenchmark


def get_benchmark() -> BaseBenchmark:
    bench = CacheAwareDisaggBenchmark(
        optimized=True,
        label="optimized_cache_aware_disagg",
        script_path=__file__,
    )
    return attach_benchmark_metadata(bench, __file__)
