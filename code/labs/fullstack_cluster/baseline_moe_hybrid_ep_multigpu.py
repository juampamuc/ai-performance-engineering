"""Baseline multi-node aware DeepSeek-style hybrid expert-parallel benchmark."""

from __future__ import annotations

from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark

from labs.fullstack_cluster.moe_hybrid_ep_common import MoEHybridEPBenchmark


def get_benchmark() -> BaseBenchmark:
    bench = MoEHybridEPBenchmark(
        optimized=False,
        multigpu=True,
        script_path=__file__,
        label="baseline_moe_hybrid_ep_multigpu",
    )
    return attach_benchmark_metadata(bench, __file__)
