"""MoE communication benchmark with hierarchical expert exchange buckets."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch15.moe_comm_exchange_benchmarks import MoeCommExchangeBenchmark, attach_benchmark_metadata


HIERARCHICAL_STORY_METADATA = {
    "pair_role": "canonical",
    "variant_role": "optimized",
    "chapter_alignment": "native",
    "chapter_native_exemplar": True,
    "comparison_axis": "flat_all_to_all_vs_overlap_and_hierarchical_moe_exchange",
    "optimization_mechanism": "hierarchical_grouped_exchange_before_rank_level_scatter",
}


def get_benchmark() -> BaseBenchmark:
    bench = MoeCommExchangeBenchmark(
        variant="hierarchical",
        label="optimized_moe_comm_exchange_hierarchical",
    )
    bench.story_metadata = dict(HIERARCHICAL_STORY_METADATA)
    return attach_benchmark_metadata(bench, __file__)
