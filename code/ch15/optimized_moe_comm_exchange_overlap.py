"""MoE communication benchmark with explicit overlap of remote exchange and local expert work."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch15.moe_comm_exchange_benchmarks import MoeCommExchangeBenchmark, attach_benchmark_metadata


OVERLAP_STORY_METADATA = {
    "pair_role": "canonical",
    "variant_role": "optimized",
    "chapter_alignment": "native",
    "chapter_native_exemplar": True,
    "comparison_axis": "flat_all_to_all_vs_overlap_and_hierarchical_moe_exchange",
    "optimization_mechanism": "overlap_remote_exchange_with_local_expert_compute",
}


def get_benchmark() -> BaseBenchmark:
    bench = MoeCommExchangeBenchmark(
        variant="overlap",
        label="optimized_moe_comm_exchange_overlap",
    )
    bench.story_metadata = dict(OVERLAP_STORY_METADATA)
    return attach_benchmark_metadata(bench, __file__)
