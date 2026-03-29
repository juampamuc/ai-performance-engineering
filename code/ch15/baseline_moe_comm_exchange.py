"""Baseline MoE communication benchmark with flat rank-by-rank exchange."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch15.moe_comm_exchange_benchmarks import MoeCommExchangeBenchmark, attach_benchmark_metadata


BASELINE_STORY_METADATA = {
    "pair_role": "canonical",
    "variant_role": "baseline",
    "chapter_alignment": "native",
    "chapter_native_exemplar": True,
    "comparison_axis": "flat_all_to_all_vs_overlap_and_hierarchical_moe_exchange",
    "optimization_mechanism": "rank_by_rank_flat_exchange_without_overlap",
}


def get_benchmark() -> BaseBenchmark:
    bench = MoeCommExchangeBenchmark(
        variant="baseline",
        label="baseline_moe_comm_exchange",
    )
    bench.story_metadata = dict(BASELINE_STORY_METADATA)
    return attach_benchmark_metadata(bench, __file__)
