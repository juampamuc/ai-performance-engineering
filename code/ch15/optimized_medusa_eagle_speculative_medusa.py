"""Medusa-style speculative decoding benchmark with explicit acceptance tradeoffs."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch15.medusa_eagle_speculative_benchmarks import (
    MedusaEagleSpeculativeBenchmark,
    attach_benchmark_metadata,
)


MEDUSA_STORY_METADATA = {
    "pair_role": "canonical",
    "variant_role": "optimized",
    "chapter_alignment": "native",
    "chapter_native_exemplar": True,
    "comparison_axis": "target_only_greedy_decode_vs_explicit_medusa_eagle_speculative_profiles",
    "optimization_mechanism": "medusa_style_multi_branch_draft_with_more_aggressive_acceptance_tradeoff",
}


def get_benchmark() -> BaseBenchmark:
    bench = MedusaEagleSpeculativeBenchmark(
        variant="medusa",
        label="optimized_medusa_eagle_speculative_medusa",
    )
    bench.story_metadata = dict(MEDUSA_STORY_METADATA)
    return attach_benchmark_metadata(bench, __file__)
