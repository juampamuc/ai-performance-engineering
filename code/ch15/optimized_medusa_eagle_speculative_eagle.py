"""EAGLE-style speculative decoding benchmark with explicit acceptance tradeoffs."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch15.medusa_eagle_speculative_benchmarks import (
    MedusaEagleSpeculativeBenchmark,
    attach_benchmark_metadata,
)


EAGLE_STORY_METADATA = {
    "pair_role": "canonical",
    "variant_role": "optimized",
    "chapter_alignment": "native",
    "chapter_native_exemplar": True,
    "comparison_axis": "target_only_greedy_decode_vs_explicit_medusa_eagle_speculative_profiles",
    "optimization_mechanism": "eagle_style_verifier_friendly_draft_with_higher_acceptance_rate",
}


def get_benchmark() -> BaseBenchmark:
    bench = MedusaEagleSpeculativeBenchmark(
        variant="eagle",
        label="optimized_medusa_eagle_speculative_eagle",
    )
    bench.story_metadata = dict(EAGLE_STORY_METADATA)
    return attach_benchmark_metadata(bench, __file__)
