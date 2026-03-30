"""Optimized single-GPU KV-handoff comparison benchmark."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch15.disaggregated_inference_single_common import (
    OptimizedDisaggregatedInferenceSingleGPUBenchmark,
    attach_benchmark_metadata,
)

OPTIMIZED_STORY_METADATA = {
    "pair_role": "comparison",
    "variant_role": "optimized",
    "chapter_alignment": "supplementary",
    "chapter_native_exemplar": False,
    "comparison_reason": (
        "Simulates single-GPU KV handoff to isolate host-staged versus device-resident "
        "cache movement. It is not the chapter-native multi-GPU disaggregated serving benchmark."
    ),
    "comparison_axis": "host_staged_vs_device_resident_single_gpu_kv_handoff",
    "execution_pattern": "batched_prefill_then_device_resident_decode_single_gpu",
    "optimization_mechanism": "batched device-resident KV cache reuse",
    "chapter_native_targets": ["disaggregated_inference_multigpu", "prefill_decode_disagg", "prefill_decode_disagg_multigpu"],
}


def get_benchmark() -> BaseBenchmark:
    bench = OptimizedDisaggregatedInferenceSingleGPUBenchmark(
        label="optimized_single_gpu_kv_handoff",
    )
    bench.story_metadata = dict(OPTIMIZED_STORY_METADATA)
    return attach_benchmark_metadata(bench, __file__)
