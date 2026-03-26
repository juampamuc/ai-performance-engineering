"""Baseline single-GPU KV-handoff control benchmark."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch15.disaggregated_inference_single_common import (
    BaselineDisaggregatedInferenceSingleGPUBenchmark,
    attach_benchmark_metadata,
)

BASELINE_STORY_METADATA = {
    "pair_role": "control",
    "variant_role": "baseline",
    "chapter_alignment": "supplementary",
    "chapter_native_exemplar": False,
    "control_reason": (
        "Simulates single-GPU KV handoff to isolate host-staged versus device-resident "
        "cache movement. It is not the chapter-native multi-GPU disaggregated serving benchmark."
    ),
    "comparison_axis": "host_staged_vs_device_resident_single_gpu_kv_handoff",
    "execution_pattern": "serialized_prefill_then_decode_single_gpu",
    "optimization_mechanism": "host-staged KV cache transfer",
    "chapter_native_targets": ["disaggregated_inference_multigpu", "prefill_decode_disagg", "prefill_decode_disagg_multigpu"],
}


def get_benchmark() -> BaseBenchmark:
    bench = BaselineDisaggregatedInferenceSingleGPUBenchmark(
        label="baseline_single_gpu_kv_handoff",
    )
    bench.story_metadata = dict(BASELINE_STORY_METADATA)
    return attach_benchmark_metadata(bench, __file__)

