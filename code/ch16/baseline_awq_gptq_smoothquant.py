"""Baseline post-training quantization benchmark without PTQ transforms."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch16.awq_gptq_smoothquant_benchmarks import PTQQuantizationBenchmark, attach_benchmark_metadata


BASELINE_STORY_METADATA = {
    "pair_role": "canonical",
    "variant_role": "baseline",
    "chapter_alignment": "native",
    "chapter_native_exemplar": True,
    "comparison_axis": "fp16_or_bf16_reference_vs_post_training_quantized_serving_path",
    "optimization_mechanism": "dense_reference_mlp_without_post_training_quantization",
}


def get_benchmark() -> BaseBenchmark:
    bench = PTQQuantizationBenchmark(
        scheme="baseline",
        label="baseline_awq_gptq_smoothquant",
    )
    bench.story_metadata = dict(BASELINE_STORY_METADATA)
    return attach_benchmark_metadata(bench, __file__)
