"""Optimized AWQ benchmark for the Chapter 16 PTQ family."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch16.awq_gptq_smoothquant_benchmarks import PTQQuantizationBenchmark, attach_benchmark_metadata


AWQ_STORY_METADATA = {
    "pair_role": "canonical",
    "variant_role": "optimized",
    "chapter_alignment": "native",
    "chapter_native_exemplar": True,
    "comparison_axis": "fp16_or_bf16_reference_vs_post_training_quantized_serving_path",
    "optimization_mechanism": "activation_aware_weight_quantization_with_int4_weights",
}


def get_benchmark() -> BaseBenchmark:
    bench = PTQQuantizationBenchmark(
        scheme="awq",
        label="optimized_awq_gptq_smoothquant_awq",
    )
    bench.story_metadata = dict(AWQ_STORY_METADATA)
    return attach_benchmark_metadata(bench, __file__)
