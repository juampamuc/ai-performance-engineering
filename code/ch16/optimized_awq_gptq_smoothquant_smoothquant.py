"""Optimized SmoothQuant benchmark for the Chapter 16 PTQ family."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch16.awq_gptq_smoothquant_benchmarks import PTQQuantizationBenchmark, attach_benchmark_metadata


SMOOTHQUANT_STORY_METADATA = {
    "pair_role": "canonical",
    "variant_role": "optimized",
    "chapter_alignment": "native",
    "chapter_native_exemplar": True,
    "comparison_axis": "fp16_or_bf16_reference_vs_post_training_quantized_serving_path",
    "optimization_mechanism": "smoothquant_activation_migration_with_int8_matmul",
}


def get_benchmark() -> BaseBenchmark:
    bench = PTQQuantizationBenchmark(
        scheme="smoothquant",
        label="optimized_awq_gptq_smoothquant_smoothquant",
    )
    bench.story_metadata = dict(SMOOTHQUANT_STORY_METADATA)
    return attach_benchmark_metadata(bench, __file__)
