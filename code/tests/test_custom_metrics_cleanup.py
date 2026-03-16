from __future__ import annotations

import math

import torch

from ch02.baseline_cublas import BaselineCublasBenchmark
from ch02.baseline_memory_transfer import BaselineMemoryTransferBenchmark
from ch02.optimized_cublas import OptimizedCublasBenchmark
from ch02.optimized_memory_transfer import OptimizedMemoryTransferBenchmark
from ch01.baseline_performance import BaselinePerformanceBenchmark
from ch01.optimized_performance_fusion import OptimizedPerformanceFusionBenchmark
from ch05.ai_common import TinyBlock as CommonTinyBlock
from ch05.baseline_ai import BaselineAIBenchmark, TinyBlock as BaselineTinyBlock
from ch05.baseline_host_staged_reduction import BaselineHostStagedReductionBenchmark
from ch05.baseline_storage_cpu import BaselineStorageCpuBenchmark
from ch05.baseline_vectorization import BaselineVectorizationBenchmark
from ch05.optimized_ai import OptimizedAIBenchmark, TinyBlock as OptimizedTinyBlock
from ch05.optimized_host_staged_reduction import OptimizedHostStagedReductionBenchmark
from ch05.optimized_storage_cpu import OptimizedStorageGdsBenchmark
from ch05.optimized_vectorization import OptimizedVectorizationBenchmark
from ch10.baseline_batch import BaselineBatchBenchmark
from ch10.baseline_double_buffered_pipeline import BaselineDoubleBufferedPipelineBenchmark
from ch10.baseline_flash_attention import BaselineFlashAttentionBenchmark
from ch10.optimized_flash_attention import OptimizedFlashAttentionBenchmark
from ch10.optimized_batch import OptimizedBatchBenchmark
from ch10.optimized_double_buffered_pipeline import OptimizedDoubleBufferedPipelineBenchmark
from ch15.baseline_inference_monolithic import SimpleLLM as BaselineMonolithicLLM
from ch15.inference_monolithic_common import SimpleLLM as CommonMonolithicLLM
from ch15.optimized_inference_monolithic import SimpleLLM as OptimizedMonolithicLLM
from ch17.baseline_prefill_decode_disagg import SimpleLLM as BaselineDisaggLLM
from ch17.optimized_prefill_decode_disagg import SimpleLLM as OptimizedDisaggLLM
from ch17.prefill_decode_disagg_monolithic_common import SimpleLLM as CommonDisaggLLM
from core.scripts.update_custom_metrics import CHAPTER_METRIC_HELPERS
from labs.flexattention.baseline_flex_attention import BaselineFlexAttentionBenchmark
from labs.flexattention.optimized_flex_attention import OptimizedFlexAttentionBenchmark


def test_ch01_environment_metrics_use_runtime_detection() -> None:
    expected_gpu_count = float(torch.cuda.device_count() if torch.cuda.is_available() else 0)

    baseline_metrics = BaselinePerformanceBenchmark().get_custom_metrics()
    optimized_metrics = OptimizedPerformanceFusionBenchmark().get_custom_metrics()

    assert baseline_metrics["env.gpu_count"] == expected_gpu_count
    assert optimized_metrics["env.gpu_count"] == expected_gpu_count
    if torch.cuda.is_available():
        assert baseline_metrics["env.gpu_memory_gb"] > 0.0
    else:
        assert baseline_metrics["env.gpu_memory_gb"] == 0.0


def test_ch05_ai_metrics_report_workload_shape_not_storage_defaults() -> None:
    baseline = BaselineAIBenchmark()
    baseline.parameter_count = 123_456
    optimized = OptimizedAIBenchmark()
    optimized.parameter_count = 654_321

    baseline_metrics = baseline.get_custom_metrics()
    optimized_metrics = optimized.get_custom_metrics()

    assert "storage.bytes_read" not in baseline_metrics
    assert baseline_metrics["ai.num_blocks"] == float(baseline.num_blocks)
    assert baseline_metrics["ai.activation_elements_per_iteration"] == float(
        baseline.batch * baseline.hidden * baseline.num_blocks
    )
    assert optimized_metrics["ai.parameters_millions"] == optimized.parameter_count / 1_000_000.0


def test_ch10_flash_attention_metrics_report_attention_shape() -> None:
    baseline = BaselineFlashAttentionBenchmark()
    optimized = OptimizedFlashAttentionBenchmark()

    baseline_metrics = baseline.get_custom_metrics()
    optimized_metrics = optimized.get_custom_metrics()

    expected_score_elements = float(
        baseline.batch_size * baseline.num_heads * baseline.seq_len * baseline.seq_len
    )
    assert "pipeline.max_stage_ms" not in baseline_metrics
    assert baseline_metrics["attention.score_matrix_elements"] == expected_score_elements
    assert optimized_metrics["attention.output_elements"] == float(
        optimized.batch_size * optimized.seq_len * optimized.hidden_dim
    )


def test_ch02_gemm_benchmarks_report_gemm_and_roofline_metrics() -> None:
    baseline = BaselineCublasBenchmark()
    optimized = OptimizedCublasBenchmark()
    baseline._last_elapsed_ms = 2.5
    optimized._last_elapsed_ms = 1.25

    baseline_metrics = baseline.get_custom_metrics()
    optimized_metrics = optimized.get_custom_metrics()
    expected_flops = float(2 * baseline.m * baseline.n * baseline.k)

    assert baseline_metrics["gemm.total_flops"] == expected_flops
    assert optimized_metrics["gemm.total_flops"] == expected_flops
    assert "roofline.achieved_tflops" in baseline_metrics
    assert "roofline.achieved_tflops" in optimized_metrics
    assert "transfer.achieved_gbps" not in baseline_metrics
    assert optimized_metrics["roofline.achieved_tflops"] > baseline_metrics["roofline.achieved_tflops"]


def test_ch02_transfer_benchmarks_still_report_transfer_metrics() -> None:
    baseline = BaselineMemoryTransferBenchmark()
    optimized = OptimizedMemoryTransferBenchmark()
    baseline._last_elapsed_ms = 4.0
    optimized._last_elapsed_ms = 2.0

    baseline_metrics = baseline.get_custom_metrics()
    optimized_metrics = optimized.get_custom_metrics()

    assert baseline_metrics["transfer.bytes"] == float(baseline.N * 4)
    assert "gemm.total_flops" not in baseline_metrics
    assert optimized_metrics["transfer.achieved_gbps"] > baseline_metrics["transfer.achieved_gbps"]


def test_flexattention_metrics_use_sparse_attention_workload_formula() -> None:
    baseline = BaselineFlexAttentionBenchmark()
    optimized = OptimizedFlexAttentionBenchmark()

    baseline_metrics = baseline.get_custom_metrics()
    optimized_metrics = optimized.get_custom_metrics()

    docs = math.ceil(baseline.seq_len / baseline.doc_span)
    active_pairs_per_head = 0
    for doc_idx in range(docs):
        start = doc_idx * baseline.doc_span
        stop = min(baseline.seq_len, start + baseline.doc_span)
        active_pairs_per_head += (stop - start) ** 2
    expected_pairs = float(baseline.batch * baseline.heads * active_pairs_per_head)
    expected_flops = 4.0 * expected_pairs * float(baseline.head_dim)

    assert baseline_metrics["flex_attention.active_score_pairs"] == expected_pairs
    assert baseline_metrics["flex_attention.total_flops"] == expected_flops
    assert optimized_metrics["flex_attention.total_flops"] == expected_flops
    assert baseline_metrics["flex_attention.arithmetic_intensity"] > 0.0


def test_ch05_remaining_metrics_no_longer_emit_storage_bandwidth_placeholders() -> None:
    baseline_vector = BaselineVectorizationBenchmark().get_custom_metrics()
    optimized_vector = OptimizedVectorizationBenchmark().get_custom_metrics()
    baseline_storage = BaselineStorageCpuBenchmark().get_custom_metrics()
    optimized_storage = OptimizedStorageGdsBenchmark().get_custom_metrics()
    baseline_reduction = BaselineHostStagedReductionBenchmark().get_custom_metrics()
    optimized_reduction = OptimizedHostStagedReductionBenchmark().get_custom_metrics()

    assert "storage.read_gbps" not in baseline_vector
    assert baseline_vector["vectorization.is_vectorized"] == 0.0
    assert optimized_vector["vectorization.is_vectorized"] == 1.0

    assert baseline_storage["storage.uses_cpu_staging"] == 1.0
    assert optimized_storage["storage.simulates_gpu_direct"] == 1.0
    assert "storage.total_gbps" not in baseline_storage

    assert baseline_reduction["reduction.host_staging_round_trips"] == 2.0
    assert optimized_reduction["reduction.keeps_reduction_on_device"] == 1.0


def test_ch10_remaining_metrics_no_longer_emit_fake_pipeline_timing() -> None:
    baseline_batch = BaselineBatchBenchmark().get_custom_metrics()
    optimized_batch = OptimizedBatchBenchmark().get_custom_metrics()
    baseline_binary = BaselineDoubleBufferedPipelineBenchmark().get_custom_metrics()
    optimized_binary = OptimizedDoubleBufferedPipelineBenchmark().get_custom_metrics()

    assert "pipeline.max_stage_ms" not in baseline_batch
    assert baseline_batch["batch.micro_batches"] == float(BaselineBatchBenchmark().micro_batches)
    assert optimized_batch["batch.micro_batches"] == 1.0

    assert "pipeline.max_stage_ms" not in baseline_binary
    assert baseline_binary["pipeline.num_stages"] == 1.0
    assert optimized_binary["pipeline.double_buffered"] == 1.0
    assert optimized_binary["workload.M"] == 2048.0


def test_chapter_models_import_shared_classes() -> None:
    assert BaselineTinyBlock is CommonTinyBlock
    assert OptimizedTinyBlock is CommonTinyBlock
    assert BaselineMonolithicLLM is CommonMonolithicLLM
    assert OptimizedMonolithicLLM is CommonMonolithicLLM
    assert BaselineDisaggLLM is CommonDisaggLLM
    assert OptimizedDisaggLLM is CommonDisaggLLM


def test_update_custom_metrics_no_longer_auto_suggests_bogus_helpers_for_ch5_and_ch10() -> None:
    assert CHAPTER_METRIC_HELPERS[5] is None
    assert CHAPTER_METRIC_HELPERS[10] is None
