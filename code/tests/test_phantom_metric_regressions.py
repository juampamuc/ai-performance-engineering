from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from ch12.baseline_cuda_graphs import BaselineCudaGraphsBenchmark
from ch13.baseline_precisionfp8 import BaselinePrecisionFP8Benchmark
from ch15.baseline_inference_placement import BaselineInferencePlacementBenchmark
from ch20.baseline_memory_standard import BaselineMemoryStandardBenchmark
from core.scripts.update_custom_metrics import analyze_get_custom_metrics

STALE_PHANTOM_REPORT_PATHS = [
    "ch04/baseline_cpu_reduction.py",
    "ch08/baseline_loop_unrolling.py",
    "ch09/baseline_sdpa_attention.py",
    "ch14/baseline_cuda_python.py",
    "ch14/baseline_model_compile_bf16.py",
    "ch15/baseline_inference_placement.py",
    "ch15/baseline_kv_cache_management.py",
    "ch16/baseline_flash_sdp.py",
    "ch16/baseline_regional_compilation.py",
    "ch17/baseline_pipeline_parallelism.py",
]


def test_stale_phantom_metric_report_paths_are_not_currently_phantom() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for rel_path in STALE_PHANTOM_REPORT_PATHS:
        analysis = analyze_get_custom_metrics(repo_root / rel_path, repo_root)
        assert analysis["classification"] != "phantom", rel_path


def test_ch12_cuda_graph_metrics_use_structural_fields_only() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    analysis = analyze_get_custom_metrics(repo_root / "ch12/baseline_cuda_graphs.py", repo_root)
    assert analysis["classification"] != "phantom"

    if not torch.cuda.is_available():
        return

    benchmark = BaselineCudaGraphsBenchmark()
    metrics = benchmark.get_custom_metrics()
    assert "graph.baseline_launch_us" not in metrics
    assert "graph.graph_launch_us" not in metrics
    assert metrics["cuda_runtime.uses_cuda_graph"] == 0.0
    assert metrics["cuda_runtime.num_iterations"] == float(benchmark.iterations)


def test_inference_placement_metrics_keep_real_totals_without_fake_latency() -> None:
    benchmark = BaselineInferencePlacementBenchmark.__new__(BaselineInferencePlacementBenchmark)
    benchmark.cfg = SimpleNamespace(batch_size=16)
    benchmark.sessions = 64
    benchmark._total_tokens = 4096
    benchmark._total_requests = 64
    benchmark.batch_size = 16
    benchmark.max_batch_size = 16
    metrics = benchmark.get_custom_metrics()
    assert "inference.ttft_ms" not in metrics
    assert "inference.tpot_ms" not in metrics
    assert metrics["inference.total_tokens"] == 4096.0
    assert metrics["inference.total_requests"] == 64.0


def test_baseline_precision_and_ai_metrics_report_current_run_only() -> None:
    precision_benchmark = BaselinePrecisionFP8Benchmark.__new__(BaselinePrecisionFP8Benchmark)
    precision_benchmark._last_elapsed_ms = 12.5
    precision_metrics = precision_benchmark.get_custom_metrics()
    assert precision_metrics["precision.fp32_ms"] == 12.5
    assert "precision.reduced_ms" not in precision_metrics
    assert "precision.speedup" not in precision_metrics

    ai_benchmark = BaselineMemoryStandardBenchmark.__new__(BaselineMemoryStandardBenchmark)
    ai_benchmark._last_elapsed_ms = 7.5
    ai_metrics = ai_benchmark.get_custom_metrics()
    assert ai_metrics["ai_opt.original_ms"] == 7.5
    assert "ai_opt.optimized_ms" not in ai_metrics
    assert "ai_opt.suggestions_applied" not in ai_metrics
    assert "ai_opt.speedup" not in ai_metrics
