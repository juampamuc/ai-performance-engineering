"""Python harness wrapper for optimized_warp_specialized_pipeline_enhanced.cu."""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from core.harness.benchmark_harness import BaseBenchmark
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedWarpSpecializedPipelineEnhancedBenchmark(CudaBinaryBenchmark):
    """Wraps the optimized CUDA binary."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_warp_specialized_pipeline_enhanced",
            friendly_name="Optimized Warp Specialized Pipeline Enhanced",
            iterations=5,
            warmup=5,
            timeout_seconds=180,
            workload_params={
                "min_tile_size": 8,
                "max_tile_size": 32,
                "tile_candidate_count": 3,
                "min_tiles": 4096,
                "max_tiles": 16384,
                "shared_tiles": 3,
                "iterations": 10,
                "batch_size": 1,
            },
        )

    def get_custom_metrics(self) -> Optional[dict]:
        from ch10.benchmark_metrics_common import compute_warp_specialization_metrics

        return compute_warp_specialization_metrics(
            self._workload_params,
            num_stages=2,
            producer_warps=1,
            compute_warps=1,
            consumer_warps=1,
            adaptive_tile_selection=True,
            uses_pipeline_api=True,
            async_staging=True,
        )


def get_benchmark() -> BaseBenchmark:
    return OptimizedWarpSpecializedPipelineEnhancedBenchmark()

