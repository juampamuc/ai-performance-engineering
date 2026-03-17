"""Python harness wrapper for optimized_warp_specialized_pipeline.cu."""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from core.harness.benchmark_harness import BaseBenchmark
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedWarpSpecializedPipelineBenchmark(CudaBinaryBenchmark):
    """Wraps the optimized CUDA binary."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_warp_specialized_pipeline",
            friendly_name="Optimized Warp Specialized Pipeline",
            iterations=5,
            warmup=5,
            timeout_seconds=180,
            workload_params={
                "TILE_SIZE": 64,
                "tiles": 512,
                "batch_size": 1,
                "elements": 512 * 64 * 64,
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
            uses_pipeline_api=True,
            async_staging=True,
        )


def get_benchmark() -> BaseBenchmark:
    return OptimizedWarpSpecializedPipelineBenchmark()

