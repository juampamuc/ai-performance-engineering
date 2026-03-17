"""Python harness wrapper for optimized_warp_specialized_cluster_pipeline.cu."""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from core.harness.benchmark_harness import BaseBenchmark
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedWarpSpecializedClusterPipelineBenchmark(CudaBinaryBenchmark):
    """Wraps the optimized CUDA binary."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_warp_specialized_cluster_pipeline",
            friendly_name="Optimized Warp Specialized Cluster Pipeline",
            iterations=5,
            warmup=5,
            timeout_seconds=180,
            workload_params={
                "tile_size": 96,
                "tiles": 8,
                "cluster_blocks": 4,
                "batch_size": 1,
                "elements": 8 * 96 * 96,
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
            uses_cluster=True,
            uses_dsmem=True,
            cluster_leader_staging=True,
            async_staging=True,
        )


def get_benchmark() -> BaseBenchmark:
    return OptimizedWarpSpecializedClusterPipelineBenchmark()

