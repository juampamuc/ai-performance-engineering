"""Python harness wrapper for optimized_warp_specialized_two_pipelines_driver.cu."""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from core.harness.benchmark_harness import BaseBenchmark
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedWarpSpecializedTwoPipelinesDriverBenchmark(CudaBinaryBenchmark):
    """Wraps the optimized CUDA binary."""

    def __init__(self) -> None:
        tiles = 128
        tile_elems = 1024
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_warp_specialized_two_pipelines_driver",
            friendly_name="Optimized Warp Specialized Two Pipelines Driver",
            iterations=5,
            warmup=5,
            timeout_seconds=180,
            workload_params={
                "tiles": tiles,
                "tile_elems": tile_elems,
                "num_streams": 2,
                "dtype": "float32",
            },
        )
        bytes_per_iteration = float(tiles * tile_elems * 2 * 4)
        self.register_workload_metadata(bytes_per_iteration=bytes_per_iteration, requests_per_iteration=1.0)

    def get_custom_metrics(self) -> Optional[dict]:
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedWarpSpecializedTwoPipelinesDriverBenchmark()

