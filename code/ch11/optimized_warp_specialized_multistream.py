"""Python harness wrapper for optimized_warp_specialized_multistream.cu."""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from core.harness.benchmark_harness import BaseBenchmark
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedWarpSpecializedMultistreamBenchmark(CudaBinaryBenchmark):
    """Wraps the optimized CUDA binary."""

    def __init__(self) -> None:
        tile = 32
        threads = 96
        batches = 4096
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_warp_specialized_multistream",
            friendly_name="Optimized Warp Specialized Multistream",
            iterations=5,
            warmup=5,
            timeout_seconds=180,
            workload_params={
                "TILE": tile,
                "THREADS": threads,
                "batches": batches,
                "dtype": 'float32',
            },
        )
        bytes_per_iteration = float(tile * threads * batches * 4)
        self.register_workload_metadata(bytes_per_iteration=bytes_per_iteration, requests_per_iteration=1.0)

    def get_custom_metrics(self) -> Optional[dict]:
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedWarpSpecializedMultistreamBenchmark()

