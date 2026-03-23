"""Python harness wrapper for optimized_launch_bounds_cuda.cu."""

from __future__ import annotations
from pathlib import Path
from typing import Optional

from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.harness.benchmark_harness import BaseBenchmark


class OptimizedLaunchBoundsCudaBenchmark(CudaBinaryBenchmark):
    """Wraps the optimized CUDA binary."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_launch_bounds_cuda",
            friendly_name="Optimized Launch Bounds CUDA",
            iterations=5,
            warmup=5,
            timeout_seconds=180,
            workload_params={
                "N": 1024 * 1024,
                "dtype": "float32",
                "batch_size": 1,
            },
        )

    def get_custom_metrics(self) -> Optional[dict]:
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedLaunchBoundsCudaBenchmark()
