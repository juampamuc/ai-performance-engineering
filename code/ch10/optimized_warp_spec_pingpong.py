"""Python harness wrapper for optimized_warp_spec_pingpong.cu."""

from __future__ import annotations
from typing import Optional

from pathlib import Path
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.benchmark.verification import simple_signature


class OptimizedWarpSpecPingPongBenchmark(CudaBinaryBenchmark):
    """Wraps the ping-pong warp-role pipeline."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_warp_spec_pingpong",
            friendly_name="Optimized Warp Spec Pingpong",
            iterations=3,
            warmup=5,
            timeout_seconds=120,
            time_regex=r"TIME_MS:\s*([0-9.]+)",
            workload_params={
                "dtype": "float32",
                "tile_size": 64,
                "tiles": 4096,
                "elements": 4096 * 64 * 64,
                "iterations": 10,
            },
        )
        self.register_workload_metadata(bytes_per_iteration=float(4096 * 64 * 64 * 3 * 4))
        self.num_stages = 2

    def get_custom_metrics(self) -> Optional[dict]:
        """Return honest pipeline metadata for the ping-pong kernel."""
        return {
            "pipeline.num_stages": float(self.num_stages),
            "pipeline.consumer_warps": 2.0,
            "pipeline.pingpong_enabled": 1.0,
        }

    def get_input_signature(self) -> dict:
        """Signature for the ping-pong warp-role pipeline."""
        return simple_signature(
            dtype="float32",
            tile_size=64,
            tiles=4096,
            elements=4096 * 64 * 64,
            iterations=10,
        ).to_dict()

    def get_output_tolerance(self) -> tuple[float, float]:
        return (0.0, 0.0)


def get_benchmark() -> OptimizedWarpSpecPingPongBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedWarpSpecPingPongBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
