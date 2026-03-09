"""Python harness wrapper for baseline_warp_spec_pingpong.cu."""

from __future__ import annotations
from typing import Optional

from pathlib import Path
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.benchmark.verification import simple_signature


class BaselineWarpSpecPingPongBenchmark(CudaBinaryBenchmark):
    """Wraps the single-stage warp-role pipeline baseline."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_warp_spec_pingpong",
            friendly_name="Baseline Warp Spec Pingpong",
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

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_pipeline_metrics
        return compute_pipeline_metrics(
            num_stages=getattr(self, 'num_stages', 4),
            stage_times_ms=getattr(self, '_stage_times_ms', [1.0]),
        )

    def get_input_signature(self) -> dict:
        """Signature for the single-stage warp-role pipeline."""
        return simple_signature(
            dtype="float32",
            tile_size=64,
            tiles=4096,
            elements=4096 * 64 * 64,
            iterations=10,
        ).to_dict()

    def get_output_tolerance(self) -> tuple[float, float]:
        return (0.0, 0.0)
def get_benchmark() -> BaselineWarpSpecPingPongBenchmark:
    """Factory for discover_benchmarks()."""
    return BaselineWarpSpecPingPongBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
