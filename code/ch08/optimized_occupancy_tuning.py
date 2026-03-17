"""Optimized occupancy tuning by increasing CTA size only."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ch08.baseline_occupancy_tuning import OccupancyBinaryBenchmark


class OptimizedOccupancyTuningBenchmark(OccupancyBinaryBenchmark):
    """Optimize occupancy by increasing block size while holding other knobs fixed."""

    def __init__(self) -> None:
        super().__init__(
            friendly_name="Occupancy Tuning (block=256)",
            run_args=[
                "--block-size",
                "256",
                "--smem-bytes",
                "0",
                "--unroll",
                "1",
                "--inner-iters",
                "1",
                "--reps",
                "60",
            ],
        )


    def get_custom_metrics(self) -> Optional[dict]:
        """Return optimization metrics for occupancy_tuning."""
        from core.benchmark.metrics import compute_speedup_metrics
        return compute_speedup_metrics(
            baseline_ms=None,
            optimized_ms=getattr(self, '_last_elapsed_ms', None),
            name="occupancy_tuning",
        )



def get_benchmark() -> OptimizedOccupancyTuningBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedOccupancyTuningBenchmark()
