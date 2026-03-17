"""Python harness wrapper for baseline_cooperative_persistent.cu."""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.benchmark.verification import simple_signature


class BaselineCooperativePersistentBenchmark(CudaBinaryBenchmark):
    """Wraps the single-launch cooperative persistent baseline kernel."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_cooperative_persistent",
            friendly_name="Baseline Cooperative Persistent",
            iterations=3,
            warmup=5,
            timeout_seconds=180,
            time_regex=r"TIME_MS:\s*([0-9.]+)",
            workload_params={
                "batch_size": 1 << 24,
                "dtype": "float32",
                "elements": 1 << 24,
                "iterations": 40,
            },
        )
        self.register_workload_metadata(bytes_per_iteration=float((1 << 24) * 2 * 4))


    def get_custom_metrics(self) -> Optional[dict]:
        """Report the persistent-kernel structure without fake stage timing."""
        from ch10.benchmark_metrics_common import compute_pipeline_variant_metrics

        return compute_pipeline_variant_metrics(
            self._workload_params,
            num_stages=1,
            persistent=True,
        )

    def get_input_signature(self) -> dict:
        """Explicit signature for the cooperative persistent baseline."""
        return simple_signature(
            batch_size=1 << 24,
            dtype="float32",
            elements=1 << 24,
            iterations=40,
        ).to_dict()

    def get_output_tolerance(self) -> tuple[float, float]:
        return (0.0, 0.0)

def get_benchmark() -> BaselineCooperativePersistentBenchmark:
    """Factory for discover_benchmarks()."""
    return BaselineCooperativePersistentBenchmark()


