"""Python harness wrapper for baseline_cluster_group.cu."""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.benchmark.verification import simple_signature

from ch10.cluster_group_utils import raise_cluster_skip

class BaselineClusterGroupBenchmark(CudaBinaryBenchmark):
    """Wraps the baseline cooperative group example without clusters."""

    allowed_benchmark_fn_antipatterns = ("io",)

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_cluster_group",
            friendly_name="Baseline Cluster Group",
            iterations=3,
            warmup=5,
            timeout_seconds=180,
            time_regex=r"TIME_MS:\s*([0-9.]+)",
            workload_params={
                "batch_size": 8192,  # chunks across the 16M elements
                "dtype": "float32",
                "elements": 1 << 24,
                "chunk_elems": 2048,
            },
        )
        self.register_workload_metadata(bytes_per_iteration=1024 * 1024)

    def get_custom_metrics(self) -> Optional[dict]:
        """Report the cooperative-group workload without fake pipeline timing."""
        from ch10.benchmark_metrics_common import compute_workload_param_metrics

        metrics = compute_workload_param_metrics(self._workload_params)
        metrics["reduction.uses_cluster"] = 0.0
        metrics["reduction.uses_dsmem"] = 0.0
        return metrics

    def benchmark_fn(self) -> None:
        try:
            super().benchmark_fn()
        except RuntimeError as exc:
            raise_cluster_skip(str(exc))
            raise

    def get_input_signature(self) -> dict:
        """Explicit workload signature for cluster group reduction baseline."""
        return simple_signature(
            batch_size=8192,
            dtype="float32",
            elements=1 << 24,
            chunk_elems=2048,
        ).to_dict()


def get_benchmark() -> BaselineClusterGroupBenchmark:
    """Factory for discover_benchmarks()."""
    return BaselineClusterGroupBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
