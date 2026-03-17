"""Optimized single-CTA fallback for cooperative group reduction workload."""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from core.harness.benchmark_harness import BaseBenchmark
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.benchmark.verification import simple_signature


class OptimizedClusterGroupSingleCtaBenchmark(CudaBinaryBenchmark):
    """Optimized fallback using shared-memory + warp reductions."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_cluster_group_single_cta",
            friendly_name="Optimized Cluster Group Single Cta",
            iterations=3,
            warmup=5,
            timeout_seconds=60,
            workload_params={
                "batch_size": 8192,
                "dtype": "float32",
                "elements": 1 << 24,
                "chunk_elems": 2048,
            },
        )
        self.register_workload_metadata(bytes_per_iteration=1024 * 1024)


    def get_custom_metrics(self) -> Optional[dict]:
        """Report the single-CTA fallback workload without fake pipeline timing."""
        from ch10.benchmark_metrics_common import compute_workload_param_metrics

        metrics = compute_workload_param_metrics(self._workload_params)
        metrics["reduction.single_cta"] = 1.0
        metrics["reduction.uses_cluster"] = 0.0
        return metrics

    def get_input_signature(self) -> dict:
        """Signature for optimized single-CTA reduction."""
        return simple_signature(
            batch_size=8192,
            dtype="float32",
            elements=1 << 24,
            chunk_elems=2048,
        ).to_dict()

def get_benchmark() -> OptimizedClusterGroupSingleCtaBenchmark:
    return OptimizedClusterGroupSingleCtaBenchmark()


