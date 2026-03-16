"""Python harness wrapper for optimized_pipeline_3stage.cu - 3-Stage Software Pipeline."""

from __future__ import annotations
from typing import Optional

from pathlib import Path
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, BenchmarkHarness, BenchmarkMode
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.benchmark.verification import simple_signature
class OptimizedPipeline3StageBenchmark(CudaBinaryBenchmark):
    """Wraps the 3-stage pipeline GEMV kernel for deeper latency hiding."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_pipeline_3stage",
            friendly_name="Optimized Pipeline 3Stage",
            iterations=10,
            warmup=5,  # Minimum warmup for CUDA binary
            timeout_seconds=120,
            workload_params={
                "batch_size": 8,
                "dtype": "float32",
                "elements": 8 * 1024 * 1024,
                "segments": 128,
                "segment_size": 65536,
            },
        )
        self.register_workload_metadata(bytes_per_iteration=65536)

    def get_custom_metrics(self) -> Optional[dict]:
        """Report the declared pipeline structure and workload params."""
        from ch10.benchmark_metrics_common import compute_pipeline_variant_metrics

        return compute_pipeline_variant_metrics(
            self._workload_params,
            num_stages=3,
        )

    def get_input_signature(self) -> dict:
        """Signature for optimized 3-stage pipeline."""
        return simple_signature(
            batch_size=8,
            dtype="float32",
            elements=8 * 1024 * 1024,
            segments=128,
            segment_size=65536,
        ).to_dict()

    def get_output_tolerance(self) -> tuple[float, float]:
        return (0.0, 0.0)
def get_benchmark() -> BaseBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedPipeline3StageBenchmark()
if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
