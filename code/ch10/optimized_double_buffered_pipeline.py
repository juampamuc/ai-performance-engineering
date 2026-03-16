"""Python harness wrapper for optimized_double_buffered_pipeline.cu."""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.benchmark.verification import simple_signature


class OptimizedDoubleBufferedPipelineBenchmark(CudaBinaryBenchmark):
    """Wraps the optimized double-buffered pipeline kernel."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_double_buffered_pipeline",
            friendly_name="Optimized Double Buffered Pipeline",
            iterations=3,
            warmup=5,
            timeout_seconds=180,
            requires_pipeline_api=True,
            time_regex=r"TIME_MS:\s*([0-9.]+)",
            workload_params={
                "batch_size": 2048,
                "dtype": "float32",
                "M": 2048,
                "N": 2048,
                "K": 2048,
            },
        )
        self.register_workload_metadata(bytes_per_iteration=1024 * 1024)


    def get_custom_metrics(self) -> Optional[dict]:
        """Report the declared pipeline structure and workload params."""
        from ch10.benchmark_metrics_common import compute_pipeline_variant_metrics

        return compute_pipeline_variant_metrics(
            self._workload_params,
            num_stages=2,
            double_buffered=True,
        )

    def get_input_signature(self) -> dict:
        """GEMM workload signature for optimized double-buffered pipeline."""
        return simple_signature(
            batch_size=2048,
            dtype="float32",
            M=2048,
            N=2048,
            K=2048,
        ).to_dict()

def get_benchmark() -> OptimizedDoubleBufferedPipelineBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedDoubleBufferedPipelineBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
