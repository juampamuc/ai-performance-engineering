"""Python harness wrapper for optimized_atomic_reduction.cu.

Chapter 10 - DSMEM-Free Optimized
This is the single-pass atomic reduction that works on ANY CUDA device.

Compare with baseline_atomic_reduction.py (two-pass).
For DSMEM-enabled hardware, use optimized_dsmem_reduction_variant1.py.
"""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.benchmark.verification import simple_signature


class OptimizedAtomicReductionBenchmark(CudaBinaryBenchmark):
    """Wraps the single-pass atomic reduction kernel (DSMEM-free optimized)."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_atomic_reduction",
            friendly_name="Optimized Atomic Reduction",
            iterations=3,
            warmup=5,
            timeout_seconds=120,
            workload_params={"type": "atomic_reduction"},
        )
        self.register_workload_metadata(bytes_per_iteration=64 * 1024 * 1024)

    def get_custom_metrics(self) -> Optional[dict]:
        """Report the actual reduction workload and measured bandwidth."""
        from ch10.benchmark_metrics_common import compute_reduction_workload_metrics

        return compute_reduction_workload_metrics(
            num_elements=64 * 1024 * 1024,
            elapsed_ms=self.last_time_ms,
            single_pass=True,
        )

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return simple_signature(batch_size=1, dtype="float32", workload=1).to_dict()

    def get_output_tolerance(self) -> tuple[float, float]:
        return (0.1, 1.0)


def get_benchmark() -> BaseBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedAtomicReductionBenchmark()


