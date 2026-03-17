"""Python harness wrapper for optimized_dsmem_reduction_cluster_atomic.cu.

DSMEM Cluster Atomic: Uses map_shared_rank() + atomicAdd for cross-CTA aggregation.

BOOK REFERENCE (Ch10): DSMEM (Distributed Shared Memory) allows CTAs within
a cluster to communicate through shared memory without global memory round-trips.

Key pattern:
  1. Each CTA performs block-level reduction
  2. Each CTA atomically adds its result to the cluster leader's smem via DSMEM
  3. Cluster leader writes final result to global memory

This is faster than two-pass reduction because:
  - No intermediate global memory writes between passes
  - Cluster sync is cheaper than kernel launch overhead
"""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.benchmark.verification import simple_signature


class OptimizedDSMEMClusterAtomicBenchmark(CudaBinaryBenchmark):
    """Wraps DSMEM cluster reduction using atomic aggregation."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_dsmem_reduction_cluster_atomic",
            friendly_name="Optimized Dsmem Reduction Cluster Atomic",
            iterations=3,
            warmup=5,
            # Full deep-dive instrumentation can slow the standalone CUDA binary
            # enough that the default subprocess budget becomes a false failure.
            timeout_seconds=600,
            workload_params={
                "batch_size": 1024,
                "dtype": "float32",
                "N": 64 * 1024 * 1024,
                "cluster_size": 4,
                "block_elems": 4096,
            },
        )
        self.register_workload_metadata(bytes_per_iteration=float(64 * 1024 * 1024 * 4))

    def get_custom_metrics(self) -> Optional[dict]:
        """Report the reduction workload without fake pipeline timing."""
        from ch10.benchmark_metrics_common import compute_workload_param_metrics

        metrics = compute_workload_param_metrics(self._workload_params)
        metrics["reduction.uses_dsmem"] = 1.0
        metrics["reduction.uses_cluster_atomic"] = 1.0
        metrics["reduction.cluster_size"] = float(self._workload_params["cluster_size"])
        return metrics

    def get_input_signature(self) -> dict:
        """Signature for DSMEM cluster atomic reduction."""
        return simple_signature(
            batch_size=1,
            dtype="float32",
            N=64 * 1024 * 1024,
        ).to_dict()

    def get_output_tolerance(self) -> tuple[float, float]:
        return (0.0, 0.0)

    def get_config(self) -> BenchmarkConfig:
        config = super().get_config()
        # This standalone CUDA binary does not benefit from full library tracing,
        # and deep-dive NSYS captures are more reliable with the light preset.
        config.nsys_timeout_seconds = 300
        config.nsys_preset_override = "light"
        config.profiling_warmup = 0
        config.profiling_iterations = 1
        config.profile_env_overrides = {
            "AISP_CUDA_BINARY_PROFILE_WARMUP": "0",
            "AISP_CUDA_BINARY_PROFILE_ITERATIONS": "1",
        }
        return config


def get_benchmark() -> BaseBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedDSMEMClusterAtomicBenchmark()


