"""Inline tcgen05 CTA-group::2 benchmark for SM100-class hardware."""

from __future__ import annotations

from labs.fullstack_cluster import optimized_matmul_tcgen05_cta2
from labs.fullstack_cluster.capstone_benchmarks import CapstoneMatmulBenchmark
from labs.fullstack_cluster.gpu_requirements import ensure_tcgen05_supported


class OptimizedCapstoneGemmTCGen05Benchmark(CapstoneMatmulBenchmark):
    def __init__(self) -> None:
        super().__init__(
            runner=optimized_matmul_tcgen05_cta2,
            label="capstone_optimized_tcgen05_cta2",
            iterations=3,
            warmup=5,
            timeout_seconds=300,
            validate_against_baseline=False,
        )

    def setup(self) -> None:
        ensure_tcgen05_supported()
        super().setup()

    def get_optimization_goal(self) -> str:
        """This tcgen05 follow-up stays runnable as a comparison benchmark."""
        return "comparison"



def get_benchmark() -> OptimizedCapstoneGemmTCGen05Benchmark:
    return OptimizedCapstoneGemmTCGen05Benchmark()
