"""Python harness wrapper for ch07's layout-transformed lookup kernel."""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedLookupBenchmark(CudaBinaryBenchmark):
    """Wrap the lookup kernel after path-table layout transformation."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        n_elems = 1 << 20
        random_steps = 64
        iterations = 200
        bytes_per_element = (random_steps + 1) * 4  # random reads + write
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_lookup",
            friendly_name="Optimized Lookup (Pretransposed Paths)",
            iterations=3,
            warmup=5,
            timeout_seconds=90,
            time_regex=r"TIME_MS:\s*([0-9.]+)",
            workload_params={
                "N": n_elems,
                "random_steps": random_steps,
                "iterations": iterations,
                "dtype": "float32",
            },
        )
        self.register_workload_metadata(
            bytes_per_iteration=float(n_elems * bytes_per_element),
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Expose the layout transformation so reports do not hide it."""
        return {
            "reads_per_output": 64.0,
            "layout_pretransposed": 1.0,
            "pointer_chase_in_kernel": 0.0,
        }

def get_benchmark() -> OptimizedLookupBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedLookupBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
