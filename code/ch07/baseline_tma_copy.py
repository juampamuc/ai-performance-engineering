"""Harness wrapper for the scalar baseline in the `tma_copy` benchmark.

The chapter's clean descriptor-backed TMA benchmark is still
`tma_bulk_tensor_2d`. This baseline stays focused on the scalar neighbor-gather
path so the `tma_copy` pair continues to measure staging and locality changes.
"""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class BaselineTMACopyBenchmark(CudaBinaryBenchmark):
    """Wrap the scalar neighbor-gather baseline for `tma_copy`."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        n_elems = 1 << 25
        lookahead = 64
        redundant_reads = 8
        bytes_per_element = (redundant_reads * 3 + 1) * 4
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_tma_copy",
            friendly_name="Scalar Neighbor Gather Copy",
            iterations=3,
            warmup=5,
            timeout_seconds=120,
            workload_params={
                "N": n_elems,
                "lookahead": lookahead,
                "redundant_reads": redundant_reads,
                "dtype": "float32",
            },
        )
        self.register_workload_metadata(
            bytes_per_iteration=float(n_elems * bytes_per_element),
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return memory access metrics."""
        return {
            "copy.async_pipeline_1d": 0.0,
            "copy.tensor_map_2d_requested": 0.0,
            "copy.tensor_map_2d_runtime_candidate": 0.0,
        }

def get_benchmark() -> BaselineTMACopyBenchmark:
    """Factory for discover_benchmarks()."""
    return BaselineTMACopyBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
