"""Harness wrapper for the optimized `tma_copy` neighbor-copy path.

The cleanest descriptor-backed TMA benchmark in the chapter remains
`tma_bulk_tensor_2d`. This wrapper now surfaces the upgraded neighbor-copy demo,
which prefers a real 2D tensor-map descriptor path when the local CUDA runtime
supports it and otherwise falls back to the async-pipeline implementation.
"""

from __future__ import annotations
from typing import Optional

from pathlib import Path

import torch

from core.harness.benchmark_harness import BaseBenchmark
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedTMACopyBenchmark(CudaBinaryBenchmark):
    """Wrap the staged neighbor-copy binary with optional descriptor-backed TMA."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        n_elems = 1 << 25
        lookahead = 64
        redundant_reads = 8
        bytes_per_element = (redundant_reads * 3 + 1) * 4
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_tma_copy",
            friendly_name="Pipeline + Tensor-Map Neighbor Copy",
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
        cuda_version = torch.version.cuda or "0.0"
        cuda_major = int(cuda_version.split(".", maxsplit=1)[0])
        return {
            "copy.async_pipeline_1d": 1.0,
            "copy.tensor_map_2d_requested": 1.0,
            "copy.tensor_map_2d_runtime_candidate": 1.0 if cuda_major >= 13 else 0.0,
        }

def get_benchmark() -> BaseBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedTMACopyBenchmark()


