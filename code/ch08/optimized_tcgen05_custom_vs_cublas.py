"""Optimized side of the custom tcgen05 versus cuBLAS comparison."""

from __future__ import annotations

import torch

from ch08.tcgen05_custom_vs_cublas_benchmark_base import Tcgen05CustomVsCublasBase
from core.harness.benchmark_harness import BaseBenchmark


class OptimizedTcgen05CustomVsCublasBenchmark(Tcgen05CustomVsCublasBase):
    """Vendor cuBLAS reference side of the comparison pair."""

    nvtx_label = "optimized_tcgen05_custom_vs_cublas"

    def benchmark_fn(self) -> None:
        if self.matrix_a is None or self.matrix_b is None:
            raise RuntimeError("Inputs not initialized")
        with self._nvtx_range(self.nvtx_label):
            with torch.no_grad():
                self.output = torch.matmul(self.matrix_a, self.matrix_b)


def get_benchmark() -> BaseBenchmark:
    return OptimizedTcgen05CustomVsCublasBenchmark()
