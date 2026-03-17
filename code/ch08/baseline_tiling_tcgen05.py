"""Baseline tcgen05 tiling benchmark using the generic B-layout path."""

from __future__ import annotations

from pathlib import Path

import torch

from ch08.tiling_benchmark_base_tcgen05 import TilingBenchmarkBaseTCGen05


class BaselineTilingBenchmarkTCGen05(TilingBenchmarkBaseTCGen05):
    """Baseline tcgen05 wrapper that transposes B inside the extension."""

    nvtx_label = "baseline_tiling_tcgen05"

    def _invoke_kernel(self) -> None:
        assert self.extension is not None
        assert self.matrix_a is not None
        assert self.matrix_b is not None
        self.output = self.extension.matmul_tiling_tcgen05(self.matrix_a, self.matrix_b)


def get_benchmark() -> BaselineTilingBenchmarkTCGen05:
    return BaselineTilingBenchmarkTCGen05()

