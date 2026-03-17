"""HBM baseline benchmark with poor memory layout."""

from __future__ import annotations

from ch08.hbm_benchmark_base import HBMBenchmarkBase


class BaselineHBMBenchmark(HBMBenchmarkBase):
    nvtx_label = "baseline_hbm"

    def _invoke_kernel(self) -> None:
        assert self.extension is not None
        assert self.host_col is not None
        assert self.matrix_col is not None
        assert self.output is not None
        self.extension.hbm_baseline(self.matrix_col, self.output)



def get_benchmark() -> HBMBenchmarkBase:
    return BaselineHBMBenchmark()


