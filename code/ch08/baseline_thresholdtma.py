"""Baseline threshold benchmark gated for Blackwell TMA comparisons."""

from __future__ import annotations

from ch08.threshold_tma_benchmark_base import ThresholdBenchmarkBaseTMA


class BaselineThresholdTMABenchmark(ThresholdBenchmarkBaseTMA):
    """Runs the branchy baseline but only on Blackwell/GB-series GPUs."""

    nvtx_label = "baseline_threshold_tma"

    def _invoke_kernel(self) -> None:
        assert self.extension is not None
        assert self.inputs is not None
        assert self.outputs is not None
        self.extension.threshold_tma_baseline(self.inputs, self.outputs, self.threshold)



def get_benchmark() -> ThresholdBenchmarkBaseTMA:
    return BaselineThresholdTMABenchmark()

