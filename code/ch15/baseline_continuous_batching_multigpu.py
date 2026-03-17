"""baseline_continuous_batching_multigpu.py - Static batching (multi-GPU)."""

from __future__ import annotations

from core.utils.continuous_batching import ContinuousBatchingBase
from core.benchmark.verification_mixin import VerificationPayloadMixin


class BaselineContinuousBatchingMultiGPUBenchmark(VerificationPayloadMixin, ContinuousBatchingBase):
    """Baseline: padded static batching across all visible GPUs."""

    multi_gpu_required = True

    def __init__(self) -> None:
        super().__init__(
            dynamic=False,
            multi_gpu=True,
            label="baseline_continuous_batching_multigpu",
        )


def get_benchmark() -> BaselineContinuousBatchingMultiGPUBenchmark:
    return BaselineContinuousBatchingMultiGPUBenchmark()


