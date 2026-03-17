"""Chapter 4 optimized continuous batching benchmark (multi-GPU)."""

from __future__ import annotations

from core.utils.continuous_batching import ContinuousBatchingBase
from core.benchmark.verification_mixin import VerificationPayloadMixin


class OptimizedContinuousBatchingMultiGPUBenchmark(VerificationPayloadMixin, ContinuousBatchingBase):
    """Optimized: dynamic batching across all visible GPUs."""

    multi_gpu_required = True

    def __init__(self) -> None:
        super().__init__(
            dynamic=True,
            multi_gpu=True,
            label="optimized_continuous_batching_multigpu",
        )


def get_benchmark() -> OptimizedContinuousBatchingMultiGPUBenchmark:
    return OptimizedContinuousBatchingMultiGPUBenchmark()


