"""optimized_continuous_batching.py - Dynamic continuous batching (single GPU)."""

from __future__ import annotations

from core.utils.continuous_batching import ContinuousBatchingBase
from core.benchmark.verification_mixin import VerificationPayloadMixin


class OptimizedContinuousBatchingBenchmark(VerificationPayloadMixin, ContinuousBatchingBase):
    """Optimized: continuous batching with dynamic batch membership."""

    def __init__(self) -> None:
        super().__init__(
            dynamic=True,
            multi_gpu=False,
            label="optimized_continuous_batching",
        )


def get_benchmark() -> OptimizedContinuousBatchingBenchmark:
    return OptimizedContinuousBatchingBenchmark()


