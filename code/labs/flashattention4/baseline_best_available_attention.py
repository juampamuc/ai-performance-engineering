"""Baseline partner for the best-available attention benchmark target."""

from __future__ import annotations

from labs.flashattention4.baseline_flashattention4 import (
    BaselineFlashAttention4Benchmark,
)


class BaselineBestAvailableAttentionBenchmark(BaselineFlashAttention4Benchmark):
    default_claim_type = "absolute"


def get_benchmark() -> BaselineFlashAttention4Benchmark:
    return BaselineBestAvailableAttentionBenchmark()


