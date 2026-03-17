"""Explicit causal-mode baseline target for the educational FlashAttention-4 lab."""

from __future__ import annotations

from labs.flashattention4.baseline_flashattention4 import BaselineFlashAttention4Benchmark
from labs.flashattention4.target_variants import FlashAttention4FixedConfigMixin


class BaselineFlashAttention4CausalBenchmark(
    FlashAttention4FixedConfigMixin, BaselineFlashAttention4Benchmark
):
    fixed_mode = "causal"


def get_benchmark() -> BaselineFlashAttention4Benchmark:
    return BaselineFlashAttention4CausalBenchmark()


