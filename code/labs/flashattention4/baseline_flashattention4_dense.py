"""Explicit dense-mode baseline target for the educational FlashAttention-4 lab."""

from __future__ import annotations

from labs.flashattention4.baseline_flashattention4 import BaselineFlashAttention4Benchmark
from labs.flashattention4.target_variants import FlashAttention4FixedConfigMixin


class BaselineFlashAttention4DenseBenchmark(
    FlashAttention4FixedConfigMixin, BaselineFlashAttention4Benchmark
):
    fixed_mode = "dense"


def get_benchmark() -> BaselineFlashAttention4Benchmark:
    return BaselineFlashAttention4DenseBenchmark()


