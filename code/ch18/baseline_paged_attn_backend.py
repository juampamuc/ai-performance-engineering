"""Dense SDPA math-backend baseline for the split paged-attention benchmarks."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch18.paged_attn_split_common import DensePagedAttnBase


class BaselinePagedAttnBackendBenchmark(DensePagedAttnBase):
    backend = "math"
    nvtx_label = "paged_attn_backend_baseline"


def get_benchmark() -> BaseBenchmark:
    return BaselinePagedAttnBackendBenchmark()
