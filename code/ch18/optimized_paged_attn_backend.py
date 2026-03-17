"""Dense SDPA flash-backend comparison target for paged attention."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch18.paged_attn_split_common import DensePagedAttnBase


class OptimizedPagedAttnBackendBenchmark(DensePagedAttnBase):
    backend = "flash"
    nvtx_label = "paged_attn_backend_optimized"


def get_benchmark() -> BaseBenchmark:
    return OptimizedPagedAttnBackendBenchmark()
