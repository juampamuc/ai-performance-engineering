"""FlexAttention block-mask path driven by a real block table."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch18.paged_attn_split_common import LayoutPagedAttnBase


class OptimizedPagedAttnLayoutBenchmark(LayoutPagedAttnBase):
    uses_paged_kv = True
    nvtx_label = "paged_attn_layout_optimized"


def get_benchmark() -> BaseBenchmark:
    return OptimizedPagedAttnLayoutBenchmark()
