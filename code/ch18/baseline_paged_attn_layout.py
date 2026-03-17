"""Dense masked-attention baseline for the paged block-table comparison."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch18.paged_attn_split_common import LayoutPagedAttnBase


class BaselinePagedAttnLayoutBenchmark(LayoutPagedAttnBase):
    uses_paged_kv = False
    nvtx_label = "paged_attn_layout_baseline"


def get_benchmark() -> BaseBenchmark:
    return BaselinePagedAttnLayoutBenchmark()
