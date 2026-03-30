"""Optimized KV offload benchmark using pinned host memory and async copies."""

from __future__ import annotations

from core.benchmark.wrapper_utils import attach_benchmark_metadata
from labs.persistent_decode.nvlink_offload_common import NvlinkOffloadBenchmark, OffloadConfig


def get_benchmark() -> NvlinkOffloadBenchmark:
    cfg = OffloadConfig(
        use_pinned=True,
        non_blocking=True,
        use_copy_stream=True,
        batch_size=4,
        num_layers=4,
        num_heads=16,
        head_dim=64,
        max_seq_len=4096,
        chunk_tokens=4096,
    )
    return attach_benchmark_metadata(
        NvlinkOffloadBenchmark(cfg, label="nvlink_offload_optimized"),
        __file__,
    )

