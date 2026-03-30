"""Baseline KV offload benchmark using pageable host memory."""

from __future__ import annotations

from core.benchmark.wrapper_utils import attach_benchmark_metadata
from labs.persistent_decode.nvlink_offload_common import NvlinkOffloadBenchmark, OffloadConfig


def get_benchmark() -> NvlinkOffloadBenchmark:
    cfg = OffloadConfig(
        use_pinned=False,
        non_blocking=False,
        use_copy_stream=False,
        batch_size=4,
        num_layers=4,
        num_heads=16,
        head_dim=64,
        max_seq_len=4096,
        chunk_tokens=4096,
    )
    return attach_benchmark_metadata(
        NvlinkOffloadBenchmark(cfg, label="nvlink_offload_baseline"),
        __file__,
    )

