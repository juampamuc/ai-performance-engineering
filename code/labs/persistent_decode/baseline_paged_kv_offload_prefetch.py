"""Baseline paged KV-cache prefetch benchmark (pageable sync copy, no prefetch).

- Uses pageable staging buffers and a synchronous copy path.
- Does not prefetch the next page, so H2D copies block the iteration.
- Uses a pageable host cache (memmap disabled) to isolate overlap effects.
"""

from __future__ import annotations

import torch

from core.benchmark.wrapper_utils import attach_benchmark_metadata
from labs.persistent_decode.paged_kv_offload_common import PagedKVConfig, PagedKVOffloadBenchmark


def get_benchmark() -> PagedKVOffloadBenchmark:
    cfg = PagedKVConfig(
        batch_size=4,
        num_heads=16,
        head_dim=128,
        max_seq_len=65536,
        page_tokens=8192,
        decode_tokens=1,
        repeat_pages=128,
        use_pinned_stage=False,
        use_async_stream=False,
        use_memmap=False,
        prefer_fp8=False,
        require_fused_fp8=False,
        fallback_dtype=torch.float16,
        prefetch_next_page=False,
        use_direct_h2d=False,
    )
    return attach_benchmark_metadata(
        PagedKVOffloadBenchmark(cfg, label="paged_kv_offload_prefetch_baseline"),
        __file__,
    )

