"""Optimized: Pinned host memory without stream or compiler changes."""

from __future__ import annotations

from labs.decode_optimization.decode_common import DecodeBenchmark, DecodeConfig, attach_benchmark_metadata


def get_benchmark() -> DecodeBenchmark:
    """Pinned-memory-only decode variant.

    Keep the workload identical to the eager baseline and change only the host
    allocation strategy so this target isolates pageable vs pinned staging.
    """
    cfg = DecodeConfig(
        batch_size=8,
        prompt_tokens=256,
        decode_tokens=64,
        hidden_size=1024,
        use_pinned_host=True,
        use_copy_stream=False,
        use_compute_stream=False,
        use_torch_compile=False,
        label="optimized_decode_pinned",
        iterations=12,
        warmup=15,
    )
    return attach_benchmark_metadata(DecodeBenchmark(cfg), __file__)

