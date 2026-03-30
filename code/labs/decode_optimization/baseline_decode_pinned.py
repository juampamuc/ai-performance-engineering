"""Baseline for pinned-host decode: pageable staging on the transfer-heavy workload."""

from __future__ import annotations

from labs.decode_optimization.decode_common import DecodeBenchmark, DecodeConfig, attach_benchmark_metadata


def get_benchmark() -> DecodeBenchmark:
    """Pageable-host decode variant for the pinned-memory comparison pair."""
    cfg = DecodeConfig(
        batch_size=64,
        prompt_tokens=2048,
        decode_tokens=64,
        prefetch_batches=2,
        host_payload_mb=512,
        hidden_size=256,
        use_pinned_host=False,
        use_copy_stream=False,
        use_compute_stream=False,
        use_cuda_graphs=False,
        use_torch_compile=False,
        label="baseline_decode_pinned",
        iterations=12,
        warmup=15,
    )
    return attach_benchmark_metadata(DecodeBenchmark(cfg), __file__)
