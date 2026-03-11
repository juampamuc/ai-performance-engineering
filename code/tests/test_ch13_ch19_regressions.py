from __future__ import annotations

import torch

from ch13.optimized_autograd_standard import OptimizedAutogradCompiledBenchmark
from ch19.mxfp8_moe_common import restore_bucketed_reduce


def test_optimized_autograd_standard_uses_wall_clock_with_full_sync() -> None:
    bench = OptimizedAutogradCompiledBenchmark()
    config = bench.get_config()
    assert config.timing_method == "wall_clock"
    assert config.full_device_sync is True


def test_optimized_autograd_standard_declares_capture_stream_when_present() -> None:
    bench = OptimizedAutogradCompiledBenchmark()
    sentinel = object()
    bench.capture_stream = sentinel  # type: ignore[assignment]
    assert bench.get_custom_streams() == [sentinel]


def test_restore_bucketed_reduce_casts_weighted_output_and_reuses_buffer() -> None:
    output = torch.tensor(
        [
            [2.0, 4.0, 6.0],
            [6.0, 8.0, 10.0],
            [1.0, 3.0, 5.0],
        ],
        dtype=torch.bfloat16,
    )
    bucket_token_ids = torch.tensor([0, 0, 1], dtype=torch.int64)
    weights = torch.tensor([0.25, 0.75, 1.0], dtype=torch.float16)
    out = torch.empty((2, 3), dtype=torch.float16)
    weight_out = torch.empty((2,), dtype=torch.float16)

    restored = restore_bucketed_reduce(
        output,
        bucket_token_ids,
        num_tokens=2,
        weights=weights,
        out=out,
        weight_out=weight_out,
    )

    expected = torch.tensor(
        [
            [5.0, 7.0, 9.0],
            [1.0, 3.0, 5.0],
        ],
        dtype=torch.float16,
    )
    assert restored.data_ptr() == out.data_ptr()
    assert torch.equal(weight_out, torch.tensor([1.0, 1.0], dtype=torch.float16))
    assert torch.allclose(restored, expected, atol=1e-3, rtol=0.0)
