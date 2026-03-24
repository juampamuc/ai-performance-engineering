from __future__ import annotations

import pytest
import torch

from ch04.baseline_nixl_tier_handoff import get_benchmark as get_baseline_benchmark
from ch04.optimized_nixl_tier_handoff import get_benchmark as get_optimized_benchmark
from labs.nccl_nixl_nvshmem.comm_stack_common import TierHandoffBenchmark


def test_ch04_nixl_tier_handoff_wrappers_surface_real_chapter_pair() -> None:
    baseline = get_baseline_benchmark()
    optimized = get_optimized_benchmark()

    assert isinstance(baseline, TierHandoffBenchmark)
    assert isinstance(optimized, TierHandoffBenchmark)
    assert baseline.optimized is False
    assert optimized.optimized is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ch04 nixl tier handoff benchmark")
def test_ch04_nixl_tier_handoff_pair_matches_selected_blocks() -> None:
    baseline = get_baseline_benchmark()
    optimized = get_optimized_benchmark()

    baseline.setup()
    optimized.setup()
    try:
        baseline.benchmark_fn()
        optimized.benchmark_fn()
        assert baseline.output is not None
        assert optimized.output is not None
        assert torch.equal(baseline.output, optimized.output)
    finally:
        baseline.teardown()
        optimized.teardown()
