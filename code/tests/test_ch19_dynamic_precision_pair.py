from __future__ import annotations

import pytest
import torch

from ch19.baseline_dynamic_precision import BaselineDynamicPrecisionBenchmark
from ch19.dynamic_precision_benchmark_common import (
    DynamicPrecisionBenchmarkConfig,
    build_model,
    build_prompt,
    decode_dynamic_precision,
    decode_fixed_precision,
)
from ch19.optimized_dynamic_precision import OptimizedDynamicPrecisionBenchmark


def test_dynamic_precision_decode_matches_fixed_precision_on_cpu() -> None:
    cfg = DynamicPrecisionBenchmarkConfig(batch_size=2, prompt_len=8, max_steps=8, vocab_size=64, hidden_dim=64)
    device = torch.device("cpu")
    prompt = build_prompt(cfg, device)
    baseline_model = build_model(cfg, device, dtype=torch.float32)
    optimized_model = build_model(cfg, device, dtype=torch.float32)

    baseline_tokens = decode_fixed_precision(baseline_model, prompt, max_steps=cfg.max_steps, device=device)
    optimized_tokens, stats = decode_dynamic_precision(optimized_model, prompt, max_steps=cfg.max_steps, device=device)

    assert torch.equal(baseline_tokens, optimized_tokens)
    assert stats is not None
    assert stats.total_tokens > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for chapter 19 dynamic-precision benchmark pair")
def test_dynamic_precision_benchmark_pair_matches_on_gpu() -> None:
    cfg = DynamicPrecisionBenchmarkConfig(batch_size=2, prompt_len=8, max_steps=8, vocab_size=64, hidden_dim=64)
    baseline = BaselineDynamicPrecisionBenchmark(cfg=cfg)
    optimized = OptimizedDynamicPrecisionBenchmark(cfg=cfg)

    baseline.setup()
    optimized.setup()
    try:
        baseline.benchmark_fn()
        optimized.benchmark_fn()
        assert baseline.output is not None
        assert optimized.output is not None
        assert torch.equal(baseline.output.cpu(), optimized.output.cpu())
    finally:
        baseline.teardown()
        optimized.teardown()
