from __future__ import annotations

import argparse
from typing import Optional

import torch

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    _lookup_target_extra_args,
)


class OverrideAwareBenchmark(BaseBenchmark):
    def __init__(self) -> None:
        super().__init__()
        self.mode = "forward"
        self.mode_seen_by_get_config: Optional[str] = None
        self.output: Optional[torch.Tensor] = None

    def apply_target_overrides(self, argv: list[str]) -> None:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--mode", choices=("forward", "fwd_bwd"), default=None)
        args, _ = parser.parse_known_args(argv)
        if args.mode:
            self.mode = args.mode

    def get_config(self) -> BenchmarkConfig:
        self.mode_seen_by_get_config = self.mode
        return BenchmarkConfig(
            iterations=1,
            warmup=0,
            enable_profiling=False,
            enable_memory_tracking=False,
            timeout_multiplier=1.0,
        )

    def setup(self) -> None:
        self.output = None

    def benchmark_fn(self) -> None:
        self.output = torch.tensor([1.0], dtype=torch.float32)

    def validate_result(self) -> Optional[str]:
        return None

    def get_verify_inputs(self):
        return {"mode": torch.tensor([0 if self.mode == "forward" else 1], dtype=torch.int64)}

    def get_verify_output(self):
        if self.output is None:
            raise RuntimeError("Output not produced")
        return self.output

    def get_output_tolerance(self):
        return (0.0, 0.0)

    def get_input_signature(self) -> dict:
        return {"mode": self.mode}


def test_lookup_target_extra_args_matches_slash_and_underscore_chapter_labels() -> None:
    overrides = {
        "labs/moe_cuda_ptx:moe_layer": ["--mode", "fwd_bwd"],
    }

    assert _lookup_target_extra_args(overrides, "labs/moe_cuda_ptx:moe_layer") == ["--mode", "fwd_bwd"]
    assert _lookup_target_extra_args(overrides, "labs_moe_cuda_ptx:moe_layer") == ["--mode", "fwd_bwd"]


def test_harness_applies_target_overrides_before_get_config() -> None:
    benchmark = OverrideAwareBenchmark()
    config = BenchmarkConfig(
        iterations=1,
        warmup=0,
        enable_profiling=False,
        enable_memory_tracking=False,
        allow_foreign_gpu_processes=True,
        enforce_environment_validation=False,
        target_label="labs_moe_cuda_ptx:moe_layer",
        target_extra_args={"labs/moe_cuda_ptx:moe_layer": ["--mode", "fwd_bwd"]},
        timeout_multiplier=1.0,
    )

    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    result = harness.benchmark(benchmark)

    assert result.errors == []
    assert benchmark.mode_seen_by_get_config == "fwd_bwd"
    assert benchmark.mode == "fwd_bwd"
