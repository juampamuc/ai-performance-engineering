"""Tests for the Blackwell grouped GEMM optimization lab."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from core.discovery import discover_benchmarks
from core.harness.benchmark_harness import BaseBenchmark
from labs.blackwell_gemm_optimizations.blackwell_grouped_gemm_autotune import (
    experimental_variant_names,
    public_variant_names,
)
from labs.blackwell_gemm_optimizations.blackwell_grouped_gemm_common import (
    BlackwellGroupedGemmWorkload,
    build_state,
    run_variant,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LAB_DIR = REPO_ROOT / "labs" / "blackwell_gemm_optimizations"


def _load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_blackwell_grouped_gemm_discovery_finds_primary_and_alias_targets() -> None:
    pairs = discover_benchmarks(LAB_DIR)
    discovered = {example_name for _, _, example_name in pairs}
    assert discovered == {
        "blackwell_grouped_gemm",
        "blackwell_grouped_gemm_large_tiles",
        "blackwell_grouped_gemm_full_stack",
        "blackwell_grouped_gemm_persistent",
    }

    primary = next(pair for pair in pairs if pair[2] == "blackwell_grouped_gemm")
    _, optimized_paths, _ = primary
    assert {path.name for path in optimized_paths} == {
        "optimized_blackwell_grouped_gemm_large_tiles.py",
        "optimized_blackwell_grouped_gemm_full_stack.py",
        "optimized_blackwell_grouped_gemm_persistent.py",
    }


@pytest.mark.parametrize(
    "relative_path",
    [
        "labs/blackwell_gemm_optimizations/baseline_blackwell_grouped_gemm.py",
        "labs/blackwell_gemm_optimizations/optimized_blackwell_grouped_gemm_large_tiles.py",
        "labs/blackwell_gemm_optimizations/optimized_blackwell_grouped_gemm_full_stack.py",
        "labs/blackwell_gemm_optimizations/optimized_blackwell_grouped_gemm_persistent.py",
    ],
)
def test_blackwell_grouped_gemm_wrappers_expose_get_benchmark(relative_path: str) -> None:
    module_path = REPO_ROOT / relative_path
    module = _load_module(module_path)
    bench = module.get_benchmark()
    assert isinstance(bench, BaseBenchmark)
    assert getattr(bench, "_module_file_override", None) == str(module_path)
    assert getattr(bench, "_factory_name_override", None) == "get_benchmark"


def test_blackwell_grouped_gemm_schedule_registry_is_complete() -> None:
    assert set(public_variant_names()) == {
        "baseline",
        "large_tiles",
        "full_stack",
        "persistent",
    }
    assert set(experimental_variant_names()) == {
        "fast_math_control",
        "latency10",
        "two_cta",
        "tile_n256",
    }


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for Blackwell grouped GEMM validation",
)
def test_blackwell_grouped_gemm_public_variants_match_reference() -> None:
    workload = BlackwellGroupedGemmWorkload(
        num_tokens=128,
        num_experts=4,
        hidden_dim=256,
        expert_ffn_dim=512,
        dtype=torch.float16,
    )
    state = build_state(workload, torch.device("cuda"))

    for variant in public_variant_names():
        packed = torch.empty(
            workload.num_experts * state.max_count,
            workload.hidden_dim,
            device="cuda",
            dtype=workload.dtype,
        )
        out = torch.empty(
            workload.num_experts,
            state.max_count,
            workload.expert_ffn_dim,
            device="cuda",
            dtype=workload.dtype,
        )
        result = run_variant(
            state,
            variant=variant,
            packed_tokens_flat=packed,
            output_buffer=out,
        )
        torch.testing.assert_close(
            result.output.float(),
            state.reference_output.float(),
            atol=3.5e-1,
            rtol=5e-2,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for Blackwell grouped GEMM validation",
)
def test_blackwell_grouped_gemm_experimental_variants_execute() -> None:
    workload = BlackwellGroupedGemmWorkload(
        num_tokens=128,
        num_experts=4,
        hidden_dim=256,
        expert_ffn_dim=512,
        dtype=torch.float16,
    )
    state = build_state(workload, torch.device("cuda"))

    for experimental in experimental_variant_names():
        packed = torch.empty(
            workload.num_experts * state.max_count,
            workload.hidden_dim,
            device="cuda",
            dtype=workload.dtype,
        )
        out = torch.empty(
            workload.num_experts,
            state.max_count,
            workload.expert_ffn_dim,
            device="cuda",
            dtype=workload.dtype,
        )
        result = run_variant(
            state,
            variant="full_stack",
            experimental=experimental,
            packed_tokens_flat=packed,
            output_buffer=out,
        )
        torch.testing.assert_close(
            result.output.float(),
            state.reference_output.float(),
            atol=3.5e-1,
            rtol=5e-2,
        )
