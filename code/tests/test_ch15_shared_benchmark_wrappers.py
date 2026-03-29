"""Smoke tests for shared Chapter 15 benchmark wrapper factories."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from core.hot_path_checks import (
    check_benchmark_fn_antipatterns,
    check_benchmark_fn_sync_calls,
)
from core.harness.benchmark_harness import BaseBenchmark


def _load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "relative_path",
    [
        "ch15/baseline_guided_decoding.py",
        "ch15/optimized_guided_decoding.py",
        "ch15/baseline_speculative_decoding.py",
        "ch15/optimized_speculative_decoding.py",
        "ch15/baseline_medusa_eagle_speculative.py",
        "ch15/optimized_medusa_eagle_speculative_medusa.py",
        "ch15/optimized_medusa_eagle_speculative_eagle.py",
        "ch15/baseline_moe_comm_exchange.py",
        "ch15/optimized_moe_comm_exchange_overlap.py",
        "ch15/optimized_moe_comm_exchange_hierarchical.py",
    ],
)
def test_shared_ch15_wrappers_attach_metadata(relative_path: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / relative_path
    module = _load_module(module_path)

    bench = module.get_benchmark()

    assert isinstance(bench, BaseBenchmark)
    assert getattr(bench, "_module_file_override", None) == str(module_path)
    assert getattr(bench, "_factory_name_override", None) == "get_benchmark"


@pytest.mark.parametrize(
    "relative_path",
    [
        "ch15/baseline_guided_decoding.py",
        "ch15/optimized_guided_decoding.py",
        "ch15/baseline_speculative_decoding.py",
        "ch15/optimized_speculative_decoding.py",
        "ch15/baseline_medusa_eagle_speculative.py",
        "ch15/optimized_medusa_eagle_speculative_medusa.py",
        "ch15/optimized_medusa_eagle_speculative_eagle.py",
        "ch15/baseline_moe_comm_exchange.py",
        "ch15/optimized_moe_comm_exchange_overlap.py",
        "ch15/optimized_moe_comm_exchange_hierarchical.py",
    ],
)
def test_shared_ch15_benchmark_fns_stay_hot_path_clean(relative_path: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(repo_root / relative_path)
    bench = module.get_benchmark()

    sync_ok, sync_warnings = check_benchmark_fn_sync_calls(bench.benchmark_fn)
    antipattern_ok, antipattern_warnings = check_benchmark_fn_antipatterns(
        bench.benchmark_fn,
        allowed_codes=getattr(bench, "allowed_benchmark_fn_antipatterns", ()),
    )

    assert sync_ok, sync_warnings
    assert antipattern_ok, antipattern_warnings


@pytest.mark.parametrize(
    "relative_path,expected_goal",
    [
        ("ch15/baseline_medusa_eagle_speculative.py", "throughput"),
        ("ch15/optimized_medusa_eagle_speculative_medusa.py", "throughput"),
        ("ch15/optimized_medusa_eagle_speculative_eagle.py", "throughput"),
        ("ch15/baseline_moe_comm_exchange.py", "speed"),
        ("ch15/optimized_moe_comm_exchange_overlap.py", "speed"),
        ("ch15/optimized_moe_comm_exchange_hierarchical.py", "speed"),
    ],
)
def test_shared_ch15_family_goals_match_benchmark_contract(relative_path: str, expected_goal: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(repo_root / relative_path)
    bench = module.get_benchmark()

    assert bench.get_optimization_goal() == expected_goal
