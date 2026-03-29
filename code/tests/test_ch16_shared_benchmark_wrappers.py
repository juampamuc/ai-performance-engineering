"""Smoke tests for the shared Chapter 16 PTQ benchmark wrappers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

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
        "ch16/baseline_awq_gptq_smoothquant.py",
        "ch16/optimized_awq_gptq_smoothquant_awq.py",
        "ch16/optimized_awq_gptq_smoothquant_gptq.py",
        "ch16/optimized_awq_gptq_smoothquant_smoothquant.py",
    ],
)
def test_shared_ch16_ptq_wrappers_attach_metadata(relative_path: str) -> None:
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
        "ch16/baseline_awq_gptq_smoothquant.py",
        "ch16/optimized_awq_gptq_smoothquant_awq.py",
        "ch16/optimized_awq_gptq_smoothquant_gptq.py",
        "ch16/optimized_awq_gptq_smoothquant_smoothquant.py",
    ],
)
def test_shared_ch16_ptq_family_is_tracked_as_memory_goal(relative_path: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / relative_path
    module = _load_module(module_path)

    bench = module.get_benchmark()

    assert bench.get_optimization_goal() == "memory"
