from __future__ import annotations

from core.benchmark.contract import BenchmarkContract
from core.scripts.generate_concept_mapping import get_chapter_name


def test_benchmark_contract_no_longer_advertises_unsupported_equivalence_hook() -> None:
    assert "get_equivalence_fn" not in BenchmarkContract.RECOMMENDED_METHODS


def test_generate_concept_mapping_uses_updated_chapter_names() -> None:
    assert get_chapter_name("ch09") == "Arithmetic Intensity and Kernel Efficiency"
    assert get_chapter_name("ch11") == "Streams and Concurrency"
    assert get_chapter_name("ch20") == "AI-Assisted Performance Optimization"


def test_ch19_performance_targets_include_quantized_cache_speedup() -> None:
    from core.benchmark.performance_targets import _DEFAULT_TARGETS

    m = _DEFAULT_TARGETS["ch19"]["metrics"]
    assert "dynamic_quantized_cache_speedup" in m
    assert m["dynamic_quantized_cache_speedup"]["unit"] == "x"
