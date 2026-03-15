from __future__ import annotations

from pathlib import Path
import warnings

from core.utils.warning_filters import (
    suppress_known_cuda_capability_warnings,
    warn_optional_component_unavailable,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_warning_filter_context_does_not_leak_global_filters() -> None:
    before = list(warnings.filters)
    with suppress_known_cuda_capability_warnings():
        inside = list(warnings.filters)
        assert len(inside) >= len(before)
    after = list(warnings.filters)
    assert after == before


def test_no_import_time_global_warning_filters_in_runtime_modules() -> None:
    paths = [
        REPO_ROOT / "core" / "harness" / "arch_config.py",
        REPO_ROOT / "core" / "utils" / "chapter_compare_template.py",
        REPO_ROOT / "core" / "benchmark" / "bench_commands.py",
        REPO_ROOT / "core" / "benchmark" / "benchmark_peak.py",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "warnings.filterwarnings(" not in text, f"global warning filter leaked back into {path}"


def test_warning_filter_summarizes_matched_warnings_with_context() -> None:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with suppress_known_cuda_capability_warnings(context="unit-test torch import"):
            warnings.warn(
                "Found GPU 0 which is of cuda capability 12.1",
                UserWarning,
            )

    messages = [str(item.message) for item in record]
    assert any("unit-test torch import" in message for message in messages)
    assert any("These warnings are no longer silently suppressed" in message for message in messages)
    assert any("Found GPU 0 which is of cuda capability 12.1" in message for message in messages)


def test_warning_filter_reemits_unmatched_warnings() -> None:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with suppress_known_cuda_capability_warnings(context="unit-test unmatched warning"):
            warnings.warn("custom unmatched warning", UserWarning)

    messages = [str(item.message) for item in record]
    assert "custom unmatched warning" in messages


def test_warn_optional_component_unavailable_includes_context_and_impact() -> None:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        warn_optional_component_unavailable(
            "demo.component",
            ImportError("missing dependency"),
            impact="feature X will not run",
            context="unit-test startup",
        )

    messages = [str(item.message) for item in record]
    assert any("demo.component" in message for message in messages)
    assert any("unit-test startup" in message for message in messages)
    assert any("feature X will not run" in message for message in messages)
