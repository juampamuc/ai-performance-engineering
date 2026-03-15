"""Chapter 1: Performance Basics - Compare baseline vs optimized implementations."""

from pathlib import Path
from typing import Any, Dict

from core.utils.warning_filters import warn_optional_component_unavailable

try:
    import ch01.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError as exc:
    warn_optional_component_unavailable(
        "ch01.arch_config",
        exc,
        impact="Chapter 1 architecture defaults were not applied; compare.py continues with stock runtime settings",
        context="ch01.compare import",
    )

from core.harness.benchmark_harness import BenchmarkConfig
from core.utils.chapter_compare_template import profile_template


def profile() -> Dict[str, Any]:
    """Compare all baseline/optimized pairs using formal harness."""
    chapter_dir = Path(__file__).parent

    return profile_template(
        chapter="ch01",
        chapter_dir=chapter_dir,
        harness_config=BenchmarkConfig(iterations=20, warmup=5),
    )


if __name__ == "__main__":
    result = profile()
    print("\nMetrics:", result)
