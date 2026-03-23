"""multiple_all_techniques.py - Combined techniques benchmark entrypoint."""

from __future__ import annotations

try:
    import ch20.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass

from ch20.optimized_bf16_mlp import OptimizedBF16MLPBenchmark
from core.harness.benchmark_harness import BaseBenchmark


def get_benchmark() -> BaseBenchmark:
    return OptimizedBF16MLPBenchmark()
