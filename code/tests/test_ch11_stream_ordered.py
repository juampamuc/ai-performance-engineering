from core.harness.benchmark_harness import BenchmarkConfig, ReadOnlyBenchmarkConfigView
from ch11.baseline_stream_ordered import BaselineStreamOrderedBenchmark
from ch11.optimized_stream_ordered import OptimizedStreamOrderedBenchmark


def test_stream_ordered_benchmarks_prefer_kernel_replay_for_ncu() -> None:
    baseline = BaselineStreamOrderedBenchmark()
    optimized = OptimizedStreamOrderedBenchmark()

    assert baseline.preferred_ncu_replay_mode == "kernel"
    assert optimized.preferred_ncu_replay_mode == "kernel"
    assert baseline.preferred_ncu_metric_set == "minimal"
    assert optimized.preferred_ncu_metric_set == "minimal"
    assert baseline.get_config().ncu_replay_mode == "application"
    assert optimized.get_config().ncu_replay_mode == "application"


def test_stream_ordered_reduces_inner_iterations_only_during_profiling() -> None:
    baseline = BaselineStreamOrderedBenchmark()
    optimized = OptimizedStreamOrderedBenchmark()

    assert baseline._active_inner_iterations() == 500
    assert optimized._active_inner_iterations() == 500

    profiling_config = BenchmarkConfig(enable_profiling=True, enable_ncu=True, enable_nvtx=True)
    baseline._config = ReadOnlyBenchmarkConfigView.from_config(profiling_config)
    optimized._config = ReadOnlyBenchmarkConfigView.from_config(profiling_config)

    assert baseline._active_inner_iterations() == 8
    assert optimized._active_inner_iterations() == 8
