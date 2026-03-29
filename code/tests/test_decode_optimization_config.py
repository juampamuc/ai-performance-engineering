from core.harness.benchmark_harness import ExecutionMode
from labs.decode_optimization.baseline_decode import get_benchmark as get_baseline_decode
from labs.decode_optimization.baseline_decode_pinned import get_benchmark as get_baseline_decode_pinned
from labs.decode_optimization.decode_common import DecodeBenchmark, DecodeConfig
from labs.decode_optimization.optimized_decode_graph import (
    get_benchmark as get_optimized_decode_graph,
)
from labs.decode_optimization.optimized_decode_pinned import get_benchmark as get_optimized_decode_pinned
from labs.decode_optimization.optimized_decode_ultimate import (
    get_benchmark as get_optimized_decode_ultimate,
)


def test_decode_benchmark_uses_subprocess_execution() -> None:
    bench = DecodeBenchmark(DecodeConfig())

    config = bench.get_config()

    assert config.use_subprocess is True
    assert config.execution_mode == ExecutionMode.SUBPROCESS


def test_decode_variants_inherit_subprocess_execution() -> None:
    for factory in (
        get_baseline_decode,
        get_baseline_decode_pinned,
        get_optimized_decode_pinned,
        get_optimized_decode_graph,
        get_optimized_decode_ultimate,
    ):
        config = factory().get_config()
        assert config.use_subprocess is True
        assert config.execution_mode == ExecutionMode.SUBPROCESS


def test_decode_pinned_pair_uses_transfer_heavy_workload_with_only_pin_state_changed() -> None:
    baseline = get_baseline_decode_pinned()
    optimized = get_optimized_decode_pinned()

    assert baseline.cfg.host_payload_mb == 512
    assert optimized.cfg.host_payload_mb == 512
    assert baseline.cfg.batch_size == optimized.cfg.batch_size == 64
    assert baseline.cfg.prompt_tokens == optimized.cfg.prompt_tokens == 2048
    assert baseline.cfg.prefetch_batches == optimized.cfg.prefetch_batches == 2
    assert baseline.cfg.hidden_size == optimized.cfg.hidden_size == 256
    assert baseline.cfg.use_pinned_host is False
    assert optimized.cfg.use_pinned_host is True
    assert baseline.cfg.use_copy_stream is False
    assert optimized.cfg.use_copy_stream is False
