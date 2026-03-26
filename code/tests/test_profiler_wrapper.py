from __future__ import annotations

from pathlib import Path

from core.harness.benchmark_harness import BenchmarkConfig
from core.profiling.profiler_wrapper import (
    _resolve_wrapper_loop_budget,
    render_ncu_python_profile_wrapper,
    render_nsys_python_profile_wrapper,
    render_torch_python_profile_wrapper,
    temporary_python_profile_wrapper,
)


def test_wrapper_loop_budget_defaults_to_existing_benchmark_counts() -> None:
    config = BenchmarkConfig(iterations=20, warmup=5)
    assert _resolve_wrapper_loop_budget(config) == (5, 10)


def test_wrapper_loop_budget_honors_profiling_specific_overrides() -> None:
    config = BenchmarkConfig(
        iterations=20,
        warmup=5,
        profiling_warmup=0,
        profiling_iterations=1,
    )
    assert _resolve_wrapper_loop_budget(config) == (0, 1)


def test_temporary_python_profile_wrapper_cleans_up_file() -> None:
    wrapper_path: Path | None = None

    with temporary_python_profile_wrapper("print('ok')\n") as created_path:
        wrapper_path = created_path
        assert wrapper_path.exists()
        assert wrapper_path.read_text(encoding="utf-8") == "print('ok')\n"

    assert wrapper_path is not None
    assert not wrapper_path.exists()


def test_render_nsys_wrapper_contains_expected_config() -> None:
    wrapper = render_nsys_python_profile_wrapper(
        benchmark_path=Path("/tmp/example.py"),
        nvtx_includes=["compute_kernel:profile/"],
        target_label="labs/moe_cuda_ptx:moe_layer",
        target_override_argv=["--mode", "fwd_bwd"],
        validity_profile="strict",
        lock_gpu_clocks_flag=True,
        gpu_sm_clock_mhz=1500,
        gpu_mem_clock_mhz=2000,
    )

    assert 'Path(r"/tmp/example.py")' in wrapper
    assert "nsys_nvtx_include=['compute_kernel:profile/']" in wrapper
    assert "validity_profile='strict'" in wrapper
    assert 'gpu_sm_clock_mhz=1500' in wrapper
    assert "_target_label = 'labs/moe_cuda_ptx:moe_layer'" in wrapper
    assert "_target_override_argv = ['--mode', 'fwd_bwd']" in wrapper
    assert "_apply_overrides(list(_target_override_argv))" in wrapper
    assert "target_extra_args={_target_label: list(_target_override_argv)}" in wrapper
    assert 'with nvtx_range("compute_kernel:profile", enable=True):' in wrapper
    assert 'if getattr(benchmark, "profile_require_teardown", False):' in wrapper
    assert "_os._exit(0)" in wrapper
    assert "raise SystemExit(0)" not in wrapper


def test_render_ncu_wrapper_contains_expected_config() -> None:
    wrapper = render_ncu_python_profile_wrapper(
        benchmark_path=Path("/tmp/example.py"),
        configured_nvtx_includes=["capture/"],
        target_label="labs/moe_cuda_ptx:moe_layer",
        target_override_argv=["--mode", "fwd_bwd"],
        profile_type="minimal",
        ncu_metric_set="minimal",
        pm_sampling_interval=1234,
        ncu_replay_mode="kernel",
        validity_profile="strict",
        lock_gpu_clocks_flag=True,
        gpu_sm_clock_mhz=1500,
        gpu_mem_clock_mhz=2000,
        profile_nvtx_label="capture",
    )

    assert "enable_ncu=True" in wrapper
    assert "ncu_metric_set='minimal'" in wrapper
    assert "pm_sampling_interval=1234" in wrapper
    assert "ncu_replay_mode='kernel'" in wrapper
    assert "_target_label = 'labs/moe_cuda_ptx:moe_layer'" in wrapper
    assert "_target_override_argv = ['--mode', 'fwd_bwd']" in wrapper
    assert "_apply_overrides(list(_target_override_argv))" in wrapper
    assert "with nvtx_range('capture', enable=True):" in wrapper
    assert 'if getattr(benchmark, "profile_require_teardown", False):' in wrapper
    assert "_os._exit(0)" in wrapper
    assert "raise SystemExit(0)" not in wrapper


def test_render_torch_wrapper_contains_expected_output_path() -> None:
    wrapper = render_torch_python_profile_wrapper(
        benchmark_path=Path("/tmp/example.py"),
        torch_output=Path("/tmp/trace.json"),
        target_label="labs/moe_cuda_ptx:moe_layer",
        target_override_argv=["--mode", "fwd_bwd"],
        validity_profile="portable",
        lock_gpu_clocks_flag=False,
        gpu_sm_clock_mhz=None,
        gpu_mem_clock_mhz=None,
    )

    assert 'Path(r"/tmp/example.py")' in wrapper
    assert "validity_profile='portable'" in wrapper
    assert "_target_label = 'labs/moe_cuda_ptx:moe_layer'" in wrapper
    assert "_target_override_argv = ['--mode', 'fwd_bwd']" in wrapper
    assert "_apply_overrides(list(_target_override_argv))" in wrapper
    assert 'prof.export_chrome_trace(r"/tmp/trace.json")' in wrapper
