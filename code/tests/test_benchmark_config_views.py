from __future__ import annotations

import pytest

from core.harness.benchmark_harness import (
    BenchmarkConfig,
    ExecutionMode,
    LaunchVia,
    ReadOnlyBenchmarkConfigView,
)


def test_benchmark_config_grouped_views_expose_stable_snapshots() -> None:
    config = BenchmarkConfig(
        iterations=12,
        warmup=5,
        measurement_timeout_seconds=321,
        profiling_timeout_seconds=654,
        nsys_timeout_seconds=111,
        ncu_timeout_seconds=222,
        enable_profiling=True,
        enable_nsys=True,
        enable_ncu=True,
        nsys_nvtx_include=["compute_kernel:profile"],
        ncu_metric_set="deep_dive",
        ncu_replay_mode="kernel",
        profile_env_overrides={"AISP_TEST_ENV": "1"},
        validity_profile="portable",
        lock_gpu_clocks=True,
        gpu_sm_clock_mhz=1500,
        gpu_mem_clock_mhz=2400,
        execution_mode="thread",
        launch_via="torchrun",
        env_passthrough=["CUDA_VISIBLE_DEVICES", "NCCL_DEBUG"],
        target_extra_args={"demo": ["--alpha"]},
        profiling_output_dir="profiles",
        subprocess_stderr_dir="stderr",
        timeout_multiplier=1.0,
    )

    assert config.timing.iterations == 12
    assert config.timing.timeout_for("measurement") == 321
    assert config.timing.timeout_for("torch") == 654
    assert config.capture_timing_snapshot() == config.timing.snapshot()

    assert config.profiling.nsys_nvtx_include == ("compute_kernel:profile",)
    assert config.profiling.ncu_metric_set == "deep_dive"
    assert config.profiling.profile_env_overrides["AISP_TEST_ENV"] == "1"

    assert config.validity.validity_profile == "portable"
    assert config.validity.gpu_sm_clock_mhz == 1500
    assert config.validity.gpu_mem_clock_mhz == 2400

    assert config.launch.execution_mode == ExecutionMode.THREAD
    assert config.launch.launch_via == LaunchVia.TORCHRUN
    assert config.launch.env_passthrough == ("CUDA_VISIBLE_DEVICES", "NCCL_DEBUG")
    assert config.launch.target_extra_args["demo"] == ("--alpha",)

    assert config.output.profiling_output_dir == "profiles"
    assert config.output.subprocess_stderr_dir == "stderr"


def test_read_only_benchmark_config_view_preserves_grouped_view_snapshot() -> None:
    config = BenchmarkConfig(
        iterations=9,
        warmup=5,
        profiling_timeout_seconds=432,
        profile_env_overrides={"AISP_TEST_ENV": "1"},
        env_passthrough=["CUDA_VISIBLE_DEVICES"],
        target_extra_args={"demo": ["--alpha"]},
        timeout_multiplier=1.0,
    )
    view = ReadOnlyBenchmarkConfigView.from_config(config)

    config.profile_env_overrides["AISP_TEST_ENV"] = "2"
    config.env_passthrough.append("NCCL_DEBUG")
    config.target_extra_args["demo"].append("--beta")

    assert view.timing.iterations == 9
    assert view.timing.timeout_for("torch") == 432
    assert view.profiling.profile_env_overrides["AISP_TEST_ENV"] == "1"
    assert view.launch.env_passthrough == ("CUDA_VISIBLE_DEVICES",)
    assert view.launch.target_extra_args["demo"] == ("--alpha",)

    with pytest.raises(AttributeError):
        view.iterations = 3
