"""Smoke tests for shared fullstack-cluster benchmark wrapper factories."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest import mock

import pytest
import torch

from core.harness.validity_checks import _list_foreign_cuda_compute_processes
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.utils.chapter_compare_template import load_benchmark

ENTRYPOINT_MODULE = "labs.fullstack_cluster.moe_hybrid_ep_entrypoint"


def _load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for benchmark wrappers")
@pytest.mark.parametrize(
    "relative_path",
    [
        "labs/fullstack_cluster/baseline_moe_hybrid_ep.py",
        "labs/fullstack_cluster/optimized_moe_hybrid_ep.py",
        "labs/fullstack_cluster/baseline_moe_hybrid_ep_multigpu.py",
        "labs/fullstack_cluster/optimized_moe_hybrid_ep_multigpu.py",
    ],
)
def test_fullstack_cluster_wrappers_attach_metadata_and_torchrun_script(relative_path: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / relative_path
    module = _load_module(module_path)

    bench = module.get_benchmark()
    spec = bench.get_torchrun_spec(BenchmarkConfig(launch_via="torchrun", nproc_per_node=1, iterations=1, warmup=5))

    assert isinstance(bench, BaseBenchmark)
    assert getattr(bench, "_module_file_override", None) == str(module_path)
    assert getattr(bench, "_factory_name_override", None) == "get_benchmark"
    assert Path(bench.script_path) == module_path
    assert spec.script_path is None
    assert spec.module_name == ENTRYPOINT_MODULE
    expected_args = ["--skip-preflight"]
    if "optimized_" in module_path.name:
        expected_args = ["--optimized", *expected_args]
    assert spec.script_args == expected_args
    assert "AISP_MOE_HYBRID_EP_METRICS_PATH" in spec.env
    assert spec.multi_gpu_required is ("multigpu" in module_path.name)
    assert bench._verify_output.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for benchmark wrappers")
@pytest.mark.parametrize(
    "relative_path",
    [
        "labs/fullstack_cluster/baseline_moe_hybrid_ep.py",
        "labs/fullstack_cluster/optimized_moe_hybrid_ep.py",
    ],
)
def test_fullstack_cluster_wrappers_expose_real_profile_torchrun_specs(relative_path: str, tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / relative_path
    module = _load_module(module_path)

    bench = module.get_benchmark()
    config = BenchmarkConfig(launch_via="torchrun", nproc_per_node=1, iterations=1, warmup=5)
    nsys_spec = bench.get_profile_torchrun_spec(profiler="nsys", config=config)
    torch_spec = bench.get_profile_torchrun_spec(
        profiler="torch",
        config=config,
        output_path=tmp_path / "trace.json",
    )

    assert nsys_spec is not None
    assert nsys_spec.script_path is None
    assert nsys_spec.module_name == ENTRYPOINT_MODULE
    expected_args = ["--skip-preflight"]
    if "optimized_" in module_path.name:
        expected_args = ["--optimized", *expected_args]
    assert nsys_spec.script_args == expected_args
    assert torch_spec is not None
    assert torch_spec.script_path is None
    assert torch_spec.module_name == ENTRYPOINT_MODULE
    expected_torch_args = [*expected_args, "--torch-profile-output", str(tmp_path / "trace.json")]
    assert torch_spec.script_args == expected_torch_args


def test_moe_hybrid_ep_entrypoint_routes_optimized_flag_and_remainder() -> None:
    from labs.fullstack_cluster import moe_hybrid_ep_entrypoint as entrypoint

    with mock.patch.object(entrypoint, "run_cli") as run_cli:
        entrypoint.main(["--optimized", "--skip-preflight", "--iters", "3"])

    run_cli.assert_called_once_with(
        optimized=True,
        argv=["--skip-preflight", "--iters", "3"],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for benchmark wrappers")
def test_load_benchmark_does_not_create_parent_cuda_compute_process() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    baseline_path = repo_root / "labs/fullstack_cluster/baseline_moe_hybrid_ep.py"

    bench = load_benchmark(baseline_path)
    assert bench is not None
    assert bench._verify_output.device.type == "cpu"

    loader_code = f"""
from pathlib import Path
import time
from core.utils.chapter_compare_template import load_benchmark

bench = load_benchmark(Path({str(baseline_path)!r}))
assert bench is not None
print("loaded", flush=True)
time.sleep(10)
"""

    process = subprocess.Popen(
        [sys.executable, "-u", "-c", loader_code],
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ready = process.stdout.readline().strip() if process.stdout is not None else ""
        assert ready == "loaded", (
            process.stderr.read() if process.stderr is not None else "loader process failed before reporting ready"
        )
        assert process.poll() is None, (
            process.stderr.read() if process.stderr is not None else "loader process exited unexpectedly"
        )

        child_visible_as_compute = False
        foreign_err = None
        for _ in range(10):
            foreign, foreign_err = _list_foreign_cuda_compute_processes(
                device_index=0,
                current_pid=os.getpid(),
            )
            if foreign_err is not None:
                break
            if any(int(record.get("pid", -1)) == int(process.pid) for record in foreign):
                child_visible_as_compute = True
                break
            time.sleep(0.1)

        if foreign_err is not None:
            pytest.skip(f"NVML unavailable for foreign-process check: {foreign_err}")

        assert not child_visible_as_compute, (
            "load_benchmark() created a CUDA compute process in the parent loader subprocess"
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
