from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from core.benchmark import run_manifest as run_manifest_module
from core.benchmark.models import BenchmarkResult, TimingStats
from core.benchmark.run_manifest import RunManifest, get_git_info
from core.harness import benchmark_harness as benchmark_harness_module
from core.harness.benchmark_harness import (
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    _maybe_write_subprocess_stderr,
)
import core.harness.run_benchmarks as run_benchmarks


def _minimal_result(*, gpu_metrics=None) -> BenchmarkResult:
    return BenchmarkResult(
        timing=TimingStats(
            mean_ms=1.0,
            median_ms=1.0,
            std_ms=0.0,
            min_ms=1.0,
            max_ms=1.0,
            iterations=1,
            warmup_iterations=0,
        ),
        benchmark_name="dummy",
        device="cpu",
        gpu_metrics=gpu_metrics,
    )


def test_benchmark_with_manifest_records_collection_warning_when_patch_fails(monkeypatch) -> None:
    class _BadHardware:
        def __setattr__(self, name, value):
            raise RuntimeError("telemetry patch exploded")

    real_create = RunManifest.create.__func__

    def _create_with_bad_hardware(cls, config=None, start_time=None):
        manifest = real_create(cls, config=config, start_time=start_time)
        manifest.hardware = _BadHardware()
        return manifest

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(
            iterations=1,
            warmup=5,
            device=torch.device("cpu"),
            use_subprocess=False,
        ),
    )
    monkeypatch.setattr(
        run_manifest_module,
        "get_git_info",
        lambda: {"commit": "deadbeef", "branch": "main", "dirty": False},
    )
    monkeypatch.setattr(RunManifest, "create", classmethod(_create_with_bad_hardware))
    monkeypatch.setattr(
        harness,
        "benchmark",
        lambda benchmark: _minimal_result(gpu_metrics={"graphics_clock_mhz": 1234}),
    )

    run = harness.benchmark_with_manifest(object(), run_id="manifest_test")

    assert run.timestamp is not None
    assert run.manifest is not None
    assert any(
        "Failed to patch locked GPU telemetry into manifest" in warning
        for warning in run.manifest.collection_warnings
    )


def test_benchmark_with_manifest_leaves_collection_warnings_empty_when_patch_succeeds(monkeypatch) -> None:
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(
            iterations=1,
            warmup=5,
            device=torch.device("cpu"),
            use_subprocess=False,
        ),
    )
    monkeypatch.setattr(
        run_manifest_module,
        "get_git_info",
        lambda: {"commit": "deadbeef", "branch": "main", "dirty": False},
    )
    monkeypatch.setattr(
        harness,
        "benchmark",
        lambda benchmark: _minimal_result(gpu_metrics={"graphics_clock_mhz": 1234, "memory_clock_mhz": 5678}),
    )

    run = harness.benchmark_with_manifest(object(), run_id="manifest_ok")

    assert run.timestamp is not None
    assert run.manifest is not None
    assert run.manifest.collection_warnings == []


def test_get_git_info_marks_repo_dirty_for_untracked_and_staged_changes(monkeypatch) -> None:
    responses = iter(
        [
            SimpleNamespace(returncode=0, stdout="deadbeef\n"),
            SimpleNamespace(returncode=0, stdout="main\n"),
            SimpleNamespace(returncode=0, stdout="?? new_file.py\nA  staged_file.py\n"),
        ]
    )

    monkeypatch.setattr(
        run_manifest_module.subprocess,
        "run",
        lambda *args, **kwargs: next(responses),
    )

    git_info = get_git_info()

    assert git_info == {"commit": "deadbeef", "branch": "main", "dirty": True}


def test_run_manifest_create_records_warning_when_git_metadata_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        run_manifest_module,
        "get_git_info",
        lambda: {"commit": None, "branch": None, "dirty": False},
    )

    manifest = RunManifest.create(config={"validity_profile": "strict"})

    assert any(
        "Git metadata unavailable for fields: commit, branch" in warning
        for warning in manifest.collection_warnings
    )


def test_ensure_cuda_executables_built_warns_when_cleanup_fails(tmp_path, monkeypatch) -> None:
    chapter_dir = tmp_path / "ch99"
    chapter_dir.mkdir()
    (chapter_dir / "Makefile").write_text("all:\n\t@echo ok\n", encoding="utf-8")
    (chapter_dir / "build").mkdir()

    warnings: list[str] = []
    monkeypatch.setattr(run_benchmarks, "detect_supported_arch", None)
    monkeypatch.setattr(
        run_benchmarks,
        "logger",
        SimpleNamespace(
            info=lambda *args, **kwargs: None,
            warning=lambda msg, *args, **kwargs: warnings.append(msg % args if args else msg),
        ),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "core.utils.build_utils",
        SimpleNamespace(
            ensure_clean_build_directory=lambda path: (_ for _ in ()).throw(RuntimeError("stale lock"))
        ),
    )
    monkeypatch.setattr(
        run_benchmarks.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr="", stdout=""),
    )

    success, reason = run_benchmarks.ensure_cuda_executables_built(chapter_dir)

    assert success is True
    assert reason is None
    assert any("Failed to clean build directory" in warning for warning in warnings)


def test_subprocess_stderr_capture_warns_when_persist_fails(monkeypatch, tmp_path) -> None:
    warnings: list[str] = []

    monkeypatch.setattr(
        benchmark_harness_module,
        "logger",
        SimpleNamespace(
            warning=lambda msg, *args, **kwargs: warnings.append(msg % args if args else msg),
        ),
    )
    monkeypatch.setattr(
        Path,
        "write_text",
        lambda self, data, *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    _maybe_write_subprocess_stderr(
        "stderr payload",
        "benchmark_name",
        BenchmarkConfig(subprocess_stderr_dir=str(tmp_path)),
    )

    assert any("Failed to persist subprocess stderr" in warning for warning in warnings)
