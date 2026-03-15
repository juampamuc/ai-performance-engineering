from pathlib import Path

from core.harness import run_benchmarks


class _DummyExpectationsStore:
    def __init__(self, chapter_dir: Path, *_args, **_kwargs) -> None:
        self.path = chapter_dir / "expectations_test.json"

    def save(self) -> None:
        return None


def test_test_chapter_impl_uses_cuda_wrapper_detector_without_nameerror(tmp_path, monkeypatch):
    chapter_dir = tmp_path / "ch03"
    chapter_dir.mkdir()
    baseline_path = chapter_dir / "baseline_numa_unaware.py"
    baseline_path.write_text(
        "from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark\n"
        "class DemoCudaWrapper(CudaBinaryBenchmark):\n"
        "    cuda_binary_path = 'demo'\n",
        encoding="utf-8",
    )

    detector_calls: list[Path] = []
    original_detector = run_benchmarks.is_cuda_binary_benchmark_file

    def _tracked_detector(path: Path) -> bool:
        detector_calls.append(path)
        return original_detector(path)

    monkeypatch.setattr(run_benchmarks, "is_cuda_binary_benchmark_file", _tracked_detector)
    monkeypatch.setattr(run_benchmarks, "dump_environment_and_capabilities", lambda: None)
    monkeypatch.setattr(run_benchmarks, "detect_expectation_key", lambda: "test")
    monkeypatch.setattr(run_benchmarks, "detect_execution_environment", lambda: {"kind": "test"})
    monkeypatch.setattr(run_benchmarks, "get_git_info", lambda: {"commit": "deadbeef"})
    monkeypatch.setattr(run_benchmarks, "clean_build_directories", lambda _chapter_dir: None)
    monkeypatch.setattr(run_benchmarks, "reset_cuda_state", lambda: None)
    monkeypatch.setattr(run_benchmarks, "reset_gpu_state", lambda: None)
    monkeypatch.setattr(run_benchmarks, "emit_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_benchmarks, "start_progress_watchdog", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(run_benchmarks, "ExpectationsStore", _DummyExpectationsStore)
    monkeypatch.setattr(run_benchmarks.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(run_benchmarks.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        run_benchmarks,
        "_discover_chapter_benchmark_pairs",
        lambda *args, **kwargs: ([(baseline_path, [], "numa_unaware")], [], None, 0, 0),
    )

    result = run_benchmarks._test_chapter_impl(chapter_dir, enable_profiling=False)

    assert detector_calls == [baseline_path]
    assert result["status"] == "completed"
    assert result["summary"]["informational"] == 1
    assert result["benchmarks"] == []


def test_benchmark_cuda_executable_treats_skip_exit_code_as_skip(tmp_path):
    executable = tmp_path / "skip_cuda_binary.sh"
    executable.write_text(
        "#!/bin/sh\n"
        "echo 'SKIPPED: cuBLASLt NVFP4 algorithm unavailable on this driver/toolchain.' >&2\n"
        "exit 3\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)

    result = run_benchmarks.benchmark_cuda_executable(executable, iterations=1, warmup=0, timeout=5)

    assert result is not None
    assert result.skip_reason == "SKIPPED: cuBLASLt NVFP4 algorithm unavailable on this driver/toolchain."


def test_is_distributed_benchmark_ignores_local_gpu_reduction_named_distributed(tmp_path):
    benchmark_path = tmp_path / "optimized_distributed.py"
    benchmark_path.write_text(
        "import torch\n"
        "\n"
        "class DemoBenchmark:\n"
        "    def benchmark_fn(self):\n"
        "        data = torch.randn(1024, device='cuda')\n"
        "        return data.sum()\n",
        encoding="utf-8",
    )

    assert run_benchmarks.is_distributed_benchmark(benchmark_path) is False


def test_append_profile_warning_persists_message_to_stderr_log(tmp_path, monkeypatch):
    log_path = tmp_path / "profile.stderr.log"
    logger_messages: list[str] = []

    monkeypatch.setattr(run_benchmarks, "LOGGER_AVAILABLE", True)
    monkeypatch.setattr(
        run_benchmarks.logger,
        "warning",
        lambda message, *args, **_kwargs: logger_messages.append(message % args if args else message),
    )

    run_benchmarks._append_profile_warning(log_path, "profiler cleanup detail")

    assert "profiler cleanup detail" in log_path.read_text(encoding="utf-8")
    assert any("profiler cleanup detail" in message for message in logger_messages)
