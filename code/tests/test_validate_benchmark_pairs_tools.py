from __future__ import annotations

import os
from pathlib import Path
import signal
import time

import torch

from core.benchmark.verification import InputSignature
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark
from core.scripts.ci.check_verification_compliance import check_file_compliance
from core.scripts.validate_benchmark_pairs import (
    discover_benchmark_pairs,
    get_input_signature_safe,
    validate_all_pairs,
    validate_pair,
)


class _DummyPayloadBenchmark(VerificationPayloadMixin, BaseBenchmark):
    allow_cpu = True

    def __init__(self) -> None:
        super().__init__()
        self.x: torch.Tensor | None = None
        self.y: torch.Tensor | None = None

    def setup(self) -> None:
        torch.manual_seed(42)
        self.x = torch.randn(2, 3, device=self.device)
        self.y = None

    def benchmark_fn(self) -> None:
        if self.x is None:
            raise RuntimeError("setup() must run before benchmark_fn()")
        self.y = self.x * 2

    def capture_verification_payload(self) -> None:
        if self.x is None or self.y is None:
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"x": self.x},
            output=self.y,
            batch_size=int(self.x.shape[0]),
            parameter_count=0,
            output_tolerance=(0.0, 0.0),
        )


def test_get_input_signature_safe_executes_payload_path() -> None:
    bench = _DummyPayloadBenchmark()
    sig, err = get_input_signature_safe(bench)
    assert err is None
    assert isinstance(sig, InputSignature)


def test_discover_benchmark_pairs_matches_optimized_variants(tmp_path: Path) -> None:
    ch_dir = tmp_path / "ch01"
    ch_dir.mkdir(parents=True)
    (ch_dir / "baseline_foo.py").write_text("# baseline\n")
    (ch_dir / "optimized_foo.py").write_text("# optimized\n")
    (ch_dir / "optimized_foo_bar.py").write_text("# optimized variant\n")

    pairs = discover_benchmark_pairs(tmp_path, chapter="ch01")
    assert "ch01:foo" in pairs
    assert "ch01:foo_bar" in pairs
    assert pairs["ch01:foo_bar"]["baseline"] == ch_dir / "baseline_foo.py"
    assert pairs["ch01:foo_bar"]["optimized"] == ch_dir / "optimized_foo_bar.py"


def test_ci_compliance_checker_accepts_payload_mixin_inheritance(tmp_path: Path) -> None:
    file_path = tmp_path / "baseline_dummy.py"
    file_path.write_text(
        "from core.benchmark.verification_mixin import VerificationPayloadMixin\n"
        "from core.harness.benchmark_harness import BaseBenchmark\n"
        "class Dummy(VerificationPayloadMixin, BaseBenchmark):\n"
        "    def benchmark_fn(self):\n"
        "        return None\n"
    )
    issues = check_file_compliance(file_path)
    assert not any("get_input_signature" in issue.message and issue.severity == "error" for issue in issues)


def test_ci_compliance_checker_accepts_local_benchmark_base_inheritance(tmp_path: Path) -> None:
    file_path = tmp_path / "baseline_dummy.py"
    file_path.write_text(
        "from core.benchmark.verification_mixin import VerificationPayloadMixin\n"
        "from core.harness.benchmark_harness import BaseBenchmark\n"
        "class ParentBenchmark(VerificationPayloadMixin, BaseBenchmark):\n"
        "    def benchmark_fn(self):\n"
        "        return None\n"
        "class ChildBenchmark(ParentBenchmark):\n"
        "    def benchmark_fn(self):\n"
        "        return None\n"
    )
    issues = check_file_compliance(file_path)
    assert not any("get_input_signature" in issue.message for issue in issues)


def test_ci_compliance_checker_accepts_imported_benchmark_base_inheritance(tmp_path: Path) -> None:
    (tmp_path / "base_module.py").write_text(
        "from core.benchmark.verification_mixin import VerificationPayloadMixin\n"
        "from core.harness.benchmark_harness import BaseBenchmark\n"
        "class ImportedParentBenchmark(VerificationPayloadMixin, BaseBenchmark):\n"
        "    allow_cpu = True\n"
        "    def benchmark_fn(self):\n"
        "        return None\n"
    )
    file_path = tmp_path / "baseline_dummy.py"
    file_path.write_text(
        "from base_module import ImportedParentBenchmark\n"
        "class ChildBenchmark(ImportedParentBenchmark):\n"
        "    def benchmark_fn(self):\n"
        "        return None\n"
    )
    issues = check_file_compliance(file_path)
    assert not any("get_input_signature" in issue.message for issue in issues)


def test_ch04_torchrun_multigpu_pairs_skip_cleanly_on_single_gpu_hosts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cases = [
        ("pipeline_parallel_multigpu_1f1b", "baseline_pipeline_parallel_multigpu.py", "optimized_pipeline_parallel_multigpu_1f1b.py"),
        ("tensor_parallel_allgather_multigpu", "baseline_tensor_parallel_allgather_multigpu.py", "optimized_tensor_parallel_allgather_multigpu.py"),
        ("tensor_parallel_multigpu", "baseline_tensor_parallel_multigpu.py", "optimized_tensor_parallel_multigpu.py"),
        ("torchcomms_multigpu", "baseline_torchcomms_multigpu.py", "optimized_torchcomms_multigpu.py"),
    ]

    for example_name, baseline_name, optimized_name in cases:
        result = validate_pair(
            "ch04",
            example_name,
            repo_root / "ch04" / baseline_name,
            repo_root / "ch04" / optimized_name,
        )
        if torch.cuda.device_count() < 2:
            assert result.skipped is True
            assert result.error is not None and "SKIPPED" in result.error
        else:
            assert result.error is None
            assert result.valid is True


def test_ch04_no_overlap_pair_validates_on_single_gpu_hosts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = validate_pair(
        "ch04",
        "no_overlap",
        repo_root / "ch04" / "baseline_no_overlap.py",
        repo_root / "ch04" / "optimized_no_overlap.py",
    )

    if torch.cuda.is_available():
        assert result.skipped is False
        assert result.error is None
        assert result.valid is True
    else:
        assert result.skipped is True
        assert result.error is not None and "SKIPPED" in result.error


def test_validate_all_pairs_timeout_kills_isolated_pair_descendants(tmp_path: Path) -> None:
    ch_dir = tmp_path / "ch01"
    ch_dir.mkdir(parents=True)
    pid_file = tmp_path / "spawned_child.pid"
    benchmark_source = f"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from core.harness.benchmark_harness import BaseBenchmark


class HangingPairBenchmark(BaseBenchmark):
    allow_cpu = True

    def __init__(self) -> None:
        super().__init__()
        self._ready = False

    def setup(self) -> None:
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        Path(r"{pid_file}").write_text(str(child.pid), encoding="utf-8")

    def benchmark_fn(self) -> None:
        time.sleep(60)
        self._ready = True

    def capture_verification_payload(self) -> None:
        if not self._ready:
            raise RuntimeError("payload not ready")

    def get_input_signature(self):
        raise RuntimeError("force execution path")
"""
    (ch_dir / "baseline_hanging.py").write_text(benchmark_source, encoding="utf-8")
    (ch_dir / "optimized_hanging.py").write_text(benchmark_source, encoding="utf-8")

    report = validate_all_pairs(tmp_path, chapter="ch01", pair_timeout_seconds=3)
    assert report.total_pairs == 1
    assert len(report.results) == 1
    result = report.results[0]
    assert result.valid is False
    assert result.error == "Validation timeout after 3 seconds"

    deadline = time.time() + 2.0
    while not pid_file.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert pid_file.exists()
    child_pid = int(pid_file.read_text(encoding="utf-8").strip())
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        os.kill(child_pid, signal.SIGKILL)
        raise AssertionError(f"isolated pair child {child_pid} survived timeout cleanup")
