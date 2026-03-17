from __future__ import annotations

from pathlib import Path

from core.verification.review_baseline_optimized_pairs import CodeReviewer, dedupe_issues


def _write(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


_BENCHMARK_PREFIX = """
from __future__ import annotations

import argparse
import torch
from core.harness.benchmark_harness import BaseBenchmark, WorkloadMetadata
"""


def test_compare_pair_does_not_flag_valid_graph_replay_as_work_reduction(tmp_path: Path) -> None:
    reviewer = CodeReviewer()
    baseline = _write(
        tmp_path,
        "baseline_graph.py",
        _BENCHMARK_PREFIX
        + """
class BaselineGraphBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self.batch_size = 16
        self.num_loops = 256
        self._workload = WorkloadMetadata(
            tokens_per_iteration=float(self.batch_size * self.num_loops),
        )

    def setup(self):
        self.tensor = torch.ones(self.batch_size, device=self.device)

    def _step(self):
        self.tensor.add_(1)

    def benchmark_fn(self):
        for _ in range(self.num_loops):
            self._step()


def get_benchmark():
    return BaselineGraphBenchmark()
""",
    )
    optimized = _write(
        tmp_path,
        "optimized_graph.py",
        _BENCHMARK_PREFIX
        + """
class OptimizedGraphBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self.batch_size = 16
        self.num_loops = 256
        self._workload = WorkloadMetadata(
            tokens_per_iteration=float(self.batch_size * self.num_loops),
        )

    def setup(self):
        self.tensor = torch.ones(self.batch_size, device=self.device)
        torch.cuda.synchronize()

    def _replay(self):
        self.tensor.add_(1)

    def benchmark_fn(self):
        self._replay()


def get_benchmark():
    return OptimizedGraphBenchmark()
""",
    )

    issues = reviewer.compare_pair(baseline, [optimized])

    assert not any(issue["type"] == "work_reduction" for issue in issues)


def test_compare_pair_ignores_setup_only_synchronizations(tmp_path: Path) -> None:
    reviewer = CodeReviewer()
    baseline = _write(
        tmp_path,
        "baseline_sync.py",
        _BENCHMARK_PREFIX
        + """
class BaselineSyncBenchmark(BaseBenchmark):
    def benchmark_fn(self):
        return None


def get_benchmark():
    return BaselineSyncBenchmark()
""",
    )
    optimized = _write(
        tmp_path,
        "optimized_sync.py",
        _BENCHMARK_PREFIX
        + """
class OptimizedSyncBenchmark(BaseBenchmark):
    def setup(self):
        torch.cuda.synchronize()

    def benchmark_fn(self):
        return None


def get_benchmark():
    return OptimizedSyncBenchmark()
""",
    )

    issues = reviewer.compare_pair(baseline, [optimized])

    assert not any(issue["type"] == "sync_mismatch" for issue in issues)


def test_check_file_ignores_dummy_rows_and_mock_decode_help_text(tmp_path: Path) -> None:
    reviewer = CodeReviewer()
    source = _write(
        tmp_path,
        "optimized_decode_graphs.py",
        _BENCHMARK_PREFIX
        + """
class DecodeGraphBenchmark(BaseBenchmark):
    def setup(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--hidden", help="Hidden size for the mock decode op.")

    def benchmark_fn(self):
        # without paying full attention cost on dummy rows.
        return None


def get_benchmark():
    return DecodeGraphBenchmark()
""",
    )

    issues = reviewer.check_file(source, "optimized")

    assert not any(issue["type"] == "suspicious_pattern" for issue in issues)


def test_dedupe_issues_collapses_alias_duplicates() -> None:
    duplicated = [
        {"type": "work_reduction", "file": "a vs b", "message": "same", "severity": "medium"},
        {"type": "work_reduction", "file": "a vs b", "message": "same", "severity": "medium"},
    ]

    assert dedupe_issues(duplicated) == [duplicated[0]]


def test_compare_pair_flags_true_workload_mismatch(tmp_path: Path) -> None:
    reviewer = CodeReviewer()
    baseline = _write(
        tmp_path,
        "baseline_workload.py",
        _BENCHMARK_PREFIX
        + """
class BaselineWorkloadBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self.batch_size = 16
        self.num_loops = 256
        self._workload = WorkloadMetadata(
            tokens_per_iteration=float(self.batch_size * self.num_loops),
        )

    def benchmark_fn(self):
        return None


def get_benchmark():
    return BaselineWorkloadBenchmark()
""",
    )
    optimized = _write(
        tmp_path,
        "optimized_workload.py",
        _BENCHMARK_PREFIX
        + """
class OptimizedWorkloadBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self.batch_size = 8
        self.num_loops = 256
        self._workload = WorkloadMetadata(
            tokens_per_iteration=float(self.batch_size * self.num_loops),
        )

    def benchmark_fn(self):
        return None


def get_benchmark():
    return OptimizedWorkloadBenchmark()
""",
    )

    issues = reviewer.compare_pair(baseline, [optimized])

    assert any(issue["type"] == "work_reduction" for issue in issues)


def test_compare_pair_flags_hot_path_helper_synchronize(tmp_path: Path) -> None:
    reviewer = CodeReviewer()
    baseline = _write(
        tmp_path,
        "baseline_helper_sync.py",
        _BENCHMARK_PREFIX
        + """
class BaselineHelperSyncBenchmark(BaseBenchmark):
    def benchmark_fn(self):
        self._step()

    def _step(self):
        return None


def get_benchmark():
    return BaselineHelperSyncBenchmark()
""",
    )
    optimized = _write(
        tmp_path,
        "optimized_helper_sync.py",
        _BENCHMARK_PREFIX
        + """
class OptimizedHelperSyncBenchmark(BaseBenchmark):
    def benchmark_fn(self):
        self._step()

    def _step(self):
        torch.cuda.synchronize()


def get_benchmark():
    return OptimizedHelperSyncBenchmark()
""",
    )

    issues = reviewer.compare_pair(baseline, [optimized])

    assert any(issue["type"] == "sync_mismatch" for issue in issues)
