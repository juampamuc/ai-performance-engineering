from __future__ import annotations

import io
from contextlib import redirect_stderr

from core.harness import isolated_runner


def test_run_benchmark_emits_warning_when_reap_fails(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        isolated_runner,
        "_reap_descendant_processes",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("reap failed")),
    )

    stderr = io.StringIO()
    with redirect_stderr(stderr):
        result = isolated_runner.run_benchmark(
            {
                "benchmark_module_path": str(tmp_path / "missing_benchmark.py"),
                "benchmark_class_name": "MissingBenchmark",
                "config_dict": {},
                "device": None,
                "initial_state": None,
            }
        )

    assert result["success"] is False
    assert "isolated_runner_warning" in stderr.getvalue()
    assert "reap failed" in stderr.getvalue()
