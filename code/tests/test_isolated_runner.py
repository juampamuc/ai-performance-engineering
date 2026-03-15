from __future__ import annotations

import io
from contextlib import redirect_stderr
import sys
from types import SimpleNamespace

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


def test_reset_cuda_state_emits_warnings_for_cleanup_failures(monkeypatch) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda: None,
            empty_cache=lambda: None,
            graph_pool_trim=lambda: (_ for _ in ()).throw(RuntimeError("graph reset failed")),
            current_device=lambda: 0,
            default_generators=[SimpleNamespace(set_offset=lambda _value: (_ for _ in ()).throw(RuntimeError("rng reset failed")))],
        ),
        _dynamo=SimpleNamespace(reset=lambda: (_ for _ in ()).throw(RuntimeError("dynamo reset failed"))),
        _inductor=SimpleNamespace(
            cudagraph_trees=SimpleNamespace(
                reset_cudagraph_trees=lambda: (_ for _ in ()).throw(RuntimeError("cudagraph reset failed"))
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    stderr = io.StringIO()
    with redirect_stderr(stderr):
        isolated_runner.reset_cuda_state()

    output = stderr.getvalue()
    assert "Failed to reset CUDA graph pool" in output
    assert "graph reset failed" in output
    assert "Failed to reset CUDA RNG state" in output
    assert "Failed to reset torch._dynamo state" in output
    assert "Failed to reset torch._inductor cudagraph trees" in output
