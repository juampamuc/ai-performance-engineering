import json

import pytest
from typer.testing import CliRunner

import core.engine as engine_module
from core.perf_core import PerfCore
from core.engine import get_engine, reset_engine
from dashboard.api import server


class _TestPerfCore(PerfCore):
    def __init__(self, *, history_root, data_file=None, bench_root=None):
        super().__init__(data_file=data_file, bench_root=bench_root)
        self._test_history_root = history_root

    def _tier1_history_root(self):
        return self._test_history_root


def test_configure_engine_uses_data_file(sample_benchmark_results_file):
    reset_engine()
    server._configure_engine(sample_benchmark_results_file)
    result = get_engine().benchmark.data()

    assert result["summary"]["total_benchmarks"] == 1
    assert result["benchmarks"][0]["name"] == "example_a"

    reset_engine()


def test_dashboard_cli_has_serve_command():
    runner = CliRunner()
    result = runner.invoke(server.cli, ["serve", "--help"])
    assert result.exit_code == 0
    assert "Start the dashboard API server." in result.output


def test_dashboard_http_benchmark_overview_route_returns_success_envelope(sample_benchmark_results_file):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    reset_engine()
    server._configure_engine(sample_benchmark_results_file)
    client = TestClient(server.fastapi_app)
    response = client.get("/api/benchmark/overview")

    assert response.status_code == 200
    payload = response.json()
    assert payload["tool"] == "benchmark.overview"
    assert payload["status"] == "ok"
    assert payload["success"] is True
    assert payload["result"]["summary"]["total"] == 1
    reset_engine()


def test_dashboard_http_compare_route_returns_error_envelope() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    client = TestClient(server.fastapi_app)
    response = client.get("/api/benchmark/compare")

    assert response.status_code == 200
    payload = response.json()
    assert payload["tool"] == "benchmark.compare"
    assert payload["status"] == "error"
    assert payload["success"] is False
    assert payload["error_type"] == "value_error"
    assert "baseline is required" in payload["error"]


def test_engine_exposes_tier1_history_and_trends(tmp_path, sample_benchmark_results_file):
    history_root = tmp_path / "artifacts" / "history" / "tier1"
    run_dir = history_root / "20260309_010000_tier1_local"
    run_dir.mkdir(parents=True)

    summary_path = run_dir / "summary.json"
    regression_path = run_dir / "regression_summary.json"
    trend_path = run_dir / "trend_snapshot.json"
    index_path = history_root / "index.json"

    summary_path.write_text(
        json.dumps(
            {
                "suite_name": "tier1",
                "suite_version": 1,
                "run_id": "20260309_010000_tier1_local",
                "generated_at": "2026-03-09T01:00:00",
                "targets": [
                    {
                        "key": "flashattention4_alibi",
                        "target": "labs/flashattention4:flashattention4_alibi",
                        "category": "attention",
                        "status": "succeeded",
                        "best_speedup": 12.5,
                        "artifacts": {
                            "baseline_nsys_rep": "artifacts/runs/demo/profiles/flash.nsys-rep",
                        },
                    }
                ],
                "summary": {
                    "target_count": 1,
                    "succeeded": 1,
                    "failed": 0,
                    "skipped": 0,
                    "missing": 0,
                    "avg_speedup": 12.5,
                    "median_speedup": 12.5,
                    "geomean_speedup": 12.5,
                    "representative_speedup": 12.5,
                    "max_speedup": 12.5,
                },
            }
        ),
        encoding="utf-8",
    )
    regression_path.write_text(
        json.dumps(
            {
                "baseline_run_id": "20260308_225441_tier1_manual",
                "current_run_id": "20260309_010000_tier1_local",
                "regressions": [],
                "improvements": [{"key": "flashattention4_alibi", "delta_pct": 4.2}],
                "new_targets": [],
                "missing_targets": [],
            }
        ),
        encoding="utf-8",
    )
    trend_path.write_text(
        json.dumps(
            {
                "suite_name": "tier1",
                "run_count": 1,
                "latest_run_id": "20260309_010000_tier1_local",
                "best_speedup_seen": 12.5,
                "history": [
                    {
                        "run_id": "20260309_010000_tier1_local",
                        "generated_at": "2026-03-09T01:00:00",
                        "avg_speedup": 12.5,
                        "median_speedup": 12.5,
                        "geomean_speedup": 12.5,
                        "representative_speedup": 12.5,
                        "max_speedup": 12.5,
                        "succeeded": 1,
                        "failed": 0,
                        "skipped": 0,
                        "missing": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    index_path.write_text(
        json.dumps(
            {
                "suite_name": "tier1",
                "suite_version": 1,
                "history_root": str(history_root),
                "runs": [
                    {
                        "run_id": "20260309_010000_tier1_local",
                        "summary_path": str(summary_path),
                        "regression_summary_path": str(run_dir / "regression_summary.md"),
                        "regression_json_path": str(regression_path),
                        "trend_snapshot_path": str(trend_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    reset_engine()
    engine_module._handler_instance = _TestPerfCore(
        history_root=history_root,
        data_file=sample_benchmark_results_file,
        bench_root=tmp_path,
    )
    history = get_engine().benchmark.tier1_history()
    trends = get_engine().benchmark.tier1_trends()
    target_history = get_engine().benchmark.tier1_target_history(key="flashattention4_alibi")

    assert history["total_runs"] == 1
    assert history["latest_run_id"] == "20260309_010000_tier1_local"
    assert history["latest"]["run"]["representative_speedup"] == 12.5
    assert history["latest"]["improvements"][0]["key"] == "flashattention4_alibi"
    assert history["latest"]["run"]["regression_summary_json_path"] == str(regression_path)
    assert trends["latest_run_id"] == "20260309_010000_tier1_local"
    assert trends["best_speedup_seen"] == 12.5
    assert target_history["selected_key"] == "flashattention4_alibi"
    assert target_history["run_count"] == 1
    assert target_history["history"][0]["target"] == "labs/flashattention4:flashattention4_alibi"
    assert target_history["history"][0]["best_speedup"] == 12.5
    assert target_history["history"][0]["artifacts"]["baseline_nsys_rep"].endswith(
        "/artifacts/runs/demo/profiles/flash.nsys-rep"
    )

    reset_engine()
