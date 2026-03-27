import json
import subprocess
import sys
from pathlib import Path

from core.analysis.compose_e2e_historical_failure_ledger import (
    attach_historical_failure_ledger_to_e2e_package,
    compose_e2e_historical_failure_ledger,
    write_e2e_historical_failure_ledger,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _make_e2e_summary(tmp_path: Path) -> Path:
    run_dir = tmp_path / "artifacts" / "e2e_runs" / "demo"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    summary_markdown_path = run_dir / "summary.md"
    payload = {
        "run_id": "demo",
        "run_dir": str(run_dir),
        "run_state": "completed",
        "overall_status": "partial",
        "generated_at": "2026-03-27T00:00:00+00:00",
        "updated_at": "2026-03-27T00:00:00+00:00",
        "success": True,
        "resume_available": False,
        "error": None,
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "summary_markdown_path": str(summary_markdown_path),
        "progress_path": str(run_dir / "progress.json"),
        "checkpoint_path": str(run_dir / "checkpoint.json"),
        "target_inventory_path": str(run_dir / "target_inventory.json"),
        "events_path": str(run_dir / "events.jsonl"),
        "inventory": {},
        "hosts": {},
        "provenance": {},
        "contract": {},
        "crash": None,
        "orchestrator_pid": None,
        "stages": [
            {"name": "tier1", "status": "succeeded", "run_id": "demo__tier1", "attempts": []},
            {
                "name": "full_sweep",
                "status": "partial",
                "run_id": "demo__full_sweep",
                "attempts": [
                    {
                        "run_id": "demo__full_sweep__single",
                        "status": "aborted",
                        "bucket": "single_gpu",
                        "benchmark_summary": {
                            "failed_benchmarks": [
                                {
                                    "target": "labs_alpha:foo",
                                    "status": "failed_error",
                                    "error": "boom",
                                },
                                {
                                    "target": "labs_beta:bar",
                                    "status": "failed_profiler",
                                    "error": "missing trace",
                                },
                                {
                                    "target": "labs_gamma:baz",
                                    "status": "failed_error",
                                    "error": "generic failure",
                                },
                            ]
                        },
                    }
                ],
            },
            {"name": "cluster", "status": "succeeded", "run_id": "demo__cluster", "attempts": []},
            {"name": "fabric", "status": "partial", "run_id": "demo__fabric", "attempts": []},
        ],
    }
    _write_json(summary_path, payload)
    _write_json(manifest_path, {"run_id": "demo"})
    summary_markdown_path.write_text("# Benchmark E2E Sweep\n", encoding="utf-8")
    return summary_path


def _make_results(tmp_path: Path, run_id: str, chapter: str, example: str, status: str, *, best_speedup=None, skip_reason=None, error=None) -> Path:
    path = tmp_path / "artifacts" / "runs" / run_id / "results" / "benchmark_test_results.json"
    _write_json(
        path,
        {
            "results": [
                {
                    "chapter": chapter,
                    "benchmarks": [
                        {
                            "example": example,
                            "status": status,
                            "best_speedup": best_speedup,
                            "skip_reason": skip_reason,
                            "error": error,
                        }
                    ],
                }
            ]
        },
    )
    return path


def test_compose_e2e_historical_failure_ledger_marks_success_skip_and_not_rerun(tmp_path: Path) -> None:
    summary_path = _make_e2e_summary(tmp_path)
    rerun_success = _make_results(tmp_path, "rerun_success", "labs_alpha", "foo", "succeeded", best_speedup=1.7)
    rerun_skip = _make_results(
        tmp_path,
        "rerun_skip",
        "labs_beta",
        "bar",
        "skipped",
        best_speedup=1.0,
        skip_reason="requires >=2 GPUs",
    )

    ledger = compose_e2e_historical_failure_ledger(
        e2e_summary_json=summary_path,
        rerun_results_json=[rerun_success, rerun_skip],
    )

    assert ledger["summary"]["total_historical_failures"] == 3
    assert ledger["summary"]["rechecked_count"] == 2
    assert ledger["summary"]["resolved_success_count"] == 1
    assert ledger["summary"]["resolved_skip_count"] == 1
    assert ledger["summary"]["not_rerun_count"] == 1

    rows = {row["target"]: row for row in ledger["rows"]}
    assert rows["labs_alpha:foo"]["disposition"] == "resolved_success"
    assert rows["labs_alpha:foo"]["best_speedup"] == 1.7
    assert rows["labs_beta:bar"]["disposition"] == "resolved_skip"
    assert rows["labs_beta:bar"]["notes"] == "requires >=2 GPUs"
    assert rows["labs_gamma:baz"]["disposition"] == "not_rerun"


def test_write_and_attach_e2e_historical_failure_ledger_updates_summary_package(tmp_path: Path) -> None:
    summary_path = _make_e2e_summary(tmp_path)
    rerun_success = _make_results(tmp_path, "rerun_success", "labs_alpha", "foo", "succeeded", best_speedup=1.7)

    outputs = write_e2e_historical_failure_ledger(
        e2e_summary_json=summary_path,
        rerun_results_json=[rerun_success],
        output_dir=summary_path.parent,
    )
    attach_historical_failure_ledger_to_e2e_package(
        e2e_summary_json=summary_path,
        ledger_json=outputs["json"],
        ledger_markdown=outputs["markdown"],
    )

    updated_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert updated_summary["historical_failure_ledger"]["json_path"] == str(outputs["json"])
    updated_manifest = json.loads(Path(updated_summary["manifest_path"]).read_text(encoding="utf-8"))
    assert updated_manifest["historical_failure_ledger"]["markdown_path"] == str(outputs["markdown"])

    markdown = Path(updated_summary["summary_markdown_path"]).read_text(encoding="utf-8")
    assert "## Historical Failure Ledger" in markdown
    assert "Resolved success" in markdown


def test_compose_e2e_historical_failure_ledger_raises_clear_error_for_malformed_summary(tmp_path: Path) -> None:
    summary_path = tmp_path / "artifacts" / "e2e_runs" / "demo" / "summary.json"
    _write_json(summary_path, {"stages": {}})
    rerun = _make_results(tmp_path, "rerun_success", "labs_alpha", "foo", "succeeded", best_speedup=1.7)

    try:
        compose_e2e_historical_failure_ledger(
            e2e_summary_json=summary_path,
            rerun_results_json=[rerun],
        )
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected malformed e2e summary to raise ValueError")

    assert "stages" in message


def test_compose_e2e_historical_failure_ledger_cli_reports_clean_error(tmp_path: Path) -> None:
    summary_path = tmp_path / "artifacts" / "e2e_runs" / "demo" / "summary.json"
    _write_json(summary_path, {"stages": {}})
    rerun = _make_results(tmp_path, "rerun_success", "labs_alpha", "foo", "succeeded", best_speedup=1.7)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.analysis.compose_e2e_historical_failure_ledger",
            "--e2e-summary-json",
            str(summary_path),
            "--rerun-results-json",
            str(rerun),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "stages" in proc.stderr
    assert "Traceback" not in proc.stderr
