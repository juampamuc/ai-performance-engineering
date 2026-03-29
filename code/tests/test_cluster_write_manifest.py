from __future__ import annotations

import json
from pathlib import Path

from cluster.scripts import write_manifest


def _write_json(path: Path, payload) -> None:  # type: ignore[no-untyped-def]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_manifest_payload_marks_cluster_run_succeeded(tmp_path: Path) -> None:
    cluster_root = tmp_path / "cluster"
    run_id = "2026-03-28_cluster_demo"
    run_dir = cluster_root / "runs" / run_id
    _write_json(
        run_dir / "structured" / f"{run_id}_suite_steps.json",
        [
            {"name": "bootstrap_nodes", "exit_code": 0},
            {"name": "manifest_refresh", "exit_code": 0},
        ],
    )
    _write_json(
        run_dir / "progress" / "run_progress.json",
        {
            "run_id": run_id,
            "current": {
                "timestamp": "2026-03-28T21:15:57.376221+00:00",
                "step": "complete",
                "step_detail": "completed 2/2 suite steps",
                "percent_complete": 100.0,
                "metrics": {
                    "status": "completed",
                    "completed_steps": 2,
                    "total_steps": 2,
                    "suite_steps_path": str(run_dir / "structured" / f"{run_id}_suite_steps.json"),
                },
            },
        },
    )

    payload, _ = write_manifest.build_manifest_payload(
        cluster_root=cluster_root,
        run_id=run_id,
        run_dir=run_dir,
        include_figures=False,
        hosts=["localhost"],
        labels=["localhost"],
    )

    assert payload["manifest_version"] == 2
    assert payload["status"] == "succeeded"
    assert payload["suite_status"] == "succeeded"
    assert payload["success"] is True
    assert payload["issues"] == []
    assert payload["progress"]["percent_complete"] == 100.0
    assert payload["suite_steps"]["failed_step_count"] == 0


def test_build_manifest_payload_marks_fabric_run_partial(tmp_path: Path) -> None:
    cluster_root = tmp_path / "cluster"
    run_id = "2026-03-28_fabric_demo"
    run_dir = cluster_root / "runs" / run_id
    _write_json(
        run_dir / "structured" / f"{run_id}_suite_steps.json",
        [
            {"name": "build_fabric_eval", "exit_code": 0},
            {"name": "manifest_refresh", "exit_code": 0},
        ],
    )
    _write_json(
        run_dir / "progress" / "run_progress.json",
        {
            "run_id": run_id,
            "current": {
                "timestamp": "2026-03-28T21:40:33.028257+00:00",
                "step": "complete",
                "step_detail": "completed 2/2 suite steps",
                "percent_complete": 100.0,
                "metrics": {
                    "status": "completed",
                    "completed_steps": 2,
                    "total_steps": 2,
                    "suite_steps_path": str(run_dir / "structured" / f"{run_id}_suite_steps.json"),
                },
            },
        },
    )
    _write_json(
        run_dir / "structured" / f"{run_id}_fabric_scorecard.json",
        {
            "status": "partial",
            "completeness": "runtime_verified",
            "families": {
                "nvlink": {"completeness": "runtime_verified"},
                "infiniband": {"completeness": "not_present"},
            },
        },
    )

    payload, _ = write_manifest.build_manifest_payload(
        cluster_root=cluster_root,
        run_id=run_id,
        run_dir=run_dir,
        include_figures=False,
        hosts=["localhost"],
        labels=["localhost"],
    )

    assert payload["status"] == "partial"
    assert payload["suite_status"] == "partial"
    assert payload["success"] is True
    assert payload["completeness"] == "runtime_verified"
    assert "fabric completeness is partial for one or more families" in payload["issues"]
    assert payload["fabric"]["scorecard_status"] == "partial"
    assert payload["fabric"]["degraded_families"] == ["infiniband"]
