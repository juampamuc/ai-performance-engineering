from __future__ import annotations

import contextlib
import getpass
import json
import time
from pathlib import Path
from types import SimpleNamespace

import mcp.mcp_server as mcp_server
from core.api import handlers
from core.benchmark import e2e_sweep


def test_discover_benchmark_e2e_inventory_buckets_targets(monkeypatch) -> None:
    monkeypatch.setattr(
        e2e_sweep,
        "_iter_discovered_targets",
        lambda _root: [
            {"target": "ch01:demo", "chapter": "ch01", "example": "demo", "bench_type": "python", "multi_gpu": False},
            {"target": "ch02:dist", "chapter": "ch02", "example": "dist", "bench_type": "python", "multi_gpu": True},
            {"target": "labs/foo:cuda_demo", "chapter": "labs/foo", "example": "cuda_demo", "bench_type": "cuda", "multi_gpu": False},
        ],
    )

    payload = e2e_sweep.discover_benchmark_e2e_inventory()

    assert payload["counts"] == {"total": 3, "single_gpu": 2, "multi_gpu": 1}
    assert payload["single_gpu"] == ["ch01:demo", "labs/foo:cuda_demo"]
    assert payload["multi_gpu"] == ["ch02:dist"]


def test_run_benchmark_e2e_sweep_derives_stage_ids_and_skips_duplicate_fabric(tmp_path: Path, monkeypatch) -> None:
    summary_path = tmp_path / "tier1" / "summary.json"
    regression_summary_path = tmp_path / "tier1" / "regression_summary.md"
    trend_snapshot_path = tmp_path / "tier1" / "trend.json"
    history_root = tmp_path / "history"
    for path in (summary_path, regression_summary_path, trend_snapshot_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    history_root.mkdir(parents=True, exist_ok=True)

    cluster_run_dir = tmp_path / "cluster_run"
    manifest_path = cluster_run_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")

    class _Tier1SuiteDefinitionLike:
        pass

    cluster_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {"counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0}, "single_gpu": [], "multi_gpu": [], "targets": []},
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(
        e2e_sweep,
        "_invoke_run_tier1_suite",
        lambda **kwargs: {
            "execution": {"run_id": kwargs["run_id"], "total_failed": 0},
            "summary_path": str(summary_path),
            "regression_summary_path": str(regression_summary_path),
            "trend_snapshot_path": str(trend_snapshot_path),
            "history_root": str(history_root),
            "suite_definitions": [_Tier1SuiteDefinitionLike()],
            "total_failed": 0,
            "total_skipped": 0,
        },
    )
    monkeypatch.setattr(e2e_sweep, "_benchmark_queue_lock", lambda *args, **kwargs: contextlib.nullcontext())

    def _fake_cluster_eval(**kwargs):
        cluster_calls.append(kwargs)
        return {
            "success": True,
            "run_id": kwargs["run_id"],
            "run_dir": str(cluster_run_dir),
            "manifest_path": str(manifest_path),
            "returncode": 0,
            "command": ["fake-cluster"],
        }

    monkeypatch.setattr(e2e_sweep, "_invoke_run_cluster_common_eval", _fake_cluster_eval)

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_001",
        run_full_sweep=False,
        cluster_preset="fabric-systems",
        run_fabric=True,
        artifacts_dir=str(tmp_path / "artifacts"),
    )

    stages = {stage["name"]: stage for stage in result["stages"]}
    assert stages["tier1"]["run_id"] == "e2e_001__tier1"
    assert stages["cluster"]["run_id"] == "e2e_001__cluster"
    assert stages["fabric"]["run_id"] == "e2e_001__fabric"
    assert stages["fabric"]["status"] == "skipped_duplicate"
    assert result["overall_status"] == "succeeded"
    assert Path(result["progress_path"]).exists()
    assert "--skip-render-localhost-report" in cluster_calls[0]["extra_args"]
    progress_payload = json.loads(Path(result["progress_path"]).read_text(encoding="utf-8"))
    assert progress_payload["run_id"] == "e2e_001"
    assert progress_payload["current"]["metrics"]["run_state"] == "completed"
    assert progress_payload["current"]["metrics"]["overall_status"] == "succeeded"
    json.dumps(result)
    assert Path(result["manifest_path"]).exists()


def test_run_benchmark_e2e_sweep_marks_partial_when_multi_gpu_bucket_is_skipped(tmp_path: Path, monkeypatch) -> None:
    output_json = tmp_path / "single" / "results.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {
            "counts": {"total": 2, "single_gpu": 1, "multi_gpu": 1},
            "single_gpu": ["ch01:demo"],
            "multi_gpu": ["ch02:dist"],
            "targets": [],
        },
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(e2e_sweep, "_benchmark_queue_lock", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(e2e_sweep, "_visible_gpu_count", lambda **kwargs: 1)
    monkeypatch.setattr(
        e2e_sweep,
        "_invoke_execute_benchmarks",
        lambda **kwargs: {
            "run_id": kwargs["run_id"],
            "output_json": str(output_json),
            "total_failed": 0,
            "total_skipped": 0,
        },
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_partial",
        run_tier1=False,
        run_full_sweep=True,
        run_cluster=False,
        run_fabric=False,
        artifacts_dir=str(tmp_path / "artifacts"),
    )

    full_stage = next(stage for stage in result["stages"] if stage["name"] == "full_sweep")
    assert full_stage["status"] == "partial"
    assert "multi-GPU bucket skipped" in " ".join(full_stage.get("issues", []))
    assert full_stage["result"]["buckets"]["multi_gpu"]["status"] == "skipped"
    assert result["overall_status"] == "partial"


def test_run_benchmark_e2e_sweep_marks_fabric_partial_for_not_configured_scorecard(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "fabric"
    structured_dir = run_dir / "structured"
    structured_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    scorecard_path = structured_dir / "e2e_fabric__fabric_fabric_scorecard.json"
    scorecard_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "families": {
                    "nvlink": {"completeness": "runtime_verified"},
                    "infiniband": {"completeness": "not_configured"},
                },
                "summary": {"configured_management_planes": 0},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {"counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0}, "single_gpu": [], "multi_gpu": [], "targets": []},
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(
        e2e_sweep,
        "_invoke_run_cluster_fabric_eval",
        lambda **kwargs: {
            "success": True,
            "run_id": kwargs["run_id"],
            "run_dir": str(run_dir),
            "manifest_path": str(manifest_path),
            "returncode": 0,
        },
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_fabric",
        run_tier1=False,
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=True,
    )

    fabric_stage = next(stage for stage in result["stages"] if stage["name"] == "fabric")
    assert fabric_stage["status"] == "partial"
    assert result["overall_status"] == "partial"
    assert fabric_stage["artifacts"]["fabric_scorecard"]["degraded_families"] == [
        {"family": "infiniband", "completeness": "not_configured"}
    ]


def test_run_benchmark_e2e_sweep_marks_fabric_partial_for_partial_scorecard_status(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "fabric_partial"
    structured_dir = run_dir / "structured"
    structured_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    scorecard_path = structured_dir / "e2e_fabric_partial__fabric_fabric_scorecard.json"
    scorecard_path.write_text(
        json.dumps(
            {
                "status": "partial",
                "families": {
                    "nvlink": {"completeness": "runtime_verified"},
                    "spectrum-x": {"completeness": "not_present"},
                },
                "summary": {"runtime_verified_families": 1},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {"counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0}, "single_gpu": [], "multi_gpu": [], "targets": []},
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(
        e2e_sweep,
        "_invoke_run_cluster_fabric_eval",
        lambda **kwargs: {
            "success": True,
            "run_id": kwargs["run_id"],
            "run_dir": str(run_dir),
            "manifest_path": str(manifest_path),
            "returncode": 0,
        },
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_fabric_partial",
        run_tier1=False,
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=True,
    )

    fabric_stage = next(stage for stage in result["stages"] if stage["name"] == "fabric")
    assert fabric_stage["status"] == "partial"
    assert "fabric completeness is partial" in " ".join(fabric_stage.get("issues", []))
    assert result["overall_status"] == "partial"


def test_run_benchmark_e2e_sweep_mirrors_cluster_stage_progress(tmp_path: Path, monkeypatch) -> None:
    observed: dict[str, object] = {}
    cluster_run_dir = tmp_path / "cluster" / "runs" / "e2e_cluster_progress__cluster"
    manifest_path = cluster_run_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {"counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0}, "single_gpu": [], "multi_gpu": [], "targets": []},
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(e2e_sweep, "_STAGE_PROGRESS_POLL_SECONDS", 0.01)

    def _fake_cluster_eval(**kwargs):
        child_progress_path = cluster_run_dir / "progress" / "run_progress.json"
        recorder = e2e_sweep.ProgressRecorder(run_id=kwargs["run_id"], progress_path=child_progress_path)
        recorder.emit(
            e2e_sweep.ProgressEvent(
                phase="cluster_eval_suite",
                phase_index=2,
                total_phases=10,
                step="vllm_serve_sweep",
                step_detail="completed 2/10 suite steps",
                percent_complete=20.0,
            )
        )
        time.sleep(0.05)
        observed["payload"] = json.loads(
            (e2e_sweep.e2e_run_dir("e2e_cluster_progress", tmp_path) / "progress.json").read_text(encoding="utf-8")
        )
        return {
            "success": True,
            "run_id": kwargs["run_id"],
            "run_dir": str(cluster_run_dir),
            "manifest_path": str(manifest_path),
            "returncode": 0,
            "command": ["fake-cluster"],
        }

    monkeypatch.setattr(e2e_sweep, "_invoke_run_cluster_common_eval", _fake_cluster_eval)

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_cluster_progress",
        run_tier1=False,
        run_full_sweep=False,
        run_cluster=True,
        run_fabric=False,
    )

    assert result["overall_status"] == "succeeded"
    payload = observed["payload"]
    assert payload["current"]["step"] == "cluster:vllm_serve_sweep"
    assert payload["current"]["step_detail"] == "completed 2/10 suite steps"
    assert payload["current"]["percent_complete"] == 20.0
    assert payload["current"]["metrics"]["current_stage_run_id"] == "e2e_cluster_progress__cluster"


def test_run_benchmark_e2e_sweep_mirrors_fabric_stage_progress(tmp_path: Path, monkeypatch) -> None:
    observed: dict[str, object] = {}
    fabric_run_dir = tmp_path / "cluster" / "runs" / "e2e_fabric_progress__fabric"
    manifest_path = fabric_run_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {"counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0}, "single_gpu": [], "multi_gpu": [], "targets": []},
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(e2e_sweep, "_STAGE_PROGRESS_POLL_SECONDS", 0.01)

    def _fake_fabric_eval(**kwargs):
        child_progress_path = fabric_run_dir / "progress" / "run_progress.json"
        recorder = e2e_sweep.ProgressRecorder(run_id=kwargs["run_id"], progress_path=child_progress_path)
        recorder.emit(
            e2e_sweep.ProgressEvent(
                phase="cluster_eval_suite",
                phase_index=5,
                total_phases=12,
                step="build_fabric_eval",
                step_detail="completed 4/12 suite steps",
                percent_complete=33.3333,
            )
        )
        time.sleep(0.05)
        observed["payload"] = json.loads(
            (e2e_sweep.e2e_run_dir("e2e_fabric_progress", tmp_path) / "progress.json").read_text(encoding="utf-8")
        )
        return {
            "success": True,
            "run_id": kwargs["run_id"],
            "run_dir": str(fabric_run_dir),
            "manifest_path": str(manifest_path),
            "returncode": 0,
            "command": ["fake-fabric"],
        }

    monkeypatch.setattr(e2e_sweep, "_invoke_run_cluster_fabric_eval", _fake_fabric_eval)

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_fabric_progress",
        run_tier1=False,
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=True,
    )

    assert result["overall_status"] == "succeeded"
    payload = observed["payload"]
    assert payload["current"]["step"] == "fabric:build_fabric_eval"
    assert payload["current"]["step_detail"] == "completed 4/12 suite steps"
    assert payload["current"]["percent_complete"] == 33.3333
    assert payload["current"]["metrics"]["current_stage_run_id"] == "e2e_fabric_progress__fabric"


def test_emit_live_progress_includes_child_stage_progress(tmp_path: Path) -> None:
    progress_path = tmp_path / "progress.json"
    recorder = e2e_sweep.ProgressRecorder(run_id="e2e_progress", progress_path=progress_path)
    stages = [
        {
            "name": "tier1",
            "enabled": True,
            "run_id": "e2e_progress__tier1",
            "status": "running",
            "attempts": [{"run_id": "e2e_progress__tier1", "status": "running"}],
        },
        {
            "name": "full_sweep",
            "enabled": True,
            "run_id": "e2e_progress__full_sweep",
            "status": "planned",
            "attempts": [],
        },
    ]

    e2e_sweep._emit_live_progress(
        recorder,
        stages=stages,
        run_state="running",
        overall_status="running",
        artifact_paths={"summary_path": tmp_path / "summary.json"},
        child_progress={
            "step": "ch04:gradient_fusion",
            "step_detail": "optimized timing (optimized_gradient_fusion)",
            "percent_complete": 50.0,
        },
        child_stage_name="tier1",
        child_run_id="e2e_progress__tier1",
    )

    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert payload["current"]["step"] == "tier1:ch04:gradient_fusion"
    assert payload["current"]["step_detail"] == "optimized timing (optimized_gradient_fusion)"
    assert payload["current"]["percent_complete"] == 25.0
    assert payload["current"]["metrics"]["current_stage_run_id"] == "e2e_progress__tier1"
    assert payload["current"]["metrics"]["child_progress"]["percent_complete"] == 50.0


def test_run_benchmark_e2e_sweep_writes_progress_and_checkpoint_before_stage_invocation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observed: dict[str, dict[str, object]] = {}
    summary_path = tmp_path / "tier1" / "summary.json"
    regression_summary_path = tmp_path / "tier1" / "regression_summary.md"
    trend_snapshot_path = tmp_path / "tier1" / "trend.json"
    history_root = tmp_path / "history"
    for path in (summary_path, regression_summary_path, trend_snapshot_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    history_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {"counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0}, "single_gpu": [], "multi_gpu": [], "targets": []},
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(e2e_sweep, "_benchmark_queue_lock", lambda *args, **kwargs: contextlib.nullcontext())

    def _fake_tier1(**kwargs):
        run_dir = e2e_sweep.e2e_run_dir("e2e_checkpoint", tmp_path)
        observed["progress"] = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
        observed["checkpoint"] = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
        return {
            "execution": {"run_id": kwargs["run_id"], "total_failed": 0},
            "summary_path": str(summary_path),
            "regression_summary_path": str(regression_summary_path),
            "trend_snapshot_path": str(trend_snapshot_path),
            "history_root": str(history_root),
            "suite_definitions": [],
            "total_failed": 0,
            "total_skipped": 0,
        }

    monkeypatch.setattr(e2e_sweep, "_invoke_run_tier1_suite", _fake_tier1)

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_checkpoint",
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
    )

    assert result["overall_status"] == "succeeded"
    assert observed["progress"]["run_id"] == "e2e_checkpoint"
    assert observed["progress"]["current"]["metrics"]["run_state"] == "running"
    assert observed["progress"]["current"]["metrics"]["overall_status"] == "running"
    assert observed["checkpoint"]["run_id"] == "e2e_checkpoint"
    assert observed["checkpoint"]["run_state"] == "running"
    assert observed["checkpoint"]["overall_status"] == "running"
    assert observed["checkpoint"]["stages"][0]["name"] == "tier1"
    assert observed["checkpoint"]["stages"][0]["status"] == "running"


def test_run_benchmark_e2e_sweep_persists_aborted_state_on_unhandled_exception(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {"counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0}, "single_gpu": [], "multi_gpu": [], "targets": []},
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(e2e_sweep, "_benchmark_queue_lock", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(
        e2e_sweep,
        "_invoke_run_tier1_suite",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("tier1 exploded")),
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_abort",
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
    )

    run_dir = e2e_sweep.e2e_run_dir("e2e_abort", tmp_path)
    summary_payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    checkpoint_payload = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    manifest_payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert result["run_state"] == "aborted"
    assert result["overall_status"] == "aborted"
    assert result["resume_available"] is True
    assert "tier1 exploded" in result["error"]
    assert summary_payload["run_state"] == "aborted"
    assert summary_payload["overall_status"] == "aborted"
    assert summary_payload["resume_available"] is True
    assert checkpoint_payload["run_state"] == "aborted"
    assert checkpoint_payload["resume_available"] is True
    assert manifest_payload["checkpoint"]["run_state"] == "aborted"
    tier1_stage = next(stage for stage in summary_payload["stages"] if stage["name"] == "tier1")
    assert tier1_stage["status"] == "aborted"
    assert tier1_stage["attempts"][-1]["status"] == "aborted"


def test_run_benchmark_e2e_sweep_resume_requires_explicit_run_id() -> None:
    result = e2e_sweep.run_benchmark_e2e_sweep(
        resume=True,
        run_tier1=False,
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
    )

    assert result["success"] is False
    assert result["overall_status"] == "failed"
    assert "requires an explicit run_id" in result["error"]


def test_run_benchmark_e2e_sweep_resume_rejects_contract_mismatch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {"counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0}, "single_gpu": [], "multi_gpu": [], "targets": []},
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )

    run_dir = e2e_sweep.e2e_run_dir("e2e_resume_mismatch", tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-03-25T00:00:00Z",
                "contract": {
                    "profile_type": "none",
                    "run_tier1": False,
                    "run_full_sweep": False,
                    "run_cluster": False,
                    "run_fabric": False,
                    "cluster_preset": "common-answer-fast",
                    "hosts": ["localhost"],
                    "labels": ["localhost"],
                    "ssh_user": getpass.getuser(),
                    "ssh_key": None,
                    "bench_root": str(tmp_path),
                    "validity_profile": "strict",
                    "single_gpu": False,
                },
                "stages": [
                    {"name": "tier1", "enabled": False, "status": "planned", "attempts": []},
                    {"name": "full_sweep", "enabled": False, "status": "planned", "attempts": []},
                    {"name": "cluster", "enabled": False, "status": "planned", "attempts": []},
                    {"name": "fabric", "enabled": False, "status": "planned", "attempts": []},
                ],
                "frozen_plan": {"full_sweep": {"single_gpu_targets": [], "multi_gpu_targets": []}},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_resume_mismatch",
        resume=True,
        run_tier1=False,
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
        profile_type="minimal",
    )

    assert result["success"] is False
    assert result["overall_status"] == "failed"
    assert result["resume_available"] is True
    assert "Resume contract mismatch" in result["error"]
    assert "profile_type" in result["error"]


def test_run_benchmark_e2e_sweep_resume_marks_superseded_running_attempt_aborted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    summary_path = tmp_path / "tier1" / "summary.json"
    regression_summary_path = tmp_path / "tier1" / "regression_summary.md"
    trend_snapshot_path = tmp_path / "tier1" / "trend.json"
    history_root = tmp_path / "history"
    for path in (summary_path, regression_summary_path, trend_snapshot_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    history_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {"counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0}, "single_gpu": [], "multi_gpu": [], "targets": []},
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(e2e_sweep, "_benchmark_queue_lock", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(
        e2e_sweep,
        "_invoke_run_tier1_suite",
        lambda **kwargs: {
            "execution": {"run_id": kwargs["run_id"], "total_failed": 0},
            "summary_path": str(summary_path),
            "regression_summary_path": str(regression_summary_path),
            "trend_snapshot_path": str(trend_snapshot_path),
            "history_root": str(history_root),
            "suite_definitions": [],
            "total_failed": 0,
            "total_skipped": 0,
        },
    )

    run_dir = e2e_sweep.e2e_run_dir("e2e_resume_running", tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-03-25T00:00:00Z",
                "contract": {
                    "profile_type": "minimal",
                    "run_tier1": True,
                    "run_full_sweep": False,
                    "run_cluster": False,
                    "run_fabric": False,
                    "cluster_preset": "common-answer-fast",
                    "hosts": ["localhost"],
                    "labels": ["localhost"],
                    "ssh_user": getpass.getuser(),
                    "ssh_key": None,
                    "bench_root": str(tmp_path),
                    "validity_profile": "strict",
                    "single_gpu": False,
                },
                "stages": [
                    {
                        "name": "tier1",
                        "enabled": True,
                        "status": "running",
                        "run_id": "e2e_resume_running__tier1",
                        "attempts": [
                            {
                                "run_id": "e2e_resume_running__tier1",
                                "status": "running",
                            }
                        ],
                    },
                    {"name": "full_sweep", "enabled": False, "status": "skipped", "run_id": "e2e_resume_running__full_sweep", "attempts": []},
                    {"name": "cluster", "enabled": False, "status": "skipped", "run_id": "e2e_resume_running__cluster", "attempts": []},
                    {"name": "fabric", "enabled": False, "status": "skipped", "run_id": "e2e_resume_running__fabric", "attempts": []},
                ],
                "frozen_plan": {"full_sweep": {"single_gpu_targets": [], "multi_gpu_targets": []}},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_resume_running",
        resume=True,
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
    )

    tier1_stage = next(stage for stage in result["stages"] if stage["name"] == "tier1")
    assert tier1_stage["status"] == "succeeded"
    assert len(tier1_stage["attempts"]) == 2
    assert tier1_stage["attempts"][0]["status"] == "aborted"
    assert "resume superseded unfinished attempt" in tier1_stage["attempts"][0]["issues"]
    assert tier1_stage["attempts"][1]["status"] == "succeeded"


def test_normalize_stale_running_resume_state_marks_dead_orchestrator_aborted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(e2e_sweep, "_pid_is_live", lambda _pid: False)

    resume_state = {
        "run_state": "running",
        "overall_status": "running",
        "success": False,
        "resume_available": False,
        "orchestrator_pid": 999999,
        "stages": [
            {
                "name": "tier1",
                "enabled": True,
                "status": "running",
                "run_id": "stale_e2e__tier1",
                "attempts": [
                    {
                        "run_id": "stale_e2e__tier1",
                        "status": "running",
                    }
                ],
            },
            {
                "name": "full_sweep",
                "enabled": False,
                "status": "skipped",
                "run_id": "stale_e2e__full_sweep",
                "attempts": [],
            },
        ],
    }

    reason = e2e_sweep._normalize_stale_running_resume_state(
        resume_state,
        repo_root=tmp_path,
        artifacts_dir=str(tmp_path / "artifacts"),
    )

    assert reason == "orchestrator process 999999 exited without finalizing run state"
    assert resume_state["run_state"] == "aborted"
    assert resume_state["overall_status"] == "aborted"
    assert resume_state["resume_available"] is True
    assert resume_state["stages"][0]["status"] == "aborted"
    assert resume_state["stages"][0]["attempts"][0]["status"] == "aborted"
    assert reason in resume_state["stages"][0]["attempts"][0]["issues"]


def test_run_benchmark_e2e_sweep_resume_resolves_current_targets_for_remaining_units(
    tmp_path: Path,
    monkeypatch,
) -> None:
    execute_calls: list[dict[str, object]] = []
    runs_root = tmp_path / "bench_runs"

    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {
            "counts": {"total": 2, "single_gpu": 2, "multi_gpu": 0},
            "single_gpu": ["ch13:torchao_quantization", "ch14:cublas_vs_cutlass"],
            "multi_gpu": [],
            "targets": [],
        },
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(e2e_sweep, "_benchmark_queue_lock", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(e2e_sweep, "_visible_gpu_count", lambda **kwargs: 1)
    monkeypatch.setattr(e2e_sweep, "_invoke_run_tier1_suite", lambda **kwargs: {})

    def _fake_execute(**kwargs):
        execute_calls.append(kwargs)
        paths = e2e_sweep._benchmark_run_event_paths(
            kwargs["run_id"],
            repo_root=tmp_path,
            artifacts_dir=str(runs_root),
        )
        paths["events"].parent.mkdir(parents=True, exist_ok=True)
        paths["output_json"].parent.mkdir(parents=True, exist_ok=True)
        paths["progress"].parent.mkdir(parents=True, exist_ok=True)
        paths["events"].write_text(
            "\n".join(
                [
                    json.dumps({"event_type": "run_start", "targets": kwargs["targets"]}),
                    json.dumps({"event_type": "chapter_start", "chapter": "ch13"}),
                    json.dumps({"event_type": "chapter_end", "chapter": "ch13"}),
                    json.dumps({"event_type": "chapter_start", "chapter": "ch14"}),
                    json.dumps({"event_type": "chapter_end", "chapter": "ch14"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        paths["output_json"].write_text(json.dumps({"results": []}), encoding="utf-8")
        paths["progress"].write_text("{}", encoding="utf-8")
        return {
            "run_id": kwargs["run_id"],
            "output_json": str(paths["output_json"]),
            "total_failed": 0,
            "total_skipped": 0,
        }

    monkeypatch.setattr(e2e_sweep, "_invoke_execute_benchmarks", _fake_execute)

    run_dir = e2e_sweep.e2e_run_dir("e2e_resume_unit_map", tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-03-25T00:00:00Z",
                "contract": {
                    "profile_type": "minimal",
                    "run_tier1": False,
                    "run_full_sweep": True,
                    "run_cluster": False,
                    "run_fabric": False,
                    "cluster_preset": "common-answer-fast",
                    "hosts": ["localhost"],
                    "labels": ["localhost"],
                    "ssh_user": getpass.getuser(),
                    "ssh_key": None,
                    "bench_root": str(tmp_path),
                    "validity_profile": "strict",
                    "single_gpu": False,
                },
                "stages": [
                    {"name": "tier1", "enabled": False, "status": "skipped", "run_id": "e2e_resume_unit_map__tier1", "attempts": []},
                    {
                        "name": "full_sweep",
                        "enabled": True,
                        "status": "aborted",
                        "run_id": "e2e_resume_unit_map__full_sweep",
                        "attempts": [
                            {
                                "run_id": "e2e_resume_unit_map__full_sweep__single",
                                "bucket": "single_gpu",
                                "status": "aborted",
                                "targets": ["ch13:torchao_quantization", "ch14:cutlass"],
                                "units": ["ch13", "ch14"],
                                "completed_units": [],
                                "active_unit": "ch13",
                            }
                        ],
                    },
                    {"name": "cluster", "enabled": False, "status": "skipped", "run_id": "e2e_resume_unit_map__cluster", "attempts": []},
                    {"name": "fabric", "enabled": False, "status": "skipped", "run_id": "e2e_resume_unit_map__fabric", "attempts": []},
                ],
                "frozen_plan": {
                    "full_sweep": {
                        "single_gpu_targets": ["ch13:torchao_quantization", "ch14:cutlass"],
                        "single_gpu_units": ["ch13", "ch14"],
                        "multi_gpu_targets": [],
                        "multi_gpu_units": [],
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_resume_unit_map",
        resume=True,
        run_tier1=False,
        run_full_sweep=True,
        run_cluster=False,
        run_fabric=False,
        artifacts_dir=str(runs_root),
    )

    assert result["overall_status"] == "succeeded"
    assert len(execute_calls) == 1
    assert execute_calls[0]["targets"] == ["ch13:torchao_quantization", "ch14:cublas_vs_cutlass"]


def test_run_benchmark_e2e_sweep_resume_reruns_partial_unit_with_resume_attempt_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    execute_calls: list[dict[str, object]] = []
    tier1_calls: list[dict[str, object]] = []
    runs_root = tmp_path / "bench_runs"

    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {
            "counts": {"total": 4, "single_gpu": 4, "multi_gpu": 0},
            "single_gpu": [
                "ch12:done",
                "ch13:torchao_quantization",
                "ch13:training_speed",
                "ch14:after",
            ],
            "multi_gpu": [],
            "targets": [],
        },
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(e2e_sweep, "_benchmark_queue_lock", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(e2e_sweep, "_visible_gpu_count", lambda **kwargs: 1)
    monkeypatch.setattr(
        e2e_sweep,
        "_invoke_run_tier1_suite",
        lambda **kwargs: tier1_calls.append(kwargs) or {},
    )

    def _fake_execute(**kwargs):
        execute_calls.append(kwargs)
        paths = e2e_sweep._benchmark_run_event_paths(
            kwargs["run_id"],
            repo_root=tmp_path,
            artifacts_dir=str(runs_root),
        )
        paths["events"].parent.mkdir(parents=True, exist_ok=True)
        paths["output_json"].parent.mkdir(parents=True, exist_ok=True)
        paths["progress"].parent.mkdir(parents=True, exist_ok=True)
        paths["events"].write_text(
            "\n".join(
                [
                    json.dumps({"event_type": "run_start", "targets": kwargs["targets"]}),
                    json.dumps({"event_type": "chapter_start", "chapter": "ch13"}),
                    json.dumps({"event_type": "chapter_end", "chapter": "ch13"}),
                    json.dumps({"event_type": "chapter_start", "chapter": "ch14"}),
                    json.dumps({"event_type": "chapter_end", "chapter": "ch14"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        paths["output_json"].write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "chapter": "ch13",
                            "benchmarks": [
                                {"example": "torchao_quantization", "status": "succeeded"},
                                {"example": "training_speed", "status": "succeeded"},
                            ],
                        },
                        {
                            "chapter": "ch14",
                            "benchmarks": [
                                {"example": "after", "status": "succeeded"},
                            ],
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        paths["progress"].write_text("{}", encoding="utf-8")
        return {
            "run_id": kwargs["run_id"],
            "output_json": str(paths["output_json"]),
            "total_failed": 0,
            "total_skipped": 0,
        }

    monkeypatch.setattr(e2e_sweep, "_invoke_execute_benchmarks", _fake_execute)

    run_dir = e2e_sweep.e2e_run_dir("e2e_resume", tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-03-25T00:00:00Z",
                "contract": {
                    "profile_type": "minimal",
                    "run_tier1": True,
                    "run_full_sweep": True,
                    "run_cluster": False,
                    "run_fabric": False,
                    "cluster_preset": "common-answer-fast",
                    "hosts": ["localhost"],
                    "labels": ["localhost"],
                    "ssh_user": getpass.getuser(),
                    "ssh_key": None,
                    "bench_root": str(tmp_path),
                    "validity_profile": "strict",
                    "single_gpu": False,
                    "accept_regressions": False,
                    "update_expectations": False,
                    "allow_mixed_provenance": False,
                    "allow_portable_expectations_update": False,
                    "timeout_multiplier": 3.0,
                    "suite_timeout": 14400,
                    "timeout_seconds": None,
                    "iterations": None,
                    "warmup": None,
                    "gpu_sm_clock_mhz": None,
                    "gpu_mem_clock_mhz": None,
                    "ncu_metric_set": "minimal",
                    "ncu_replay_mode": None,
                    "nsys_timeout_seconds": None,
                    "ncu_timeout_seconds": None,
                },
                "stages": [
                    {
                        "name": "tier1",
                        "enabled": True,
                        "status": "succeeded",
                        "run_id": "e2e_resume__tier1",
                        "attempts": [{"run_id": "e2e_resume__tier1", "status": "succeeded"}],
                    },
                    {
                        "name": "full_sweep",
                        "enabled": True,
                        "status": "aborted",
                        "run_id": "e2e_resume__full_sweep",
                        "attempts": [
                            {
                                "run_id": "e2e_resume__full_sweep__single",
                                "bucket": "single_gpu",
                                "status": "aborted",
                                "targets": [
                                    "ch12:done",
                                    "ch13:torchao_quantization",
                                    "ch13:training_speed",
                                    "ch14:after",
                                ],
                                "units": ["ch12", "ch13", "ch14"],
                                "completed_units": ["ch12"],
                                "active_unit": "ch13",
                            }
                        ],
                    },
                    {"name": "cluster", "enabled": False, "status": "planned", "run_id": "e2e_resume__cluster", "attempts": []},
                    {"name": "fabric", "enabled": False, "status": "planned", "run_id": "e2e_resume__fabric", "attempts": []},
                ],
                "frozen_plan": {
                    "full_sweep": {
                        "single_gpu_targets": [
                            "ch12:done",
                            "ch13:torchao_quantization",
                            "ch13:training_speed",
                            "ch14:after",
                        ],
                        "single_gpu_units": ["ch12", "ch13", "ch14"],
                        "multi_gpu_targets": [],
                        "multi_gpu_units": [],
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_resume",
        resume=True,
        run_tier1=True,
        run_full_sweep=True,
        run_cluster=False,
        run_fabric=False,
        artifacts_dir=str(runs_root),
    )

    assert tier1_calls == []
    assert len(execute_calls) == 1
    assert execute_calls[0]["run_id"] == "e2e_resume__full_sweep__single__resume1"
    assert execute_calls[0]["targets"] == [
        "ch13:torchao_quantization",
        "ch13:training_speed",
        "ch14:after",
    ]

    full_stage = next(stage for stage in result["stages"] if stage["name"] == "full_sweep")
    assert full_stage["status"] == "succeeded"
    assert len(full_stage["attempts"]) == 2
    assert full_stage["attempts"][0]["run_id"] == "e2e_resume__full_sweep__single"
    assert full_stage["attempts"][1]["run_id"] == "e2e_resume__full_sweep__single__resume1"
    assert full_stage["attempts"][1]["completed_units"] == ["ch13", "ch14"]
    assert full_stage["result"]["buckets"]["single_gpu"]["latest_attempt_run_id"] == "e2e_resume__full_sweep__single__resume1"
    assert result["overall_status"] == "succeeded"


def test_run_benchmark_e2e_sweep_resume_canonicalizes_lab_unit_names(
    tmp_path: Path,
    monkeypatch,
) -> None:
    execute_calls: list[dict[str, object]] = []
    runs_root = tmp_path / "bench_runs"

    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {
            "counts": {"total": 2, "single_gpu": 2, "multi_gpu": 0},
            "single_gpu": [
                "labs/async_input_pipeline:async_input_pipeline",
                "labs/trtllm_phi_3_5_moe:trtllm_phi_3_5_moe",
            ],
            "multi_gpu": [],
            "targets": [],
        },
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(e2e_sweep, "_benchmark_queue_lock", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(e2e_sweep, "_visible_gpu_count", lambda **kwargs: 1)

    def _fake_execute(**kwargs):
        execute_calls.append(kwargs)
        paths = e2e_sweep._benchmark_run_event_paths(
            kwargs["run_id"],
            repo_root=tmp_path,
            artifacts_dir=str(runs_root),
        )
        paths["events"].parent.mkdir(parents=True, exist_ok=True)
        paths["output_json"].parent.mkdir(parents=True, exist_ok=True)
        paths["progress"].parent.mkdir(parents=True, exist_ok=True)
        paths["events"].write_text(
            "\n".join(
                [
                    json.dumps({"event_type": "run_start", "targets": kwargs["targets"]}),
                    json.dumps({"event_type": "chapter_start", "chapter": "labs_trtllm_phi_3_5_moe"}),
                    json.dumps({"event_type": "chapter_end", "chapter": "labs_trtllm_phi_3_5_moe"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        paths["output_json"].write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "chapter": "labs_trtllm_phi_3_5_moe",
                            "benchmarks": [
                                {"example": "trtllm_phi_3_5_moe", "status": "succeeded"},
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        paths["progress"].write_text("{}", encoding="utf-8")
        return {
            "run_id": kwargs["run_id"],
            "output_json": str(paths["output_json"]),
            "total_failed": 0,
            "total_skipped": 0,
        }

    monkeypatch.setattr(e2e_sweep, "_invoke_execute_benchmarks", _fake_execute)

    prior_paths = e2e_sweep._benchmark_run_event_paths(
        "e2e_resume_labs__full_sweep__single",
        repo_root=tmp_path,
        artifacts_dir=str(runs_root),
    )
    prior_paths["events"].parent.mkdir(parents=True, exist_ok=True)
    prior_paths["output_json"].parent.mkdir(parents=True, exist_ok=True)
    prior_paths["progress"].parent.mkdir(parents=True, exist_ok=True)
    prior_paths["events"].write_text(
        "\n".join(
            [
                json.dumps({"event_type": "run_start", "targets": [
                    "labs/async_input_pipeline:async_input_pipeline",
                    "labs/trtllm_phi_3_5_moe:trtllm_phi_3_5_moe",
                ]}),
                json.dumps({"event_type": "chapter_start", "chapter": "labs_async_input_pipeline"}),
                json.dumps({"event_type": "chapter_end", "chapter": "labs_async_input_pipeline"}),
                json.dumps({"event_type": "chapter_start", "chapter": "labs_trtllm_phi_3_5_moe"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    prior_paths["output_json"].write_text(
        json.dumps(
            {
                "results": [
                    {
                        "chapter": "labs_async_input_pipeline",
                        "benchmarks": [
                            {"example": "async_input_pipeline", "status": "succeeded"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    prior_paths["progress"].write_text("{}", encoding="utf-8")

    run_dir = e2e_sweep.e2e_run_dir("e2e_resume_labs", tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-03-25T00:00:00Z",
                "run_state": "running",
                "orchestrator_pid": 999999,
                "contract": {
                    "profile_type": "minimal",
                    "run_tier1": False,
                    "run_full_sweep": True,
                    "run_cluster": False,
                    "run_fabric": False,
                    "cluster_preset": "common-answer-fast",
                    "hosts": ["localhost"],
                    "labels": ["localhost"],
                    "ssh_user": getpass.getuser(),
                    "ssh_key": None,
                    "bench_root": str(tmp_path),
                    "validity_profile": "strict",
                    "single_gpu": False,
                },
                "stages": [
                    {"name": "tier1", "enabled": False, "status": "skipped", "run_id": "e2e_resume_labs__tier1", "attempts": []},
                    {
                        "name": "full_sweep",
                        "enabled": True,
                        "status": "running",
                        "run_id": "e2e_resume_labs__full_sweep",
                        "attempts": [
                            {
                                "run_id": "e2e_resume_labs__full_sweep__single",
                                "bucket": "single_gpu",
                                "status": "running",
                                "targets": [
                                    "labs/async_input_pipeline:async_input_pipeline",
                                    "labs/trtllm_phi_3_5_moe:trtllm_phi_3_5_moe",
                                ],
                                "units": ["labs/async_input_pipeline", "labs/trtllm_phi_3_5_moe"],
                                "completed_units": [],
                                "active_unit": "labs/async_input_pipeline",
                            }
                        ],
                    },
                    {"name": "cluster", "enabled": False, "status": "skipped", "run_id": "e2e_resume_labs__cluster", "attempts": []},
                    {"name": "fabric", "enabled": False, "status": "skipped", "run_id": "e2e_resume_labs__fabric", "attempts": []},
                ],
                "frozen_plan": {
                    "full_sweep": {
                        "single_gpu_targets": [
                            "labs/async_input_pipeline:async_input_pipeline",
                            "labs/trtllm_phi_3_5_moe:trtllm_phi_3_5_moe",
                        ],
                        "single_gpu_units": ["labs/async_input_pipeline", "labs/trtllm_phi_3_5_moe"],
                        "multi_gpu_targets": [],
                        "multi_gpu_units": [],
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_resume_labs",
        resume=True,
        run_tier1=False,
        run_full_sweep=True,
        run_cluster=False,
        run_fabric=False,
        artifacts_dir=str(runs_root),
    )

    assert len(execute_calls) == 1
    assert execute_calls[0]["run_id"] == "e2e_resume_labs__full_sweep__single__resume1"
    assert execute_calls[0]["targets"] == ["labs/trtllm_phi_3_5_moe:trtllm_phi_3_5_moe"]
    full_stage = next(stage for stage in result["stages"] if stage["name"] == "full_sweep")
    assert full_stage["attempts"][0]["status"] == "aborted"
    assert full_stage["attempts"][0]["completed_units"] == ["labs/async_input_pipeline"]
    assert full_stage["attempts"][0]["active_unit"] == "labs/trtllm_phi_3_5_moe"
    assert full_stage["attempts"][1]["completed_units"] == ["labs/trtllm_phi_3_5_moe"]
    assert result["overall_status"] == "succeeded"


def test_run_benchmark_e2e_sweep_rejects_portable_expectation_writes() -> None:
    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_invalid",
        run_tier1=False,
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
        validity_profile="portable",
        update_expectations=True,
        dry_run=True,
    )

    assert result["success"] is False
    assert result["overall_status"] == "failed"
    assert "allow-portable-expectations-update" in result["error"]


def test_run_benchmark_e2e_sweep_rejects_non_local_hosts_without_ssh_credentials(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {"counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0}, "single_gpu": [], "multi_gpu": [], "targets": []},
    )
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_bad_hosts",
        run_tier1=False,
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
        hosts=["gpu-node-1"],
        artifacts_dir=str(tmp_path / "artifacts"),
    )

    assert result["success"] is False
    assert result["overall_status"] == "failed"
    assert "Non-local hosts require explicit ssh_user and ssh_key" in result["error"]


def test_benchmark_e2e_sweep_handler_returns_async_ticket(monkeypatch) -> None:
    monkeypatch.setattr(
        "core.benchmark.e2e_sweep.run_benchmark_e2e_sweep",
        lambda **kwargs: {"success": True, "run_id": kwargs["run_id"]},
    )

    ticket = handlers.benchmark_e2e_sweep(
        {
            "async": True,
            "run_tier1": False,
            "run_full_sweep": False,
            "run_cluster": False,
            "run_fabric": False,
            "run_id": "handler_async_e2e",
        }
    )

    assert ticket["status"] == "queued"
    assert ticket["run_id"] == "handler_async_e2e"
    assert ticket["run_dir"].endswith("artifacts/e2e_runs/handler_async_e2e")
    assert ticket["progress_path"].endswith("artifacts/e2e_runs/handler_async_e2e/progress.json")

    record = handlers.JobStore.get().get_status(ticket["job_id"])
    if record is not None:
        handlers.JobStore.get().update_job(ticket["job_id"], status="completed")


def test_tool_benchmark_e2e_sweep_delegates_to_handler(monkeypatch) -> None:
    monkeypatch.setattr(
        "core.api.handlers.benchmark_e2e_sweep",
        lambda params: {"success": True, "run_id": "tool_e2e", "echo": params.get("dry_run", False)},
    )

    payload = mcp_server.tool_benchmark_e2e_sweep({"dry_run": True})

    assert payload["success"] is True
    assert payload["run_id"] == "tool_e2e"
    assert payload["echo"] is True


def test_benchmark_e2e_sweep_handler_passes_resume_flag(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run(**kwargs):
        captured.update(kwargs)
        return {"success": True, "run_id": kwargs["run_id"], "resume": kwargs["resume"]}

    monkeypatch.setattr("core.benchmark.e2e_sweep.run_benchmark_e2e_sweep", _fake_run)

    payload = handlers.benchmark_e2e_sweep(
        {
            "run_id": "handler_resume_e2e",
            "resume": True,
            "run_tier1": False,
            "run_full_sweep": False,
            "run_cluster": False,
            "run_fabric": False,
        }
    )

    assert payload["success"] is True
    assert payload["resume"] is True
    assert captured["resume"] is True
