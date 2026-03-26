from __future__ import annotations

import contextlib
import json
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
