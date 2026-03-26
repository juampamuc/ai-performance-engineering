"""End-to-end benchmark orchestration for tier1, full sweep, and cluster eval."""

from __future__ import annotations

import contextlib
import getpass
import json
import os
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import fcntl  # POSIX-only; optional for portability.
except Exception:  # pragma: no cover - non-POSIX environments
    fcntl = None  # type: ignore[assignment]

from core.benchmark.artifact_manager import build_run_id
from core.benchmark.expectations import detect_expectation_key
from core.benchmark.run_manifest import get_git_info
from core.discovery import chapter_slug, discover_all_chapters, discover_benchmarks, is_cuda_binary_benchmark_file
from core.harness.progress import ProgressEvent, ProgressRecorder
from core.harness.validity_checks import detect_execution_environment
from core.harness.validity_profile import normalize_validity_profile


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def e2e_runs_root(repo_root: Optional[Path] = None) -> Path:
    return Path(repo_root or _repo_root()) / "artifacts" / "e2e_runs"


def e2e_run_dir(run_id: str, repo_root: Optional[Path] = None) -> Path:
    return e2e_runs_root(repo_root) / run_id


def e2e_progress_path(run_dir: Path) -> Path:
    return Path(run_dir) / "progress.json"


def resolve_e2e_run_id(run_id: Optional[str] = None, *, repo_root: Optional[Path] = None) -> str:
    if run_id and str(run_id).strip():
        return str(run_id).strip()
    return build_run_id("benchmark_e2e_sweep", base_dir=e2e_runs_root(repo_root))


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _append_event(events_path: Path, event: str, **fields: Any) -> None:
    events_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts": _utc_now(), "event": event, **fields}
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=_json_default) + "\n")


def _queue_lock_timeout_seconds() -> int:
    raw = os.environ.get("AISP_MCP_QUEUE_RUNNER_LOCK_TIMEOUT_SEC", "1800")
    try:
        value = int(raw)
    except Exception:
        value = 1800
    return max(0, value)


@contextlib.contextmanager
def _benchmark_queue_lock(stage_name: str, run_id: str, *, repo_root: Optional[Path] = None):
    if fcntl is None:
        yield
        return

    root = Path(repo_root or _repo_root())
    queue_dir = root / "artifacts" / "parallel_runs"
    queue_dir.mkdir(parents=True, exist_ok=True)
    lock_path = queue_dir / "queue.runner.lock"
    lock_path.touch(exist_ok=True)
    timeout = _queue_lock_timeout_seconds()
    started = time.monotonic()
    handle = lock_path.open("a+", encoding="utf-8")
    acquired = False
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                if timeout > 0 and (time.monotonic() - started) >= timeout:
                    raise RuntimeError(
                        f"Benchmark queue lock timeout after {timeout}s "
                        f"(stage={stage_name}, run_id={run_id})"
                    )
                time.sleep(1.0)
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "ts": _utc_now(),
                    "stage": stage_name,
                    "run_id": run_id,
                    "pid": os.getpid(),
                }
            )
            + "\n"
        )
        handle.flush()
        yield
    finally:
        if acquired:
            with contextlib.suppress(OSError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            handle.close()


def _is_local_host(host: str) -> bool:
    normalized = str(host or "").strip().lower()
    if normalized in {"localhost", "127.0.0.1", "::1"}:
        return True

    local_hostnames = {
        socket.gethostname().lower(),
        socket.getfqdn().lower(),
    }
    local_hostnames.update({name.split(".", 1)[0] for name in local_hostnames})
    return normalized in local_hostnames


def _normalize_cluster_hosts_and_labels(
    *,
    hosts: Optional[List[str]],
    labels: Optional[List[str]],
    ssh_user: Optional[str],
    ssh_key: Optional[str],
) -> Dict[str, Any]:
    normalized_hosts = [str(host).strip() for host in (hosts or []) if str(host).strip()]
    if not normalized_hosts:
        normalized_hosts = ["localhost"]

    normalized_labels = [str(label).strip() for label in (labels or []) if str(label).strip()]
    local_only = all(_is_local_host(host) for host in normalized_hosts)
    if not normalized_labels:
        if local_only:
            normalized_labels = ["localhost"] * len(normalized_hosts)
        else:
            normalized_labels = list(normalized_hosts)
    if len(normalized_labels) != len(normalized_hosts):
        raise ValueError("labels must match hosts count")

    effective_ssh_user = ssh_user
    if local_only and not effective_ssh_user:
        effective_ssh_user = getpass.getuser()
    if not local_only and (not ssh_user or not ssh_key):
        raise ValueError("Non-local hosts require explicit ssh_user and ssh_key for run-e2e.")

    return {
        "hosts": normalized_hosts,
        "labels": normalized_labels,
        "ssh_user": effective_ssh_user,
        "ssh_key": ssh_key,
        "local_only": local_only,
    }


def _with_e2e_cluster_extra_args(extra_args: Optional[List[str]]) -> Optional[List[str]]:
    merged = [str(arg) for arg in (extra_args or []) if str(arg).strip()]
    render_flags = {"--render-localhost-report", "--skip-render-localhost-report"}
    if not any(flag in render_flags for flag in merged):
        merged.append("--skip-render-localhost-report")
    return merged or None


def _visible_gpu_count(*, single_gpu: bool) -> int:
    if single_gpu:
        return 1

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip():
        devices = [token.strip() for token in visible.split(",") if token.strip() and token.strip() != "-1"]
        if devices:
            return len(devices)

    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:
        pass
    return 0


def _validate_expectation_policy(
    *,
    validity_profile: str,
    allow_portable_expectations_update: bool,
    update_expectations: bool,
    accept_regressions: bool,
    allow_mixed_provenance: bool,
) -> Optional[str]:
    if validity_profile != "portable" or allow_portable_expectations_update:
        return None
    requested = []
    if update_expectations:
        requested.append("--update-expectations")
    if accept_regressions:
        requested.append("--accept-regressions")
    if allow_mixed_provenance:
        requested.append("--allow-mixed-provenance")
    if not requested:
        return None
    requested_summary = ", ".join(requested)
    return (
        "Invalid flag combination: "
        f"{requested_summary} requested with --validity-profile portable. "
        "Portable validity profile disables expectation writes by default. "
        "Add --allow-portable-expectations-update to enable writes in portable mode, "
        "or use --validity-profile strict."
    )


def _iter_discovered_targets(active_bench_root: Path) -> List[Dict[str, Any]]:
    from core.benchmark.bench_commands import _collect_multi_gpu_examples, _expectation_example_key

    discovered: Dict[str, Dict[str, Any]] = {}
    chapter_dirs = discover_all_chapters(active_bench_root, bench_roots=[active_bench_root])
    for chapter_dir in chapter_dirs:
        chapter_id = chapter_slug(chapter_dir, active_bench_root, bench_root=active_bench_root)
        multi_gpu_examples = _collect_multi_gpu_examples(chapter_dir)
        for baseline_path, _optimized_paths, example_name in discover_benchmarks(chapter_dir):
            bench_type = "cuda" if is_cuda_binary_benchmark_file(baseline_path) else "python"
            example_key = _expectation_example_key(example_name, bench_type)
            target = f"{chapter_id}:{example_name}"
            discovered.setdefault(
                target,
                {
                    "target": target,
                    "chapter": chapter_id,
                    "example": example_name,
                    "bench_type": bench_type,
                    "multi_gpu": bool(multi_gpu_examples.get(example_key, False)),
                },
            )
    return [discovered[key] for key in sorted(discovered)]


def discover_benchmark_e2e_inventory(bench_root: Optional[Path] = None) -> Dict[str, Any]:
    repo_root = _repo_root()
    active_bench_root = Path(bench_root).resolve() if bench_root else repo_root
    discovered = _iter_discovered_targets(active_bench_root)
    single_gpu_targets = sorted(entry["target"] for entry in discovered if not entry.get("multi_gpu"))
    multi_gpu_targets = sorted(entry["target"] for entry in discovered if entry.get("multi_gpu"))
    return {
        "generated_at": _utc_now(),
        "bench_root": str(active_bench_root),
        "targets": discovered,
        "single_gpu": single_gpu_targets,
        "multi_gpu": multi_gpu_targets,
        "counts": {
            "total": len(discovered),
            "single_gpu": len(single_gpu_targets),
            "multi_gpu": len(multi_gpu_targets),
        },
    }


def _invoke_run_tier1_suite(**kwargs: Any) -> Dict[str, Any]:
    from core.benchmark.suites.tier1 import run_tier1_suite

    return run_tier1_suite(**kwargs)


def _invoke_execute_benchmarks(**kwargs: Any) -> Dict[str, Any]:
    from core.benchmark.bench_commands import _execute_benchmarks

    return _execute_benchmarks(**kwargs)


def _invoke_run_cluster_common_eval(**kwargs: Any) -> Dict[str, Any]:
    from core.cluster import run_cluster_common_eval

    return run_cluster_common_eval(**kwargs)


def _invoke_run_cluster_fabric_eval(**kwargs: Any) -> Dict[str, Any]:
    from core.cluster import run_cluster_fabric_eval

    return run_cluster_fabric_eval(**kwargs)


def _result_path_exists(path_value: Optional[str]) -> bool:
    if not path_value:
        return False
    return Path(path_value).exists()


def _benchmark_stage_status(
    result: Dict[str, Any],
    *,
    required_paths: List[str],
) -> Tuple[str, List[str]]:
    issues: List[str] = []
    if result.get("error"):
        issues.append(str(result["error"]))
        return "failed", issues

    missing = [path_key for path_key in required_paths if not _result_path_exists(result.get(path_key))]
    if missing:
        issues.append(f"missing required artifacts: {', '.join(missing)}")
        return "failed", issues

    total_failed = int(result.get("total_failed", 0) or 0)
    if total_failed > 0:
        issues.append(f"{total_failed} benchmark target(s) failed")
        return "failed", issues

    total_skipped = int(result.get("total_skipped", 0) or 0)
    if total_skipped > 0:
        issues.append(f"{total_skipped} benchmark target(s) skipped")
        return "partial", issues

    return "succeeded", issues


def _fabric_scorecard_details(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    run_dir = result.get("run_dir")
    run_id = result.get("run_id")
    if not run_dir or not run_id:
        return None

    scorecard_path = Path(str(run_dir)) / "structured" / f"{run_id}_fabric_scorecard.json"
    if not scorecard_path.exists():
        return None

    try:
        payload = json.loads(scorecard_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "path": str(scorecard_path),
            "status": "error",
            "error": f"Failed to parse fabric scorecard: {exc}",
        }

    families = payload.get("families") or {}
    degraded = []
    for family_name, family_payload in families.items():
        completeness = str((family_payload or {}).get("completeness") or "unknown")
        if completeness in {"not_present", "not_configured"}:
            degraded.append({"family": family_name, "completeness": completeness})
    return {
        "path": str(scorecard_path),
        "status": payload.get("status"),
        "summary": payload.get("summary"),
        "degraded_families": degraded,
    }


def _cluster_stage_status(result: Dict[str, Any]) -> Tuple[str, List[str], Optional[Dict[str, Any]]]:
    issues: List[str] = []
    if not result.get("success", False):
        error = result.get("error") or result.get("stderr") or "cluster stage failed"
        issues.append(str(error))
        return "failed", issues, None

    if not _result_path_exists(result.get("manifest_path")):
        issues.append("missing required artifacts: manifest_path")
        return "failed", issues, None

    scorecard = _fabric_scorecard_details(result)
    if scorecard:
        if scorecard.get("status") not in {None, "ok"}:
            issues.append(str(scorecard.get("error") or "fabric scorecard did not report ok"))
            return "failed", issues, scorecard
        degraded = scorecard.get("degraded_families") or []
        if degraded:
            issues.append("fabric completeness is partial for one or more families")
            return "partial", issues, scorecard

    return "succeeded", issues, scorecard


def _roll_up_overall_status(stage_statuses: List[str]) -> str:
    relevant = [status for status in stage_statuses if status not in {"skipped", "planned"}]
    if not relevant:
        return "succeeded"
    if any(status == "failed" for status in relevant):
        return "failed"
    if any(status == "partial" for status in relevant):
        return "partial"
    if relevant and all(status == "skipped_duplicate" for status in relevant):
        return "skipped_duplicate"
    return "succeeded"


def _stage_entry(
    *,
    name: str,
    enabled: bool,
    stage_run_id: str,
    status: str,
    reason: Optional[str] = None,
    command: Optional[List[str]] = None,
    returncode: Optional[int] = None,
    result: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    issues: Optional[List[str]] = None,
    duration_ms: Optional[int] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "name": name,
        "enabled": enabled,
        "run_id": stage_run_id,
        "status": status,
    }
    if reason:
        payload["reason"] = reason
    if command is not None:
        payload["command"] = command
    if returncode is not None:
        payload["returncode"] = returncode
    if result is not None:
        payload["result"] = result
    if artifacts:
        payload["artifacts"] = artifacts
    if issues:
        payload["issues"] = issues
    if duration_ms is not None:
        payload["duration_ms"] = duration_ms
    return payload


def _planned_stage_entries(
    *,
    run_tier1: bool,
    run_full_sweep: bool,
    run_cluster: bool,
    run_fabric: bool,
    cluster_preset: str,
    stage_run_ids: Dict[str, str],
) -> List[Dict[str, Any]]:
    stages = [
        _stage_entry(
            name="tier1",
            enabled=run_tier1,
            stage_run_id=stage_run_ids["tier1"],
            status="planned" if run_tier1 else "skipped",
            reason=None if run_tier1 else "disabled by flag",
        ),
        _stage_entry(
            name="full_sweep",
            enabled=run_full_sweep,
            stage_run_id=stage_run_ids["full_sweep"],
            status="planned" if run_full_sweep else "skipped",
            reason=None if run_full_sweep else "disabled by flag",
        ),
        _stage_entry(
            name="cluster",
            enabled=run_cluster,
            stage_run_id=stage_run_ids["cluster"],
            status="planned" if run_cluster else "skipped",
            reason=None if run_cluster else "disabled by flag",
        ),
    ]
    fabric_duplicate = run_fabric and run_cluster and cluster_preset.strip().lower() == "fabric-systems"
    fabric_status = "planned" if run_fabric else "skipped"
    fabric_reason = None if run_fabric else "disabled by flag"
    if fabric_duplicate:
        fabric_status = "skipped_duplicate"
        fabric_reason = "cluster preset already includes fabric evaluation"
    stages.append(
        _stage_entry(
            name="fabric",
            enabled=run_fabric,
            stage_run_id=stage_run_ids["fabric"],
            status=fabric_status,
            reason=fabric_reason,
        )
    )
    return stages


def _stage_index(stages: List[Dict[str, Any]], name: str) -> int:
    for idx, stage in enumerate(stages):
        if stage.get("name") == name:
            return idx
    raise KeyError(f"Unknown E2E stage '{name}'")


def _replace_stage(
    stages: List[Dict[str, Any]],
    *,
    name: str,
    status: str,
    reason: Optional[str] = None,
    command: Optional[List[str]] = None,
    returncode: Optional[int] = None,
    result: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    issues: Optional[List[str]] = None,
    duration_ms: Optional[int] = None,
) -> None:
    index = _stage_index(stages, name)
    existing = stages[index]
    stages[index] = _stage_entry(
        name=name,
        enabled=bool(existing.get("enabled")),
        stage_run_id=str(existing.get("run_id")),
        status=status,
        reason=reason,
        command=command,
        returncode=returncode,
        result=result,
        artifacts=artifacts,
        issues=issues,
        duration_ms=duration_ms,
    )


def _enabled_stages(stages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [stage for stage in stages if stage.get("enabled")]


def _completed_enabled_stage_count(stages: List[Dict[str, Any]]) -> int:
    return sum(
        1
        for stage in _enabled_stages(stages)
        if stage.get("status") not in {"planned", "running"}
    )


def _current_stage_name(stages: List[Dict[str, Any]]) -> Optional[str]:
    for stage in _enabled_stages(stages):
        if stage.get("status") == "running":
            return str(stage["name"])
    for stage in _enabled_stages(stages):
        if stage.get("status") == "planned":
            return str(stage["name"])
    return None


def _progress_percent(stages: List[Dict[str, Any]], *, run_state: str) -> float:
    enabled = _enabled_stages(stages)
    if not enabled:
        return 100.0 if run_state == "completed" else 0.0
    if run_state == "completed":
        return 100.0
    return (float(_completed_enabled_stage_count(stages)) / float(len(enabled))) * 100.0


def _emit_live_progress(
    progress_recorder: Optional[ProgressRecorder],
    *,
    stages: List[Dict[str, Any]],
    run_state: str,
    overall_status: str,
    artifact_paths: Dict[str, Path],
) -> None:
    if progress_recorder is None:
        return

    enabled = _enabled_stages(stages)
    total_phases = max(1, len(enabled))
    current_stage = _current_stage_name(stages)
    if current_stage and enabled:
        phase_index = next(
            (idx for idx, stage in enumerate(enabled, start=1) if stage.get("name") == current_stage),
            1,
        )
    elif enabled:
        phase_index = len(enabled)
    else:
        phase_index = 1

    progress_recorder.emit(
        ProgressEvent(
            phase="e2e_sweep",
            phase_index=phase_index,
            total_phases=total_phases,
            step=current_stage or ("complete" if run_state == "completed" else "idle"),
            step_detail=f"run_state={run_state}, overall_status={overall_status}",
            percent_complete=_progress_percent(stages, run_state=run_state),
            artifacts=[str(path) for path in artifact_paths.values()],
            metrics={
                "run_state": run_state,
                "overall_status": overall_status,
                "current_stage": current_stage,
                "completed_stages": _completed_enabled_stage_count(stages),
                "total_stages": len(enabled),
                "stages": _json_safe(stages),
            },
        )
    )


def _render_summary_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# Benchmark E2E Sweep",
        "",
        f"- Run id: `{summary['run_id']}`",
        f"- Overall status: `{summary['overall_status']}`",
        f"- Success: `{summary['success']}`",
        f"- Generated at: `{summary['generated_at']}`",
        "",
        "## Stages",
        "",
        "| Stage | Status | Run id | Notes |",
        "| --- | --- | --- | --- |",
    ]
    for stage in summary.get("stages", []):
        notes = stage.get("reason") or "; ".join(stage.get("issues", [])) or ""
        lines.append(
            f"| `{stage['name']}` | `{stage['status']}` | `{stage['run_id']}` | {notes} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Manifest: `{summary['manifest_path']}`",
            f"- Summary JSON: `{summary['summary_path']}`",
            f"- Summary Markdown: `{summary['summary_markdown_path']}`",
            f"- Progress JSON: `{summary['progress_path']}`",
            f"- Target inventory: `{summary['target_inventory_path']}`",
            f"- Events: `{summary['events_path']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run_benchmark_e2e_sweep(
    *,
    run_tier1: bool = True,
    run_full_sweep: bool = False,
    run_cluster: bool = True,
    run_fabric: bool = True,
    cluster_preset: str = "common-answer-fast",
    hosts: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    ssh_user: Optional[str] = None,
    ssh_key: Optional[str] = None,
    oob_if: Optional[str] = None,
    socket_ifname: Optional[str] = None,
    nccl_ib_hca: Optional[str] = None,
    nmx_url: Optional[str] = None,
    nmx_token: Optional[str] = None,
    ib_mgmt_host: Optional[str] = None,
    ib_mgmt_user: Optional[str] = None,
    ib_mgmt_ssh_key: Optional[str] = None,
    cumulus_hosts: Optional[List[str]] = None,
    cumulus_user: Optional[str] = None,
    cumulus_ssh_key: Optional[str] = None,
    primary_label: Optional[str] = None,
    coverage_baseline_run_id: Optional[str] = None,
    extra_cluster_args: Optional[List[str]] = None,
    bench_root: Optional[Path] = None,
    profile_type: str = "minimal",
    output_format: str = "both",
    suite_timeout: Optional[int] = 14400,
    timeout_multiplier: float = 3.0,
    validity_profile: str = "strict",
    allow_portable_expectations_update: bool = False,
    reproducible: bool = False,
    cold_start: bool = False,
    force_synchronize: bool = False,
    iterations: Optional[int] = None,
    warmup: Optional[int] = None,
    gpu_sm_clock_mhz: Optional[int] = None,
    gpu_mem_clock_mhz: Optional[int] = None,
    artifacts_dir: Optional[str] = None,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    single_gpu: bool = False,
    accept_regressions: bool = False,
    update_expectations: bool = False,
    allow_mixed_provenance: bool = False,
    ncu_metric_set: str = "minimal",
    ncu_replay_mode: Optional[str] = None,
    nsys_timeout_seconds: Optional[int] = None,
    ncu_timeout_seconds: Optional[int] = None,
    timeout_seconds: Optional[int] = None,
    run_id: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    repo_root = _repo_root()
    active_bench_root = Path(bench_root).resolve() if bench_root else repo_root
    normalized_validity_profile = normalize_validity_profile(validity_profile, field_name="validity_profile")
    resolved_run_id = resolve_e2e_run_id(run_id, repo_root=repo_root)
    run_dir = e2e_run_dir(resolved_run_id, repo_root)
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.json"
    summary_markdown_path = run_dir / "summary.md"
    progress_path = e2e_progress_path(run_dir)
    target_inventory_path = run_dir / "target_inventory.json"
    events_path = run_dir / "events.jsonl"
    generated_at = _utc_now()
    progress_recorder = None if dry_run else ProgressRecorder(run_id=resolved_run_id, progress_path=progress_path)
    artifact_paths = {
        "manifest_path": manifest_path,
        "summary_path": summary_path,
        "summary_markdown_path": summary_markdown_path,
        "progress_path": progress_path,
        "target_inventory_path": target_inventory_path,
        "events_path": events_path,
    }
    stage_run_ids = {
        "tier1": f"{resolved_run_id}__tier1",
        "full_sweep": f"{resolved_run_id}__full_sweep",
        "cluster": f"{resolved_run_id}__cluster",
        "fabric": f"{resolved_run_id}__fabric",
    }
    planned_stages = _planned_stage_entries(
        run_tier1=run_tier1,
        run_full_sweep=run_full_sweep,
        run_cluster=run_cluster,
        run_fabric=run_fabric,
        cluster_preset=cluster_preset,
        stage_run_ids=stage_run_ids,
    )

    def _finalize_result(
        payload: Dict[str, Any],
        *,
        persist: bool,
        stages_for_progress: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        safe_payload = _json_safe(payload)
        if persist:
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_json(target_inventory_path, inventory)
            _write_json(summary_path, safe_payload)
            summary_markdown_path.write_text(_render_summary_markdown(safe_payload), encoding="utf-8")
            _write_json(manifest_path, safe_payload)
            _emit_live_progress(
                progress_recorder,
                stages=stages_for_progress or safe_payload.get("stages", planned_stages),
                run_state="completed",
                overall_status=str(safe_payload.get("overall_status", "failed")),
                artifact_paths=artifact_paths,
            )
        return safe_payload

    expectation_error = _validate_expectation_policy(
        validity_profile=normalized_validity_profile,
        allow_portable_expectations_update=allow_portable_expectations_update,
        update_expectations=update_expectations,
        accept_regressions=accept_regressions,
        allow_mixed_provenance=allow_mixed_provenance,
    )
    inventory = discover_benchmark_e2e_inventory(active_bench_root)
    environment = detect_execution_environment()
    try:
        cluster_host_config = _normalize_cluster_hosts_and_labels(
            hosts=hosts,
            labels=labels,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
        )
    except Exception as exc:
        return _finalize_result(
            {
                "success": False,
                "run_id": resolved_run_id,
                "run_dir": str(run_dir),
                "overall_status": "failed",
                "error": str(exc),
                "generated_at": generated_at,
                "manifest_path": str(manifest_path),
                "summary_path": str(summary_path),
                "summary_markdown_path": str(summary_markdown_path),
                "progress_path": str(progress_path),
                "target_inventory_path": str(target_inventory_path),
                "events_path": str(events_path),
                "inventory": inventory,
                "stages": planned_stages,
                "provenance": {
                    "generated_at": generated_at,
                    "git": get_git_info(),
                    "bench_root": str(active_bench_root),
                },
            },
            persist=not dry_run,
            stages_for_progress=planned_stages,
        )
    gpu_count = _visible_gpu_count(single_gpu=single_gpu)
    cluster_extra_args = _with_e2e_cluster_extra_args(extra_cluster_args)
    provenance = {
        "generated_at": generated_at,
        "git": get_git_info(),
        "expectation_hardware_key": detect_expectation_key(),
        "execution_environment": {
            "kind": environment.kind,
            "virtualized": environment.virtualized,
            "dmi_product_name": environment.dmi_product_name,
        },
        "gpu_count": gpu_count,
        "bench_root": str(active_bench_root),
    }

    if expectation_error:
        result = {
            "success": False,
            "run_id": resolved_run_id,
            "run_dir": str(run_dir),
            "overall_status": "failed",
            "error": expectation_error,
            "generated_at": generated_at,
            "manifest_path": str(manifest_path),
            "summary_path": str(summary_path),
            "summary_markdown_path": str(summary_markdown_path),
            "progress_path": str(progress_path),
            "target_inventory_path": str(target_inventory_path),
            "events_path": str(events_path),
            "inventory": inventory,
            "stages": planned_stages,
            "provenance": provenance,
        }
        if not dry_run:
            _append_event(events_path, "run_failed_preflight", error=expectation_error, run_id=resolved_run_id)
        return _finalize_result(result, persist=not dry_run, stages_for_progress=planned_stages)

    stages: List[Dict[str, Any]] = [dict(stage) for stage in planned_stages]
    if dry_run:
        return _json_safe({
            "success": True,
            "dry_run": True,
            "run_id": resolved_run_id,
            "run_dir": str(run_dir),
            "overall_status": "dry_run",
            "generated_at": generated_at,
            "manifest_path": str(manifest_path),
            "summary_path": str(summary_path),
            "summary_markdown_path": str(summary_markdown_path),
            "progress_path": str(progress_path),
            "target_inventory_path": str(target_inventory_path),
            "events_path": str(events_path),
            "inventory": inventory,
            "hosts": cluster_host_config,
            "provenance": provenance,
            "stages": stages,
        })

    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(target_inventory_path, inventory)
    _append_event(events_path, "run_started", run_id=resolved_run_id)
    _emit_live_progress(
        progress_recorder,
        stages=stages,
        run_state="running",
        overall_status=_roll_up_overall_status([stage["status"] for stage in stages if stage.get("enabled")]),
        artifact_paths=artifact_paths,
    )

    if run_tier1:
        started = time.monotonic()
        _replace_stage(stages, name="tier1", status="running")
        _append_event(events_path, "stage_started", stage="tier1", run_id=stage_run_ids["tier1"])
        _emit_live_progress(
            progress_recorder,
            stages=stages,
            run_state="running",
            overall_status=_roll_up_overall_status([stage["status"] for stage in stages if stage.get("enabled")]),
            artifact_paths=artifact_paths,
        )
        tier1_command = [
            "python",
            "-m",
            "cli.aisp",
            "bench",
            "run-tier1",
            "--run-id",
            stage_run_ids["tier1"],
            "--profile",
            profile_type,
            "--validity-profile",
            normalized_validity_profile,
        ]
        with _benchmark_queue_lock("tier1", stage_run_ids["tier1"], repo_root=repo_root):
            tier1_result = _invoke_run_tier1_suite(
                bench_root=active_bench_root,
                profile_type=profile_type,
                output_format=output_format,
                suite_timeout=suite_timeout,
                timeout_multiplier=timeout_multiplier,
                validity_profile=normalized_validity_profile,
                allow_portable_expectations_update=allow_portable_expectations_update,
                reproducible=reproducible,
                cold_start=cold_start,
                force_synchronize=force_synchronize,
                iterations=iterations,
                warmup=warmup,
                gpu_sm_clock_mhz=gpu_sm_clock_mhz,
                gpu_mem_clock_mhz=gpu_mem_clock_mhz,
                artifacts_dir=artifacts_dir,
                run_id=stage_run_ids["tier1"],
                log_level=log_level,
                log_file=log_file,
                single_gpu=single_gpu,
                accept_regressions=accept_regressions,
                update_expectations=update_expectations,
                allow_mixed_provenance=allow_mixed_provenance,
                ncu_metric_set=ncu_metric_set,
                ncu_replay_mode=ncu_replay_mode,
                nsys_timeout_seconds=nsys_timeout_seconds,
                ncu_timeout_seconds=ncu_timeout_seconds,
            )
        tier1_status, tier1_issues = _benchmark_stage_status(
            tier1_result,
            required_paths=["summary_path", "regression_summary_path", "trend_snapshot_path"],
        )
        tier1_returncode = 1 if tier1_status == "failed" else 0
        _replace_stage(
            stages,
            name="tier1",
            status=tier1_status,
            command=tier1_command,
            returncode=tier1_returncode,
            result=tier1_result,
            artifacts={
                "summary_path": tier1_result.get("summary_path"),
                "regression_summary_path": tier1_result.get("regression_summary_path"),
                "trend_snapshot_path": tier1_result.get("trend_snapshot_path"),
                "history_root": tier1_result.get("history_root"),
            },
            issues=tier1_issues,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        _append_event(events_path, "stage_finished", stage="tier1", status=tier1_status, run_id=stage_run_ids["tier1"])
        _emit_live_progress(
            progress_recorder,
            stages=stages,
            run_state="running",
            overall_status=_roll_up_overall_status([stage["status"] for stage in stages if stage.get("enabled")]),
            artifact_paths=artifact_paths,
        )

    if run_full_sweep:
        started = time.monotonic()
        _replace_stage(stages, name="full_sweep", status="running")
        _append_event(events_path, "stage_started", stage="full_sweep", run_id=stage_run_ids["full_sweep"])
        _emit_live_progress(
            progress_recorder,
            stages=stages,
            run_state="running",
            overall_status=_roll_up_overall_status([stage["status"] for stage in stages if stage.get("enabled")]),
            artifact_paths=artifact_paths,
        )
        full_stage_issues: List[str] = []
        bucket_results: Dict[str, Any] = {}
        single_targets = list(inventory.get("single_gpu", []))
        multi_targets = list(inventory.get("multi_gpu", []))
        if not single_targets and not multi_targets:
            full_status = "failed"
            full_stage_issues.append("no benchmark targets discovered for full sweep")
        else:
            full_status = "succeeded"
            if single_targets:
                single_run_id = f"{stage_run_ids['full_sweep']}__single"
                single_command = [
                    "python",
                    "-m",
                    "cli.aisp",
                    "bench",
                    "run",
                    "--run-id",
                    single_run_id,
                    "--profile",
                    profile_type,
                    "--validity-profile",
                    normalized_validity_profile,
                    *sum([["-t", target] for target in single_targets], []),
                ]
                with _benchmark_queue_lock("full_sweep_single", single_run_id, repo_root=repo_root):
                    single_result = _invoke_execute_benchmarks(
                        targets=single_targets,
                        bench_root=active_bench_root,
                        output_format=output_format,
                        profile_type=profile_type,
                        suite_timeout=suite_timeout,
                        timeout_multiplier=timeout_multiplier,
                        validity_profile=normalized_validity_profile,
                        allow_portable_expectations_update=allow_portable_expectations_update,
                        reproducible=reproducible,
                        cold_start=cold_start,
                        force_synchronize=force_synchronize,
                        iterations=iterations,
                        warmup=warmup,
                        gpu_sm_clock_mhz=gpu_sm_clock_mhz,
                        gpu_mem_clock_mhz=gpu_mem_clock_mhz,
                        artifacts_dir=artifacts_dir,
                        run_id=single_run_id,
                        log_level=log_level,
                        log_file=log_file,
                        single_gpu=single_gpu,
                        accept_regressions=accept_regressions,
                        update_expectations=update_expectations,
                        allow_mixed_provenance=allow_mixed_provenance,
                        ncu_metric_set=ncu_metric_set,
                        ncu_replay_mode=ncu_replay_mode,
                        nsys_timeout_seconds=nsys_timeout_seconds,
                        ncu_timeout_seconds=ncu_timeout_seconds,
                        exit_on_failure=False,
                    )
                single_status, single_issues = _benchmark_stage_status(single_result, required_paths=["output_json"])
                bucket_results["single_gpu"] = {
                    "targets": single_targets,
                    "run_id": single_run_id,
                    "command": single_command,
                    "status": single_status,
                    "issues": single_issues,
                    "result": single_result,
                }
                if single_status == "failed":
                    full_status = "failed"
                elif single_status == "partial" and full_status != "failed":
                    full_status = "partial"
                full_stage_issues.extend(single_issues)

            if multi_targets:
                if gpu_count < 2:
                    bucket_results["multi_gpu"] = {
                        "targets": multi_targets,
                        "run_id": f"{stage_run_ids['full_sweep']}__multi",
                        "status": "skipped",
                        "reason": f"requires >=2 visible GPUs; detected {gpu_count}",
                    }
                    if full_status != "failed":
                        full_status = "partial"
                    full_stage_issues.append(f"multi-GPU bucket skipped because only {gpu_count} visible GPU(s) were detected")
                else:
                    multi_run_id = f"{stage_run_ids['full_sweep']}__multi"
                    multi_command = [
                        "python",
                        "-m",
                        "cli.aisp",
                        "bench",
                        "run",
                        "--run-id",
                        multi_run_id,
                        "--profile",
                        profile_type,
                        "--validity-profile",
                        normalized_validity_profile,
                        *sum([["-t", target] for target in multi_targets], []),
                    ]
                    with _benchmark_queue_lock("full_sweep_multi", multi_run_id, repo_root=repo_root):
                        multi_result = _invoke_execute_benchmarks(
                            targets=multi_targets,
                            bench_root=active_bench_root,
                            output_format=output_format,
                            profile_type=profile_type,
                            suite_timeout=suite_timeout,
                            timeout_multiplier=timeout_multiplier,
                            validity_profile=normalized_validity_profile,
                            allow_portable_expectations_update=allow_portable_expectations_update,
                            reproducible=reproducible,
                            cold_start=cold_start,
                            force_synchronize=force_synchronize,
                            iterations=iterations,
                            warmup=warmup,
                            gpu_sm_clock_mhz=gpu_sm_clock_mhz,
                            gpu_mem_clock_mhz=gpu_mem_clock_mhz,
                            artifacts_dir=artifacts_dir,
                            run_id=multi_run_id,
                            log_level=log_level,
                            log_file=log_file,
                            single_gpu=single_gpu,
                            accept_regressions=accept_regressions,
                            update_expectations=update_expectations,
                            allow_mixed_provenance=allow_mixed_provenance,
                            ncu_metric_set=ncu_metric_set,
                            ncu_replay_mode=ncu_replay_mode,
                            nsys_timeout_seconds=nsys_timeout_seconds,
                            ncu_timeout_seconds=ncu_timeout_seconds,
                            exit_on_failure=False,
                        )
                    multi_status, multi_issues = _benchmark_stage_status(multi_result, required_paths=["output_json"])
                    bucket_results["multi_gpu"] = {
                        "targets": multi_targets,
                        "run_id": multi_run_id,
                        "command": multi_command,
                        "status": multi_status,
                        "issues": multi_issues,
                        "result": multi_result,
                    }
                    if multi_status == "failed":
                        full_status = "failed"
                    elif multi_status == "partial" and full_status != "failed":
                        full_status = "partial"
                    full_stage_issues.extend(multi_issues)

        _replace_stage(
            stages,
            name="full_sweep",
            status=full_status,
            returncode=1 if full_status == "failed" else 0,
            result={"buckets": bucket_results},
            artifacts={"target_inventory_path": str(target_inventory_path)},
            issues=full_stage_issues,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        _append_event(events_path, "stage_finished", stage="full_sweep", status=full_status, run_id=stage_run_ids["full_sweep"])
        _emit_live_progress(
            progress_recorder,
            stages=stages,
            run_state="running",
            overall_status=_roll_up_overall_status([stage["status"] for stage in stages if stage.get("enabled")]),
            artifact_paths=artifact_paths,
        )

    if run_cluster:
        started = time.monotonic()
        _replace_stage(stages, name="cluster", status="running")
        _append_event(events_path, "stage_started", stage="cluster", run_id=stage_run_ids["cluster"])
        _emit_live_progress(
            progress_recorder,
            stages=stages,
            run_state="running",
            overall_status=_roll_up_overall_status([stage["status"] for stage in stages if stage.get("enabled")]),
            artifact_paths=artifact_paths,
        )
        cluster_result = _invoke_run_cluster_common_eval(
            preset=cluster_preset,
            run_id=stage_run_ids["cluster"],
            hosts=cluster_host_config["hosts"],
            labels=cluster_host_config["labels"],
            ssh_user=cluster_host_config["ssh_user"],
            ssh_key=cluster_host_config["ssh_key"],
            oob_if=oob_if,
            socket_ifname=socket_ifname,
            nccl_ib_hca=nccl_ib_hca,
            nmx_url=nmx_url,
            nmx_token=nmx_token,
            ib_mgmt_host=ib_mgmt_host,
            ib_mgmt_user=ib_mgmt_user,
            ib_mgmt_ssh_key=ib_mgmt_ssh_key,
            cumulus_hosts=cumulus_hosts,
            cumulus_user=cumulus_user,
            cumulus_ssh_key=cumulus_ssh_key,
            primary_label=primary_label,
            coverage_baseline_run_id=coverage_baseline_run_id,
            extra_args=cluster_extra_args,
            timeout_seconds=timeout_seconds,
        )
        cluster_status, cluster_issues, cluster_scorecard = _cluster_stage_status(cluster_result)
        _replace_stage(
            stages,
            name="cluster",
            status=cluster_status,
            command=cluster_result.get("command"),
            returncode=int(cluster_result.get("returncode", 0) or 0),
            result=cluster_result,
            artifacts={
                "run_dir": cluster_result.get("run_dir"),
                "manifest_path": cluster_result.get("manifest_path"),
                "fabric_scorecard": cluster_scorecard,
            },
            issues=cluster_issues,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        _append_event(events_path, "stage_finished", stage="cluster", status=cluster_status, run_id=stage_run_ids["cluster"])
        _emit_live_progress(
            progress_recorder,
            stages=stages,
            run_state="running",
            overall_status=_roll_up_overall_status([stage["status"] for stage in stages if stage.get("enabled")]),
            artifact_paths=artifact_paths,
        )

    fabric_duplicate = run_fabric and run_cluster and cluster_preset.strip().lower() == "fabric-systems"
    if fabric_duplicate:
        _emit_live_progress(
            progress_recorder,
            stages=stages,
            run_state="running",
            overall_status=_roll_up_overall_status([stage["status"] for stage in stages if stage.get("enabled")]),
            artifact_paths=artifact_paths,
        )
    elif run_fabric:
        started = time.monotonic()
        _replace_stage(stages, name="fabric", status="running")
        _append_event(events_path, "stage_started", stage="fabric", run_id=stage_run_ids["fabric"])
        _emit_live_progress(
            progress_recorder,
            stages=stages,
            run_state="running",
            overall_status=_roll_up_overall_status([stage["status"] for stage in stages if stage.get("enabled")]),
            artifact_paths=artifact_paths,
        )
        fabric_result = _invoke_run_cluster_fabric_eval(
            run_id=stage_run_ids["fabric"],
            hosts=cluster_host_config["hosts"],
            labels=cluster_host_config["labels"],
            ssh_user=cluster_host_config["ssh_user"],
            ssh_key=cluster_host_config["ssh_key"],
            oob_if=oob_if,
            socket_ifname=socket_ifname,
            nccl_ib_hca=nccl_ib_hca,
            nmx_url=nmx_url,
            nmx_token=nmx_token,
            ib_mgmt_host=ib_mgmt_host,
            ib_mgmt_user=ib_mgmt_user,
            ib_mgmt_ssh_key=ib_mgmt_ssh_key,
            cumulus_hosts=cumulus_hosts,
            cumulus_user=cumulus_user,
            cumulus_ssh_key=cumulus_ssh_key,
            primary_label=primary_label,
            coverage_baseline_run_id=coverage_baseline_run_id,
            extra_args=cluster_extra_args,
            timeout_seconds=timeout_seconds,
        )
        fabric_status, fabric_issues, fabric_scorecard = _cluster_stage_status(fabric_result)
        _replace_stage(
            stages,
            name="fabric",
            status=fabric_status,
            command=fabric_result.get("command"),
            returncode=int(fabric_result.get("returncode", 0) or 0),
            result=fabric_result,
            artifacts={
                "run_dir": fabric_result.get("run_dir"),
                "manifest_path": fabric_result.get("manifest_path"),
                "fabric_scorecard": fabric_scorecard,
            },
            issues=fabric_issues,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        _append_event(events_path, "stage_finished", stage="fabric", status=fabric_status, run_id=stage_run_ids["fabric"])
        _emit_live_progress(
            progress_recorder,
            stages=stages,
            run_state="running",
            overall_status=_roll_up_overall_status([stage["status"] for stage in stages if stage.get("enabled")]),
            artifact_paths=artifact_paths,
        )

    overall_status = _roll_up_overall_status([stage["status"] for stage in stages if stage.get("enabled")])
    success = overall_status != "failed"
    summary = {
        "success": success,
        "run_id": resolved_run_id,
        "run_dir": str(run_dir),
        "overall_status": overall_status,
        "generated_at": provenance["generated_at"],
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "summary_markdown_path": str(summary_markdown_path),
        "progress_path": str(progress_path),
        "target_inventory_path": str(target_inventory_path),
        "events_path": str(events_path),
        "inventory": {
            "counts": inventory["counts"],
            "single_gpu": inventory["single_gpu"],
            "multi_gpu": inventory["multi_gpu"],
        },
        "hosts": cluster_host_config,
        "provenance": provenance,
        "stages": stages,
    }
    manifest = {
        **summary,
        "inventory": inventory,
    }
    _append_event(events_path, "run_finished", run_id=resolved_run_id, overall_status=overall_status, success=success)
    _write_json(summary_path, _json_safe(summary))
    summary_markdown_path.write_text(_render_summary_markdown(_json_safe(summary)), encoding="utf-8")
    _write_json(manifest_path, _json_safe(manifest))
    _emit_live_progress(
        progress_recorder,
        stages=stages,
        run_state="completed",
        overall_status=overall_status,
        artifact_paths=artifact_paths,
    )
    return _json_safe(summary)
