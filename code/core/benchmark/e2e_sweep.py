"""End-to-end benchmark orchestration for tier1, full sweep, and cluster eval."""

from __future__ import annotations

import contextlib
import getpass
import json
import os
import signal
import socket
import threading
import time
import traceback
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

_STAGE_PROGRESS_POLL_SECONDS = 2.0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def e2e_runs_root(repo_root: Optional[Path] = None) -> Path:
    return Path(repo_root or _repo_root()) / "artifacts" / "e2e_runs"


def e2e_run_dir(run_id: str, repo_root: Optional[Path] = None) -> Path:
    return e2e_runs_root(repo_root) / run_id


def e2e_progress_path(run_dir: Path) -> Path:
    return Path(run_dir) / "progress.json"


def e2e_checkpoint_path(run_dir: Path) -> Path:
    return Path(run_dir) / "checkpoint.json"


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


def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _pid_is_live(pid: Optional[int]) -> bool:
    if pid is None:
        return False
    try:
        pid_int = int(pid)
    except Exception:
        return False
    if pid_int <= 0:
        return False
    try:
        os.kill(pid_int, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


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


def _group_targets_by_unit(targets: List[str]) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    index_by_name: Dict[str, int] = {}
    for target in targets:
        unit_name = _canonical_unit_name(str(target).split(":", 1)[0].strip() or str(target).strip())
        if unit_name not in index_by_name:
            index_by_name[unit_name] = len(units)
            units.append({"name": unit_name, "targets": []})
        units[index_by_name[unit_name]]["targets"].append(str(target))
    return units


def _canonical_unit_name(unit_name: Optional[str]) -> str:
    value = str(unit_name or "").strip()
    if not value:
        return ""
    if value.startswith("labs_"):
        return f"labs/{value[len('labs_'):]}"
    return value


def _completed_units_from_attempts(attempts: List[Dict[str, Any]], *, ordered_units: List[str]) -> List[str]:
    ordered_units = [_canonical_unit_name(unit) for unit in ordered_units]
    completed_lookup = {
        _canonical_unit_name(unit)
        for attempt in attempts
        for unit in (attempt.get("completed_units") or [])
        if _canonical_unit_name(unit)
    }
    return [unit for unit in ordered_units if unit in completed_lookup]


def _remaining_targets_after_completed_units(
    targets: List[str],
    *,
    completed_units: List[str],
) -> List[str]:
    grouped_units = _group_targets_by_unit(targets)
    completed_lookup = {_canonical_unit_name(unit) for unit in completed_units if _canonical_unit_name(unit)}
    first_incomplete_index: Optional[int] = None
    for index, unit in enumerate(grouped_units):
        if unit["name"] not in completed_lookup:
            first_incomplete_index = index
            break
    if first_incomplete_index is None:
        return []
    remaining: List[str] = []
    for unit in grouped_units[first_incomplete_index:]:
        remaining.extend([str(target) for target in unit.get("targets", [])])
    return remaining


def _remaining_units_after_completed_units(
    ordered_units: List[str],
    *,
    completed_units: List[str],
) -> List[str]:
    ordered_units = [_canonical_unit_name(unit) for unit in ordered_units]
    completed_lookup = {_canonical_unit_name(unit) for unit in completed_units if _canonical_unit_name(unit)}
    first_incomplete_index: Optional[int] = None
    for index, unit in enumerate(ordered_units):
        if unit not in completed_lookup:
            first_incomplete_index = index
            break
    if first_incomplete_index is None:
        return []
    return [str(unit) for unit in ordered_units[first_incomplete_index:]]


def _resolve_targets_for_units(
    available_targets: List[str],
    *,
    ordered_units: List[str],
) -> Tuple[List[str], List[str]]:
    grouped_targets: Dict[str, List[str]] = {}
    for target in available_targets:
        unit_name = _canonical_unit_name(str(target).split(":", 1)[0].strip() or str(target).strip())
        grouped_targets.setdefault(unit_name, []).append(str(target))

    resolved_targets: List[str] = []
    missing_units: List[str] = []
    for unit_name in ordered_units:
        canonical_unit_name = _canonical_unit_name(unit_name)
        matches = grouped_targets.get(canonical_unit_name, [])
        if not matches:
            missing_units.append(canonical_unit_name)
            continue
        resolved_targets.extend(matches)
    return resolved_targets, missing_units


def _benchmark_stage_details_from_output(output_json_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not output_json_path or not Path(output_json_path).exists():
        return None
    try:
        payload = json.loads(Path(output_json_path).read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    status_counts: Dict[str, int] = {}
    failed_benchmarks: List[Dict[str, Any]] = []
    skipped_benchmarks: List[Dict[str, Any]] = []
    results = payload.get("results") or []
    for chapter_entry in results:
        chapter_name = str((chapter_entry or {}).get("chapter") or "").strip() or None
        for benchmark in (chapter_entry or {}).get("benchmarks", []) or []:
            status = str((benchmark or {}).get("status") or "unknown").strip() or "unknown"
            status_counts[status] = status_counts.get(status, 0) + 1
            example = str((benchmark or {}).get("example") or "").strip()
            target = f"{chapter_name}:{example}" if chapter_name and example else example or chapter_name or "<unknown>"
            if status.startswith("failed"):
                error_detail = str(
                    (benchmark or {}).get("error")
                    or (benchmark or {}).get("failure_reason")
                    or "benchmark target failed"
                )
                failed_benchmarks.append(
                    {
                        "target": target,
                        "status": status,
                        "error": error_detail,
                    }
                )
            elif status == "skipped":
                skip_reason = str(
                    (benchmark or {}).get("error")
                    or (benchmark or {}).get("skip_reason")
                    or "benchmark target skipped"
                )
                skipped_benchmarks.append(
                    {
                        "target": target,
                        "status": status,
                        "reason": skip_reason,
                    }
                )

    return {
        "status_counts": status_counts,
        "failed_benchmarks": failed_benchmarks,
        "skipped_benchmarks": skipped_benchmarks,
    }


def _benchmark_stage_status(
    result: Dict[str, Any],
    *,
    required_paths: List[str],
) -> Tuple[str, List[str], Optional[Dict[str, Any]]]:
    issues: List[str] = []
    if result.get("error"):
        issues.append(str(result["error"]))
        return "failed", issues, None

    missing = [path_key for path_key in required_paths if not _result_path_exists(result.get(path_key))]
    if missing:
        issues.append(f"missing required artifacts: {', '.join(missing)}")
        return "failed", issues, None

    benchmark_details = _benchmark_stage_details_from_output(result.get("output_json"))
    if benchmark_details is not None:
        total_failed = sum(
            count
            for status, count in (benchmark_details.get("status_counts") or {}).items()
            if str(status).startswith("failed")
        )
        total_skipped = int((benchmark_details.get("status_counts") or {}).get("skipped", 0) or 0)
    else:
        total_failed = int(result.get("total_failed", 0) or 0)
        total_skipped = int(result.get("total_skipped", 0) or 0)

    if total_failed > 0:
        failed_benchmarks = (benchmark_details or {}).get("failed_benchmarks") or []
        if failed_benchmarks:
            issues.extend(
                f"{entry['target']}: {entry['error']}"
                for entry in failed_benchmarks
            )
        else:
            issues.append(f"{total_failed} benchmark target(s) failed")
        return "failed", issues, benchmark_details

    if total_skipped > 0:
        issues.append(f"{total_skipped} benchmark target(s) skipped")
        return "partial", issues, benchmark_details

    return "succeeded", issues, benchmark_details


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
        scorecard_status = str(scorecard.get("status") or "").strip().lower()
        if scorecard_status in {"error", "failed"}:
            issues.append(str(scorecard.get("error") or "fabric scorecard reported a fatal status"))
            return "failed", issues, scorecard
        degraded = scorecard.get("degraded_families") or []
        if scorecard_status == "partial":
            if degraded:
                issues.append("fabric completeness is partial for one or more families")
            else:
                issues.append("fabric scorecard reported partial runtime verification")
            return "partial", issues, scorecard
        if scorecard_status not in {"", "ok"}:
            issues.append(str(scorecard.get("error") or f"unexpected fabric scorecard status: {scorecard_status}"))
            return "failed", issues, scorecard
        if degraded:
            issues.append("fabric completeness is partial for one or more families")
            return "partial", issues, scorecard

    return "succeeded", issues, scorecard


def _roll_up_overall_status(stage_statuses: List[str]) -> str:
    relevant = [status for status in stage_statuses if status not in {"skipped", "planned"}]
    if not relevant:
        return "succeeded"
    if any(status == "aborted" for status in relevant):
        return "aborted"
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
    attempts: Optional[List[Dict[str, Any]]] = None,
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
    if attempts is not None:
        payload["attempts"] = attempts
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
    attempts: Optional[List[Dict[str, Any]]] = None,
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
        attempts=existing.get("attempts") if attempts is None else attempts,
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


def _progress_percent(
    stages: List[Dict[str, Any]],
    *,
    run_state: str,
    child_percent: Optional[float] = None,
) -> float:
    enabled = _enabled_stages(stages)
    if not enabled:
        return 100.0 if run_state == "completed" else 0.0
    if run_state == "completed":
        return 100.0
    completed = float(_completed_enabled_stage_count(stages))
    if child_percent is not None:
        bounded_child = max(0.0, min(100.0, float(child_percent)))
        return ((completed + (bounded_child / 100.0)) / float(len(enabled))) * 100.0
    return (completed / float(len(enabled))) * 100.0


def _load_progress_current(progress_path: Path) -> Optional[Dict[str, Any]]:
    payload = _read_json_if_exists(progress_path)
    if payload is None:
        return None
    current = payload.get("current")
    if not isinstance(current, dict):
        return None
    return current


def _emit_live_progress(
    progress_recorder: Optional[ProgressRecorder],
    *,
    stages: List[Dict[str, Any]],
    run_state: str,
    overall_status: str,
    artifact_paths: Dict[str, Path],
    emit_lock: Optional[threading.Lock] = None,
    child_progress: Optional[Dict[str, Any]] = None,
    child_stage_name: Optional[str] = None,
    child_run_id: Optional[str] = None,
    child_bucket: Optional[str] = None,
    orchestrator_pid: Optional[int] = None,
) -> None:
    if progress_recorder is None:
        return

    enabled = _enabled_stages(stages)
    total_phases = max(1, len(enabled))
    current_stage = child_stage_name or _current_stage_name(stages)
    if current_stage and enabled:
        phase_index = next(
            (idx for idx, stage in enumerate(enabled, start=1) if stage.get("name") == current_stage),
            1,
        )
    elif enabled:
        phase_index = len(enabled)
    else:
        phase_index = 1

    child_percent = None
    step = current_stage or ("complete" if run_state == "completed" else "idle")
    step_detail = f"run_state={run_state}, overall_status={overall_status}"
    if child_progress and run_state == "running" and current_stage:
        child_step = str(child_progress.get("step") or "").strip()
        child_detail = str(child_progress.get("step_detail") or "").strip()
        raw_child_percent = child_progress.get("percent_complete")
        if isinstance(raw_child_percent, (int, float)):
            child_percent = float(raw_child_percent)
        step_prefix = current_stage if not child_bucket else f"{current_stage}/{child_bucket}"
        if child_step:
            step = f"{step_prefix}:{child_step}"
        else:
            step = step_prefix
        if child_detail:
            step_detail = child_detail

    event = ProgressEvent(
        phase="e2e_sweep",
        phase_index=phase_index,
        total_phases=total_phases,
        step=step,
        step_detail=step_detail,
        percent_complete=_progress_percent(stages, run_state=run_state, child_percent=child_percent),
        artifacts=[str(path) for path in artifact_paths.values()],
        metrics={
            "run_state": run_state,
            "overall_status": overall_status,
            "current_stage": current_stage,
            "current_stage_run_id": child_run_id,
            "current_bucket": child_bucket,
            "orchestrator_pid": orchestrator_pid,
            "completed_stages": _completed_enabled_stage_count(stages),
            "total_stages": len(enabled),
            "stages": _json_safe(stages),
            "child_progress": _json_safe(child_progress) if child_progress is not None else None,
        },
    )
    if emit_lock is None:
        progress_recorder.emit(event)
        return
    with emit_lock:
        progress_recorder.emit(event)


def _render_summary_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# Benchmark E2E Sweep",
        "",
        f"- Run id: `{summary['run_id']}`",
        f"- Overall status: `{summary['overall_status']}`",
        f"- Success: `{summary['success']}`",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Resume available: `{summary.get('resume_available', False)}`",
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
            f"- Checkpoint JSON: `{summary['checkpoint_path']}`",
            f"- Target inventory: `{summary['target_inventory_path']}`",
            f"- Events: `{summary['events_path']}`",
        ]
    )
    historical_failure_ledger = summary.get("historical_failure_ledger")
    if isinstance(historical_failure_ledger, dict):
        ledger_summary = historical_failure_ledger.get("summary") or {}
        lines.extend(
            [
                "",
                "## Historical Failure Ledger",
                "",
                f"- Ledger JSON: `{historical_failure_ledger.get('json_path', '')}`",
                f"- Ledger Markdown: `{historical_failure_ledger.get('markdown_path', '')}`",
                f"- Total historical failures: `{ledger_summary.get('total_historical_failures', 0)}`",
                f"- Rechecked: `{ledger_summary.get('rechecked_count', 0)}`",
                f"- Resolved success: `{ledger_summary.get('resolved_success_count', 0)}`",
                f"- Resolved skip: `{ledger_summary.get('resolved_skip_count', 0)}`",
                f"- Still failing: `{ledger_summary.get('still_failing_count', 0)}`",
                f"- Not rerun: `{ledger_summary.get('not_rerun_count', 0)}`",
            ]
        )
    return "\n".join(lines) + "\n"


def _stage_attempt_entry(
    *,
    run_id: str,
    bucket: Optional[str] = None,
    status: str,
    targets: Optional[List[str]] = None,
    units: Optional[List[str]] = None,
    completed_units: Optional[List[str]] = None,
    active_unit: Optional[str] = None,
    reason: Optional[str] = None,
    command: Optional[List[str]] = None,
    returncode: Optional[int] = None,
    result: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    issues: Optional[List[str]] = None,
    duration_ms: Optional[int] = None,
    benchmark_summary: Optional[Dict[str, Any]] = None,
    started_at: Optional[str] = None,
    ended_at: Optional[str] = None,
    recovered: bool = False,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "run_id": run_id,
        "status": status,
        "recovered": recovered,
    }
    if bucket:
        payload["bucket"] = bucket
    if targets is not None:
        payload["targets"] = list(targets)
    if units is not None:
        payload["units"] = list(units)
    if completed_units is not None:
        payload["completed_units"] = list(completed_units)
    if active_unit:
        payload["active_unit"] = active_unit
    if reason:
        payload["reason"] = reason
    if command is not None:
        payload["command"] = list(command)
    if returncode is not None:
        payload["returncode"] = int(returncode)
    if result is not None:
        payload["result"] = result
    if artifacts:
        payload["artifacts"] = artifacts
    if issues:
        payload["issues"] = list(issues)
    if duration_ms is not None:
        payload["duration_ms"] = int(duration_ms)
    if benchmark_summary is not None:
        payload["benchmark_summary"] = benchmark_summary
    if started_at:
        payload["started_at"] = started_at
    if ended_at:
        payload["ended_at"] = ended_at
    return payload


def _build_e2e_contract(**kwargs: Any) -> Dict[str, Any]:
    return _json_safe(kwargs)


def _find_stage(stages: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    for stage in stages:
        if stage.get("name") == name:
            stage.setdefault("attempts", [])
            return stage
    raise KeyError(f"Unknown E2E stage '{name}'")


def _compute_stage_status_from_attempts(stage: Dict[str, Any]) -> str:
    attempts = list(stage.get("attempts") or [])
    if attempts:
        latest_status = str(attempts[-1].get("status") or "").strip()
        if latest_status:
            return latest_status
    return str(stage.get("status") or "planned")


def _bucket_attempt_run_id(stage_base_run_id: str, bucket: str, attempt_index: int) -> str:
    bucket_suffix = "single" if bucket == "single_gpu" else "multi"
    base_run_id = f"{stage_base_run_id}__{bucket_suffix}"
    if attempt_index <= 0:
        return base_run_id
    return f"{base_run_id}__resume{attempt_index}"


def _stage_attempt_run_id(stage_base_run_id: str, attempt_index: int) -> str:
    if attempt_index <= 0:
        return stage_base_run_id
    return f"{stage_base_run_id}__resume{attempt_index}"


def _bucket_attempts(stage: Dict[str, Any], bucket: str) -> List[Dict[str, Any]]:
    return [attempt for attempt in (stage.get("attempts") or []) if attempt.get("bucket") == bucket]


def _benchmark_run_dir(
    run_id: str,
    *,
    repo_root: Path,
    artifacts_dir: Optional[str],
) -> Path:
    artifacts_root = Path(artifacts_dir).resolve() if artifacts_dir else (repo_root / "artifacts" / "runs")
    return artifacts_root / run_id


def _benchmark_run_event_paths(
    run_id: str,
    *,
    repo_root: Path,
    artifacts_dir: Optional[str],
) -> Dict[str, Path]:
    run_dir = _benchmark_run_dir(run_id, repo_root=repo_root, artifacts_dir=artifacts_dir)
    return {
        "run_dir": run_dir,
        "events": run_dir / "logs" / "benchmark_events.jsonl",
        "output_json": run_dir / "results" / "benchmark_test_results.json",
        "progress": run_dir / "progress" / "run_progress.json",
    }


def _cluster_run_progress_path(
    run_id: str,
    *,
    repo_root: Path,
) -> Path:
    return repo_root / "cluster" / "runs" / run_id / "progress" / "run_progress.json"


def _load_benchmark_run_start(
    run_id: str,
    *,
    repo_root: Path,
    artifacts_dir: Optional[str],
) -> Optional[Dict[str, Any]]:
    events_path = _benchmark_run_event_paths(run_id, repo_root=repo_root, artifacts_dir=artifacts_dir)["events"]
    for payload in _read_jsonl(events_path):
        if payload.get("event_type") == "run_start":
            return payload
    return None


def _load_benchmark_unit_progress(
    run_id: str,
    *,
    repo_root: Path,
    artifacts_dir: Optional[str],
) -> Dict[str, Any]:
    events_path = _benchmark_run_event_paths(run_id, repo_root=repo_root, artifacts_dir=artifacts_dir)["events"]
    completed_units: List[str] = []
    started_units: List[str] = []
    for payload in _read_jsonl(events_path):
        event_type = str(payload.get("event_type") or "")
        unit_name = _canonical_unit_name(payload.get("chapter"))
        if not unit_name:
            continue
        if event_type == "chapter_start" and unit_name not in started_units:
            started_units.append(unit_name)
        elif event_type == "chapter_end" and unit_name not in completed_units:
            completed_units.append(unit_name)
    active_unit = None
    for unit_name in started_units:
        if unit_name not in completed_units:
            active_unit = unit_name
            break
    return {
        "completed_units": completed_units,
        "active_unit": active_unit,
        "started_units": started_units,
    }


def _attach_benchmark_attempt_state(
    stage_name: str,
    attempt: Dict[str, Any],
    *,
    repo_root: Path,
    artifacts_dir: Optional[str],
) -> None:
    run_id = str(attempt.get("run_id") or "").strip()
    if not run_id:
        return
    benchmark_paths = _benchmark_run_event_paths(
        run_id,
        repo_root=repo_root,
        artifacts_dir=artifacts_dir,
    )
    attempt["artifacts"] = {
        **dict(attempt.get("artifacts") or {}),
        "run_dir": str(benchmark_paths["run_dir"]),
        "events_path": str(benchmark_paths["events"]),
        "output_json": str(benchmark_paths["output_json"]),
        "progress_path": str(benchmark_paths["progress"]),
    }
    benchmark_summary = _benchmark_stage_details_from_output(str(benchmark_paths["output_json"]))
    if benchmark_summary is not None:
        attempt["benchmark_summary"] = benchmark_summary
    if stage_name == "full_sweep":
        unit_progress = _load_benchmark_unit_progress(
            run_id,
            repo_root=repo_root,
            artifacts_dir=artifacts_dir,
        )
        attempt["completed_units"] = unit_progress.get("completed_units") or []
        attempt["active_unit"] = unit_progress.get("active_unit")


def _mark_attempt_aborted(
    stage_name: str,
    attempt: Dict[str, Any],
    *,
    reason: str,
    repo_root: Path,
    artifacts_dir: Optional[str],
    ended_at: Optional[str] = None,
) -> None:
    attempt["status"] = "aborted"
    attempt["ended_at"] = ended_at or _utc_now()
    attempt_issues = list(attempt.get("issues") or [])
    if reason not in attempt_issues:
        attempt_issues.append(reason)
    attempt["issues"] = attempt_issues
    _attach_benchmark_attempt_state(
        stage_name,
        attempt,
        repo_root=repo_root,
        artifacts_dir=artifacts_dir,
    )


def _normalize_incomplete_attempts_for_resume(
    stages: List[Dict[str, Any]],
    *,
    repo_root: Path,
    artifacts_dir: Optional[str],
    reason: str,
) -> None:
    for stage in stages:
        if not stage.get("enabled"):
            continue
        changed = False
        for attempt in stage.get("attempts") or []:
            if str(attempt.get("status") or "") != "running":
                continue
            _mark_attempt_aborted(
                str(stage.get("name") or ""),
                attempt,
                reason=reason,
                repo_root=repo_root,
                artifacts_dir=artifacts_dir,
            )
            changed = True
        if changed and str(stage.get("status") or "") == "running":
            stage["status"] = _compute_stage_status_from_attempts(stage)


def _summarize_inventory_for_summary(inventory: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "counts": dict(inventory.get("counts") or {}),
        "single_gpu": list(inventory.get("single_gpu") or []),
        "multi_gpu": list(inventory.get("multi_gpu") or []),
    }


def _build_frozen_full_sweep_plan(single_targets: List[str], multi_targets: List[str]) -> Dict[str, Any]:
    return {
        "single_gpu_targets": list(single_targets),
        "single_gpu_units": [entry["name"] for entry in _group_targets_by_unit(single_targets)],
        "multi_gpu_targets": list(multi_targets),
        "multi_gpu_units": [entry["name"] for entry in _group_targets_by_unit(multi_targets)],
    }


def _build_frozen_plan(*, inventory: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "full_sweep": _build_frozen_full_sweep_plan(
            list(inventory.get("single_gpu") or []),
            list(inventory.get("multi_gpu") or []),
        )
    }


def _stage_finish_event_map(events_path: Path) -> Dict[str, str]:
    statuses: Dict[str, str] = {}
    for payload in _read_jsonl(events_path):
        if payload.get("event") != "stage_finished":
            continue
        stage_name = str(payload.get("stage") or "").strip()
        if stage_name:
            statuses[stage_name] = str(payload.get("status") or "unknown")
    return statuses


def _recover_legacy_resume_state(
    *,
    resolved_run_id: str,
    run_dir: Path,
    stage_run_ids: Dict[str, str],
    inventory: Dict[str, Any],
    planned_stages: List[Dict[str, Any]],
    requested_contract: Dict[str, Any],
    repo_root: Path,
    artifacts_dir: Optional[str],
) -> Dict[str, Any]:
    events_path = run_dir / "events.jsonl"
    stage_statuses = _stage_finish_event_map(events_path)
    stages = _json_safe(planned_stages)
    tier1_stage = _find_stage(stages, "tier1")
    if "tier1" in stage_statuses:
        tier1_stage["status"] = stage_statuses["tier1"]
        tier1_stage["attempts"] = [
            _stage_attempt_entry(
                run_id=stage_run_ids["tier1"],
                status=stage_statuses["tier1"],
                recovered=True,
            )
        ]

    full_sweep_stage = _find_stage(stages, "full_sweep")
    single_run_id = _bucket_attempt_run_id(stage_run_ids["full_sweep"], "single_gpu", 0)
    single_run_start = _load_benchmark_run_start(
        single_run_id,
        repo_root=repo_root,
        artifacts_dir=artifacts_dir,
    )
    single_targets = list((single_run_start or {}).get("targets") or [])
    if single_targets:
        unit_progress = _load_benchmark_unit_progress(
            single_run_id,
            repo_root=repo_root,
            artifacts_dir=artifacts_dir,
        )
        benchmark_paths = _benchmark_run_event_paths(
            single_run_id,
            repo_root=repo_root,
            artifacts_dir=artifacts_dir,
        )
        benchmark_summary = _benchmark_stage_details_from_output(str(benchmark_paths["output_json"]))
        full_sweep_stage["status"] = "aborted"
        full_sweep_stage["attempts"] = [
            _stage_attempt_entry(
                run_id=single_run_id,
                bucket="single_gpu",
                status="aborted",
                targets=single_targets,
                units=[entry["name"] for entry in _group_targets_by_unit(single_targets)],
                completed_units=unit_progress.get("completed_units") or [],
                active_unit=unit_progress.get("active_unit"),
                artifacts={
                    "run_dir": str(benchmark_paths["run_dir"]),
                    "events_path": str(benchmark_paths["events"]),
                    "output_json": str(benchmark_paths["output_json"]),
                    "progress_path": str(benchmark_paths["progress"]),
                },
                benchmark_summary=benchmark_summary,
                issues=[
                    entry["error"]
                    for entry in (benchmark_summary or {}).get("failed_benchmarks", [])
                ] or ["full_sweep single bucket aborted before stage completion"],
                recovered=True,
            )
        ]
    elif "full_sweep" in stage_statuses:
        full_sweep_stage["status"] = stage_statuses["full_sweep"]

    frozen_plan = {
        "full_sweep": _build_frozen_full_sweep_plan(
            single_targets or list(inventory.get("single_gpu") or []),
            list(inventory.get("multi_gpu") or []),
        )
    }
    recovered_contract = dict(requested_contract)
    if single_run_start:
        for key in (
            "profile_type",
            "validity_profile",
            "allow_portable_expectations_update",
            "timeout_multiplier",
            "ncu_metric_set",
            "ncu_replay_mode",
            "nsys_timeout_seconds",
            "ncu_timeout_seconds",
            "update_expectations",
            "allow_mixed_provenance",
        ):
            run_start_key = key
            if key == "profile_type":
                run_start_key = "profile_type"
            recovered_value = single_run_start.get(run_start_key)
            if recovered_value is not None:
                recovered_contract[key] = recovered_value
        recovered_contract["run_tier1"] = bool(
            stage_statuses.get("tier1") or any(payload.get("event") == "stage_started" and payload.get("stage") == "tier1" for payload in _read_jsonl(events_path))
        )
        recovered_contract["run_full_sweep"] = True
    return {
        "generated_at": _utc_now(),
        "stages": stages,
        "contract": recovered_contract,
        "frozen_plan": frozen_plan,
        "legacy_recovered": True,
    }


def _load_resume_state(
    *,
    run_dir: Path,
    resolved_run_id: str,
    stage_run_ids: Dict[str, str],
    inventory: Dict[str, Any],
    planned_stages: List[Dict[str, Any]],
    requested_contract: Dict[str, Any],
    repo_root: Path,
    artifacts_dir: Optional[str],
) -> Dict[str, Any]:
    checkpoint = _read_json_if_exists(e2e_checkpoint_path(run_dir))
    if checkpoint is not None:
        return checkpoint
    return _recover_legacy_resume_state(
        resolved_run_id=resolved_run_id,
        run_dir=run_dir,
        stage_run_ids=stage_run_ids,
        inventory=inventory,
        planned_stages=planned_stages,
        requested_contract=requested_contract,
        repo_root=repo_root,
        artifacts_dir=artifacts_dir,
    )


def _normalize_stale_running_resume_state(
    resume_state: Dict[str, Any],
    *,
    repo_root: Path,
    artifacts_dir: Optional[str],
) -> Optional[str]:
    if str(resume_state.get("run_state") or "").strip() != "running":
        return None
    orchestrator_pid = resume_state.get("orchestrator_pid")
    if _pid_is_live(orchestrator_pid):
        return None

    stale_reason = (
        "orchestrator process exited without finalizing run state"
        if orchestrator_pid is None
        else f"orchestrator process {orchestrator_pid} exited without finalizing run state"
    )
    stages = _json_safe(resume_state.get("stages") or [])
    for stage in stages:
        if isinstance(stage, dict):
            stage.setdefault("attempts", [])
    _normalize_incomplete_attempts_for_resume(
        stages,
        repo_root=repo_root,
        artifacts_dir=artifacts_dir,
        reason=stale_reason,
    )
    for stage in stages:
        if str(stage.get("status") or "") == "running":
            stage["status"] = _compute_stage_status_from_attempts(stage)
            stage_issues = list(stage.get("issues") or [])
            if stale_reason not in stage_issues:
                stage_issues.append(stale_reason)
            stage["issues"] = stage_issues
    crash = dict(resume_state.get("crash") or {})
    crash.setdefault("type", "orchestrator_exit")
    crash.setdefault("message", stale_reason)
    resume_state["crash"] = crash
    resume_state["run_state"] = "aborted"
    resume_state["overall_status"] = "aborted"
    resume_state["success"] = False
    resume_state["resume_available"] = True
    resume_state["error"] = stale_reason
    resume_state["stages"] = stages
    return stale_reason


def _validate_resume_contract(
    *,
    requested: Dict[str, Any],
    stored: Dict[str, Any],
) -> Optional[str]:
    fields_to_validate = [
        "profile_type",
        "validity_profile",
        "single_gpu",
        "bench_root",
        "run_tier1",
        "run_full_sweep",
        "run_cluster",
        "run_fabric",
        "cluster_preset",
        "hosts",
        "labels",
        "ssh_user",
        "ssh_key",
        "oob_if",
        "socket_ifname",
        "nccl_ib_hca",
        "nmx_url",
        "nmx_token",
        "ib_mgmt_host",
        "ib_mgmt_user",
        "ib_mgmt_ssh_key",
        "cumulus_hosts",
        "cumulus_user",
        "cumulus_ssh_key",
        "primary_label",
        "coverage_baseline_run_id",
        "timeout_seconds",
        "suite_timeout",
        "timeout_multiplier",
        "accept_regressions",
        "update_expectations",
        "allow_mixed_provenance",
        "allow_portable_expectations_update",
        "iterations",
        "warmup",
        "gpu_sm_clock_mhz",
        "gpu_mem_clock_mhz",
        "ncu_metric_set",
        "ncu_replay_mode",
        "nsys_timeout_seconds",
        "ncu_timeout_seconds",
    ]
    mismatches: List[str] = []
    for field_name in fields_to_validate:
        if field_name not in stored:
            continue
        requested_value = _json_safe(requested.get(field_name))
        stored_value = _json_safe(stored.get(field_name))
        if requested_value != stored_value:
            mismatches.append(
                f"{field_name}: requested={requested_value!r}, original={stored_value!r}"
            )
    if not mismatches:
        return None
    return "Resume contract mismatch: " + "; ".join(mismatches)


def _build_checkpoint_payload(
    *,
    run_id: str,
    run_dir: Path,
    generated_at: str,
    updated_at: str,
    run_state: str,
    overall_status: str,
    success: bool,
    resume_available: bool,
    error: Optional[str],
    contract: Dict[str, Any],
    inventory: Dict[str, Any],
    frozen_plan: Dict[str, Any],
    hosts: Dict[str, Any],
    provenance: Dict[str, Any],
    stages: List[Dict[str, Any]],
    artifact_paths: Dict[str, Path],
    crash: Optional[Dict[str, Any]],
    orchestrator_pid: Optional[int],
) -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "generated_at": generated_at,
        "updated_at": updated_at,
        "run_state": run_state,
        "overall_status": overall_status,
        "success": success,
        "resume_available": resume_available,
        "error": error,
        "contract": contract,
        "inventory": _summarize_inventory_for_summary(inventory),
        "frozen_plan": frozen_plan,
        "hosts": hosts,
        "provenance": provenance,
        "stages": stages,
        "artifact_paths": {key: str(value) for key, value in artifact_paths.items()},
        "crash": crash,
        "orchestrator_pid": orchestrator_pid,
    }


def _build_summary_payload(
    *,
    run_id: str,
    run_dir: Path,
    generated_at: str,
    updated_at: str,
    run_state: str,
    overall_status: str,
    success: bool,
    resume_available: bool,
    error: Optional[str],
    contract: Dict[str, Any],
    inventory: Dict[str, Any],
    hosts: Dict[str, Any],
    provenance: Dict[str, Any],
    stages: List[Dict[str, Any]],
    artifact_paths: Dict[str, Path],
    crash: Optional[Dict[str, Any]],
    orchestrator_pid: Optional[int],
) -> Dict[str, Any]:
    return {
        "success": success,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "run_state": run_state,
        "overall_status": overall_status,
        "generated_at": generated_at,
        "updated_at": updated_at,
        "resume_available": resume_available,
        "error": error,
        "manifest_path": str(artifact_paths["manifest_path"]),
        "summary_path": str(artifact_paths["summary_path"]),
        "summary_markdown_path": str(artifact_paths["summary_markdown_path"]),
        "progress_path": str(artifact_paths["progress_path"]),
        "checkpoint_path": str(artifact_paths["checkpoint_path"]),
        "target_inventory_path": str(artifact_paths["target_inventory_path"]),
        "events_path": str(artifact_paths["events_path"]),
        "inventory": _summarize_inventory_for_summary(inventory),
        "hosts": hosts,
        "provenance": provenance,
        "contract": contract,
        "stages": stages,
        "crash": crash,
        "orchestrator_pid": orchestrator_pid,
    }


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
    resume: bool = False,
) -> Dict[str, Any]:
    repo_root = _repo_root()
    active_bench_root = Path(bench_root).resolve() if bench_root else repo_root
    normalized_validity_profile = normalize_validity_profile(validity_profile, field_name="validity_profile")
    if resume and not str(run_id or "").strip():
        return _json_safe(
            {
                "success": False,
                "overall_status": "failed",
                "run_state": "completed",
                "error": "resume=true requires an explicit run_id",
            }
        )
    resolved_run_id = resolve_e2e_run_id(run_id, repo_root=repo_root)
    run_dir = e2e_run_dir(resolved_run_id, repo_root)
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.json"
    summary_markdown_path = run_dir / "summary.md"
    progress_path = e2e_progress_path(run_dir)
    checkpoint_path = e2e_checkpoint_path(run_dir)
    target_inventory_path = run_dir / "target_inventory.json"
    events_path = run_dir / "events.jsonl"
    generated_at = _utc_now()
    progress_recorder = None if dry_run else ProgressRecorder(run_id=resolved_run_id, progress_path=progress_path)
    progress_emit_lock = threading.Lock()
    artifact_paths = {
        "manifest_path": manifest_path,
        "summary_path": summary_path,
        "summary_markdown_path": summary_markdown_path,
        "progress_path": progress_path,
        "checkpoint_path": checkpoint_path,
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
        failure_payload = {
            "success": False,
            "run_id": resolved_run_id,
            "run_dir": str(run_dir),
            "run_state": "completed",
            "overall_status": "failed",
            "generated_at": generated_at,
            "updated_at": generated_at,
            "resume_available": False,
            "error": str(exc),
            "manifest_path": str(manifest_path),
            "summary_path": str(summary_path),
            "summary_markdown_path": str(summary_markdown_path),
            "progress_path": str(progress_path),
            "checkpoint_path": str(checkpoint_path),
            "target_inventory_path": str(target_inventory_path),
            "events_path": str(events_path),
            "inventory": _summarize_inventory_for_summary(inventory),
            "stages": _json_safe(planned_stages),
            "provenance": {
                "generated_at": generated_at,
                "git": get_git_info(),
                "bench_root": str(active_bench_root),
            },
        }
        safe_failure = _json_safe(failure_payload)
        if not dry_run:
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_json(target_inventory_path, inventory)
            _write_json(summary_path, safe_failure)
            summary_markdown_path.write_text(_render_summary_markdown(safe_failure), encoding="utf-8")
            _write_json(manifest_path, {**safe_failure, "inventory": inventory})
        return safe_failure
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
    requested_contract = _build_e2e_contract(
        run_tier1=run_tier1,
        run_full_sweep=run_full_sweep,
        run_cluster=run_cluster,
        run_fabric=run_fabric,
        cluster_preset=cluster_preset,
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
        cumulus_hosts=list(cumulus_hosts or []),
        cumulus_user=cumulus_user,
        cumulus_ssh_key=cumulus_ssh_key,
        primary_label=primary_label,
        coverage_baseline_run_id=coverage_baseline_run_id,
        bench_root=str(active_bench_root),
        profile_type=profile_type,
        suite_timeout=suite_timeout,
        timeout_multiplier=timeout_multiplier,
        timeout_seconds=timeout_seconds,
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

    if expectation_error:
        result = {
            "success": False,
            "run_id": resolved_run_id,
            "run_dir": str(run_dir),
            "run_state": "completed",
            "overall_status": "failed",
            "updated_at": generated_at,
            "resume_available": False,
            "error": expectation_error,
            "generated_at": generated_at,
            "manifest_path": str(manifest_path),
            "summary_path": str(summary_path),
            "summary_markdown_path": str(summary_markdown_path),
            "progress_path": str(progress_path),
            "checkpoint_path": str(checkpoint_path),
            "target_inventory_path": str(target_inventory_path),
            "events_path": str(events_path),
            "inventory": _summarize_inventory_for_summary(inventory),
            "stages": _json_safe(planned_stages),
            "provenance": provenance,
            "contract": requested_contract,
        }
        if not dry_run:
            _append_event(events_path, "run_failed_preflight", error=expectation_error, run_id=resolved_run_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_json(target_inventory_path, inventory)
            _write_json(summary_path, _json_safe(result))
            summary_markdown_path.write_text(_render_summary_markdown(_json_safe(result)), encoding="utf-8")
            _write_json(manifest_path, {**_json_safe(result), "inventory": inventory})
        return _json_safe(result)

    def _materialize_stages(loaded_stages: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        loaded_by_name = {
            str(stage.get("name")): dict(stage)
            for stage in (loaded_stages or [])
            if str(stage.get("name", "")).strip()
        }
        materialized: List[Dict[str, Any]] = []
        for planned_stage in planned_stages:
            payload = dict(planned_stage)
            loaded = loaded_by_name.get(str(planned_stage.get("name")))
            if loaded:
                payload.update(loaded)
            payload["name"] = planned_stage["name"]
            payload.setdefault("enabled", planned_stage.get("enabled", False))
            payload.setdefault("run_id", planned_stage.get("run_id"))
            payload.setdefault("description", planned_stage.get("description"))
            payload.setdefault("attempts", [])
            materialized.append(payload)
        return materialized

    frozen_plan = _build_frozen_plan(inventory=inventory)
    if resume:
        if not checkpoint_path.exists() and not events_path.exists():
            return _json_safe(
                {
                    "success": False,
                    "run_id": resolved_run_id,
                    "run_dir": str(run_dir),
                    "run_state": "completed",
                    "overall_status": "failed",
                    "generated_at": generated_at,
                    "updated_at": generated_at,
                    "resume_available": False,
                    "error": f"No prior run state found for run_id={resolved_run_id!r}",
                    "manifest_path": str(manifest_path),
                    "summary_path": str(summary_path),
                    "summary_markdown_path": str(summary_markdown_path),
                    "progress_path": str(progress_path),
                    "checkpoint_path": str(checkpoint_path),
                    "target_inventory_path": str(target_inventory_path),
                    "events_path": str(events_path),
                    "inventory": _summarize_inventory_for_summary(inventory),
                    "stages": _json_safe(planned_stages),
                    "provenance": provenance,
                    "contract": requested_contract,
                }
            )
        resume_state = _load_resume_state(
            run_dir=run_dir,
            resolved_run_id=resolved_run_id,
            stage_run_ids=stage_run_ids,
            inventory=inventory,
            planned_stages=planned_stages,
            requested_contract=requested_contract,
            repo_root=repo_root,
            artifacts_dir=artifacts_dir,
        )
        stale_reason = _normalize_stale_running_resume_state(
            resume_state,
            repo_root=repo_root,
            artifacts_dir=artifacts_dir,
        )
        if stale_reason:
            _append_event(
                events_path,
                "run_recovered_aborted",
                run_id=resolved_run_id,
                error=stale_reason,
            )
        mismatch_error = _validate_resume_contract(
            requested=requested_contract,
            stored=resume_state.get("contract") or {},
        )
        if mismatch_error:
            return _json_safe(
                {
                    "success": False,
                    "run_id": resolved_run_id,
                    "run_dir": str(run_dir),
                    "run_state": "completed",
                    "overall_status": "failed",
                    "generated_at": str(resume_state.get("generated_at") or generated_at),
                    "updated_at": generated_at,
                    "resume_available": True,
                    "error": mismatch_error,
                    "manifest_path": str(manifest_path),
                    "summary_path": str(summary_path),
                    "summary_markdown_path": str(summary_markdown_path),
                    "progress_path": str(progress_path),
                    "checkpoint_path": str(checkpoint_path),
                    "target_inventory_path": str(target_inventory_path),
                    "events_path": str(events_path),
                    "inventory": _summarize_inventory_for_summary(inventory),
                    "stages": _json_safe(resume_state.get("stages") or planned_stages),
                    "provenance": provenance,
                    "contract": requested_contract,
                }
            )
        generated_at = str(resume_state.get("generated_at") or generated_at)
        frozen_plan = _json_safe(resume_state.get("frozen_plan") or frozen_plan)
        stages = _materialize_stages(resume_state.get("stages"))
        _normalize_incomplete_attempts_for_resume(
            stages,
            repo_root=repo_root,
            artifacts_dir=artifacts_dir,
            reason="resume superseded unfinished attempt",
        )
        requested_contract = _json_safe(resume_state.get("contract") or requested_contract)
    else:
        stages = _materialize_stages()

    run_state = "running"
    error: Optional[str] = None
    crash: Optional[Dict[str, Any]] = None
    resume_available = False
    run_finished_event_emitted = False
    current_stage_name: Optional[str] = None
    current_stage_started_at: Optional[float] = None
    current_stage_event_run_id: Optional[str] = None
    current_bucket: Optional[str] = None

    def _current_overall_status() -> str:
        if run_state == "aborted":
            return "aborted"
        statuses = [str(stage.get("status") or "planned") for stage in stages if stage.get("enabled")]
        if run_state == "running":
            if any(status == "aborted" for status in statuses):
                return "aborted"
            if any(status == "failed" for status in statuses):
                return "failed"
            if any(status == "partial" for status in statuses):
                return "partial"
            return "running"
        return _roll_up_overall_status(statuses)

    def _persist_state() -> Dict[str, Any]:
        updated_at = _utc_now()
        overall_status = _current_overall_status()
        success = run_state == "completed" and overall_status not in {"failed", "aborted"}
        summary_payload = _build_summary_payload(
            run_id=resolved_run_id,
            run_dir=run_dir,
            generated_at=generated_at,
            updated_at=updated_at,
            run_state=run_state,
            overall_status=overall_status,
            success=success,
            resume_available=resume_available,
            error=error,
            contract=requested_contract,
            inventory=inventory,
            hosts=cluster_host_config,
            provenance=provenance,
            stages=stages,
            artifact_paths=artifact_paths,
            crash=crash,
            orchestrator_pid=os.getpid(),
        )
        checkpoint_payload = _build_checkpoint_payload(
            run_id=resolved_run_id,
            run_dir=run_dir,
            generated_at=generated_at,
            updated_at=updated_at,
            run_state=run_state,
            overall_status=overall_status,
            success=success,
            resume_available=resume_available,
            error=error,
            contract=requested_contract,
            inventory=inventory,
            frozen_plan=frozen_plan,
            hosts=cluster_host_config,
            provenance=provenance,
            stages=stages,
            artifact_paths=artifact_paths,
            crash=crash,
            orchestrator_pid=os.getpid(),
        )
        if not dry_run:
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_json(target_inventory_path, inventory)
            _write_json(checkpoint_path, _json_safe(checkpoint_payload))
            _write_json(summary_path, _json_safe(summary_payload))
            summary_markdown_path.write_text(_render_summary_markdown(_json_safe(summary_payload)), encoding="utf-8")
            _write_json(
                manifest_path,
                {
                    **_json_safe(summary_payload),
                    "inventory": inventory,
                    "checkpoint": _json_safe(checkpoint_payload),
                    "frozen_plan": _json_safe(frozen_plan),
                },
            )
            _emit_live_progress(
                progress_recorder,
                stages=stages,
                run_state=run_state,
                overall_status=overall_status,
                artifact_paths=artifact_paths,
                emit_lock=progress_emit_lock,
                orchestrator_pid=os.getpid(),
            )
        return _json_safe(summary_payload)

    def _emit_stage_progress_snapshot(
        stage_name: str,
        child_run_id: str,
        child_progress_path: Path,
        *,
        bucket: Optional[str] = None,
    ) -> None:
        child_progress = _load_progress_current(child_progress_path)
        if child_progress is None:
            return
        _emit_live_progress(
            progress_recorder,
            stages=stages,
            run_state=run_state,
            overall_status=_current_overall_status(),
            artifact_paths=artifact_paths,
            emit_lock=progress_emit_lock,
            child_progress=child_progress,
            child_stage_name=stage_name,
            child_run_id=child_run_id,
            child_bucket=bucket,
            orchestrator_pid=os.getpid(),
        )

    def _run_with_stage_progress_mirror(
        stage_name: str,
        child_run_id: str,
        child_progress_path: Path,
        invoke,
        *,
        bucket: Optional[str] = None,
    ):
        if progress_recorder is None:
            return invoke()

        stop_event = threading.Event()

        def _mirror_worker() -> None:
            while True:
                _emit_stage_progress_snapshot(
                    stage_name,
                    child_run_id,
                    child_progress_path,
                    bucket=bucket,
                )
                if stop_event.wait(_STAGE_PROGRESS_POLL_SECONDS):
                    break

        thread = threading.Thread(
            target=_mirror_worker,
            name=f"e2e-progress-{stage_name}-{child_run_id}",
            daemon=True,
        )
        thread.start()
        try:
            return invoke()
        finally:
            stop_event.set()
            thread.join(timeout=5.0)
            _emit_stage_progress_snapshot(
                stage_name,
                child_run_id,
                child_progress_path,
                bucket=bucket,
            )

    def _start_stage(stage_name: str, event_run_id: str) -> None:
        nonlocal current_stage_name, current_stage_started_at, current_stage_event_run_id, current_bucket
        current_stage_name = stage_name
        current_stage_started_at = time.monotonic()
        current_stage_event_run_id = event_run_id
        current_bucket = None
        _replace_stage(stages, name=stage_name, status="running")
        _append_event(events_path, "stage_started", stage=stage_name, run_id=event_run_id)
        _persist_state()

    def _finish_stage(
        stage_name: str,
        *,
        status: str,
        command: Optional[List[str]] = None,
        returncode: Optional[int] = None,
        result: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, Any]] = None,
        issues: Optional[List[str]] = None,
        duration_ms: Optional[int] = None,
    ) -> None:
        nonlocal current_stage_name, current_stage_started_at, current_stage_event_run_id, current_bucket
        stage = _find_stage(stages, stage_name)
        _replace_stage(
            stages,
            name=stage_name,
            status=status,
            command=command,
            returncode=returncode,
            result=result,
            artifacts=artifacts,
            issues=issues,
            duration_ms=duration_ms,
            attempts=stage.get("attempts") or [],
        )
        _append_event(
            events_path,
            "stage_finished",
            stage=stage_name,
            status=status,
            run_id=current_stage_event_run_id or stage.get("run_id"),
        )
        current_stage_name = None
        current_stage_started_at = None
        current_stage_event_run_id = None
        current_bucket = None
        _persist_state()

    def _abort_current_stage(message: str) -> None:
        nonlocal current_stage_name, current_stage_started_at, current_stage_event_run_id, current_bucket
        if not current_stage_name:
            return
        stage = _find_stage(stages, current_stage_name)
        attempts = stage.get("attempts") or []
        if attempts:
            latest_attempt = attempts[-1]
            if latest_attempt.get("status") == "running":
                _mark_attempt_aborted(
                    current_stage_name,
                    latest_attempt,
                    reason=message,
                    repo_root=repo_root,
                    artifacts_dir=artifacts_dir,
                    ended_at=_utc_now(),
                )
        issues = list(stage.get("issues") or [])
        if message not in issues:
            issues.append(message)
        _replace_stage(
            stages,
            name=current_stage_name,
            status="aborted",
            issues=issues,
            attempts=attempts,
            duration_ms=int((time.monotonic() - current_stage_started_at) * 1000)
            if current_stage_started_at is not None
            else stage.get("duration_ms"),
        )
        _append_event(
            events_path,
            "stage_finished",
            stage=current_stage_name,
            status="aborted",
            run_id=current_stage_event_run_id or stage.get("run_id"),
            error=message,
        )
        current_stage_name = None
        current_stage_started_at = None
        current_stage_event_run_id = None
        current_bucket = None
        _persist_state()

    if dry_run:
        return _json_safe(
            {
                "success": True,
                "dry_run": True,
                "run_id": resolved_run_id,
                "run_dir": str(run_dir),
                "run_state": "dry_run",
                "overall_status": "dry_run",
                "generated_at": generated_at,
                "updated_at": generated_at,
                "resume_available": bool(resume),
                "manifest_path": str(manifest_path),
                "summary_path": str(summary_path),
                "summary_markdown_path": str(summary_markdown_path),
                "progress_path": str(progress_path),
                "checkpoint_path": str(checkpoint_path),
                "target_inventory_path": str(target_inventory_path),
                "events_path": str(events_path),
                "inventory": _summarize_inventory_for_summary(inventory),
                "hosts": cluster_host_config,
                "provenance": provenance,
                "contract": requested_contract,
                "stages": stages,
                "frozen_plan": frozen_plan,
            }
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(target_inventory_path, inventory)
    _append_event(events_path, "run_resumed" if resume else "run_started", run_id=resolved_run_id)
    _persist_state()

    class _E2EAbort(RuntimeError):
        pass

    previous_handlers: Dict[int, Any] = {}
    abort_signal: Dict[str, Optional[str]] = {"signal": None}

    def _handle_abort_signal(signum: int, _frame: Any) -> None:
        signame = signal.Signals(signum).name
        abort_signal["signal"] = signame
        raise _E2EAbort(f"received {signame}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _handle_abort_signal)

    try:
        if run_tier1:
            tier1_stage = _find_stage(stages, "tier1")
            if tier1_stage.get("status") not in {"succeeded", "skipped_duplicate"}:
                attempt_index = len(tier1_stage.get("attempts") or [])
                tier1_run_id = _stage_attempt_run_id(stage_run_ids["tier1"], attempt_index)
                tier1_command = [
                    "python",
                    "-m",
                    "cli.aisp",
                    "bench",
                    "run-tier1",
                    "--run-id",
                    tier1_run_id,
                    "--profile",
                    profile_type,
                    "--validity-profile",
                    normalized_validity_profile,
                ]
                tier1_attempt = _stage_attempt_entry(
                    run_id=tier1_run_id,
                    status="running",
                    command=tier1_command,
                    started_at=_utc_now(),
                )
                tier1_stage.setdefault("attempts", []).append(tier1_attempt)
                _start_stage("tier1", tier1_run_id)
                tier1_progress_path = _benchmark_run_event_paths(
                    tier1_run_id,
                    repo_root=repo_root,
                    artifacts_dir=artifacts_dir,
                )["progress"]
                with _benchmark_queue_lock("tier1", tier1_run_id, repo_root=repo_root):
                    tier1_result = _run_with_stage_progress_mirror(
                        "tier1",
                        tier1_run_id,
                        tier1_progress_path,
                        lambda: _invoke_run_tier1_suite(
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
                            run_id=tier1_run_id,
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
                        ),
                    )
                tier1_status, tier1_issues, tier1_benchmark_summary = _benchmark_stage_status(
                    tier1_result,
                    required_paths=["summary_path", "regression_summary_path", "trend_snapshot_path"],
                )
                tier1_attempt.update(
                    {
                        "status": tier1_status,
                        "ended_at": _utc_now(),
                        "returncode": 1 if tier1_status == "failed" else 0,
                        "result": tier1_result,
                        "artifacts": {
                            "summary_path": tier1_result.get("summary_path"),
                            "regression_summary_path": tier1_result.get("regression_summary_path"),
                            "trend_snapshot_path": tier1_result.get("trend_snapshot_path"),
                            "history_root": tier1_result.get("history_root"),
                        },
                        "issues": tier1_issues,
                        "duration_ms": int((time.monotonic() - (current_stage_started_at or time.monotonic())) * 1000),
                    }
                )
                if tier1_benchmark_summary is not None:
                    tier1_attempt["benchmark_summary"] = tier1_benchmark_summary
                _finish_stage(
                    "tier1",
                    status=tier1_status,
                    command=tier1_command,
                    returncode=1 if tier1_status == "failed" else 0,
                    result=tier1_result,
                    artifacts=tier1_attempt.get("artifacts"),
                    issues=tier1_issues,
                    duration_ms=tier1_attempt.get("duration_ms"),
                )

        if run_full_sweep:
            full_stage = _find_stage(stages, "full_sweep")
            if full_stage.get("status") not in {"succeeded", "skipped_duplicate"}:
                _start_stage("full_sweep", stage_run_ids["full_sweep"])
                full_stage_issues: List[str] = []
                full_stage_result: Dict[str, Any] = {"buckets": {}}
                bucket_outcomes: List[str] = []
                single_targets = list((frozen_plan.get("full_sweep") or {}).get("single_gpu_targets") or [])
                multi_targets = list((frozen_plan.get("full_sweep") or {}).get("multi_gpu_targets") or [])
                if not single_targets and not multi_targets:
                    full_stage_issues.append("no benchmark targets discovered for full sweep")
                    full_status = "failed"
                else:
                    full_status = "succeeded"
                    for bucket_name, all_targets in (
                        ("single_gpu", single_targets),
                        ("multi_gpu", multi_targets),
                    ):
                        if not all_targets:
                            continue
                        current_bucket_targets = list(
                            inventory.get("single_gpu" if bucket_name == "single_gpu" else "multi_gpu") or []
                        )
                        all_units = list(
                            (frozen_plan.get("full_sweep") or {}).get(
                                "single_gpu_units" if bucket_name == "single_gpu" else "multi_gpu_units"
                            )
                            or [entry["name"] for entry in _group_targets_by_unit(all_targets)]
                        )
                        bucket_attempts = _bucket_attempts(full_stage, bucket_name)
                        completed_units = _completed_units_from_attempts(bucket_attempts, ordered_units=all_units)
                        remaining_units = _remaining_units_after_completed_units(
                            all_units,
                            completed_units=completed_units,
                        )
                        remaining_targets, missing_units = _resolve_targets_for_units(
                            current_bucket_targets,
                            ordered_units=remaining_units,
                        )
                        if bucket_name == "multi_gpu" and gpu_count < 2:
                            skip_reason = f"requires >=2 visible GPUs; detected {gpu_count}"
                            if not bucket_attempts:
                                skip_attempt = _stage_attempt_entry(
                                    run_id=_bucket_attempt_run_id(stage_run_ids["full_sweep"], bucket_name, 0),
                                    bucket=bucket_name,
                                    status="skipped",
                                    targets=all_targets,
                                    units=all_units,
                                    completed_units=completed_units,
                                    reason=skip_reason,
                                    issues=[skip_reason],
                                    started_at=_utc_now(),
                                    ended_at=_utc_now(),
                                )
                                full_stage.setdefault("attempts", []).append(skip_attempt)
                                bucket_attempts = _bucket_attempts(full_stage, bucket_name)
                                _persist_state()
                            full_stage_result["buckets"][bucket_name] = {
                                "targets": all_targets,
                                "status": "skipped",
                                "reason": skip_reason,
                                "attempts": bucket_attempts,
                            }
                            full_stage_issues.append(f"multi-GPU bucket skipped because only {gpu_count} visible GPU(s) were detected")
                            bucket_outcomes.append("partial")
                            continue
                        if missing_units:
                            missing_issue = (
                                f"resume could not resolve current benchmark targets for unit(s): {', '.join(missing_units)}"
                            )
                            full_stage_result["buckets"][bucket_name] = {
                                "targets": all_targets,
                                "status": "failed",
                                "reason": missing_issue,
                                "attempts": bucket_attempts,
                                "missing_units": missing_units,
                            }
                            full_stage_issues.append(missing_issue)
                            bucket_outcomes.append("failed")
                            continue
                        if not remaining_targets:
                            latest_attempt = bucket_attempts[-1] if bucket_attempts else None
                            if latest_attempt is not None:
                                latest_status = str(latest_attempt.get("status") or "succeeded")
                                full_stage_result["buckets"][bucket_name] = {
                                    "targets": all_targets,
                                    "status": latest_status,
                                    "attempts": bucket_attempts,
                                    "latest_attempt_run_id": latest_attempt.get("run_id"),
                                }
                                bucket_outcomes.append("partial" if latest_status == "skipped" else latest_status)
                            continue

                        attempt_index = len(bucket_attempts)
                        bucket_run_id = _bucket_attempt_run_id(stage_run_ids["full_sweep"], bucket_name, attempt_index)
                        bucket_command = [
                            "python",
                            "-m",
                            "cli.aisp",
                            "bench",
                            "run",
                            "--run-id",
                            bucket_run_id,
                            "--profile",
                            profile_type,
                            "--validity-profile",
                            normalized_validity_profile,
                            *sum([["-t", target] for target in remaining_targets], []),
                        ]
                        attempt_units = [entry["name"] for entry in _group_targets_by_unit(remaining_targets)]
                        bucket_attempt = _stage_attempt_entry(
                            run_id=bucket_run_id,
                            bucket=bucket_name,
                            status="running",
                            targets=remaining_targets,
                            units=attempt_units,
                            completed_units=[],
                            active_unit=attempt_units[0] if attempt_units else None,
                            command=bucket_command,
                            started_at=_utc_now(),
                        )
                        full_stage.setdefault("attempts", []).append(bucket_attempt)
                        current_bucket = bucket_name
                        _persist_state()
                        bucket_progress_path = _benchmark_run_event_paths(
                            bucket_run_id,
                            repo_root=repo_root,
                            artifacts_dir=artifacts_dir,
                        )["progress"]
                        with _benchmark_queue_lock(f"full_sweep_{bucket_name}", bucket_run_id, repo_root=repo_root):
                            bucket_result = _run_with_stage_progress_mirror(
                                "full_sweep",
                                bucket_run_id,
                                bucket_progress_path,
                                lambda: _invoke_execute_benchmarks(
                                    targets=remaining_targets,
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
                                    run_id=bucket_run_id,
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
                                ),
                                bucket=bucket_name,
                            )
                        bucket_status, bucket_issues, bucket_benchmark_summary = _benchmark_stage_status(
                            bucket_result,
                            required_paths=["output_json"],
                        )
                        unit_progress = _load_benchmark_unit_progress(
                            bucket_run_id,
                            repo_root=repo_root,
                            artifacts_dir=artifacts_dir,
                        )
                        benchmark_paths = _benchmark_run_event_paths(
                            bucket_run_id,
                            repo_root=repo_root,
                            artifacts_dir=artifacts_dir,
                        )
                        bucket_attempt.update(
                            {
                                "status": bucket_status,
                                "ended_at": _utc_now(),
                                "returncode": 1 if bucket_status == "failed" else 0,
                                "result": bucket_result,
                                "artifacts": {
                                    "run_dir": str(benchmark_paths["run_dir"]),
                                    "events_path": str(benchmark_paths["events"]),
                                    "output_json": str(benchmark_paths["output_json"]),
                                    "progress_path": str(benchmark_paths["progress"]),
                                },
                                "issues": bucket_issues,
                                "duration_ms": int((time.monotonic() - (current_stage_started_at or time.monotonic())) * 1000),
                                "completed_units": unit_progress.get("completed_units") or [],
                                "active_unit": unit_progress.get("active_unit"),
                            }
                        )
                        if bucket_benchmark_summary is not None:
                            bucket_attempt["benchmark_summary"] = bucket_benchmark_summary
                        full_stage_result["buckets"][bucket_name] = {
                            "targets": all_targets,
                            "status": bucket_status,
                            "attempts": _bucket_attempts(full_stage, bucket_name),
                            "latest_attempt_run_id": bucket_run_id,
                        }
                        full_stage_issues.extend(bucket_issues)
                        bucket_outcomes.append(bucket_status)
                        _persist_state()
                        current_bucket = None

                    if any(status == "aborted" for status in bucket_outcomes):
                        full_status = "aborted"
                    elif any(status == "failed" for status in bucket_outcomes):
                        full_status = "failed"
                    elif any(status in {"partial", "skipped"} for status in bucket_outcomes):
                        full_status = "partial"
                    elif bucket_outcomes:
                        full_status = "succeeded"
                    else:
                        full_status = "failed"
                        full_stage_issues.append("no full-sweep bucket produced a terminal result")
                _finish_stage(
                    "full_sweep",
                    status=full_status,
                    returncode=1 if full_status in {"failed", "aborted"} else 0,
                    result=full_stage_result,
                    artifacts={"target_inventory_path": str(target_inventory_path)},
                    issues=full_stage_issues,
                    duration_ms=int((time.monotonic() - (current_stage_started_at or time.monotonic())) * 1000),
                )

        if run_cluster:
            cluster_stage = _find_stage(stages, "cluster")
            if cluster_stage.get("status") not in {"succeeded", "skipped_duplicate"}:
                attempt_index = len(cluster_stage.get("attempts") or [])
                cluster_run_id = _stage_attempt_run_id(stage_run_ids["cluster"], attempt_index)
                cluster_attempt = _stage_attempt_entry(
                    run_id=cluster_run_id,
                    status="running",
                    started_at=_utc_now(),
                )
                cluster_stage.setdefault("attempts", []).append(cluster_attempt)
                _start_stage("cluster", cluster_run_id)
                cluster_result = _run_with_stage_progress_mirror(
                    "cluster",
                    cluster_run_id,
                    _cluster_run_progress_path(cluster_run_id, repo_root=repo_root),
                    lambda: _invoke_run_cluster_common_eval(
                        preset=cluster_preset,
                        run_id=cluster_run_id,
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
                    ),
                )
                cluster_status, cluster_issues, cluster_scorecard = _cluster_stage_status(cluster_result)
                cluster_attempt.update(
                    {
                        "status": cluster_status,
                        "ended_at": _utc_now(),
                        "command": cluster_result.get("command"),
                        "returncode": int(cluster_result.get("returncode", 0) or 0),
                        "result": cluster_result,
                        "artifacts": {
                            "run_dir": cluster_result.get("run_dir"),
                            "manifest_path": cluster_result.get("manifest_path"),
                            "fabric_scorecard": cluster_scorecard,
                        },
                        "issues": cluster_issues,
                        "duration_ms": int((time.monotonic() - (current_stage_started_at or time.monotonic())) * 1000),
                    }
                )
                _finish_stage(
                    "cluster",
                    status=cluster_status,
                    command=cluster_result.get("command"),
                    returncode=int(cluster_result.get("returncode", 0) or 0),
                    result=cluster_result,
                    artifacts=cluster_attempt.get("artifacts"),
                    issues=cluster_issues,
                    duration_ms=cluster_attempt.get("duration_ms"),
                )

        fabric_duplicate = run_fabric and run_cluster and cluster_preset.strip().lower() == "fabric-systems"
        fabric_stage = _find_stage(stages, "fabric")
        if fabric_duplicate:
            if fabric_stage.get("status") != "skipped_duplicate":
                duplicate_attempt = _stage_attempt_entry(
                    run_id=stage_run_ids["fabric"],
                    status="skipped_duplicate",
                    reason="fabric stage duplicated by cluster preset fabric-systems",
                    issues=["fabric stage duplicated by cluster preset fabric-systems"],
                    started_at=_utc_now(),
                    ended_at=_utc_now(),
                )
                fabric_stage.setdefault("attempts", []).append(duplicate_attempt)
                _finish_stage(
                    "fabric",
                    status="skipped_duplicate",
                    issues=["fabric stage duplicated by cluster preset fabric-systems"],
                    duration_ms=0,
                )
        elif run_fabric and fabric_stage.get("status") not in {"succeeded", "skipped_duplicate"}:
            attempt_index = len(fabric_stage.get("attempts") or [])
            fabric_run_id = _stage_attempt_run_id(stage_run_ids["fabric"], attempt_index)
            fabric_attempt = _stage_attempt_entry(
                run_id=fabric_run_id,
                status="running",
                started_at=_utc_now(),
            )
            fabric_stage.setdefault("attempts", []).append(fabric_attempt)
            _start_stage("fabric", fabric_run_id)
            fabric_result = _run_with_stage_progress_mirror(
                "fabric",
                fabric_run_id,
                _cluster_run_progress_path(fabric_run_id, repo_root=repo_root),
                lambda: _invoke_run_cluster_fabric_eval(
                    run_id=fabric_run_id,
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
                ),
            )
            fabric_status, fabric_issues, fabric_scorecard = _cluster_stage_status(fabric_result)
            fabric_attempt.update(
                {
                    "status": fabric_status,
                    "ended_at": _utc_now(),
                    "command": fabric_result.get("command"),
                    "returncode": int(fabric_result.get("returncode", 0) or 0),
                    "result": fabric_result,
                    "artifacts": {
                        "run_dir": fabric_result.get("run_dir"),
                        "manifest_path": fabric_result.get("manifest_path"),
                        "fabric_scorecard": fabric_scorecard,
                    },
                    "issues": fabric_issues,
                    "duration_ms": int((time.monotonic() - (current_stage_started_at or time.monotonic())) * 1000),
                }
            )
            _finish_stage(
                "fabric",
                status=fabric_status,
                command=fabric_result.get("command"),
                returncode=int(fabric_result.get("returncode", 0) or 0),
                result=fabric_result,
                artifacts=fabric_attempt.get("artifacts"),
                issues=fabric_issues,
                duration_ms=fabric_attempt.get("duration_ms"),
            )

        run_state = "completed"
        resume_available = False
        _append_event(
            events_path,
            "run_finished",
            run_id=resolved_run_id,
            overall_status=_roll_up_overall_status(
                [str(stage.get("status") or "planned") for stage in stages if stage.get("enabled")]
            ),
            success=_roll_up_overall_status(
                [str(stage.get("status") or "planned") for stage in stages if stage.get("enabled")]
            )
            not in {"failed", "aborted"},
        )
        run_finished_event_emitted = True
        return _persist_state()
    except (_E2EAbort, KeyboardInterrupt) as exc:
        error = str(exc)
        crash = {
            "type": type(exc).__name__,
            "message": str(exc),
            "signal": abort_signal.get("signal"),
            "traceback": traceback.format_exc(),
        }
        run_state = "aborted"
        resume_available = True
        _abort_current_stage(error)
    except Exception as exc:
        error = str(exc)
        crash = {
            "type": type(exc).__name__,
            "message": str(exc),
            "signal": abort_signal.get("signal"),
            "traceback": traceback.format_exc(),
        }
        run_state = "aborted"
        resume_available = True
        _abort_current_stage(error)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    if not run_finished_event_emitted:
        _append_event(
            events_path,
            "run_finished",
            run_id=resolved_run_id,
            overall_status="aborted",
            success=False,
            error=error,
        )
    return _persist_state()
