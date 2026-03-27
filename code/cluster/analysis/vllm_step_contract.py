from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_latest_suite_step(suite_steps_path: Path, step_name: str) -> Optional[Dict[str, Any]]:
    if not suite_steps_path.exists():
        return None
    payload = json.loads(suite_steps_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return None
    latest: Optional[Dict[str, Any]] = None
    for row in payload:
        if not isinstance(row, dict):
            continue
        if str(row.get("name") or "").strip() != step_name:
            continue
        latest = row
    return latest


def summarize_upstream_failure(
    suite_steps_path: Path,
    step_name: str,
    startup_artifact_path: Optional[Path] = None,
) -> Dict[str, Any]:
    startup_payload: Dict[str, Any] = {}
    if startup_artifact_path is not None and startup_artifact_path.exists():
        startup_payload = _load_json(startup_artifact_path)

    step_row = _load_latest_suite_step(suite_steps_path, step_name)
    step_rc: Optional[int] = None
    step_log_path: Optional[str] = None
    if step_row is not None:
        raw_rc = step_row.get("exit_code")
        if raw_rc is not None:
            step_rc = int(raw_rc)
        raw_log = str(step_row.get("log_path") or "").strip()
        if raw_log:
            step_log_path = raw_log

    status = str(startup_payload.get("status") or "").strip()
    ready = bool(startup_payload.get("ready"))
    elapsed = startup_payload.get("elapsed_seconds")
    detail = str(startup_payload.get("detail") or "").strip()
    server_log_path = str(startup_payload.get("server_log_path") or "").strip() or None

    if step_rc in (None, 0) and status in {"", "ok"}:
        return {
            "has_failure": False,
            "step_name": step_name,
            "step_exit_code": step_rc,
            "startup_status": status or None,
            "ready": ready,
            "elapsed_seconds": elapsed,
            "server_log_path": server_log_path,
            "detail": detail or None,
            "message": None,
        }

    if status in {"startup_timeout", "startup_error", "benchmark_failed_after_ready"}:
        detail_suffix = f"; detail={detail}" if detail else ""
        elapsed_suffix = f"; elapsed_seconds={elapsed}" if elapsed is not None else ""
        log_suffix = f"; server_log_path={server_log_path}" if server_log_path else ""
        return {
            "has_failure": True,
            "step_name": step_name,
            "step_exit_code": step_rc,
            "startup_status": status,
            "ready": ready,
            "elapsed_seconds": elapsed,
            "server_log_path": server_log_path,
            "detail": detail or None,
            "message": (
                f"{step_name} failed upstream with status={status}; ready={ready}"
                f"{elapsed_suffix}{log_suffix}{detail_suffix}"
            ),
        }

    if step_rc not in (None, 0):
        detail_suffix = f"; detail={detail}" if detail else ""
        server_log_suffix = f"; server_log_path={server_log_path}" if server_log_path else ""
        log_suffix = f"; suite_log_path={step_log_path}" if step_log_path else ""
        return {
            "has_failure": True,
            "step_name": step_name,
            "step_exit_code": step_rc,
            "startup_status": status or None,
            "ready": ready,
            "elapsed_seconds": elapsed,
            "server_log_path": server_log_path,
            "detail": detail or None,
            "message": (
                f"{step_name} failed upstream with exit_code={step_rc}"
                f"{log_suffix}{server_log_suffix}{detail_suffix}"
            ),
        }

    return {
        "has_failure": False,
        "step_name": step_name,
        "step_exit_code": step_rc,
        "startup_status": status or None,
        "ready": ready,
        "elapsed_seconds": elapsed,
        "server_log_path": server_log_path,
        "detail": detail or None,
        "message": None,
    }


def validate_vllm_serve_csv(csv_path: Path) -> None:
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8", newline="")))
    if not rows:
        raise ValueError("vLLM serve sweep csv has no rows")
    for idx, row in enumerate(rows, start=1):
        completed = int(float((row.get("completed") or 0.0)))
        failed = int(float((row.get("failed") or 0.0)))
        total_tok = float((row.get("total_token_throughput") or 0.0))
        if completed <= 0:
            raise ValueError(f"vLLM serve sweep row {idx} has completed={completed} (must be > 0)")
        if failed > 0:
            raise ValueError(f"vLLM serve sweep row {idx} has failed={failed} (must be 0)")
        if total_tok <= 0.0:
            raise ValueError(
                f"vLLM serve sweep row {idx} has total_token_throughput={total_tok} (must be > 0)"
            )


def validate_vllm_request_rate_csv(csv_path: Path) -> None:
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8", newline="")))
    if not rows:
        raise ValueError("request-rate sweep csv has no rows")
    for idx, row in enumerate(rows, start=1):
        completed = int(float((row.get("completed") or 0.0)))
        failed = int(float((row.get("failed") or 0.0)))
        total_tok = float((row.get("total_token_throughput") or 0.0))
        if completed <= 0:
            raise ValueError(f"request-rate sweep row {idx} has completed={completed} (must be > 0)")
        if failed > 0:
            raise ValueError(f"request-rate sweep row {idx} has failed={failed} (must be 0)")
        if total_tok <= 0.0:
            raise ValueError(
                f"request-rate sweep row {idx} has total_token_throughput={total_tok} (must be > 0)"
            )


def validate_vllm_stability_summary(stability_path: Path, *, threshold: Optional[float] = None, label: str) -> None:
    payload = _load_json(stability_path)
    summary = payload.get("summary") or {}
    points = int(summary.get("points") or 0)
    if points <= 0:
        raise ValueError(f"{label} has no points")
    value = summary.get("total_token_throughput_cv_pct_p95")
    if value is None:
        raise ValueError(f"{label} missing total_token_throughput_cv_pct_p95")
    cv = float(value)
    if cv < 0:
        raise ValueError(f"{label} CV is negative")
    if threshold is not None and cv > threshold:
        raise ValueError(f"{label} CV p95={cv} exceeds threshold {threshold}")


def validate_vllm_slo_goodput_summary(slo_path: Path, *, request_rate: bool = False) -> None:
    payload = _load_json(slo_path)
    status = payload.get("status")
    if status != "ok":
        raise ValueError(f"SLO goodput status is not ok: {status}")
    summary = payload.get("summary") or {}
    if request_rate:
        if int(summary.get("request_rate_points", 0)) <= 0:
            raise ValueError("request-rate SLO goodput request_rate_points is not positive")
        return
    if int(summary.get("concurrency_points", 0)) <= 0:
        raise ValueError("vLLM SLO goodput concurrency_points is not positive")
    if float(summary.get("peak_total_tok_s", 0.0)) <= 0:
        raise ValueError("vLLM SLO goodput peak_total_tok_s is not positive")
    if float(summary.get("max_goodput_tok_s", 0.0)) < 0:
        raise ValueError("vLLM SLO goodput max_goodput_tok_s is negative")
