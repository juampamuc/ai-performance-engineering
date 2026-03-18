#!/usr/bin/env python3
"""Run the advisory benchmark-pair audit workflow and write structured artifacts."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from core.scripts.ci import check_verification_compliance as compliance_check
from core.scripts import validate_benchmark_pairs as pair_validation
from core.verification import review_baseline_optimized_pairs as pair_review


REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PYTEST_TARGETS = [
    "tests/test_review_pair_scanner.py",
    "tests/test_validate_benchmark_pairs_tools.py",
    "tests/test_benchmark_hygiene_regressions.py",
    "tests/test_review_findings_regressions.py",
    "tests/test_run_benchmark_pair_audit.py",
]
REVIEW_REPORT_NAME = "BENCHMARK_PAIR_REVIEW_REPORT.md"
AUDIT_PAIR_VALIDATION_TIMEOUT_SECONDS = 30


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ__benchmark-pair-audit")


def _host_capability_bucket(*, cuda_available: bool, gpu_count: int) -> str:
    if not cuda_available or gpu_count <= 0:
        return "static-only"
    if gpu_count == 1:
        return "single-gpu"
    return "multi-gpu"


def _resolve_scopes(scope_args: Optional[List[str]]) -> List[str]:
    if not scope_args:
        return ["all"]
    cleaned = [item for item in scope_args if item]
    if not cleaned or "all" in cleaned:
        return ["all"]
    deduped: List[str] = []
    for item in cleaned:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _discover_pairs(scopes: List[str]) -> Dict[str, Dict[str, Path]]:
    if scopes == ["all"]:
        return pair_validation.discover_benchmark_pairs(REPO_ROOT)
    merged: Dict[str, Dict[str, Path]] = {}
    for scope in scopes:
        merged.update(pair_validation.discover_benchmark_pairs(REPO_ROOT, chapter=scope))
    return merged


def _sanitize_scope(scope: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", scope)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _compliance_report_to_dict(report: compliance_check.ComplianceReport) -> Dict[str, Any]:
    return {
        "files_checked": report.files_checked,
        "errors": report.errors,
        "warnings": report.warnings,
        "issues": [
            {
                "file_path": issue.file_path,
                "severity": issue.severity,
                "message": issue.message,
                "line_number": issue.line_number,
            }
            for issue in report.issues
        ],
    }


def _merge_validation_reports(reports: Iterable[pair_validation.ValidationReport]) -> pair_validation.ValidationReport:
    merged = pair_validation.ValidationReport(timestamp=_utc_now_iso())
    for report in reports:
        merged.total_pairs += report.total_pairs
        merged.valid_pairs += report.valid_pairs
        merged.invalid_pairs += report.invalid_pairs
        merged.missing_signature_pairs += report.missing_signature_pairs
        merged.signature_mismatch_pairs += report.signature_mismatch_pairs
        merged.skipped_pairs += report.skipped_pairs
        merged.error_pairs += report.error_pairs
        merged.results.extend(report.results)
    return merged


def _run_review_step(scopes: List[str], output_dir: Path) -> Tuple[Dict[str, Any], pair_review.ReviewReport]:
    report = pair_review.run_review(None if scopes == ["all"] else scopes)
    log_buffer = io.StringIO()
    with redirect_stdout(log_buffer):
        pair_review.print_review_summary(report)
    review_dir = output_dir / "review"
    written = pair_review.write_review_outputs(report, review_dir, write_json=True, write_markdown=True)
    stdout_path = review_dir / "review_stdout.log"
    stdout_path.write_text(log_buffer.getvalue(), encoding="utf-8")
    step = {
        "status": "completed_with_findings" if report.findings else "completed",
        "summary": report.to_dict()["summary"],
        "artifacts": {
            "stdout": str(stdout_path),
            **written,
        },
    }
    return step, report


def _run_compliance_step(files: List[Path], output_dir: Path) -> Tuple[Dict[str, Any], compliance_check.ComplianceReport]:
    compliance_dir = output_dir / "compliance"
    report = compliance_check.check_compliance(files=files, root_dir=REPO_ROOT, validate_pairs=False)
    log_buffer = io.StringIO()
    with redirect_stdout(log_buffer):
        compliance_check.print_report(report)
    stdout_path = compliance_dir / "compliance_stdout.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(log_buffer.getvalue(), encoding="utf-8")
    report_path = compliance_dir / "compliance_report.json"
    _write_json(report_path, _compliance_report_to_dict(report))
    step = {
        "status": "completed_with_findings" if report.errors or report.warnings else "completed",
        "summary": {
            "files_checked": report.files_checked,
            "errors": report.errors,
            "warnings": report.warnings,
        },
        "artifacts": {
            "stdout": str(stdout_path),
            "json": str(report_path),
        },
    }
    return step, report


def _run_pair_validation_step(
    scopes: List[str],
    output_dir: Path,
    *,
    cuda_available: bool,
) -> Tuple[Dict[str, Any], Optional[pair_validation.ValidationReport]]:
    validation_dir = output_dir / "pair_validation"
    stdout_path = validation_dir / "pair_validation_stdout.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    if not cuda_available:
        stdout_path.write_text("SKIPPED: CUDA unavailable on audit host\n", encoding="utf-8")
        step = {
            "status": "skipped",
            "summary": {"reason": "CUDA unavailable on audit host"},
            "artifacts": {"stdout": str(stdout_path)},
        }
        return step, None

    reports: List[pair_validation.ValidationReport] = []
    log_buffer = io.StringIO()
    with redirect_stdout(log_buffer):
        if scopes == ["all"]:
            reports.append(
                pair_validation.validate_all_pairs(
                    REPO_ROOT,
                    pair_timeout_seconds=AUDIT_PAIR_VALIDATION_TIMEOUT_SECONDS,
                )
            )
        else:
            for scope in scopes:
                reports.append(
                    pair_validation.validate_all_pairs(
                        REPO_ROOT,
                        chapter=scope,
                        pair_timeout_seconds=AUDIT_PAIR_VALIDATION_TIMEOUT_SECONDS,
                    )
                )
    stdout_path.write_text(log_buffer.getvalue(), encoding="utf-8")
    merged = _merge_validation_reports(reports)
    report_path = validation_dir / "pair_validation_report.json"
    _write_json(report_path, merged.to_dict())
    has_findings = bool(merged.signature_mismatch_pairs or merged.error_pairs or merged.missing_signature_pairs)
    step = {
        "status": "completed_with_findings" if has_findings else "completed",
        "summary": merged.to_dict()["summary"],
        "artifacts": {
            "stdout": str(stdout_path),
            "json": str(report_path),
        },
        "pair_timeout_seconds": AUDIT_PAIR_VALIDATION_TIMEOUT_SECONDS,
    }
    return step, merged


def _run_pytest_audit_step(output_dir: Path) -> Dict[str, Any]:
    pytest_dir = output_dir / "pytest"
    stdout_path = pytest_dir / "pytest_stdout.log"
    stderr_path = pytest_dir / "pytest_stderr.log"
    cmd = [sys.executable, "-m", "pytest", *AUDIT_PYTEST_TARGETS, "-v", "--tb=short"]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    return {
        "status": "completed" if proc.returncode == 0 else "completed_with_findings",
        "summary": {"returncode": proc.returncode, "targets": AUDIT_PYTEST_TARGETS},
        "artifacts": {
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        },
        "command": cmd,
    }


def _extract_multi_gpu_required(path: Path) -> bool:
    if "multigpu" in path.stem.lower():
        return True
    benchmark, load_error = pair_validation.load_benchmark_class(path)
    if benchmark is None:
        if load_error and ">=2 GPU" in load_error:
            return True
        return False
    try:
        config = benchmark.get_config() if hasattr(benchmark, "get_config") else None
    except Exception:
        config = None
    return bool(
        getattr(config, "multi_gpu_required", False)
        or getattr(benchmark, "multi_gpu_required", False)
    )


def _validation_lookup(report: Optional[pair_validation.ValidationReport]) -> Dict[str, pair_validation.PairValidationResult]:
    if report is None:
        return {}
    return {f"{result.chapter}:{result.example_name}": result for result in report.results}


def _pair_statuses(
    pairs: Dict[str, Dict[str, Path]],
    review_report: pair_review.ReviewReport,
    validation_report: Optional[pair_validation.ValidationReport],
) -> Dict[str, str]:
    statuses = {pair_key: "PASS" for pair_key in pairs}
    validation_by_key = _validation_lookup(validation_report)
    for pair_key, result in validation_by_key.items():
        if result.skipped:
            if result.error and ">=2 GPU" in result.error:
                statuses[pair_key] = "REQUIRES >=2 GPUs"
            else:
                statuses[pair_key] = "SKIPPED"
        elif result.error or result.mismatches or not result.valid:
            statuses[pair_key] = "FLAG"
    for pair_key in review_report.pair_statuses():
        statuses[pair_key] = "FLAG"
    return statuses


def _parse_report_status_table(report_path: Path) -> Dict[str, str]:
    statuses: Dict[str, str] = {}
    pattern = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*([^|]+?)\s*\|")
    for line in report_path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        statuses[match.group(1)] = match.group(2).strip()
    return statuses


def _run_report_drift_step(
    scopes: List[str],
    pairs: Dict[str, Dict[str, Path]],
    review_report: pair_review.ReviewReport,
    validation_report: Optional[pair_validation.ValidationReport],
    output_dir: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    drift_dir = output_dir / "report_drift"
    pair_statuses = _pair_statuses(pairs, review_report, validation_report)
    findings: List[Dict[str, Any]] = []

    scoped_reports: List[Tuple[str, Path]] = []
    for pair_key in pairs:
        scope, _example = pair_key.split(":", 1)
        report_path = REPO_ROOT / scope / REVIEW_REPORT_NAME
        if report_path.exists() and (scope, report_path) not in scoped_reports:
            scoped_reports.append((scope, report_path))

    for scope, report_path in scoped_reports:
        reported_statuses = _parse_report_status_table(report_path)
        for example_name, reported_status in reported_statuses.items():
            pair_key = f"{scope}:{example_name}"
            current_status = pair_statuses.get(pair_key)
            if current_status is None or current_status == reported_status:
                continue
            pair_paths = pairs[pair_key]
            findings.append(
                pair_review._make_issue(
                    file=str(report_path),
                    issue_type="report_drift",
                    severity="medium",
                    message=(
                        f"Review report status drift for {pair_key}: "
                        f"reported={reported_status}, current={current_status}"
                    ),
                    baseline_path=pair_paths.get("baseline"),
                    optimized_path=pair_paths.get("optimized"),
                    evidence={
                        "pair_key": pair_key,
                        "reported_status": reported_status,
                        "current_status": current_status,
                    },
                )
            )

    report_path = drift_dir / "report_drift.json"
    _write_json(report_path, {"findings": findings})
    step = {
        "status": "completed_with_findings" if findings else "completed",
        "summary": {"findings": len(findings)},
        "artifacts": {"json": str(report_path)},
    }
    return step, findings


def _run_gpu_rescan_step(scopes: List[str], output_dir: Path, include_gpu_rescan: bool) -> Dict[str, Any]:
    gpu_dir = output_dir / "gpu_rescan"
    gpu_dir.mkdir(parents=True, exist_ok=True)
    if not include_gpu_rescan:
        return {
            "status": "not_requested",
            "summary": {"reason": "GPU rescan not requested"},
            "artifacts": {},
        }
    if not torch.cuda.is_available():
        return {
            "status": "skipped",
            "summary": {"reason": "CUDA unavailable on audit host"},
            "artifacts": {},
        }

    commands: List[Dict[str, Any]] = []
    requested_scopes = scopes
    if scopes == ["all"]:
        requested_scopes = sorted({pair_key.split(":", 1)[0] for pair_key in _discover_pairs(scopes)})

    for scope in requested_scopes:
        run_id = f"{_default_run_id()}__{_sanitize_scope(scope)}"
        stdout_path = gpu_dir / f"{_sanitize_scope(scope)}.stdout.log"
        stderr_path = gpu_dir / f"{_sanitize_scope(scope)}.stderr.log"
        cmd = [
            sys.executable,
            "-m",
            "cli.aisp",
            "bench",
            "run",
            "--targets",
            scope,
            "--profile",
            "minimal",
            "--format",
            "json",
            "--verify-phase",
            "detect",
            "--suite-timeout",
            "0",
            "--artifacts-dir",
            str(gpu_dir / "artifacts"),
            "--run-id",
            run_id,
        ]
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        commands.append(
            {
                "scope": scope,
                "command": cmd,
                "returncode": proc.returncode,
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
                "status": "completed" if proc.returncode == 0 else "completed_with_findings",
            }
        )

    any_failures = any(item["returncode"] != 0 for item in commands)
    return {
        "status": "completed_with_findings" if any_failures else "completed",
        "summary": {
            "commands": len(commands),
            "nonzero_returncodes": sum(1 for item in commands if item["returncode"] != 0),
        },
        "artifacts": {"runs": commands},
    }


def _pair_results(
    pairs: Dict[str, Dict[str, Path]],
    review_report: pair_review.ReviewReport,
    validation_report: Optional[pair_validation.ValidationReport],
    *,
    cuda_available: bool,
) -> List[Dict[str, Any]]:
    pair_statuses = _pair_statuses(pairs, review_report, validation_report)
    validation_by_key = _validation_lookup(validation_report)
    results: List[Dict[str, Any]] = []
    for pair_key, paths in sorted(pairs.items()):
        validation = validation_by_key.get(pair_key)
        if validation and validation.skipped:
            bucket = "skipped"
            skip_reason = validation.error
        elif validation is None and not cuda_available:
            bucket = "static-only"
            skip_reason = "CUDA unavailable on audit host"
        else:
            bucket = "multi-gpu" if any(_extract_multi_gpu_required(path) for path in paths.values()) else "single-gpu"
            skip_reason = None
        results.append(
            {
                "pair_key": pair_key,
                "scope": pair_key.split(":", 1)[0],
                "status": pair_statuses.get(pair_key, "PASS"),
                "bucket": bucket,
                "skip_reason": skip_reason,
                "baseline_path": str(paths["baseline"]),
                "optimized_path": str(paths["optimized"]),
            }
        )
    return results


def _render_summary_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# Benchmark Pair Advisory Audit Summary",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Timestamp: `{summary['timestamp']}`",
        f"- Scope: `{', '.join(summary['scopes'])}`",
        f"- CUDA available: `{summary['host']['cuda_available']}`",
        f"- GPU count: `{summary['host']['gpu_count']}`",
        "",
        "## Steps",
        "",
        "| Step | Status |",
        "|---|---|",
    ]
    for step_name, step in summary["steps"].items():
        lines.append(f"| {step_name} | {step['status']} |")
    lines.extend(
        [
            "",
            "## Pair Statuses",
            "",
            "| Pair | Status | Bucket |",
            "|---|---|---|",
        ]
    )
    for item in summary["pair_results"]:
        lines.append(f"| `{item['pair_key']}` | {item['status']} | {item['bucket']} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the advisory benchmark pair audit workflow")
    parser.add_argument(
        "--scope",
        action="append",
        help="Limit the audit to a chapter or lab path. Repeatable. Use 'all' or omit for full scope.",
    )
    parser.add_argument("--include-gpu-rescan", action="store_true", help="Run exhaustive bench invocations for the requested scopes")
    parser.add_argument("--run-id", default=None, help="Run identifier for the audit output directory")
    parser.add_argument("--output-dir", type=Path, default=None, help="Audit output directory")
    args = parser.parse_args(argv)

    scopes = _resolve_scopes(args.scope)
    run_id = args.run_id or _default_run_id()
    output_dir = args.output_dir or (REPO_ROOT / "artifacts" / "audits" / run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = _discover_pairs(scopes)
    files = sorted({path for item in pairs.values() for path in item.values()})
    cuda_available = bool(torch.cuda.is_available())
    gpu_count = int(torch.cuda.device_count()) if cuda_available else 0
    host_bucket = _host_capability_bucket(cuda_available=cuda_available, gpu_count=gpu_count)
    started_at = _utc_now_iso()

    review_step, review_report = _run_review_step(scopes, output_dir)
    compliance_step, _compliance_report = _run_compliance_step(files, output_dir)
    validation_step, validation_report = _run_pair_validation_step(scopes, output_dir, cuda_available=cuda_available)
    pytest_step = _run_pytest_audit_step(output_dir)
    report_drift_step, report_drift_findings = _run_report_drift_step(
        scopes,
        pairs,
        review_report,
        validation_report,
        output_dir,
    )
    gpu_step = _run_gpu_rescan_step(scopes, output_dir, args.include_gpu_rescan)

    summary = {
        "run_id": run_id,
        "timestamp": _utc_now_iso(),
        "scopes": scopes,
        "host": {"cuda_available": cuda_available, "gpu_count": gpu_count},
        "steps": {
            "review": review_step,
            "compliance": compliance_step,
            "pair_validation": validation_step,
            "pytest_audit": pytest_step,
            "report_drift": report_drift_step,
            "gpu_rescan": gpu_step,
        },
        "summary": {
            "review_findings": len(review_report.findings),
            "report_drift_findings": len(report_drift_findings),
            "pair_validation_ran": validation_report is not None,
        },
        "pair_results": _pair_results(
            pairs,
            review_report,
            validation_report,
            cuda_available=cuda_available,
        ),
    }

    summary_json_path = output_dir / "summary.json"
    summary_markdown_path = output_dir / "summary.md"
    _write_json(summary_json_path, summary)
    summary_markdown_path.write_text(_render_summary_markdown(summary), encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "timestamp_started": started_at,
        "timestamp_completed": _utc_now_iso(),
        "scopes": scopes,
        "pair_count": len(pairs),
        "output_dir": str(output_dir),
        "host": {
            "cuda_available": cuda_available,
            "gpu_count": gpu_count,
            "capability_bucket": host_bucket,
        },
        "commands": {
            "review": [
                sys.executable,
                "-m",
                "core.verification.review_baseline_optimized_pairs",
                "--json",
                "--markdown",
                "--output-dir",
                str(output_dir / "review"),
                *[arg for scope in ([] if scopes == ["all"] else scopes) for arg in ("--chapter", scope)],
            ],
            "compliance": [
                sys.executable,
                "-m",
                "core.scripts.ci.check_verification_compliance",
            ],
            "pair_validation": (
                [
                    sys.executable,
                    "-m",
                    "core.scripts.validate_benchmark_pairs",
                    "--pair-timeout-seconds",
                    str(AUDIT_PAIR_VALIDATION_TIMEOUT_SECONDS),
                ]
                if scopes == ["all"]
                else [
                    [
                        sys.executable,
                        "-m",
                        "core.scripts.validate_benchmark_pairs",
                        "--pair-timeout-seconds",
                        str(AUDIT_PAIR_VALIDATION_TIMEOUT_SECONDS),
                        "--chapter",
                        scope,
                    ]
                    for scope in scopes
                ]
            ),
            "pytest_audit": pytest_step.get("command"),
            "gpu_rescan": gpu_step.get("artifacts", {}).get("runs", []),
        },
        "artifacts": {
            "summary_json": str(summary_json_path),
            "summary_markdown": str(summary_markdown_path),
            "review": review_step.get("artifacts", {}),
            "compliance": compliance_step.get("artifacts", {}),
            "pair_validation": validation_step.get("artifacts", {}),
            "pytest_audit": pytest_step.get("artifacts", {}),
            "report_drift": report_drift_step.get("artifacts", {}),
            "gpu_rescan": gpu_step.get("artifacts", {}),
        },
    }
    _write_json(output_dir / "manifest.json", manifest)

    print(f"Audit manifest: {output_dir / 'manifest.json'}")
    print(f"Audit summary:  {output_dir / 'summary.json'}")
    print(f"Audit markdown: {output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
