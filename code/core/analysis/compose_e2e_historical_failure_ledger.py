#!/usr/bin/env python3
"""Compose a historical failure ledger for an e2e sweep from preserved attempts plus reruns."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _load_json(path: Path, *, label: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {label} {path}, got {type(payload).__name__}")
    return payload


def _run_id_from_results_path(path: Path) -> str:
    try:
        return path.resolve().parents[1].name
    except IndexError:
        return path.stem


def _find_full_sweep_stage(summary_payload: Dict[str, Any]) -> Dict[str, Any]:
    stages = summary_payload.get("stages")
    if not isinstance(stages, list):
        raise ValueError("Expected e2e summary 'stages' to be a list")
    for stage in stages:
        if isinstance(stage, dict) and stage.get("name") == "full_sweep":
            return stage
    raise ValueError("E2E summary does not contain a 'full_sweep' stage")


def _extract_historical_failures(summary_json: Path) -> Dict[str, Dict[str, Any]]:
    payload = _load_json(summary_json, label="e2e summary")
    stage = _find_full_sweep_stage(payload)
    attempts = stage.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError("Expected full_sweep attempts to be a list")

    rows: Dict[str, Dict[str, Any]] = {}
    for attempt_index, attempt in enumerate(attempts):
        if not isinstance(attempt, dict):
            raise ValueError(f"Expected attempt object in full_sweep attempts, got {type(attempt).__name__}")
        benchmark_summary = attempt.get("benchmark_summary") or {}
        failed_rows = benchmark_summary.get("failed_benchmarks") or []
        if not isinstance(failed_rows, list):
            raise ValueError("Expected benchmark_summary.failed_benchmarks to be a list")
        run_id = str(attempt.get("run_id") or "").strip() or f"attempt_{attempt_index}"
        for failed in failed_rows:
            if not isinstance(failed, dict):
                raise ValueError(f"Expected failed benchmark row to be a dict, got {type(failed).__name__}")
            target = str(failed.get("target") or "").strip()
            if not target:
                raise ValueError(f"Missing target in failed benchmark row for attempt {run_id}")
            status = str(failed.get("status") or "failed_error").strip() or "failed_error"
            error = str(failed.get("error") or "").strip() or None
            row = rows.setdefault(
                target,
                {
                    "target": target,
                    "original_attempt_run_id": run_id,
                    "original_status": status,
                    "original_error": error,
                    "historical_attempt_run_ids": [],
                    "historical_occurrences": [],
                },
            )
            row["historical_attempt_run_ids"].append(run_id)
            row["historical_occurrences"].append(
                {
                    "run_id": run_id,
                    "status": status,
                    "error": error,
                }
            )
    return rows


def _extract_rerun_targets(results_json: Path) -> Dict[str, Dict[str, Any]]:
    payload = _load_json(results_json, label="benchmark results")
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Expected 'results' list in benchmark results {results_json}, got {type(results).__name__}")
    run_id = _run_id_from_results_path(results_json)
    extracted: Dict[str, Dict[str, Any]] = {}
    for chapter in results:
        if not isinstance(chapter, dict):
            raise ValueError(
                f"Expected chapter object in benchmark results {results_json}, got {type(chapter).__name__}"
            )
        chapter_name = str(chapter.get("chapter") or "").strip() or "<unknown>"
        benchmarks = chapter.get("benchmarks")
        if not isinstance(benchmarks, list):
            raise ValueError(
                f"Expected benchmarks list for chapter {chapter_name} in benchmark results {results_json}, got {type(benchmarks).__name__}"
            )
        for benchmark in benchmarks:
            if not isinstance(benchmark, dict):
                raise ValueError(
                    f"Expected benchmark object for chapter {chapter_name} in benchmark results {results_json}, got {type(benchmark).__name__}"
                )
            example = str(benchmark.get("example") or "").strip() or "<unknown>"
            target = f"{chapter_name}:{example}"
            extracted[target] = {
                "target": target,
                "run_id": run_id,
                "status": str(benchmark.get("status") or "").strip() or "unknown",
                "error": str(benchmark.get("error") or "").strip() or None,
                "best_speedup": benchmark.get("best_speedup"),
                "skip_reason": str(benchmark.get("skip_reason") or "").strip() or None,
            }
    return extracted


def _disposition_for_rerun(row: Optional[Dict[str, Any]]) -> str:
    if row is None:
        return "not_rerun"
    status = str(row.get("status") or "").strip()
    if status == "succeeded":
        return "resolved_success"
    if status == "skipped":
        return "resolved_skip"
    return "still_failing"


def compose_e2e_historical_failure_ledger(
    *,
    e2e_summary_json: Path,
    rerun_results_json: Iterable[Path],
) -> Dict[str, Any]:
    historical = _extract_historical_failures(e2e_summary_json)
    latest_by_target: Dict[str, Dict[str, Any]] = {}
    rerun_run_ids: List[str] = []
    for rerun_path in rerun_results_json:
        rerun_targets = _extract_rerun_targets(rerun_path)
        rerun_run_ids.append(_run_id_from_results_path(rerun_path))
        for target, row in rerun_targets.items():
            if target in historical:
                latest_by_target[target] = row

    rows: List[Dict[str, Any]] = []
    for target in sorted(historical):
        original = historical[target]
        rerun = latest_by_target.get(target)
        disposition = _disposition_for_rerun(rerun)
        notes = None
        if rerun is None:
            notes = "No rerun evidence recorded."
        elif rerun.get("status") == "skipped":
            notes = rerun.get("skip_reason") or rerun.get("error")
        elif rerun.get("status") not in {"succeeded", "skipped"}:
            notes = rerun.get("error")
        rows.append(
            {
                "target": target,
                "original_attempt_run_id": original["original_attempt_run_id"],
                "original_status": original["original_status"],
                "original_error": original["original_error"],
                "historical_attempt_run_ids": original["historical_attempt_run_ids"],
                "rerun_run_id": rerun.get("run_id") if rerun else None,
                "rerun_status": rerun.get("status") if rerun else None,
                "best_speedup": rerun.get("best_speedup") if rerun else None,
                "disposition": disposition,
                "notes": notes,
            }
        )

    summary = {
        "e2e_run_id": _load_json(e2e_summary_json, label="e2e summary").get("run_id"),
        "total_historical_failures": len(rows),
        "rechecked_count": sum(1 for row in rows if row["rerun_run_id"]),
        "resolved_success_count": sum(1 for row in rows if row["disposition"] == "resolved_success"),
        "resolved_skip_count": sum(1 for row in rows if row["disposition"] == "resolved_skip"),
        "still_failing_count": sum(1 for row in rows if row["disposition"] == "still_failing"),
        "not_rerun_count": sum(1 for row in rows if row["disposition"] == "not_rerun"),
        "rerun_run_ids": rerun_run_ids,
    }
    return {"summary": summary, "rows": rows}


def _markdown_for_ledger(ledger: Dict[str, Any]) -> str:
    summary = ledger["summary"]
    lines = [
        "# Historical Failure Ledger",
        "",
        f"- E2E run: `{summary['e2e_run_id']}`",
        f"- Total historical failures: `{summary['total_historical_failures']}`",
        f"- Rechecked: `{summary['rechecked_count']}`",
        f"- Resolved success: `{summary['resolved_success_count']}`",
        f"- Resolved skip: `{summary['resolved_skip_count']}`",
        f"- Still failing: `{summary['still_failing_count']}`",
        f"- Not rerun: `{summary['not_rerun_count']}`",
        "",
        "| Target | Original Attempt | Original Status | Rerun | Rerun Status | Disposition | Best Speedup | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in ledger["rows"]:
        speedup = row.get("best_speedup")
        speedup_text = "" if speedup is None else f"{float(speedup):.3f}x"
        notes = (row.get("notes") or "").replace("\n", " ")
        lines.append(
            f"| `{row['target']}` | `{row['original_attempt_run_id']}` | `{row['original_status']}` | "
            f"`{row.get('rerun_run_id') or ''}` | `{row.get('rerun_status') or ''}` | "
            f"`{row['disposition']}` | `{speedup_text}` | {notes} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_e2e_historical_failure_ledger(
    *,
    e2e_summary_json: Path,
    rerun_results_json: Iterable[Path],
    output_dir: Path,
) -> Dict[str, Path]:
    ledger = compose_e2e_historical_failure_ledger(
        e2e_summary_json=e2e_summary_json,
        rerun_results_json=rerun_results_json,
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "historical_failure_ledger.json"
    md_path = output_dir / "historical_failure_ledger.md"
    json_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_markdown_for_ledger(ledger), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def attach_historical_failure_ledger_to_e2e_package(
    *,
    e2e_summary_json: Path,
    ledger_json: Path,
    ledger_markdown: Path,
) -> None:
    ledger_json = ledger_json.resolve()
    ledger_markdown = ledger_markdown.resolve()
    summary_payload = _load_json(e2e_summary_json, label="e2e summary")
    ledger_payload = _load_json(ledger_json, label="historical failure ledger")
    ledger_ref = {
        "json_path": str(ledger_json),
        "markdown_path": str(ledger_markdown),
        "summary": ledger_payload.get("summary"),
    }
    summary_payload["historical_failure_ledger"] = ledger_ref
    e2e_summary_json.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")

    manifest_path = Path(str(summary_payload["manifest_path"]))
    manifest_payload = _load_json(manifest_path, label="e2e manifest")
    manifest_payload["historical_failure_ledger"] = ledger_ref
    manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")

    summary_markdown_path = Path(str(summary_payload["summary_markdown_path"]))
    from core.benchmark.e2e_sweep import _render_summary_markdown

    summary_markdown_path.write_text(_render_summary_markdown(summary_payload), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose and attach an e2e historical failure ledger.")
    parser.add_argument("--e2e-summary-json", type=Path, required=True, help="Top-level e2e summary.json")
    parser.add_argument(
        "--rerun-results-json",
        type=Path,
        action="append",
        default=[],
        help="Rerun benchmark_test_results.json (repeatable)",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for historical_failure_ledger.{json,md}")
    parser.add_argument("--attach", action="store_true", help="Attach the generated ledger to the e2e package summary/manifest")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        outputs = write_e2e_historical_failure_ledger(
            e2e_summary_json=args.e2e_summary_json,
            rerun_results_json=args.rerun_results_json,
            output_dir=args.output_dir,
        )
        if args.attach:
            attach_historical_failure_ledger_to_e2e_package(
                e2e_summary_json=args.e2e_summary_json,
                ledger_json=outputs["json"],
                ledger_markdown=outputs["markdown"],
            )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
