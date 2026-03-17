#!/usr/bin/env python3
"""Derive weak deep-dive review candidates directly from benchmark results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _best_optimization(bench: Dict[str, Any]) -> Dict[str, Any]:
    optimizations = list(bench.get("optimizations", []))
    if not optimizations:
        return {}
    goal = str(bench.get("optimization_goal") or "speed").strip().lower()
    if goal == "memory":
        return max(optimizations, key=lambda row: float(row.get("memory_savings_pct") or -1e9))
    return max(optimizations, key=lambda row: float(row.get("speedup") or 0.0))


def _source_run_ids(results_payload: Dict[str, Any]) -> List[str]:
    source_run_ids = []
    for run_id in results_payload.get("source_run_ids", []):
        text = str(run_id)
        if text and text not in source_run_ids:
            source_run_ids.append(text)
    run_id = results_payload.get("run_id")
    if run_id:
        text = str(run_id)
        if text not in source_run_ids:
            source_run_ids.append(text)
    return source_run_ids


def build_review_candidates(
    results_payload: Dict[str, Any],
    *,
    weak_positive_threshold: float = 1.20,
    flat_threshold: float = 1.05,
    target_filter: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    source_run_ids = _source_run_ids(results_payload)
    if len(source_run_ids) == 1:
        run_id = source_run_ids[0]
    elif source_run_ids:
        run_id = "merged_review_set"
    else:
        run_id = None
    for chapter_result in results_payload.get("results", []):
        chapter = chapter_result.get("chapter")
        for bench in chapter_result.get("benchmarks", []):
            example = bench.get("example")
            if not chapter or not example:
                continue
            target = f"{chapter}:{example}"
            if target_filter is not None and target not in target_filter:
                continue
            goal = str(bench.get("optimization_goal") or "speed").strip().lower()
            best_opt = _best_optimization(bench)
            speedup = float(best_opt.get("speedup") or 1.0)
            row = {
                "chapter": chapter,
                "example": example,
                "goal": goal,
                "speedup": speedup,
                "baseline_ms": bench.get("baseline_time_ms"),
                "best_opt_ms": best_opt.get("time_ms"),
                "technique": best_opt.get("technique", ""),
                "status": bench.get("status"),
            }

            if goal != "speed":
                row["bucket"] = "non_speed_goal"
                rows.append(row)
                continue

            if best_opt.get("time_ms") is None or not best_opt.get("technique"):
                row["bucket"] = "flat_or_negative"
                rows.append(row)
                continue

            if speedup <= flat_threshold:
                row["bucket"] = "flat_or_negative"
                rows.append(row)
            elif speedup <= weak_positive_threshold:
                row["bucket"] = "weak_positive"
                rows.append(row)

    rows.sort(key=lambda row: (str(row.get("chapter")), str(row.get("example"))))
    return {
        "run_id": run_id,
        "source_run_ids": source_run_ids,
        "count": len(rows),
        "rows": rows,
        "thresholds": {
            "flat_or_negative_max_speedup": flat_threshold,
            "weak_positive_max_speedup": weak_positive_threshold,
        },
    }


def _to_markdown(payload: Dict[str, Any], *, results_json: Path) -> str:
    lines = [
        "# Deep-Dive Review Candidates",
        "",
        f"- Source results: `{results_json}`",
        f"- Source run IDs: `{payload.get('source_run_ids', [])}`",
        f"- Candidate count: `{payload.get('count', 0)}`",
        f"- Thresholds: `{payload.get('thresholds', {})}`",
        "",
        "| Target | Goal | Bucket | Speedup | Baseline ms | Best Opt ms | Technique | Status |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in payload.get("rows", []):
        lines.append(
            f"| `{row.get('chapter')}:{row.get('example')}` | `{row.get('goal')}` | `{row.get('bucket')}` | "
            f"`{row.get('speedup')}` | `{row.get('baseline_ms')}` | `{row.get('best_opt_ms')}` | "
            f"`{row.get('technique')}` | `{row.get('status')}` |"
        )
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    *,
    results_json: Path,
    output_dir: Path | None = None,
    weak_positive_threshold: float = 1.20,
    flat_threshold: float = 1.05,
    target_filter: Optional[Set[str]] = None,
) -> Dict[str, Path]:
    payload = build_review_candidates(
        _load_json(results_json),
        weak_positive_threshold=weak_positive_threshold,
        flat_threshold=flat_threshold,
        target_filter=target_filter,
    )
    out_dir = output_dir or results_json.parent.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "deep_dive_review_candidates_refined.json"
    md_path = out_dir / "deep_dive_review_candidates_refined.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_to_markdown(payload, results_json=results_json), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive weak deep-dive review candidates from benchmark results.")
    parser.add_argument("--results-json", type=Path, required=True, help="Path to benchmark_test_results.json")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (defaults to sibling analysis/)")
    parser.add_argument("--weak-positive-threshold", type=float, default=1.20, help="Upper speedup bound for weak-positive candidates")
    parser.add_argument("--flat-threshold", type=float, default=1.05, help="Upper speedup bound for flat/negative candidates")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = write_outputs(
        results_json=args.results_json,
        output_dir=args.output_dir,
        weak_positive_threshold=args.weak_positive_threshold,
        flat_threshold=args.flat_threshold,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
