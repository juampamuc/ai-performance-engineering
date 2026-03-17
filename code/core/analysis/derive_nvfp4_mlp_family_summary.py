#!/usr/bin/env python3
"""Derive a canonical NVFP4 MLP family summary from benchmark results."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_family_rows(results_payload: Dict[str, Any], *, source_name: str) -> Iterable[Dict[str, Any]]:
    for chapter_result in results_payload.get("results", []):
        chapter = chapter_result.get("chapter")
        for bench in chapter_result.get("benchmarks", []):
            if bench.get("example") != "nvfp4_mlp":
                continue
            optimizations = list(bench.get("optimizations", []))
            best_opt = optimizations[0] if optimizations else {}
            baseline_ncu = ((bench.get("baseline_profiler_metrics") or {}).get("ncu") or {})
            optimized_ncu = ((best_opt.get("optimized_profiler_metrics") or {}).get("ncu") or {})
            yield {
                "source_results": source_name,
                "chapter": chapter,
                "example": bench.get("example"),
                "optimization_goal": bench.get("optimization_goal", "speed"),
                "status": bench.get("status"),
                "baseline_time_ms": bench.get("baseline_time_ms"),
                "optimized_time_ms": best_opt.get("time_ms"),
                "speedup": best_opt.get("speedup"),
                "baseline_memory_mb": bench.get("baseline_memory_mb"),
                "optimized_memory_mb": best_opt.get("memory_mb"),
                "memory_savings_pct": best_opt.get("memory_savings_pct"),
                "optimized_file": best_opt.get("file"),
                "baseline_kernel_time_ms": baseline_ncu.get("kernel_time_ms"),
                "optimized_kernel_time_ms": optimized_ncu.get("kernel_time_ms"),
                "baseline_occupancy": baseline_ncu.get("occupancy"),
                "optimized_occupancy": optimized_ncu.get("occupancy"),
                "baseline_sm_throughput_pct": baseline_ncu.get("sm_throughput_percent"),
                "optimized_sm_throughput_pct": optimized_ncu.get("sm_throughput_percent"),
            }


def build_summary(results_payload: Dict[str, Any]) -> Dict[str, Any]:
    return build_summary_from_sources([("inline", results_payload)])


def build_summary_from_sources(sources: Iterable[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    source_names: List[str] = []
    for source_name, payload in sources:
        source_names.append(source_name)
        rows.extend(_iter_family_rows(payload, source_name=source_name))
    rows = sorted(rows, key=lambda row: (str(row.get("chapter")), str(row.get("source_results"))))
    if not rows:
        return {
            "total": 0,
            "rows": [],
            "decision": "missing_family_rows",
            "summary": {},
            "sources": source_names,
        }

    goals = Counter(str(row.get("optimization_goal") or "speed") for row in rows)
    memory_rows = [row for row in rows if row.get("memory_savings_pct") is not None]
    speed_rows = [row for row in rows if row.get("speedup") is not None]
    kernel_delta_rows = [
        row for row in rows
        if row.get("baseline_kernel_time_ms") is not None and row.get("optimized_kernel_time_ms") is not None
    ]
    occupancy_delta_rows = [
        row for row in rows
        if row.get("baseline_occupancy") is not None and row.get("optimized_occupancy") is not None
    ]

    avg_speedup = mean(float(row["speedup"]) for row in speed_rows) if speed_rows else None
    avg_memory_savings_pct = (
        mean(float(row["memory_savings_pct"]) for row in memory_rows) if memory_rows else None
    )
    avg_kernel_delta_ms = (
        mean(float(row["optimized_kernel_time_ms"]) - float(row["baseline_kernel_time_ms"]) for row in kernel_delta_rows)
        if kernel_delta_rows
        else None
    )
    avg_occupancy_delta = (
        mean(float(row["optimized_occupancy"]) - float(row["baseline_occupancy"]) for row in occupancy_delta_rows)
        if occupancy_delta_rows
        else None
    )

    if goals.get("memory", 0) == len(rows) and (avg_memory_savings_pct or 0.0) >= 20.0:
        decision = "memory_tradeoff_story"
    elif (avg_speedup or 0.0) > 1.20:
        decision = "speed_story"
    else:
        decision = "hold_and_reframe"

    return {
        "total": len(rows),
        "sources": source_names,
        "decision": decision,
        "summary": {
            "optimization_goal_counts": dict(goals),
            "avg_speedup": avg_speedup,
            "avg_memory_savings_pct": avg_memory_savings_pct,
            "avg_kernel_delta_ms": avg_kernel_delta_ms,
            "avg_occupancy_delta": avg_occupancy_delta,
        },
        "rows": rows,
    }


def _to_markdown(summary: Dict[str, Any], *, results_jsons: List[Path]) -> str:
    lines = [
        "# NVFP4 MLP Family Summary",
        "",
        f"- Source results: `{', '.join(str(path) for path in results_jsons)}`",
        f"- Family rows: `{summary.get('total', 0)}`",
        f"- Decision: `{summary.get('decision', 'unknown')}`",
        f"- Aggregate summary: `{summary.get('summary', {})}`",
        "",
        "## Rows",
        "",
        "| Source | Target | Goal | Baseline ms | Optimized ms | Speedup | Memory Saved | Baseline NCU kernel ms | Optimized NCU kernel ms |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.get("rows", []):
        lines.append(
            f"| `{row.get('source_results')}` | `{row.get('chapter')}:{row.get('example')}` | `{row.get('optimization_goal')}` | "
            f"`{row.get('baseline_time_ms')}` | `{row.get('optimized_time_ms')}` | "
            f"`{row.get('speedup')}` | `{row.get('memory_savings_pct')}` | "
            f"`{row.get('baseline_kernel_time_ms')}` | `{row.get('optimized_kernel_time_ms')}` |"
        )
    lines.append("")
    return "\n".join(lines)


def write_outputs(*, results_jsons: Iterable[Path], output_dir: Path | None = None) -> Dict[str, Path]:
    materialized_results = list(results_jsons)
    sources = [(str(path), _load_json(path)) for path in materialized_results]
    summary = build_summary_from_sources(sources)
    first_results = materialized_results[0]
    out_dir = output_dir or first_results.parent.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "nvfp4_mlp_family_summary.json"
    md_path = out_dir / "nvfp4_mlp_family_summary.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_to_markdown(summary, results_jsons=materialized_results), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the shared NVFP4 MLP family from one or more benchmark results.")
    parser.add_argument("--results-json", type=Path, action="append", required=True, help="Path to benchmark_test_results.json (repeatable)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (defaults to sibling analysis/)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = write_outputs(results_jsons=args.results_json, output_dir=args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
