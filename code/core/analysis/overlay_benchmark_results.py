#!/usr/bin/env python3
"""Overlay focused benchmark results onto a canonical results JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_run_id(path: Path) -> str | None:
    parts = path.parts
    for idx, part in enumerate(parts[:-1]):
        if part == "runs" and idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def _chapter_map(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for chapter in payload.get("results", []):
        chapter_name = str(chapter.get("chapter"))
        benchmark_map = {
            str(bench.get("example")): bench
            for bench in chapter.get("benchmarks", [])
            if bench.get("example") is not None
        }
        mapping[chapter_name] = {"chapter_payload": chapter, "benchmarks": benchmark_map}
    return mapping


def overlay_results(base_payload: Dict[str, Any], overlays: Iterable[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    result = json.loads(json.dumps(base_payload))
    base_map = _chapter_map(result)
    source_runs: List[str] = []

    for source_name, overlay in overlays:
        source_runs.append(source_name)
        for overlay_chapter in overlay.get("results", []):
            chapter_name = str(overlay_chapter.get("chapter"))
            if chapter_name not in base_map:
                chapter_payload = {"chapter": chapter_name, "status": overlay_chapter.get("status"), "benchmarks": []}
                result.setdefault("results", []).append(chapter_payload)
                base_map[chapter_name] = {"chapter_payload": chapter_payload, "benchmarks": {}}
            chapter_payload = base_map[chapter_name]["chapter_payload"]
            benchmark_map = base_map[chapter_name]["benchmarks"]
            bench_list = list(chapter_payload.get("benchmarks", []))
            bench_index = {str(bench.get("example")): idx for idx, bench in enumerate(bench_list) if bench.get("example") is not None}

            for overlay_bench in overlay_chapter.get("benchmarks", []):
                example = str(overlay_bench.get("example"))
                if example in bench_index:
                    bench_list[bench_index[example]] = overlay_bench
                else:
                    bench_list.append(overlay_bench)
                benchmark_map[example] = overlay_bench

            chapter_payload["benchmarks"] = bench_list

    result["overlay_sources"] = source_runs
    return result


def write_outputs(
    *,
    base_results_json: Path,
    overlay_results_jsons: Iterable[Path],
    output_json: Path,
) -> Path:
    overlay_paths = list(overlay_results_jsons)
    overlays = [(str(path), _load_json(path)) for path in overlay_paths]
    merged = overlay_results(_load_json(base_results_json), overlays)
    base_run_id = _infer_run_id(base_results_json)
    source_run_ids: List[str] = []
    if base_run_id:
        source_run_ids.append(base_run_id)
    for path in overlay_paths:
        run_id = _infer_run_id(path)
        if run_id and run_id not in source_run_ids:
            source_run_ids.append(run_id)
    merged["base_results_json"] = str(base_results_json)
    merged["base_run_id"] = base_run_id
    merged["overlay_results_jsons"] = [str(path) for path in overlay_paths]
    merged["source_results_jsons"] = [str(base_results_json), *[str(path) for path in overlay_paths]]
    merged["source_run_ids"] = source_run_ids
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    return output_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay focused benchmark results onto a canonical benchmark_test_results.json.")
    parser.add_argument("--base-results-json", type=Path, required=True, help="Canonical benchmark_test_results.json")
    parser.add_argument("--overlay-results-json", type=Path, action="append", required=True, help="Focused rerun benchmark_test_results.json (repeatable)")
    parser.add_argument("--output-json", type=Path, required=True, help="Path to write the merged benchmark_test_results.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = write_outputs(
        base_results_json=args.base_results_json,
        overlay_results_jsons=args.overlay_results_json,
        output_json=args.output_json,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
