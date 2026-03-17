#!/usr/bin/env python3
"""Derive root-cause ledgers for weak deep-dive review candidates."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_rows(path: Path) -> Tuple[str | None, List[Dict[str, Any]]]:
    payload = _load_json(path)
    if isinstance(payload, dict):
        return payload.get("run_id"), list(payload.get("rows", []))
    if isinstance(payload, list):
        return None, list(payload)
    raise TypeError(f"Unsupported payload type for {path}: {type(payload)!r}")


def _family_summary_map(paths: Iterable[Path]) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        payload = _load_json(path)
        rows = list(payload.get("rows", []))
        if not rows:
            continue
        family_name = str(rows[0].get("example") or path.stem)
        mapping[family_name] = payload
    return mapping


def derive_rows(
    *,
    review_rows: Iterable[Dict[str, Any]],
    family_summaries: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in review_rows]
    by_example: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_example.setdefault(str(row.get("example")), []).append(row)

    derived: List[Dict[str, Any]] = []

    nvfp4_family = by_example.get("nvfp4_mlp", [])
    if nvfp4_family:
        summary = family_summaries.get("nvfp4_mlp", {})
        decision = str(summary.get("decision") or "hold_and_reframe")
        summary_stats = dict(summary.get("summary", {}))
        avg_memory = summary_stats.get("avg_memory_savings_pct")
        avg_speedup = summary_stats.get("avg_speedup")
        category = "repeated_family"
        root_cause = (
            "All chapters use the same shared NVFP4 MLP benchmark shape and the same Transformer Engine NVFP4 path. "
            "On this environment the optimized path shows near-identical kernel metrics and no durable end-to-end win "
            "over the BF16 baseline."
        )
        next_action = (
            "Hold expectations/story across the family. Retune shape or replace with a stronger NVFP4 example before "
            "blessing any chapter instance."
        )
        if decision == "memory_tradeoff_story":
            category = "shared_family_non_speed_tradeoff"
            root_cause = (
                "All chapters use the same shared NVFP4 MLP benchmark shape and the same Transformer Engine NVFP4 path. "
                "The family does not produce a durable latency win here, but it does deliver a stable memory reduction "
                "with nearly identical kernel behavior, so the validated story is memory/capability tradeoff rather than speed."
            )
            next_action = (
                "Refresh expectations against the memory goal, keep the family out of speedup claims, and reframe all chapter "
                "copies consistently as a memory/precision tradeoff."
            )
        evidence = [
            "shared implementation: core/benchmark/nvfp4_mlp.py",
            f"family summary decision: {decision}",
        ]
        if avg_speedup is not None:
            evidence.append(f"average family speedup: {avg_speedup:.4f}x")
        if avg_memory is not None:
            evidence.append(f"average family memory savings: {avg_memory:.2f}%")
        derived.append(
            {
                "category": category,
                "targets": sorted(f"{row.get('chapter')}:{row.get('example')}" for row in nvfp4_family),
                "root_cause": root_cause,
                "evidence": evidence,
                "next_action": next_action,
            }
        )

    distributed_rows = [
        row for row in rows if str(row.get("chapter")) == "ch05" and str(row.get("example")) == "distributed"
    ]
    if distributed_rows:
        derived.append(
            {
                "category": "no_winner",
                "targets": ["ch05:distributed"],
                "root_cause": (
                    "The optimized path is intentionally multi-GPU/distributed and is skipped on this 1-GPU host, while "
                    "the baseline still runs as a single-GPU host-staged reduction. This is not an optimization miss; it is "
                    "a scope/hardware mismatch for the current environment."
                ),
                "evidence": [
                    "optimized result status=skipped on 1 GPU",
                    "source: ch05/optimized_distributed.py requires multi-GPU scope",
                ],
                "next_action": (
                    "Keep expectations and story on hold for single-GPU evaluation. Reclassify as hardware-scoped or rerun "
                    "on multi-GPU hardware before treating it as a performance comparison."
                ),
            }
        )

    cublaslt_rows = [
        row for row in rows if str(row.get("chapter")) == "ch09" and str(row.get("example")) == "cublaslt_gemm_fp4"
    ]
    if cublaslt_rows:
        derived.append(
            {
                "category": "no_winner",
                "targets": ["ch09:cublaslt_gemm_fp4"],
                "root_cause": (
                    "The optimized cuBLASLt NVFP4 algorithm is unavailable on the current driver/toolchain stack, so no "
                    "optimized winner is produced."
                ),
                "evidence": [
                    "optimized result status=skipped",
                    "optimized cuBLASLt NVFP4 path unavailable on current stack",
                ],
                "next_action": (
                    "Do not refresh expectations here. Treat it as a capability-gated example and rerun only after the "
                    "NVFP4 cuBLASLt path is available on the target stack."
                ),
            }
        )

    derived.sort(key=lambda row: tuple(row.get("targets", [])))
    return derived


def _to_markdown(*, generated_on: str, rows: List[Dict[str, Any]]) -> str:
    lines = [
        "# Deep-Dive Weak Case Root Causes",
        "",
        f"Generated: {generated_on}",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {', '.join(row.get('targets', []))}",
                "",
                f"- Category: `{row.get('category')}`",
                f"- Targets: `{', '.join(row.get('targets', []))}`",
                f"- Root cause: {row.get('root_cause')}",
                "- Evidence:",
            ]
        )
        for item in row.get("evidence", []):
            lines.append(f"  - {item}")
        lines.append(f"- Next action: {row.get('next_action')}")
        lines.append("")
    return "\n".join(lines)


def write_outputs(
    *,
    review_candidates_json: Path,
    family_summary_jsons: Iterable[Path] = (),
    output_dir: Path | None = None,
) -> Dict[str, Path]:
    run_id, rows = _load_rows(review_candidates_json)
    family_summaries = _family_summary_map(family_summary_jsons)
    derived_rows = derive_rows(review_rows=rows, family_summaries=family_summaries)
    out_dir = output_dir or review_candidates_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    generated_on = datetime.now(tz=UTC).strftime("%Y-%m-%d UTC")
    payload = {
        "run_id": run_id,
        "generated_on": generated_on,
        "rows": derived_rows,
    }
    json_path = out_dir / "deep_dive_weak_case_root_causes.json"
    md_path = out_dir / "deep_dive_weak_case_root_causes.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_to_markdown(generated_on=generated_on, rows=derived_rows), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive root causes for weak deep-dive review candidates.")
    parser.add_argument("--review-candidates-json", type=Path, required=True, help="Path to deep_dive_review_candidates_refined.json")
    parser.add_argument("--family-summary-json", type=Path, action="append", default=[], help="Optional family summary JSON (repeatable)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (defaults to the review candidates directory)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = write_outputs(
        review_candidates_json=args.review_candidates_json,
        family_summary_jsons=args.family_summary_json,
        output_dir=args.output_dir,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
