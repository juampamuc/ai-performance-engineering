#!/usr/bin/env python3
"""Derive a concrete `book-after` editing plan from final weak-case disposition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stays_within_current_narrative(target: str, disposition: str, narrative_decision: str) -> str:
    if target.endswith(":distributed") or target.endswith(":cublaslt_gemm_fp4"):
        return "No"
    if target.endswith(":nvfp4_mlp"):
        return "No"
    if narrative_decision == "qualify":
        return "Yes"
    if narrative_decision == "goal_specific":
        return "Yes"
    if disposition == "hold_expectations_reframe_narrative":
        return "Maybe"
    return "Maybe"


def _recommended_edit(row: Dict[str, Any]) -> str:
    target = str(row.get("target"))
    disposition = str(row.get("disposition"))
    narrative = str(row.get("narrative_decision"))

    if target.endswith(":distributed"):
        return "Scope explicitly to multi-GPU in this pass. Current result is a host mismatch, not an optimization miss."
    if target.endswith(":cublaslt_gemm_fp4"):
        return "Reframe as a capability-gated path on this stack in this pass; do not treat it as a replacement-search item yet."
    if target.endswith(":nvfp4_mlp"):
        return "Reframe consistently as a memory/precision tradeoff story. Do not present as a durable speed win."
    if narrative == "goal_specific":
        return "Judge it on its declared non-speed goal and remove raw-speedup framing from prose, tables, and callouts."
    if narrative == "qualify":
        return "Keep it in the chapter, but narrow the claim to a modest or context-dependent win. Avoid headline language."
    if disposition == "hold_expectations_reframe_narrative":
        return "Keep the topic, but demote this benchmark from headline-win language. Recast as cautionary, conditional, or replace with a stronger validated example."
    return "Manual review required."


def build_plan(final_disposition: Dict[str, Any]) -> Dict[str, Any]:
    rows = [row for row in final_disposition.get("rows", []) if row.get("narrative_decision") not in {None, "keep"}]
    rows.sort(key=lambda row: (str(row.get("chapter")), str(row.get("example"))))

    plan_rows: List[Dict[str, Any]] = []
    for row in rows:
        target = str(row.get("target"))
        entry = {
            "chapter": row.get("chapter"),
            "target": target,
            "disposition": row.get("disposition"),
            "stays_within_current_narrative": _stays_within_current_narrative(
                target=target,
                disposition=str(row.get("disposition")),
                narrative_decision=str(row.get("narrative_decision")),
            ),
            "recommended_edit": _recommended_edit(row),
        }
        plan_rows.append(entry)

    return {
        "original_run_id": final_disposition.get("original_run_id"),
        "total": len(plan_rows),
        "rows": plan_rows,
    }


def _to_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        "# Book-After Narrative Edit Plan",
        "",
        "Derived from:",
        "- `deep_dive_final_disposition.md`",
        "- `chapter_narrative_review_queue.md`",
        "- `deep_dive_expectation_refresh_summary.md`",
        "",
        "## Decision Rules",
        "",
        "- `refresh + keep`",
        "  Keep the benchmark and the current chapter story. No narrative demotion needed.",
        "- `refresh + qualify`",
        "  Keep the benchmark and refresh expectations, but rewrite the prose so it reads as a modest or context-dependent win instead of a headline improvement.",
        "- `hold + reframe`",
        "  Do not bless the current benchmark as a strong optimization win. Either demote the claim, change the claim dimension, or replace the benchmark in the chapter story.",
        "- `goal_specific`",
        "  Keep the example, but evaluate it against its declared non-speed goal rather than speedup.",
        "",
        "## Chapter Actions",
        "",
        "| Chapter | Target | Disposition | Stay Within Current Narrative? | Recommended Edit |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in payload.get("rows", []):
        lines.append(
            f"| `{row.get('chapter')}` | `{row.get('target').split(':', 1)[1]}` | `{row.get('disposition')}` | "
            f"`{row.get('stays_within_current_narrative')}` | {row.get('recommended_edit')} |"
        )
    lines.extend(
        [
            "",
            "## Recommended Editing Order",
            "",
            "1. Update chapter prose/callouts for the `qualify` cases so the refreshed expectations and the narrative match.",
            "2. Reframe or demote the flat/weak local cases.",
            "3. Keep hardware/capability-gated examples explicitly scoped rather than claiming an optimization miss.",
            "4. Apply one consistent family-level narrative decision for shared-family cases like `nvfp4_mlp`.",
            "5. Keep non-speed examples out of speedup tables and speed-focused prose.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(*, final_disposition_json: Path, output_dir: Path | None = None) -> Dict[str, Path]:
    payload = build_plan(_load_json(final_disposition_json))
    out_dir = output_dir or final_disposition_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "book_after_narrative_edit_plan.json"
    md_path = out_dir / "book_after_narrative_edit_plan.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_to_markdown(payload), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive a concrete book-after editing plan from final deep-dive disposition.")
    parser.add_argument("--final-disposition-json", type=Path, required=True, help="Path to deep_dive_final_disposition.json")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (defaults to the disposition directory)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = write_outputs(final_disposition_json=args.final_disposition_json, output_dir=args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
