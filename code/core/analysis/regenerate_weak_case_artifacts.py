#!/usr/bin/env python3
"""Regenerate weak-case analysis artifacts from base and focused rerun evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from core.analysis.derive_book_after_narrative_edit_plan import write_outputs as write_book_plan
from core.analysis.derive_chapter_narrative_review_queue import write_outputs as write_review_queue
from core.analysis.derive_deep_dive_expectation_refresh_summary import (
    write_outputs as write_expectation_refresh_summary,
)
from core.analysis.derive_deep_dive_final_disposition import write_outputs as write_final_disposition
from core.analysis.derive_deep_dive_review_candidates import write_outputs as write_review_candidates
from core.analysis.derive_nvfp4_mlp_family_summary import write_outputs as write_family_summary
from core.analysis.derive_weak_case_actions import write_outputs as write_weak_actions
from core.analysis.derive_weak_case_root_causes import write_outputs as write_root_causes
from core.analysis.overlay_benchmark_results import write_outputs as write_overlay_results


def _load_focus_targets(path: Path) -> Set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    targets: Set[str] = set()
    for rows in payload.get("by_chapter", {}).values():
        for row in rows:
            target = row.get("target")
            if target:
                targets.add(str(target))
    return targets


def regenerate(
    *,
    base_results_json: Path,
    final_failure_ledger_json: Path,
    output_dir: Path,
    overlay_results_jsons: Iterable[Path] = (),
    apply_summary_jsons: Iterable[Path] = (),
    focus_review_queue_json: Optional[Path] = None,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_list = list(overlay_results_jsons)
    merged_results_json = output_dir / "merged_benchmark_test_results.json"
    if overlay_list:
        write_overlay_results(
            base_results_json=base_results_json,
            overlay_results_jsons=overlay_list,
            output_json=merged_results_json,
        )
    else:
        merged_results_json.write_text(base_results_json.read_text(encoding="utf-8"), encoding="utf-8")

    target_filter = _load_focus_targets(focus_review_queue_json) if focus_review_queue_json else None
    review_outputs = write_review_candidates(
        results_json=merged_results_json,
        output_dir=output_dir,
        target_filter=target_filter,
    )
    weak_actions_outputs = write_weak_actions(review_candidates_json=review_outputs["json"], output_dir=output_dir)
    family_summary_outputs = write_family_summary(results_jsons=overlay_list or [base_results_json], output_dir=output_dir)
    root_cause_outputs = write_root_causes(
        review_candidates_json=review_outputs["json"],
        family_summary_jsons=[family_summary_outputs["json"]],
        output_dir=output_dir,
    )
    final_disposition_outputs = write_final_disposition(
        final_failure_ledger_json=final_failure_ledger_json,
        weak_actions_json=weak_actions_outputs["json"],
        weak_root_causes_json=root_cause_outputs["json"],
        output_dir=output_dir,
    )
    review_queue_outputs = write_review_queue(
        final_disposition_json=final_disposition_outputs["json"],
        output_dir=output_dir,
    )
    book_plan_outputs = write_book_plan(
        final_disposition_json=final_disposition_outputs["json"],
        output_dir=output_dir,
    )

    outputs: Dict[str, Path] = {
        "merged_results_json": merged_results_json,
        "review_candidates_json": review_outputs["json"],
        "review_candidates_markdown": review_outputs["markdown"],
        "weak_actions_json": weak_actions_outputs["json"],
        "weak_actions_markdown": weak_actions_outputs["markdown"],
        "family_summary_json": family_summary_outputs["json"],
        "family_summary_markdown": family_summary_outputs["markdown"],
        "weak_root_causes_json": root_cause_outputs["json"],
        "weak_root_causes_markdown": root_cause_outputs["markdown"],
        "final_disposition_json": final_disposition_outputs["json"],
        "final_disposition_markdown": final_disposition_outputs["markdown"],
        "review_queue_json": review_queue_outputs["json"],
        "review_queue_markdown": review_queue_outputs["markdown"],
        "book_plan_json": book_plan_outputs["json"],
        "book_plan_markdown": book_plan_outputs["markdown"],
    }

    apply_summaries = list(apply_summary_jsons)
    if apply_summaries:
        expectation_outputs = write_expectation_refresh_summary(
            final_disposition_json=final_disposition_outputs["json"],
            apply_summary_jsons=apply_summaries,
            output_dir=output_dir,
        )
        outputs["expectation_refresh_summary_json"] = expectation_outputs["json"]
        outputs["expectation_refresh_summary_markdown"] = expectation_outputs["markdown"]

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate weak-case analysis artifacts from base and focused rerun results.")
    parser.add_argument("--base-results-json", type=Path, required=True, help="Canonical benchmark_test_results.json")
    parser.add_argument("--final-failure-ledger-json", type=Path, required=True, help="Path to final_failure_ledger.json")
    parser.add_argument("--overlay-results-json", type=Path, action="append", default=[], help="Focused rerun benchmark_test_results.json (repeatable)")
    parser.add_argument("--apply-summary-json", type=Path, action="append", default=[], help="Expectation refresh apply summary JSON (repeatable)")
    parser.add_argument("--focus-review-queue-json", type=Path, default=None, help="Optional prior chapter_narrative_review_queue.json used to lock review scope")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for regenerated artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = regenerate(
        base_results_json=args.base_results_json,
        final_failure_ledger_json=args.final_failure_ledger_json,
        overlay_results_jsons=args.overlay_results_json,
        apply_summary_jsons=args.apply_summary_json,
        focus_review_queue_json=args.focus_review_queue_json,
        output_dir=args.output_dir,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
