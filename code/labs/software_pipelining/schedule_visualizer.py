"""Render deterministic software-pipelining schedules as ASCII or JSON."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from labs.software_pipelining.pipeline_graph import (
    PipelineExample,
    get_pipeline_example,
    list_pipeline_examples,
    pipeline_example_to_dict,
    validate_schedule,
)


def render_json_payload(example: PipelineExample) -> dict:
    return pipeline_example_to_dict(example)


def render_ascii(example: PipelineExample) -> str:
    validation = validate_schedule(example)
    max_time = max(slot.time for slot in example.schedule)
    slots_by_unit_time: dict[tuple[str, int], str] = {}
    units = []
    for node in example.nodes:
        if node.unit not in units:
            units.append(node.unit)
    for slot in example.schedule:
        slots_by_unit_time[(next(node.unit for node in example.nodes if node.name == slot.node), slot.time)] = (
            f"{slot.node}[{slot.iteration}]"
        )

    phase_spans: dict[str, list[int]] = defaultdict(list)
    for slot in example.schedule:
        phase_spans[slot.phase].append(slot.time)

    lines = [
        f"Example: {example.name}",
        example.description,
        "",
        f"Stages: {example.stage_count}",
        f"Iterations: {example.iteration_count}",
        f"Validation: {'PASS' if validation.is_valid else 'FAIL'}",
    ]
    if validation.errors:
        lines.append("Errors:")
        for error in validation.errors:
            lines.append(f"  - {error}")
    lines.extend(
        [
            "",
            "Phase spans:",
        ]
    )
    for phase in ("prologue", "steady", "epilogue"):
        points = phase_spans.get(phase, [])
        if not points:
            continue
        lines.append(f"  - {phase}: t={min(points)}..{max(points)}")

    lines.extend(["", "Dependency summary:"])
    for edge in example.edges:
        lines.append(
            f"  - {edge.kind}: {edge.src}(i) -> {edge.dst}(i+{edge.distance}) "
            f"with min_delay={edge.min_delay}"
        )

    header_cells = ["unit".ljust(12)] + [f"t{time}".ljust(18) for time in range(max_time + 1)]
    lines.extend(["", "Schedule:", " | ".join(header_cells)])
    lines.append("-" * len(lines[-1]))

    for unit in units:
        row = [unit.ljust(12)]
        for time in range(max_time + 1):
            row.append(slots_by_unit_time.get((unit, time), ".").ljust(18))
        lines.append(" | ".join(row))

    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a deterministic software-pipelining schedule example."
    )
    parser.add_argument(
        "--example",
        choices=list_pipeline_examples(),
        default="gemm_mainloop",
        help="Named schedule example to render.",
    )
    parser.add_argument(
        "--format",
        choices=("ascii", "json"),
        default="ascii",
        help="Output format.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    example = get_pipeline_example(args.example)
    if args.format == "json":
        print(json.dumps(render_json_payload(example), indent=2, sort_keys=True))
        return
    print(render_ascii(example))


if __name__ == "__main__":
    main()
