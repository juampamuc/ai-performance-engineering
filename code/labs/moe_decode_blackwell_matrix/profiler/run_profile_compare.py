from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch

from core.harness.benchmark_harness import lock_gpu_clocks

from ..artifact_io import read_jsonl, write_json
from ..matrix_types import MatrixScenario
from ..preflight import profiler_tool_status, query_app_clock_state, require_blackwell_cuda
from .capture import profile_scenario
from .compare import auto_select_graph_pair, compare_profiles


def _scenario_from_row(row: dict[str, object]) -> MatrixScenario:
    return MatrixScenario(
        playbook_name=str(row["playbook_name"]),
        description=str(row["description"]),
        seed=int(row["seed"]),
        dtype=str(row["dtype"]),
        hidden_size=int(row["hidden_size"]),
        intermediate_size=int(row["intermediate_size"]),
        steps=int(row["steps"]),
        warmup=int(row["warmup"]),
        repeats=int(row["repeats"]),
        num_experts=int(row["num_experts"]),
        top_k=int(row["top_k"]),
        decode_batch=int(row["decode_batch"]),
        routing_policy=str(row["routing_policy"]),
        schedule_mode=str(row["schedule_mode"]),
        launch_mode=str(row["launch_mode"]),
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config-a", default=None)
    parser.add_argument("--config-b", default=None)
    parser.add_argument("--top-ops", type=int, default=10)
    parser.add_argument("--output-name", default="auto_pair")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = args.run_dir.resolve()
    rows = read_jsonl(run_dir / "matrix.jsonl")
    require_blackwell_cuda(allow_non_blackwell=False)
    device = torch.device("cuda")

    if args.config_a and args.config_b:
        row_map = {row["config_id"]: row for row in rows}
        row_a = row_map[args.config_a]
        row_b = row_map[args.config_b]
    elif not args.config_a and not args.config_b:
        row_a, row_b = auto_select_graph_pair(rows)
    else:
        raise ValueError("--config-a and --config-b must be provided together")

    scenario_a = _scenario_from_row(row_a)
    scenario_b = _scenario_from_row(row_b)
    out_dir = run_dir / "profiles" / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    target_sm_clock = int(row_a.get("app_clock_sm_mhz") or row_b.get("app_clock_sm_mhz") or 0)
    target_mem_clock = int(row_a.get("app_clock_mem_mhz") or row_b.get("app_clock_mem_mhz") or 0)
    with lock_gpu_clocks(
        device=0,
        sm_clock_mhz=target_sm_clock or None,
        mem_clock_mhz=target_mem_clock or None,
    ) as _locked:
        clock_state = query_app_clock_state(0)
        profile_a = profile_scenario(
            scenario_a,
            device=device,
            trace_path=out_dir / f"{scenario_a.config_id}.trace.json",
            top_ops=args.top_ops,
        )
        profile_b = profile_scenario(
            scenario_b,
            device=device,
            trace_path=out_dir / f"{scenario_b.config_id}.trace.json",
            top_ops=args.top_ops,
        )

    summary = {
        "config_a": row_a,
        "config_b": row_b,
        "profile_a": profile_a,
        "profile_b": profile_b,
        "delta": compare_profiles(profile_a, profile_b),
        "clock_state": clock_state,
        "profiler_tools": profiler_tool_status(),
    }
    write_json(out_dir / "summary.json", summary)
    print(f"Profile comparison directory: {out_dir}")
    print(f"Config A: {scenario_a.config_id}")
    print(f"Config B: {scenario_b.config_id}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
