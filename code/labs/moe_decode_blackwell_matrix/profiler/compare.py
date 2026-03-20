from __future__ import annotations

from typing import Any, Sequence


def auto_select_graph_pair(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    eager_by_workload = {
        row["workload_key"]: row
        for row in ok_rows
        if row["schedule_mode"] == "persistent" and row["launch_mode"] == "eager"
    }
    graph_rows = [
        row
        for row in ok_rows
        if row["schedule_mode"] == "persistent" and row["launch_mode"] == "cuda_graph"
    ]
    ranked_pairs = []
    for graph_row in graph_rows:
        eager_row = eager_by_workload.get(graph_row["workload_key"])
        if eager_row is None:
            continue
        speedup = float(eager_row["step_mean_ms"]) / float(graph_row["step_mean_ms"])
        ranked_pairs.append((speedup, eager_row, graph_row))
    if not ranked_pairs:
        raise RuntimeError("No persistent eager vs cuda_graph pair found in the run directory")
    ranked_pairs.sort(key=lambda item: item[0], reverse=True)
    _speedup, eager_row, graph_row = ranked_pairs[0]
    return eager_row, graph_row


def compare_profiles(
    profile_a: dict[str, Any],
    profile_b: dict[str, Any],
) -> dict[str, Any]:
    return {
        "config_a": profile_a["config_id"],
        "config_b": profile_b["config_id"],
        "delta_total_self_cuda_time_us": round(
            profile_b["total_self_cuda_time_us"] - profile_a["total_self_cuda_time_us"],
            3,
        ),
        "delta_total_cuda_time_us": round(
            profile_b["total_cuda_time_us"] - profile_a["total_cuda_time_us"],
            3,
        ),
        "delta_total_cpu_time_us": round(
            profile_b["total_cpu_time_us"] - profile_a["total_cpu_time_us"],
            3,
        ),
    }
