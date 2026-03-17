"""Shared workload metrics for Chapter 12 launch and graph benchmarks."""

from __future__ import annotations

from typing import Dict, Optional


def compute_ch12_workload_metrics(
    *,
    uses_cuda_graph: bool,
    num_iterations: int,
    workload_elements: Optional[float] = None,
    num_nodes: Optional[int] = None,
    conditional_support: Optional[bool] = None,
    bytes_per_iteration: Optional[float] = None,
) -> Dict[str, float]:
    """Return structural graph metrics without inventing timing deltas."""
    metrics: Dict[str, float] = {
        "cuda_runtime.uses_cuda_graph": float(uses_cuda_graph),
        "cuda_runtime.num_iterations": float(num_iterations),
    }
    if workload_elements is not None:
        metrics["cuda_runtime.workload_elements"] = float(workload_elements)
    if num_nodes is not None:
        metrics["cuda_runtime.num_nodes"] = float(num_nodes)
    if conditional_support is not None:
        metrics["cuda_runtime.conditional_support"] = float(conditional_support)
    if bytes_per_iteration is not None:
        metrics["cuda_runtime.bytes_per_iteration"] = float(bytes_per_iteration)
    return metrics
