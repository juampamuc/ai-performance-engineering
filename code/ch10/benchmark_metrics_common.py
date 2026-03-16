"""Shared custom-metric helpers for Chapter 10 benchmarks."""

from __future__ import annotations

from typing import Mapping, Any

from core.benchmark.metrics import compute_bandwidth_metrics


def _bool_metric(value: bool) -> float:
    return 1.0 if value else 0.0


def compute_workload_param_metrics(params: Mapping[str, Any], *, prefix: str = "workload") -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in params.items():
        if isinstance(value, bool):
            metrics[f"{prefix}.{key}"] = _bool_metric(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[f"{prefix}.{key}"] = float(value)
    return metrics


def compute_pipeline_variant_metrics(
    params: Mapping[str, Any],
    *,
    num_stages: int,
    microbatches: int = 1,
    **flags: bool | int | float,
) -> dict[str, float]:
    metrics = compute_workload_param_metrics(params)
    metrics["pipeline.num_stages"] = float(num_stages)
    metrics["pipeline.microbatches"] = float(microbatches)
    for key, value in flags.items():
        if isinstance(value, bool):
            metrics[f"pipeline.{key}"] = _bool_metric(value)
        else:
            metrics[f"pipeline.{key}"] = float(value)
    return metrics


def compute_batch_workload_metrics(
    *,
    total_batch_size: int,
    micro_batch_size: int,
    micro_batches: int,
    hidden_dim: int,
    ffn_dim: int,
) -> dict[str, float]:
    return {
        "batch.total_batch_size": float(total_batch_size),
        "batch.micro_batch_size": float(micro_batch_size),
        "batch.micro_batches": float(micro_batches),
        "batch.hidden_dim": float(hidden_dim),
        "batch.ffn_dim": float(ffn_dim),
        "batch.activation_elements": float(total_batch_size * hidden_dim),
    }


def compute_reduction_workload_metrics(
    *,
    num_elements: int,
    bytes_per_element: int = 4,
    elapsed_ms: float | None = None,
    **flags: bool | int | float,
) -> dict[str, float]:
    total_bytes = float(num_elements) * float(bytes_per_element)
    metrics = {
        "reduction.num_elements": float(num_elements),
        "reduction.bytes_per_element": float(bytes_per_element),
        "reduction.total_bytes": total_bytes,
    }
    metrics.update(compute_bandwidth_metrics(total_bytes=total_bytes, elapsed_ms=elapsed_ms))
    for key, value in flags.items():
        if isinstance(value, bool):
            metrics[f"reduction.{key}"] = _bool_metric(value)
        else:
            metrics[f"reduction.{key}"] = float(value)
    return metrics
