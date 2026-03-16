"""Shared custom-metric helpers for Chapter 5 benchmarks."""

from __future__ import annotations


def _bool_metric(value: bool) -> float:
    return 1.0 if value else 0.0


def compute_storage_path_metrics(
    *,
    bytes_read: int,
    bytes_written: int,
    file_count: int,
    uses_cpu_staging: bool,
    simulates_gpu_direct: bool,
) -> dict[str, float]:
    return {
        "storage.bytes_read": float(bytes_read),
        "storage.bytes_written": float(bytes_written),
        "storage.file_count": float(file_count),
        "storage.uses_cpu_staging": _bool_metric(uses_cpu_staging),
        "storage.simulates_gpu_direct": _bool_metric(simulates_gpu_direct),
    }


def compute_host_reduction_metrics(
    *,
    num_elements: int,
    host_staging_round_trips: int,
    keeps_reduction_on_device: bool,
) -> dict[str, float]:
    return {
        "reduction.num_elements": float(num_elements),
        "reduction.host_staging_round_trips": float(host_staging_round_trips),
        "reduction.keeps_reduction_on_device": _bool_metric(keeps_reduction_on_device),
        "reduction.logical_bytes": float(num_elements * 4),
    }


def compute_vectorization_metrics(
    *,
    num_elements: int,
    chunk_elements: int,
    is_vectorized: bool,
) -> dict[str, float]:
    chunk_elements = max(int(chunk_elements), 1)
    return {
        "vectorization.num_elements": float(num_elements),
        "vectorization.chunk_elements": float(chunk_elements),
        "vectorization.chunk_count": float((int(num_elements) + chunk_elements - 1) // chunk_elements),
        "vectorization.is_vectorized": _bool_metric(is_vectorized),
    }


def compute_decompression_metrics(
    *,
    run_count: int,
    run_length: int,
    decompressed_elements: int,
    runs_on_device: bool,
) -> dict[str, float]:
    compressed_elements = max(int(run_count), 1)
    return {
        "compression.run_count": float(run_count),
        "compression.run_length": float(run_length),
        "compression.decompressed_elements": float(decompressed_elements),
        "compression.expansion_ratio": float(decompressed_elements) / float(compressed_elements),
        "compression.runs_on_device": _bool_metric(runs_on_device),
    }


def compute_multi_gpu_reduction_metrics(
    *,
    num_elements_per_gpu: int,
    device_count: int,
    uses_cpu_staging: bool,
) -> dict[str, float]:
    total_elements = int(num_elements_per_gpu) * max(int(device_count), 1)
    return {
        "distributed.device_count": float(device_count),
        "distributed.elements_per_gpu": float(num_elements_per_gpu),
        "distributed.total_elements": float(total_elements),
        "distributed.uses_cpu_staging": _bool_metric(uses_cpu_staging),
        "distributed.logical_bytes": float(total_elements * 4),
    }
