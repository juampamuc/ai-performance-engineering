"""Distributed CLI commands wired to the unified PerformanceEngine."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

try:
    import torch
except Exception:  # pragma: no cover - torch may be missing in docs-only envs
    torch = None  # type: ignore

from core.engine import get_engine


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _print_result(result: Dict[str, Any], json_output: bool) -> None:
    if json_output:
        print(json.dumps(result, indent=2, default=_json_default))
        return
    print(json.dumps(result, indent=2, default=_json_default))


def _resolve_model_preset(model_size: float) -> str:
    known_presets = {
        7.0: "llama-7b",
        8.0: "llama-3.1-8b",
        13.0: "llama-13b",
        70.0: "llama-70b",
        405.0: "llama-3.1-405b",
    }
    closest = min(known_presets, key=lambda candidate: abs(candidate - model_size))
    return known_presets[closest]


def _detect_gpu_memory_gb(default: float = 80.0) -> float:
    if torch is None or not torch.cuda.is_available():
        return default
    props = torch.cuda.get_device_properties(0)
    return round(props.total_memory / (1024 ** 3), 2)


def plan_parallelism(args: Any) -> int:
    """Plan TP/PP/DP strategy for a model size and cluster shape."""
    result = get_engine().distributed.plan(
        model_size=float(getattr(args, "model_size", 7.0)),
        gpus=int(getattr(args, "gpus", 8)),
        nodes=int(getattr(args, "nodes", 1)),
    )
    _print_result(result, getattr(args, "json", False))
    return 0


def topology(args: Any) -> int:
    """Show detected parallelism topology information."""
    from core.optimization.parallelism_planner import TopologyDetector

    try:
        result = TopologyDetector().detect().to_dict()
        result["available"] = True
    except Exception as exc:
        result = {
            "available": False,
            "error": str(exc),
        }
    _print_result(result, getattr(args, "json", False))
    return 0 if result.get("available", True) else 1


def nccl_tuning(args: Any) -> int:
    """Print NCCL tuning recommendations."""
    result = get_engine().distributed.nccl(
        nodes=int(getattr(args, "nodes", 1)),
        gpus=int(getattr(args, "gpus", 8)),
        diagnose=bool(getattr(args, "diagnose", False)),
    )
    _print_result(result, getattr(args, "json", False))
    return 0 if result.get("available", True) else 1


def zero_config(args: Any) -> int:
    """Print ZeRO/FSDP recommendations for the requested model size."""
    model_size = float(getattr(args, "model_size", 70.0))
    preset = _resolve_model_preset(model_size)

    from core.optimization.parallelism_planner import ModelAnalyzer
    from core.optimization.parallelism_planner.sharding_strategies import ShardingOptimizer

    analyzer = ModelAnalyzer()
    optimizer = ShardingOptimizer()
    model = analyzer.analyze(preset)
    recommendations = optimizer.recommend(
        model=model,
        dp_size=int(getattr(args, "gpus", 8)),
        gpu_memory_gb=_detect_gpu_memory_gb(),
    )
    result = {
        "success": True,
        "input_model_size_b": model_size,
        "model_preset_used": preset,
        "recommendations": [item.to_dict() for item in recommendations],
    }
    _print_result(result, getattr(args, "json", False))
    return 0 if result.get("success", True) else 1
