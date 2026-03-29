import json
from pathlib import Path

from core.benchmark.expectations import ExpectationEntry, RunProvenance
from core.analysis.refresh_expectations_from_run import _expectation_example_key as refresh_expectation_example_key
from core.benchmark.bench_commands import _expectation_example_key as bench_command_expectation_example_key
from core.harness.run_benchmarks import (
    _format_failed_no_speedup,
    _resolve_local_contract_from_expectation,
    _should_fail_no_speedup,
    build_expectation_metadata,
    expectation_example_key,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _provenance() -> RunProvenance:
    return RunProvenance(
        git_commit="deadbee",
        hardware_key="b200",
        profile_name="minimal",
        timestamp="2026-03-28T00:00:00+00:00",
        iterations=10,
        warmup_iterations=5,
        execution_environment="virtualized",
        validity_profile="portable",
    )


def test_should_fail_no_speedup_respects_local_minimum_speedup() -> None:
    assert _should_fail_no_speedup(
        {
            "optimization_goal": "speed",
            "best_speedup": 1.01,
            "minimum_required_speedup": 1.02,
        }
    ) is True
    assert _should_fail_no_speedup(
        {
            "optimization_goal": "speed",
            "best_speedup": 1.03,
            "minimum_required_speedup": 1.02,
        }
    ) is False


def test_should_fail_no_speedup_defaults_to_generic_gate_without_local_contract() -> None:
    assert _should_fail_no_speedup(
        {
            "optimization_goal": "speed",
            "best_speedup": 1.04,
        }
    ) is True
    assert _should_fail_no_speedup(
        {
            "optimization_goal": "speed",
            "best_speedup": 1.05,
        }
    ) is False


def test_format_failed_no_speedup_uses_local_minimum_speedup() -> None:
    rendered = _format_failed_no_speedup(
        {
            "optimization_goal": "speed",
            "best_speedup": 1.01,
            "minimum_required_speedup": 1.02,
        }
    )
    assert "1.01x" in rendered
    assert "1.02x threshold" in rendered


def test_build_expectation_metadata_includes_minimum_required_speedup() -> None:
    metadata = build_expectation_metadata(
        {
            "example": "cache_aware_disagg",
            "type": "python",
            "optimization_goal": "speed",
            "baseline_time_ms": 10.0,
            "minimum_required_speedup": 1.02,
        },
        {"file": "optimized_cache_aware_disagg.py", "technique": "optimized", "time_ms": 9.2},
        git_commit="deadbee",
    )
    assert metadata["minimum_required_speedup"] == 1.02


def test_expectation_entry_round_trip_preserves_minimum_required_speedup() -> None:
    entry = ExpectationEntry(
        example="cache_aware_disagg",
        type="python",
        optimization_goal="speed",
        baseline_time_ms=10.0,
        best_optimized_time_ms=9.2,
        provenance=_provenance(),
        best_optimization_name="optimized",
        best_optimization_file="optimized_cache_aware_disagg.py",
        best_optimization_technique="optimized",
        minimum_required_speedup=1.02,
    )

    serialized = entry.to_dict()
    assert serialized["metadata"]["minimum_required_speedup"] == 1.02

    restored = ExpectationEntry.from_dict(serialized)
    assert restored.minimum_required_speedup == 1.02


def test_resolve_local_contract_from_expectation_requires_explicit_speed_floor() -> None:
    entry = ExpectationEntry(
        example="launch_bounds",
        type="python",
        optimization_goal="speed",
        baseline_time_ms=10.072,
        best_optimized_time_ms=10.0,
        provenance=_provenance(),
        best_optimization_name="optimized_launch_bounds",
    )

    goal, minimum_required_speedup = _resolve_local_contract_from_expectation(entry)

    assert goal == "speed"
    assert minimum_required_speedup is None


def test_resolve_local_contract_from_expectation_keeps_non_speed_goal_without_threshold() -> None:
    entry = ExpectationEntry(
        example="padding_aware_transformer",
        type="python",
        optimization_goal="memory",
        baseline_time_ms=10.0,
        best_optimized_time_ms=12.0,
        baseline_memory_mb=1000.0,
        best_optimized_memory_mb=250.0,
        provenance=_provenance(),
        best_optimization_name="optimized_padding_aware_transformer",
    )

    goal, minimum_required_speedup = _resolve_local_contract_from_expectation(entry)

    assert goal == "memory"
    assert minimum_required_speedup is None


def test_block_scaling_expectation_uses_explicit_local_floor_below_historical_best() -> None:
    payload = json.loads((REPO_ROOT / "labs" / "block_scaling" / "expectations_b200.json").read_text(encoding="utf-8"))
    entry = ExpectationEntry.from_dict(payload["examples"]["block_scaling"])

    assert entry.minimum_required_speedup == 1.75
    assert entry.best_speedup > entry.minimum_required_speedup


def test_small_effect_b200_examples_keep_explicit_local_speed_floors() -> None:
    ch06_payload = json.loads((REPO_ROOT / "ch06" / "expectations_b200.json").read_text(encoding="utf-8"))
    launch_bounds = ExpectationEntry.from_dict(ch06_payload["examples"]["launch_bounds"])
    launch_bounds_cuda = ExpectationEntry.from_dict(ch06_payload["examples"]["launch_bounds_cuda"])

    assert launch_bounds.minimum_required_speedup == 1.005
    assert launch_bounds.best_speedup > launch_bounds.minimum_required_speedup
    assert launch_bounds_cuda.minimum_required_speedup == 1.005
    assert launch_bounds_cuda.best_speedup > launch_bounds_cuda.minimum_required_speedup


def test_model_compile_reduced_precision_b200_uses_explicit_local_floor() -> None:
    payload = json.loads((REPO_ROOT / "ch14" / "expectations_b200.json").read_text(encoding="utf-8"))
    entry = ExpectationEntry.from_dict(payload["examples"]["model_compile_reduced_precision"])

    assert entry.minimum_required_speedup == 1.01
    assert entry.best_speedup > entry.minimum_required_speedup


def test_expectation_example_key_keeps_cuda_examples_with_cuda_suffix_stable() -> None:
    for helper in (
        expectation_example_key,
        bench_command_expectation_example_key,
        refresh_expectation_example_key,
    ):
        assert helper("launch_bounds_cuda", "cuda") == "launch_bounds_cuda"
        assert helper("double_buffered_pipeline", "cuda") == "double_buffered_pipeline_cuda"


def test_persistent_decode_nvlink_offload_is_control_contract_while_paged_offload_stays_speed() -> None:
    payload = json.loads((REPO_ROOT / "labs" / "persistent_decode" / "expectations_b200.json").read_text(encoding="utf-8"))
    nvlink_entry = ExpectationEntry.from_dict(payload["examples"]["nvlink_offload"])
    paged_entry = ExpectationEntry.from_dict(payload["examples"]["paged_kv_offload"])

    assert nvlink_entry.optimization_goal == "control"
    assert nvlink_entry.minimum_required_speedup is None
    assert paged_entry.optimization_goal == "speed"
    assert paged_entry.minimum_required_speedup is None
