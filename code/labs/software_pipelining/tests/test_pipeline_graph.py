from __future__ import annotations

from dataclasses import replace

from labs.software_pipelining.baseline_tile_pipeline import get_benchmark as get_baseline_benchmark
from labs.software_pipelining.optimized_tile_pipeline import get_benchmark as get_optimized_benchmark
from labs.software_pipelining.pipeline_graph import (
    PipelineExample,
    ScheduleSlot,
    get_pipeline_example,
    validate_schedule,
)
from labs.software_pipelining.schedule_visualizer import render_ascii, render_json_payload


def _replace_slot(
    example: PipelineExample,
    *,
    node: str,
    iteration: int,
    time: int,
) -> PipelineExample:
    updated_schedule = []
    for slot in example.schedule:
        if slot.node == node and slot.iteration == iteration:
            updated_schedule.append(replace(slot, time=time))
        else:
            updated_schedule.append(slot)
    return replace(example, schedule=tuple(updated_schedule))


def test_valid_examples_pass_legality_checks() -> None:
    for name in ("gemm_mainloop", "fa_like_inner_loop"):
        result = validate_schedule(get_pipeline_example(name))
        assert result.is_valid, result.errors


def test_gemm_example_rejects_stage_reuse_too_early() -> None:
    example = get_pipeline_example("gemm_mainloop")
    invalid = _replace_slot(example, node="tma_load", iteration=2, time=1)
    result = validate_schedule(invalid)
    assert not result.is_valid
    assert any("anti_dependency violation" in error for error in result.errors)


def test_fa_like_example_rejects_loop_carried_fixup_violation() -> None:
    example = get_pipeline_example("fa_like_inner_loop")
    invalid = _replace_slot(example, node="softmax_fixup", iteration=1, time=2)
    result = validate_schedule(invalid)
    assert not result.is_valid
    assert any("loop_carried violation" in error for error in result.errors)


def test_json_payload_has_stable_schema() -> None:
    payload = render_json_payload(get_pipeline_example("fa_like_inner_loop"))
    assert set(payload.keys()) == {
        "description",
        "edges",
        "example",
        "iteration_count",
        "nodes",
        "schedule",
        "stage_count",
        "validation",
    }
    assert payload["example"] == "fa_like_inner_loop"
    assert payload["validation"]["is_valid"] is True
    assert payload["schedule"]


def test_ascii_output_mentions_pipeline_phases() -> None:
    text = render_ascii(get_pipeline_example("gemm_mainloop"))
    lowered = text.lower()
    assert "prologue" in lowered
    assert "steady" in lowered
    assert "epilogue" in lowered
    assert "validation: pass" in lowered


def test_benchmark_modules_share_workload_contract() -> None:
    baseline = get_baseline_benchmark()
    optimized = get_optimized_benchmark()

    assert baseline.workload == optimized.workload
    assert baseline.pipeline_stage_count == 1
    assert optimized.pipeline_stage_count == 2
    assert baseline.get_config().iterations == optimized.get_config().iterations
