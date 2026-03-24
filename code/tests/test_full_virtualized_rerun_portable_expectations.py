from __future__ import annotations

from pathlib import Path

from core.benchmark.expectations import ExpectationEntry, RunProvenance
from scripts.full_virtualized_rerun import (
    _canonicalize_state,
    _initialize_state,
    _write_eligible_expectation,
)


def _sample_entry() -> ExpectationEntry:
    return ExpectationEntry(
        example="performance",
        type="python",
        optimization_goal="speed",
        baseline_time_ms=2.0,
        best_optimized_time_ms=1.0,
        provenance=RunProvenance(
            git_commit="deadbeef",
            hardware_key="b200",
            profile_name="none",
            timestamp="2026-03-24T00:00:00+00:00",
            iterations=5,
            warmup_iterations=5,
            execution_environment="virtualized",
            validity_profile="portable",
        ),
    )


def test_portable_rerun_state_tracks_expectation_write_opt_in(tmp_path: Path) -> None:
    state = _initialize_state(
        run_root=tmp_path / "runs",
        queue_root=tmp_path / "queue",
        profile="none",
        suite_timeout=0,
        gpu_sm_clock_mhz=1965,
        gpu_mem_clock_mhz=3996,
        allow_portable_expectations_update=True,
    )

    assert state["allow_portable_expectations_update"] is True
    assert _canonicalize_state({})["allow_portable_expectations_update"] is False


def test_portable_rerun_does_not_write_expectations_without_explicit_opt_in(tmp_path: Path) -> None:
    repo_root = tmp_path
    (repo_root / "ch01").mkdir(parents=True)

    action, reasons, details = _write_eligible_expectation(
        repo_root=repo_root,
        target="ch01:performance",
        bench={"example": "performance", "type": "python"},
        entry=_sample_entry(),
        hardware_key="b200",
        allow_portable_expectations_update=False,
    )

    assert action == "queued"
    assert reasons == ["portable_expectation_write_not_allowed"]
    assert details["allow_portable_expectations_update"] is False
    assert not (repo_root / "ch01" / "expectations_b200.json").exists()
