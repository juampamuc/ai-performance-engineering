from __future__ import annotations

from pathlib import Path
import signal

import pytest

from core.benchmark.bench_commands import (
    _apply_suite_timeout,
    _read_progress_payload_for_heartbeat,
    _validate_validity_profile,
)


def test_validate_validity_profile_default_is_strict() -> None:
    assert _validate_validity_profile(None) == "strict"


def test_validate_validity_profile_accepts_portable() -> None:
    assert _validate_validity_profile("portable") == "portable"


def test_validate_validity_profile_rejects_invalid_value() -> None:
    with pytest.raises(Exception, match="Invalid --validity-profile value 'invalid'"):
        _validate_validity_profile("invalid")


def test_read_progress_payload_for_heartbeat_surfaces_invalid_json(tmp_path: Path) -> None:
    progress_path = tmp_path / "run_progress.json"
    progress_path.write_text("{not-json", encoding="utf-8")

    payload, warning = _read_progress_payload_for_heartbeat(progress_path)

    assert payload is None
    assert warning is not None
    assert "Failed to read progress payload" in warning


def test_read_progress_payload_for_heartbeat_rejects_non_object_json(tmp_path: Path) -> None:
    progress_path = tmp_path / "run_progress.json"
    progress_path.write_text("[]", encoding="utf-8")

    payload, warning = _read_progress_payload_for_heartbeat(progress_path)

    assert payload is None
    assert warning is not None
    assert "expected JSON object, got list" in warning


@pytest.mark.skipif(
    not hasattr(signal, "SIGALRM") or not hasattr(signal, "getitimer"),
    reason="SIGALRM interval timers unavailable on this platform",
)
def test_apply_suite_timeout_clears_inherited_alarm_when_disabled() -> None:
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_delay, previous_interval = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, lambda _signum, _frame: None)
    signal.setitimer(signal.ITIMER_REAL, 5.0, 0.0)

    restore = _apply_suite_timeout(0)
    try:
        cleared_delay, cleared_interval = signal.getitimer(signal.ITIMER_REAL)
        assert cleared_delay == pytest.approx(0.0, abs=0.05)
        assert cleared_interval == pytest.approx(0.0, abs=0.05)
    finally:
        restore()
        signal.setitimer(signal.ITIMER_REAL, 0.0, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_delay > 0.0 or previous_interval > 0.0:
            signal.setitimer(signal.ITIMER_REAL, previous_delay, previous_interval)


@pytest.mark.skipif(
    not hasattr(signal, "SIGALRM") or not hasattr(signal, "getitimer"),
    reason="SIGALRM interval timers unavailable on this platform",
)
def test_apply_suite_timeout_restores_previous_alarm_state() -> None:
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_delay, previous_interval = signal.getitimer(signal.ITIMER_REAL)

    def _handler(_signum, _frame) -> None:
        return None

    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, 5.0, 0.0)

    restore = _apply_suite_timeout(1)
    try:
        active_delay, active_interval = signal.getitimer(signal.ITIMER_REAL)
        assert 0.0 < active_delay <= 1.0
        assert active_interval == pytest.approx(0.0, abs=0.05)
    finally:
        restore()
        restored_delay, restored_interval = signal.getitimer(signal.ITIMER_REAL)
        restored_handler = signal.getsignal(signal.SIGALRM)
        signal.setitimer(signal.ITIMER_REAL, 0.0, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_delay > 0.0 or previous_interval > 0.0:
            signal.setitimer(signal.ITIMER_REAL, previous_delay, previous_interval)

    assert restored_handler is _handler
    assert 0.0 < restored_delay <= 5.0
    assert restored_interval == pytest.approx(0.0, abs=0.05)
