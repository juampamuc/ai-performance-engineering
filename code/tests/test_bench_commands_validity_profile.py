from __future__ import annotations

from pathlib import Path

import pytest

from core.benchmark.bench_commands import _read_progress_payload_for_heartbeat, _validate_validity_profile


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
