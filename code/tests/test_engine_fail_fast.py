from __future__ import annotations

import pytest

from core.engine import _safe_call


def _raise_value_error() -> dict:
    raise ValueError("bad input")


def _raise_runtime_error() -> dict:
    raise RuntimeError("unexpected bug")


def test_safe_call_returns_structured_error_for_expected_user_failure() -> None:
    result = _safe_call(_raise_value_error)

    assert result["success"] is False
    assert result["error"] == "bad input"
    assert result["error_type"] == "value_error"


def test_safe_call_reraises_unexpected_failures() -> None:
    with pytest.raises(RuntimeError, match="unexpected bug"):
        _safe_call(_raise_runtime_error)
