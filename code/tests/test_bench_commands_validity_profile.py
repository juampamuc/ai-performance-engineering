from __future__ import annotations

import pytest

from core.benchmark.bench_commands import _validate_validity_profile


def test_validate_validity_profile_default_is_strict() -> None:
    assert _validate_validity_profile(None) == "strict"


def test_validate_validity_profile_accepts_portable() -> None:
    assert _validate_validity_profile("portable") == "portable"


def test_validate_validity_profile_rejects_invalid_value() -> None:
    with pytest.raises(Exception, match="Invalid --validity-profile value 'invalid'"):
        _validate_validity_profile("invalid")
