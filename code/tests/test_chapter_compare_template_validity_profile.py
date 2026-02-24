from __future__ import annotations

import pytest

from core.utils.chapter_compare_template import _extract_cli_validity_profile


def test_extract_cli_validity_profile_absent() -> None:
    assert _extract_cli_validity_profile([]) is None


def test_extract_cli_validity_profile_separate_token() -> None:
    assert _extract_cli_validity_profile(["--validity-profile", "portable"]) == "portable"


def test_extract_cli_validity_profile_equals_token() -> None:
    assert _extract_cli_validity_profile(["--validity-profile=strict"]) == "strict"


def test_extract_cli_validity_profile_invalid_value() -> None:
    with pytest.raises(ValueError, match="Invalid --validity-profile value"):
        _extract_cli_validity_profile(["--validity-profile", "bad"])
