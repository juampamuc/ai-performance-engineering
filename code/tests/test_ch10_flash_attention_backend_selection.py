from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_flash_attention_is_tried_before_other_sdpa_backends() -> None:
    source = (REPO_ROOT / "ch10/optimized_flash_attention.py").read_text()
    flash_idx = source.index("SDPBackend.FLASH_ATTENTION")
    efficient_idx = source.index("SDPBackend.EFFICIENT_ATTENTION")
    assert flash_idx < efficient_idx
    assert "major >= 10" not in source
    assert "[SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]" not in source


def test_external_flashattention_engines_are_probed_in_preference_order() -> None:
    source = (REPO_ROOT / "ch10/optimized_flash_attention.py").read_text()

    flash3_idx = source.index("flash_attn_3.flash_attn_interface")
    flash2_idx = source.index("flash_attn.flash_attn_interface")

    assert flash3_idx < flash2_idx
