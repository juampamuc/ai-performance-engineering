from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text()


def test_optimized_performance_disables_and_restores_tf32() -> None:
    source = _read("ch01/optimized_performance.py")
    assert "capture_tf32_state" in source
    assert "set_tf32_state(False)" in source
    assert "self._tf32_state" in source
    assert "restore_tf32_state(self._tf32_state)" in source


def test_optimized_performance_fp16_disables_and_restores_tf32() -> None:
    source = _read("ch01/optimized_performance_fp16.py")
    assert "capture_tf32_state" in source
    assert "set_tf32_state(False)" in source
    assert "self._tf32_state" in source
    assert "restore_tf32_state(self._tf32_state)" in source
