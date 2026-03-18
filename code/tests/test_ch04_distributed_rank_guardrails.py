from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parent.parent
CH04_DIR = REPO_ROOT / "ch04"

LOCAL_RANK_FALLBACK = re.compile(r'os\.environ\.get\("LOCAL_RANK",')
GLOBAL_RANK_DEVICE = re.compile(
    r'(?:torch\.device\(f"cuda:\{rank\}"\)|device\s*=\s*f"cuda:\{rank\}")'
)


def test_ch04_avoids_ad_hoc_local_rank_fallbacks() -> None:
    offenders = []
    for path in sorted(CH04_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if LOCAL_RANK_FALLBACK.search(text):
            offenders.append(path.name)
    assert offenders == []


def test_ch04_distributed_scripts_do_not_bind_cuda_device_from_global_rank() -> None:
    offenders = []
    for path in sorted(CH04_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "dist.get_rank()" in text and GLOBAL_RANK_DEVICE.search(text):
            offenders.append(path.name)
    assert offenders == []
