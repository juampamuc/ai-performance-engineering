from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parent.parent
SKIP_PARTS = {"third_party", "cluster/env", "tests"}
DISALLOWED_PATTERNS = {
    "os.environ.get(\"LOCAL_RANK\", ...)": re.compile(r'os\.environ\.get\(["\']LOCAL_RANK["\']\s*,'),
    "os.getenv(\"LOCAL_RANK\", ...)": re.compile(r'os\.getenv\(["\']LOCAL_RANK["\']\s*,'),
    "_env_int(\"LOCAL_RANK\", 0)": re.compile(r'_env_int\(["\']LOCAL_RANK["\']\s*,\s*0\s*\)'),
    "_get_env_int(\"OMPI_COMM_WORLD_LOCAL_RANK\", 0)": re.compile(
        r'_get_env_int\(["\']OMPI_COMM_WORLD_LOCAL_RANK["\']\s*,\s*0\s*\)'
    ),
    "manual LOCAL_RANK else-0 fallback": re.compile(
        r'if\s+["\']LOCAL_RANK["\']\s+in\s+os\.environ:\s*\n'
        r'\s*local_rank\s*=\s*int\(os\.environ\[["\']LOCAL_RANK["\']\]\)\s*\n'
        r'\s*else:[^\n]*\n'
        r'\s*local_rank\s*=\s*0',
        re.MULTILINE,
    ),
}


def _iter_repo_sources() -> list[Path]:
    paths: list[Path] = []
    for pattern in ("**/*.py", "**/*.sh"):
        for path in REPO_ROOT.glob(pattern):
            relative = path.relative_to(REPO_ROOT).as_posix()
            if any(part in relative for part in SKIP_PARTS):
                continue
            paths.append(path)
    return sorted(set(paths))


def test_repo_avoids_silent_local_rank_fallbacks() -> None:
    offenders: list[str] = []
    for path in _iter_repo_sources():
        text = path.read_text(encoding="utf-8")
        hits = [label for label, pattern in DISALLOWED_PATTERNS.items() if pattern.search(text)]
        if hits:
            offenders.append(f"{path.relative_to(REPO_ROOT)}: {', '.join(hits)}")
    assert offenders == []
