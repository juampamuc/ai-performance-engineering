from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ABSOLUTE_PREFIX = "/home/cfregly/ai-performance-engineering/code/"
LOCAL_MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\((?!https?:|#|mailto:)([^)]+)\)")
README_EXCLUDE_PARTS = {
    "third_party",
    "node_modules",
    ".git",
    ".pytest_cache",
    ".venv_pdf",
    ".venv_pdfqa",
}


def _maintained_readmes() -> list[Path]:
    return [
        path
        for path in sorted(REPO_ROOT.rglob("README*.md"))
        if not any(part in README_EXCLUDE_PARTS for part in path.parts)
    ]


def test_maintained_readmes_have_no_workspace_absolute_links() -> None:
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in _maintained_readmes()
        if WORKSPACE_ABSOLUTE_PREFIX in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"README files contain workspace-absolute links: {offenders}"


def test_maintained_readme_local_markdown_links_resolve() -> None:
    missing: list[str] = []
    for path in _maintained_readmes():
        text = path.read_text(encoding="utf-8")
        for target in LOCAL_MARKDOWN_LINK.findall(text):
            normalized = target[1:-1] if target.startswith("<") and target.endswith(">") else target
            if normalized.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path_part = normalized.split("#", 1)[0].split("?", 1)[0].strip()
            if not path_part:
                continue
            if not (path.parent / path_part).resolve().exists():
                missing.append(f"{path.relative_to(REPO_ROOT)} -> {target}")
    assert not missing, f"README files contain missing local links: {missing}"
