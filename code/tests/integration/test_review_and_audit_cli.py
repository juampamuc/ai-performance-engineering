from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_pair_review_cli_writes_real_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "pair_review"
    cmd = [
        sys.executable,
        "-m",
        "core.verification.review_baseline_optimized_pairs",
        "--chapter",
        "ch12",
        "--json",
        "--markdown",
        "--output-dir",
        str(output_dir),
    ]

    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Pairs reviewed:" in proc.stdout

    json_path = output_dir / "review_findings.json"
    md_path = output_dir / "review_findings.md"
    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["chapters"] == ["ch12"]
    assert payload["summary"]["total_pairs"] > 0


@pytest.mark.slow
def test_benchmark_pair_audit_cli_writes_real_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "audit"
    cmd = [
        sys.executable,
        "-m",
        "core.scripts.run_benchmark_pair_audit",
        "--scope",
        "ch11",
        "--output-dir",
        str(output_dir),
    ]

    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Audit manifest:" in proc.stdout
    assert "Audit summary:" in proc.stdout

    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"
    assert manifest_path.exists()
    assert summary_path.exists()
    assert markdown_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert manifest["pair_count"] > 0
    assert summary["scopes"] == ["ch11"]
    assert "review" in summary["steps"]
    assert "scope_contract" in summary["steps"]
