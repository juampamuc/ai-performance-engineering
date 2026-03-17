from __future__ import annotations

import subprocess
import uuid
from pathlib import Path
import json


def test_modern_profile_honors_explicit_skip_vllm_request_rate() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "cluster" / "scripts" / "run_cluster_eval_suite.sh"
    run_id = f"pytest_skip_vllm_rate_{uuid.uuid4().hex[:8]}"
    proc = subprocess.run(
        [
            "bash",
            str(script),
            "--run-id",
            run_id,
            "--hosts",
            "localhost",
            "--labels",
            "localhost",
            "--primary-label",
            "localhost",
            "--modern-llm-profile",
            "--skip-vllm-request-rate-sweep",
            "--multinode-readiness-check-only",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output

    readiness_path = repo_root / "cluster" / "runs" / run_id / "structured" / f"{run_id}_multinode_readiness.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    assert readiness["inputs"]["modern_llm_profile"] is True
    assert readiness["inputs"]["run_vllm_request_rate_sweep"] is False


def test_modern_profile_honors_explicit_no_strict_canonical_completeness() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "cluster" / "scripts" / "run_cluster_eval_suite.sh"
    run_id = f"pytest_no_strict_{uuid.uuid4().hex[:8]}"
    proc = subprocess.run(
        [
            "bash",
            str(script),
            "--run-id",
            run_id,
            "--hosts",
            "localhost",
            "--labels",
            "localhost",
            "--primary-label",
            "localhost",
            "--modern-llm-profile",
            "--no-strict-canonical-completeness",
            "--multinode-readiness-check-only",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output

    readiness_path = repo_root / "cluster" / "runs" / run_id / "structured" / f"{run_id}_multinode_readiness.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    assert readiness["inputs"]["modern_llm_profile"] is True
    assert readiness["inputs"]["strict_canonical_completeness"] is False


def test_fio_file_size_flag_is_reported_in_suite_summary() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "cluster" / "scripts" / "run_cluster_eval_suite.sh"
    run_id = f"pytest_fio_size_{uuid.uuid4().hex[:8]}"
    proc = subprocess.run(
        [
            "bash",
            str(script),
            "--run-id",
            run_id,
            "--hosts",
            "localhost",
            "--labels",
            "localhost",
            "--primary-label",
            "localhost",
            "--fio-file-size",
            "256M",
            "--multinode-readiness-check-only",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output

    readiness_path = repo_root / "cluster" / "runs" / run_id / "structured" / f"{run_id}_multinode_readiness.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    assert readiness["inputs"]["fio_file_size"] == "256M"
