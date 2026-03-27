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


def test_localhost_fabric_eval_defaults_to_canary_vllm_profile() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "cluster" / "scripts" / "run_cluster_eval_suite.sh"
    run_id = f"pytest_local_fabric_canary_{uuid.uuid4().hex[:8]}"
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
            "--run-fabric-eval",
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
    profile = readiness["inputs"]["vllm_profile"]
    assert profile["class"] == "localhost_canary"
    assert profile["model"] == "openai-community/gpt2"
    assert profile["tp"] == 1
    assert profile["isl"] == 64
    assert profile["osl"] == 32
    assert profile["concurrency_range"] == ["1"]
    assert profile["repeats"] == 1
    assert profile["request_rate_enabled"] is False
    assert "localhost fabric evaluation" in profile["selection_reason"]


def test_localhost_fabric_eval_honors_explicit_vllm_override() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "cluster" / "scripts" / "run_cluster_eval_suite.sh"
    run_id = f"pytest_local_fabric_override_{uuid.uuid4().hex[:8]}"
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
            "--run-fabric-eval",
            "--model",
            "openai/gpt-oss-20b",
            "--isl",
            "512",
            "--osl",
            "128",
            "--concurrency-range",
            "8 16 32",
            "--run-vllm-request-rate-sweep",
            "--vllm-request-rate-range",
            "1 2 4",
            "--vllm-request-rate-max-concurrency",
            "32",
            "--vllm-request-rate-num-prompts",
            "128",
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
    profile = readiness["inputs"]["vllm_profile"]
    assert profile["class"] == "explicit_override"
    assert profile["model"] == "openai/gpt-oss-20b"
    assert profile["isl"] == 512
    assert profile["osl"] == 128
    assert profile["concurrency_range"] == ["8", "16", "32"]
    assert profile["request_rate_enabled"] is True
    assert profile["request_rate_range"] == ["1", "2", "4"]
    assert profile["request_rate_max_concurrency"] == 32
    assert profile["request_rate_num_prompts"] == 128
