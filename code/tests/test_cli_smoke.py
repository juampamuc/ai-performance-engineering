import subprocess
import sys
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(args, *, timeout=30, env=None):
    return subprocess.run(
        [sys.executable, "-m", "cli.aisp", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        cwd=REPO_ROOT,
        env=env,
    )


def test_help_exits_cleanly():
    result = _run_cli(["help"])
    assert result.returncode == 0
    assert "aisp" in result.stdout.lower()


def test_mcp_server_import():
    import mcp.mcp_server as mcp_server

    assert isinstance(mcp_server.TOOLS, dict)
    # ensure harness tools are registered
    assert "run_benchmarks" in mcp_server.TOOLS


def test_bench_list_targets_help():
    result = _run_cli(["bench", "list-targets", "--help"])
    assert result.returncode == 0
    assert "list-targets" in result.stdout


def test_bench_analyze_help():
    result = _run_cli(["bench", "analyze", "--help"])
    assert result.returncode == 0
    assert "analyze" in result.stdout.lower()


def test_bench_whatif_help():
    result = _run_cli(["bench", "whatif", "--help"])
    assert result.returncode == 0
    assert "whatif" in result.stdout.lower()


def test_distributed_plan_outputs_json_serializable_payload():
    result = _run_cli(
        ["--json", "distributed", "plan", "--model-size", "7", "--gpus", "8", "--nodes", "1"],
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["success"] is True
    assert "best" in payload


def test_distributed_topology_returns_structured_payload():
    result = _run_cli(["--json", "distributed", "topology"], timeout=60)
    assert result.returncode in (0, 1), result.stderr
    payload = json.loads(result.stdout)
    assert "available" in payload


def test_inference_vllm_outputs_json_serializable_payload():
    result = _run_cli(["inference", "vllm", "--model-size", "7", "--target", "throughput"], timeout=60)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["success"] is True
    assert isinstance(payload["vllm_config"], dict)
    assert payload["vllm_config"]["model"] == "model"
    assert payload["vllm_config"]["tensor_parallel_size"] >= 1
    assert payload["vllm_config"]["launch_command"]


def test_cluster_nmx_partition_lab_requires_explicit_nmx_url():
    env = os.environ.copy()
    env["AISP_FABRIC_NMX_URL"] = "https://ignored.example"
    result = _run_cli(["cluster", "nmx-partition-lab", "--alpha-name", "AlphaPartition"], env=env)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["success"] is False
    assert payload["error"] == "NMX URL is required. Pass --nmx-url to target the fabric management plane."


def test_cluster_common_eval_rejects_unknown_preset_via_real_cli():
    result = _run_cli(["cluster", "common-eval", "--preset", "not-a-preset", "--hosts", "localhost"])
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["success"] is False
    assert "Unknown preset" in payload["error"]


def test_cluster_promote_run_reports_missing_run_dir_via_real_cli():
    result = _run_cli(["cluster", "promote-run", "--run-id", "does-not-exist"])
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["success"] is False
    assert payload["run_id"] == "does-not-exist"
    assert "missing run dir" in payload["stderr"]
