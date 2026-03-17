import subprocess
import sys
import json
import os


def test_help_exits_cleanly():
    result = subprocess.run(
        [sys.executable, "-m", "cli.aisp", "help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "aisp" in result.stdout.lower()


def test_mcp_server_import():
    import mcp.mcp_server as mcp_server

    assert isinstance(mcp_server.TOOLS, dict)
    # ensure harness tools are registered
    assert "run_benchmarks" in mcp_server.TOOLS


def test_bench_list_targets_help():
    result = subprocess.run(
        [sys.executable, "-m", "cli.aisp", "bench", "list-targets", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "list-targets" in result.stdout


def test_bench_analyze_help():
    result = subprocess.run(
        [sys.executable, "-m", "cli.aisp", "bench", "analyze", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "analyze" in result.stdout.lower()


def test_bench_whatif_help():
    result = subprocess.run(
        [sys.executable, "-m", "cli.aisp", "bench", "whatif", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "whatif" in result.stdout.lower()


def test_distributed_plan_outputs_json_serializable_payload():
    result = subprocess.run(
        [sys.executable, "-m", "cli.aisp", "--json", "distributed", "plan", "--model-size", "7", "--gpus", "8", "--nodes", "1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["success"] is True
    assert "best" in payload


def test_distributed_topology_returns_structured_payload():
    result = subprocess.run(
        [sys.executable, "-m", "cli.aisp", "--json", "distributed", "topology"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
    )
    assert result.returncode in (0, 1), result.stderr
    payload = json.loads(result.stdout)
    assert "available" in payload


def test_inference_vllm_outputs_json_serializable_payload():
    result = subprocess.run(
        [sys.executable, "-m", "cli.aisp", "inference", "vllm", "--model-size", "7", "--target", "throughput"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
    )
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
    result = subprocess.run(
        [sys.executable, "-m", "cli.aisp", "cluster", "nmx-partition-lab", "--alpha-name", "AlphaPartition"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["success"] is False
    assert payload["error"] == "NMX URL is required. Pass --nmx-url to target the fabric management plane."
