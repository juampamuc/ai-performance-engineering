from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def test_analyze_vllm_slo_goodput_detects_knee_and_goodput(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    tmp_dir = tmp_path / "vllm_slo_goodput"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    input_csv = tmp_dir / "vllm_serve_sweep.csv"
    with input_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "tp",
                "isl",
                "osl",
                "concurrency",
                "num_prompts",
                "request_throughput",
                "output_throughput",
                "total_token_throughput",
                "mean_ttft_ms",
                "median_ttft_ms",
                "p99_ttft_ms",
                "mean_tpot_ms",
                "median_tpot_ms",
                "p99_tpot_ms",
                "gpu_util_mean_pct",
                "gpu_util_p95_pct",
                "mem_used_mean_mb",
                "mem_used_max_mb",
                "completed",
                "failed",
            ]
        )
        writer.writerow(["model", 1, 128, 64, 1, 10, 10, 70, 100, 200, 180, 400, 20, 19, 30, 50, 70, 1200, 1300, 10, 0])
        writer.writerow(["model", 1, 128, 64, 2, 20, 18, 140, 180, 250, 230, 900, 28, 27, 50, 60, 80, 1300, 1400, 20, 0])
        writer.writerow(["model", 1, 128, 64, 4, 40, 22, 180, 220, 500, 450, 2500, 35, 34, 80, 75, 90, 1450, 1600, 40, 0])

    output_json = tmp_dir / "vllm_slo_goodput.json"
    output_csv = tmp_dir / "vllm_slo_goodput.csv"
    cmd = [
        sys.executable,
        "cluster/analysis/analyze_vllm_slo_goodput.py",
        "--input",
        str(input_csv),
        "--run-id",
        "2026-03-04",
        "--label",
        "node1",
        "--slo-p99-ttft-ms",
        "2000",
        "--slo-p99-tpot-ms",
        "200",
        "--output-json",
        str(output_json),
        "--output-csv",
        str(output_csv),
    ]
    proc = subprocess.run(cmd, cwd=repo_root, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert output_json.exists()
    assert output_csv.exists()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    summary = payload["summary"]
    assert summary["concurrency_at_max_goodput_tok_s"] == 2
    assert summary["knee_concurrency"] == 4
    assert summary["max_goodput_tok_s"] == 180.0
    assert summary["peak_total_tok_s"] == 220.0
