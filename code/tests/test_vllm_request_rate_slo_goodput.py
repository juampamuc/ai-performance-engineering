from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def test_analyze_vllm_request_rate_slo_goodput(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    tmp_dir = tmp_path / "vllm_request_rate_slo"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    input_csv = tmp_dir / "vllm_serve_request_rate_sweep.csv"
    with input_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "tp",
                "isl",
                "osl",
                "request_rate",
                "max_concurrency",
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
                "completed",
                "failed",
            ]
        )
        writer.writerow(["model", 1, 128, 64, 1, 32, 200, 1, 90, 120, 220, 210, 400, 25, 24, 50, 200, 0])
        writer.writerow(["model", 1, 128, 64, 2, 32, 200, 2, 140, 190, 260, 250, 700, 30, 29, 70, 200, 0])
        writer.writerow(["model", 1, 128, 64, 4, 32, 200, 3.5, 200, 260, 600, 550, 2500, 50, 49, 180, 200, 0])

    output_json = tmp_dir / "request_rate_slo_goodput.json"
    output_csv = tmp_dir / "request_rate_slo_goodput.csv"
    cmd = [
        sys.executable,
        "cluster/analysis/analyze_vllm_request_rate_slo_goodput.py",
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
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    summary = payload["summary"]
    assert summary["request_rate_at_max_goodput_tok_s"] == 2.0
    assert summary["knee_request_rate"] == 4.0
    assert summary["max_goodput_tok_s"] == 190.0
    assert summary["peak_total_tok_s"] == 260.0
