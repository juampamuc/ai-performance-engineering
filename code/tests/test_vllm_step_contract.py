from __future__ import annotations

import json
from pathlib import Path

import pytest

from cluster.analysis.vllm_step_contract import (
    summarize_upstream_failure,
    validate_vllm_request_rate_csv,
    validate_vllm_serve_csv,
    validate_vllm_slo_goodput_summary,
    validate_vllm_stability_summary,
)


def test_summarize_upstream_failure_prefers_startup_status(tmp_path: Path) -> None:
    suite_steps = tmp_path / "suite_steps.json"
    suite_steps.write_text(
        json.dumps(
            [
                {
                    "name": "vllm_serve_sweep",
                    "exit_code": 1,
                    "log_path": "/tmp/vllm_serve_sweep.log",
                }
            ]
        ),
        encoding="utf-8",
    )
    startup = tmp_path / "serve_startup.json"
    startup.write_text(
        json.dumps(
            {
                "status": "startup_timeout",
                "ready": False,
                "elapsed_seconds": 1800,
                "server_log_path": "/tmp/server.log",
                "detail": "server failed to start before ready timeout",
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_upstream_failure(suite_steps, "vllm_serve_sweep", startup)
    assert summary["has_failure"] is True
    assert summary["startup_status"] == "startup_timeout"
    assert "startup_timeout" in summary["message"]
    assert "/tmp/server.log" in summary["message"]


def test_validate_vllm_serve_csv_rejects_zero_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "vllm_serve_sweep.csv"
    csv_path.write_text(
        "model,tp,isl,osl,concurrency,num_prompts,request_throughput,output_throughput,total_token_throughput,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_tpot_ms,median_tpot_ms,p99_tpot_ms,gpu_util_mean_pct,gpu_util_p95_pct,mem_used_mean_mb,mem_used_max_mb,gpu_power_mean_w,gpu_power_p95_w,completed,failed\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="has no rows"):
        validate_vllm_serve_csv(csv_path)


def test_validate_vllm_step_outputs_accept_successful_artifacts(tmp_path: Path) -> None:
    serve_csv = tmp_path / "vllm_serve_sweep.csv"
    serve_csv.write_text(
        "model,tp,isl,osl,concurrency,num_prompts,request_throughput,output_throughput,total_token_throughput,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_tpot_ms,median_tpot_ms,p99_tpot_ms,gpu_util_mean_pct,gpu_util_p95_pct,mem_used_mean_mb,mem_used_max_mb,gpu_power_mean_w,gpu_power_p95_w,completed,failed\n"
        "openai-community/gpt2,1,64,32,1,10,12.0,34.0,46.0,10.0,9.0,12.0,1.0,1.0,1.5,50.0,60.0,1024.0,1024.0,100.0,110.0,10,0\n",
        encoding="utf-8",
    )
    request_rate_csv = tmp_path / "vllm_serve_request_rate_sweep.csv"
    request_rate_csv.write_text(
        "model,tp,isl,osl,request_rate,max_concurrency,num_prompts,request_throughput,output_throughput,total_token_throughput,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_tpot_ms,median_tpot_ms,p99_tpot_ms,gpu_util_mean_pct,gpu_util_p95_pct,mem_used_mean_mb,mem_used_max_mb,gpu_power_mean_w,gpu_power_p95_w,completed,failed\n"
        "openai-community/gpt2,1,64,32,1.000000,4,80,12.0,34.0,46.0,10.0,9.0,12.0,1.0,1.0,1.5,50.0,60.0,1024.0,1024.0,100.0,110.0,10,0\n",
        encoding="utf-8",
    )
    stability = tmp_path / "serve_stability.json"
    stability.write_text(
        json.dumps({"summary": {"points": 1, "total_token_throughput_cv_pct_p95": 3.0}}),
        encoding="utf-8",
    )
    slo = tmp_path / "serve_slo.json"
    slo.write_text(
        json.dumps({"status": "ok", "summary": {"concurrency_points": 1, "peak_total_tok_s": 46.0, "max_goodput_tok_s": 40.0}}),
        encoding="utf-8",
    )
    rate_slo = tmp_path / "rate_slo.json"
    rate_slo.write_text(
        json.dumps({"status": "ok", "summary": {"request_rate_points": 1}}),
        encoding="utf-8",
    )

    validate_vllm_serve_csv(serve_csv)
    validate_vllm_request_rate_csv(request_rate_csv)
    validate_vllm_stability_summary(stability, threshold=10.0, label="vLLM sweep stability")
    validate_vllm_slo_goodput_summary(slo)
    validate_vllm_slo_goodput_summary(rate_slo, request_rate=True)
