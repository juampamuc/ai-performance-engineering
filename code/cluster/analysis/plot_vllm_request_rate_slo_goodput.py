#!/usr/bin/env python3
"""Plot SLO-aware goodput curves from request-rate analysis JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from plot_style import apply_plot_style


def main() -> int:
    apply_plot_style()
    p = argparse.ArgumentParser(description="Plot vLLM request-rate SLO goodput.")
    p.add_argument("--input-json", required=True, help="JSON from analyze_vllm_request_rate_slo_goodput.py")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-id", default="")
    args = p.parse_args()

    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    points = payload.get("points") or []
    if not points:
        raise SystemExit(f"No points in {args.input_json}")

    if args.run_id:
        run_id = args.run_id
    else:
        run_id = str(payload.get("run_id", "run")).strip() or "run"
        label = str(payload.get("label", "")).strip()
        if label:
            run_id = f"{run_id}_{label}"

    rates = [float(p.get("request_rate", 0.0) or 0.0) for p in points]
    total_tok = [float(p.get("total_token_throughput", 0.0) or 0.0) for p in points]
    goodput_tok = [float(p.get("goodput_tok_s", 0.0) or 0.0) for p in points]
    p99_ttft = [float(p.get("p99_ttft_ms", 0.0) or 0.0) for p in points]
    p99_tpot = [float(p.get("p99_tpot_ms", 0.0) or 0.0) for p in points]
    thresholds = payload.get("slo_thresholds_ms") or {}
    ttft_slo = float(thresholds.get("p99_ttft_ms", 0.0) or 0.0)
    tpot_slo = float(thresholds.get("p99_tpot_ms", 0.0) or 0.0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rates, total_tok, marker="o", label="total tok/s")
    ax.plot(rates, goodput_tok, marker="o", label="goodput tok/s")
    ax.set_xlabel("Request rate (req/s)")
    ax.set_ylabel("Tokens/sec")
    ax.set_title("vLLM request-rate SLO-aware goodput")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_id}_vllm_request_rate_goodput_tok_s.png", dpi=200)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.plot(rates, p99_ttft, marker="o", color="#1f77b4", label="p99 TTFT")
    ax2.plot(rates, p99_tpot, marker="o", color="#d62728", label="p99 TPOT")
    if ttft_slo > 0:
        ax1.axhline(ttft_slo, color="#1f77b4", linestyle="--", alpha=0.5)
    if tpot_slo > 0:
        ax2.axhline(tpot_slo, color="#d62728", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Request rate (req/s)")
    ax1.set_ylabel("p99 TTFT (ms)", color="#1f77b4")
    ax2.set_ylabel("p99 TPOT (ms)", color="#d62728")
    ax1.set_title("vLLM request-rate tail latency vs SLO")
    ax1.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_id}_vllm_request_rate_tail_slo_ms.png", dpi=200)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
