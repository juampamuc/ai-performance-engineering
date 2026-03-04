#!/usr/bin/env python3
"""Plot SLO-aware vLLM goodput curves from analyze_vllm_slo_goodput output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from plot_style import apply_plot_style


def _plot_goodput(points: list[dict], output: Path, title: str) -> None:
    conc = [int(point.get("concurrency", 0)) for point in points]
    total_tok = [float(point.get("total_token_throughput", 0.0)) for point in points]
    goodput_tok = [float(point.get("goodput_tok_s", 0.0)) for point in points]
    total_req = [float(point.get("request_throughput", 0.0)) for point in points]
    goodput_req = [float(point.get("goodput_req_s", 0.0)) for point in points]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(conc, total_tok, marker="o", label="total tok/s")
    axes[0].plot(conc, goodput_tok, marker="o", label="goodput tok/s")
    axes[0].set_title("Token Throughput")
    axes[0].set_xlabel("Concurrency")
    axes[0].set_ylabel("Tokens/sec")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(fontsize=9)

    axes[1].plot(conc, total_req, marker="o", label="total req/s")
    axes[1].plot(conc, goodput_req, marker="o", label="goodput req/s")
    axes[1].set_title("Request Throughput")
    axes[1].set_xlabel("Concurrency")
    axes[1].set_ylabel("Requests/sec")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(fontsize=9)

    fig.suptitle(title)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _plot_tail_latency(points: list[dict], output: Path, title: str, slo_ttft_ms: float, slo_tpot_ms: float) -> None:
    conc = [int(point.get("concurrency", 0)) for point in points]
    p99_ttft = [float(point.get("p99_ttft_ms", 0.0)) for point in points]
    p99_tpot = [float(point.get("p99_tpot_ms", 0.0)) for point in points]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    line_ttft = ax1.plot(conc, p99_ttft, marker="o", color="#1f77b4", label="p99 TTFT (ms)")
    line_tpot = ax2.plot(conc, p99_tpot, marker="o", color="#d62728", label="p99 TPOT (ms)")

    if slo_ttft_ms > 0:
        ax1.axhline(slo_ttft_ms, color="#1f77b4", linestyle="--", alpha=0.5, label=f"TTFT SLO ({slo_ttft_ms:g} ms)")
    if slo_tpot_ms > 0:
        ax2.axhline(slo_tpot_ms, color="#d62728", linestyle="--", alpha=0.5, label=f"TPOT SLO ({slo_tpot_ms:g} ms)")

    ax1.set_xlabel("Concurrency")
    ax1.set_ylabel("p99 TTFT (ms)", color="#1f77b4")
    ax2.set_ylabel("p99 TPOT (ms)", color="#d62728")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.set_title(title)

    lines = line_ttft + line_tpot
    labels = [line.get_label() for line in lines]
    if slo_ttft_ms > 0:
        labels.append(f"TTFT SLO ({slo_ttft_ms:g} ms)")
    if slo_tpot_ms > 0:
        labels.append(f"TPOT SLO ({slo_tpot_ms:g} ms)")
    handles = lines
    if slo_ttft_ms > 0:
        handles.append(ax1.lines[-1])
    if slo_tpot_ms > 0:
        handles.append(ax2.lines[-1])
    ax1.legend(handles, labels, loc="best", fontsize=9)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> int:
    apply_plot_style()
    p = argparse.ArgumentParser(description="Plot vLLM SLO goodput curves.")
    p.add_argument("--input-json", required=True, help="JSON from analyze_vllm_slo_goodput.py")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--run-id", default="", help="Output filename prefix (default: payload run_id + label)")
    args = p.parse_args()

    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    points = payload.get("points") or []
    if not points:
        raise SystemExit(f"No points found in {args.input_json}")

    if args.run_id:
        run_id = args.run_id
    else:
        run_id = str(payload.get("run_id", "run")).strip() or "run"
        label = str(payload.get("label", "")).strip()
        if label:
            run_id = f"{run_id}_{label}"

    slo = payload.get("slo_thresholds_ms") or {}
    slo_ttft_ms = float(slo.get("p99_ttft_ms", 0.0) or 0.0)
    slo_tpot_ms = float(slo.get("p99_tpot_ms", 0.0) or 0.0)

    out_dir = Path(args.out_dir)
    _plot_goodput(
        points,
        out_dir / f"{run_id}_vllm_serve_goodput_vs_concurrency.png",
        "vLLM SLO-aware goodput vs concurrency",
    )
    _plot_tail_latency(
        points,
        out_dir / f"{run_id}_vllm_serve_tail_slo_vs_concurrency.png",
        "vLLM p99 tail latency vs concurrency",
        slo_ttft_ms,
        slo_tpot_ms,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
