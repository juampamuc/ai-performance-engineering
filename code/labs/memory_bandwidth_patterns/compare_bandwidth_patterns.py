"""Standalone runner for the memory-bandwidth-patterns lab."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.harness.benchmark_harness import lock_gpu_clocks
from labs.memory_bandwidth_patterns.bandwidth_patterns_common import (
    BandwidthLabConfig,
    DECK_TITLE,
    apply_workload_overrides,
    build_source_matrix,
    copy_reference,
    effective_bandwidth_gbps,
    is_async_copy_supported,
    load_lab_config_from_env,
    make_copy_output,
    make_transpose_output,
    measure_cuda_callable,
    require_async_copy_supported,
    transpose_reference,
)
from labs.memory_bandwidth_patterns.bandwidth_patterns_extension import (
    load_memory_bandwidth_patterns_extension,
)

ALL_VARIANTS = (
    "copy_scalar",
    "copy_vectorized",
    "copy_async_double_buffered",
    "transpose_naive",
    "transpose_tiled",
)


def _variant_list(value: str) -> list[str]:
    variants = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in variants if item not in ALL_VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    return variants


def _run_copy_variant(extension, variant: str, src_matrix: torch.Tensor, config: BandwidthLabConfig):
    src = src_matrix.view(-1)
    dst = make_copy_output(config, src_matrix.device)
    reference = copy_reference(src)
    if variant == "copy_scalar":
        fn = lambda: extension.copy_scalar(src, dst)
    elif variant == "copy_vectorized":
        fn = lambda: extension.copy_vectorized(src, dst)
    elif variant == "copy_async_double_buffered":
        require_async_copy_supported(src_matrix.device)
        fn = lambda: extension.copy_async_double_buffered(src, dst)
    else:
        raise ValueError(f"Unknown copy variant: {variant}")
    fn()
    torch.cuda.synchronize()
    torch.testing.assert_close(dst, reference, rtol=0.0, atol=0.0)
    return fn, dst


def _run_transpose_variant(extension, variant: str, src_matrix: torch.Tensor, config: BandwidthLabConfig):
    dst = make_transpose_output(config, src_matrix.device)
    reference = transpose_reference(src_matrix)
    if variant == "transpose_naive":
        fn = lambda: extension.transpose_naive(src_matrix, dst)
    elif variant == "transpose_tiled":
        fn = lambda: extension.transpose_tiled(src_matrix, dst)
    else:
        raise ValueError(f"Unknown transpose variant: {variant}")
    fn()
    torch.cuda.synchronize()
    torch.testing.assert_close(dst, reference, rtol=0.0, atol=0.0)
    return fn, dst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=None, help="Input matrix rows")
    parser.add_argument("--cols", type=int, default=None, help="Input matrix cols")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations per variant")
    parser.add_argument("--iterations", type=int, default=30, help="Timed iterations per variant")
    parser.add_argument(
        "--variants",
        type=str,
        default=",".join(ALL_VARIANTS),
        help=f"Comma-separated subset of {','.join(ALL_VARIANTS)}",
    )
    parser.add_argument(
        "--no-lock-gpu-clocks",
        dest="lock_gpu_clocks",
        action="store_false",
        help="Skip harness clock locking for quick local iteration",
    )
    parser.add_argument("--sm-clock-mhz", type=int, default=None, help="Optional SM application clock")
    parser.add_argument("--mem-clock-mhz", type=int, default=None, help="Optional memory application clock")
    parser.add_argument("--json", action="store_true", help="Print JSON payload")
    parser.add_argument("--json-out", type=Path, default=None, help="Write JSON payload to file")
    parser.set_defaults(lock_gpu_clocks=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("labs.memory_bandwidth_patterns requires CUDA.")

    config = apply_workload_overrides(
        load_lab_config_from_env(),
        [piece for piece in (f"--rows={args.rows}" if args.rows else None, f"--cols={args.cols}" if args.cols else None) if piece],
    )
    variants = _variant_list(args.variants)
    extension = load_memory_bandwidth_patterns_extension()
    lock_ctx = (
        lock_gpu_clocks(device=0, sm_clock_mhz=args.sm_clock_mhz, mem_clock_mhz=args.mem_clock_mhz)
        if args.lock_gpu_clocks
        else nullcontext()
    )

    with lock_ctx:
        src_matrix = build_source_matrix(config, torch.device("cuda"))
        results = []
        for variant in variants:
            if variant.startswith("copy_"):
                fn, _ = _run_copy_variant(extension, variant, src_matrix, config)
            else:
                fn, _ = _run_transpose_variant(extension, variant, src_matrix, config)
            latency_ms = measure_cuda_callable(fn, warmup=args.warmup, iterations=args.iterations)
            results.append(
                {
                    "variant": variant,
                    "latency_ms": latency_ms,
                    "effective_bandwidth_gbps": effective_bandwidth_gbps(
                        config.bytes_per_iteration,
                        latency_ms,
                    ),
                    "category": "copy" if variant.startswith("copy_") else "transpose",
                }
            )
        torch.cuda.synchronize()

    by_name = {item["variant"]: item for item in results}
    for item in results:
        if item["category"] == "copy" and "copy_scalar" in by_name and item["variant"] != "copy_scalar":
            item["relative_to_copy_scalar"] = (
                by_name[item["variant"]]["effective_bandwidth_gbps"] / by_name["copy_scalar"]["effective_bandwidth_gbps"]
                if by_name["copy_scalar"]["effective_bandwidth_gbps"] > 0
                else 0.0
            )
        if item["category"] == "transpose" and "transpose_naive" in by_name and item["variant"] != "transpose_naive":
            item["relative_to_transpose_naive"] = (
                by_name[item["variant"]]["effective_bandwidth_gbps"] / by_name["transpose_naive"]["effective_bandwidth_gbps"]
                if by_name["transpose_naive"]["effective_bandwidth_gbps"] > 0
                else 0.0
            )

    payload = {
        "deck_title": DECK_TITLE,
        "rows": config.rows,
        "cols": config.cols,
        "bytes_per_iteration": config.bytes_per_iteration,
        "async_copy_supported": is_async_copy_supported(torch.device("cuda")),
        "lock_gpu_clocks": args.lock_gpu_clocks,
        "sm_clock_mhz": args.sm_clock_mhz,
        "mem_clock_mhz": args.mem_clock_mhz,
        "variants": results,
    }

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("memory_bandwidth_patterns")
        print(f"shape={config.rows}x{config.cols} bytes_per_iteration={config.bytes_per_iteration}")
        print(f"async_copy_supported={payload['async_copy_supported']}")
        for item in results:
            line = (
                f"{item['variant']}: {item['latency_ms']:.6f} ms "
                f"({item['effective_bandwidth_gbps']:.3f} GB/s)"
            )
            if "relative_to_copy_scalar" in item:
                line += f" relative_to_copy_scalar={item['relative_to_copy_scalar']:.3f}x"
            if "relative_to_transpose_naive" in item:
                line += f" relative_to_transpose_naive={item['relative_to_transpose_naive']:.3f}x"
            print(line)
        if args.json_out is not None:
            print(f"wrote: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
