"""Entry point for ZeRO-2 demos (baseline vs optimized)."""

from __future__ import annotations

import argparse
import sys

import labs.train_distributed.baseline_zero2 as baseline_single_run
import labs.train_distributed.baseline_zero2_multigpu as baseline_multi_run
import labs.train_distributed.optimized_zero2 as optimized_single_run
import labs.train_distributed.optimized_zero2_multigpu as optimized_multi_run


def main():
    parser = argparse.ArgumentParser(description="ZeRO-2 training examples.")
    parser.add_argument(
        "--mode",
        choices=["baseline", "optimized"],
        default="optimized",
        help="Select which variant to run.",
    )
    parser.add_argument(
        "--variant",
        choices=["single", "multigpu"],
        default="single",
        help="Select the single-GPU or multi-GPU implementation.",
    )
    args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining
    if args.variant == "multigpu":
        run = baseline_multi_run if args.mode == "baseline" else optimized_multi_run
    else:
        run = baseline_single_run if args.mode == "baseline" else optimized_single_run
    run.main()


if __name__ == "__main__":
    main()
