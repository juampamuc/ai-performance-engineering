"""Executable torchrun entrypoint for hybrid expert-parallel benchmarks."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from labs.fullstack_cluster.moe_hybrid_ep_common import run_cli


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--optimized", action="store_true", help=argparse.SUPPRESS)
    args, remainder = parser.parse_known_args(list(argv) if argv is not None else None)
    run_cli(optimized=bool(args.optimized), argv=remainder)


if __name__ == "__main__":
    main()
