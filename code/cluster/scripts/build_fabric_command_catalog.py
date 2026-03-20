#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from cluster.fabric.catalog import DEFAULT_SOURCE_ROOT, generate_catalog_payload, write_catalog_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the curated machine-readable fabric command catalog.")
    parser.add_argument("--run-id", default="", help="Optional run ID for embedded catalog metadata")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT), help="Root directory of archived HTML fabric docs")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "fabric" / "fabric_command_catalog.json"),
        help="Catalog output path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = generate_catalog_payload(Path(args.source_root).resolve(), run_id=args.run_id)
    write_catalog_payload(output, payload)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
