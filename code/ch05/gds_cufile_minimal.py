#!/usr/bin/env python3

"""Minimal cuFile/GDS capability probe.

This script no longer publishes host-mediated throughput under a GPUDirect
Storage name. It either confirms that usable cuFile/GDS support is present or
fails fast with `SKIPPED:`.
"""

from __future__ import annotations

import sys

import argparse
import os
from pathlib import Path

import torch


def _ensure_test_file(path: Path, size: int) -> None:
    """Create a test file filled with random bytes if missing or undersized."""
    if path.exists() and path.stat().st_size >= size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(os.urandom(size))
    print(f"[OK] Generated test file: {path} ({size} bytes)")


def _require_gds_support() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("SKIPPED: gds_cufile_minimal requires CUDA")
    try:
        from cuda.bindings import cufile  # type: ignore
    except (ImportError, OSError) as exc:
        raise RuntimeError("SKIPPED: gds_cufile_minimal requires usable cuFile/GDS support") from exc

    try:
        version = str(cufile.get_version())
    except Exception as exc:  # pragma: no cover - driver/library specific
        raise RuntimeError("SKIPPED: gds_cufile_minimal requires usable cuFile/GDS support") from exc

    if not Path("/etc/cufile.json").exists():
        raise RuntimeError("SKIPPED: gds_cufile_minimal requires usable cuFile/GDS support")
    return version


def run_gds_probe(path: Path, num_bytes: int) -> None:
    """Verify that cuFile/GDS prerequisites are present for the target file."""
    version = _require_gds_support()
    print(f"[OK] cuFile bindings detected (version: {version})")
    print(f"[OK] GDS probe target: {path} ({num_bytes} bytes requested)")
    print("[OK] Capability probe passed; use a real cuFile benchmark on a supported host for throughput numbers.")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path("/tmp/gds_test_file.bin"),
        help="Path to the binary file to read (default: /tmp/gds_test_file.bin).",
    )
    parser.add_argument(
        "num_bytes",
        type=int,
        nargs="?",
        default=1 << 20,
        help="Number of bytes to read (default: 1 MiB).",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Create the file with random bytes if it is missing or too small.",
    )
    parser.add_argument(
        "--profile-output-dir",
        type=Path,
        default=None,
        help="Optional profiler output directory (ignored).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    path = args.path

    # Auto-generate test file if it doesn't exist or is smaller than requested.
    try:
        need_generate = args.generate or not path.exists()
        if not need_generate:
            try:
                need_generate = path.stat().st_size < args.num_bytes
            except OSError:
                need_generate = True
        if need_generate:
            _ensure_test_file(path, args.num_bytes)
    except OSError as exc:
        print(f"SKIPPED: Unable to prepare file '{path}': {exc}")
        return 3

    try:
        run_gds_probe(path, args.num_bytes)
    except RuntimeError as exc:
        message = str(exc).strip()
        if message.startswith("SKIPPED:"):
            print(message)
            return 3
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
