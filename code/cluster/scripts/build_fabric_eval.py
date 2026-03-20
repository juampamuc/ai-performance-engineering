#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cluster.fabric import build_fabric_payloads
from cluster.fabric.evaluator import make_management_config


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build structured fabric verification artifacts for a cluster run.")
    parser.add_argument("--run-id", required=True, help="Canonical run ID")
    parser.add_argument("--run-dir", default="", help="Canonical run directory (default: cluster/runs/<run_id>)")
    parser.add_argument("--primary-label", default="", help="Primary host label for workload correlation")
    parser.add_argument("--labels", default="", help="Optional comma-separated labels to scope meta discovery")
    parser.add_argument("--ssh-user", default="", help="Optional SSH user for management-plane checks")
    parser.add_argument("--ssh-key", default="", help="Optional SSH key for management-plane checks")
    parser.add_argument("--nmx-url", default="", help="Optional NMX base URL override for NVLink management-plane checks")
    parser.add_argument("--nmx-token", default="", help="Optional NMX bearer token for the management plane")
    parser.add_argument("--ib-mgmt-host", default="", help="Optional InfiniBand management host for fabric CLI checks")
    parser.add_argument("--ib-mgmt-user", default="", help="Optional SSH user for the InfiniBand management host")
    parser.add_argument("--ib-mgmt-ssh-key", default="", help="Optional SSH key for the InfiniBand management host")
    parser.add_argument("--cumulus-hosts", default="", help="Optional comma-separated Cumulus/Spectrum-X switch hosts")
    parser.add_argument("--cumulus-user", default="", help="Optional SSH user for Cumulus/Spectrum-X switches")
    parser.add_argument("--cumulus-ssh-key", default="", help="Optional SSH key for Cumulus/Spectrum-X switches")
    parser.add_argument("--source-root", default="", help="Optional fabric source-doc root override")
    parser.add_argument(
        "--require-management-plane",
        action="store_true",
        help="Fail when publish-grade fabric validation is requested but no management plane is configured",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cluster_root = Path(__file__).resolve().parents[1]
    run_dir = Path(args.run_dir).resolve() if args.run_dir else (cluster_root / "runs" / args.run_id).resolve()
    structured_dir = run_dir / "structured"
    structured_dir.mkdir(parents=True, exist_ok=True)
    management = make_management_config(
        ib_mgmt_host=args.ib_mgmt_host or None,
        ib_mgmt_user=args.ib_mgmt_user or None,
        ib_mgmt_ssh_key=args.ib_mgmt_ssh_key or None,
        nmx_url=args.nmx_url or None,
        nmx_token=args.nmx_token or None,
        cumulus_hosts=_parse_csv(args.cumulus_hosts) if args.cumulus_hosts else None,
        cumulus_user=args.cumulus_user or None,
        cumulus_ssh_key=args.cumulus_ssh_key or None,
        ssh_user=args.ssh_user or None,
        ssh_key=args.ssh_key or None,
    )

    payloads = build_fabric_payloads(
        run_id=args.run_id,
        run_dir=run_dir,
        primary_label=args.primary_label or None,
        labels=_parse_csv(args.labels) or None,
        management=management,
        ssh_user=args.ssh_user or None,
        ssh_key=args.ssh_key or None,
        source_root=Path(args.source_root).resolve() if args.source_root else None,
        require_management_plane=bool(args.require_management_plane),
    )

    output_paths = {
        "fabric_command_catalog": structured_dir / f"{args.run_id}_fabric_command_catalog.json",
        "fabric_capability_matrix": structured_dir / f"{args.run_id}_fabric_capability_matrix.json",
        "fabric_verification": structured_dir / f"{args.run_id}_fabric_verification.json",
        "fabric_ai_correlation": structured_dir / f"{args.run_id}_fabric_ai_correlation.json",
        "fabric_scorecard": structured_dir / f"{args.run_id}_fabric_scorecard.json",
        "fabric_scorecard_md": structured_dir / f"{args.run_id}_fabric_scorecard.md",
    }

    for key, path in output_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if key.endswith("_md"):
            markdown = str((payloads.get(key) or {}).get("markdown") or "")
            path.write_text(markdown, encoding="utf-8")
        else:
            _write_json(path, payloads[key])
        print(f"Wrote {path}")

    if args.require_management_plane and payloads["fabric_scorecard"]["status"] == "error":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
