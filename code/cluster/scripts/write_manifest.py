#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sanitize_label(raw: str) -> str:
    return raw.replace(".", "_").replace(":", "_")


def _read_json_if_exists(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write a manifest JSON for a cluster eval RUN_ID.")
    p.add_argument("--root", default="", help="Cluster root (default: inferred from this script location)")
    p.add_argument("--run-id", required=True, help="RUN_ID prefix")
    p.add_argument("--run-dir", default="", help="Explicit run directory (default: <cluster_root>/runs/<run_id> when present)")
    p.add_argument("--hosts", default="", help="Comma-separated host list (optional)")
    p.add_argument("--labels", default="", help="Comma-separated labels (optional; must match host count)")
    p.add_argument(
        "--include-figures",
        action="store_true",
        help="Include figure artifacts in the manifest.",
    )
    return p.parse_args()


def _resolve_cluster_root(raw_root: str) -> Path:
    return Path(raw_root).resolve() if raw_root else Path(__file__).resolve().parents[1]


def _resolve_run_dir(cluster_root: Path, run_id: str, raw_run_dir: str) -> Path | None:
    if raw_run_dir:
        return Path(raw_run_dir).resolve()
    candidate = cluster_root / "runs" / run_id
    if candidate.exists():
        return candidate
    return None


def _collect_run_dir_paths(run_dir: Path, include_figures: bool) -> List[Path]:
    paths: List[Path] = []
    for subdir in ("structured", "raw", "reports"):
        base = run_dir / subdir
        if not base.exists():
            continue
        paths.extend(sorted(path for path in base.rglob("*") if path.is_file()))
    if include_figures:
        figures_dir = run_dir / "figures"
        if figures_dir.exists():
            paths.extend(sorted(path for path in figures_dir.rglob("*") if path.is_file()))
    return sorted(set(paths))


def _collect_legacy_paths(cluster_root: Path, run_id: str, include_figures: bool) -> List[Path]:
    struct_dir = cluster_root / "results" / "structured"
    raw_dir = cluster_root / "results" / "raw"
    fig_dir = cluster_root / "docs" / "figures"

    paths_set = set()
    if struct_dir.exists():
        for p in struct_dir.glob(f"{run_id}_*"):
            if p.is_file():
                paths_set.add(p)
    if raw_dir.exists():
        for p in raw_dir.rglob(f"{run_id}_*"):
            if p.is_file():
                paths_set.add(p)
            elif p.is_dir():
                for fp in p.rglob("*"):
                    if fp.is_file():
                        paths_set.add(fp)
    if include_figures and fig_dir.exists():
        for p in fig_dir.glob(f"{run_id}_*"):
            if p.is_file():
                paths_set.add(p)
    return sorted(paths_set)


def _artifact_role_for(rel_path: str) -> str | None:
    name = Path(rel_path).name
    role_patterns = [
        ("fabric_command_catalog", "_fabric_command_catalog.json"),
        ("fabric_capability_matrix", "_fabric_capability_matrix.json"),
        ("fabric_verification", "_fabric_verification.json"),
        ("fabric_ai_correlation", "_fabric_ai_correlation.json"),
        ("fabric_scorecard", "_fabric_scorecard.json"),
        ("cluster_scorecard", "_cluster_scorecard.json"),
        ("benchmark_coverage_analysis", "_benchmark_coverage_analysis.json"),
        ("mlperf_alignment", "_mlperf_alignment.json"),
        ("suite_steps", "_suite_steps.json"),
        ("meta", "_meta.json"),
        ("nccl_allreduce", "_nccl.json"),
        ("nccl_alltoall", "_alltoall_nccl_alltoall.json"),
        ("vllm_request_rate_sweep", "_vllm_serve_request_rate_sweep.csv"),
        ("vllm_concurrency_sweep", "_vllm_serve_sweep.csv"),
        ("gemm_sanity", "_gemm_gpu_sanity.csv"),
        ("fio", "_fio.json"),
        ("nvbandwidth", "_nvbandwidth.json"),
        ("gpu_stream", "_gpu_stream.json"),
        ("allreduce_stability", "_allreduce_stability.json"),
        ("allreduce_latency_comp", "_allreduce_latency_comp.json"),
        ("allgather_control_plane", "_allgather_control_plane.json"),
        ("nccl_algo_comparison", "_nccl_algo_comparison.json"),
        ("train_step", "_train_step.json"),
        ("multinode_readiness", "_multinode_readiness.json"),
    ]
    for role, suffix in role_patterns:
        if name.endswith(suffix):
            return role
    return None


def _suite_steps_summary(run_dir: Path, run_id: str) -> Dict[str, Any]:
    suite_steps_path = run_dir / "structured" / f"{run_id}_suite_steps.json"
    payload = _read_json_if_exists(suite_steps_path)
    steps = payload if isinstance(payload, list) else []
    failed_steps: List[Dict[str, Any]] = []
    completed_steps = 0
    for step in steps:
        if not isinstance(step, dict):
            continue
        try:
            exit_code = int(step.get("exit_code", 0) or 0)
        except Exception:
            exit_code = 0
        if exit_code == 0:
            completed_steps += 1
        else:
            failed_steps.append(
                {
                    "name": step.get("name"),
                    "exit_code": exit_code,
                    "status": step.get("status"),
                    "log": step.get("log"),
                }
            )
    return {
        "suite_steps_path": str(suite_steps_path) if suite_steps_path.exists() else None,
        "step_count": len(steps),
        "completed_step_count": completed_steps,
        "failed_step_count": len(failed_steps),
        "failed_steps": failed_steps[:20],
    }


def _progress_summary(run_dir: Path) -> Dict[str, Any]:
    progress_path = run_dir / "progress" / "run_progress.json"
    payload = _read_json_if_exists(progress_path)
    current = payload.get("current") if isinstance(payload, dict) and isinstance(payload.get("current"), dict) else {}
    metrics = current.get("metrics") if isinstance(current.get("metrics"), dict) else {}
    return {
        "progress_path": str(progress_path) if progress_path.exists() else None,
        "timestamp": current.get("timestamp"),
        "phase": current.get("phase"),
        "step": current.get("step"),
        "step_detail": current.get("step_detail"),
        "percent_complete": current.get("percent_complete"),
        "elapsed_seconds": current.get("elapsed_seconds"),
        "eta_seconds": current.get("eta_seconds"),
        "status": metrics.get("status"),
        "completed_steps": metrics.get("completed_steps"),
        "total_steps": metrics.get("total_steps"),
        "current_step": metrics.get("current_step"),
        "suite_steps_path": metrics.get("suite_steps_path"),
    }


def _fabric_scorecard_summary(run_dir: Path, run_id: str) -> Dict[str, Any]:
    scorecard_path = run_dir / "structured" / f"{run_id}_fabric_scorecard.json"
    payload = _read_json_if_exists(scorecard_path)
    if not isinstance(payload, dict):
        return {
            "scorecard_path": str(scorecard_path) if scorecard_path.exists() else None,
            "scorecard_status": None,
            "completeness": None,
            "degraded_families": [],
        }
    families = payload.get("families") if isinstance(payload.get("families"), dict) else {}
    degraded_families = [
        name
        for name, values in families.items()
        if str((values or {}).get("completeness") or "") not in {"full_stack_verified", "runtime_verified"}
    ]
    return {
        "scorecard_path": str(scorecard_path),
        "scorecard_status": payload.get("status"),
        "completeness": payload.get("completeness"),
        "degraded_families": degraded_families,
    }


def _semantic_manifest_fields(run_dir: Optional[Path], run_id: str) -> Dict[str, Any]:
    if run_dir is None:
        return {
            "status": None,
            "suite_status": None,
            "success": None,
            "completeness": None,
            "issues": [],
            "progress": None,
        }

    suite = _suite_steps_summary(run_dir, run_id)
    progress = _progress_summary(run_dir)
    fabric = _fabric_scorecard_summary(run_dir, run_id)

    issues: List[str] = []
    if suite["failed_step_count"]:
        issues.append(f"{suite['failed_step_count']} suite step(s) failed")
    if fabric.get("scorecard_status") == "partial":
        issues.append("fabric completeness is partial for one or more families")

    if suite["failed_step_count"]:
        suite_status = "failed"
        success = False
    elif fabric.get("scorecard_status") == "partial":
        suite_status = "partial"
        success = True
    elif str(progress.get("status") or "") == "running":
        suite_status = "running"
        success = None
    elif str(progress.get("status") or "") == "completed" or suite["step_count"]:
        suite_status = "succeeded"
        success = True
    else:
        suite_status = "unknown"
        success = None

    return {
        "status": suite_status,
        "suite_status": suite_status,
        "success": success,
        "completeness": fabric.get("completeness"),
        "issues": issues,
        "progress": progress,
        "suite_steps": suite,
        "fabric": fabric,
    }


def build_manifest_payload(
    *,
    cluster_root: Path,
    run_id: str,
    run_dir: Optional[Path],
    include_figures: bool,
    hosts: List[str],
    labels: List[str],
) -> Tuple[Dict[str, Any], Path]:
    if run_dir is not None:
        paths = _collect_run_dir_paths(run_dir, include_figures)
        files = [str(p.relative_to(run_dir)) for p in paths]
        hashes = {str(p.relative_to(run_dir)): _sha256(p) for p in paths}
        out_path = run_dir / "manifest.json"
        artifact_root = str(run_dir.relative_to(cluster_root))
        manifest_mode = "run_dir"
    else:
        paths = _collect_legacy_paths(cluster_root, run_id, include_figures)
        files = [str(p.relative_to(cluster_root)) for p in paths]
        hashes = {str(p.relative_to(cluster_root)): _sha256(p) for p in paths}
        out_path = cluster_root / "results" / "structured" / f"{run_id}_manifest.json"
        artifact_root = "results"
        manifest_mode = "legacy_flat"

    artifact_counts: Dict[str, int] = {}
    for p in paths:
        suffix = p.suffix.lstrip(".") or "no_ext"
        artifact_counts[suffix] = artifact_counts.get(suffix, 0) + 1
    artifact_roles: Dict[str, List[str]] = {}
    for rel_path in files:
        role = _artifact_role_for(rel_path)
        if role is None:
            continue
        artifact_roles.setdefault(role, []).append(rel_path)

    nodes: List[Dict[str, Any]] = []
    for i, h in enumerate(hosts):
        label = labels[i] if labels else _sanitize_label(h)
        nodes.append({"label": label, "host": h})

    semantic = _semantic_manifest_fields(run_dir, run_id)
    manifest: Dict[str, Any] = {
        "manifest_version": 2,
        "run_id": run_id,
        "artifact_root": artifact_root,
        "manifest_mode": manifest_mode,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "nodes": nodes,
        "status": semantic["status"],
        "suite_status": semantic["suite_status"],
        "success": semantic["success"],
        "completeness": semantic["completeness"],
        "issues": semantic["issues"],
        "progress": semantic["progress"],
        "artifact_roles": artifact_roles,
        "files": files,
        "summary": {
            "file_count": len(files),
            "artifact_counts": artifact_counts,
            "sha256": hashes,
        },
    }
    if semantic.get("suite_steps"):
        manifest["suite_steps"] = semantic["suite_steps"]
    if semantic.get("fabric") and any(semantic["fabric"].values()):
        manifest["fabric"] = semantic["fabric"]
    return manifest, out_path


def main() -> int:
    args = parse_args()
    cluster_root = _resolve_cluster_root(args.root)
    run_id = args.run_id
    run_dir = _resolve_run_dir(cluster_root, run_id, args.run_dir)

    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    if labels and len(labels) != len(hosts):
        raise SystemExit("--labels count must match --hosts count")
    manifest, out_path = build_manifest_payload(
        cluster_root=cluster_root,
        run_id=run_id,
        run_dir=run_dir,
        include_figures=args.include_figures,
        hosts=hosts,
        labels=labels,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
