from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.benchmark.e2e_sweep import watch_benchmark_e2e_sweep_foreground


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch an e2e benchmark sweep and auto-resume stale runs.")
    parser.add_argument("--run-id", required=True, help="Top-level e2e run id to supervise.")
    parser.add_argument("--repo-root", type=Path, default=None, help="Optional repository root override.")
    parser.add_argument("--poll-interval-seconds", type=int, default=15, help="Watcher poll interval in seconds.")
    parser.add_argument("--max-auto-resumes", type=int, default=3, help="Maximum auto-resume attempts before giving up.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = watch_benchmark_e2e_sweep_foreground(
        run_id=args.run_id,
        repo_root=args.repo_root.resolve() if args.repo_root else None,
        poll_interval_seconds=max(1, int(args.poll_interval_seconds)),
        max_auto_resumes=max(0, int(args.max_auto_resumes)),
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
