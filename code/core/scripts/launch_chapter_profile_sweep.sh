#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  launch_chapter_profile_sweep.sh \
    --targets-file <path> \
    --minimal-run-id <run_id> \
    --deep-run-id <run_id> \
    [--sweep-dir <path>] \
    [--refresh-target <chapter:example>] \
    [--refresh-run-id <run_id>] \
    [--artifacts-dir <path>] \
    [--timeout-multiplier <float>] \
    [--ncu-metric-set <preset>]

Runs an optional refresh target, then a minimal sweep, then a deep-dive sweep.
Status is written to <sweep-dir>/launch_status.txt, including explicit *_rc fields
when a phase fails so stale "minimal_running" states do not linger.
EOF
}

SWEEP_DIR=""
TARGETS_FILE=""
MINIMAL_RUN_ID=""
DEEP_RUN_ID=""
REFRESH_TARGET=""
REFRESH_RUN_ID=""
ARTIFACTS_DIR="artifacts/runs"
TIMEOUT_MULTIPLIER="3.0"
NCU_METRIC_SET="minimal"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sweep-dir)
      SWEEP_DIR="$2"
      shift 2
      ;;
    --targets-file)
      TARGETS_FILE="$2"
      shift 2
      ;;
    --minimal-run-id)
      MINIMAL_RUN_ID="$2"
      shift 2
      ;;
    --deep-run-id)
      DEEP_RUN_ID="$2"
      shift 2
      ;;
    --refresh-target)
      REFRESH_TARGET="$2"
      shift 2
      ;;
    --refresh-run-id)
      REFRESH_RUN_ID="$2"
      shift 2
      ;;
    --artifacts-dir)
      ARTIFACTS_DIR="$2"
      shift 2
      ;;
    --timeout-multiplier)
      TIMEOUT_MULTIPLIER="$2"
      shift 2
      ;;
    --ncu-metric-set)
      NCU_METRIC_SET="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$TARGETS_FILE" || -z "$MINIMAL_RUN_ID" || -z "$DEEP_RUN_ID" ]]; then
  usage >&2
  exit 2
fi

if [[ -z "$SWEEP_DIR" ]]; then
  SWEEP_DIR="artifacts/sweeps/${MINIMAL_RUN_ID}__launch"
fi

mkdir -p "$SWEEP_DIR"
STATUS_FILE="$SWEEP_DIR/launch_status.txt"
MIN_LOG="$SWEEP_DIR/minimal.log"
DEEP_LOG="$SWEEP_DIR/deep_dive.log"
REFRESH_LOG="$SWEEP_DIR/refresh.log"

: > "$MIN_LOG"
: > "$DEEP_LOG"
: > "$REFRESH_LOG"

{
  echo "queued_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "targets_file=$TARGETS_FILE"
  echo "minimal_run_id=$MINIMAL_RUN_ID"
  echo "deep_run_id=$DEEP_RUN_ID"
  echo "phase=queued"
} > "$STATUS_FILE"

if [[ -n "$REFRESH_TARGET" ]]; then
  if [[ -z "$REFRESH_RUN_ID" ]]; then
    echo "refresh-target requires --refresh-run-id" >&2
    exit 2
  fi
  {
    echo "refresh_target=$REFRESH_TARGET"
    echo "refresh_run_id=$REFRESH_RUN_ID"
    echo "refresh_started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "phase=refresh_running"
  } >> "$STATUS_FILE"
  if python -m cli.aisp bench run \
      --targets "$REFRESH_TARGET" \
      --profile none \
      --single-gpu \
      --run-id "$REFRESH_RUN_ID" \
      --artifacts-dir "$ARTIFACTS_DIR" \
      --suite-timeout 0 \
      --timeout-multiplier "$TIMEOUT_MULTIPLIER" \
      --update-expectations 2>&1 | tee -a "$REFRESH_LOG"; then
    REFRESH_RC=0
  else
    REFRESH_RC=${PIPESTATUS[0]}
  fi
  {
    echo "refresh_finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "refresh_rc=$REFRESH_RC"
  } >> "$STATUS_FILE"
  if [[ "$REFRESH_RC" -ne 0 ]]; then
    echo "phase=refresh_failed" >> "$STATUS_FILE"
    exit "$REFRESH_RC"
  fi
fi

mapfile -t TARGETS < "$TARGETS_FILE"
ARGS=()
for target in "${TARGETS[@]}"; do
  [[ -n "$target" ]] || continue
  ARGS+=(--targets "$target")
done

{
  echo "minimal_started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "phase=minimal_running"
} >> "$STATUS_FILE"
if python -m cli.aisp bench run \
    "${ARGS[@]}" \
    --profile minimal \
    --single-gpu \
    --run-id "$MINIMAL_RUN_ID" \
    --artifacts-dir "$ARTIFACTS_DIR" \
    --suite-timeout 0 \
    --timeout-multiplier "$TIMEOUT_MULTIPLIER" \
    --ncu-metric-set "$NCU_METRIC_SET" 2>&1 | tee -a "$MIN_LOG"; then
  MINIMAL_RC=0
else
  MINIMAL_RC=${PIPESTATUS[0]}
fi
{
  echo "minimal_finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "minimal_rc=$MINIMAL_RC"
} >> "$STATUS_FILE"
if [[ "$MINIMAL_RC" -ne 0 ]]; then
  echo "phase=minimal_failed" >> "$STATUS_FILE"
  exit "$MINIMAL_RC"
fi

{
  echo "deep_started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "phase=deep_running"
} >> "$STATUS_FILE"
if python -m cli.aisp bench run \
    "${ARGS[@]}" \
    --profile deep_dive \
    --single-gpu \
    --run-id "$DEEP_RUN_ID" \
    --artifacts-dir "$ARTIFACTS_DIR" \
    --suite-timeout 0 \
    --timeout-multiplier "$TIMEOUT_MULTIPLIER" 2>&1 | tee -a "$DEEP_LOG"; then
  DEEP_RC=0
else
  DEEP_RC=${PIPESTATUS[0]}
fi
{
  echo "deep_finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "deep_rc=$DEEP_RC"
} >> "$STATUS_FILE"
if [[ "$DEEP_RC" -ne 0 ]]; then
  echo "phase=deep_failed" >> "$STATUS_FILE"
  exit "$DEEP_RC"
fi

echo "phase=completed" >> "$STATUS_FILE"
