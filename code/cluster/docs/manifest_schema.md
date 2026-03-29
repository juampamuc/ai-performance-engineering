# Manifest Schema

Current version: `2`

## Purpose
This document pins the manifest schema used by `scripts/collect_discovery_and_tcp_sysctl.sh` so future changes are explicit and traceable.

## Versioning
- Increment `manifest_version` when fields are added, removed, or meaningfully changed.
- Keep backwards compatibility notes in a new section for each version.

## Schema (v2)

| Field | Type | Description |
| --- | --- | --- |
| manifest_version | integer | Schema version. |
| run_id | string | Run identifier (e.g., `2026-02-05`). |
| timestamp_utc | string | ISO-8601 timestamp in UTC. |
| nodes | array | List of nodes in the run (`label` + `host`). |
| status | string or null | Semantic top-level run status (`succeeded`, `partial`, `failed`, `running`, `unknown`). |
| suite_status | string or null | Alias of `status` for suite consumers. |
| success | boolean or null | Whether the run completed semantically successfully. `partial` fabric runs can still set `success=true`. |
| completeness | string or null | Optional completeness signal, currently populated from fabric scorecards when present. |
| issues | array | Short semantic issues suitable for direct surfacing in tooling. |
| progress | object or null | Compact copy of the latest `progress/run_progress.json` current snapshot. |
| suite_steps | object | Step-count summary and failed-step preview when suite-steps are available. |
| fabric | object | Fabric scorecard summary when a fabric evaluation was part of the run. |
| files | array | Relative paths of artifacts created for the run. |
| summary | object | Integrity summary and artifact counts. |
| summary.file_count | integer | Total number of files in `files`. |
| summary.artifact_counts | object | Counts by file extension (e.g., `json`, `txt`). |
| summary.sha256 | object | Mapping of relative file path to SHA-256 hash. |

## Backwards Compatibility
- `v1` manifests remain valid for inventory-only consumers.
- `v2` adds semantic status/progress fields without removing any `v1` fields.

## Example (v2)
```json
{
  "manifest_version": 2,
  "run_id": "2026-02-05",
  "timestamp_utc": "2026-02-05T04:55:12.345678+00:00",
  "status": "succeeded",
  "suite_status": "succeeded",
  "success": true,
  "completeness": null,
  "issues": [],
  "progress": {
    "step": "complete",
    "step_detail": "completed 32/32 suite steps",
    "percent_complete": 100.0
  },
  "nodes": [
    {"label": "node1", "host": "node1.example.internal"},
    {"label": "node2", "host": "node2.example.internal"}
  ],
  "files": [
    "results/structured/2026-02-05_node1_meta.json",
    "results/structured/2026-02-05_node1_tcp_sysctl.json"
  ],
  "summary": {
    "file_count": 2,
    "artifact_counts": {"json": 2},
    "sha256": {
      "results/structured/2026-02-05_node1_meta.json": "...",
      "results/structured/2026-02-05_node1_tcp_sysctl.json": "..."
    }
  }
}
```
