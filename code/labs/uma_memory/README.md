# Lab - UMA Memory Diagnostics

## Summary
Diagnostics for UMA / unified-memory systems: capture device-visible free memory, host reclaimable memory, and JSON snapshots before you make claims about allocator behavior or memory-fit limits.

## Problem
On UMA-capable systems, a "GPU memory" question is often really a host-memory and reclaimability question. This lab exists to make that boundary explicit instead of inferring it from a single `nvidia-smi` or `cudaMemGetInfo()` number.

## What This Lab Is
- a diagnostics/tool lab
- human-readable reporting plus JSON snapshots
- useful before larger UMA, memory-fit, or allocation experiments

It is **not** currently a baseline/optimized benchmark pair, and discovery intentionally reports zero harness targets here.

## Learning Goals
- Measure device-free memory and host-available memory together.
- Estimate UMA allocatable capacity with an explicit reclaim assumption.
- Capture reproducible snapshots before and after runtime, allocator, or workload changes.

## Directory Layout
| Path | Description |
| --- | --- |
| `uma_memory_reporting.py` | Main reporting tool that prints a human-readable report and can emit JSON snapshots. |
| `uma_memory_utils.py` | Helpers for parsing `/proc/meminfo`, formatting byte counts, and detecting integrated GPUs. |
| `__init__.py` | Package marker for the lab/tool module. |

## Running the Tool
Use the tool entrypoint for standard reporting or call the script directly when iterating locally.
```bash
python -m cli.aisp tools uma-memory -- --device-index 0
python -m cli.aisp tools uma-memory -- --device-index 0 --snapshot --snapshot-dir artifacts/uma_memory_snapshots
python labs/uma_memory/uma_memory_reporting.py --json --device-index 0
```
- This lab is tool-driven, not benchmark-pair driven.
- Snapshot JSON is the artifact to compare across allocator, driver, or system-configuration changes.

## Validation Checklist
- `python -m cli.aisp tools uma-memory -- --device-index 0` prints a readable summary without requiring manual parsing.
- `python -m cli.aisp tools uma-memory -- --device-index 0 --snapshot` writes a structured JSON artifact under `artifacts/uma_memory_snapshots/`.
- Direct-script output and tool-registry output should agree for the same `--device-index` and reclaim settings.

## Notes
- The tool combines `torch.cuda.mem_get_info()` with `/proc/meminfo` so the report stays explicit about what is device memory vs host-side reclaimability.
- Use this as environment evidence for chapters/labs that discuss UMA or memory fit; it is not a substitute for workload benchmarks.
