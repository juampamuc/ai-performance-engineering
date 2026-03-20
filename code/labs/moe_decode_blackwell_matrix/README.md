# Lab - MoE Decode Blackwell Matrix

## Summary
A Blackwell-class MoE decode scenario matrix for routing locality, persistent scheduling, and graph-launch tradeoffs. This is the canonical public surface for the deck-aligned Blackwell MoE decode work.

## Why This Exists
The source PDF in `.cursor/plans/` overlaps enough with the repo's existing MoE and decode material that a second public benchmark pair would be redundant. The more honest surface is a matrix/playbook lab that answers the actual tradeoff questions:

- how routing locality changes the decode-step shape
- when persistent scheduling helps
- when graph replay is worth the setup cost

This lab is the shared-doc Blackwell MoE decode entry.

## What This Lab Is
- a decode-step MoE scenario matrix for Blackwell-class GPUs
- a playbook surface with named sweep presets and structured artifacts
- focused on routing, scheduling, and launch tradeoffs rather than a single baseline/optimized claim

It reuses the repo's existing MoE primitives instead of duplicating the staged MoE or decode benchmark stories.

## Why This Is Not A Benchmark Pair
The important comparisons here are multiple named variants, not one universal optimized path. Keeping this as a matrix lab avoids inventing a fake one-dimensional story.

- `labs/moe_optimization_journey` is the staged MoE benchmark narrative.
- `labs/decode_optimization` is the broader decode hot-path microbenchmark suite.
- `labs/blackwell_gemm_optimizations` is the grouped expert / GEMM kernel journey.

`labs/moe_decode_blackwell_matrix` stays at the scenario/playbook layer above those labs.

## What It Measures
Each matrix point runs a deterministic single-step decode-style MoE expert path and records:

- per-step mean / stdev / p95 latency
- tokens per second and dispatch-tokens per second
- routing entropy, active-expert fraction, and max tokens per expert
- CUDA-graph capture cost when graph mode is enabled
- max absolute difference against the non-persistent grouped reference
- locked application clocks and theoretical device ceilings

## Learning Goals
- Keep the deck-aligned Blackwell MoE decode work visible without inventing a redundant benchmark pair.
- Make routing locality, persistent scheduling, and graph-launch tradeoffs reproducible and auditable.
- Give readers a MoE-specific decode scenario lab that is clearly distinct from the staged MoE, decode microbenchmark, and grouped-GEMM labs.

## Directory Layout
| Path | Description |
| --- | --- |
| `matrix_types.py`, `matrix_catalog.py` | Typed scenario definitions plus YAML playbook loading. |
| `preflight.py`, `runner.py`, `artifact_io.py`, `run_matrix.py` | Preflight checks, deterministic decode-step execution, artifact writing, and the main matrix runner. |
| `playbooks/deck_matrix.yaml`, `playbooks/smoke_b200.yaml` | Broad deck-style sweep plus fast local smoke preset. |
| `profiler/run_profile_compare.py`, `profiler/capture.py`, `profiler/compare.py` | Profiler-backed graph-vs-eager comparison helpers for prior matrix runs. |
| `tests/test_matrix_lab.py` | Lab-local pytest coverage for playbook loading, batch generation, and summary contracts. |

## Running The Matrix
Use the matrix runner for scenario sweeps. Use the profiler helper when you want a concrete graph-vs-eager comparison from an existing run.
```bash
python -m labs.moe_decode_blackwell_matrix.run_matrix --playbook smoke_b200 --run-id smoke_20260320
python -m labs.moe_decode_blackwell_matrix.run_matrix --playbook deck_matrix --run-id deck_20260320
```
- Use `smoke_b200` first; it is the fast local evidence path for a Blackwell host.
- This lab is script-first, not harness-target driven.
- Use this lab when you want the scenario/playbook view of Blackwell MoE decode tradeoffs rather than a single benchmark-pair claim.

## Validation Checklist
- `python -m compileall labs/moe_decode_blackwell_matrix` should succeed.
- `python -m pytest labs/moe_decode_blackwell_matrix/tests -q` should keep the playbook and summary contracts green.
- `python -m labs.moe_decode_blackwell_matrix.run_matrix --playbook smoke_b200 --run-id smoke_<date>` should emit a self-contained artifact directory under `artifacts/moe_decode_blackwell_matrix/runs/`.
- `python -m labs.moe_decode_blackwell_matrix.profiler.run_profile_compare --run-dir artifacts/moe_decode_blackwell_matrix/runs/<RUN_ID>` should write trace outputs plus a structured comparison summary.

## Playbooks

| Playbook | Intent |
| --- | --- |
| `deck_matrix` | Broad tradeoff sweep for the duplicated deck surface. |
| `smoke_b200` | Fast local B200 smoke run with artifact generation and graph-vs-eager evidence. |

## Artifact Layout

Every matrix run writes a self-contained directory under `artifacts/moe_decode_blackwell_matrix/runs/<RUN_ID>/`:

| Artifact | Contents |
| --- | --- |
| `sys_meta.json` | Playbook config, GPU metadata, profiler-tool visibility, lock state, and command. |
| `matrix.jsonl` | One row per matrix point, including unsupported and error states. |
| `matrix.csv` | Flat export for spreadsheet-style inspection. |
| `summary.json` | Best overall point plus persistent-vs-dynamic and graph-vs-eager deltas. |
| `manifest.json` | File hashes for reproducibility. |
| `profiles/<name>/summary.json` | Profiler comparison metadata and deltas. |
| `profiles/<name>/*.trace.json` | Chrome-trace exports from `torch.profiler`. |

## Profiler-Backed Comparison

```bash
python -m labs.moe_decode_blackwell_matrix.profiler.run_profile_compare               --run-dir artifacts/moe_decode_blackwell_matrix/runs/smoke_20260320
```

Or choose explicit config ids from `matrix.jsonl`:

```bash
python -m labs.moe_decode_blackwell_matrix.profiler.run_profile_compare               --run-dir artifacts/moe_decode_blackwell_matrix/runs/smoke_20260320               --config-a e8_k2_b8_stk_pst_egr               --config-b e8_k2_b8_stk_pst_grf
```

## Notes
- `dynamic + cuda_graph` is emitted as `unsupported` on purpose; the lab makes that constraint visible instead of silently falling back.
- `sticky` routing is the locality-friendly control, while the other policies help explain when persistent scheduling does or does not pay off.
- This lab remains the matrix/playbook companion to `labs/moe_optimization_journey`, `labs/decode_optimization`, and `labs/blackwell_gemm_optimizations`, not a replacement for those benchmark stories.
