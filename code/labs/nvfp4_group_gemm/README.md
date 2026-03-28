# Lab - NVFP4 Grouped GEMM

## Summary
Explores grouped-GEMM routing and schedule variants across multiple cases so you can see where the grouped NVFP4 path is actually winning and where it is merely legal.

## Problem
Grouped GEMM tuning is noisy and easy to overclaim. This lab keeps the case routing explicit and benchmarked so promotions are based on repeated verified wins instead of one-off lows.

## Baseline Path
- per-case baseline grouped GEMM paths
- stable routing reference for cases 0-3
- useful for showing which grouped shapes are hard versus easy

## Optimized Path
- per-case tuned NVFP4 grouped GEMM variants
- same grouped workloads, but explicit schedule/routing choices
- designed to keep promotions tied to repeated verify and ABAB checks

## Measured Delta
Fresh portable B200 reruns on this host kept cases 0-2 in the small-effect band:

| Target | Baseline | Optimized | Measured delta | Contract |
| --- | ---: | ---: | ---: | --- |
| `nvfp4_group_gemm_case0` | `2.408 ms` | `2.377 ms` | `1.01x` | informational control surface |
| `nvfp4_group_gemm_case1` | `2.079 ms` | `2.021 ms` | `1.03x` | informational control surface |
| `nvfp4_group_gemm_case2` | `0.615 ms` | `0.594 ms` | `1.04x` | informational control surface |

The older strict all-case snapshots in `artifacts/runs/20260302_rerun_all_labschapters_strict/` are still useful historical router evidence, but they are not the current runnable truth for these three harness targets on this host. Treat `case0`, `case1`, and `case2` as supplementary informational control surfaces; keep canonical speed claims on the still-winning case routes and on the stricter ABAB/router tuning workflow.

## Profiler Evidence
```bash
python -m cli.aisp bench run --targets labs/nvfp4_group_gemm --profile deep_dive --single-gpu
```

Use the harness artifacts for schedule attribution, then use the router/ABAB tooling for promotion decisions. The benchmark pair tells you the shape of the win; the tuning scripts decide whether a default should actually move.

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/nvfp4_group_gemm
python -m cli.aisp bench run --targets labs/nvfp4_group_gemm --profile minimal
```

## Learning Goals
- Keep grouped-GEMM tuning grounded in repeated verified case-by-case evidence.
- Keep `case0`, `case1`, and `case2` visible as routing controls without forcing them to carry the lab's canonical speed claim on every host.
- Separate exploration scripts from the regression-tracked benchmark defaults.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_nvfp4_group_gemm_case0.py` ... `baseline_nvfp4_group_gemm_case3.py` | Per-case baseline grouped-GEMM entrypoints. |
| `optimized_nvfp4_group_gemm_case0*.py` ... `optimized_nvfp4_group_gemm_case3*.py` | Per-case tuned grouped-GEMM variants. |
| `WORKLOG.md`, `custom_cuda_submission.py`, `cutlass_extension.py` | Tuning log and implementation plumbing for the promoted routes. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/nvfp4_group_gemm
python -m cli.aisp bench run --targets labs/nvfp4_group_gemm --profile minimal
```
- Targets follow the `labs/nvfp4_group_gemm:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/nvfp4_group_gemm:<workload>="--flag value"` to sweep schedule knobs.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Portable runs do not write expectation files unless `--allow-portable-expectations-update` is also provided.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/nvfp4_group_gemm:nvfp4_group_gemm_case3 --profile minimal` should keep the promoted case3 route verification-clean.
- `nvfp4_group_gemm_case0`, `nvfp4_group_gemm_case1`, and `nvfp4_group_gemm_case2` remain informational control surfaces; use the ABAB/router tooling when deciding whether any of them should become canonical speed-claim targets again.
- Default changes should still be gated by the stricter ABAB/verify process documented in the codebase notes, not by a single benchmark run.

## Notes
- This lab is intentionally stricter than a normal benchmark pair because grouped-GEMM route tuning is unusually noise-prone.
- The benchmark harness now treats `case0`, `case1`, and `case2` as informational control surfaces on this host-aligned repo surface, because fresh portable B200 reruns only reproduced 1.01-1.04x gains while preserving clean verification and profiler coverage.
