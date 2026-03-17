# Lab - Cache-Aware Disaggregated Inference

## Summary
Recreates the article-level scheduler story behind cache-aware prefill/decode disaggregation: a cache-unaware round-robin baseline versus cache-affine decode placement with a shared KV hierarchy and a warm/cold request mix.

## Problem
Naive prefill/decode disaggregation makes the topology look correct while still wasting locality. If chunk handoff bounces each request across decode workers, warm prefixes stop being warm and KV reload traffic overwhelms the supposed benefit of disaggregation.

## Baseline Path
- round-robin logical decode placement
- shared-prefix reloads whenever chunk ownership changes
- a useful control for showing how temporal locality gets destroyed

## Optimized Path
- cache-affine worker assignment
- warm prefixes stay resident on the same logical decode worker
- the benchmark reports cache hit rate, KV transfer volume, worker switches, and TTFT/TPOT so the win is explained, not just timed

## Learning Goals
- Compare cache-unaware round-robin handoff against cache-aware decode affinity.
- Make temporal and spatial locality visible through custom metrics rather than narrative alone.
- Keep the lab runnable on one GPU by simulating logical workers instead of requiring a full cluster.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_cache_aware_disagg.py`, `optimized_cache_aware_disagg.py`, `cache_aware_disagg_common.py` | Single-GPU logical-worker benchmark pair plus the shared scheduler/cache model that reproduces the article's core behavior. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/cache_aware_disagg_inference
python -m cli.aisp bench run --targets labs/cache_aware_disagg_inference --profile minimal
```
- Targets follow the `labs/cache_aware_disagg_inference:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/cache_aware_disagg_inference:<workload>="--flag value"` to sweep schedule knobs.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Portable runs do not write expectation files unless `--allow-portable-expectations-update` is also provided.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/cache_aware_disagg_inference --profile minimal` compares the cache-unaware and cache-aware paths through the standard harness.
- `python -m labs.cache_aware_disagg_inference.baseline_cache_aware_disagg` prints JSON metrics for the round-robin control path.
- `python -m labs.cache_aware_disagg_inference.optimized_cache_aware_disagg` prints JSON metrics for the cache-affine path with lower KV transfer and fewer worker switches.

## Notes
- This lab is intentionally a logical reproduction of the scheduler/caching story, not a full serving engine.
- The defaults model chunked prefill, warm/cold requests, and a 2P1D-style control problem without forcing an 8-GPU host.
