# NVFP4 Group GEMM (B200 Competition Shapes)

This lab mirrors the input shapes from the GPU MODE `nvfp4_group_gemm` leaderboard and is intended
to let us iterate on a strong contender using the AISP harness (clock locking, correctness checks,
Nsight traces), then port the resulting kernel strategy back into a Popcorn `submission.py`.

## Targets

List targets:
```bash
python -m cli.aisp bench list-targets --chapter labs/nvfp4_group_gemm
```

Run the suite (baseline vs optimized) with profiling:
```bash
python -m cli.aisp bench run --targets labs/nvfp4_group_gemm --profile deep_dive --update-expectations
```

Notes:
- The baseline uses a straightforward CuTe DSL implementation (close to the official starter).
- The CuTe DSL optimized variants remove per-call allocations by caching metadata tensors and
  tensormap buffers in `setup()`, matching the competition evaluation style where `custom_kernel()`
  is called repeatedly with identical shapes.
- The CUTLASS optimized variants (`optimized_*_cutlass.py`) use a single-launch, device-scheduled
  grouped GEMM kernel adapted from `third_party/cutlass/examples/75_blackwell_grouped_gemm/`.
  Tuning knobs (optional):
  - `AISP_NVFP4_GROUP_GEMM_CLUSTER_M`, `AISP_NVFP4_GROUP_GEMM_CLUSTER_N`
  - `AISP_NVFP4_GROUP_GEMM_RASTER_ORDER`
  - `AISP_NVFP4_GROUP_GEMM_USE_PDL`

## Case2/Case3 Optimization Log (Correctness-First)

This section documents the recent kernel-level case2/case3 work so others can reproduce decisions,
understand trade-offs, and continue from the current state.

### Ground rules used for every promotion

- `--verify` is the hard gate before any latency claim.
- ABAB interleaved comparisons are required before changing defaults.
- No fallback routing for correctness or legality failures.
- Candidates that are fast but fail verify or fail compile are rejected.

### Current promoted defaults (router)

File: `labs/nvfp4_group_gemm_v2/popcorn_submission_tuned_router.py`

| Case | Variant | cluster_m | cluster_n | raster_order | use_pdl | max_swizzle |
|---|---|---:|---:|---:|---:|---:|
| 0 | `2sm` | 2 | 1 | 2 | 1 | 8 |
| 1 | `2sm` | 2 | 1 | 2 | 0 | 8 |
| 2 | `1sm_n128_case2_nvf4` | 1 | 2 | 2 | 1 | 16 |
| 3 | `1sm_n128_case3` | 1 | 2 | 0 | 0 | 8 |

### Current strict baseline snapshot

Artifact: `/tmp/nvfp4_defaults_post_kernelpass_verify_20260228_010106.json`

- verify: pass on all 4 cases
- geomean_per_group_us: `10.7945`
- case0: `9.5766`
- case1: `7.2268`
- case2: `15.6016`
- case3: `12.5744`

### Decisions made in the latest kernel pass

| Candidate | Type | Verify | Key result | Decision |
|---|---|---|---|---|
| `1sm_n128_case2_s2` | New case2 block-scaled StageCount=2 lane | Pass | geomean `11.7811`, case2 `23.4424` (worse than baseline) | Rejected |
| `1sm_n256_case2_nvf4` | New case2 NVF4 lane (N=256, K=256) | Pass | geomean `12.6116`, case2 `29.4173` (worse than baseline) | Rejected |
| `1sm_n256_case2_nvf4` vs incumbent | ABAB (4 pairs) | N/A | delta_mean_B_minus_A `+1.8393 us` (candidate slower), stdev `0.0974` | Rejected |
| `1sm_n128_case2_nvf4_s5` | New case2 NVF4 StageCount=5 lane | Compile fails | CUTLASS static assert: `SMEM usage exceeded capacity` | Removed |

ABAB artifact for the `n256` decision:
- `/tmp/nvfp4_case2_n256_abab_20260228_1772240988.json`

### Focused case2 family screen (same harness conditions)

Screen run (`--no-verify`, warmup=2, repeats=16) with case3 pinned to `1sm_n128_case3` showed
the current incumbent still best on geomean:

| Case2 variant | geomean_per_group_us | case2_us | note |
|---|---:|---:|---|
| `1sm_n128_case2_nvf4` | `10.6771` | `15.3860` | incumbent best geomean |
| `1sm_n128_case2_nvf4_epi64` | `10.7131` | `15.4910` | slightly slower geomean |
| `1sm_n128_case2_nvf4_epi64x128` | `10.7910` | `16.1270` | slower |
| `1sm_n128_case2_nvf4_epi128` | `10.8234` | `15.9970` | slower |
| `1sm_n128_case2_nvf4_s4` | `10.9761` | `15.2970` | case2 a bit better, case3 much worse |
| `1sm_n128_case2_nvf4_s1` | `16.1286` | `74.7810` | pathological |

### Root-cause findings (important)

- Case2 forced onto 2SM families (`2sm`, `2sm_n64*`, `2sm_n128*`) currently fails `CUTLASS can_implement()`
  in this submission path.
- NVF4 StageCount=5 for the 1SM N=128 case2 lane is not legal on SM100 because shared memory exceeds
  capacity at compile time.
- Several candidates can be verify-green but still regress geomean materially; verify alone is necessary
  but not sufficient for promotion.

### Code changes from this pass

Main file touched:
- `labs/nvfp4_group_gemm/cutlass_nvfp4_grouped_gemm_sm100.cu`

What was added:
- New exploratory variant `1sm_n256_case2_nvf4` (metadata + plan bindings + pybind exports).

What was added and then removed (invalid):
- `1sm_n128_case2_nvf4_s5` (removed after SMEM-capacity compile failure).

Supporting documentation updated:
- `AGENTS.md` (new findings and non-promotable paths recorded)

### Repro commands

Strict verify on current defaults:
```bash
python -u /tmp/nvfp4_router_eval.py \
  --submission-file labs/nvfp4_group_gemm_v2/popcorn_submission_tuned_router.py \
  --json --warmup 2 --repeats 20
```

Strict verify for a case2 candidate:
```bash
python -u /tmp/nvfp4_router_eval.py \
  --submission-file labs/nvfp4_group_gemm_v2/popcorn_submission_tuned_router.py \
  --json --warmup 1 --repeats 12 \
  --set AISP_NVFP4_GROUP_GEMM_CASE2_VARIANT=1sm_n256_case2_nvf4 \
  --set AISP_NVFP4_GROUP_GEMM_CASE3_VARIANT=1sm_n128_case3
```

### Next directions (open-ended)

Current router/tunable space for case2 appears saturated. Next high-ROI directions are kernel-contract
or scheduling changes, not more environment flips:

1. Build a truly different legal case2 kernel family (new tile/schedule contract) that changes
   execution behavior without tripping `can_implement()` or SMEM limits.
2. Explore case2-specific epilogue/mainloop pairings that improve case2 without harming case3,
   and gate every candidate with strict verify + ABAB.
3. Keep all default promotions tied to repeated geomean evidence (ABAB + stability reruns), not
   single-pass wins.
4. Continue documenting rejected branches with quantitative evidence so future sessions avoid rework.