## 2026-03-19T23:18:00Z

- Objective: Run a strict sequential stabilization pass over every chapter and every real lab, using the pair-audit plus harness workflow as the source of truth; fix fairness/runtime/doc mismatches and keep README claims aligned with `book-after`.
- Acceptance criteria:
  - Each touched scope has `0` static review findings.
  - Each touched scope has `0` compliance warnings/errors.
  - Pair validation is clean, with explicit skips for true multi-GPU-only paths on this 1-GPU host.
  - GPU rescans complete without repo-caused failures.
  - Benchmark-facing README/docs stay consistent with current code and `book-after`.
  - Canonical pairs either show a real improvement on this host or are honestly reclassified.
- Constraints:
  - Single B200 host only; multi-GPU runtime proof is not required, but gating and documentation must still be credible.
  - Keep the full `--include-gpu-rescan` path as requested, even though it is slow.
  - Do not hand-edit generated chapter/lab READMEs; update generator sources and regenerate.
- Current status:
  - Preflight entrypoint cleanup completed.
  - README/doc surface tests completed.
  - `ch01` through `ch03` audited and green.
  - Root README generator command example corrected to use repeatable `--targets` instead of shell-glob `ch*`.
  - `ch04` needed multi-GPU constructor-skip fixes plus a harness classification fix so single-process torchrun benchmarks are not auto-skipped as distributed.
  - The generated `ch04` README validation checklist now routes the bandwidth benchmark through `cli.aisp bench run` instead of direct benchmark-module execution.
  - Remaining indirect benchmark-module CLI shims in `labs/cache_aware_disagg_inference` and `labs/fullstack_cluster` were removed, and the AST contract/tests now reject that shim pattern.
  - `ch04` strict rerun is in progress with no competing GPU workloads; the live run is advancing cleanly through the chapter with strict virtualization warnings only.

## 2026-03-20T03:53:27Z

- Objective: Create `labs/nccl_nixl_nvshmem/` from the GTC deck as a self-contained lab that matches repo style, owns only that directory, and honestly models communication-stack tradeoffs.
- Acceptance criteria:
  - New directory contains a README, auto-discoverable benchmark-pair target, and a standalone runnable probe/compare entrypoint.
  - The benchmark pair is honest on this host: it measures a real control-path or transport decision we can execute locally instead of faking multi-GPU NIXL/NVSHMEM results.
  - NCCL, NIXL, NVSHMEM, and symmetric-memory availability are surfaced explicitly with clear failure/constraint messages.
  - At least one real repo invocation is dogfooded and recorded with exact command plus outcome.
  - No shared files outside `labs/nccl_nixl_nvshmem/` are edited unless a blocker makes that unavoidable.
- Constraints:
  - Current host is `1x NVIDIA B200`; multi-GPU runtime proof is unavailable locally.
  - `torch`, `cupy`, and PyTorch symmetric memory are available; `nixl` import is absent; `nvshmemrun` is absent.
  - Repo-local `tools/nccl-tests/build/all_reduce_perf` exists, but meaningful NCCL collective validation still requires `>=2` GPUs.
  - Install only the local tools needed to inspect the source deck and validate the lab honestly.
- Current status:
  - Reference labs (`labs/fullstack_cluster`, `labs/train_distributed`, `labs/dynamic_router`) reviewed.
  - Lab class chosen: benchmark-pair, backed by a single-GPU transport analogue plus an explicit stack probe runner.
  - Shared planner files are being updated by append-only sections to avoid colliding with the older stabilization thread.


## 2026-03-20T03:53:51Z

- Objective: Create `labs/moe_decode_blackwell_matrix/` from the duplicate GTC deck "Optimize MoE for low-latency inference decode on Blackwell" as a self-contained matrix/playbook lab rather than a cloned benchmark-pair lab.
- Acceptance criteria:
  - Only the new lab directory plus plan-and-implement state files are edited.
  - The lab delivers a README, sweep runner(s), structured artifact output, and profiler-backed comparison support.
  - The README explicitly states the source PDF is duplicated and that the repo benefits more from a matrix/playbook surface than from cloning a second pair lab.
  - At least one real local smoke invocation on the B200 host produces reproducible artifacts.
  - Validation includes syntax/import checks, a targeted pytest path inside the new lab, and a real GPU execution path.
- Constraints:
  - Do not edit shared repo files outside `labs/moe_decode_blackwell_matrix/` and the required shared-state files.
  - Keep the new lab intentionally non-overlapping with any future/core `moe_decode_blackwell` pair lab.
  - Install missing local tooling if needed for PDF inspection or local validation.
- Current status:
  - Adjacent lab patterns inspected (`moe_optimization_journey`, `dynamic_router`, `blackwell_gemm_optimizations`).
  - Local host confirmed as single NVIDIA B200 with CUDA 13 / Triton 3.5.1.
  - PDF text extraction tooling is currently missing; local installation is needed if deck extraction remains necessary.

## 2026-03-20T03:56:50Z

- Objective: Create `labs/moe_decode_blackwell/` as a compact benchmark-pair lab from the GTC deck, scoped to a single honest decode-story on the local B200 host.
- Acceptance criteria:
  - Only `labs/moe_decode_blackwell/` and the required shared-state files are edited.
  - The new lab exposes an auto-discoverable baseline/optimized benchmark pair with runnable `cli.aisp bench` targets.
  - The optimized path is faithful to an implementable deck slice: grouped-by-expert decode packing plus fused FC1 / gated-SiLU style expert work on a Blackwell-class GPU.
  - The README states directly which deck ideas are intentionally out of scope here (`PDL`, TMA `gather4`, runtime dynamic TMA box size, NVLink / multi-GPU expert parallelism, native tcgen05 kernels).
  - Validation includes syntax/import checks and at least one real `cli.aisp bench` invocation on the local B200 host.
- Constraints:
  - Keep production code edits inside `labs/moe_decode_blackwell/`; shared follow-up must be reported, not patched.
  - The deck PDF is image-based; use local OCR/tooling rather than guessing slide content.
  - This lab is single-GPU and Blackwell-class only; do not make multi-GPU or TensorRT-LLM claims the repo cannot prove here.
- Current status:
  - Adjacent lab patterns reviewed: `labs/moe_optimization_journey`, `labs/blackwell_gemm_optimizations`, `labs/decode_optimization`, plus `labs/moe_cuda_ptx` for routed grouped-expert structure.
  - Local deck extraction now works through OCR (`pdftoppm` + `tesseract`); key deck themes recovered: latency regime, grouped-by-expert execution, fused FC1 / gated-SiLU, Blackwell warp-specialization, PDL overlap, TMA sparse-token loading, and dynamic box-size TMA.
  - A user-local PDF tooling venv was created at `~/.local/share/aipe-codex-tools/pypdf-venv` after the system Python blocked direct package install.
  - Local runtime confirmed as `1x NVIDIA B200 (SM100)`.

## 2026-03-20T04:13:30Z

- Objective: Finalize `labs/nccl_nixl_nvshmem/` with honest local tool/runtime detection, recorded validation, and no shared-surface edits.
- Acceptance criteria status:
  - New lab directory is complete with README, benchmark pair, and standalone runner.
  - Real repo invocation completed successfully on the local host.
  - Local stack probe now reports the installed state accurately: `nixl` importable, NVSHMEM launcher present, multi-GPU data paths still blocked by topology.
  - Shared files outside `labs/nccl_nixl_nvshmem/` remain untouched except append-only planner state.
- Current status:
  - `nixl` installed into the user Python environment with `pip --user --break-system-packages`.
  - CUDA 13 NVSHMEM dev/runtime packages upgraded locally; launcher discovered at `/usr/bin/nvshmem_13/nvshmrun`.
  - Final portable/shared-host harness artifact recorded under `artifacts/review/labs/nccl_nixl_nvshmem_portable_shared_v3/20260320_041146__bench__profile_none_targets_labs_nccl_nixl_nvshmem_tier_handoff/`.
  - Remaining hard blocker is physical topology: this host still exposes only `1x B200`, so true NCCL collective and NVSHMEM one-sided multi-GPU validation remains out of scope locally.


## 2026-03-20T04:08:30Z

- Current status:
  - Local PDF tooling installed in `/home/cfregly/.local/share/aipe-pdftools-venv`; `pypdf` confirms both duplicate PDFs are 58-page documents but the extracted text is empty on sampled pages, so implementation stayed anchored on the explicit user scope plus adjacent lab patterns.
  - `labs/moe_decode_blackwell_matrix/` now exists as a self-contained matrix/playbook lab with playbook presets, deterministic sweep logic, structured artifacts, and profiler comparison support.
  - Validation is complete: compile/import checks passed, lab-local pytest passed, a real B200 smoke matrix run produced artifacts, the lab-local profiler comparison produced trace artifacts, and both Nsight Systems and Nsight Compute captured the new runtime path successfully.
  - Shared repo surfaces remain untouched outside `.plan-and-implement/*`; merge-stage follow-up is documented separately.


## 2026-03-20T04:10:05Z

- Current status:
  - Profiler comparison helper updated to relock at the matrix run's recorded app clocks (`1500/3996 MHz`) instead of the device max, and the auto-pair comparison was rerun successfully.

## 2026-03-20T04:12:00Z

- Objective: Create `labs/moe_decode_blackwell/` as a compact benchmark-pair lab from the GTC deck, scoped to a single honest decode story on the local B200 host.
- Acceptance criteria:
  - Only `labs/moe_decode_blackwell/` and the required shared-state files are edited.
  - The new lab exposes an auto-discoverable baseline/optimized benchmark pair with runnable `cli.aisp bench` targets.
  - The optimized path stays inside an implementable deck slice: grouped-by-expert decode packing plus fused FC1 / gated-SiLU style expert work on a Blackwell-class GPU.
  - The README states directly which deck ideas are intentionally out of scope here.
  - Validation includes syntax/import checks and at least one real `cli.aisp bench` invocation on the local B200 host.
- Constraints:
  - Keep production code edits inside `labs/moe_decode_blackwell/`; shared follow-up must be reported, not patched.
  - The deck PDF is image-based; use local OCR/tooling rather than guessing slide content.
  - This lab is single-GPU and Blackwell-class only; do not make multi-GPU or TensorRT-LLM claims the repo cannot prove here.
- Current status:
  - Completed.
  - Local OCR/tooling installed and used to recover deck themes.
  - `labs/moe_decode_blackwell/` is implemented with README, benchmark wrappers, shared runtime, and lab-local tests.
  - Validation is complete: compile/import checks passed, pytest passed, target discovery passed, benchmark lint passed, and a real `cli.aisp bench run` on the local B200 host produced a successful artifact-backed result.
  - Measured outcome from the real run: baseline `7.739 ms`, optimized `0.159 ms`, `48.80x` speedup, verification passed, app clocks recorded as `1500/3996 MHz`, provenance marked `hardware_key=b200` and `execution_environment=virtualized`.
