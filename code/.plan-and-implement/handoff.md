## 2026-03-19T23:18:00Z

- `book-after` content is not under `code/book-after`; resolve the correct path before doing deeper README/book intent comparisons for later chapters.
- `ch04` rerun must be judged after removing the conflicting background CUDA process; the previous nonzero GPU-rescan return code was environmental, not a benchmark logic failure.
- The interrupted global pair-validation preflight already exposed future constructor-time multi-GPU skip issues in `ch13`, `ch15`, `ch17`, and `labs/cache_aware_disagg_inference`; confirm them through the sequential scope loop rather than restarting another competing GPU scan right now.

## 2026-03-20T03:53:27Z

- The local host can support a real single-GPU transport analogue, but not true multi-GPU NCCL/NIXL/NVSHMEM performance claims. Keep the README and validation language explicit about that boundary.
- The deck PDF is image-heavy; plain text extraction only recovered the cover summary. Local PDF/OCR tooling may need to be installed to extract more slide content.
- If true NIXL or NVSHMEM runtime enablement requires shared system setup outside the new lab directory, capture it as follow-up instead of reaching into shared repo files.


## 2026-03-20T03:53:51Z

- New task is isolated to `labs/moe_decode_blackwell_matrix/`; do not modify shared harness/docs surfaces for discovery or target registration in this pass.
- If a future merge wants bench CLI discovery or README index wiring, report that separately as merge-stage follow-up instead of doing it now.
- Prefer a synthetic decode-step MoE matrix that reuses existing MoE primitives over cloning the staged journey or adding new public benchmark-pair targets.

## 2026-03-20T03:56:50Z

- New active task supersedes the earlier matrix idea: build `labs/moe_decode_blackwell/` as a single benchmark pair, not a second matrix/playbook lab.
- Deck-derived features that should stay as README-only follow-up unless the repo grows new kernels: `PDL`, Blackwell warp-specialized tcgen05 kernels, `gather4`, runtime dynamic TMA box size, split-K cluster reductions, and multi-GPU EP / TP transport claims.
- The implementable kernel story for this pass is grouped-by-expert decode packing with fused FC1/Gated-SiLU style execution using existing PyTorch/CUDA primitives on Blackwell-class hardware.

## 2026-03-20T03:56:46Z

- There are mixed task threads in shared state; current execution target is strictly `labs/nccl_nixl_nvshmem/`. Ignore the later `moe_decode_blackwell_matrix` thread for this pass.
- Tooling/runtime facts that must stay explicit in implementation:
  - host has `1x B200` (no true multi-GPU collective measurement locally)
  - `nixl` import missing
  - `nvshmemrun` missing from `PATH`
  - PyTorch symmetric memory reports available
  - repo-local `tools/nccl-tests/build/all_reduce_perf` exists but meaningful collectives still require `>=2` GPUs
- Risk controls:
  - avoid any shared registry edits; rely on auto-discovery from local wrapper filenames
  - prefer explicit `SKIPPED:` and capability diagnostics over fallback behavior
  - if multi-GPU or external runtime setup is needed, record as follow-up instead of expanding scope

## 2026-03-20T04:13:30Z

- Final task state:
  - `labs/nccl_nixl_nvshmem/` is complete and validated locally.
  - `nixl` is importable and NVSHMEM runtime/launcher packages are installed locally, but the host still exposes only `1x B200`, so multi-GPU NCCL and NVSHMEM transport claims remain blocked by topology, not packaging.
- Shared follow-up intentionally left out of scope:
  - if the repo later wants a Python-level NVSHMEM binding probe beyond launcher/runtime detection, decide the supported binding/module name first
  - if publish-grade NCCL or NVSHMEM results are needed, rerun on a clean `>=2 GPU` host without the portable/shared-host concessions used here


## 2026-03-20T04:08:30Z

- Merge-stage follow-up only; not done here by design:
  - If the repo wants bench CLI discovery for this lab, add it in a separate change that intentionally touches shared harness/discovery surfaces.
  - If the stakeholder docs index should link this lab, do that separately with shared README refresh/regeneration.
  - If OCR-grade PDF extraction becomes important later, install a proper image-PDF OCR stack; `pypdf` alone does not recover slide text from these copies.

## 2026-03-20T04:12:00Z

- Active task is complete in `labs/moe_decode_blackwell/`; ignore the older matrix and transport-lab planning threads for merge review.
- Shared follow-up only; intentionally not done in this change:
  - If the repo wants this lab surfaced from any shared README/index page, do that separately.
  - If the repo wants a standardized PDF/OCR toolchain for future deck-based labs, promote the user-local tooling into a documented shared setup in a separate change.
- Important interpretation note:
  - The benchmark result is real and artifact-backed on the local B200 host, but it is a virtualized single-GPU implementation-state run. Treat it as evidence for the lab's narrow grouped-versus-control decode story, not as a publish-grade Blackwell serving claim.
