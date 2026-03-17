## 2026-03-17T21:21:34+00:00

- Objective: create a new lab from Together AI's "Cache-Aware Disaggregated Inference" article.
- Acceptance criteria:
  - add a repo-native lab with benchmark entrypoints and shared implementation code
  - baseline and optimized paths clearly map to article concepts
  - include README/discovery wiring so the lab is discoverable
  - add targeted tests
  - dogfood at least one realistic CLI benchmark path and record outcomes
- Constraints:
  - keep the existing dirty worktree intact
  - do not fold article logic into the current `labs/fullstack_cluster/moe_hybrid_ep_*` WIP unless explicitly requested
  - prefer additive changes with minimal impact on unrelated lab flows
- Current status: planning complete enough to start implementation; proceeding with a new additive lab under `labs/cache_aware_disagg_inference`.

## 2026-03-17T21:33:45+00:00

- Outcome: implementation complete.
- The new lab lives under `labs/cache_aware_disagg_inference` and is wired into generator-owned README targets.
- Validation status:
  - targeted test file passed
  - direct module invocation passed
  - harness CLI benchmark passed on the local B200 host
- Remaining caveat: the shared host reports foreign CUDA Python processes, so the harness dogfood run required `--allow-foreign-gpu-processes` together with `--validity-profile portable`.
