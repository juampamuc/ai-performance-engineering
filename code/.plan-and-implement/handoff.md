## 2026-03-17T21:21:34+00:00

- Assumption: the requested lab should be additive and article-faithful, not a rewrite of the existing `moe_hybrid_ep` WIP in `labs/fullstack_cluster`.
- Implementation bias: favor a reproducible single-GPU logical-worker simulation over an 8-GPU hard requirement so the lab can be dogfooded locally while still teaching the article's cache-locality ideas.
- Open decision to revisit only if evidence demands it: whether to add multi-GPU wrappers in this pass or keep the first version single-GPU only.

## 2026-03-17T21:33:45+00:00

- No blocker remains for this pass.
- Deferred enhancement only: add a multi-GPU or torchrun-backed variant if the user wants a closer reproduction of the article's full cluster split beyond the current logical-worker lab.
- Host caveat recorded: harness validation on this machine needed `--allow-foreign-gpu-processes` because other CUDA Python processes were already resident on the GPU.
