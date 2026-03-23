## 2026-03-20T14:39:00Z

1. Inventory the audit surface.
   Confirm chapter coverage, lab coverage, manuscript/README sources of truth, and current benchmark methodology constraints.
2. Build the audit rubric and lightweight tooling.
   Capture the checks that distinguish a valid comparison from a misleading one: frozen workload, isolated variable, correctness parity, honest naming/doc shape, reproducible harness path, and measurable artifact coverage.
3. Audit chapters batch by batch.
   Start with chapters that have either existing remediation packets or active worktree changes, then continue through the remaining chapters.
   First target batch: `ch10`, `ch11`, `ch14`, `ch16`, `ch18`, followed by the currently touched `ch01`, `ch02`, `ch09`, `ch12`, `ch13`, `ch20`.
4. Audit labs batch by batch.
   Separate benchmark-pair labs from playbook/matrix labs and fix code/docs/harness mismatches instead of forcing fake pair semantics.
   Use the new `scope_contract` audit step to decide whether each lab should be treated as a strict pair review or an honest workflow/playbook review.
5. Verify every touched path and maintain an evidence ledger.
   Record exact commands, outcomes, and any remaining blockers or non-feasible runs.

## 2026-03-20T16:33:50Z

1. Close the remaining untouched chapter scopes from Batch 1 and Batch 2.
   Run `run_benchmark_pair_audit` for `ch16`, then for `ch01` and `ch09`.
   For each scope: read the `book-after` chapter intent, inspect benchmark metadata/comments/tolerances, fix any misleading comparison details locally, rerun the scope audit, and dogfood any changed runtime path through `cli.aisp bench run`.
2. Finish the untouched Batch 3 chapter sweep.
   Audit `ch03` through `ch08`, then `ch15`, `ch17`, and `ch19`.
   Focus especially on tolerance correctness, signature publication, and whether the code still matches the manuscript optimization story without introducing extra optimization variables.
3. Finish the repo-wide lab sweep in review-mode order.
   First remaining `benchmark-pair` and `benchmark-story` labs, then `challenge-kernel-lab`, then `playbook-matrix`.
   For non-pair labs, enforce honest README classification and `scope_contract` output rather than inventing synthetic baseline/optimized claims.
4. Re-run the full audit evidence bundle after each milestone boundary.
   Use the nested pytest bundle as a hard gate, rerun static review after meaningful repo-wide batches, and keep realistic benchmark invocations for every touched runtime path.
5. Finalize the evidence ledger and remaining-risk list.
   Record exact artifact paths, note any expected skips or unavailable-hardware blockers, and clearly separate virtualized dogfood evidence from canonical publish-grade evidence.

## 2026-03-20T17:42:30Z

1. Close the continuation sweep with evidence, not more broad edits.
   The requested chapter scopes and remaining lab scopes are now clean through the audit workflow.
   Any further work should be targeted follow-up, not another repo-wide pass.
2. Preserve the current classification contract.
   Keep `labs/README.md` aligned with the actual `labs/` directory set and preserve the explicit benchmark-pair / benchmark-story / playbook-matrix / challenge-kernel split.
3. Treat known skips as environment findings unless the user asks for remediation.
   The remaining `labs/flashattention4` windowed skips are due to unstable experimental kernels on this `torch 2.9.1 + sm_100` stack.
   They are no longer benchmark-validity bugs; fixing them would be an environment/kernel stability task.
4. Use bare-metal reruns for any publish-grade next step.
   The current evidence is good for repo-local validation and virtualized dogfood, but any final published numbers should be rerun on bare metal with the same strict harness flow.

## 2026-03-20T18:00:42Z

1. Review the other session's edits semantically, not cosmetically.
   Keep the broad cleanup if it matches helper semantics and workload intent.
   Overwrite any caller-only change that leaves the helper contract inconsistent or introduces unsupported verification metadata.
2. Fix only the local inconsistencies uncovered by that review.
   Update `compute_precision_metrics()` for `fp32`/`int8`, correct the CUTLASS GEMM precision/byte metadata, and remove unsupported `precision_flags` keys from the touched benchmarks.
3. Re-verify the touched scopes through the existing audit workflow.
   Re-run `run_benchmark_pair_audit` for `ch14`, `labs/speculative_decode`, and `labs/flashattention4`.
4. Dogfood at least one clean chapter target and one clean lab target under the strict validity profile.
   Use isolated `cli.aisp bench run` invocations and let the foreign-process guard stay strict.
5. Re-check the `labs/flashattention4` windowed path in isolation.
   Confirm whether the remaining windowed skip is still a provider/kernel instability on the current stack or if a repo change is required.
## 2026-03-21T17:28:00Z

1. Reconcile canonical expectation files with the verified fairness runs.
   Remove stale speed expectations for informational-only controls and rewrite `ch13:kv_cache_naive` as a memory-goal benchmark using the measured portable B200 run.
2. Regenerate the affected chapter READMEs from the updated expectation surface.
   Refresh `ch05`, `ch13`, and `ch20` after the expectation edits so docs stop advertising removed or reclassified canonical deltas.
3. Rebuild the portable rerun queue/state from the repaired canonical surface.
   Remove stale `problem_queue.jsonl` entries for resolved targets, preserve `expected_unsupported.jsonl`, and keep aggregate counts synchronized in `state.json`.
4. Re-run static and repository-level verification.
   Confirm py_compile/pytest/readme-refresh/rerun-status all pass against the reconciled state and record the exact evidence trail in `results.md`.

## 2026-03-21T18:20:00Z

1. Finish the outstanding lab failure bucket from the portable rerun.
   Re-run `labs/custom_vs_cublas:tcgen05_matmul`, `labs/nvfp4_gemv:nvfp4_gemv`, `labs/moe_cuda_ptx:moe_layer`, and `labs/parameterized_cuda_graphs:parameterized_graph_launch` on fresh run ids after fixing the local benchmark/runtime defects.
2. Convert runtime-capability failures into explicit unsupported classifications.
   Keep `labs/train_distributed:zero2` in `expected_unsupported.jsonl` with a runtime-capability reason rather than a generic queue failure.
3. Rewrite the portable queue/state to reflect post-fix truth, not stale rerun noise.
   Remove resolved lab failures from `problem_queue.jsonl`, preserve `moe_layer` only as a verified `non_speedup`, and keep the aggregate totals synchronized in `state.json`.
4. Re-run the repo verification bundle after the last NVFP4 loader change.
   Confirm compile, queue status, and the targeted pytest subset all still pass.

## 2026-03-23T04:33:00Z

1. Treat `ch10:persistent_matmul_tma` as locally resolved on the portable `b200` host.
   Keep the tuned launch meta unless a later canonical rerun disproves the benefit on bare metal.
2. Carry `ch13:torchao_quantization` as the next explicit decision point.
   The local evidence now shows a real B200 non-speedup for both the current int8 dynamic path and a quick float8 dynamic screen.
   Any next change here must stay honest to the chapter story; do not silently “fix” it by folding in `torch.compile`.
3. Continue queue cleanup with the next highest-value local cases that are still benchmark-local and actionable on one GPU.
   Priority after this loop: `ch12:cuda_graphs_conditional` contract cleanup, then the remaining chapter `out_of_tolerance` cases that do not require canonical multi-GPU hardware.
4. Keep real multi-GPU validation deferred to a host that actually has `>=2` GPUs.
   The current host remains valid for structural checks, local single-GPU benchmark fixes, and portable queue reconciliation only.

## 2026-03-23T05:10:00Z

1. Preserve the completed local fixes.
   Keep `ch12:cuda_graphs_conditional` informational/non-canonical and keep the `ch11:stream_ordered_kv_cache` three-stream tuning unless a later canonical rerun disproves it.
2. Use user confirmation before workload-shape retunes.
   The next promising local improvements (`ch10:attention`, `ch13:precisionmixed`) have clear directional evidence, but they likely require increasing benchmark workload size rather than just changing implementation details.
3. If workload retunes are approved, prioritize the most defensible story first.
   First `ch10:attention` (Flash/SDPA benefit scales cleanly with sequence length), then `ch13:precisionmixed` (mixed-precision training benefit scales with model/workload size).
4. Keep multi-GPU semantics unchanged.
   Do not use any single-GPU emulation or fallback paths as evidence for multi-GPU targets.

## 2026-03-23T05:46:00Z

1. Preserve the successful workload retunes.
   Keep `ch10:attention` at `seq_len=1280` and `ch13:precisionmixed` at `hidden_dim=3072` unless a later canonical rerun disproves them.
2. Do not churn the compile-scope benchmarks without a contract decision.
   The local benchmark-class probes do not show a persuasive retune path for `ch13:regional_compile` or `ch14:regional_triton`.
3. Next local target, if continuing without a contract rewrite, is `ch13:torchao_quantization`.
   It remains a real local non-speedup and is the clearest benchmark-local problem still standing on this host.
4. Keep multi-GPU semantics unchanged.
   Real multi-GPU validation remains deferred to a host with `>=2` GPUs.

## 2026-03-23T13:35:00Z

1. Preserve the Chapter 12 rerun-helper fix.
   Keep the typed CUDA expectation-key handling in `scripts/full_virtualized_rerun.py`; this was a real false-positive queue bug.
2. Keep the Chapter 14 warmup fix, but do not treat the pair as locally settled.
   The optimized setup warmup is still the right benchmark-local behavior, yet repeated locked-clock reruns remain unstable on this virtualized host.
3. Prioritize `ch13:torchao_quantization` next.
   It remains the clearest unresolved single-GPU benchmark-local non-speedup, and any next move must stay quantization-only rather than quietly folding in compilation.
4. Do not remove or de-list `ch14:regional_triton` yet.
   The right next step is either deeper diagnosis or a contract discussion after more evidence, not removal by convenience.
