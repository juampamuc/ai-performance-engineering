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
