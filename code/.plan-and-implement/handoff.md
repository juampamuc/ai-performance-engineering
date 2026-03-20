## 2026-03-20T14:39:00Z

- Decision: use the `plan-and-implement` shared-state workflow, but keep all planning and implementation in the main thread because the user did not authorize sub-agent delegation.
- Open question to resolve during the audit: which issues can be enforced mechanically with a repo audit script and which require manuscript-level manual review.
- Working assumption: chapter source of truth is `book-after/chN.md` or `book-after/chNN.md`; lab source of truth is the corresponding lab `README.md`, plus `labs/README.md` for pair-vs-playbook expectations.
- Resolved for tooling: the benchmark-pair audit now records a `scope_contract` for each reviewed scope, including source doc path, lab review mode (`benchmark-pair`, `benchmark-story`, `playbook-matrix`, or `challenge-kernel-lab`), and contract findings.
- New blocker removed: the nested audit pytest bundle was failing on a too-tight isolated-pair timeout test; the timeout now allows the descendant process to spawn before cleanup assertions run.

## 2026-03-20T16:33:50Z

- Resolved for this phase: the immediate benchmark-validity cleanup is complete, and the remaining work is now a pure coverage problem across untouched scopes.
- Resolved for tolerance review so far:
- `ch13:fp8_static` remains exact-match by design and by audit evidence.
- `ch15`, `ch17`, and `ch18` inference-style paths should not be loosened casually because they compare discrete token outputs rather than float activations.
- Next execution order:
- `ch16` scope audit and manuscript review
- `ch01` and `ch09` scope audit and manuscript review
- then untouched Batch 3 chapters
- then remaining labs by review mode
- Remaining question to settle during later batches: whether any untouched chapter or lab has a real workload-equivalence defect rather than the metadata/signature/comment issues fixed so far.

## 2026-03-20T16:42:31Z

- Resolved for the next untouched chapter pass:
- `ch16`, `ch01`, and `ch09` are clean by audit evidence.
- The `ch16` manual fixes were narrative/runtime-path cleanup, not harness relaxations or workload changes.
- Carry-forward focus for Batch 3:
- confirm `ch03` and `ch04` finish cleanly
- then continue with `ch06`, `ch07`, `ch08`, `ch15`, `ch17`, and `ch19`
- then move to the remaining lab sweep
- Remaining question to watch for in later scopes: whether any untouched benchmark still encodes a conceptual mismatch like a stale optimization claim or misnamed comparison that the current static pair checks do not catch automatically.

## 2026-03-20T16:44:23Z

- Resolved in Batch 3 so far:
- `ch03` is clean by full audit evidence.
- `ch04` currently shows no review/compliance defects; only the slower later audit stages are still pending.
- Carry-forward execution order:
- wait for `ch04` to finish or inspect its pair-validation artifacts once written
- then continue with `ch06`, `ch07`, `ch08`, `ch15`, `ch17`, and `ch19`
- then proceed to the remaining lab sweep

## 2026-03-20T17:42:30Z

- Resolved in this continuation pass:
- `ch04`, `ch06`, `ch07`, `ch08`, `ch15`, `ch17`, and `ch19` are clean by full audit evidence.
- All remaining lab batches are closed:
- playbook/matrix scopes are honestly classified and documented
- benchmark-story scopes are clean after adding static signatures where the audit previously had to execute runtime paths just to compare workloads
- benchmark-pair and challenge-kernel scopes are clean after adding static signatures to the timeout-prone families and explicitly classifying all `nvfp4_*` labs
- Additional benchmark-local cleanup completed after the broad lab reruns:
- `labs/persistent_decode` now publishes one consistent workload signature across the baseline, graph, Triton, and CUDA variants, and the targeted scope rerun is green on the latest code.
- Carry-forward focus if more work is requested:
- investigate the known `labs/flashattention4` sliding-window kernel instability on this host if the goal is environment remediation rather than honest audit classification
- otherwise the requested review/fix sweep is complete, and any next pass should be a publish-grade bare-metal rerun rather than more repo-local cleanup

## 2026-03-20T18:00:42Z

- Resolved in the targeted follow-up:
- the other session's changes were only partially correct
- the repo keeps the broad metric/comment cleanup, but not the inconsistent parts
- Final semantic decisions from this pass:
- keep the newer `precision_type=\"fp32\"` and `precision_type=\"int8\"` callers, but support them explicitly in `core/benchmark/metrics.py`
- keep `compute_gemm_metrics()` for `ch14:cutlass`, but report the real FP16 storage/roofline inputs
- do not extend the verification precision schema just to add `fp32` or `int8`; remove those unsupported keys from per-benchmark payloads instead
- Confirmed runtime conclusions:
- `ch14:cutlass` succeeded cleanly under strict validity with corrected FP16 GEMM metrics
- `labs/speculative_decode:speculative_decode` succeeded cleanly under strict validity with the cleaned payload contract
- `labs/flashattention4:flashattention4_windowed` reproduces as an optimized-path skip because all candidate providers yield non-finite outputs on this stack
- Carry-forward focus if more work is requested:
- publish-grade reruns should happen on bare metal
- `labs/flashattention4` remediation would be a provider/kernel-stack task, not a benchmark-validity cleanup task
