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
## 2026-03-21T17:28:00Z

- No architectural blocker is open.
- The only active decision is classification, and the current implementation direction is fixed by runtime evidence:
- `ch05:ai` stays informational because the fair baseline is slower than the optimized path after removing the artificial sync inflation
- `ch20:pipeline_sequential` stays informational because the repaired overlap story is not a canonical speedup on this verified run
- `ch13:kv_cache_naive` stays canonical but changes to `optimization_goal="memory"` because the fair optimized path materially reduces memory while slightly regressing runtime

## 2026-03-21T18:20:00Z

- No unresolved implementation blocker remains for the requested benchmark-validity + rerun-queue plan.
- Final classification decisions from the lab follow-up:
- `labs/custom_vs_cublas:tcgen05_matmul`, `labs/nvfp4_gemv:nvfp4_gemv`, and `labs/parameterized_cuda_graphs:parameterized_graph_launch` are resolved manual rerun successes and should stay out of `problem_queue.jsonl`.
- `labs/train_distributed:zero2` should remain in `expected_unsupported.jsonl` because the current PyTorch build lacks the required `batched_reduce_scatter_hook`; this is now an explicit runtime-capability classification rather than a generic failure.
- `labs/moe_cuda_ptx:moe_layer` is no longer a correctness failure. It now belongs in the queue only as a verified `non_speedup` target until someone improves the optimized implementation or decides to de-list it from speed-goal coverage.
- Carry-forward note if more work is requested:
- the next logical step is optional lab non-speedup triage (`moe_layer` and any other lab targets still queued for `non_speedup`), not more queue-noise cleanup

## 2026-03-22T23:31:00Z

- Resolved in this follow-up:
- three canonical non-speedup targets now have verified wins on the current single virtualized `b200` host:
  - `ch13:matmul_pytorch`
  - `ch20:end_to_end_bandwidth`
  - `ch20:memory_standard`
- Their stale March 21 problem-queue entries are removed, and the rerun snapshot now reports `queued_problem_total=158`.
- Current blocker / decision point for any next pass:
- the remaining `ch04` single-GPU non-speedups are not quick kernel swaps; they need a principled single-GPU virtual-parallel redesign if we want honest wins without de-listing
- Do not “fix” the `ch04` items by weakening checks or injecting strawman overhead. If work continues there, it should be a benchmark-local redesign with fresh locked-clock reruns.

## 2026-03-22T23:45:00Z

- Additional progress in the `ch04` follow-up:
- `ch04:tensor_parallel_async` is now a verified manual rerun success on the current host after adding a one-rank optimized fast path that preserves exact math for the degenerate single-rank case.
- The March 21 queue entry for `ch04:tensor_parallel_async` has been removed and the rerun state now records it as `classification=manual_rerun_success`.
- `ch04:pipeline_parallel` / `ch04:pipeline_parallel_1f1b` remain the open decision point:
  - honest single-stage specializations were attempted
  - repeated locked-clock reruns still leave them at parity or slight slowdown
  - removing or de-listing them was not done, per user instruction
- If more work is requested here, the next sensible move is to design a stronger single-GPU stage-virtualization benchmark rather than keep iterating on the current one-stage stand-in.

## 2026-03-23T00:15:00Z

- The deeper `ch04` redesign has now landed and verified on the local virtualized `b200` host.
- Final local outcomes for the original single-GPU non-speedup set:
  - `ch04:pipeline_parallel` -> `1.0165x`
  - `ch04:pipeline_parallel_1f1b` -> `1.0479x`
  - `ch04:tensor_parallel_async` -> `1.0006x`
- Important semantic improvement:
  - the single-rank `ch04` pipeline baseline no longer relies on fake CPU round-trips to simulate inter-stage communication
  - the one-rank story is now a GPU-only virtual-stage benchmark comparing naive cloned handoffs vs reusable-buffer 1F1B scheduling
- The March 21 rerun queue is synchronized for these three targets and now reports `queued_problem_total=155`.
- Carry-forward note:
- the current evidence is still portable/non-canonical because it was gathered on a virtualized single-GPU host with locked clocks, not bare metal

## 2026-03-23T00:30:00Z

- Superseding note for `ch04`:
- do not quote the earlier one-rank “wins” as evidence for pipeline/tensor-parallel behavior
- the user explicitly chose the stricter contract: on a 1-GPU host these benchmarks should skip as expected unsupported, not emulate multi-GPU performance locally
- the current code now reflects that choice:
  - `ch04/baseline_pipeline_parallel.py`
  - `ch04/optimized_pipeline_parallel_1f1b.py`
  - `ch04/baseline_tensor_parallel.py`
  - `ch04/optimized_tensor_parallel_async.py`
  - all require `>=2` GPUs and advertise `multi_gpu_required=True`
- important constraint:
  - the old CPU round-trip pipeline baseline was not restored
  - the benchmark story remains GPU-native, but the single-GPU publication path is now intentionally disabled
- current single-GPU truth on this host:
  - `ch04:pipeline_parallel` -> `expected_unsupported_portable_single_gpu`
  - `ch04:pipeline_parallel_1f1b` -> `expected_unsupported_portable_single_gpu`
  - `ch04:tensor_parallel` -> `expected_unsupported_portable_single_gpu`
  - `ch04:tensor_parallel_async` -> `expected_unsupported_portable_single_gpu`
- queue/state were updated to match, and the latest queue snapshot is:
  - `queued_problem_total=154`
  - `expected_unsupported_total=65`
  - `written_expectation_total=218`

## 2026-03-23T04:33:00Z

- Resolved in this loop:
- `ch10:persistent_matmul_tma` is no longer a local non-speedup on the portable `b200` host
- benchmark-local TMA tuning that preserved math and tile shape was sufficient:
  - `GROUP_M=2`
  - `NUM_WARPS=4`
  - `NUM_STAGES=5`
- fresh rerun evidence:
  - baseline `0.2784 ms`
  - optimized `0.2501 ms`
  - `1.1133x`
- queue state was reconciled to `classification=manual_rerun_success`, and the stale `problem_queue.jsonl` entry for `ch10:persistent_matmul_tma` was removed

- Unresolved after explicit reproduction:
- `ch13:torchao_quantization` remains a real local non-speedup on `b200`
- clean harness rerun:
  - baseline `1.0389 ms`
  - optimized int8 dynamic torchao `5.0256 ms`
  - `0.2067x`
- direct local repro with TF32 enabled confirms the benchmark result is directionally real on this host:
  - FP32/TF32 baseline `~0.97 ms`
  - torchao int8 dynamic `~4.77 ms`
  - torchao float8 dynamic `~1.49 ms`
- implication:
  - this is not a stale queue record and not a harness-isolation defect
  - any future fix for `ch13:torchao_quantization` needs either a benchmark-local semantics change that remains honest for the chapter story, or a later discussion about classification; do not “fix” it by folding in `torch.compile` because `torchao_quantization_compiled` is already intentionally informational

## 2026-03-23T05:10:00Z

- New local win cleared from the portable queue:
- `ch11:stream_ordered_kv_cache`
  - changed `ch11/optimized_stream_ordered_kv_cache.py` from `num_streams=2` to `num_streams=3`
  - kept `num_segments=8` and the same chunked workload/order semantics
  - fresh harness rerun on the locked-clock portable `b200` host:
    - baseline `3.1422 ms`
    - optimized `2.0206 ms`
    - `1.5551x`
  - queue/state now mark it as `manual_rerun_success`

- `ch12:cuda_graphs_conditional` status:
  - confirmed again as `informational_noncanonical`
  - no code change was needed in the benchmark pair itself; the important part is that it should not return to the canonical expectation queue

- Most important carry-forward decision point:
- the next out-of-tolerance chapter cases look increasingly like workload-shape issues, not obvious benchmark-local code defects
- directional evidence gathered locally before changing anything:
  - `ch10:attention`
    - current `seq_len=1024` with Flash-only SDPA is about `3.02x`
    - `seq_len=1280-2048` moves the same story into roughly `5.49-5.65x`
  - `ch13:precisionmixed`
    - current `(hidden_dim=2048, batch_size=512)` is about `3.97x`
    - `(3072, 512)` and `(4096, 512)` move the same story into roughly `7.0x+`
- implication for the next pass:
  - if the user approves workload retunes, `ch10:attention` and `ch13:precisionmixed` are strong candidates
  - if workload retunes are not approved, the next best work is further diagnosis/documentation, not more blind code churn

## 2026-03-23T05:46:00Z

- Approved retunes that are now complete:
- `ch10:attention`
  - `seq_len` increased from `1024` to `1280` in both baseline and optimized
  - real harness rerun on the locked-clock portable `b200` host:
    - baseline `3.4380 ms`
    - optimized `0.6216 ms`
    - `5.5311x`
  - queue/state reconciled to `manual_rerun_success`
- `ch13:precisionmixed`
  - `hidden_dim` increased from `2048` to `3072` in both baseline and optimized
  - real harness rerun on the locked-clock portable `b200` host:
    - baseline `77.7916 ms`
    - optimized `11.0806 ms`
    - `7.0205x`
  - queue/state reconciled to `manual_rerun_success`

- Important remaining local decision point:
- the compile-scope drift cases do not currently look like clean local fixes
- using the actual benchmark classes with overridden shapes:
- `ch13:regional_compile` only screened up to about `1.0766x`
- `ch14:regional_triton` screened roughly in the `1.07x-1.15x` band
- implication:
- these no longer look like simple shape-retune wins
- any next change there should probably be a benchmark-contract discussion (for example whether the measurement should include compile/shape-switch cost) rather than another blind performance-tuning pass

## 2026-03-23T13:35:00Z

- `ch12:cuda_graphs_conditional_enhanced` contract cleanup is complete.
  - The rerun helper bug was that CUDA examples were looked up by raw example name instead of typed expectation key.
  - `scripts/full_virtualized_rerun.py` now uses `<example>_cuda` for CUDA targets.
  - Regression coverage was added.
  - The main queue now treats `ch12:cuda_graphs_conditional_enhanced` as expectation-bearing and clean on this host.

- `ch14:regional_triton` has a real benchmark-local improvement, but not a stable local resolution.
  - Added explicit bucket warmup in `setup()` for the optimized path.
  - Repeated locked-clock reruns on the same virtualized `b200` host now show:
    - `1.6741x`
    - `1.2509x`
    - `0.8417x` on the main queue rerun
  - Carry-forward interpretation:
    - keep the warmup fix
    - do not claim the pair is resolved on this host
    - treat it as unstable on virtualized hardware until a canonical rerun or deeper benchmark-contract decision clarifies it

- Queue carry-forward:
  - main queue is now `queued_problem_total=148`, `expected_unsupported_total=65`, `written_expectation_total=219`
  - stale JSONL lines for the false `ch12` miss and the superseded older `ch14` run were removed
  - the newest `ch14:regional_triton` queue entry remains intentionally

- Best next local target after this handoff:
- `ch13:torchao_quantization`
- a broader pure-quantization sweep is running / should be continued before any benchmark rewrite decision

- Follow-up evidence on `ch13:torchao_quantization`:
  - broader pure-quantization sweeps on this B200 host did not find a stable steady-state winner for BF16-vs-quantized execution
  - some FP32-at-`batch=8192` runs looked superficially positive, but they conflict with the repeated harness behavior and should be treated as cold-start / first-case noise until reproduced under the actual benchmark contract
  - practical implication:
    - do not patch `optimized_torchao_quantization.py` to a new pure-quantization config yet
    - the next useful step is a benchmark-contract discussion or deeper measurement audit, not another blind config swap
