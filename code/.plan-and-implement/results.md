## 2026-03-20T14:39:00Z

- Completed:
- Collected top-level inventory for `ch01`-`ch20` and discoverable `labs/*` benchmark surfaces.
- Confirmed benchmark methodology constraints from `docs/benchmark_methodology.md`.
- Confirmed `book-after` structure and the chapter/fix packet mapping from `book-after/README.md`.
- Added a `scope_contract` step to `core/scripts/run_benchmark_pair_audit.py` so each audited scope records:
- source-of-truth doc path
- chapter-vs-lab type
- lab review mode from `labs/README.md`
- contract findings for missing docs, missing pair targets, or ambiguous non-pair lab docs
- Extended audit summary/manifest output to include `scope_contract` results.
- Fixed the flaky timeout regression in `tests/test_validate_benchmark_pairs_tools.py` so the audit workflow's nested pytest bundle passes reliably on this host.
- Pending verification:
- Full repo-wide audit not run yet.
- Chapter batch review not started yet.
- Remaining risks:
- Chapter manuscripts describe concepts more often than exact file names, so a fully trustworthy audit will require a mix of automation and manual review.
- Verification evidence:
- `python -m pytest tests/test_run_benchmark_pair_audit.py tests/test_review_pair_scanner.py -q` -> `26 passed`
- `python -m pytest tests/test_validate_benchmark_pairs_tools.py::test_validate_all_pairs_timeout_kills_isolated_pair_descendants -q` -> `1 passed`
- `python -m pytest tests/test_review_pair_scanner.py tests/test_validate_benchmark_pairs_tools.py tests/test_benchmark_hygiene_regressions.py tests/test_review_findings_regressions.py tests/test_run_benchmark_pair_audit.py -q` -> `81 passed`
- `python -m core.scripts.run_benchmark_pair_audit --scope labs/block_scaling --output-dir artifacts/audits/20260320_block_scaling_scope_contract_smoke2` -> completed with all steps green, including `scope_contract` and nested `pytest_audit`

## 2026-03-20T16:33:50Z

- Completed since the initial bootstrap:
- Implemented the immediate metadata/comment/tolerance fixes for `ch02`, `ch10`, `ch12`, `ch14`, `ch20`, and `labs/cache_aware_disagg_inference`.
- Removed additional stale optimization comments surfaced by the broader review in `ch05`, `ch11`, `ch13`, and `ch18`.
- Added static input-signature publication to the `ch12` kernel-fusion family, `ch13:bandwidth_naive`, and `ch18:vllm_v1_integration` so pair validation no longer times out on those comparisons.
- Classified `labs/cache_aware_disagg_inference` as a `benchmark-story` lab in `labs/README.md` so the audit treats it honestly.
- Repo-wide static pair review now reports zero findings.
- Scope audits completed for the touched and timeout-prone chapter set; remaining work is now the untouched chapter/lab sweep rather than cleanup of the already-fixed scopes.
- Verification evidence added:
- `python -m pytest tests/test_benchmark_hygiene_regressions.py tests/test_run_benchmark_pair_audit.py -q` -> `37 passed, 7 warnings`
- `python -m pytest tests/test_review_pair_scanner.py tests/test_validate_benchmark_pairs_tools.py tests/test_benchmark_hygiene_regressions.py tests/test_review_findings_regressions.py tests/test_run_benchmark_pair_audit.py -q` -> `84 passed, 7 warnings`
- `python -m core.verification.review_baseline_optimized_pairs --json --markdown --output-dir artifacts/reviews/20260320_repo_static_review` -> `457 pairs reviewed, 0 findings`
- `python -m core.scripts.run_benchmark_pair_audit --scope ch05 --scope ch11 --scope ch12 --scope ch13 --scope ch18 --scope labs/cache_aware_disagg_inference --output-dir artifacts/audits/20260320_validity_batch2` -> only expected multi-GPU skips remained
- `python -m core.scripts.run_benchmark_pair_audit --scope ch13 --scope ch18 --output-dir artifacts/audits/20260320_validity_batch3` -> `41 valid pairs, 0 error pairs`
- `python -m cli.aisp bench run --targets ch02:cublas --profile minimal --format json --suite-timeout 0 --run-id 20260320_dogfood_ch02_cublas` -> strict/input verification passed; `6.29x` speedup on this virtualized host
- `python -m cli.aisp bench run --targets ch10:pipeline_3stage --profile minimal --format json --suite-timeout 0 --run-id 20260320_dogfood_ch10_pipeline_3stage` -> strict verification passed; corrected workload-byte accounting exercised end to end
- `python -m cli.aisp bench run --targets labs/cache_aware_disagg_inference --profile minimal --format json --suite-timeout 0 --run-id 20260320_dogfood_cache_aware_disagg` -> single-GPU path passed with `(1e-2, 1e-2)` tolerance; multi-GPU path skipped correctly on a 1-GPU host

## 2026-03-20T16:42:31Z

- Completed in the next untouched-scope pass:
- `ch16` full audit completed cleanly, then rerun cleanly after a small manual cleanup patch.
- `ch01` full audit completed cleanly.
- `ch09` full audit completed cleanly.
- Manual `ch16` cleanup applied:
- removed a stale `torch.compile` claim from `optimized_flash_sdp.py`
- corrected the stale chapter docstring in `ch16/compare.py`
- corrected the baseline story text in `baseline_regional_compilation.py`
- fixed `baseline_piece_graphs.validate_result()` to check the initialized verification input instead of a nonexistent `self.inputs`
- Verification evidence added:
- `python -m py_compile ch16/optimized_flash_sdp.py ch16/baseline_piece_graphs.py ch16/baseline_regional_compilation.py ch16/compare.py` -> succeeded
- `python -m core.scripts.run_benchmark_pair_audit --scope ch16 --output-dir artifacts/audits/20260320_ch16_audit_rerun` -> completed cleanly
- `python - <<'PY' ... from ch16.baseline_piece_graphs import get_benchmark ... validate_result ... PY` -> `validate_result None`
- `python -m core.scripts.run_benchmark_pair_audit --scope ch01 --output-dir artifacts/audits/20260320_ch01_audit` -> completed cleanly
- `python -m core.scripts.run_benchmark_pair_audit --scope ch09 --output-dir artifacts/audits/20260320_ch09_audit` -> completed cleanly
- In progress:
- `python -m core.scripts.run_benchmark_pair_audit --scope ch03 --output-dir artifacts/audits/20260320_ch03_audit`
- `python -m core.scripts.run_benchmark_pair_audit --scope ch04 --output-dir artifacts/audits/20260320_ch04_audit`
- Current partial status for the in-progress scopes: review findings `0`, compliance errors `0`, warnings `0`.

## 2026-03-20T16:44:23Z

- Additional completed evidence:
- `python -m core.scripts.run_benchmark_pair_audit --scope ch03 --output-dir artifacts/audits/20260320_ch03_audit` -> completed cleanly
- `ch03` summary: review findings `0`, scope-contract findings `0`, report-drift findings `0`, pair validation ran successfully
- Additional partial evidence:
- `ch04` review stage -> `52` pairs reviewed, `0` findings
- `ch04` compliance stage -> `98` files checked, `0` errors, `0` warnings
- Still running:
- `python -m core.scripts.run_benchmark_pair_audit --scope ch04 --output-dir artifacts/audits/20260320_ch04_audit`

## 2026-03-20T17:42:30Z

- Completed since the prior checkpoint:
- Remaining chapter scopes closed cleanly:
- `python -m core.scripts.run_benchmark_pair_audit --scope ch04 --output-dir artifacts/audits/20260320_ch04_audit` -> completed cleanly; only expected multi-GPU skips remained.
- `python -m core.scripts.run_benchmark_pair_audit --scope ch06 --scope ch07 --scope ch08 --output-dir artifacts/audits/20260320_ch06_ch07_ch08_audit` -> completed cleanly.
- `python -m core.scripts.run_benchmark_pair_audit --scope ch15 --scope ch17 --scope ch19 --output-dir artifacts/audits/20260320_ch15_ch17_ch19_audit` -> completed cleanly.
- Remaining lab batches closed cleanly:
- `python -m core.scripts.run_benchmark_pair_audit --scope labs/cutlass_profiler_kernel_selector --scope labs/moe_decode_blackwell_matrix --scope labs/moe_parallelism --scope labs/nanochat_fullstack --scope labs/python_concurrency --scope labs/tcgen05_cluster_shapes --scope labs/uma_memory --scope labs/vllm-deepseek-tuning --output-dir artifacts/audits/20260320_labs_playbook_audit_rerun` -> completed cleanly.
- `python -m core.scripts.run_benchmark_pair_audit --scope labs/cache_aware_disagg_inference --scope labs/decode_optimization --scope labs/dynamic_router --scope labs/fullstack_cluster --scope labs/kv_cache_compression --scope labs/kv_optimization --scope labs/moe_cuda --scope labs/moe_cuda_ptx --scope labs/moe_optimization_journey --scope labs/nanochat_fullstack --scope labs/ozaki_scheme --scope labs/real_world_models --scope labs/speculative_decode --scope labs/train_distributed --scope labs/trtllm_phi_3_5_moe --output-dir artifacts/audits/20260320_labs_benchmark_story_audit_rerun` -> completed cleanly; summary shows `83` valid pairs, `3` expected skips, `0` errors.
- `python -m core.scripts.run_benchmark_pair_audit --scope labs/async_input_pipeline --scope labs/blackwell_gemm_optimizations --scope labs/blackwell_matmul --scope labs/block_scaling --scope labs/cudnn_sdpa_bench --scope labs/custom_vs_cublas --scope labs/flashattention4 --scope labs/flashattention_gluon --scope labs/flashinfer_attention --scope labs/flexattention --scope labs/memory_bandwidth_patterns --scope labs/nccl_nixl_nvshmem --scope labs/nvfp4_dual_gemm --scope labs/nvfp4_gemm --scope labs/nvfp4_gemv --scope labs/nvfp4_group_gemm --scope labs/occupancy_tuning --scope labs/parameterized_cuda_graphs --scope labs/persistent_decode --scope labs/recsys_sequence_ranking --scope labs/top_k_kernel --scope labs/training_hotpath --output-dir artifacts/audits/20260320_labs_benchmark_pair_audit_rerun` -> completed cleanly; summary shows `54` valid pairs, `4` expected `labs/flashattention4` windowed skips, `0` errors.
- Persistent-decode signature normalization completed after the broad batch:
- `python -m py_compile labs/persistent_decode/persistent_decode_common.py labs/persistent_decode/baseline_persistent_decode.py labs/persistent_decode/optimized_persistent_decode_graphs.py labs/persistent_decode/optimized_persistent_decode_triton.py labs/persistent_decode/optimized_persistent_decode_cuda.py` -> succeeded.
- `python - <<'PY' ... baseline/graphs/triton/cuda get_benchmark().get_input_signature() ... PY` -> all four variants now publish identical workload-shaped signatures (`q`, `k`, `v`, `output`; batch `8`; quantization `fp32` in the default path).
- `python -m core.scripts.run_benchmark_pair_audit --scope labs/persistent_decode --output-dir artifacts/audits/20260320_labs_persistent_decode_audit_after_signature_v2` -> completed cleanly; `10` valid pairs, `0` mismatches, `0` errors.
- Runtime dogfood evidence added:
- `python -m cli.aisp bench run --targets labs/nanochat_fullstack:nanochat_inference --profile minimal --format json --suite-timeout 0 --run-id 20260320_dogfood_nanochat_signature` -> completed successfully; verification passed; profiles succeeded; optimized path reported `2.309x`.
- `python -m cli.aisp bench run --targets labs/persistent_decode:persistent_decode_full_and_piecewise --profile minimal --format json --suite-timeout 0 --run-id 20260320_dogfood_persistent_decode_signature` -> completed successfully; verification passed; profiles succeeded; optimized path reported `0.0898 ms`, `13.42x`, with expectation refresh correctly rejected on provenance mismatch rather than silently updating.
- Repo-wide hard gate rerun after the latest lab patches:
- `python -m pytest tests/test_review_pair_scanner.py tests/test_validate_benchmark_pairs_tools.py tests/test_benchmark_hygiene_regressions.py tests/test_review_findings_regressions.py tests/test_run_benchmark_pair_audit.py -q` -> `84 passed, 7 warnings`.
- Lab classification coverage check:
- `python - <<'PY' ... compare labs/README.md classified paths vs actual labs/ directories ... PY` -> `44` actual lab directories classified, `0` missing, `0` extra.

## 2026-03-20T18:00:42Z

- Completed targeted follow-up on the post-sweep questions:
- reviewed the other session's claimed metric/helper edits against current helper semantics and live harness behavior
- kept the broad workload-intent cleanup where it matched the code
- adjusted the places where the edits would otherwise leave misleading or unsupported metadata behind
- Concrete code changes from this pass:
- `core/benchmark/metrics.py`
  - `compute_precision_metrics()` now explicitly supports `precision_type=\"fp32\"` and `precision_type=\"int8\"`
  - the helper docs now describe `precision_type` as a path-level precision mode with memory-reduction factors interpreted relative to FP32
- `ch14/baseline_cutlass.py`
- `ch14/optimized_cutlass.py`
  - corrected `compute_gemm_metrics()` inputs from FP32/4-byte to FP16/2-byte so the custom metrics match the actual benchmark tensors
- `ch13/optimized_quantization.py`
- `ch13/optimized_torchao_quantization.py`
- `labs/speculative_decode/baseline_speculative_decode.py`
- `labs/speculative_decode/optimized_speculative_decode.py`
  - removed unsupported `precision_flags` keys (`int8`, `fp32`) so the payloads stay within the current verification schema instead of relying on silently ignored fields
- `tests/test_benchmark_metrics.py`
  - added coverage for `compute_precision_metrics(... precision_type=\"int8\")`
  - added coverage for neutral FP32 precision reporting
- Verification evidence added:
- `python -m py_compile core/benchmark/metrics.py ch14/baseline_cutlass.py ch14/optimized_cutlass.py ch13/optimized_quantization.py ch13/optimized_torchao_quantization.py labs/speculative_decode/baseline_speculative_decode.py labs/speculative_decode/optimized_speculative_decode.py tests/test_benchmark_metrics.py` -> succeeded
- `python -m pytest tests/test_benchmark_metrics.py -q` -> `45 passed`
- `python -m core.scripts.run_benchmark_pair_audit --scope ch14 --scope labs/speculative_decode --output-dir artifacts/audits/20260320_ch14_specdecode_postreview` -> review findings `0`; pair validation stdout shows all `10` `ch14` pairs and `labs/speculative_decode:speculative_decode` validated successfully
- `python -m core.scripts.run_benchmark_pair_audit --scope labs/flashattention4 --output-dir artifacts/audits/20260320_flashattention4_recheck` -> review findings `0`, compliance errors `0`, warnings `0`
- `python -m cli.aisp bench run --targets ch14:cutlass --profile minimal --format json --suite-timeout 0 --run-id 20260320_dogfood_ch14_cutlass_postreview_clean` -> succeeded under strict validity on this virtualized host; baseline `1.361 ms`, optimized `0.529 ms`, `2.575x`; verification passed; `nsys`, `ncu`, and `torch` profiling all succeeded; custom metrics now report `gemm.bytes_per_element=2.0` and FP16 roofline inputs
- `python -m cli.aisp bench run --targets labs/speculative_decode:speculative_decode --profile minimal --format json --suite-timeout 0 --run-id 20260320_dogfood_speculative_decode_postreview_clean` -> succeeded under strict validity on this virtualized host; baseline `102.057 ms`, optimized `27.316 ms`, `3.736x`; verification passed with exact-match output tolerance; `nsys`, `ncu`, and `torch` profiling all succeeded
- `python -m cli.aisp bench run --targets labs/flashattention4:flashattention4_windowed --profile minimal --format json --suite-timeout 0 --run-id 20260320_repro_flashattention4_windowed_clean` -> baseline succeeded under strict validity; optimized path skipped with the benchmark-local message `experimental sliding-window FlashAttention-4 kernels are unstable on this torch 2.9.1 + sm_100 stack (all candidate providers produced non-finite outputs)`
- Remaining risks:
- none from benchmark-validity semantics in this follow-up pass
- the `labs/flashattention4` windowed optimized path still needs provider/kernel-stack remediation if the goal is to enable that benchmark rather than classify it honestly
