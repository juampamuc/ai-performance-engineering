## 2026-03-17T21:21:34+00:00

- No implementation results yet.
- Planned validation:
  - targeted pytest coverage for the new lab
  - `python -m compileall` or `python -m py_compile` on new files
  - `python -m cli.aisp bench list-targets --chapter labs/cache_aware_disagg_inference`
  - `python -m cli.aisp bench run --targets labs/cache_aware_disagg_inference:<target> ...`

## 2026-03-17T21:33:45+00:00

- Files changed:
  - `labs/cache_aware_disagg_inference/__init__.py`
  - `labs/cache_aware_disagg_inference/cache_aware_disagg_common.py`
  - `labs/cache_aware_disagg_inference/baseline_cache_aware_disagg.py`
  - `labs/cache_aware_disagg_inference/optimized_cache_aware_disagg.py`
  - `labs/cache_aware_disagg_inference/README.md`
  - `tests/test_cache_aware_disagg_lab.py`
  - `core/scripts/refresh_readmes.py`
  - `labs/README.md`

- Validation commands and outcomes:
  - `python -m py_compile labs/cache_aware_disagg_inference/cache_aware_disagg_common.py labs/cache_aware_disagg_inference/baseline_cache_aware_disagg.py labs/cache_aware_disagg_inference/optimized_cache_aware_disagg.py tests/test_cache_aware_disagg_lab.py core/scripts/refresh_readmes.py`
    - passed
  - `pytest -q tests/test_cache_aware_disagg_lab.py -q`
    - passed (`...`)
  - `python core/scripts/refresh_readmes.py --check --target labs/cache_aware_disagg_inference --target labs/README.md`
    - passed (`All 2 README target(s) are in sync.`)
  - `python -m cli.aisp bench list-targets --chapter labs/cache_aware_disagg_inference`
    - passed (`labs/cache_aware_disagg_inference:cache_aware_disagg`)
  - `python -m labs.cache_aware_disagg_inference.optimized_cache_aware_disagg --requests-per-iteration 4 --context-window 384 --chunk-size 96 --decode-tokens 24 --hidden-size 128 --num-layers 2 --batch-size 1 --logical-decode-workers 3`
    - passed; emitted `cache_aware.cache_hit_rate=0.8889`, `cache_aware.kv_transfer_mb=0.098304`, `validation_error=null`
  - `python -m cli.aisp bench run --targets labs/cache_aware_disagg_inference:cache_aware_disagg --profile none --iterations 1 --warmup 1 --single-gpu --validity-profile portable --allow-foreign-gpu-processes --suite-timeout 300 --target-extra-arg 'labs/cache_aware_disagg_inference:cache_aware_disagg=--requests-per-iteration 4 --context-window 384 --chunk-size 96 --decode-tokens 24 --hidden-size 128 --num-layers 2 --batch-size 1 --logical-decode-workers 3'`
    - passed on the local B200 host
    - baseline: `164.09 ms`
    - optimized: `149.91 ms`
    - speedup: `1.09x`
    - baseline locality: `cache_hit_rate=0.1667`, `kv_transfer_mb=368.050176`, `worker_switches_per_request=4.5`
    - optimized locality: `cache_hit_rate=0.9167`, `kv_transfer_mb=28.311552`, `worker_switches_per_request=0.0`
    - artifacts: `artifacts/runs/20260317_213255__bench__profile_none_targets_labs_cache_aware_disagg_inference_cache_aware_disagg/`

- Remaining risks:
  - The lab is currently a single-GPU logical-worker reproduction, not a full multi-node serving implementation.
  - Harness dogfood on this host required `--allow-foreign-gpu-processes` because other CUDA Python processes were already resident on the shared GPU.
