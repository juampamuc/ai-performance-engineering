## 2026-03-17T21:21:34+00:00

1. Create a new lab package with shared benchmark logic plus baseline/optimized wrappers.
2. Model the article's scheduler contrast:
   - baseline: round-robin cache-unaware chunk/decode placement
   - optimized: cache-affine placement with shared KV hierarchy and warm/cold mix
3. Add README and refresh-readme metadata so the lab is documented and discoverable.
4. Add targeted tests for wrapper/discovery behavior and emitted custom metrics.
5. Validate with syntax/tests plus at least one CLI benchmark invocation.

## 2026-03-17T21:33:45+00:00

- Completed milestone 1: added the new lab package, shared benchmark logic, wrappers, and generator-owned README.
- Completed milestone 2: baseline/optimized mapping now matches the article's round-robin vs cache-affine scheduler story.
- Completed milestone 3: added `tests/test_cache_aware_disagg_lab.py`.
- Completed milestone 4: validated module and harness entrypoints on the local GPU host.
