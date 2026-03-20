2026-03-20T04:09:00Z

Milestones
1. Study `labs/block_scaling`, `labs/blackwell_matmul`, and `labs/custom_vs_cublas` to copy the repo’s lab structure.
2. Reduce the deck to a small kernel set with a single benchmark pair plus a standalone milestone runner.
3. Implement a self-contained CUDA extension for copy and transpose bandwidth kernels.
4. Add harness-discoverable baseline/optimized modules for the transpose pair.
5. Add a README and validate both the standalone runner and the repo harness path.

Decisions
- Main pair is `transpose_naive` vs `transpose_tiled`.
- Vectorization is measured on the contiguous copy roofline path, not hidden inside the pair.
- The advanced async milestone is `copy_async_double_buffered`. It is measurable but not assumed to be the winner.
- No shared-file edits were made; any repo-wide README or registry follow-up is out of scope for this task.
