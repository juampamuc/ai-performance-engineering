#!/usr/bin/env python3
"""Patch and verify the Chapter 10 sample fixes in the book-after PDF.

This targets the known high-risk Chapter 10 manuscript rows that were fixed in the
book-after packet but can still drift in the rendered PDF:

- CH10-01: unified CTA pipeline legality wording
- CH10-02/03: warp-specialized pipeline listing + handoff explanation
- CH10-04/08: cluster pipeline listing legality
- CH10-05/06/07/09: surrounding narrative consistency

Usage:
    python -m core.scripts.fix_ch10_book_after_pdf --check
    python -m core.scripts.fix_ch10_book_after_pdf --patch --output /tmp/fixed.pdf
    python -m core.scripts.fix_ch10_book_after_pdf --patch --in-place
"""
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "PyMuPDF is required. Install it in a venv first, for example:\n"
        "  python -m venv /tmp/aisp-pdf-verify\n"
        "  /tmp/aisp-pdf-verify/bin/pip install pymupdf"
    ) from exc


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF = ROOT / "book-after" / "AI_Systems_Performance_Engineering_Mar_2026_updated.pdf"


@dataclass(frozen=True)
class TextPatch:
    page: int
    rect: tuple[float, float, float, float]
    text: str
    fontname: str
    max_fontsize: float
    min_fontsize: float = 6.3


@dataclass(frozen=True)
class PhraseCheck:
    page: int
    phrase: str
    should_exist: bool
    label: str


PATCHES: tuple[TextPatch, ...] = (
    TextPatch(
        page=407,
        rect=(68, 458, 438, 587),
        fontname="helv",
        max_fontsize=9.0,
        text="""Concurrently, other warps in the block invoke pipe.consumer_wait(). This stalls only
threads dependent on the stage indexed by s = tile % STAGES until that stage is
committed by the producer.

After finishing the computation on the current stage, the warps invoke
pipe.consumer_release() to free that stage for reuse in subsequent iterations.

At the end of each iteration, the ring buffer advances by stage index; the next
prefetch uses nextTile = tile + STAGES and writes into the same slot s modulo
STAGES. This keeps the narrative consistent with the code's s / nextTile naming
and logic.""",
    ),
    TextPatch(
        page=413,
        rect=(68, 170, 438, 547),
        fontname="helv",
        max_fontsize=8.9,
        text="""Using CUDA Pipeline API for Warp Specialization

Warp specialization builds on the CUDA Pipeline API by allowing specialized warps
to communicate using fine-grained producer and consumer primitives. These calls
avoid full block barriers while composing naturally with asynchronous copies such as
cuda::memcpy_async.

For the unified block-scoped pipeline shown here (cuda::make_pipeline(cta,
&pipe_state)), the producer and consumer calls (pipe.producer_acquire(),
pipe.producer_commit(), pipe.consumer_wait(), and pipe.consumer_release()) are
collective across the participating CTA threads and must execute in a consistent order.

A block-wide barrier would stall every warp, even those that are not involved with the
producer-consumer pipeline. All execution in that block must pause until every thread
reaches the barrier, as shown in Figure 10-7.

By comparison, the Pipeline API maintains per-stage state internally while still
requiring ordered collective participation at each acquire / commit / wait / release point
in this unified example. Warp specialization remains valid because only role-specific
warps perform role-specific work between those collective calls, with explicit handoff
synchronization points. If you want different warps to call different pipeline primitives
directly, use a partitioned pipeline with explicit producer / consumer roles instead of a
unified CTA pipeline.

You can implement warp specialization with the CUDA Pipeline API in a three-role
pattern. A loader warp produces inputs for a compute warp, the compute warp
consumes those inputs and produces results, and a storer warp consumes those results
and writes them out. The pipeline object is block scoped, and it tracks the stage order
internally.""",
    ),
    TextPatch(
        page=414,
        rect=(68, 356, 438, 587),
        fontname="cour",
        max_fontsize=7.4,
        text="""// warp_specialized_pipeline.cu
// Persistent kernel with warp-specialized roles using the CUDA Pipeline API
#include <cuda/pipeline>
#include <cooperative_groups.h>
using namespace cooperative_groups;

// smem_bytes = 3 * TILE_SIZE^2 * sizeof(float)
// 3 tiles * 112 * 112 * 4 bytes per float = 150,528 bytes < 227,328 bytes
// per-block dynamic SMEM limit on Blackwell
#define TILE_SIZE 112

__device__ void compute_full_tile(const float* __restrict__ A_tile,
                                  const float* __restrict__ B_tile,
                                  float* __restrict__ C_tile,
                                  int lane_id) {
    for (int idx = lane_id; idx < TILE_SIZE * TILE_SIZE; idx += warpSize) {
        int row = idx / TILE_SIZE;
        int col = idx % TILE_SIZE;
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += A_tile[row * TILE_SIZE + k] * B_tile[k * TILE_SIZE + col];
        }
        C_tile[idx] = acc;
    }
}""",
    ),
    TextPatch(
        page=415,
        rect=(68, 50, 438, 587),
        fontname="cour",
        max_fontsize=7.3,
        text="""extern "C"
__global__ void warp_specialized_pipeline_kernel(
    const float* __restrict__ A_global,
    const float* __restrict__ B_global,
    float* __restrict__ C_global,
    int numTiles) {
    thread_block cta = this_thread_block();

    extern __shared__ float shared_mem[];
    float* A_tile = shared_mem;
    float* B_tile = A_tile + TILE_SIZE * TILE_SIZE;
    float* C_tile = B_tile + TILE_SIZE * TILE_SIZE;

    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    const int bytes = TILE_SIZE * TILE_SIZE * sizeof(float);

    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 3>
        pipe_state;
    auto pipe = cuda::make_pipeline(cta, &pipe_state);

    for (int tile = blockIdx.x; tile < numTiles; tile += gridDim.x) {
        size_t offset = static_cast<size_t>(tile) * TILE_SIZE * TILE_SIZE;

        pipe.producer_acquire();
        cuda::memcpy_async(cta, A_tile, A_global + offset,
                           cuda::aligned_size_t<16>{bytes}, pipe);
        cuda::memcpy_async(cta, B_tile, B_global + offset,
                           cuda::aligned_size_t<16>{bytes}, pipe);
        pipe.producer_commit();

        pipe.consumer_wait();
        cta.sync();""",
    ),
    TextPatch(
        page=416,
        rect=(68, 60, 438, 547),
        fontname="cour",
        max_fontsize=7.4,
        text="""        if (warp_id == 1) {
            compute_full_tile(A_tile, B_tile, C_tile, lane_id);
        }
        cta.sync();

        if (warp_id == 2) {
            for (int idx = lane_id; idx < TILE_SIZE * TILE_SIZE; idx += warpSize) {
                C_global[offset + idx] = C_tile[idx];
            }
        }
        cta.sync();
        pipe.consumer_release();
        cta.sync();
    }
}

Here each warp works in a distinct role for compute and store, while the unified
block-scoped pipeline collectives execute uniformly before role-specific branches.
Each iteration calls producer_acquire(), performs cooperative staged copy under
pipeline collectives, then calls producer_commit(). After consumer_wait(),
cta.sync() enforces handoff order: load -> compute -> store. Another cta.sync()
plus consumer_release() recycles the stage.

Important correctness rule for this unified example: producer_acquire(),
producer_commit(), consumer_wait(), and consumer_release() must occur in a
consistent collective sequence across the participating CTA threads. Do not place
those collectives inside divergent warp-role branches unless you intentionally switch
to a partitioned pipeline API with explicit producer / consumer roles.

This kernel runs as a persistent kernel across many tiles to amortize launch
overhead. More on persistent kernels in a bit.""",
    ),
    TextPatch(
        page=417,
        rect=(68, 48, 438, 122),
        fontname="helv",
        max_fontsize=8.8,
        text="""In short, using the CUDA Pipeline API together with cooperative groups allows
fine-grained, SM-wide producer-consumer handoffs with minimal CTA barriers at
explicit handoff points. Table 10-3 compares three implementations: a naive tiled
kernel, a two-stage double-buffered GEMM using double_buffered_pipeline, and our
warp-specialized pipeline kernel warp_specialized_pipeline.""",
    ),
    TextPatch(
        page=417,
        rect=(68, 390, 438, 586),
        fontname="helv",
        max_fontsize=8.8,
        text="""Here, the double-buffered pipeline finishes GEMM in 20.5 ms, whereas the
warp-specialized version completes in 18.4 ms. In contrast, the warp-specialized
kernel keeps pipeline collectives legal by issuing them before branch divergence,
then uses explicit handoff points between loader, compute, and storer work.

This finer-grained structure limits synchronization to explicit CTA handoff points
and lets role-specific work overlap between those points. As a result, average SM
utilization rises from roughly 92% in the double-buffered design to about 96% in
the warp-specialized version, and warp-stall cycles drop substantially.

From a scalability standpoint, Nsight Compute shows that the naive tiling kernel
saturates after just two to three active warps per SM. This is because each tile load
must complete before any computation can start.""",
    ),
    TextPatch(
        page=418,
        rect=(68, 150, 438, 430),
        fontname="helv",
        max_fontsize=8.7,
        text="""As Table 10-3 shows, both the double-buffered and warp-specialized approaches
substantially outperform the naive tiled kernel. The double_buffer_pipeline halves
the runtime by overlapping tile loads and computation, while the
warp_specialized_pipeline adds another 10.2% speedup by avoiding unnecessary
block-wide producer / consumer participation and enforcing only role-specific waits.

Instruction counts drop from 1.7 billion in the naive version to 1.05 billion in the
two-stage pipeline, a 38% reduction, and further to about 1.00 billion in the
warp-specialized kernel for an additional 4.76% reduction.

L2 load throughput climbs from 80 GB/s in naive tiling to 155 GB/s in the two-stage
approach (+94%) and then to 165 GB/s in the warp-specialized kernel (+6.45%
versus two-stage). This is because the loader, compute, and storer phases are
handed off explicitly while shared-memory reuse keeps the pipeline fed without
redundant block-wide waits.

In practice, the two-stage double-buffered pipeline is ideal for uniformly tiled GEMM
workloads. The warp-specialized approach is better suited to irregular or deeper
pipelines when explicit load / compute / store role separation helps keep the SM busy.""",
    ),
    TextPatch(
        page=447,
        rect=(68, 214, 438, 587),
        fontname="cour",
        max_fontsize=7.3,
        text="""extern "C"
__global__ void warp_specialized_cluster_pipeline(
    const float* __restrict__ A_global,
    const float* __restrict__ B_global,
    float* __restrict__ C_global,
    int numTiles) {
    thread_block cta = this_thread_block();
    cluster_group cluster = this_cluster();

    extern __shared__ float shared_mem[];
    float* A_tile_local = shared_mem;
    float* B_tile_local = A_tile_local + TILE_ELEMS;
    float* C_tile_local = B_tile_local + TILE_ELEMS;

    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 1>
        pipe_state;
    auto pipe = cuda::make_pipeline(cta, &pipe_state);

    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    auto warp = tiled_partition<32>(cta);

    const int cluster_rank = cluster.block_rank();
    const dim3 cluster_dims = cluster.dim_blocks();
    const int blocks_in_cluster =
        cluster_dims.x * cluster_dims.y * cluster_dims.z;

    for (int tile = blockIdx.x / cluster_dims.x; tile < numTiles;
         tile += gridDim.x / cluster_dims.x) {
        const size_t offset = static_cast<size_t>(tile) * TILE_ELEMS;""",
    ),
    TextPatch(
        page=448,
        rect=(68, 50, 438, 587),
        fontname="cour",
        max_fontsize=7.1,
        text="""        // Leader block stages A and B once for the entire cluster.
        // For block-scoped pipeline collectives, the entire leader CTA participates.
        if (cluster_rank == 0) {
            pipe.producer_acquire();
            if (warp_id == 0) {
                cuda::memcpy_async(warp, A_tile_local, A_global + offset,
                                   TILE_ELEMS * sizeof(float), pipe);
                cuda::memcpy_async(warp, B_tile_local, B_global + offset,
                                   TILE_ELEMS * sizeof(float), pipe);
            }
            pipe.producer_commit();
            cuda::pipeline_consumer_wait_prior<1>(pipe);
            pipe.consumer_release();
        }

        cluster.sync();

        const float* A_src = cluster.map_shared_rank(A_tile_local, 0);
        const float* B_src = cluster.map_shared_rank(B_tile_local, 0);

        const int rows_per_block =
            (TILE_SIZE + blocks_in_cluster - 1) / blocks_in_cluster;
        const int row_begin = min(cluster_rank * rows_per_block, TILE_SIZE);
        const int row_end = min(row_begin + rows_per_block, TILE_SIZE);

        if (warp_id == 1) {
            compute_rows_from_ds(A_src, B_src, C_tile_local, row_begin, row_end,
                                 lane_id);
        }

        cta.sync();

        if (warp_id == 2) {
            for (int row = row_begin + lane_id; row < row_end; row += warpSize) {
                for (int col = 0; col < TILE_SIZE; ++col) {
                    C_global[offset + row * TILE_SIZE + col] =
                        C_tile_local[row * TILE_SIZE + col];
                }
            }
        }

        cluster.sync();
    }
    // dynamic shared memory size: 3 * TILE_ELEMS * sizeof(float)
}""",
    ),
)


CHECKS: tuple[PhraseCheck, ...] = (
    PhraseCheck(407, "stage indexed by s = tile % STAGES", True, "CH10-09 replacement present"),
    PhraseCheck(407, "curr buffer", False, "CH10-09 stale wording removed"),
    PhraseCheck(413, "For the unified block-scoped pipeline shown here", True, "CH10-01 corrected legality wording"),
    PhraseCheck(413, "synchronize only the specific warps or", False, "CH10-01 stale legality wording removed"),
    PhraseCheck(416, "Important correctness rule for this unified example", True, "CH10-03 correctness note present"),
    PhraseCheck(416, "The loader warp calls producer_acquire()", False, "CH10-03 stale handoff explanation removed"),
    PhraseCheck(417, "minimal CTA barriers at explicit handoff points", True, "CH10-05 corrected summary present"),
    PhraseCheck(417, "only stalling the consumer warp", False, "CH10-06 stale warp-stall claim removed"),
    PhraseCheck(418, "avoiding unnecessary\nblock-wide producer / consumer participation", True, "CH10-07 corrected table interpretation present"),
    PhraseCheck(447, "auto warp = tiled_partition<32>(cta);", True, "CH10-08 warp declaration present"),
    PhraseCheck(448, "Leader block stages A and B once for the entire cluster.", True, "CH10-04 cluster leader rewrite present"),
    PhraseCheck(448, "if (cluster_rank == 0 && warp_id == 0)", False, "CH10-04 stale divergent leader branch removed"),
)


def _page_text(doc: fitz.Document, page_number: int) -> str:
    return doc[page_number - 1].get_text()


def _replace_text(page: fitz.Page, spec: TextPatch) -> float:
    rect = fitz.Rect(*spec.rect)
    page.add_redact_annot(rect, fill=(1, 1, 1))
    page.apply_redactions(
        images=fitz.PDF_REDACT_IMAGE_NONE,
        graphics=fitz.PDF_REDACT_LINE_ART_NONE,
        text=fitz.PDF_REDACT_TEXT_REMOVE,
    )
    fontsize = spec.max_fontsize
    while fontsize >= spec.min_fontsize:
        remaining = page.insert_textbox(
            rect,
            spec.text,
            fontname=spec.fontname,
            fontsize=fontsize,
            color=(0, 0, 0),
            align=fitz.TEXT_ALIGN_LEFT,
        )
        if remaining >= 0:
            return fontsize
        page.draw_rect(rect, color=None, fill=(1, 1, 1), overlay=True)
        fontsize -= 0.2
    raise RuntimeError(f"Patch on page {spec.page} overflowed its target rectangle")


def patch_pdf(input_pdf: Path, output_pdf: Path, write_pages_text: Path | None = None) -> Path:
    input_pdf = input_pdf.resolve()
    output_pdf = output_pdf.resolve()
    temp_output = output_pdf.with_suffix(output_pdf.suffix + ".tmp")

    with fitz.open(input_pdf) as doc:
        for spec in PATCHES:
            _replace_text(doc[spec.page - 1], spec)
        doc.save(temp_output, garbage=4, deflate=True)

    if output_pdf == input_pdf:
        backup = input_pdf.with_name(f"{input_pdf.stem}_before_ch10_sample_fix{input_pdf.suffix}")
        if not backup.exists():
            shutil.copy2(input_pdf, backup)
        temp_output.replace(output_pdf)
        patched_path = output_pdf
    else:
        temp_output.replace(output_pdf)
        patched_path = output_pdf

    if write_pages_text is not None:
        write_pages_snapshot(patched_path, write_pages_text)

    return patched_path


def normalize_for_phrase_search(text: str) -> str:
    return " ".join(text.split())


def run_checks(pdf_path: Path, checks: Sequence[PhraseCheck] = CHECKS) -> int:
    pdf_path = pdf_path.resolve()
    failures = 0
    with fitz.open(pdf_path) as doc:
        for check in checks:
            page_text = normalize_for_phrase_search(_page_text(doc, check.page))
            phrase = normalize_for_phrase_search(check.phrase)
            found = phrase in page_text
            passed = found if check.should_exist else not found
            status = "PASS" if passed else "FAIL"
            print(f"[{status}] p{check.page} {check.label}")
            if not passed:
                failures += 1
    return failures


def write_pages_snapshot(pdf_path: Path, output_path: Path, pages: Iterable[int] | None = None) -> None:
    pdf_path = pdf_path.resolve()
    output_path = output_path.resolve()
    if pages is None:
        pages = sorted({spec.page for spec in PATCHES})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[str] = []
    with fitz.open(pdf_path) as doc:
        for page in pages:
            chunks.append(f"===== PAGE {page} =====")
            chunks.append(_page_text(doc, page).rstrip())
            chunks.append("")
    output_path.write_text("\n".join(chunks), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="Input PDF path")
    parser.add_argument("--check", action="store_true", help="Verify the Chapter 10 sample fixes")
    parser.add_argument("--patch", action="store_true", help="Apply the Chapter 10 fix patches")
    parser.add_argument("--in-place", action="store_true", help="Overwrite the input PDF in place")
    parser.add_argument("--output", type=Path, help="Patched PDF output path")
    parser.add_argument(
        "--write-pages-text",
        type=Path,
        default=None,
        help="Optional text snapshot of the affected pages after patch/check",
    )
    args = parser.parse_args()
    if not args.check and not args.patch:
        parser.error("choose at least one of --check or --patch")
    if args.in_place and args.output:
        parser.error("use either --in-place or --output, not both")
    if args.patch and not args.in_place and args.output is None:
        parser.error("--patch requires either --in-place or --output")
    return args


def main() -> None:
    args = parse_args()
    pdf_path = args.pdf

    if args.patch:
        output_path = pdf_path if args.in_place else args.output
        assert output_path is not None
        patched = patch_pdf(pdf_path, output_path, write_pages_text=args.write_pages_text)
        print(f"patched PDF: {patched}")
        failures = run_checks(patched)
        sys.exit(1 if failures else 0)

    if args.write_pages_text is not None:
        write_pages_snapshot(pdf_path, args.write_pages_text)
    failures = run_checks(pdf_path)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
