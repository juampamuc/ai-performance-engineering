// baseline_warp_spec_pingpong.cu - Single-stage warp-role pipeline (Ch10)
//
// WHAT:
//   Three dedicated warps cooperate on each tile:
//   - Warp 0 loads the next tile from global memory
//   - Warp 1 computes the tile
//   - Warp 2 stores the finished tile
//
// WHY THIS IS THE BASELINE:
//   The block has only one shared-memory stage, so the producer warp cannot
//   start loading the next tile until compute and store finish on the current
//   tile. That exposes the cost of a non-ping-pong pipeline.
//
// COMPARE WITH:
//   optimized_warp_spec_pingpong.cu
//   - Adds a two-stage ping-pong buffer
//   - Keeps the same warp roles, but overlaps producer work across tiles

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"

namespace cg = cooperative_groups;

namespace {
constexpr int TILE_SIZE = 64;
constexpr int TILE_ELEMS = TILE_SIZE * TILE_SIZE;
constexpr int WARPS_PER_BLOCK = 3;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
constexpr int TOTAL_TILES = 4096;
constexpr int ITERATIONS = 10;

constexpr int PRODUCER_WARP = 0;
constexpr int COMPUTE_WARP = 1;
constexpr int CONSUMER_WARP = 2;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t status_ = (call);                                          \
        if (status_ != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(status_));                         \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

__device__ void compute_tile(const float* __restrict__ a_tile,
                             const float* __restrict__ b_tile,
                             float* __restrict__ c_tile,
                             int lane_id) {
    for (int idx = lane_id; idx < TILE_ELEMS; idx += warpSize) {
        float x = a_tile[idx];
        float y = b_tile[idx];
        c_tile[idx] = sqrtf(x * x + y * y);
    }
}

__global__ void warp_specialized_single_stage(const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              float* __restrict__ C,
                                              int total_tiles) {
    cg::thread_block block = cg::this_thread_block();

    extern __shared__ float shared_mem[];
    float* tile_a = shared_mem;
    float* tile_b = tile_a + TILE_ELEMS;
    float* tile_c = tile_b + TILE_ELEMS;

    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;

    const bool is_producer = warp_id == PRODUCER_WARP;
    const bool is_compute = warp_id == COMPUTE_WARP;
    const bool is_consumer = warp_id == CONSUMER_WARP;

    for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
        const size_t offset = static_cast<size_t>(tile) * TILE_ELEMS;

        if (is_producer) {
            for (int idx = lane_id; idx < TILE_ELEMS / 4; idx += warpSize) {
                float4 a4 = *reinterpret_cast<const float4*>(&A[offset + idx * 4]);
                float4 b4 = *reinterpret_cast<const float4*>(&B[offset + idx * 4]);
                *reinterpret_cast<float4*>(&tile_a[idx * 4]) = a4;
                *reinterpret_cast<float4*>(&tile_b[idx * 4]) = b4;
            }
        }
        block.sync();

        if (is_compute) {
            compute_tile(tile_a, tile_b, tile_c, lane_id);
        }
        block.sync();

        if (is_consumer) {
            for (int idx = lane_id; idx < TILE_ELEMS / 4; idx += warpSize) {
                float4 c4 = *reinterpret_cast<const float4*>(&tile_c[idx * 4]);
                *reinterpret_cast<float4*>(&C[offset + idx * 4]) = c4;
            }
        }
        block.sync();
    }
}

double checksum(const std::vector<float>& data) {
    double acc = 0.0;
    for (float v : data) {
        NVTX_RANGE("verify");
        acc += static_cast<double>(v);
    }
    return acc / static_cast<double>(data.size());
}
}  // namespace

int main() {
    NVTX_RANGE("main");

    const size_t elems = static_cast<size_t>(TOTAL_TILES) * TILE_ELEMS;
    const size_t bytes = elems * sizeof(float);

    std::vector<float> h_A(elems), h_B(elems), h_C(elems);
    for (size_t i = 0; i < elems; ++i) {
        NVTX_RANGE("setup");
        h_A[i] = static_cast<float>(i % 251) * 0.01f;
        h_B[i] = static_cast<float>((i + 17) % 127) * 0.02f;
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    const dim3 block(THREADS_PER_BLOCK);
    const dim3 grid(std::min(TOTAL_TILES, std::max(1, prop.multiProcessorCount * 2)));
    const size_t shared_bytes = 3 * TILE_ELEMS * sizeof(float);

    warp_specialized_single_stage<<<grid, block, shared_bytes>>>(d_A, d_B, d_C, TOTAL_TILES);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < ITERATIONS; ++iter) {
        NVTX_RANGE("compute_kernel:warp_specialized_single_stage");
        warp_specialized_single_stage<<<grid, block, shared_bytes>>>(d_A, d_B, d_C, TOTAL_TILES);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const float avg_ms = total_ms / static_cast<float>(ITERATIONS);

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));
    const double chk = checksum(h_C);

    std::printf("Baseline warp-specialized single-stage pipeline: %.3f ms\n", avg_ms);
    std::printf("TIME_MS: %.6f\n", avg_ms);
    std::printf("Checksum: %.6f\n", chk);
    VERIFY_PRINT_CHECKSUM(static_cast<float>(chk));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}
