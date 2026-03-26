// baseline_double_buffered_pipeline.cu -- Single-buffered tiled GEMM baseline.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t status = (call);                                         \
        if (status != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,\
                        cudaGetErrorString(status));                         \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

template<int TILE_M, int TILE_N, int CHUNK_K, int THREAD_TILE_M, int THREAD_TILE_N>
__global__ void gemm_single_buffered_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {
    const int block_row = blockIdx.y * TILE_M;
    const int block_col = blockIdx.x * TILE_N;

    __shared__ float A_tile[TILE_M][CHUNK_K];
    __shared__ float B_tile[CHUNK_K][TILE_N];

    float accum[THREAD_TILE_M][THREAD_TILE_N] = {0.0f};
    const int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int threads = blockDim.x * blockDim.y;

    for (int chunk_base = 0; chunk_base < K; chunk_base += CHUNK_K) {
        const int valid_k = max(0, min(CHUNK_K, K - chunk_base));

        for (int idx = linear_tid; idx < TILE_M * CHUNK_K; idx += threads) {
            const int row = idx / CHUNK_K;
            const int col = idx % CHUNK_K;
            const int global_row = block_row + row;
            const int global_col = chunk_base + col;
            A_tile[row][col] = (global_row < M && global_col < K)
                ? A[global_row * K + global_col]
                : 0.0f;
        }

        for (int idx = linear_tid; idx < CHUNK_K * TILE_N; idx += threads) {
            const int row = idx / TILE_N;
            const int col = idx % TILE_N;
            const int global_row = chunk_base + row;
            const int global_col = block_col + col;
            B_tile[row][col] = (row < valid_k && global_col < N && global_row < K)
                ? B[global_row * N + global_col]
                : 0.0f;
        }
        __syncthreads();

        for (int kk = 0; kk < valid_k; ++kk) {
            float a_frag[THREAD_TILE_M];
            float b_frag[THREAD_TILE_N];
            for (int i = 0; i < THREAD_TILE_M; ++i) {
                const int local_row = threadIdx.y * THREAD_TILE_M + i;
                a_frag[i] = A_tile[local_row][kk];
            }
            for (int j = 0; j < THREAD_TILE_N; ++j) {
                const int local_col = threadIdx.x * THREAD_TILE_N + j;
                b_frag[j] = B_tile[kk][local_col];
            }
            for (int i = 0; i < THREAD_TILE_M; ++i) {
                for (int j = 0; j < THREAD_TILE_N; ++j) {
                    accum[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < THREAD_TILE_M; ++i) {
        const int row = block_row + threadIdx.y * THREAD_TILE_M + i;
        if (row >= M) {
            continue;
        }
        for (int j = 0; j < THREAD_TILE_N; ++j) {
            const int col = block_col + threadIdx.x * THREAD_TILE_N + j;
            if (col >= N) {
                continue;
            }
            C[row * N + col] = accum[i][j];
        }
    }
}

int main() {
    NVTX_RANGE("main");
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;
    const size_t bytes_A = static_cast<size_t>(M) * K * sizeof(float);
    const size_t bytes_B = static_cast<size_t>(K) * N * sizeof(float);
    const size_t bytes_C = static_cast<size_t>(M) * N * sizeof(float);

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : h_A) {
        NVTX_RANGE("setup");
        v = dist(rng);
    }
    for (auto& v : h_B) {
        NVTX_RANGE("setup");
        v = dist(rng);
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, bytes_C));

    constexpr int THREAD_TILE_M = 4;
    constexpr int THREAD_TILE_N = 4;
    constexpr int TILE_M = THREAD_TILE_M * 16;
    constexpr int TILE_N = THREAD_TILE_N * 16;
    constexpr int CHUNK_K = 32;
    dim3 block(16, 16);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iterations = 10;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        NVTX_RANGE("compute_kernel:smem_single_buffered");
        gemm_single_buffered_kernel<TILE_M, TILE_N, CHUNK_K, THREAD_TILE_M, THREAD_TILE_N>
            <<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    const float avg_ms = elapsed_ms / static_cast<float>(iterations);
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));

    double checksum = 0.0;
    for (float v : h_C) {
        NVTX_RANGE("verify");
        checksum += v;
    }
    checksum /= static_cast<double>(M * N);

    std::printf(
        "Baseline GEMM (single-buffered shared-memory tiles): %.3f ms (avg over %d iters)\n",
        avg_ms,
        iterations);
    std::printf("TIME_MS: %.6f\n", avg_ms);
    std::printf("Checksum: %.6f\n", checksum);
    VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}
