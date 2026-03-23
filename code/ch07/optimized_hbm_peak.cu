// optimized_hbm_peak.cu -- HBM peak bandwidth kernel for Blackwell.
// CUDA 13 + Blackwell: Uses Float8 (32-byte aligned) for 256-bit loads

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// CUDA 13 + Blackwell: 32-byte aligned type for 256-bit loads
struct alignas(32) Float8 {
    float elems[8];
};
static_assert(sizeof(Float8) == 32, "Float8 must be 32 bytes");
static_assert(alignof(Float8) == 32, "Float8 must be 32-byte aligned");

// HBM peak bandwidth kernel - Blackwell B200/B300 with 256-bit loads
__global__ void hbm_peak_copy(const Float8* __restrict__ src,
                               Float8* __restrict__ dst,
                               size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for (size_t base = tid * 4; base < n; base += stride * 4) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            size_t idx = base + j;
            if (idx >= n) break;
            dst[idx] = src[idx];  // 256-bit load/store
        }
    }
}

int main() {
    NVTX_RANGE("main");
    const size_t target_bytes = 512ULL * 1024 * 1024;  // Match baseline workload
    const size_t n_floats = target_bytes / sizeof(float);
    const size_t n_vec8 = n_floats / 8;
    
    Float8 *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, target_bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, target_bytes));
    CUDA_CHECK(cudaMemset(d_src, 1, target_bytes));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int iterations = 10;
    const int block_size = 256;
    const int grid_size = static_cast<int>((n_vec8 + block_size - 1) / block_size);
    const int launch_blocks = grid_size > 4096 ? 4096 : grid_size;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        NVTX_RANGE("compute_kernel:hbm_peak_copy");
        hbm_peak_copy<<<launch_blocks, block_size>>>(d_src, d_dst, n_vec8);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    double bytes_transferred = 2.0 * target_bytes * iterations;
    double bandwidth_tbs = (bytes_transferred / elapsed_ms) / 1e9;
    
    printf("HBM peak (Float8): %.2f ms, %.2f TB/s\n", elapsed_ms / iterations, bandwidth_tbs);
    printf("TIME_MS: %.6f\n", elapsed_ms / iterations);
    printf("Vectorized path reuses the same 512 MB workload with wider loads/stores.\n");

#ifdef VERIFY
    float* h_verify = static_cast<float*>(malloc(target_bytes));
    if (!h_verify) {
        fprintf(stderr, "Host allocation failed\n");
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaMemcpy(h_verify, d_dst, target_bytes, cudaMemcpyDeviceToHost));
    float checksum = 0.0f;
    VERIFY_CHECKSUM(h_verify, static_cast<int>(n_floats), &checksum);
    VERIFY_PRINT_CHECKSUM(checksum);
    free(h_verify);
#endif
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    
    return 0;
}


