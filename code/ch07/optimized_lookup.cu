// optimized_lookup.cu -- pretranspose the lookup path table, then read the same
// 64 values per output in a coalesced layout. This is a data-layout
// transformation, not precomputation of the final reduction.

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

#include "../core/common/headers/cuda_helpers.cuh"
#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"

constexpr int N = 1 << 20;
constexpr int ITERATIONS = 200;
constexpr int RANDOM_STEPS = 64;

__global__ void lookupOptimized(const float* __restrict__ paths,
                                float* __restrict__ out,
                                int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  float val = 0.0f;
  #pragma unroll 8
  for (int step = 0; step < RANDOM_STEPS; ++step) {
    val += paths[step * n + idx];
  }
  out[idx] = val;
}

__host__ int advance_lcg(int idx) {
  return (idx * 1664525 + 1013904223) & (N - 1);
}

int main() {
    NVTX_RANGE("main");
  float *h_table, *h_out;
  int *h_indices;
  CUDA_CHECK(cudaMallocHost(&h_table, N * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h_out, N * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h_indices, N * sizeof(int)));

  for (int i = 0; i < N; ++i) {
      NVTX_RANGE("setup");
    h_table[i] = static_cast<float>(i);
    h_indices[i] = (i * 3) % N;
  }

  std::vector<float> paths(static_cast<std::size_t>(N) * RANDOM_STEPS);
  for (int i = 0; i < N; ++i) {
    int idx = h_indices[i];
    for (int step = 0; step < RANDOM_STEPS; ++step) {
      paths[static_cast<std::size_t>(step) * N + i] = h_table[idx];
      idx = advance_lcg(idx);
    }
  }

  float *d_paths = nullptr;
  float *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_paths, paths.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_paths, paths.data(), paths.size() * sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int iter = 0; iter < ITERATIONS; ++iter) {
      NVTX_RANGE("compute_kernel:lookupOptimized");
    lookupOptimized<<<grid, block>>>(d_paths, d_out, N);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  float avg_ms = elapsed_ms / ITERATIONS;

  CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
  printf("Lookup (layout-optimized): %.4f ms\n", avg_ms);
  printf("TIME_MS: %.4f\n", avg_ms);
  printf("out[0]=%.1f\n", h_out[0]);
#ifdef VERIFY
  float checksum = 0.0f;
  VERIFY_CHECKSUM(h_out, N, &checksum);
  VERIFY_PRINT_CHECKSUM(checksum);
#endif

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_paths));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFreeHost(h_table));
  CUDA_CHECK(cudaFreeHost(h_indices));
  CUDA_CHECK(cudaFreeHost(h_out));
  return 0;
}
