// baseline_cooperative_persistent.cu
// Single-launch cooperative persistent kernel with synchronous staging.

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../core/common/headers/cuda_helpers.cuh"
#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"

namespace cg = cooperative_groups;

constexpr int ELEMENTS = 1 << 24;          // 16M elements (~64 MB)
constexpr int ITERATIONS = 40;
constexpr int WARMUP_ITERATIONS = 2;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int ITEMS_PER_THREAD = 4;
constexpr int TILE_ELEMS = THREADS_PER_BLOCK * ITEMS_PER_THREAD;

__device__ __forceinline__ float fused_transform(float x, float scale, float bias) {
  x = __fmaf_rn(x, scale, bias);
  x = tanhf(x);
  x = x + 0.01f * __sinf(x);
  return __expf(x) - 1.0f;
}

__device__ __forceinline__ int tile_remaining(int elements, int base) {
  int remaining = elements - base;
  remaining = remaining > TILE_ELEMS ? TILE_ELEMS : remaining;
  return remaining > 0 ? remaining : 0;
}

__global__ void cooperative_persistent_baseline(float* data,
                                                int elements,
                                                float scale,
                                                float bias,
                                                int iterations) {
  cg::thread_block block = cg::this_thread_block();
  extern __shared__ float tile_buffer[];

  const int lane_base = threadIdx.x * ITEMS_PER_THREAD;
  const int total_tiles = (elements + TILE_ELEMS - 1) / TILE_ELEMS;

  for (int iter = 0; iter < iterations; ++iter) {
    for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
      const int base = tile * TILE_ELEMS;
      const int remaining = tile_remaining(elements, base);

#pragma unroll
      for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        const int local_idx = lane_base + item;
        if (local_idx < remaining) {
          tile_buffer[local_idx] = data[base + local_idx];
        }
      }
      block.sync();

#pragma unroll
      for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        const int local_idx = lane_base + item;
        if (local_idx < remaining) {
          tile_buffer[local_idx] = fused_transform(tile_buffer[local_idx], scale, bias);
        }
      }
      block.sync();

#pragma unroll
      for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        const int local_idx = lane_base + item;
        if (local_idx < remaining) {
          data[base + local_idx] = tile_buffer[local_idx];
        }
      }
      block.sync();
    }
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

int main() {
  NVTX_RANGE("main");
  std::vector<float> h_data(ELEMENTS);
  std::mt19937 rng(1337);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : h_data) {
    NVTX_RANGE("setup");
    v = dist(rng);
  }

  float* d_data = nullptr;
  const size_t bytes = static_cast<size_t>(ELEMENTS) * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_data, bytes));
  CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  const int total_tiles = (ELEMENTS + TILE_ELEMS - 1) / TILE_ELEMS;
  const int max_blocks = prop.multiProcessorCount * 4;
  const int grid_blocks = max_blocks < total_tiles ? max_blocks : total_tiles;
  const dim3 grid(grid_blocks > 0 ? grid_blocks : 1);
  const dim3 block(THREADS_PER_BLOCK);
  const size_t shared_bytes = TILE_ELEMS * sizeof(float);

  cooperative_persistent_baseline<<<grid, block, shared_bytes>>>(
      d_data, ELEMENTS, 1.001f, 0.05f, WARMUP_ITERATIONS);
  CUDA_CHECK_LAST_ERROR();
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  NVTX_RANGE("compute_kernel:cooperative_persistent_baseline");
  cooperative_persistent_baseline<<<grid, block, shared_bytes>>>(
      d_data, ELEMENTS, 1.001f, 0.05f, ITERATIONS);
  CUDA_CHECK_LAST_ERROR();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  const float avg_ms = total_ms / static_cast<float>(ITERATIONS);

  CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, bytes, cudaMemcpyDeviceToHost));
  const double chk = checksum(h_data);

  std::printf("Baseline cooperative persistent pipeline: %.3f ms (%d iterations)\n",
              avg_ms,
              ITERATIONS);
  std::printf("TIME_MS: %.6f\n", avg_ms);
  std::printf("Checksum: %.6f\n", chk);
  VERIFY_PRINT_CHECKSUM(static_cast<float>(chk));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_data));
  return 0;
}
