#include "sgemm.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <string>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

#define CUDA_CHECK(err)                                                        \
  if (err != cudaSuccess) {                                                    \
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at "           \
              << __LINE__ << std::endl;                                        \
    exit(EXIT_FAILURE);                                                        \
  }

void launch_sgemm_naive(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C, dim3 gridDim, dim3 blockDim) {
  sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

void launch_sgemm_coalesce(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C, dim3 gridDim, dim3 blockDim) {
  sgemm_coalesce<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

void launch_sgemm_coalesce_and_shared(int M, int N, int K, float alpha,
                                      float *A, float *B, float beta, float *C,
                                      dim3 gridDim, dim3 blockDim,
                                      size_t A_region) {
  size_t shared_bytes = A_region * 2 * sizeof(float);
  sgemm_coalesce_w_shared<<<gridDim, blockDim, shared_bytes>>>(
      M, N, K, alpha, A, B, beta, C, A_region);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

void launch_1d_tiling(int M, int N, int K, float alpha, float *A, float *B,
                      float beta, float *C) {
  const uint BM = 64;
  const uint BN = 64;
  const uint BK = 8;
  const uint TM = 8;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim(CEIL_DIV(BN * BM, TM));
  sgemm_1d_tiling<BM, BN, BK, TM>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

void launch_2d_tiling(int M, int N, int K, float alpha, float *A, float *B,
                      float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128) {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemm_2d_tiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    // this is a hacky solution to the underlying problem
    // of not having proper bounds checking in the kernel
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemm_2d_tiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " [naive|coalesce]" << std::endl;
    return 1;
  }

  std::string mode = argv[1];

  const int M = 4092;
  const int N = 4092;
  const int K = 4092;
  const float alpha = 1.0f;
  const float beta = 1.0f;

  size_t size_A = M * K * sizeof(float);
  size_t size_B = K * N * sizeof(float);
  size_t size_C = M * N * sizeof(float);

  float *h_A = new float[M * K];
  float *h_B = new float[K * N];
  float *h_C = new float[M * N];

  for (int i = 0; i < M * K; ++i)
    h_A[i] = 1.0f;
  for (int i = 0; i < K * N; ++i)
    h_B[i] = 1.0f;
  for (int i = 0; i < M * N; ++i)
    h_C[i] = 0.0f;

  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc((void **)&d_A, size_A));
  CUDA_CHECK(cudaMalloc((void **)&d_B, size_B));
  CUDA_CHECK(cudaMalloc((void **)&d_C, size_C));

  CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));

  dim3 blockDim(TILE_SIZE, TILE_SIZE);
  dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
               (N + blockDim.y - 1) / blockDim.y);

  if (mode == "naive") {
    std::cout << "Running SGEMM Naive" << std::endl;
    launch_sgemm_naive(M, N, K, alpha, d_A, d_B, beta, d_C, gridDim, blockDim);
  } else if (mode == "coalesce") {
    std::cout << "Running SGEMM Coalesce" << std::endl;
    launch_sgemm_coalesce(M, N, K, alpha, d_A, d_B, beta, d_C, gridDim,
                          blockDim);
  } else if (mode == "shared") {
    std::cout << "Running SGEMM Coalesce w shared" << std::endl;
    size_t A_region = TILE_SIZE * TILE_SIZE;
    launch_sgemm_coalesce_and_shared(M, N, K, alpha, d_A, d_B, beta, d_C,
                                     gridDim, blockDim, A_region);
  } else if (mode == "1d_tiling") {
    std::cout << "Running SGEMM 1d_tiling" << std::endl;
    launch_1d_tiling(M, N, K, alpha, d_A, d_B, beta, d_C);

  } else if (mode == "2d_tiling") {
    std::cout << "Running SGEMM 2d_tiling" << std::endl;
    launch_2d_tiling(M, N, K, alpha, d_A, d_B, beta, d_C);
  } else {
    std::cerr << "Invalid mode. Use 'naive' or 'coalesce', or 'shared'."
              << std::endl;
    return 1;
  }

  CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
  std::cout << "C[0] = " << h_C[0] << std::endl;

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}
