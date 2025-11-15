#include "sgemm.cuh"
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  // compute position in C that this thread is responsible for
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

__global__ void sgemm_coalesce(int M, int N, int K, float alpha, const float *A,
                               const float *B, float beta, float *C) {
  // compute position in C that this thread is responsible for
  const uint x = blockIdx.x * blockDim.x + (threadIdx.x / blockDim.x);
  const uint y = blockIdx.y * blockDim.y + (threadIdx.x % blockDim.x);

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

__global__ void sgemm_coalesce_w_shared(int M, int N, int K, float alpha,
                                        const float *A, const float *B,
                                        float beta, float *C, size_t A_region) {
  // have both As, and Bs. As first half, Bs second half
  extern __shared__ float sdata[];
  float *As = sdata;
  float *Bs = sdata + A_region;
  // compute position in C that this thread is responsible for
  const uint row = blockIdx.x * blockDim.x + (threadIdx.x / blockDim.x);
  const uint col = blockIdx.y * blockDim.y + (threadIdx.x % blockDim.x);
  float sum = 0.0f;
  // Populate shared mem and loop through K per tile t
  for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
    // global indicies
    int A_col = t * TILE_SIZE + threadIdx.x;
    int B_row = t * TILE_SIZE + threadIdx.y;

    // Load Tile into shared mem:
    if (row < M && A_col < K) {
      As[threadIdx.y * TILE_SIZE + threadIdx.x] = A[row * K + A_col];
    } else {
      As[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
    }

    if (B_row < K && col < N) {
      Bs[threadIdx.y * TILE_SIZE + threadIdx.x] = B[B_row * N + col];
    } else {
      Bs[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
    }

    __syncthreads();

    for (int i = 0; i < TILE_SIZE; ++i) {
      sum += As[threadIdx.y * TILE_SIZE + i] * Bs[i * TILE_SIZE + threadIdx.x];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum * alpha + beta * C[row * N + col];
  }
}

// template <const int BM, const int BN, const int BK, const int TM>
// __global__ void sgemm_1d_tiling(int M, int N, int K, float alpha,
//                                 const float *A, const float *B, float beta,
//                                 float *C) {
//   const uint cRow = blockIdx.y;
//   const uint cCol = blockIdx.x;
//
//   const int threadCol = threadIdx.x % BN;
//   const int threadRow = threadIdx.x / BN;
//
//   // allocate shared mem for A and B
//   __shared__ float As[BM * BK];
//   __shared__ float Bs[BN * BK];
//
//   // advance pointers to correct view for threads
//   A += cRow * BM * K;
//   B += cCol * BN;
//   C += cRow * BM * N + cCol * BN;
//
//   // define the col and row indexes for within blocks
//   const uint innerRowA = threadIdx.x / BK;
//   const uint innerColA = threadIdx.x % BK;
//   const uint innerRowB = threadIdx.x / BN;
//   const uint innerColB = threadIdx.x % BN;
//
//   // allocate register memory for the results calculated by each thread
//   float threadResults[TM] = {0.0};
//
//   // outer tile loop
//   for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
//
//     As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
//     Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
//     __syncthreads(); // make sure to sync to maintain L1 cache coherency
//
//     A += BK;
//     B += BK * N;
//
//     for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
//       // cache B value for dot product;
//       float Btmp = Bs[dotIdx * BN + threadCol];
//       for (uint resIdx = 0; resIdx < TM; ++resIdx) {
//         threadResults[resIdx] +=
//             As[(threadRow * TM + dotIdx) * BK + dotIdx] * Btmp;
//       }
//     }
//     __syncthreads();
//   }
//   // write out the results
//   for (uint resIdx = 0; resIdx < TM; ++resIdx) {
//     C[(threadRow * TM + resIdx) * N + threadCol] =
//         alpha * threadResults[resIdx] +
//         beta * C[(threadRow * TM + resIdx) * N + threadCol];
//   }
// }
