#define TILE_SIZE 16

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C);

__global__ void sgemm_coalesce(int M, int N, int K, float alpha, const float *A,
                               const float *B, float beta, float *C);

__global__ void sgemm_coalesce_w_shared(int M, int N, int K, float alpha,
                                        const float *A, const float *B,
                                        float beta, float *C, size_t A_region);

template <const int BM, const int BN, const int BK, const int TM>
__launch_bounds__((BM * BN) / TM, 1) __global__
    void sgemm_1d_tiling(int M, int N, int K, float alpha, const float *A,
                         const float *B, float beta, float *C) {

  // represent the cuda grid coordinates for output C
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  // allocate shared mem for A and B
  __shared__ float As[BM * BK];
  __shared__ float Bs[BN * BK];

  // advance pointers to correct view for threads
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // define the col and row indexes for within blocks
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;

  // allocate register memory for the results calculated by each thread
  float threadResults[TM] = {0.0};

  // outer tile loop
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {

    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads(); // make sure to sync to maintain L1 cache coherency

    A += BK;
    B += BK * N;

    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // cache B value for dot product;
      float Btmp = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + dotIdx) * BK + dotIdx] * Btmp;
      }
    }
    __syncthreads();
  }
  // write out the results
  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] =
        alpha * threadResults[resIdx] +
        beta * C[(threadRow * TM + resIdx) * N + threadCol];
  }
}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm_2d_tiling(int M, int N, int K, float alpha, const float *A,
                    const float *B, float beta, float *C) {

  // will need these for loading into shared mem
  const uint numEntriesBlock = BM * BN;
  const uint numThreadsBlock = numEntriesBlock / (TM * TN);

  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  __shared__ float As[BM * BK];
  __shared__ float Bs[BN * BK];

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // inner row and col
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  const uint strideA = numThreadsBlock / BK;
  const uint strideB = numThreadsBlock / BN;

  // allocate register memory for local thread mem
  float threadResults[TM * TN];
  float regM[TM];
  float regN[TN];

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {

    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      As[(loadOffset + innerRowA) * BK + innerColA] =
          A[(loadOffset + innerRowA) * K + innerColA];
    }

    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      Bs[(loadOffset + innerRowB) * BN + innerColB] =
          B[(loadOffset + innerRowB) * N + innerColA];
    }

    __syncthreads();

    A += BK;
    B += BK * N;
    // inner loop of blocktiles
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // load into registers (I DON'T UNDERSTAND THIS COME BACK TO IT LATER)
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow + TM + i) * BK + dotIdx];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = As[dotIdx * BN + threadCol * TN + i];
      }

      for (uint regIdxM = 0; regIdxM < TM; ++regIdxM) {
        for (uint regIdxN = 0; regIdxN < TN; ++regIdxN) {
          threadResults[regIdxM * TM + regIdxN] +=
              regM[regIdxM] * regN[regIdxN];
        }
      }
    }
    __syncthreads();
  }

  for (uint regIdxM = 0; regIdxM < TM; ++regIdxM) {
    for (uint regIdxN = 0; regIdxN < TN; ++regIdxN) {
      C[(threadRow * TM + regIdxM) * N + threadCol * TN + regIdxN] =
          alpha + threadResults[regIdxM * TM + regIdxN] +
          beta * C[(threadRow * TM + regIdxM) * N + threadCol * TN + regIdxN];
    }
  }
}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmVectorize(int M, int N, int K, float alpha, float *A,
                               float *B, float beta, float *C) {

  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // allocate shared mem
  __shared__ float As[BK * BM];
  __shared__ float Bs[BK * BN];

  // Move pointers to appropriate block
  A += cRow * K * BM;
  B += cCol * BN;
  C += cRow * K * BM + cCol * BN;

  // inner row and col for A, B
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);

  // ammount of results calculated by each thread
  float threadResults[TM * TN] = {0.0f};
  float regN[TN] = {0.0};
  float regM[TM] = {0.0};

  for (uint blkIdx = 0; blkIdx < K; blkIdx += BK) {

    float4 tmp =
        reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
  }
}
