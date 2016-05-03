#include "hip_runtime.h"
#include "hipsgemm.h"

extern "C" __global__ void hip_sgemm_kernel(hipLaunchParm lp, const int M, const int N, const int K, const float alpha,
                                          float *A, const int lda, float *B, const int ldb, const float beta, float *C,
                                          const int ldc)
{
        //column major NN
        size_t idx_x = hipBlockIdx_x*hipBlockDim_x + hipThreadIdx_x;
        size_t idx_y = hipBlockIdx_y*hipBlockDim_y + hipThreadIdx_y;
        size_t dim_x = hipGridDim_x*hipBlockDim_x;
        size_t myIdx = idx_y * dim_x + idx_x;

        float local_c = beta * C[myIdx];
        for(int k = 0; k < K; k++)
        {
          local_c += alpha * A[ idx_y + k * K] * B[ idx_x * K + k];
        }
        C[myIdx] = local_c;
}
