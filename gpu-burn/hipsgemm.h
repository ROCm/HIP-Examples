extern "C" __global__ void hip_sgemm_kernel(hipLaunchParm lp, const int M, const int N, const int K, const float alpha,
                                          float *A, const int lda, float *B, const int ldb, const float beta, float *C, const int ldc);
