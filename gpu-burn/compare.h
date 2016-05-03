extern "C" __global__ void compare_kernel(hipLaunchParm lp, float *C, int *faultyElems, size_t iters);

extern "C" __global__ void compareD_kernel(hipLaunchParm lp, double *C, int *faultyElems, size_t iters);
