#include "hip_runtime.h"
#include "compare.h"
// Actually, there are no rounding errors due to results being accumulated in an arbitrary order..
// Therefore EPSILON = 0.0f is OK
#define EPSILON 0.001f
#define EPSILOND 0.0000001

extern "C" __global__ void compare_kernel(hipLaunchParm lp, float *C, int *faultyElems, size_t iters) {
	size_t iterStep = hipBlockDim_x*hipBlockDim_y*hipGridDim_x*hipGridDim_y;
	size_t myIndex = (hipBlockIdx_y*hipBlockDim_y + hipThreadIdx_y)* // Y
		hipGridDim_x*hipBlockDim_x + // W
		hipBlockIdx_x*hipBlockDim_x + hipThreadIdx_x; // X

	int myFaulty = 0;
	for (size_t i = 1; i < iters; ++i)
		if (fabsf(C[myIndex] - C[myIndex + i*iterStep]) > EPSILON)
			myFaulty++;

	atomicAdd(faultyElems, myFaulty);
}

extern "C" __global__ void compareD_kernel(hipLaunchParm lp, double *C, int *faultyElems, size_t iters) {
	size_t iterStep = hipBlockDim_x*hipBlockDim_y*hipGridDim_x*hipGridDim_y;
	size_t myIndex = (hipBlockIdx_y*hipBlockDim_y + hipThreadIdx_y)* // Y
		hipGridDim_x*hipBlockDim_x + // W
		hipBlockIdx_x*hipBlockDim_x + hipThreadIdx_x; // X

	int myFaulty = 0;
	for (size_t i = 1; i < iters; ++i)
		if (fabs(C[myIndex] - C[myIndex + i*iterStep]) > EPSILOND)
			myFaulty++;

	atomicAdd(faultyElems, myFaulty);
}
