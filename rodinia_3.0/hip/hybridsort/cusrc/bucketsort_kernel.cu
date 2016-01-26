#ifndef _BUCKETSORT_KERNEL_H_
#define _BUCKETSORT_KERNEL_H_

#include <stdio.h>

#define BUCKET_WARP_LOG_SIZE	5
#define BUCKET_WARP_N			1
#ifdef BUCKET_WG_SIZE_1
#define BUCKET_THREAD_N BUCKET_WG_SIZE_1
#else
#define BUCKET_THREAD_N			(BUCKET_WARP_N << BUCKET_WARP_LOG_SIZE)
#endif
#define BUCKET_BLOCK_MEMORY		(DIVISIONS * BUCKET_WARP_N)
#define BUCKET_BAND				128

texture<float, 1, cudaReadModeElementType> texPivot; 

__device__ int addOffset(volatile unsigned int *s_offset, unsigned int data, unsigned int threadTag){
    unsigned int count;

    do{
        count = s_offset[data] & 0x07FFFFFFU;
        count = threadTag | (count + 1);
        s_offset[data] = count;
    }while(s_offset[data] != count);

	return (count & 0x07FFFFFFU) - 1;
}

__global__ void
bucketcount( float *input, int *indice, unsigned int *d_prefixoffsets, int size)
{
	volatile __shared__ unsigned int s_offset[BUCKET_BLOCK_MEMORY]; 

    const unsigned int threadTag = threadIdx.x << (32 - BUCKET_WARP_LOG_SIZE);
    const int warpBase = (threadIdx.x >> BUCKET_WARP_LOG_SIZE) * DIVISIONS; 
    const int numThreads = blockDim.x * gridDim.x;
	for (int i = threadIdx.x; i < BUCKET_BLOCK_MEMORY; i += blockDim.x)
		s_offset[i] = 0; 

	__syncthreads(); 

	for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < size; tid += numThreads) {
		float elem = input[tid]; 

		int idx  = DIVISIONS/2 - 1; 
		int jump = DIVISIONS/4; 
		float piv = tex1Dfetch(texPivot, idx); //s_pivotpoints[idx]; 

		while(jump >= 1){
			idx = (elem < piv) ? (idx - jump) : (idx + jump);
			piv = tex1Dfetch(texPivot, idx); //s_pivotpoints[idx]; 
			jump /= 2; 
		}
		idx = (elem < piv) ? idx : (idx + 1); 

		indice[tid] = (addOffset(s_offset + warpBase, idx, threadTag) << LOG_DIVISIONS) + idx;  //atomicInc(&offsets[idx], size + 1);
	}

	__syncthreads(); 

	int prefixBase = blockIdx.x * BUCKET_BLOCK_MEMORY; 

	for (int i = threadIdx.x; i < BUCKET_BLOCK_MEMORY; i += blockDim.x)
		d_prefixoffsets[prefixBase + i] = s_offset[i] & 0x07FFFFFFU; 
}

__global__ void bucketprefixoffset(unsigned int *d_prefixoffsets, unsigned int *d_offsets, int blocks) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x; 
	int size = blocks * BUCKET_BLOCK_MEMORY; 
	int sum = 0; 

	for (int i = tid; i < size; i += DIVISIONS) {
		int x = d_prefixoffsets[i]; 
		d_prefixoffsets[i] = sum; 
		sum += x; 
	}

	d_offsets[tid] = sum; 
}

__global__ void
bucketsort(float *input, int *indice, float *output, int size, unsigned int *d_prefixoffsets, 
		   unsigned int *l_offsets)
{
	volatile __shared__ unsigned int s_offset[BUCKET_BLOCK_MEMORY]; 

	int prefixBase = blockIdx.x * BUCKET_BLOCK_MEMORY; 
    const int warpBase = (threadIdx.x >> BUCKET_WARP_LOG_SIZE) * DIVISIONS; 
    const int numThreads = blockDim.x * gridDim.x;
	for (int i = threadIdx.x; i < BUCKET_BLOCK_MEMORY; i += blockDim.x)
		s_offset[i] = l_offsets[i & (DIVISIONS - 1)] + d_prefixoffsets[prefixBase + i]; 

	__syncthreads(); 

	for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < size; tid += numThreads) {

		float elem = input[tid]; 
		int id = indice[tid]; 

		output[s_offset[warpBase + (id & (DIVISIONS - 1))] + (id >> LOG_DIVISIONS)] = elem;
		int test = s_offset[warpBase + (id & (DIVISIONS - 1))] + (id >> LOG_DIVISIONS);

	}
}

#endif 
