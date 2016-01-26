#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>

// declare texture reference for 1D float texture
texture<float4, 1, cudaReadModeElementType> tex;
texture<float, 1, cudaReadModeElementType> txt; 

__device__ float4 sortElem(float4 r) {
	float4 nr;

	nr.x = (r.x > r.y) ? r.y : r.x; 
	nr.y = (r.y > r.x) ? r.y : r.x; 
	nr.z = (r.z > r.w) ? r.w : r.z; 
	nr.w = (r.w > r.z) ? r.w : r.z; 

	r.x = (nr.x > nr.z) ? nr.z : nr.x; 
	r.y = (nr.y > nr.w) ? nr.w : nr.y; 
	r.z = (nr.z > nr.x) ? nr.z : nr.x; 
	r.w = (nr.w > nr.y) ? nr.w : nr.y; 

	nr.x = r.x; 
	nr.y = (r.y > r.z) ? r.z : r.y; 
	nr.z = (r.z > r.y) ? r.z : r.y; 
	nr.w = r.w; 
	return nr; 
}

__device__ float4 getLowest(float4 a, float4 b)
{
	//float4 na; 
	a.x = (a.x < b.w) ? a.x : b.w; 
	a.y = (a.y < b.z) ? a.y : b.z; 
	a.z = (a.z < b.y) ? a.z : b.y; 
	a.w = (a.w < b.x) ? a.w : b.x; 
	return a; 
}

__device__ float4 getHighest(float4 a, float4 b)
{
	b.x = (a.w >= b.x) ? a.w : b.x; 
	b.y = (a.z >= b.y) ? a.z : b.y; 
	b.z = (a.y >= b.z) ? a.y : b.z; 
	b.w = (a.x >= b.w) ? a.x : b.w; 
	return b; 
}


__constant__ int constStartAddr[DIVISIONS + 1]; 
__constant__ int finalStartAddr[DIVISIONS + 1]; 
__constant__ int nullElems[DIVISIONS]; 

__global__ void
mergeSortFirst(float4 *result, int listsize) 
{
    // Block index
    int bx = blockIdx.x;
    // Thread index
    //int tx = threadIdx.x;
		if(bx*blockDim.x + threadIdx.x < listsize/4){
			float4 r = tex1Dfetch(tex, (int)(bx*blockDim.x + threadIdx.x));
			result[bx * blockDim.x + threadIdx.x] = sortElem(r); 
		}
}

__global__ void
mergeSortPass(float4 *result, int nrElems, int threadsPerDiv) 
{
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x; 
	// The division to work on
	int division = tid / threadsPerDiv; 
	if(division >= DIVISIONS) return; 
	// The block within the division
	int int_tid = tid - division * threadsPerDiv; 
	int Astart = constStartAddr[division] + int_tid * nrElems; 

	int Bstart = Astart + nrElems/2;
	float4 *resStart = &(result[Astart]); 

	if(Astart >= constStartAddr[division + 1]) 
		return; 
	if(Bstart >= constStartAddr[division + 1]){
		for(int i=0; i<(constStartAddr[division + 1] - Astart); i++)
		{
			resStart[i] = tex1Dfetch(tex, Astart + i); 
		}
		return; 
	}

	int aidx = 0; 
	int bidx = 0; 
	int outidx = 0; 
	float4 a, b;
	a = tex1Dfetch(tex, Astart + aidx);  
	b = tex1Dfetch(tex, Bstart + bidx); 
	
	while(true)//aidx < nrElems/2)// || (bidx < nrElems/2  && (Bstart + bidx < constEndAddr[division])))
	{
		/**
		 * For some reason, it's faster to do the texture fetches here than
		 * after the merge
		 */
		float4 nextA = tex1Dfetch(tex, Astart + aidx + 1); 
		float4 nextB = tex1Dfetch(tex, Bstart + bidx + 1); 

		float4 na = getLowest(a,b); 
		float4 nb = getHighest(a,b); 
		a = sortElem(na); 
		b = sortElem(nb); 
		// Now, a contains the lowest four elements, sorted
		resStart[outidx++] = a; 

		bool elemsLeftInA; 
		bool elemsLeftInB;

		elemsLeftInA = (aidx + 1 < nrElems/2); // Astart + aidx + 1 is allways less than division border 
		elemsLeftInB = (bidx + 1 < nrElems/2) && (Bstart + bidx + 1 < constStartAddr[division + 1]); 

		if(elemsLeftInA){
			if(elemsLeftInB){
				if(nextA.x < nextB.x) { aidx += 1; a = nextA; }  
				else { bidx += 1;  a = nextB; }
			}
			else {
				aidx += 1; a = nextA;
			}
		}
		else {
			if(elemsLeftInB){
				bidx += 1;  a = nextB;
			}
			else {
				break; 
			}
		}

	}
	resStart[outidx++] = b;
}

__global__ void
mergepack(float *orig, float *result) 
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
	int division = blockIdx.y; 

	if((finalStartAddr[division] + idx) >= finalStartAddr[division + 1]) return; 
	result[finalStartAddr[division] + idx] = orig[constStartAddr[division]*4 + nullElems[division] + idx]; 
}


#endif // #ifndef _MATRIXMUL_KERNEL_H_
