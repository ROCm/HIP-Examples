#include "hip/hip_runtime.h"
////////////////////////////////////////////////////////////////////////////////
// Includes
////////////////////////////////////////////////////////////////////////////////
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "mergesort.cuh"
#include "mergesort_kernel.hip.cu"
////////////////////////////////////////////////////////////////////////////////
// Defines
////////////////////////////////////////////////////////////////////////////////
#define BLOCKSIZE	256
#define ROW_LENGTH	BLOCKSIZE * 4
#define ROWS		4096

////////////////////////////////////////////////////////////////////////////////
// The mergesort algorithm
////////////////////////////////////////////////////////////////////////////////
float4* runMergeSort(int listsize, int divisions, 
				     float4 *d_origList, float4 *d_resultList, 
				     int *sizes, int *nullElements,
					 unsigned int *origOffsets)
{
	int *startaddr = (int *)malloc((divisions + 1)*sizeof(int)); 
	int largestSize = -1; 
	startaddr[0] = 0; 
	for(int i=1; i<=divisions; i++)
	{
		startaddr[i] = startaddr[i-1] + sizes[i-1];
		if(sizes[i-1] > largestSize) largestSize = sizes[i-1]; 
	}
	largestSize *= 4; 
//	printf("\n *** list size = %d *** \n",listsize);
	#ifdef USE_TEXTURES
	// Setup texture
	hipChannelFormatDesc channelDesc = hipCreateChannelDesc(32, 32, 32, 32, hipChannelFormatKindFloat);
	tex.addressMode[0] = hipAddressModeWrap;
	tex.addressMode[1] = hipAddressModeWrap;
	tex.filterMode = hipFilterModePoint;
	tex.normalized = false;
	/*#else
	float4 *dg_origList;
	hipMalloc((void**) &dg_origList, (listsize/4)*sizeof(float4));
        hipMemcpy(dg_origList,d_origList,(listsize/4)*sizeof(float4), hipMemcpyHostToDevice);*/
	#endif
	////////////////////////////////////////////////////////////////////////////
	// First sort all float4 elements internally
	////////////////////////////////////////////////////////////////////////////
	#ifdef MERGE_WG_SIZE_0
	const int THREADS = MERGE_WG_SIZE_0;
	#else
	const int THREADS = 256; 
	#endif
	dim3 threads(THREADS, 1);
	int blocks = ((listsize/4)%THREADS == 0) ? (listsize/4)/THREADS : (listsize/4)/THREADS + 1; 
	dim3 grid(blocks, 1);
	#ifdef USE_TEXTURES
	hipBindTexture(0,tex,d_origList, channelDesc, listsize*sizeof(float)); 
	hipLaunchKernel(mergeSortFirst, dim3(grid), dim3(threads ), 0, 0, d_resultList, listsize);
	#else
	hipLaunchKernel(mergeSortFirst, dim3(grid), dim3(threads ), 0, 0,d_origList, d_resultList, listsize);
	#endif
	////////////////////////////////////////////////////////////////////////////
	// Then, go level by level
	////////////////////////////////////////////////////////////////////////////
	#ifdef MEMCPYTOSYMBOL
	hipMemcpyToSymbol(constStartAddr, startaddr, (divisions + 1)*sizeof(int)); 
	hipMemcpyToSymbol(finalStartAddr, origOffsets, (divisions + 1)*sizeof(int)); 
	hipMemcpyToSymbol(nullElems, nullElements, (divisions)*sizeof(int));
	#else
	int *dconstStartAddr ;
	unsigned int *dfinalStartAddr ;
	int *dnullElements ;
	hipMalloc((void**) &dconstStartAddr, (divisions + 1)*sizeof(int));
	hipMalloc((void**) &dfinalStartAddr, (divisions + 1)*sizeof(unsigned int));
	hipMalloc((void**) &dnullElements, (divisions)*sizeof(int));
	hipMemcpy(dconstStartAddr,startaddr,(divisions + 1)*sizeof(int), hipMemcpyHostToDevice);
	hipMemcpy(dfinalStartAddr,origOffsets,(divisions + 1)*sizeof(unsigned int), hipMemcpyHostToDevice);
	hipMemcpy(dnullElements, nullElements, (divisions)*sizeof(int), hipMemcpyHostToDevice);
	#endif
	int nrElems = 2;
	while(true){
		int floatsperthread = (nrElems*4); 
		int threadsPerDiv = (int)ceil(largestSize/(float)floatsperthread); 
		int threadsNeeded = threadsPerDiv * divisions; 
		#ifdef MERGE_WG_SIZE_1
		threads.x = MERGE_WG_SIZE_1;
		#else
		threads.x = 208; 
		#endif
		grid.x = ((threadsNeeded%threads.x) == 0) ?
			threadsNeeded/threads.x : 
			(threadsNeeded/threads.x) + 1; 
		if(grid.x < 8){
			grid.x = 8; 
			threads.x = ((threadsNeeded%grid.x) == 0) ? 
				threadsNeeded / grid.x : 
				(threadsNeeded / grid.x) + 1; 
		}
		// Swap orig/result list
		float4 *tempList = d_origList; 
		d_origList = d_resultList; 
		d_resultList = tempList;
	/*	#ifdef USE_TEXTURES 
		hipBindTexture(0,tex, d_origList, channelDesc, listsize*sizeof(float));
                #ifdef MEMCPYTOSYMBOL
                hipLaunchKernel(mergeSortPass, dim3(grid), dim3(threads ), 0, 0, d_resultList, nrElems, threadsPerDiv);
                #else
                hipLaunchKernel(mergeSortPass, dim3(grid), dim3(threads ), 0, 0,d_resultList, nrElems, threadsPerDiv,startaddr);
                #endif
                #endif */

		#ifdef MEMCPYTOSYMBOL
		hipLaunchKernel(mergeSortPass, dim3(grid), dim3(threads ), 0, 0,d_origList, d_resultList, nrElems, threadsPerDiv);
		#else
		hipLaunchKernel(mergeSortPass, dim3(grid), dim3(threads ), 0, 0,d_origList, d_resultList, nrElems, threadsPerDiv,dconstStartAddr); 
		#endif
		nrElems *= 2; 
		floatsperthread = (nrElems*4); 
		if(threadsPerDiv == 1) break; 
	}
	////////////////////////////////////////////////////////////////////////////
	// Now, get rid of the NULL elements
	////////////////////////////////////////////////////////////////////////////
	#ifdef MERGE_WG_SIZE_0
	threads.x = MERGE_WG_SIZE_0;
	#else
	threads.x = 256; 
	#endif
	grid.x = ((largestSize%threads.x) == 0) ?
			largestSize/threads.x : 
			(largestSize/threads.x) + 1; 
	grid.y = divisions;
	#ifdef MEMCPYTOSYMBOL 
	hipLaunchKernel(mergepack, dim3(grid), dim3(threads ), 0, 0, (float *)d_resultList, (float *)d_origList);
	#else
	hipLaunchKernel(mergepack, dim3(grid), dim3(threads ), 0, 0, (float *)d_resultList, (float *)d_origList,dconstStartAddr,dfinalStartAddr,dnullElements);
	#endif
	free(startaddr);
	return d_origList; 
}
