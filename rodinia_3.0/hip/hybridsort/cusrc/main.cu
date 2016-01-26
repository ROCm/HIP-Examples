#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "helper_cuda.h"
#include "helper_timer.h"
#include <iostream>
#include "bucketsort.cuh"
#include "mergesort.cuh"
using namespace std; 

////////////////////////////////////////////////////////////////////////////////
// Size of the testset (Bitwise shift of 1 over 22 places)
////////////////////////////////////////////////////////////////////////////////
#define SIZE	(1 << 22)
////////////////////////////////////////////////////////////////////////////////
// Number of tests to average over
////////////////////////////////////////////////////////////////////////////////
#define TEST	4
////////////////////////////////////////////////////////////////////////////////
// The timers for the different parts of the algo
////////////////////////////////////////////////////////////////////////////////
StopWatchInterface  *uploadTimer, *downloadTimer, *bucketTimer, 
			 *mergeTimer, *totalTimer, *cpuTimer; 
////////////////////////////////////////////////////////////////////////////////
// Compare method for CPU sort
////////////////////////////////////////////////////////////////////////////////
inline int compare(const void *a, const void *b) {
	if(*((float *)a) < *((float *)b)) return -1; 
	else if(*((float *)a) > *((float *)b)) return 1; 
	else return 0; 
}
////////////////////////////////////////////////////////////////////////////////
// Forward declaration
////////////////////////////////////////////////////////////////////////////////
void cudaSort(float *origList, float minimum, float maximum,
			  float *resultList, int numElements);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv)
{ 

  // Create timers for each sort
    sdkCreateTimer(&uploadTimer);
    sdkCreateTimer(&downloadTimer);
    sdkCreateTimer(&bucketTimer);
    sdkCreateTimer(&mergeTimer);
    sdkCreateTimer(&totalTimer);
    sdkCreateTimer(&cpuTimer);
	int numElements = 0;
    // Number of elements in the test bed
       	if(strcmp(argv[1],"r") ==0) {
	numElements = SIZE; 
	}
	else {
		FILE *fp;
	fp = fopen(argv[1],"r");
	if(fp == NULL) {
	      cout << "Error reading file" << endl;
	      exit(EXIT_FAILURE);
	      }
	int count = 0;
	float c;

	while(fscanf(fp,"%f",&c) != EOF) {
	 count++;
}
	fclose(fp);

	numElements = count;
}
	cout << "Sorting list of " << numElements << " floats\n";
	// Generate random data
	// Memory space the list of random floats will take up
	int mem_size = numElements * sizeof(float); 
	// Allocate enough for the input list
	float *cpu_idata = (float *)malloc(mem_size);
	// Allocate enough for the output list on the cpu side
	float *cpu_odata = (float *)malloc(mem_size);
	// Allocate enough memory for the output list on the gpu side
	float *gpu_odata = (float *)malloc(mem_size);

	float datamin = FLT_MAX; 
	float datamax = -FLT_MAX; 
	if(strcmp(argv[1],"r")==0) {
	for (int i = 0; i < numElements; i++) {
	// Generate random floats between 0 and 1 for the input data
		cpu_idata[i] = ((float) rand() / RAND_MAX); 
	//Compare data at index to data minimum, if less than current minimum, set that element as new minimum
		datamin = min(cpu_idata[i], datamin);
	//Same as above but for maximum
		datamax = max(cpu_idata[i], datamax);
	}

}	else {
	FILE *fp;
	fp = fopen(argv[1],"r");
	for(int i = 0; i < numElements; i++) {
	fscanf(fp,"%f",&cpu_idata[i]);
	datamin = min(cpu_idata[i], datamin);
	datamax = max(cpu_idata[i],datamax);
	}
	}

	cout << "Sorting on GPU..." << flush; 
	// GPU Sort
	for (int i = 0; i < TEST; i++) 
		cudaSort(cpu_idata, datamin, datamax, gpu_odata, numElements);		
	cout << "done.\n";
#ifdef VERIFY
	cout << "Sorting on CPU..." << flush; 
	// CPU Sort
	memcpy(cpu_odata, cpu_idata, mem_size); 		
	sdkStartTimer(&cpuTimer); 
		qsort(cpu_odata, numElements, sizeof(float), compare);
	sdkStopTimer(&cpuTimer); 
	cout << "done.\n";
	cout << "Checking result..." << flush; 
	// Result checking
	int count = 0; 
	for(int i = 0; i < numElements; i++)
		if(cpu_odata[i] != gpu_odata[i])
		{
			printf("Sort missmatch on element %d: \n", i); 
			printf("CPU = %f : GPU = %f\n", cpu_odata[i], gpu_odata[i]); 
			count++; 
			break; 
		}
	if(count == 0) cout << "PASSED.\n";
	else cout << "FAILED.\n";
#endif
	// Timer report
	printf("GPU iterations: %d\n", TEST); 
#ifdef TIMER
#ifdef VERIFY
	printf("Average CPU execution time: %f ms\n", sdkGetTimerValue(&cpuTimer));
#endif
	printf("Average GPU execution time: %f ms\n", sdkGetTimerValue(&totalTimer) / TEST);
	printf("    - Upload		: %f ms\n", sdkGetTimerValue(&uploadTimer) / TEST);
	printf("    - Download		: %f ms\n", sdkGetTimerValue(&downloadTimer) / TEST);
	printf("    - Bucket sort	: %f ms\n", sdkGetTimerValue(&bucketTimer) / TEST);
	printf("    - Merge sort	: %f ms\n", sdkGetTimerValue(&mergeTimer) / TEST);
#endif

#ifdef OUTPUT
    FILE *tp;
    const char filename2[]="./hybridoutput.txt";
    tp = fopen(filename2,"w");
    for(int i = 0; i < numElements; i++) {
        fprintf(tp,"%f ",cpu_idata[i]);
    }
    
    fclose(tp);
#endif
	
	// Release memory
    sdkDeleteTimer(&uploadTimer);
    sdkDeleteTimer(&downloadTimer);
    sdkDeleteTimer(&bucketTimer);
    sdkDeleteTimer(&mergeTimer);
    sdkDeleteTimer(&totalTimer);
    sdkDeleteTimer(&cpuTimer);
	free(cpu_idata); free(cpu_odata); free(gpu_odata); 
}


void cudaSort(float *origList, float minimum, float maximum,
			  float *resultList, int numElements)
{
	// Initialization and upload data
	float *d_input  = NULL; 
	float *d_output = NULL; 
	int mem_size = (numElements + DIVISIONS * 4) * sizeof(float); 
	sdkStartTimer(&uploadTimer);
	{
		cudaMalloc((void**) &d_input, mem_size);
		cudaMalloc((void**) &d_output, mem_size);
		cudaMemcpy((void *) d_input, (void *)origList, numElements * sizeof(float),
				   cudaMemcpyHostToDevice);
		init_bucketsort(numElements); 
	}
	sdkStopTimer(&uploadTimer); 

	sdkStartTimer(&totalTimer); 

	// Bucketsort the list
	sdkStartTimer(&bucketTimer); 
		int *sizes = (int*) malloc(DIVISIONS * sizeof(int)); 
		int *nullElements = (int*) malloc(DIVISIONS * sizeof(int));  
		unsigned int *origOffsets = (unsigned int *) malloc((DIVISIONS + 1) * sizeof(int)); 
		bucketSort(d_input, d_output, numElements, sizes, nullElements, 
				   minimum, maximum, origOffsets); 
	sdkStopTimer(&bucketTimer); 

	// Mergesort the result
	sdkStartTimer(&mergeTimer); 
		float4 *d_origList = (float4*) d_output, 
		*d_resultList = (float4*) d_input;
		int newlistsize = 0; 
	
		for(int i = 0; i < DIVISIONS; i++)
			newlistsize += sizes[i] * 4;
		
		float4 *mergeresult = runMergeSort(	newlistsize, DIVISIONS, d_origList, d_resultList, 
			sizes, nullElements, origOffsets); //d_origList; 
		cudaThreadSynchronize(); 
	sdkStopTimer(&mergeTimer); 
	sdkStopTimer(&totalTimer); 

	// Download result
	sdkStartTimer(&downloadTimer); 
		checkCudaErrors(	cudaMemcpy((void *) resultList, 
				(void *)mergeresult, numElements * sizeof(float), cudaMemcpyDeviceToHost) );
	sdkStopTimer(&downloadTimer); 

	// Clean up
	finish_bucketsort(); 
	cudaFree(d_input); cudaFree(d_output); 
	free(nullElements); free(sizes); 
}
