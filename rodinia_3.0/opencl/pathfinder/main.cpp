/***********************************************************************
 * PathFinder uses dynamic programming to find a path on a 2-D grid from
 * the bottom row to the top row with the smallest accumulated weights,
 * where each step of the path moves straight ahead or diagonally ahead.
 * It iterates row by row, each node picks a neighboring node in the
 * previous row that has the smallest accumulated weight, and adds its
 * own weight to the sum.
 *
 * This kernel uses the technique of ghost zone optimization
 ***********************************************************************/

// Other header files.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <iostream>
#include "OpenCL.h"

using namespace std;

// halo width along one direction when advancing to the next iteration
#define HALO     1
#define STR_SIZE 256
#define DEVICE   0
#define M_SEED   9
#define BENCH_PRINT
#define IN_RANGE(x, min, max)	((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

// Program variables.
int   rows, cols;
int   Ne = rows * cols;
int*  data;
int** wall;
int*  result;
int   pyramid_height;

void init(int argc, char** argv)
{
	if (argc == 4)
	{
		cols = atoi(argv[1]);
		rows = atoi(argv[2]);
		pyramid_height = atoi(argv[3]);
	}
	else
	{
		printf("Usage: dynproc row_len col_len pyramid_height\n");
		exit(0);
	}
	data = new int[rows * cols];
	wall = new int*[rows];
	for (int n = 0; n < rows; n++)
	{
		// wall[n] is set to be the nth row of the data array.
		wall[n] = data + cols * n;
	}
	result = new int[cols];

	int seed = M_SEED;
	srand(seed);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			wall[i][j] = rand() % 10;
		}
	}
#ifdef BENCH_PRINT
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			printf("%d ", wall[i][j]);
		}
		printf("\n");
	}
#endif
}

void fatal(char *s)
{
	fprintf(stderr, "error: %s\n", s);
}

int main(int argc, char** argv)
{
	init(argc, argv);
	
	// Pyramid parameters.
	int borderCols = (pyramid_height) * HALO;
	// int smallBlockCol = ?????? - (pyramid_height) * HALO * 2;
	// int blockCols = cols / smallBlockCol + ((cols % smallBlockCol == 0) ? 0 : 1);

	
	/* printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",
	   pyramid_height, cols, borderCols, NUMBER_THREADS, blockCols, smallBlockCol); */

	int size = rows * cols;

	// Create and initialize the OpenCL object.
	OpenCL cl(1);  // 1 means to display output (debugging mode).
	cl.init(1);    // 1 means to use GPU. 0 means use CPU.
	cl.gwSize(rows * cols);

	// Create and build the kernel.
	string kn = "dynproc_kernel";  // the kernel name, for future use.
	cl.createKernel(kn);

	// Allocate device memory.
	cl_mem d_gpuWall = clCreateBuffer(cl.ctxt(),
	                                  CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
	                                  sizeof(cl_int)*(size-cols),
	                                  (data + cols),
	                                  NULL);

	cl_mem d_gpuResult[2];

	d_gpuResult[0] = clCreateBuffer(cl.ctxt(),
	                                CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
	                                sizeof(cl_int)*cols,
	                                data,
	                                NULL);

	d_gpuResult[1] = clCreateBuffer(cl.ctxt(),
	                                CL_MEM_READ_WRITE,
	                                sizeof(cl_int)*cols,
	                                NULL,
	                                NULL);

	cl_int* h_outputBuffer = (cl_int*)malloc(16384*sizeof(cl_int));
	for (int i = 0; i < 16384; i++)
	{
		h_outputBuffer[i] = 0;
	}
	cl_mem d_outputBuffer = clCreateBuffer(cl.ctxt(),
	                                       CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
	                                       sizeof(cl_int)*16384,
	                                       h_outputBuffer,
	                                       NULL);

	int src = 1, final_ret = 0;
	for (int t = 0; t < rows - 1; t += pyramid_height)
	{
		int temp = src;
		src = final_ret;
		final_ret = temp;

		// Calculate this for the kernel argument...
		int arg0 = MIN(pyramid_height, rows-t-1);
		int theHalo = HALO;

		// Set the kernel arguments.
		clSetKernelArg(cl.kernel(kn), 0,  sizeof(cl_int), (void*) &arg0);
		clSetKernelArg(cl.kernel(kn), 1,  sizeof(cl_mem), (void*) &d_gpuWall);
		clSetKernelArg(cl.kernel(kn), 2,  sizeof(cl_mem), (void*) &d_gpuResult[src]);
		clSetKernelArg(cl.kernel(kn), 3,  sizeof(cl_mem), (void*) &d_gpuResult[final_ret]);
		clSetKernelArg(cl.kernel(kn), 4,  sizeof(cl_int), (void*) &cols);
		clSetKernelArg(cl.kernel(kn), 5,  sizeof(cl_int), (void*) &rows);
		clSetKernelArg(cl.kernel(kn), 6,  sizeof(cl_int), (void*) &t);
		clSetKernelArg(cl.kernel(kn), 7,  sizeof(cl_int), (void*) &borderCols);
		clSetKernelArg(cl.kernel(kn), 8,  sizeof(cl_int), (void*) &theHalo);
		clSetKernelArg(cl.kernel(kn), 9,  sizeof(cl_int) * (cl.localSize()), 0);
		clSetKernelArg(cl.kernel(kn), 10, sizeof(cl_int) * (cl.localSize()), 0);
		clSetKernelArg(cl.kernel(kn), 11, sizeof(cl_mem), (void*) &d_outputBuffer);
		cl.launch(kn);
	}

	// Copy results back to host.
	clEnqueueReadBuffer(cl.q(),                   // The command queue.
	                    d_gpuResult[final_ret],   // The result on the device.
	                    CL_TRUE,                  // Blocking? (ie. Wait at this line until read has finished?)
	                    0,                        // Offset. None in this case.
	                    sizeof(cl_int)*cols,      // Size to copy.
	                    result,                   // The pointer to the memory on the host.
	                    0,                        // Number of events in wait list. Not used.
	                    NULL,                     // Event wait list. Not used.
	                    NULL);                    // Event object for determining status. Not used.


	// Copy string buffer used for debugging from device to host.
	clEnqueueReadBuffer(cl.q(),                   // The command queue.
	                    d_outputBuffer,           // Debug buffer on the device.
	                    CL_TRUE,                  // Blocking? (ie. Wait at this line until read has finished?)
	                    0,                        // Offset. None in this case.
	                    sizeof(cl_char)*16384,    // Size to copy.
	                    h_outputBuffer,           // The pointer to the memory on the host.
	                    0,                        // Number of events in wait list. Not used.
	                    NULL,                     // Event wait list. Not used.
	                    NULL);                    // Event object for determining status. Not used.
	
	// Tack a null terminator at the end of the string.
	h_outputBuffer[16383] = '\0';
	
#ifdef BENCH_PRINT
	for (int i = 0; i < cols; i++)
		printf("%d ", data[i]);
	printf("\n");
	for (int i = 0; i < cols; i++)
		printf("%d ", result[i]);
	printf("\n");
#endif

	// Memory cleanup here.
	delete[] data;
	delete[] wall;
	delete[] result;
	
	return EXIT_SUCCESS;
}
