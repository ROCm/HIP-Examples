#include "hip/hip_runtime.h"
#include "find_ellipse_kernel.h"
// #include <cutil.h>
#include <stdio.h>

// The number of sample points in each ellipse (stencil)
#define NPOINTS 150
// The maximum radius of a sample ellipse
#define MAX_RAD 20
// The total number of sample ellipses
#define NCIRCLES 7
// The size of the structuring element used in dilation
#define STREL_SIZE (12 * 2 + 1)


// Matrix used to store the maximal GICOV score at each pixels
// Produced by the GICOV kernel and consumed by the dilation kernel
float *device_gicov;

#ifdef MEMCPYTOSYMBOL 
// Constant device arrays holding the stencil parameters used by the GICOV kernel
__constant__ float c_sin_angle[NPOINTS];
__constant__ float c_cos_angle[NPOINTS];
__constant__ int c_tX[NCIRCLES * NPOINTS];
__constant__ int c_tY[NCIRCLES * NPOINTS];
#else
#define PI 3.14159
#define RADIUS 10
#define MIN_RAD RADIUS - 2
#endif
// Texture references to the gradient matrices used by the GICOV kernel
texture<float, 1, hipReadModeElementType> t_grad_x;
texture<float, 1, hipReadModeElementType> t_grad_y;

// Kernel to find the maximal GICOV value at each pixel of a
//  video frame, based on the input x- and y-gradient matrices
__global__ void GICOV_kernel(hipLaunchParm lp, int grad_m, float *gicov
#ifdef MEMCPYTOSYMBOL
)
#else
, float *c_sin_angle, float *c_cos_angle, int *c_tX, int *c_tY) 
#endif
{
	int i, j, k, n, x, y;
	
	// Determine this thread's pixel
	i = hipBlockIdx_x + MAX_RAD + 2;
	j = hipThreadIdx_x + MAX_RAD + 2;

	// Initialize the maximal GICOV score to 0
	float max_GICOV = 0.f;

	// Iterate across each stencil
	for (k = 0; k < NCIRCLES; k++) {
		// Variables used to compute the mean and variance
		//  of the gradients along the current stencil
		float sum = 0.f, M2 = 0.f, mean = 0.f;		
		
		// Iterate across each sample point in the current stencil
		for (n = 0; n < NPOINTS; n++) {
			// Determine the x- and y-coordinates of the current sample point
			y = j + c_tY[(k * NPOINTS) + n];
			x = i + c_tX[(k * NPOINTS) + n];
			
			// Compute the combined gradient value at the current sample point
			int addr = x * grad_m + y;
			float p = tex1Dfetch(t_grad_x,addr) * c_cos_angle[n] + 
					  tex1Dfetch(t_grad_y,addr) * c_sin_angle[n];
			
			// Update the running total
			sum += p;
			
			// Partially compute the variance
			float delta = p - mean;
			mean = mean + (delta / (float) (n + 1));
			M2 = M2 + (delta * (p - mean));
		}
		
		// Finish computing the mean
		mean = sum / ((float) NPOINTS);
		
		// Finish computing the variance
		float var = M2 / ((float) (NPOINTS - 1));
		
		// Keep track of the maximal GICOV value seen so far
		if (((mean * mean) / var) > max_GICOV) max_GICOV = (mean * mean) / var;
	}
	
	// Store the maximal GICOV value
	gicov[(i * grad_m) + j] = max_GICOV;
}

#ifndef MEMCPYTOSYMBOL
void compute_copy_constants(float *c_sin_angle, float *c_cos_angle, int *c_tX, int *c_tY)
{
        int n, k;
        // Compute the sine and cosine of the angle to each point in each sample circle
        //  (which are the same across all sample circles)
        float host_sin_angle[NPOINTS], host_cos_angle[NPOINTS], theta[NPOINTS];
        for(n = 0; n < NPOINTS; n++) {
                theta[n] = (((double) n) * 2.0 * PI) / ((double) NPOINTS);
                host_sin_angle[n] = sin(theta[n]);
                host_cos_angle[n] = cos(theta[n]);
        }

        // Compute the (x,y) pixel offsets of each sample point in each sample circle
        int host_tX[NCIRCLES * NPOINTS], host_tY[NCIRCLES * NPOINTS];
        for (k = 0; k < NCIRCLES; k++) {
                double rad = (double) (MIN_RAD + (2 * k));
                for (n = 0; n < NPOINTS; n++) {
                        host_tX[(k * NPOINTS) + n] = (int)(cos(theta[n]) * rad);
                        host_tY[(k * NPOINTS) + n] = (int)(sin(theta[n]) * rad);
                }
        }

	hipMemcpy(c_sin_angle, host_sin_angle, (sizeof(float) * NPOINTS), hipMemcpyHostToDevice);
	hipMemcpy(c_cos_angle, host_cos_angle, (sizeof(float) * NPOINTS), hipMemcpyHostToDevice);
	hipMemcpy(c_tX, host_tX, (sizeof(int) * NCIRCLES *  NPOINTS), hipMemcpyHostToDevice);
	hipMemcpy(c_tY, host_tY, (sizeof(int) * NCIRCLES *  NPOINTS), hipMemcpyHostToDevice);
}
#endif
// Sets up and invokes the GICOV kernel and returns its output
float *GICOV_CUDA(int grad_m, int grad_n, float *host_grad_x, float *host_grad_y) {

	int MaxR = MAX_RAD + 2;
#ifndef MEMCPYTOSYMBOL
	float *c_sin_angle;//[NPOINTS];
	float *c_cos_angle;//[NPOINTS];
	int *c_tX;//[NCIRCLES * NPOINTS];
	int *c_tY;//[NCIRCLES * NPOINTS];
#endif

	// Allocate device memory
	unsigned int grad_mem_size = sizeof(float) * grad_m * grad_n;
	float *device_grad_x, *device_grad_y;
	hipMalloc((void**) &device_grad_x, grad_mem_size);
	hipMalloc((void**) &device_grad_y, grad_mem_size);
#ifndef MEMCPYTOSYMBOL
	hipMalloc((void **) &c_sin_angle, (sizeof(float) * NPOINTS));
	hipMalloc((void **) &c_cos_angle, (sizeof(float) * NPOINTS));
	hipMalloc((void **) &c_tX, (sizeof(int) * NCIRCLES *  NPOINTS));
	hipMalloc((void **) &c_tY, (sizeof(int) * NCIRCLES *  NPOINTS));

	compute_copy_constants(c_sin_angle, c_cos_angle, c_tX, c_tY);   
#endif
	// Copy the input gradients to the device
	hipMemcpy(device_grad_x, host_grad_x, grad_mem_size, hipMemcpyHostToDevice);
	hipMemcpy(device_grad_y, host_grad_y, grad_mem_size, hipMemcpyHostToDevice);
    
	// Bind the device arrays to texture references
    hipBindTexture(0, t_grad_x, device_grad_x, grad_mem_size);
    hipBindTexture(0, t_grad_y, device_grad_y, grad_mem_size);

	// Allocate & initialize device memory for result
	// (some elements are not assigned values in the kernel)
	hipMalloc((void**) &device_gicov, grad_mem_size);
	hipMemset(device_gicov, 0, grad_mem_size);

	// Setup execution parameters
	int num_blocks = grad_n - (2 * MaxR);
	int threads_per_block = grad_m - (2 * MaxR);
    
	// Execute the GICOV kernel
	hipLaunchKernel(GICOV_kernel, dim3(num_blocks), dim3(threads_per_block ), 0, 0, grad_m, device_gicov
#ifdef MEMCPYTOSYMBOL
);
#else
, c_sin_angle, c_cos_angle, c_tX, c_tY); 
#endif	
	// Check for kernel errors
	hipDeviceSynchronize();
	hipError_t error = hipGetLastError();
	if (error != hipSuccess) {
		printf("GICOV kernel error: %s\n", hipGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Copy the result to the host
	float *host_gicov = (float *) malloc(grad_mem_size);
	hipMemcpy(host_gicov, device_gicov, grad_mem_size, hipMemcpyDeviceToHost);

	// Cleanup memory
	hipUnbindTexture(&t_grad_x);
	hipUnbindTexture(&t_grad_y);
	hipFree(device_grad_x);
	hipFree(device_grad_y);
#ifndef MEMCPYTOSYMBOL
	hipFree(c_sin_angle);
	hipFree(c_cos_angle);
	hipFree(c_tX);
	hipFree(c_tY);
#endif
	return host_gicov;
}


// Constant device array holding the structuring element used by the dilation kernel
#ifdef MEMCPYTOSYMBOL
__constant__ float c_strel[STREL_SIZE * STREL_SIZE];
#endif

// Texture reference to the GICOV matrix used by the dilation kernel
texture<float, 1, hipReadModeElementType> t_img;

// Kernel to compute the dilation of the GICOV matrix produced by the GICOV kernel
// Each element (i, j) of the output matrix is set equal to the maximal value in
//  the neighborhood surrounding element (i, j) in the input matrix
// Here the neighborhood is defined by the structuring element (c_strel)
__global__ void dilate_kernel(grid_launch_parm lp, int img_m, int img_n, int strel_m, int strel_n, float *dilated
#ifdef MEMCPYTOSYMBOL
)
#else
, float *c_strel)
#endif
{	
	// Find the center of the structuring element
	int el_center_i = strel_m / 2;
	int el_center_j = strel_n / 2;

	// Determine this thread's location in the matrix
	int thread_id = (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;
	int i = thread_id % img_m;
	int j = thread_id / img_m;

	// Initialize the maximum GICOV score seen so far to zero
	float max = 0.0;

	// Iterate across the structuring element in one dimension
	int el_i, el_j, x, y;
	for(el_i = 0; el_i < strel_m; el_i++) {
		y = i - el_center_i + el_i;
		// Make sure we have not gone off the edge of the matrix
		if( (y >= 0) && (y < img_m) ) {
			// Iterate across the structuring element in the other dimension
			for(el_j = 0; el_j < strel_n; el_j++) {
				x = j - el_center_j + el_j;
				// Make sure we have not gone off the edge of the matrix
				//  and that the current structuring element value is not zero
				if( (x >= 0) &&
					(x < img_n) &&
					(c_strel[(el_i * strel_n) + el_j] != 0) ) {
						// Determine if this is maximal value seen so far
						int addr = (x * img_m) + y;
						float temp = tex1Dfetch(t_img, addr);
						if (temp > max) max = temp;
				}
			}
		}
	}
	
	// Store the maximum value found
	dilated[(i * img_n) + j] = max;
}
#ifndef MEMCPYTOSYMBOL
// Returns a circular structuring element of the specified radius
float * structuring_element(int radius) {
        int m = radius*2+1;
        int n = radius*2+1;
        float *result = (float *) malloc(sizeof(float) * m * n);

        int i, j;
        for (i = 0; i < m; i++) {
                for (j = 0; j < n; j++) {
                        if (sqrt((float)((i-radius)*(i-radius)+(j-radius)*(j-radius))) <= radius)
                                result[(i * m) + j] = 1.0;
                        else
                                result[(i * m) + j] = 0.0;
                }
        }

        return result;
}

#endif

// Sets up and invokes the dilation kernel and returns its output
float *dilate_CUDA(int max_gicov_m, int max_gicov_n, int strel_m, int strel_n) {
	// Allocate device memory for result
	unsigned int max_gicov_mem_size = sizeof(float) * max_gicov_m * max_gicov_n;
	float* device_img_dilated;
	hipMalloc( (void**) &device_img_dilated, max_gicov_mem_size);
	
#ifndef MEMCPYTOSYMBOL
	float *c_strel;
	hipMalloc( (void**) &c_strel, (sizeof(float) * STREL_SIZE * STREL_SIZE));
	float *host_strel = structuring_element(12);
	hipMemcpy(host_strel, c_strel, (sizeof(float) * STREL_SIZE * STREL_SIZE), hipMemcpyHostToDevice);
	free(host_strel);
#endif
	// Bind the input matrix of GICOV values to a texture reference
	hipBindTexture(0, t_img, device_gicov, max_gicov_mem_size);
    
	// Setup execution parameters
	int num_threads = max_gicov_m * max_gicov_n;
	int threads_per_block = 176;
	int num_blocks = (int) (((float) num_threads / (float) threads_per_block) + 0.5);

	// Execute the dilation kernel
	hipLaunchKernel(dilate_kernel, dim3(num_blocks), dim3(threads_per_block ), 0, 0, max_gicov_m, max_gicov_n, strel_m, strel_n, device_img_dilated
#ifdef MEMCPYTOSYMBOL
);
#else
, c_strel);
#endif
	
	// Check for kernel errors
	hipDeviceSynchronize();
	hipError_t error = hipGetLastError();
	if (error != hipSuccess) {
		printf("Dilation kernel error: %s\n", hipGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Copy the result to the host
	float *host_img_dilated = (float*) malloc(max_gicov_mem_size);
	hipMemcpy(host_img_dilated, device_img_dilated, max_gicov_mem_size, hipMemcpyDeviceToHost);

	// Cleanup memory
	hipUnbindTexture(&t_img);
	hipFree(device_gicov);
	hipFree(device_img_dilated);
#ifndef MEMCPYTOSYMBOL
	hipFree(c_strel);
#endif
	return host_img_dilated;
}


// Chooses the most appropriate GPU on which to execute
void select_device() {
	// Figure out how many devices exist
	int num_devices, device;
	hipGetDeviceCount(&num_devices);
	// Choose the device with the largest number of multiprocessors
	if (num_devices > 0) {
		int max_multiprocessors = 0, max_device = -1;
		for (device = 0; device < num_devices; device++) {
			hipDeviceProp_t properties;
			hipGetDeviceProperties(&properties, device);
			if (max_multiprocessors < properties.multiProcessorCount) {
				max_multiprocessors = properties.multiProcessorCount;
				max_device = device;
			}
		}
		hipSetDevice(max_device);
	}
	
	// The following is to remove the API initialization overhead from the runtime measurements
	hipFree(0);
}


#ifdef MEMCPYTOSYMBOL
// Transfers pre-computed constants used by the two kernels to the GPU
void transfer_constants(float *host_sin_angle, float *host_cos_angle, int *host_tX, int *host_tY, int strel_m, int strel_n, float *host_strel) {

	// Compute the sizes of the matrices
	unsigned int angle_mem_size = sizeof(float) * NPOINTS;
	unsigned int t_mem_size = sizeof(int) * NCIRCLES * NPOINTS;
	unsigned int strel_mem_size = sizeof(float) * strel_m * strel_n;

	// Copy the matrices from host memory to device constant memory
	hipMemcpyToSymbol((const void *)(c_sin_angle), (const void *)host_sin_angle, angle_mem_size, 0, hipMemcpyHostToDevice);
	hipMemcpyToSymbol((const void *)(c_cos_angle), (const void *)host_cos_angle, angle_mem_size, 0, hipMemcpyHostToDevice);
	hipMemcpyToSymbol((const void *)(c_tX), (const void *)host_tX, t_mem_size, 0, hipMemcpyHostToDevice);
	hipMemcpyToSymbol((const void *)(c_tY), (const void *)host_tY, t_mem_size, 0, hipMemcpyHostToDevice);
	hipMemcpyToSymbol((const void *)(c_strel), (const void *)host_strel, strel_mem_size, 0, hipMemcpyHostToDevice);
}
#endif
