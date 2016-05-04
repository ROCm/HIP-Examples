/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "mysecond.c"

#define nt 30
#define nx 680
#define ny 134
#define nz 450

inline __host__ __device__ int indexTo1D(int x, int y, int z){
    return x + y*ny + z*ny*nz;
}

__global__ void
rtm8(float* vsq, float* current_s, float* current_r, float* next_s, float* next_r, float* image, float* a, size_t N)
{
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned z = blockIdx.z * blockDim.z + threadIdx.z;
    float div;
    if ((4 <= x && x < (nx - 4) ) && (4 <= y && y < (ny - 4)) && (4 <= z && z < (nz - 4))){
        div =
            a[0] * current_s[indexTo1D(x,y,z)] +
            a[1] * (current_s[indexTo1D(x+1,y,z)] + current_s[indexTo1D(x-1,y,z)] +
                    current_s[indexTo1D(x,y+1,z)] + current_s[indexTo1D(x,y-1,z)] +
                    current_s[indexTo1D(x,y,z+1)] + current_s[indexTo1D(x,y,z-1)]) +
            a[2] * (current_s[indexTo1D(x+2,y,z)] + current_s[indexTo1D(x-2,y,z)] +
                    current_s[indexTo1D(x,y+2,z)] + current_s[indexTo1D(x,y-2,z)] +
                    current_s[indexTo1D(x,y,z+2)] + current_s[indexTo1D(x,y,z-2)]) +
            a[3] * (current_s[indexTo1D(x+3,y,z)] + current_s[indexTo1D(x-3,y,z)] +
                    current_s[indexTo1D(x,y+3,z)] + current_s[indexTo1D(x,y-3,z)] +
                    current_s[indexTo1D(x,y,z+3)] + current_s[indexTo1D(x,y,z-3)]) +
            a[4] * (current_s[indexTo1D(x+4,y,z)] + current_s[indexTo1D(x-4,y,z)] +
                    current_s[indexTo1D(x,y+4,z)] + current_s[indexTo1D(x,y-4,z)] +
                    current_s[indexTo1D(x,y,z+4)] + current_s[indexTo1D(x,y,z-4)]);

        next_s[indexTo1D(x,y,z)] = 2*current_s[indexTo1D(x,y,z)] - next_s[indexTo1D(x,y,z)]
            + vsq[indexTo1D(x,y,z)]*div;
        div =
            a[0] * current_r[indexTo1D(x,y,z)] +
            a[1] * (current_r[indexTo1D(x+1,y,z)] + current_r[indexTo1D(x-1,y,z)] +
                    current_r[indexTo1D(x,y+1,z)] + current_r[indexTo1D(x,y-1,z)] +
                    current_r[indexTo1D(x,y,z+1)] + current_r[indexTo1D(x,y,z-1)]) +
            a[2] * (current_r[indexTo1D(x+2,y,z)] + current_r[indexTo1D(x-2,y,z)] +
                    current_r[indexTo1D(x,y+2,z)] + current_r[indexTo1D(x,y-2,z)] +
                    current_r[indexTo1D(x,y,z+2)] + current_r[indexTo1D(x,y,z-2)]) +
            a[3] * (current_r[indexTo1D(x+3,y,z)] + current_r[indexTo1D(x-3,y,z)] +
                    current_r[indexTo1D(x,y+3,z)] + current_r[indexTo1D(x,y-3,z)] +
                    current_r[indexTo1D(x,y,z+3)] + current_r[indexTo1D(x,y,z-3)]) +
            a[4] * (current_r[indexTo1D(x+4,y,z)] + current_r[indexTo1D(x-4,y,z)] +
                    current_r[indexTo1D(x,y+4,z)] + current_r[indexTo1D(x,y-4,z)] +
                    current_r[indexTo1D(x,y,z+4)] + current_r[indexTo1D(x,y,z-4)]);

        next_r[indexTo1D(x,y,z)] = 2 * current_r[indexTo1D(x,y,z)]
            - next_r[indexTo1D(x,y,z)] + vsq[indexTo1D(x,y,z)] * div;

        image[indexTo1D(x,y,z)] = next_s[indexTo1D(x,y,z)] * next_r[indexTo1D(x,y,z)];
    }
}

// Code to check CUDA errors
void check_cuda_error(void)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr
            << "Error: "
            << cudaGetErrorString(err)
            << std::endl;
            exit(err);
    }
}

int main(){
    const int ArraySize = nx + nx*ny + nx*ny*nz;

    float* next_s = (float*)malloc(ArraySize * sizeof(float));
    float* current_s = (float*)malloc(ArraySize * sizeof(float));
    float* next_r = (float*)malloc(ArraySize * sizeof(float));
    float* current_r = (float*)malloc(ArraySize * sizeof(float));
    float* vsq = (float*)malloc(ArraySize * sizeof(float));
    float* image = (float*)malloc(ArraySize * sizeof(float));

    float a[5];

    double pts, t0, t1, dt, flops, pt_rate, flop_rate, speedup, memory;

    memory = nx*ny*nz*4*6;
    pts = nt;
	pts = pts*(nx-8)*(ny-8)*(nz-8);
	flops = 67*pts;
    printf("memory (MB) = %f\n", memory/1e6);
    printf("pts (billions) = %f\n", pts/1e9);
    printf("Tflops = %f\n", flops/1e12);

// Initialization of matrix
	a[0] = -1./560.;
	a[1] = 8./315;
	a[2] = -0.2;
	a[3] = 1.6;
	a[4] = -1435./504.;

    for (int z = 0; z < nz; z++) {
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                vsq[indexTo1D(x,y,z)] = 1.0;
                next_s[indexTo1D(x,y,z)] = 0;
                current_s[indexTo1D(x,y,z)] = 0;
                next_r[indexTo1D(x,y,z)] = 0;
                current_r[indexTo1D(x,y,z)] = 0;
                image[indexTo1D(x,y,z)] = 0;
            }
        }
    }

    t0 = mysecond();
    //allocate and copy matrix to device
    float* vsq_d;
    float* next_s_d;
    float* current_s_d;
    float* next_r_d;
    float* current_r_d;
    float* image_d;
    float* a_d;

	cudaMalloc(&vsq_d, ArraySize * sizeof(float));
	cudaMalloc(&next_s_d, ArraySize * sizeof(float));
	cudaMalloc(&current_s_d, ArraySize * sizeof(float));
	cudaMalloc(&next_r_d, ArraySize * sizeof(float));
	cudaMalloc(&current_r_d, ArraySize * sizeof(float));
	cudaMalloc(&image_d, ArraySize * sizeof(float));
	cudaMalloc(&a_d, 5 * sizeof(float));
    check_cuda_error();
    cudaMemcpy(vsq_d, vsq, ArraySize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(next_s_d, next_s, ArraySize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(current_s_d, current_s, ArraySize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(next_r_d, next_r, ArraySize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(current_r_d, current_r, ArraySize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(image_d, image, ArraySize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(a_d, a, 5 * sizeof(float), cudaMemcpyHostToDevice);
    check_cuda_error();
    // Make sure the copies are finished
    cudaDeviceSynchronize();
    check_cuda_error();

    int gridSize = 256*256;
    int groupSize = 256;


    for (int t = 0; t < nt; t++) {
        //Launch the HIP kernel
        rtm8<<<dim3(gridSize), dim3(groupSize)>>>((float*)vsq_d, (float*)current_s_d, (float*)next_s_d, (float*)current_r_d,(float*)next_r_d, (float*)image_d, (float*)a_d, ArraySize);
    }
    //copy back image value
    cudaMemcpy(image, image_d,ArraySize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    t1 = mysecond();

    dt = t1 - t0;
    pt_rate = pts/dt;
    flop_rate = flops/dt;
    speedup = 2*pow(10, 9)/3/pt_rate;
    printf("dt = %f\n", dt);
    printf("pt_rate (millions/sec) = %f\n", pt_rate/1e6);
    printf("flop_rate (Gflops) = %f\n", flop_rate/1e9);
    printf("speedup = %f\n", speedup);

    //release arrays
    free(vsq);
    free(next_s);
    free(current_s);
    free(next_r);
    free(current_r);
    free(image);
    return 0;
}

