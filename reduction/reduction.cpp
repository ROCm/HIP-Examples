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

#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cfloat>
#include <iomanip>
#include <cmath>
#include "hip/hip_runtime.h"


void check_hip_error(void)
{
hipError_t err = hipGetLastError();
if (err != hipSuccess)
{
    std::cerr
        << "Error: "
        << hipGetErrorString(err)
        << std::endl;
        exit(err);
}
}

__global__ void atomic_reduction_kernel(int *in, int* out, int ARRAYSIZE) {
    int sum=int(0);
    int idx = hipBlockIdx_x*hipBlockDim_x+hipThreadIdx_x;
    for(int i= idx;i<ARRAYSIZE;i+=hipBlockDim_x*hipGridDim_x) {
        sum+=in[i];
    }
    atomicAdd(out,sum);
}

__global__ void atomic_reduction_kernel2(int *in, int* out, int ARRAYSIZE) {
    int sum=int(0);
    int idx = hipBlockIdx_x*hipBlockDim_x+hipThreadIdx_x;
    for(int i= idx*16;i<ARRAYSIZE;i+=hipBlockDim_x*hipGridDim_x*16) {
        sum+=in[i] + in[i+1] + in[i+2] + in[i+3] +in[i+4] +in[i+5] +in[i+6] +in[i+7] +in[i+8] +in[i+9] +in[i+10]
            +in[i+11] +in[i+12] +in[i+13] +in[i+14] +in[i+15] ;
    }
    atomicAdd(out,sum);
}

__global__ void atomic_reduction_kernel3(int *in, int* out, int ARRAYSIZE) {
    int sum=int(0);
    int idx = hipBlockIdx_x*hipBlockDim_x+hipThreadIdx_x;
    for(int i= idx*4;i<ARRAYSIZE;i+=hipBlockDim_x*hipGridDim_x*4) {
        sum+=in[i] + in[i+1] + in[i+2] + in[i+3];
    }
    atomicAdd(out,sum);
}

int main(int argc, char** argv)
{
    unsigned int ARRAYSIZE = 52428800;
    if(argc<2) {
        printf("Usage: ./reduction num_of_elems\n");
        printf("using default value: %d\n",ARRAYSIZE);
    }else
        ARRAYSIZE=atoi(argv[1]);
    int N = 10;
    printf("ARRAYSIZE: %d\n", ARRAYSIZE);

    std::cout << "Array size: " << ARRAYSIZE*sizeof(int)/1024.0/1024.0 << " MB"<<std::endl;
    int* array=(int*)malloc(ARRAYSIZE*sizeof(int));
    int checksum =0;
    for(int i=0;i<ARRAYSIZE;i++) {
        array[i]=rand()%2;
        checksum+=array[i];
    }
    int *in, *out;

    // Declare timers
    std::chrono::high_resolution_clock::time_point t1, t2;


    long long size=sizeof(int)*ARRAYSIZE;

    hipMalloc(&in,size);
    hipMalloc(&out,sizeof(int));
    check_hip_error();

    hipMemcpy(in,array,ARRAYSIZE*sizeof(int),hipMemcpyHostToDevice);
    hipDeviceSynchronize();
    check_hip_error();
    // Get device properties
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, 0);


    int threads=256;
    int blocks=std::min((ARRAYSIZE+threads-1)/threads,2048u);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i=0;i<N;i++) {
        hipMemsetAsync(out,0,sizeof(int));
        hipLaunchKernelGGL(atomic_reduction_kernel, dim3(blocks), dim3(threads), 0, 0, in,out,ARRAYSIZE);
        //hipLaunchKernelGGL(atomic_reduction_kernel2, dim3(blocks), dim3(threads), 0, 0, in,out,ARRAYSIZE);
        //hipLaunchKernelGGL(atomic_reduction_kernel3, dim3(blocks), dim3(threads), 0, 0, in,out,ARRAYSIZE);

        check_hip_error();
        hipDeviceSynchronize();
        check_hip_error();
    }
    t2 = std::chrono::high_resolution_clock::now();
    double times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    float GB=(float)ARRAYSIZE*sizeof(int)*N;
    std::cout
        << "The average performance of reduction is "<< 1.0E-09 * GB/times<<" GBytes/sec"<<std::endl;

    int sum;
    hipMemcpy(&sum,out,sizeof(int),hipMemcpyDeviceToHost);
    check_hip_error();

    if(sum==checksum)
        std::cout<<"VERIFICATION: result is CORRECT"<<std::endl<<std::endl;
    else
        std::cout<<"VERIFICATION: result is INCORRECT!!"<<std::endl<<std::endl;

    hipFree(in);
    hipFree(out);
    check_hip_error();

    free(array);

}
