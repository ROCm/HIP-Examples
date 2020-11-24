/*
Copyright (c) 2020-present Advanced Micro Devices, Inc. All rights reserved.

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

// OpenMP program to print Hello World
// using C language is supported by HIP

// HIP header
#include <hip/hip_runtime.h>

//OpenMP header
#include <omp.h>

#include <stdio.h>
#include <stdlib.h>

__global__
void hip_helloworld(unsigned omp_id)
{
    printf("Hello World... from HIP thread = %u\n", omp_id);
}

int main(int argc, char* argv[])
{
    // Beginning of parallel region
    #pragma omp parallel
    {
        printf("Hello World... from OMP thread = %d\n",
               omp_get_thread_num());

        hipLaunchKernelGGL(hip_helloworld, dim3(1), dim3(1), 0, 0, omp_get_thread_num());
    }
    // Ending of parallel region

    hipLaunchKernelGGL(hip_helloworld, dim3(1), dim3(1), 0, 0, /*id=*/ 0);
    hipStreamSynchronize(0);

    printf ("PASSED!\n");
    return 0;
}
