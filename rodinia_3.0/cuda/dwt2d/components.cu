/* 
 * Copyright (c) 2009, Jiri Matela
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
 
#include <unistd.h>
#include <error.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <assert.h>

#include "components.h"
#include "common.h"

#define THREADS 256

/* Store 3 RGB float components */
__device__ void storeComponents(float *d_r, float *d_g, float *d_b, float r, float g, float b, int pos)
{
    d_r[pos] = (r/255.0f) - 0.5f;
    d_g[pos] = (g/255.0f) - 0.5f;
    d_b[pos] = (b/255.0f) - 0.5f;
}

/* Store 3 RGB intege components */
__device__ void storeComponents(int *d_r, int *d_g, int *d_b, int r, int g, int b, int pos)
{
    d_r[pos] = r - 128;
    d_g[pos] = g - 128;
    d_b[pos] = b - 128;
} 

/* Store float component */
__device__ void storeComponent(float *d_c, float c, int pos)
{
    d_c[pos] = (c/255.0f) - 0.5f;
}

/* Store integer component */
__device__ void storeComponent(int *d_c, int c, int pos)
{
    d_c[pos] = c - 128;
}

/* Copy img src data into three separated component buffers */
template<typename T>
__global__ void c_CopySrcToComponents(T *d_r, T *d_g, T *d_b, 
                                  unsigned char * d_src, 
                                  int pixels)
{
    int x  = threadIdx.x;
    int gX = blockDim.x*blockIdx.x;

    __shared__ unsigned char sData[THREADS*3];

    /* Copy data to shared mem by 4bytes 
       other checks are not necessary, since 
       d_src buffer is aligned to sharedDataSize */
    if ( (x*4) < THREADS*3 ) {
        float *s = (float *)d_src;
        float *d = (float *)sData;
        d[x] = s[((gX*3)>>2) + x];
    }
    __syncthreads();

    T r, g, b;

    int offset = x*3;
    r = (T)(sData[offset]);
    g = (T)(sData[offset+1]);
    b = (T)(sData[offset+2]);

    int globalOutputPosition = gX + x;
    if (globalOutputPosition < pixels) {
        storeComponents(d_r, d_g, d_b, r, g, b, globalOutputPosition);
    }
}

/* Copy img src data into three separated component buffers */
template<typename T>
__global__ void c_CopySrcToComponent(T *d_c, unsigned char * d_src, int pixels)
{
    int x  = threadIdx.x;
    int gX = blockDim.x*blockIdx.x;

    __shared__ unsigned char sData[THREADS];

    /* Copy data to shared mem by 4bytes 
       other checks are not necessary, since 
       d_src buffer is aligned to sharedDataSize */
    if ( (x*4) < THREADS) {
        float *s = (float *)d_src;
        float *d = (float *)sData;
        d[x] = s[(gX>>2) + x];
    }
    __syncthreads();

    T c;

    c = (T)(sData[x]);

    int globalOutputPosition = gX + x;
    if (globalOutputPosition < pixels) {
        storeComponent(d_c, c, globalOutputPosition);
    }
}


/* Separate compoents of 8bit RGB source image */
template<typename T>
void rgbToComponents(T *d_r, T *d_g, T *d_b, unsigned char * src, int width, int height)
{
    unsigned char * d_src;
    int pixels      = width*height;
    int alignedSize =  DIVANDRND(width*height, THREADS) * THREADS * 3; //aligned to thread block size -- THREADS

    /* Alloc d_src buffer */
    cudaMalloc((void **)&d_src, alignedSize);
    cudaCheckAsyncError("Cuda malloc")
    cudaMemset(d_src, 0, alignedSize);

    /* Copy data to device */
    cudaMemcpy(d_src, src, pixels*3, cudaMemcpyHostToDevice);
    cudaCheckError("Copy data to device")

    /* Kernel */
    dim3 threads(THREADS);
    dim3 grid(alignedSize/(THREADS*3));
    assert(alignedSize%(THREADS*3) == 0);
    c_CopySrcToComponents<<<grid, threads>>>(d_r, d_g, d_b, d_src, pixels);
    cudaCheckAsyncError("CopySrcToComponents kernel")

    /* Free Memory */
    cudaFree(d_src);
    cudaCheckAsyncError("Free memory")
}
template void rgbToComponents<float>(float *d_r, float *d_g, float *d_b, unsigned char * src, int width, int height);
template void rgbToComponents<int>(int *d_r, int *d_g, int *d_b, unsigned char * src, int width, int height);


/* Copy a 8bit source image data into a color compoment of type T */
template<typename T>
void bwToComponent(T *d_c, unsigned char * src, int width, int height)
{
    unsigned char * d_src;
    int pixels      = width*height;
    int alignedSize =  DIVANDRND(pixels, THREADS) * THREADS; //aligned to thread block size -- THREADS

    /* Alloc d_src buffer */
    cudaMalloc((void **)&d_src, alignedSize);
    cudaCheckAsyncError("Cuda malloc")
    cudaMemset(d_src, 0, alignedSize);

    /* Copy data to device */
    cudaMemcpy(d_src, src, pixels, cudaMemcpyHostToDevice);
    cudaCheckError("Copy data to device")

    /* Kernel */
    dim3 threads(THREADS);
    dim3 grid(alignedSize/(THREADS));
    assert(alignedSize%(THREADS) == 0);
    c_CopySrcToComponent<<<grid, threads>>>(d_c, d_src, pixels);
    cudaCheckAsyncError("CopySrcToComponent kernel")

    /* Free Memory */
    cudaFree(d_src);
    cudaCheckAsyncError("Free memory")
}

template void bwToComponent<float>(float *d_c, unsigned char *src, int width, int height);
template void bwToComponent<int>(int *d_c, unsigned char *src, int width, int height);
