#include "hip/hip_runtime.h"
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

#include <stdio.h>
#include <fcntl.h>
#include <assert.h>
#include <errno.h>
#include <sys/time.h>
#include <unistd.h>
#include <error.h>
#include "dwt_cuda/dwt.h"
#include "dwt_cuda/common.h"
#include "dwt.h"
#include "common.h"

inline void fdwt(float *in, float *out, int width, int height, int levels)
{
        dwt_cuda::fdwt97(in, out, width, height, levels);
}

inline void fdwt(int *in, int *out, int width, int height, int levels)
{
        dwt_cuda::fdwt53(in, out, width, height, levels);
}

inline void rdwt(float *in, float *out, int width, int height, int levels)
{
        dwt_cuda::rdwt97(in, out, width, height, levels);
}

inline void rdwt(int *in, int *out, int width, int height, int levels)
{
        dwt_cuda::rdwt53(in, out, width, height, levels);
}

template<typename T>
int nStage2dDWT(T * in, T * out, T * backup, int pixWidth, int pixHeight, int stages, bool forward)
{
    printf("\n*** %d stages of 2D forward DWT:\n", stages);
    
    /* create backup of input, because each test iteration overwrites it */
    const int size = pixHeight * pixWidth * sizeof(T);
    hipMemcpy(backup, in, size, hipMemcpyDeviceToDevice);
    
    /* Measure time of individual levels. */
    if(forward)
        fdwt(in, out, pixWidth, pixHeight, stages);
    else
        rdwt(in, out, pixWidth, pixHeight, stages);
    
    return 0;
}
template int nStage2dDWT<float>(float*, float*, float*, int, int, int, bool);
template int nStage2dDWT<int>(int*, int*, int*, int, int, int, bool);

void samplesToChar(unsigned char * dst, float * src, int samplesNum)
{
    int i;

    for(i = 0; i < samplesNum; i++) {
        float r = (src[i]+0.5f) * 255;
        if (r > 255) r = 255; 
        if (r < 0)   r = 0; 
        dst[i] = (unsigned char)r;
    }
}

void samplesToChar(unsigned char * dst, int * src, int samplesNum)
{
    int i;

    for(i = 0; i < samplesNum; i++) {
        int r = src[i]+128;
        if (r > 255) r = 255;
        if (r < 0)   r = 0; 
        dst[i] = (unsigned char)r;
    }
}

///* Write output linear orderd*/
template<typename T>
int writeLinear(T *component_cuda, int pixWidth, int pixHeight,
                const char * filename, const char * suffix)
{
    unsigned char * result;
    T *gpu_output;
    int i;
    int size;
    int samplesNum = pixWidth*pixHeight;

    size = samplesNum*sizeof(T);
    hipHostMalloc((void **)&gpu_output, size);
    memset(gpu_output, 0, size);
    result = (unsigned char *)malloc(samplesNum);
    hipMemcpy(gpu_output, component_cuda, size, hipMemcpyDeviceToHost);

    /* T to char */
    samplesToChar(result, gpu_output, samplesNum);

    /* Write component */
    char outfile[strlen(filename)+strlen(suffix)];
    strcpy(outfile, filename);
    strcpy(outfile+strlen(filename), suffix);
    i = open(outfile, O_CREAT|O_WRONLY, 0644);
    if (i == -1) {
        error(0,errno,"cannot access %s", outfile);
        return -1;
    }
    printf("\nWriting to %s (%d x %d)\n", outfile, pixWidth, pixHeight);
    ssize_t x ;
    x = write(i, result, samplesNum);
    close(i);

    /* Clean up */
    hipHostFree(gpu_output);
    free(result);
    if(x == 0) return 1;
    return 0;
}
template int writeLinear<float>(float *component_cuda, int pixWidth, int pixHeight, const char * filename, const char * suffix); 
template int writeLinear<int>(int *component_cuda, int pixWidth, int pixHeight, const char * filename, const char * suffix); 

/* Write output visual ordered */
template<typename T>
int writeNStage2DDWT(T *component_cuda, int pixWidth, int pixHeight, 
                     int stages, const char * filename, const char * suffix) 
{
    struct band {
        int dimX; 
        int dimY;
    };
    struct dimensions {
        struct band LL;
        struct band HL;
        struct band LH;
        struct band HH;
    };

    unsigned char * result;
    T *src, *dst;
    int i,s;
    int size;
    int offset;
    int yOffset;
    int samplesNum = pixWidth*pixHeight;
    struct dimensions * bandDims;

    bandDims = (struct dimensions *)malloc(stages * sizeof(struct dimensions));

    bandDims[0].LL.dimX = DIVANDRND(pixWidth,2);
    bandDims[0].LL.dimY = DIVANDRND(pixHeight,2);
    bandDims[0].HL.dimX = pixWidth - bandDims[0].LL.dimX;
    bandDims[0].HL.dimY = bandDims[0].LL.dimY;
    bandDims[0].LH.dimX = bandDims[0].LL.dimX;
    bandDims[0].LH.dimY = pixHeight - bandDims[0].LL.dimY;
    bandDims[0].HH.dimX = bandDims[0].HL.dimX;
    bandDims[0].HH.dimY = bandDims[0].LH.dimY;

    for (i = 1; i < stages; i++) {
        bandDims[i].LL.dimX = DIVANDRND(bandDims[i-1].LL.dimX,2);
        bandDims[i].LL.dimY = DIVANDRND(bandDims[i-1].LL.dimY,2);
        bandDims[i].HL.dimX = bandDims[i-1].LL.dimX - bandDims[i].LL.dimX;
        bandDims[i].HL.dimY = bandDims[i].LL.dimY;
        bandDims[i].LH.dimX = bandDims[i].LL.dimX;
        bandDims[i].LH.dimY = bandDims[i-1].LL.dimY - bandDims[i].LL.dimY;
        bandDims[i].HH.dimX = bandDims[i].HL.dimX;
        bandDims[i].HH.dimY = bandDims[i].LH.dimY;
    }

    size = samplesNum*sizeof(T);
    hipHostMalloc((void **)&src, size);
    dst = (T*)malloc(size);
    memset(src, 0, size);
    memset(dst, 0, size);
    result = (unsigned char *)malloc(samplesNum);
    hipMemcpy(src, component_cuda, size, hipMemcpyDeviceToHost);

    // LL Band
    size = bandDims[stages-1].LL.dimX * sizeof(T);
    for (i = 0; i < bandDims[stages-1].LL.dimY; i++) {
        memcpy(dst+i*pixWidth, src+i*bandDims[stages-1].LL.dimX, size);
    }

    for (s = stages - 1; s >= 0; s--) {
        // HL Band
        size = bandDims[s].HL.dimX * sizeof(T);
        offset = bandDims[s].LL.dimX * bandDims[s].LL.dimY;
        for (i = 0; i < bandDims[s].HL.dimY; i++) {
            memcpy(dst+i*pixWidth+bandDims[s].LL.dimX,
                src+offset+i*bandDims[s].HL.dimX, 
                size);
        }

        // LH band
        size = bandDims[s].LH.dimX * sizeof(T);
        offset += bandDims[s].HL.dimX * bandDims[s].HL.dimY;
        yOffset = bandDims[s].LL.dimY;
        for (i = 0; i < bandDims[s].HL.dimY; i++) {
            memcpy(dst+(yOffset+i)*pixWidth,
                src+offset+i*bandDims[s].LH.dimX, 
                size);
        }

        //HH band
        size = bandDims[s].HH.dimX * sizeof(T);
        offset += bandDims[s].LH.dimX * bandDims[s].LH.dimY;
        yOffset = bandDims[s].HL.dimY;
        for (i = 0; i < bandDims[s].HH.dimY; i++) {
            memcpy(dst+(yOffset+i)*pixWidth+bandDims[s].LH.dimX,
                src+offset+i*bandDims[s].HH.dimX, 
                size);
        }
    }

    /* Write component */
    samplesToChar(result, dst, samplesNum);

    char outfile[strlen(filename)+strlen(suffix)];
    strcpy(outfile, filename);
    strcpy(outfile+strlen(filename), suffix);
    i = open(outfile, O_CREAT|O_WRONLY, 0644);
    if (i == -1) {
        error(0,errno,"cannot access %s", outfile);
        return -1;
    }
    printf("\nWriting to %s (%d x %d)\n", outfile, pixWidth, pixHeight);
    ssize_t x;
    x = write(i, result, samplesNum);
    close(i);

    hipHostFree(src);
    free(dst);
    free(result);
    free(bandDims);
    if (x == 0) return 1;
    return 0;
}
template int writeNStage2DDWT<float>(float *component_cuda, int pixWidth, int pixHeight, int stages, const char * filename, const char * suffix); 
template int writeNStage2DDWT<int>(int *component_cuda, int pixWidth, int pixHeight, int stages, const char * filename, const char * suffix); 
