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
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include <getopt.h>

#include "common.h"
#include "components.h"
#include "dwt.h"

struct dwt {
    char * srcFilename;
    char * outFilename;
    unsigned char *srcImg;
    int pixWidth;
    int pixHeight;
    int components;
    int dwtLvls;
};

int getImg(char * srcFilename, unsigned char *srcImg, int inputSize)
{
    // printf("Loading ipnput: %s\n", srcFilename);
    char *path = "../../data/dwt2d/";
    char *newSrc = NULL;
    
    if((newSrc = (char *)malloc(strlen(srcFilename)+strlen(path)+1)) != NULL)
    {
        newSrc[0] = '\0';
        strcat(newSrc, path);
        strcat(newSrc, srcFilename);
        srcFilename= newSrc;
    }
    printf("Loading ipnput: %s\n", srcFilename);

    //srcFilename = strcat("../../data/dwt2d/",srcFilename);
    //read image
    int i = open(srcFilename, O_RDONLY, 0644);
    if (i == -1) { 
        error(0,errno,"cannot access %s", srcFilename);
        return -1;
    }
    int ret = read(i, srcImg, inputSize);
    printf("precteno %d, inputsize %d\n", ret, inputSize);
    close(i);

    return 0;
}


void usage() {
    printf("dwt [otpions] src_img.rgb <out_img.dwt>\n\
  -d, --dimension\t\tdimensions of src img, e.g. 1920x1080\n\
  -c, --components\t\tnumber of color components, default 3\n\
  -b, --depth\t\t\tbit depth, default 8\n\
  -l, --level\t\t\tDWT level, default 3\n\
  -D, --device\t\t\tcuda device\n\
  -f, --forward\t\t\tforward transform\n\
  -r, --reverse\t\t\treverse transform\n\
  -9, --97\t\t\t9/7 transform\n\
  -5, --53\t\t\t5/3 transform\n\
  -w  --write-visual\t\twrite output in visual (tiled) fashion instead of the linear\n");
}

template <typename T>
void processDWT(struct dwt *d, int forward, int writeVisual)
{
    int componentSize = d->pixWidth*d->pixHeight*sizeof(T);
    
    T *c_r_out, *backup ;
    cudaMalloc((void**)&c_r_out, componentSize); //< aligned component size
    cudaCheckError("Alloc device memory");
    cudaMemset(c_r_out, 0, componentSize);
    cudaCheckError("Memset device memory");
    
    cudaMalloc((void**)&backup, componentSize); //< aligned component size
    cudaCheckError("Alloc device memory");
    cudaMemset(backup, 0, componentSize);
    cudaCheckError("Memset device memory");
	
    if (d->components == 3) {
        /* Alloc two more buffers for G and B */
        T *c_g_out, *c_b_out;
        cudaMalloc((void**)&c_g_out, componentSize); //< aligned component size
        cudaCheckError("Alloc device memory");
        cudaMemset(c_g_out, 0, componentSize);
        cudaCheckError("Memset device memory");
        
        cudaMalloc((void**)&c_b_out, componentSize); //< aligned component size
        cudaCheckError("Alloc device memory");
        cudaMemset(c_b_out, 0, componentSize);
        cudaCheckError("Memset device memory");
        
        /* Load components */
        T *c_r, *c_g, *c_b;
        cudaMalloc((void**)&c_r, componentSize); //< R, aligned component size
        cudaCheckError("Alloc device memory");
        cudaMemset(c_r, 0, componentSize);
        cudaCheckError("Memset device memory");

        cudaMalloc((void**)&c_g, componentSize); //< G, aligned component size
        cudaCheckError("Alloc device memory");
        cudaMemset(c_g, 0, componentSize);
        cudaCheckError("Memset device memory");

        cudaMalloc((void**)&c_b, componentSize); //< B, aligned component size
        cudaCheckError("Alloc device memory");
        cudaMemset(c_b, 0, componentSize);
        cudaCheckError("Memset device memory");

        rgbToComponents(c_r, c_g, c_b, d->srcImg, d->pixWidth, d->pixHeight);
		

        /* Compute DWT and always store into file */

        nStage2dDWT(c_r, c_r_out, backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward);
        nStage2dDWT(c_g, c_g_out, backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward);
        nStage2dDWT(c_b, c_b_out, backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward);
     
        // -------test----------
        // T *h_r_out=(T*)malloc(componentSize);
		// cudaMemcpy(h_r_out, c_g_out, componentSize, cudaMemcpyDeviceToHost);
        // int ii;
		// for(ii=0;ii<componentSize/sizeof(T);ii++) {
			// fprintf(stderr, "%d ", h_r_out[ii]);
			// if((ii+1) % (d->pixWidth) == 0) fprintf(stderr, "\n");
        // }
        // -------test----------
        
		
        /* Store DWT to file */
#ifdef OUTPUT        
        if (writeVisual) {
            writeNStage2DDWT(c_r_out, d->pixWidth, d->pixHeight, d->dwtLvls, d->outFilename, ".r");
            writeNStage2DDWT(c_g_out, d->pixWidth, d->pixHeight, d->dwtLvls, d->outFilename, ".g");
            writeNStage2DDWT(c_b_out, d->pixWidth, d->pixHeight, d->dwtLvls, d->outFilename, ".b");
        } else {
            writeLinear(c_r_out, d->pixWidth, d->pixHeight, d->outFilename, ".r");
            writeLinear(c_g_out, d->pixWidth, d->pixHeight, d->outFilename, ".g");
            writeLinear(c_b_out, d->pixWidth, d->pixHeight, d->outFilename, ".b");
        }
#endif


        cudaFree(c_r);
        cudaCheckError("Cuda free");
        cudaFree(c_g);
        cudaCheckError("Cuda free");
        cudaFree(c_b);
        cudaCheckError("Cuda free");
        cudaFree(c_g_out);
        cudaCheckError("Cuda free");
        cudaFree(c_b_out);
        cudaCheckError("Cuda free");

    } 
    else if (d->components == 1) {
        //Load component
        T *c_r;
        cudaMalloc((void**)&(c_r), componentSize); //< R, aligned component size
        cudaCheckError("Alloc device memory");
        cudaMemset(c_r, 0, componentSize);
        cudaCheckError("Memset device memory");

        bwToComponent(c_r, d->srcImg, d->pixWidth, d->pixHeight);

        // Compute DWT 
        nStage2dDWT(c_r, c_r_out, backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward);

        // Store DWT to file 
// #ifdef OUTPUT        
        if (writeVisual) {
            writeNStage2DDWT(c_r_out, d->pixWidth, d->pixHeight, d->dwtLvls, d->outFilename, ".out");
        } else {
            writeLinear(c_r_out, d->pixWidth, d->pixHeight, d->outFilename, ".lin.out");
        }
// #endif
        cudaFree(c_r);
        cudaCheckError("Cuda free");
    }

    cudaFree(c_r_out);
    cudaCheckError("Cuda free device");
    cudaFree(backup);
    cudaCheckError("Cuda free device");
}

int main(int argc, char **argv) 
{
    int optindex = 0;
    char ch;
    struct option longopts[] = {
        {"dimension",   required_argument, 0, 'd'}, //dimensions of src img
        {"components",  required_argument, 0, 'c'}, //numger of components of src img
        {"depth",       required_argument, 0, 'b'}, //bit depth of src img
        {"level",       required_argument, 0, 'l'}, //level of dwt
        {"device",      required_argument, 0, 'D'}, //cuda device
        {"forward",     no_argument,       0, 'f'}, //forward transform
        {"reverse",     no_argument,       0, 'r'}, //reverse transform
        {"97",          no_argument,       0, '9'}, //9/7 transform
        {"53",          no_argument,       0, '5' }, //5/3transform
        {"write-visual",no_argument,       0, 'w' }, //write output (subbands) in visual (tiled) order instead of linear
        {"help",        no_argument,       0, 'h'}  
    };
    
    int pixWidth    = 0; //<real pixWidth
    int pixHeight   = 0; //<real pixHeight
    int compCount   = 3; //number of components; 3 for RGB or YUV, 4 for RGBA
    int bitDepth    = 8; 
    int dwtLvls     = 3; //default numuber of DWT levels
    int device      = 0;
    int forward     = 1; //forward transform
    int dwt97       = 1; //1=dwt9/7, 0=dwt5/3 transform
    int writeVisual = 0; //write output (subbands) in visual (tiled) order instead of linear
    char * pos;

    while ((ch = getopt_long(argc, argv, "d:c:b:l:D:fr95wh", longopts, &optindex)) != -1) {
        switch (ch) {
        case 'd':
            pixWidth = atoi(optarg);
            pos = strstr(optarg, "x");
            if (pos == NULL || pixWidth == 0 || (strlen(pos) >= strlen(optarg))) {
                usage();
                return -1;
            }
            pixHeight = atoi(pos+1);
            break;
        case 'c':
            compCount = atoi(optarg);
            break;
        case 'b':
            bitDepth = atoi(optarg);
            break;
        case 'l':
            dwtLvls = atoi(optarg);
            break;
        case 'D':
            device = atoi(optarg);
            break;
        case 'f':
            forward = 1;
            break;
        case 'r':
            forward = 0;
            break;
        case '9':
            dwt97 = 1;
            break;
        case '5':
            dwt97 = 0;
            break;
        case 'w':
            writeVisual = 1;
            break;
        case 'h':
            usage();
            return 0;
        case '?':
            return -1;
        default :
            usage();
            return -1;
        }
    }
	argc -= optind;
	argv += optind;

    if (argc == 0) { // at least one filename is expected
        printf("Please supply src file name\n");
        usage();
        return -1;
    }

    if (pixWidth <= 0 || pixHeight <=0) {
        printf("Wrong or missing dimensions\n");
        usage();
        return -1;
    }

    if (forward == 0) {
        writeVisual = 0; //do not write visual when RDWT
    }

    // device init
    int devCount;
    cudaGetDeviceCount(&devCount);
    cudaCheckError("Get device count");
    if (devCount == 0) {
        printf("No CUDA enabled device\n");
        return -1;
    } 
    if (device < 0 || device > devCount -1) {
        printf("Selected device %d is out of bound. Devices on your system are in range %d - %d\n", 
               device, 0, devCount -1);
        return -1;
    }
    cudaDeviceProp devProp;                                          
    cudaGetDeviceProperties(&devProp, device);  
    cudaCheckError("Get device properties");
    if (devProp.major < 1) {                                         
        printf("Device %d does not support CUDA\n", device);
        return -1;
    }                                                                   
    printf("Using device %d: %s\n", device, devProp.name);
    cudaSetDevice(device);
    cudaCheckError("Set selected device");

    struct dwt *d;
    d = (struct dwt *)malloc(sizeof(struct dwt));
    d->srcImg = NULL;
    d->pixWidth = pixWidth;
    d->pixHeight = pixHeight;
    d->components = compCount;
    d->dwtLvls  = dwtLvls;

    // file names
    d->srcFilename = (char *)malloc(strlen(argv[0]));
    strcpy(d->srcFilename, argv[0]);
    if (argc == 1) { // only one filename supplyed
        d->outFilename = (char *)malloc(strlen(d->srcFilename)+4);
        strcpy(d->outFilename, d->srcFilename);
        strcpy(d->outFilename+strlen(d->srcFilename), ".dwt");
    } else {
        d->outFilename = strdup(argv[1]);
    }

    //Input review
    printf("Source file:\t\t%s\n", d->srcFilename);
    printf(" Dimensions:\t\t%dx%d\n", pixWidth, pixHeight);
    printf(" Components count:\t%d\n", compCount);
    printf(" Bit depth:\t\t%d\n", bitDepth);
    printf(" DWT levels:\t\t%d\n", dwtLvls);
    printf(" Forward transform:\t%d\n", forward);
    printf(" 9/7 transform:\t\t%d\n", dwt97);
    
    //data sizes
    int inputSize = pixWidth*pixHeight*compCount; //<amount of data (in bytes) to proccess

    //load img source image
    cudaMallocHost((void **)&d->srcImg, inputSize);
    cudaCheckError("Alloc host memory");
    if (getImg(d->srcFilename, d->srcImg, inputSize) == -1) 
        return -1;

    /* DWT */
    if (forward == 1) {
        if(dwt97 == 1 )
            processDWT<float>(d, forward, writeVisual);
        else // 5/3
            processDWT<int>(d, forward, writeVisual);
    }
    else { // reverse
        if(dwt97 == 1 )
            processDWT<float>(d, forward, writeVisual);
        else // 5/3
            processDWT<int>(d, forward, writeVisual);
    }

    //writeComponent(r_cuda, pixWidth, pixHeight, srcFilename, ".g");
    //writeComponent(g_wave_cuda, 512000, ".g");
    //writeComponent(g_cuda, componentSize, ".g");
    //writeComponent(b_wave_cuda, componentSize, ".b");
    cudaFreeHost(d->srcImg);
    cudaCheckError("Cuda free host");

    return 0;
}
