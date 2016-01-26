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

#ifndef _DWT97_KERNEL_CU
#define _DWT97_KERNEL_CU

//#define A -1.586134342
//#define B -0.05298011854
//#define C 0.8829110762
//#define D 0.4435068522
//#define K 1.149604398 

/* 9/7 filters */
#define A -1.58613434205992
#define B -0.05298011857296
#define C  0.88291107553093
#define D  0.44350685204397
#define K  1.23017410491400

/* 5/3 filters*/
#define P53 -0.5
#define U53 0.25

__shared__ float f_blockData[2*DWT_BLOCK_SIZE_Y][DWT_BLOCK_SIZE_X + 1];

enum dwtfilter
{
    dwt97,
    dwt53
};

__device__ void rowPredict(float a, int n, float f_blockData[2*DWT_BLOCK_SIZE_Y][DWT_BLOCK_SIZE_X + 1])
{
    float _x,x,x_;
    if (n < blockDim.x-1) {
        _x = f_blockData[threadIdx.x][n-1]; 
        x  = f_blockData[threadIdx.x][n]; 
        x_ = f_blockData[threadIdx.x][n+1]; 
        x += a * (_x + x_);
        f_blockData[threadIdx.x][n] = x;
    }
    if (n == blockDim.x-1) { //last sample on the threadIdx.x line
        _x = f_blockData[threadIdx.x][n-1];
        x  = f_blockData[threadIdx.x][n];
        x += 2*a*_x;
        f_blockData[threadIdx.x][n] = x;
    }
}

__device__ void rowUpdate(float a, int n, float f_blockData[2*DWT_BLOCK_SIZE_Y][DWT_BLOCK_SIZE_X + 1])
{
    float _x,x,x_;
    if (n > 0) {
        _x = f_blockData[threadIdx.x][n-1]; 
         x = f_blockData[threadIdx.x][n]; 
        x_ = f_blockData[threadIdx.x][n+1]; 
        x += a * (_x + x_);
        f_blockData[threadIdx.x][n] = x;
    }
    if (n == 0) {
        x = f_blockData[threadIdx.x][0]; 
        x_ =f_blockData[threadIdx.x][1]; 
        x += a*(2*x_);
        f_blockData[threadIdx.x][0] = x;
    }
}

__device__ void colPredict(float a, int n, float f_blockData[2*DWT_BLOCK_SIZE_Y][DWT_BLOCK_SIZE_X + 1])
{
    float _x,x,x_;
    if (n < blockDim.x-1) {
        _x = f_blockData[n-1][threadIdx.x]; 
        x  = f_blockData[n][threadIdx.x]; 
        x_ = f_blockData[n+1][threadIdx.x]; 
        x += a * (_x + x_);
        f_blockData[n][threadIdx.x] = x;
    }
    if (n == blockDim.x-1) { //last sample on the threadIdx.x line
        _x = f_blockData[n-1][threadIdx.x];
        x  = f_blockData[n][threadIdx.x];
        x += 2*a*_x;
        f_blockData[n][threadIdx.x] = x;
    }
}

__device__ void colUpdate(float a, int n, float f_blockData[2*DWT_BLOCK_SIZE_Y][DWT_BLOCK_SIZE_X + 1])
{
    float _x,x,x_;
    if (n > 0) {
        _x = f_blockData[n-1][threadIdx.x]; 
         x = f_blockData[n][threadIdx.x]; 
        x_ = f_blockData[n+1][threadIdx.x]; 
        x += a * (_x + x_);
        f_blockData[n][threadIdx.x] = x;
    }
    if (n == 0) {
        x = f_blockData[0][threadIdx.x]; 
        x_ =f_blockData[1][threadIdx.x]; 
        x += a*(2*x_);
        f_blockData[0][threadIdx.x] = x;
    }
}

__device__ void computeFDwt97(float f_blockData[2*DWT_BLOCK_SIZE_Y][DWT_BLOCK_SIZE_X + 1])
{
    /*** ROW-WISE ***/
    //Predict 1
    //float a = -1.586134342;
    int n = threadIdx.y*2 + 1; //odd samples
    rowPredict((float)A, n, f_blockData);
    __syncthreads();

    // Update 1
    //a = -0.05298011854;
    n--; //even samples
    rowUpdate((float)B, n, f_blockData); 
    __syncthreads();

    //Predict 2
    //a = 0.8829110762;
    n++; //odd samples
    rowPredict((float)C, n, f_blockData);
    __syncthreads();

    // Update 2
    //a = 0.4435068522;
    n--; //even samples
    rowUpdate((float)D, n, f_blockData); 
    __syncthreads();

    //scale 
    //a = 1.149604398;
    f_blockData[threadIdx.x][n] = f_blockData[threadIdx.x][n] / (float)K; // sude
    f_blockData[threadIdx.x][n+1] = f_blockData[threadIdx.x][n+1] * (float)K; //liche
    __syncthreads();

    /*** COL-WISE ***/
    //Predict 1
    //float a = -1.586134342;
    n = threadIdx.y*2 + 1; //odd samples
    colPredict((float)A, n, f_blockData);
    __syncthreads();

    // Update 1
    //a = -0.05298011854;
    n--; //even samples
    colUpdate((float)B, n, f_blockData); 
    __syncthreads();

    //Predict 2
    //a = 0.8829110762;
    n++; //odd samples
    colPredict((float)C, n, f_blockData);
    __syncthreads();

    // Update 2
    //a = 0.4435068522;
    n--; //even samples
    colUpdate((float)D, n, f_blockData); 
    __syncthreads();

    //scale 
    //a = 1.149604398;
    f_blockData[n][threadIdx.x] = f_blockData[n][threadIdx.x] / (float)K; // sude
    f_blockData[n+1][threadIdx.x] = f_blockData[n+1][threadIdx.x] * (float)K; //liche
    __syncthreads();
}

__device__ void computeFDwt53(float f_blockData[2*DWT_BLOCK_SIZE_Y][DWT_BLOCK_SIZE_X + 1])
{
    /*** ROW-WISE ***/
    //Predict 1
    //float a = -1.586134342;
    int n = threadIdx.y*2 + 1; //odd samples
    rowPredict((float)P53, n, f_blockData);
    __syncthreads();

    // Update 1
    //a = -0.05298011854;
    n--; //even samples
    rowUpdate((float)U53, n, f_blockData); 
    __syncthreads();

    /*** COL-WISE ***/
    //Predict 1
    //float a = -1.586134342;
    n = threadIdx.y*2 + 1; //odd samples
    colPredict((float)P53, n, f_blockData);
    __syncthreads();

    // Update 1
    //a = -0.05298011854;
    n--; //even samples
    colUpdate((float)U53, n, f_blockData); 
    __syncthreads();
}


template<typename T>
__global__ void fdwt(T *src, T *out, int width, int height, enum dwtfilter filter)
{

    const int   globalTileX = IMUL(blockIdx.x, DWT_BLOCK_SIZE_X);
    const int   globalTileY = IMUL(blockIdx.y, DWT_BLOCK_SIZE_Y);   
    int       globalThreadX = globalTileX + threadIdx.x;
    int       globalThreadY = IMUL((globalTileY + threadIdx.y), 2); //even lines

    //mirroing along right edge of image: uvwxyZyxw...  'Z' is the the edge sample, everything
    //beyond is mirrored with the edge as axis.  2 * imgWidth - globalThreadX - 2
    if (globalThreadX >= width) 
        globalThreadX = IMUL(width, 2) - (globalThreadX + 2);
    if (globalThreadY >= height) 
        globalThreadY = IMUL(height, 2) - (globalThreadY + 2);

    int rowStart = IMUL(globalThreadY, width); //row start index in *src
    int sharedIdxY = IMUL(threadIdx.y, 2);

    //load even lines 
    f_blockData[sharedIdxY][threadIdx.x] = src[rowStart + globalThreadX];

    //compute odd lines indexes and load them
    if ((globalThreadY+1) < height) {
        rowStart += width; 
    } else
        rowStart -= width; 
    f_blockData[sharedIdxY+1][threadIdx.x] = src[rowStart + globalThreadX];

    __syncthreads();

    /* Compute DWT */
    switch (filter) {
        case dwt97:
            computeFDwt97(f_blockData);
            break;
        case dwt53:
            computeFDwt53(f_blockData);
            break;
    }

    //store data
    globalThreadX = globalTileX + threadIdx.x;
    globalThreadY = (globalTileY + threadIdx.y) * 2; //even lines

    /** Output DWT bands as 2x2 matrix **/
#if 0
    if (globalThreadX < width && globalThreadY < height) {
        globalThreadX /= 2;
        globalThreadY /= 2;
        rowStart = globalThreadY * width;

        if (threadIdx.x % 2 == 0) {
            out[rowStart + globalThreadX] = f_blockData[threadIdx.y*2][threadIdx.x];
        } else {
            out[rowStart + globalThreadX + width/2] = f_blockData[threadIdx.y*2][threadIdx.x];
        }

        rowStart += width*height/2;

        if (threadIdx.x % 2 == 0) {
            out[rowStart + globalThreadX] = f_blockData[(threadIdx.y*2)+1][threadIdx.x];
        } else {
            out[rowStart + globalThreadX + width/2] = f_blockData[(threadIdx.y*2)+1][threadIdx.x];
        }
    }
#else

    /** Output DWT bands linearly **/

    // tidX / 16; -- 0 determines first half of X, 1 determines second
    //               first half reads every 4th line (0, 4, 8, 12 ..), second half reads
    //               every 4th + 2 line(2, 6, 10, 14 ..)
    int sharedHalfX = threadIdx.x>>4;
    // tidY / 8;  -- 0 determines first half of Y, 1 determines second
    //               first half reads even samples, second half reads odd samples
    int sharedHalfY = threadIdx.y>>3;

    // threadIdx.x<<1       -- even samples on the line
    // - sharedHalfX*32     -- start reading from zero when in second half (tid 16 to 31)
    //                         second half reads from line sharedY+2
    // + sharedHalfY*1      -- read odd samples when Y is in second half
    int sharedX = threadIdx.x*2 - sharedHalfX*32 + sharedHalfY;

    // threadIdx.y<<2       -- go by four lines
    // - sharedHalfY*32     -- reset to line 0 when in second half (tid 8 to 16)
    // + sharedHalfX*2      -- read from base_line+2 when threadIdx.x is in second half
    //                      (base_line is every 4th line: threadIdx.y<<2 - sharedHalfY*32)
    int sharedY = threadIdx.y*4 - sharedHalfY*32  + sharedHalfX*2;

    int oddSampCount   = width >> 1;
    int evenSampCount  = width - oddSampCount;
    int oddLinesCount  = height >> 1;
    int evenLinesCount = height - oddLinesCount;

    if (globalTileX + sharedX < width && globalTileY * 2 + sharedY < height) {
        /** Storing even lines (LL and LH) **/
        // globalTileX >> 1         -- each of bands is half of width of original image,
        //                          so that samples from the every 4th (or 4th+1) line goes to
        //                          the first half of corresponding line of transformed image
        // sharedHalfX*(width>>1)   -- samples from the 4th + 2 (or 4th + 2 + 1) line goes to
        //                          the second half of corresponding line of transformed image 
        //                          (sharedHalfX determines threads with id 16 - 31, those threads
        //                          read every 4th + 2 or 4th + 2 + 1 line)
        // threadIdx.x              -- samples are stored at possiotion of threadIdx.x (0 - 15) 
        //                          when storing 4th (+1) line
        // -sharedHalfX*(DWT_BLOCK_SIZE_X>>1) -- when storing to second half (every 4th+2(+1) shared line) 
        //                                we need to reset threadIdx.x to values 0 - 15 so that 
        //                                substracting half of DWT_BLOCK_SIZE_X
        //globalThreadX = (globalTileX>>1) + sharedHalfX*(sharedHalfY*evenSampCount + (!sharedHalfY)*oddSampCount) 
         //               + threadIdx.x - sharedHalfX*(DWT_BLOCK_SIZE_X>>1);
        globalThreadX = (globalTileX>>1) + threadIdx.x - sharedHalfX*(DWT_BLOCK_SIZE_X>>1);

        // globalTileY>>1           -- we are storing even lines, this is just half of resulting 
        //                          image and this half contains LL (first quarter) and LH (second quarter)
        // sharedHalfY*(height>>2)  -- odd samples (LH or HH) on each line are stored to second quarter of 
        //                          current image half
        // threadIdx.y              -- storing on line number "threadIdx.y", which is 0 - 7 
        // -sharedHalfY*(DWT_BLOCK_SIZE_Y>>1) -- reseting line numbers "threadIdx.y" to 0 - 7 when storing odd
        //                                samples to the second half
        //globalThreadY = (globalTileY>>1) + sharedHalfY*(height>>2) + threadIdx.y - sharedHalfY*(DWT_BLOCK_SIZE_Y>>1);
        globalThreadY = globalTileY + threadIdx.y*2 + sharedHalfX - sharedHalfY*(DWT_BLOCK_SIZE_Y);
        

        // even lines (LL and HL)
        if (sharedHalfY == 0) {
            rowStart = IMUL(globalThreadY, evenSampCount);
        } else {
            rowStart = IMUL(globalThreadY, oddSampCount);
        }
        rowStart += sharedHalfY*IMUL(evenLinesCount, evenSampCount);

        out[rowStart + globalThreadX] = f_blockData[sharedY][sharedX];

        // odd lines (LH and HH)
        sharedY++;
        if (globalTileY * 2 + sharedY < height) {
            //rowStart += IMUL(width,(height>>1));
            rowStart = IMUL(evenLinesCount, evenSampCount) + IMUL(evenLinesCount, oddSampCount);
            if (sharedHalfY == 0) {
                rowStart += IMUL(globalThreadY, evenSampCount);
            } else {
                rowStart += IMUL(globalThreadY, oddSampCount);
            }
        //    rowStart += IMUL(globalThreadY, (sharedHalfY*evenSampCount + (!sharedHalfY)*oddSampCount));
            rowStart += sharedHalfY*IMUL(oddLinesCount, evenSampCount);
            out[rowStart + globalThreadX] = f_blockData[sharedY][sharedX];
        }
    }
#endif

}

__device__ void computeRDwt97(float f_blockData[2*DWT_BLOCK_SIZE_Y][DWT_BLOCK_SIZE_X + 1])
{
    int n = threadIdx.y*2; //even samples

    /*** COL-WISE ***/
    //unscale 
    f_blockData[n][threadIdx.x] = f_blockData[n][threadIdx.x] * (float)K; // sude
    f_blockData[n+1][threadIdx.x] = f_blockData[n+1][threadIdx.x] / (float)K; //liche
    __syncthreads();

    // Undo Update 2
    colUpdate(-1*(float)D, n, f_blockData); 
    __syncthreads();

    //Undo Predict 2
    n++; //odd samples
    colPredict(-1*(float)C, n, f_blockData);
    __syncthreads();

    // Undo Update 1
    n--; //even samples
    colUpdate(-1*(float)B, n, f_blockData); 
    __syncthreads();

    //Undo Predict 1
    n++; //odd samples
    colPredict(-1*(float)A, n, f_blockData);
    __syncthreads();

    /*** ROW-WISE ***/
    //unscale 
    n--; //even samples
    f_blockData[threadIdx.x][n] = f_blockData[threadIdx.x][n] * (float)K; // sude
    f_blockData[threadIdx.x][n+1] = f_blockData[threadIdx.x][n+1] / (float)K; //liche
    __syncthreads();

    // Undo Update 2
    rowUpdate(-1*(float)D, n, f_blockData); 
    __syncthreads();

    // Undo Predict 2
    n++; //odd samples
    rowPredict(-1*(float)C, n, f_blockData);
    __syncthreads();

    // Undo Update 1
    n--; //even samples
    rowUpdate(-1*(float)B, n, f_blockData); 
    __syncthreads();

    // Undo Predict 1
    n++; //even samples
    rowPredict(-1*(float)A, n, f_blockData);
    __syncthreads();
}

__device__ void computeRDwt53(float f_blockData[2*DWT_BLOCK_SIZE_Y][DWT_BLOCK_SIZE_X + 1])
{
    int n = threadIdx.y*2; //even samples

    /*** COL-WISE ***/
    // Undo Update 1
    colUpdate(-1*(float)U53, n, f_blockData); 
    __syncthreads();

    //Undo Predict 1
    n++; //odd samples
    colPredict(-1*(float)P53, n, f_blockData);
    __syncthreads();

    /*** ROW-WISE ***/
    // Undo Update 1
    rowUpdate(-1*(float)U53, n, f_blockData); 
    __syncthreads();

    // Undo Predict 1
    n++; //odd samples
    rowPredict(-1*(float)P53, n, f_blockData);
    __syncthreads();
}

template<typename T>
__global__ void rdwt(T *src, T *out, int width, int height, enum dwtfilter filter)
{

    const int   globalTileX = IMUL(blockIdx.x, DWT_BLOCK_SIZE_X);
    const int   globalTileY = IMUL(blockIdx.y, DWT_BLOCK_SIZE_Y);   
    int       globalThreadX;// = globalTileX + threadIdx.x;
    int       globalThreadY;// = IMUL((globalTileY + threadIdx.y), 2); //even lines

    // tidX / 16; -- 0 determines first half of X, 1 determines second
    //               first half reads every 4th line (0, 4, 8, 12 ..), second half reads
    //               every 4th + 2 line(2, 6, 10, 14 ..)
    int sharedHalfX = threadIdx.x>>4;
    // tidY / 8;  -- 0 determines first half of Y, 1 determines second
    //               first half reads even samples, second half reads odd samples
    int sharedHalfY = threadIdx.y>>3;

    // threadIdx.x<<1       -- even samples on the line
    // - sharedHalfX*32     -- start reading from zero when in second half (tid 16 to 31)
    //                         second half reads from line sharedY+2
    // + sharedHalfY*1      -- read odd samples when Y is in second half
    int sharedX = threadIdx.x*2 - sharedHalfX*32 + sharedHalfY;

    // threadIdx.y<<2       -- go by four lines
    // - sharedHalfY*32     -- reset to line 0 when in second half (tid 8 to 16)
    // + sharedHalfX*2      -- read from base_line+2 when threadIdx.x is in second half
    //                      (base_line is every 4th line: threadIdx.y<<2 - sharedHalfY*32)
    int sharedY = threadIdx.y*4 - sharedHalfY*32  + sharedHalfX*2;

    int oddSampCount   = width >> 1;
    int evenSampCount  = width - oddSampCount;
    int oddLinesCount  = height >> 1;
    int evenLinesCount = height - oddLinesCount;

    globalThreadX = (globalTileX>>1) + threadIdx.x - sharedHalfX*(DWT_BLOCK_SIZE_X>>1);
    globalThreadY = globalTileY + threadIdx.y*2 + sharedHalfX - sharedHalfY*(DWT_BLOCK_SIZE_Y);

    if (globalThreadY >= evenLinesCount)
        globalThreadY = (evenLinesCount<<1) - (globalThreadY + 2);

    int rowStart;
    if (sharedHalfY == 0) {
        if (globalThreadX >= evenSampCount)
            globalThreadX = (evenSampCount<<1) - (globalThreadX + 2);
        rowStart = IMUL(globalThreadY, evenSampCount);
    } else {
        if (globalThreadX >= oddSampCount)
            globalThreadX = (oddSampCount<<1) - (globalThreadX + 2);
        rowStart = IMUL(globalThreadY, oddSampCount);
    }
    rowStart += sharedHalfY*IMUL(evenLinesCount, evenSampCount);

    f_blockData[sharedY][sharedX] = src[rowStart + globalThreadX];

    globalThreadY = globalTileY + threadIdx.y*2 + sharedHalfX - sharedHalfY*(DWT_BLOCK_SIZE_Y);
    if (globalThreadY >= oddLinesCount)
        globalThreadY = (oddLinesCount<<1) - (globalThreadY + 2);

    sharedY++;
    rowStart = IMUL(evenLinesCount, evenSampCount) + IMUL(evenLinesCount, oddSampCount); // second img half
    rowStart += sharedHalfY*IMUL(oddLinesCount, evenSampCount); // + quater (the good one), only in case of we are reading odd samples
    if (sharedHalfY == 0) {
        rowStart += IMUL(globalThreadY, evenSampCount);
    } else {
        rowStart += IMUL(globalThreadY, oddSampCount);
    }
    f_blockData[sharedY][sharedX] = src[rowStart + globalThreadX];

    __syncthreads();

    /* Compute DWT */
    switch (filter) {
        case dwt97:
            computeRDwt97(f_blockData);
            break;
        case dwt53:
            computeRDwt53(f_blockData);
            break;
    }

    //store data
    int sharedIdxY = IMUL(threadIdx.y, 2);
    globalThreadX = globalTileX + threadIdx.x;
    globalThreadY = (globalTileY + threadIdx.y) * 2; //even lines
    if (globalThreadX < width && (globalThreadY) < height) {
        rowStart = IMUL(globalThreadY, width); //row start index in *src
        out[rowStart + globalThreadX] = f_blockData[sharedIdxY][threadIdx.x];
        if ((globalThreadY+1) < height) {
            rowStart += width; 
            out[rowStart + globalThreadX] = f_blockData[sharedIdxY+1][threadIdx.x];
        }
    }
}

#endif
