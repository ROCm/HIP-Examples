/// @file    fdwt53.cu
/// @brief   CUDA implementation of forward 5/3 2D DWT.
/// @author  Martin Jirman (207962@mail.muni.cz)
/// @date    2011-02-04 13:23
///
///
/// Copyright (c) 2011 Martin Jirman
/// All rights reserved.
/// 
/// Redistribution and use in source and binary forms, with or without
/// modification, are permitted provided that the following conditions are met:
/// 
///     * Redistributions of source code must retain the above copyright
///       notice, this list of conditions and the following disclaimer.
///     * Redistributions in binary form must reproduce the above copyright
///       notice, this list of conditions and the following disclaimer in the
///       documentation and/or other materials provided with the distribution.
/// 
/// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
/// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
/// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
/// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
/// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
/// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
/// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
/// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
/// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
/// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
/// POSSIBILITY OF SUCH DAMAGE.
///


#include "common.h"
#include "transform_buffer.h"
#include "io.h"

namespace dwt_cuda {


  /// Wraps buffer and methods needed for computing one level of 5/3 FDWT
  /// using sliding window approach.
  /// @tparam WIN_SIZE_X  width of sliding window
  /// @tparam WIN_SIZE_Y  height of sliding window
  template <int WIN_SIZE_X, int WIN_SIZE_Y>
  class FDWT53 {
  private:
    
    /// Info needed for processing of one input column.
    /// @tparam CHECKED_LOADER  true if column's loader should check boundaries
    ///                         false if there are no near boudnaries to check
    template <bool CHECKED_LOADER>
    struct FDWT53Column {
      /// loader for the column
      VerticalDWTPixelLoader<int, CHECKED_LOADER> loader;
      
      /// offset of the column in shared buffer
      int offset;                   
      
      // backup of first 3 loaded pixels (not transformed)
      int pixel0, pixel1, pixel2;
      
      /// Sets all fields to anything to prevent 'uninitialized' warnings.
      __device__ void clear() {
        offset = pixel0 = pixel1 = pixel2 = 0;
        loader.clear();
      }
    };


    /// Type of shared memory buffer for 5/3 FDWT transforms.
    typedef TransformBuffer<int, WIN_SIZE_X, WIN_SIZE_Y + 3, 2> FDWT53Buffer;

    /// Actual shared buffer used for forward 5/3 DWT.
    FDWT53Buffer buffer;

    /// Difference between indices of two vertical neighbors in buffer.
    enum { STRIDE = FDWT53Buffer::VERTICAL_STRIDE };


    /// Forward 5/3 DWT predict operation.
    struct Forward53Predict {
      __device__ void operator() (const int p, int & c, const int n) const {
        // c = n;
        c -= (p + n) / 2;      // F.8, page 126, ITU-T Rec. T.800 final draft the real one
      }
    };


    /// Forward 5/3 DWT update operation.
    struct Forward53Update {
      __device__ void operator() (const int p, int & c, const int n) const {
        c += (p + n + 2) / 4;  // F.9, page 126, ITU-T Rec. T.800 final draft
      }
    };


    /// Initializes one column: computes offset of the column in shared memory
    /// buffer, initializes loader and finally uses it to load first 3 pixels.
    /// @tparam CHECKED  true if loader of the column checks boundaries
    /// @param column    (uninitialized) column info to be initialized
    /// @param input     input image
    /// @param sizeX     width of the input image
    /// @param sizeY     height of the input image
    /// @param colIndex  x-axis coordinate of the column (relative to the left
    ///                  side of this threadblock's block of input pixels)
    /// @param firstY    y-axis coordinate of first image row to be transformed
	
	template <bool CHECKED>
    __device__ void initColumn(FDWT53Column<CHECKED> & column,
                               const int * const input,
                               const int sizeX, const int sizeY,
                               const int colIndex, const int firstY) {
      // get offset of the column with index 'cId'
      column.offset = buffer.getColumnOffset(colIndex);

      // coordinates of the first pixel to be loaded
      const int firstX = blockIdx.x * WIN_SIZE_X + colIndex;

      if(blockIdx.y == 0) {
        // topmost block - apply mirroring rules when loading first 3 rows
        column.loader.init(sizeX, sizeY, firstX, firstY);

        // load pixels in mirrored way
        column.pixel2 = column.loader.loadFrom(input);  // loaded pixel #0
        column.pixel1 = column.loader.loadFrom(input);  // loaded pixel #1
        column.pixel0 = column.loader.loadFrom(input);  // loaded pixel #2

        // reinitialize loader to start with pixel #1 again
        column.loader.init(sizeX, sizeY, firstX, firstY + 1);
      } else {
        // non-topmost row - regular loading:
        column.loader.init(sizeX, sizeY, firstX, firstY - 2);

        // load 3 rows into the column
        column.pixel0 = column.loader.loadFrom(input);
        column.pixel1 = column.loader.loadFrom(input);
        column.pixel2 = column.loader.loadFrom(input);
        // Now, the next pixel, which will be loaded by loader, is pixel #1.
      }
		
	}


    /// Loads and vertically transforms given column. Assumes that first 3
    /// pixels are already loaded in column fields pixel0 ... pixel2.
    /// @tparam CHECKED  true if loader of the column checks boundaries
    /// @param column    column to be loaded and vertically transformed
    /// @param input     pointer to input image data
    template <bool CHECKED>
    __device__ void loadAndVerticallyTransform(FDWT53Column<CHECKED> & column,
                                               const int * const input) {
	  // take 3 loaded pixels and put them into shared memory transform buffer
      buffer[column.offset + 0 * STRIDE] = column.pixel0;
      buffer[column.offset + 1 * STRIDE] = column.pixel1;
      buffer[column.offset + 2 * STRIDE] = column.pixel2;
	
      // load remaining pixels to be able to vertically transform the window

      for(int i = 3; i < (3 + WIN_SIZE_Y); i++) 
      {
        buffer[column.offset + i * STRIDE] = column.loader.loadFrom(input);
      }
 
      // remember last 3 pixels for use in next iteration
      column.pixel0 = buffer[column.offset + (WIN_SIZE_Y + 0) * STRIDE];
      column.pixel1 = buffer[column.offset + (WIN_SIZE_Y + 1) * STRIDE];
      column.pixel2 = buffer[column.offset + (WIN_SIZE_Y + 2) * STRIDE];

      // vertically transform the column in transform buffer
	  buffer.forEachVerticalOdd(column.offset, Forward53Predict());
      buffer.forEachVerticalEven(column.offset, Forward53Update());
	  
    }


    /// Actual implementation of 5/3 FDWT.
    /// @tparam CHECK_LOADS   true if input loader must check boundaries
    /// @tparam CHECK_WRITES  true if output writer must check boundaries
    /// @param in        input image
    /// @param out       output buffer
    /// @param sizeX     width of the input image 
    /// @param sizeY     height of the input image
    /// @param winSteps  number of sliding window steps
    template <bool CHECK_LOADS, bool CHECK_WRITES>
    __device__ void transform(const int * const in, int * const out,
                              const int sizeX, const int sizeY,
                              const int winSteps) {
      // info about one main and one boundary columns processed by this thread
      FDWT53Column<CHECK_LOADS> column;    
      FDWT53Column<CHECK_LOADS> boundaryColumn;  // only few threads use this

      // Initialize all column info: initialize loaders, compute offset of 
      // column in shared buffer and initialize loader of column.
      const int firstY = blockIdx.y * WIN_SIZE_Y * winSteps;
	  initColumn(column, in, sizeX, sizeY, threadIdx.x, firstY); //has been checked Mar 9th

	  
      // first 3 threads initialize boundary columns, others do not use them
      boundaryColumn.clear();
      if(threadIdx.x < 3) {
        // index of boundary column (relative x-axis coordinate of the column)
        const int colId = threadIdx.x + ((threadIdx.x == 0) ? WIN_SIZE_X : -3);

        // initialize the column
        initColumn(boundaryColumn, in, sizeX, sizeY, colId, firstY);

      }
	  
	  
      // index of column which will be written into output by this thread      
	  const int outColumnIndex = parityIdx<WIN_SIZE_X>();

      // offset of column which will be written by this thread into output
      const int outColumnOffset = buffer.getColumnOffset(outColumnIndex);

      // initialize output writer for this thread
      const int outputFirstX = blockIdx.x * WIN_SIZE_X + outColumnIndex;
      VerticalDWTBandWriter<int, CHECK_WRITES> writer;
	  writer.init(sizeX, sizeY, outputFirstX, firstY);
		
	  
      // Sliding window iterations:
      // Each iteration assumes that first 3 pixels of each column are loaded.
     for(int w = 0; w < winSteps; w++) {

	 // For each column (including boundary columns): load and vertically
        // transform another WIN_SIZE_Y lines.
        loadAndVerticallyTransform(column, in);
        if(threadIdx.x < 3) { 
          loadAndVerticallyTransform(boundaryColumn, in); 
        }
 		
        // wait for all columns to be vertically transformed and transform all
        // output rows horizontally
        __syncthreads();
		

		buffer.forEachHorizontalOdd(2, WIN_SIZE_Y, Forward53Predict());
        __syncthreads();
		
        buffer.forEachHorizontalEven(2, WIN_SIZE_Y, Forward53Update());

        // wait for all output rows to be transformed horizontally and write
        // them into output buffer
        __syncthreads();	


        for(int r = 2; r < (2 + WIN_SIZE_Y); r += 2) {
          // Write low coefficients from output column into low band ...
			writer.writeLowInto(out, buffer[outColumnOffset + r * STRIDE]);
          // ... and high coeficients into the high band.
			writer.writeHighInto(out, buffer[outColumnOffset + (r+1) * STRIDE]);
        }

        // before proceeding to next iteration, wait for all output columns
        // to be written into the output
        __syncthreads();
			
	}
	
    }

    
  public:
    /// Determines, whether this block's pixels touch boundary and selects
    /// right version of algorithm according to it - for many threadblocks, it
    /// selects version which does not deal with boundary mirroring and thus is 
    /// slightly faster.
    /// @param in     input image
    /// @param out    output buffer
    /// @param sx     width of the input image 
    /// @param sy     height of the input image
    /// @param steps  number of sliding window steps
    __device__ static void run(const int * const in, int * const out,
                               const int sx, const int sy, const int steps) {
        // if(blockIdx.x==0 && blockIdx.y ==11 && threadIdx.x >=0&&threadIdx.x <64){
      // object with transform buffer in shared memory
      __shared__ FDWT53<WIN_SIZE_X, WIN_SIZE_Y> fdwt53;

	  // Compute limits of this threadblock's block of pixels and use them to
      // determine, whether this threadblock will have to deal with boundary.
      // (1 in next expressions is for radius of impulse response of 9/7 FDWT.)
      const int maxX = (blockIdx.x + 1) * WIN_SIZE_X + 1;
      const int maxY = (blockIdx.y + 1) * WIN_SIZE_Y * steps + 1;
      const bool atRightBoudary = maxX >= sx;
      const bool atBottomBoudary = maxY >= sy;

      // Select specialized version of code according to distance of this
      // threadblock's pixels from image boundary.
      if(atBottomBoudary) 
      {
        // near bottom boundary => check both writing and reading
        fdwt53.transform<true, true>(in, out, sx, sy, steps);
      } else if(atRightBoudary) 
      {
        // near right boundary only => check writing only
        fdwt53.transform<false, true>(in, out, sx, sy, steps);
      } else 
      {
        // no nearby boundary => check nothing
        fdwt53.transform<false, false>(in, out, sx, sy, steps);
      }
    }
    // }
    
  }; // end of class FDWT53
  
  
  
  /// Main GPU 5/3 FDWT entry point.
  /// @tparam WIN_SX   width of sliding window to be used
  /// @tparam WIN_SY   height of sliding window to be used
  /// @param input     input image
  /// @param output    output buffer
  /// @param sizeX     width of the input image 
  /// @param sizeY     height of the input image
  /// @param winSteps  number of sliding window steps
  template <int WIN_SX, int WIN_SY>
  __launch_bounds__(WIN_SX, CTMIN(SHM_SIZE/sizeof(FDWT53<WIN_SX, WIN_SY>), 8))
  __global__ void fdwt53Kernel(const int * const input, int * const output,
                               const int sizeX, const int sizeY,
                               const int winSteps) {
    FDWT53<WIN_SX, WIN_SY>::run(input, output, sizeX, sizeY, winSteps);
  }

  

  /// Only computes optimal number of sliding window steps, 
  /// number of threadblocks and then lanches the 5/3 FDWT kernel.
  /// @tparam WIN_SX  width of sliding window
  /// @tparam WIN_SY  height of sliding window
  /// @param in       input image
  /// @param out      output buffer
  /// @param sx       width of the input image 
  /// @param sy       height of the input image
  template <int WIN_SX, int WIN_SY>
  void launchFDWT53Kernel (int * in, int * out, int sx, int sy) {
    // compute optimal number of steps of each sliding window
	
    const int steps = divRndUp(sy, 15 * WIN_SY);

	int gx = divRndUp(sx, WIN_SX);
	int gy = divRndUp(sy, WIN_SY * steps);

	printf("\n sliding steps = %d , gx = %d , gy = %d \n", steps, gx, gy);

    // prepare grid size
    dim3 gSize(divRndUp(sx, WIN_SX), divRndUp(sy, WIN_SY * steps));
    // printf("\n globalx=%d, globaly=%d, blocksize=%d\n", gSize.x, gSize.y, WIN_SX);
    
    // run kernel, possibly measure time and finally check the call
    // PERF_BEGIN
    fdwt53Kernel<WIN_SX, WIN_SY><<<gSize, WIN_SX>>>(in, out, sx, sy, steps);
    // PERF_END("        FDWT53", sx, sy)
    // CudaDWTTester::checkLastKernelCall("FDWT 5/3 kernel");
    printf("fdwt53Kernel in launchFDWT53Kernel has finished");
	
  }
  
  
  
  /// Forward 5/3 2D DWT. See common rules (above) for more details.
  /// @param in      Expected to be normalized into range [-128, 127].
  ///                Will not be preserved (will be overwritten).
  /// @param out     output buffer on GPU
  /// @param sizeX   width of input image (in pixels)
  /// @param sizeY   height of input image (in pixels)
  /// @param levels  number of recursive DWT levels
  void fdwt53(int * in, int * out, int sizeX, int sizeY, int levels) {
    // select right width of kernel for the size of the image
	
    if(sizeX >= 960) {
      launchFDWT53Kernel<192, 8>(in, out, sizeX, sizeY);
    } else if (sizeX >= 480) {
      launchFDWT53Kernel<128, 8>(in, out, sizeX, sizeY);
    } else {
      launchFDWT53Kernel<64, 8>(in, out, sizeX, sizeY);
    }
    
    // if this was not the last level, continue recursively with other levels
    if(levels > 1) {
      // copy output's LL band back into input buffer
      const int llSizeX = divRndUp(sizeX, 2); 
      const int llSizeY = divRndUp(sizeY, 2);
	 // printf("\n llSizeX = %d , llSizeY = %d \n", llSizeX, llSizeY);
      memCopy(in, out, llSizeX, llSizeY); //the function memCopy in cuda_dwt/common.h line 238
      
      // run remaining levels of FDWT
      fdwt53(in, out, llSizeX, llSizeY, levels - 1);
    }
  }
  
  

} // end of namespace dwt_cuda
