/// 
/// @file    rdwt97.cu
/// @brief   CUDA implementation of reverse 9/7 2D DWT.
/// @author  Martin Jirman (207962@mail.muni.cz)
/// @date    2011-02-03 21:59
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

  
  /// Wraps shared memory buffer and methods for computing 9/7 RDWT using
  /// lifting schema and sliding window.
  /// @tparam WIN_SIZE_X  width of the sliding window
  /// @tparam WIN_SIZE_Y  height of the sliding window
  template <int WIN_SIZE_X, int WIN_SIZE_Y>
  class RDWT97 {
  private:
    
    /// Info related to loading of one input column.
    /// @tparam CHECKED true if boundary chould be checked,
    ///                 false if there is no near boudnary
    template <bool CHECKED>
    struct RDWT97Column  {
      /// laoder of input pxels for given column.
      VerticalDWTBandLoader<float, CHECKED> loader;
      
      /// Offset of loaded column in shared memory buffer.
      int offset;
      
      /// Sets all fields to some values to avoid 'uninitialized' warnings.
      __device__ void clear() {
        loader.clear();
        offset = 0;
      }
    };


    /// Shared memory buffer used for 9/7 DWT transforms.
    typedef TransformBuffer<float, WIN_SIZE_X, WIN_SIZE_Y + 7, 4> RDWT97Buffer;

    /// Shared buffer used for reverse 9/7 DWT.
    RDWT97Buffer buffer;

    /// Difference between indices of two vertical neighbors in buffer.
    enum { STRIDE = RDWT97Buffer::VERTICAL_STRIDE };


    /// Horizontal 9/7 RDWT on specified lines of transform buffer.
    /// @param lines      number of lines to be transformed
    /// @param firstLine  index of the first line to be transformed
    __device__ void horizontalRDWT97(int lines, int firstLine) {
      __syncthreads();
      buffer.scaleHorizontal(scale97Mul, scale97Div, firstLine, lines);
      __syncthreads();
      buffer.forEachHorizontalEven(firstLine, lines, AddScaledSum(r97update2));
      __syncthreads();
      buffer.forEachHorizontalOdd(firstLine, lines, AddScaledSum(r97predict2));
      __syncthreads();
      buffer.forEachHorizontalEven(firstLine, lines, AddScaledSum(r97update1));
      __syncthreads();
      buffer.forEachHorizontalOdd(firstLine, lines, AddScaledSum(r97Predict1));
      __syncthreads();
    }


    /// Initializes one column of shared transform buffer with 7 input pixels.
    /// Those 7 pixels will not be transformed. Also initializes given loader.
    /// @tparam CHECKED  true if there are near image boundaries
    /// @param colIndex  index of column in shared transform buffer
    /// @param input     input image
    /// @param sizeX     width of the input image
    /// @param sizeY     height of the input image
    /// @param column    (uninitialized) info about loading one column
    /// @param firstY    index of first image row to be transformed
    template <bool CHECKED>
    __device__ void initColumn(const int colIndex, const float * const input, 
                               const int sizeX, const int sizeY,
                               RDWT97Column<CHECKED> & column,
                               const int firstY) {
      // coordinates of the first coefficient to be loaded
      const int firstX = blockIdx.x * WIN_SIZE_X + colIndex;

      // offset of the column with index 'colIndex' in the transform buffer
      column.offset = buffer.getColumnOffset(colIndex);

      if(blockIdx.y == 0) {
        // topmost block - apply mirroring rules when loading first 7 rows
        column.loader.init(sizeX, sizeY, firstX, firstY);

        // load pixels in mirrored way
        buffer[column.offset + 3 * STRIDE] = column.loader.loadLowFrom(input);
        buffer[column.offset + 4 * STRIDE] =
        buffer[column.offset + 2 * STRIDE] = column.loader.loadHighFrom(input);
        buffer[column.offset + 5 * STRIDE] =
        buffer[column.offset + 1 * STRIDE] = column.loader.loadLowFrom(input);
        buffer[column.offset + 6 * STRIDE] = 
        buffer[column.offset + 0 * STRIDE] = column.loader.loadHighFrom(input);
      } else {
        // non-topmost row - regular loading:
        column.loader.init(sizeX, sizeY, firstX, firstY - 3);
        buffer[column.offset + 0 * STRIDE] = column.loader.loadHighFrom(input);
        buffer[column.offset + 1 * STRIDE] = column.loader.loadLowFrom(input);
        buffer[column.offset + 2 * STRIDE] = column.loader.loadHighFrom(input);
        buffer[column.offset + 3 * STRIDE] = column.loader.loadLowFrom(input);
        buffer[column.offset + 4 * STRIDE] = column.loader.loadHighFrom(input);
        buffer[column.offset + 5 * STRIDE] = column.loader.loadLowFrom(input);
        buffer[column.offset + 6 * STRIDE] = column.loader.loadHighFrom(input);
      }
      // Now, the next coefficient, which will be loaded by loader, is #4.
    }


    /// Using given loader, it loads another WIN_SIZE_Y coefficients
    /// into specified column.
    /// @tparam CHECKED  true if there are near image boundaries
    /// @param col       info about loaded column
    /// @param input     buffer with input coefficients
    template <bool CHECKED>
    inline __device__ void loadWindowIntoColumn(RDWT97Column<CHECKED> & col,
                                                const float * const input) {
      for(int i = 7; i < (7 + WIN_SIZE_Y); i += 2) {
        buffer[col.offset + i * STRIDE] = col.loader.loadLowFrom(input);
        buffer[col.offset + (i + 1) * STRIDE] = col.loader.loadHighFrom(input);
      }
    }


    /// Actual GPU 9/7 RDWT sliding window lifting schema implementation.
    /// @tparam CHECKED_LOADS   true if loader should check boundaries
    /// @tparam CHECKED_WRITES  true if boundaries should be taken into account
    ///                         when writing into output buffer
    /// @param in        input image (9/7 transformed coefficients)
    /// @param out       output buffer (for reverse transformed image)
    /// @param sizeX     width of the output image 
    /// @param sizeY     height of the output image
    /// @param winSteps  number of steps of sliding window
    template <bool CHECKED_LOADS, bool CHECKED_WRITES>
    __device__ void transform(const float * const in, float * const out,
                              const int sizeX, const int sizeY,
                              const int winSteps) {
      // info about one main column and one boundary column
      RDWT97Column<CHECKED_LOADS> column;
      RDWT97Column<CHECKED_LOADS> boundaryColumn;

      // index of first image row to be transformed
      const int firstY = blockIdx.y * WIN_SIZE_Y * winSteps;

      // initialize boundary columns
      boundaryColumn.clear();
      if(threadIdx.x < 7) {
        // each thread among first 7 ones gets index of one of boundary columns
        const int colId = threadIdx.x + ((threadIdx.x < 4) ? WIN_SIZE_X : -7);

        // Thread initializes offset of the boundary column (in shared  
        // buffer), first 7 pixels of the column and a loader for this column.
        initColumn(colId, in, sizeX, sizeY, boundaryColumn, firstY);
      }

      // All threads initialize central columns.
      initColumn(parityIdx<WIN_SIZE_X>(), in, sizeX, sizeY, column, firstY);

      // horizontally transform first 7 rows
      horizontalRDWT97(7, 0);

      // writer of output pixels - initialize it
      const int outputX = blockIdx.x * WIN_SIZE_X + threadIdx.x;
      VerticalDWTPixelWriter<float, CHECKED_WRITES> writer;
      writer.init(sizeX, sizeY, outputX, firstY);

      // offset of column (in transform buffer) saved by this thread
      const int outColumnOffset = buffer.getColumnOffset(threadIdx.x);

      // (Each iteration assumes that first 7 rows of transform buffer are 
      // already loaded with horizontally transformed pixels.)
      for(int w = 0; w < winSteps; w++) {
        // Load another WIN_SIZE_Y lines of this thread's column
        // into the transform buffer.
        loadWindowIntoColumn(column, in);

        // possibly load boundary columns
        if(threadIdx.x < 7) {
          loadWindowIntoColumn(boundaryColumn, in);
        }

        // horizontally transform all newly loaded lines
        horizontalRDWT97(WIN_SIZE_Y, 7);

        // Using 7 registers, remember current values of last 7 rows 
        // of transform buffer. These rows are transformed horizontally 
        // only and will be used in next iteration.
        float last7Lines[7];
        for(int i = 0; i < 7; i++) {
          last7Lines[i] = buffer[outColumnOffset + (WIN_SIZE_Y + i) * STRIDE];
        }

        // vertically transform all central columns
        buffer.scaleVertical(scale97Div, scale97Mul, outColumnOffset,
                             WIN_SIZE_Y + 7, 0);
        buffer.forEachVerticalOdd(outColumnOffset, AddScaledSum(r97update2));
        buffer.forEachVerticalEven(outColumnOffset, AddScaledSum(r97predict2));
        buffer.forEachVerticalOdd(outColumnOffset, AddScaledSum(r97update1));
        buffer.forEachVerticalEven(outColumnOffset, AddScaledSum(r97Predict1));

        // Save all results of current window. Results are in transform buffer
        // at rows from #3 to #(3 + WIN_SIZE_Y). Other rows are invalid now.
        // (They only served as a boundary for vertical RDWT.)
        for(int i = 3; i < (3 + WIN_SIZE_Y); i++) {
          writer.writeInto(out, buffer[outColumnOffset + i * STRIDE]);
        }

        // Use last 7 remembered lines as first 7 lines for next iteration.
        // As expected, these lines are already horizontally transformed.
        for(int i = 0; i < 7; i++) {
          buffer[outColumnOffset + i * STRIDE] = last7Lines[i];
        }

        // Wait for all writing threads before proceeding to loading new
        // coeficients in next iteration. (Not to overwrite those which
        // are not written yet.)
        __syncthreads();
      }
    }


  public:
    /// Main GPU 9/7 RDWT entry point.
    /// @param in     input image (9/7 transformed coefficients)
    /// @param out    output buffer (for reverse transformed image)
    /// @param sizeX  width of the output image 
    /// @param sizeY  height of the output image
    __device__ static void run(const float * const input, float * const output,
                               const int sx, const int sy, const int steps) {
      // prepare instance with buffer in shared memory
      __shared__ RDWT97<WIN_SIZE_X, WIN_SIZE_Y> rdwt97;
      
      // Compute limits of this threadblock's block of pixels and use them to
      // determine, whether this threadblock will have to deal with boundary.
      // (3 in next expressions is for radius of impulse response of 9/7 RDWT.)
      const int maxX = (blockIdx.x + 1) * WIN_SIZE_X + 3;
      const int maxY = (blockIdx.y + 1) * WIN_SIZE_Y * steps + 3;
      const bool atRightBoudary = maxX >= sx;
      const bool atBottomBoudary = maxY >= sy;

      // Select specialized version of code according to distance of this
      // threadblock's pixels from image boundary.
      if(atBottomBoudary) {
        // near bottom boundary => check both writing and reading
        rdwt97.transform<true, true>(input, output, sx, sy, steps);
      } else if(atRightBoudary) {
        // near right boundary only => check writing only
        rdwt97.transform<false, true>(input, output, sx, sy, steps);
      } else {
        // no nearby boundary => check nothing
        rdwt97.transform<false, false>(input, output, sx, sy, steps);
      }
    }
    
  }; // end of class RDWT97
  
    
  
  /// Main GPU 9/7 RDWT entry point.
  /// @param in     input image (9/7 transformed coefficients)
  /// @param out    output buffer (for reverse transformed image)
  /// @param sizeX  width of the output image 
  /// @param sizeY  height of the output image
  template <int WIN_SX, int WIN_SY>
  __launch_bounds__(WIN_SX, CTMIN(SHM_SIZE/sizeof(RDWT97<WIN_SX, WIN_SY>), 8))
  __global__ void rdwt97Kernel(const float * const in, float * const out,
                               const int sx, const int sy, const int steps) {
    RDWT97<WIN_SX, WIN_SY>::run(in, out, sx, sy, steps);
  }
  
  
  
  /// Only computes optimal number of sliding window steps, 
  /// number of threadblocks and then lanches the 9/7 RDWT kernel.
  /// @tparam WIN_SX  width of sliding window
  /// @tparam WIN_SY  height of sliding window
  /// @param in       input image
  /// @param out      output buffer
  /// @param sx       width of the input image 
  /// @param sy       height of the input image
  template <int WIN_SX, int WIN_SY>
  void launchRDWT97Kernel (float * in, float * out, int sx, int sy) {
    // compute optimal number of steps of each sliding window
    const int steps = divRndUp(sy, 15 * WIN_SY);
    
    // prepare grid size
    dim3 gSize(divRndUp(sx, WIN_SX), divRndUp(sy, WIN_SY * steps));
    
    // finally launch kernel
    PERF_BEGIN
    rdwt97Kernel<WIN_SX, WIN_SY><<<gSize, WIN_SX>>>(in, out, sx, sy, steps);
    PERF_END("        RDWT97", sx, sy)
    CudaDWTTester::checkLastKernelCall("RDWT 9/7 kernel");
  }
  
  
  
  /// Reverse 9/7 2D DWT. See common rules (dwt.h) for more details.
  /// @param in      Input DWT coefficients. Format described in common rules.
  ///                Will not be preserved (will be overwritten).
  /// @param out     output buffer on GPU - will contain original image
  ///                in normalized range [-0.5, 0.5].
  /// @param sizeX   width of input image (in pixels)
  /// @param sizeY   height of input image (in pixels)
  /// @param levels  number of recursive DWT levels
  void rdwt97(float * in, float * out, int sizeX, int sizeY, int levels) {
    if(levels > 1) {
      // let this function recursively reverse transform deeper levels first
      const int llSizeX = divRndUp(sizeX, 2);
      const int llSizeY = divRndUp(sizeY, 2);
      rdwt97(in, out, llSizeX, llSizeY, levels - 1);
      
      // copy reverse transformed LL band from output back into the input
      memCopy(in, out, llSizeX, llSizeY);
    }
    
    // select right width of kernel for the size of the image
    if(sizeX >= 960) {
      launchRDWT97Kernel<192, 8>(in, out, sizeX, sizeY);
    } else if (sizeX >= 480) {
      launchRDWT97Kernel<128, 6>(in, out, sizeX, sizeY);
    } else {
      launchRDWT97Kernel<64, 6>(in, out, sizeX, sizeY);
    }
  }
  

  
} // end of namespace dwt_cuda
