/// 
/// @file    rdwt53.cu
/// @brief   CUDA implementation of reverse 5/3 2D DWT.
/// @author  Martin Jirman (207962@mail.muni.cz)
/// @date    2011-02-04 14:19
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

  

  /// Wraps shared momory buffer and algorithms needed for computing 5/3 RDWT
  /// using sliding window and lifting schema.
  /// @tparam WIN_SIZE_X  width of sliding window
  /// @tparam WIN_SIZE_Y  height of sliding window
  template <int WIN_SIZE_X, int WIN_SIZE_Y>
  class RDWT53 {
  private: 
    
    /// Shared memory buffer used for 5/3 DWT transforms.
    typedef TransformBuffer<int, WIN_SIZE_X, WIN_SIZE_Y + 3, 2> RDWT53Buffer;

    /// Shared buffer used for reverse 5/3 DWT.
    RDWT53Buffer buffer;

    /// Difference between indices of two vertically neighboring items in buffer.
    enum { STRIDE = RDWT53Buffer::VERTICAL_STRIDE };


    /// Info needed for loading of one input column from input image.
    /// @tparam CHECKED  true if loader should check boundaries
    template <bool CHECKED>
    struct RDWT53Column {
      /// loader of pixels from column in input image
      VerticalDWTBandLoader<int, CHECKED> loader;
      
      /// Offset of corresponding column in shared buffer.
      int offset;
      
      /// Sets all fields to some values to avoid 'uninitialized' warnings.
      __device__ void clear() {
        offset = 0;
        loader.clear();
      }
    };


    /// 5/3 DWT reverse update operation.
    struct Reverse53Update {
      __device__ void operator() (const int p, int & c, const int n) const {
        c -= (p + n + 2) / 4;  // F.3, page 118, ITU-T Rec. T.800 final draft
      }
    };


    /// 5/3 DWT reverse predict operation.
    struct Reverse53Predict {
      __device__ void operator() (const int p, int & c, const int n) const {
        c += (p + n) / 2;      // F.4, page 118, ITU-T Rec. T.800 final draft
      }
    };


    /// Horizontal 5/3 RDWT on specified lines of transform buffer.
    /// @param lines      number of lines to be transformed
    /// @param firstLine  index of the first line to be transformed
    __device__ void horizontalTransform(const int lines, const int firstLine) {
      __syncthreads();
      buffer.forEachHorizontalEven(firstLine, lines, Reverse53Update());
      __syncthreads();
      buffer.forEachHorizontalOdd(firstLine, lines, Reverse53Predict());
      __syncthreads();
    }


    /// Using given loader, it loads another WIN_SIZE_Y coefficients
    /// into specified column.
    /// @tparam CHECKED  true if loader should check image boundaries
    /// @param input     input coefficients to load from
    /// @param col       info about loaded column
    template <bool CHECKED>
    inline __device__ void loadWindowIntoColumn(const int * const input,
                                                RDWT53Column<CHECKED> & col) {
      for(int i = 3; i < (3 + WIN_SIZE_Y); i += 2) {
        buffer[col.offset + i * STRIDE] = col.loader.loadLowFrom(input);
        buffer[col.offset + (i + 1) * STRIDE] = col.loader.loadHighFrom(input);
      }
    }


    /// Initializes one column of shared transform buffer with 7 input pixels.
    /// Those 7 pixels will not be transformed. Also initializes given loader.
    /// @tparam CHECKED  true if loader should check image boundaries
    /// @param columnX   x coordinate of column in shared transform buffer
    /// @param input     input image
    /// @param sizeX     width of the input image
    /// @param sizeY     height of the input image
    /// @param loader    (uninitialized) info about loaded column
    template <bool CHECKED>
    __device__ void initColumn(const int columnX, const int * const input, 
                               const int sizeX, const int sizeY,
                               RDWT53Column<CHECKED> & column,
                               const int firstY) {
      // coordinates of the first coefficient to be loaded
      const int firstX = blockIdx.x * WIN_SIZE_X + columnX;

      // offset of the column with index 'colIndex' in the transform buffer
      column.offset = buffer.getColumnOffset(columnX);

      if(blockIdx.y == 0) {
        // topmost block - apply mirroring rules when loading first 3 rows
        column.loader.init(sizeX, sizeY, firstX, firstY);

        // load pixels in mirrored way
        buffer[column.offset + 1 * STRIDE] = column.loader.loadLowFrom(input);
        buffer[column.offset + 0 * STRIDE] =
        buffer[column.offset + 2 * STRIDE] = column.loader.loadHighFrom(input);
      } else {
        // non-topmost row - regular loading:
        column.loader.init(sizeX, sizeY, firstX, firstY - 1);
        buffer[column.offset + 0 * STRIDE] = column.loader.loadHighFrom(input);
        buffer[column.offset + 1 * STRIDE] = column.loader.loadLowFrom(input);
        buffer[column.offset + 2 * STRIDE] = column.loader.loadHighFrom(input);
      }
      // Now, the next coefficient, which will be loaded by loader, is #2.
    }


    /// Actual GPU 5/3 RDWT implementation.
    /// @tparam CHECKED_LOADS   true if boundaries must be checked when reading
    /// @tparam CHECKED_WRITES  true if boundaries must be checked when writing
    /// @param in        input image (5/3 transformed coefficients)
    /// @param out       output buffer (for reverse transformed image)
    /// @param sizeX     width of the output image 
    /// @param sizeY     height of the output image
    /// @param winSteps  number of sliding window steps
    template<bool CHECKED_LOADS, bool CHECKED_WRITES>
    __device__ void transform(const int * const in, int * const out,
                              const int sizeX, const int sizeY,
                              const int winSteps) {
      // info about one main and one boundary column
      RDWT53Column<CHECKED_LOADS> column, boundaryColumn;

      // index of first row to be transformed
      const int firstY = blockIdx.y * WIN_SIZE_Y * winSteps;

      // some threads initialize boundary columns
      boundaryColumn.clear();
      if(threadIdx.x < 3) {
        // First 3 threads also handle boundary columns. Thread #0 gets right
        // column #0, thread #1 get right column #1 and thread #2 left column.
        const int colId = threadIdx.x + ((threadIdx.x != 2) ? WIN_SIZE_X : -3);

        // Thread initializes offset of the boundary column (in shared 
        // buffer), first 3 pixels of the column and a loader for this column.
        initColumn(colId, in, sizeX, sizeY, boundaryColumn, firstY);
      }

      // All threads initialize central columns.
      initColumn(parityIdx<WIN_SIZE_X>(), in, sizeX, sizeY, column, firstY);

      // horizontally transform first 3 rows
      horizontalTransform(3, 0);

      // writer of output pixels - initialize it
      const int outX = blockIdx.x * WIN_SIZE_X + threadIdx.x;
      VerticalDWTPixelWriter<int, CHECKED_WRITES> writer;
      writer.init(sizeX, sizeY, outX, firstY);

      // offset of column (in transform buffer) saved by this thread
      const int outputColumnOffset = buffer.getColumnOffset(threadIdx.x);

      // (Each iteration assumes that first 3 rows of transform buffer are
      // already loaded with horizontally transformed pixels.)
      for(int w = 0; w < winSteps; w++) {
        // Load another WIN_SIZE_Y lines of this thread's column
        // into the transform buffer.
        loadWindowIntoColumn(in, column);

        // possibly load boundary columns
        if(threadIdx.x < 3) {
          loadWindowIntoColumn(in, boundaryColumn);
        }

        // horizontally transform all newly loaded lines
        horizontalTransform(WIN_SIZE_Y, 3);

        // Using 3 registers, remember current values of last 3 rows 
        // of transform buffer. These rows are transformed horizontally 
        // only and will be used in next iteration.
        int last3Lines[3];
        last3Lines[0] = buffer[outputColumnOffset + (WIN_SIZE_Y + 0) * STRIDE];
        last3Lines[1] = buffer[outputColumnOffset + (WIN_SIZE_Y + 1) * STRIDE];
        last3Lines[2] = buffer[outputColumnOffset + (WIN_SIZE_Y + 2) * STRIDE];

        // vertically transform all central columns
        buffer.forEachVerticalOdd(outputColumnOffset, Reverse53Update());
        buffer.forEachVerticalEven(outputColumnOffset, Reverse53Predict());

        // Save all results of current window. Results are in transform buffer
        // at rows from #1 to #(1 + WIN_SIZE_Y). Other rows are invalid now.
        // (They only served as a boundary for vertical RDWT.)
        for(int i = 1; i < (1 + WIN_SIZE_Y); i++) {
          writer.writeInto(out, buffer[outputColumnOffset + i * STRIDE]);
        }

        // Use last 3 remembered lines as first 3 lines for next iteration.
        // As expected, these lines are already horizontally transformed.
        buffer[outputColumnOffset + 0 * STRIDE] = last3Lines[0];
        buffer[outputColumnOffset + 1 * STRIDE] = last3Lines[1];
        buffer[outputColumnOffset + 2 * STRIDE] = last3Lines[2];

        // Wait for all writing threads before proceeding to loading new
        // coeficients in next iteration. (Not to overwrite those which
        // are not written yet.)
        __syncthreads();
      }
    }


  public:
    /// Main GPU 5/3 RDWT entry point.
    /// @param in     input image (5/3 transformed coefficients)
    /// @param out    output buffer (for reverse transformed image)
    /// @param sizeX  width of the output image 
    /// @param sizeY  height of the output image
    /// @param winSteps  number of sliding window steps
    __device__ static void run(const int * const input, int * const output,
                               const int sx, const int sy, const int steps) {
      // prepare instance with buffer in shared memory
      __shared__ RDWT53<WIN_SIZE_X, WIN_SIZE_Y> rdwt53;

      // Compute limits of this threadblock's block of pixels and use them to
      // determine, whether this threadblock will have to deal with boundary.
      // (1 in next expressions is for radius of impulse response of 5/3 RDWT.)
      const int maxX = (blockIdx.x + 1) * WIN_SIZE_X + 1;
      const int maxY = (blockIdx.y + 1) * WIN_SIZE_Y * steps + 1;
      const bool atRightBoudary = maxX >= sx;
      const bool atBottomBoudary = maxY >= sy;

      // Select specialized version of code according to distance of this
      // threadblock's pixels from image boundary.
      if(atBottomBoudary) {
        // near bottom boundary => check both writing and reading
        rdwt53.transform<true, true>(input, output, sx, sy, steps);
      } else if(atRightBoudary) {
        // near right boundary only => check writing only
        rdwt53.transform<false, true>(input, output, sx, sy, steps);
      } else {
        // no nearby boundary => check nothing
        rdwt53.transform<false, false>(input, output, sx, sy, steps);
      }
    }

  }; // end of class RDWT53
  
  
  
  /// Main GPU 5/3 RDWT entry point.
  /// @param in     input image (5/3 transformed coefficients)
  /// @param out    output buffer (for reverse transformed image)
  /// @param sizeX  width of the output image 
  /// @param sizeY  height of the output image
  /// @param winSteps  number of sliding window steps
  template <int WIN_SX, int WIN_SY>
  __launch_bounds__(WIN_SX, CTMIN(SHM_SIZE/sizeof(RDWT53<WIN_SX, WIN_SY>), 8))
  __global__ void rdwt53Kernel(const int * const in, int * const out,
                               const int sx, const int sy, const int steps) {
    RDWT53<WIN_SX, WIN_SY>::run(in, out, sx, sy, steps);
  }
  
  
  
  /// Only computes optimal number of sliding window steps, 
  /// number of threadblocks and then lanches the 5/3 RDWT kernel.
  /// @tparam WIN_SX  width of sliding window
  /// @tparam WIN_SY  height of sliding window
  /// @param in       input image
  /// @param out      output buffer
  /// @param sx       width of the input image 
  /// @param sy       height of the input image
  template <int WIN_SX, int WIN_SY>
  void launchRDWT53Kernel (int * in, int * out, const int sx, const int sy) {
    // compute optimal number of steps of each sliding window
    const int steps = divRndUp(sy, 15 * WIN_SY);
    
    // prepare grid size
    dim3 gSize(divRndUp(sx, WIN_SX), divRndUp(sy, WIN_SY * steps));
    
    // finally transform this level
    PERF_BEGIN
    rdwt53Kernel<WIN_SX, WIN_SY><<<gSize, WIN_SX>>>(in, out, sx, sy, steps);
    PERF_END("        RDWT53", sx, sy)
    CudaDWTTester::checkLastKernelCall("RDWT 5/3 kernel");
  }
    
  
  
  /// Reverse 5/3 2D DWT. See common rules (above) for more details.
  /// @param in      Input DWT coefficients. Format described in common rules.
  ///                Will not be preserved (will be overwritten).
  /// @param out     output buffer on GPU - will contain original image
  ///                in normalized range [-128, 127].
  /// @param sizeX   width of input image (in pixels)
  /// @param sizeY   height of input image (in pixels)
  /// @param levels  number of recursive DWT levels
  void rdwt53(int * in, int * out, int sizeX, int sizeY, int levels) {
    if(levels > 1) {
      // let this function recursively reverse transform deeper levels first
      const int llSizeX = divRndUp(sizeX, 2);
      const int llSizeY = divRndUp(sizeY, 2);
      rdwt53(in, out, llSizeX, llSizeY, levels - 1);
      
      // copy reverse transformed LL band from output back into the input
      memCopy(in, out, llSizeX, llSizeY);
    }
    
    // select right width of kernel for the size of the image
    if(sizeX >= 960) {
      launchRDWT53Kernel<192, 8>(in, out, sizeX, sizeY);
    } else if (sizeX >= 480) {
      launchRDWT53Kernel<128, 8>(in, out, sizeX, sizeY);
    } else {
      launchRDWT53Kernel<64, 8>(in, out, sizeX, sizeY);
    }
  }
  

} // end of namespace dwt_cuda
