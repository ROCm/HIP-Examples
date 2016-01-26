/// line 248 the index
/// @file    transform_buffer.h
/// @brief   Buffer with separated even and odd columns and related algorithms.
/// @author  Martin Jirman (207962@mail.muni.cz)
/// @date    2011-01-20 18:33
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


#ifndef TRANSFORM_BUFFER_H
#define	TRANSFORM_BUFFER_H


namespace dwt_cuda {
  
  
  /// Buffer (in shared memory of GPU) where block of input image is stored,
  /// but odd and even lines are separated. (Generates less bank conflicts when 
  /// using lifting schema.) All operations expect SIZE_X threads.
  /// Also implements basic building blocks of lifting schema.
  /// @tparam SIZE_X      width of the buffer excluding two boundaries (Also
  ///                     a number of threads participating on all operations.)
  ///                     Must be divisible by 4.
  /// @tparam SIZE_Y      height of buffer (total number of lines)
  /// @tparam BOUNDARY_X  number of extra pixels at the left and right side
  ///                     boundary is expected to be smaller than half SIZE_X
  ///                     Must be divisible by 2.
  template <typename T, int SIZE_X, int SIZE_Y, int BOUNDARY_X>
  class TransformBuffer {
  public:
    enum {
      /// difference between pointers to two vertical neigbors
      VERTICAL_STRIDE = BOUNDARY_X + (SIZE_X / 2)
    };
    
  private:
    enum {
      /// number of shared memory banks - needed for correct padding
      #ifdef __CUDA_ARCH__
      SHM_BANKS = ((__CUDA_ARCH__ >= 200) ? 32 : 16),
      #else
      SHM_BANKS = 16,  // for host code only - can be anything, won't be used
      #endif
      
      /// size of one of two buffers (odd or even)
      BUFFER_SIZE = VERTICAL_STRIDE * SIZE_Y,
      
      /// unused space between two buffers
      PADDING = SHM_BANKS - ((BUFFER_SIZE + SHM_BANKS / 2) % SHM_BANKS),
      
      /// offset of the odd columns buffer from the beginning of data buffer
      ODD_OFFSET = BUFFER_SIZE + PADDING,
    };

    /// buffer for both even and odd columns
    T data[2 * BUFFER_SIZE + PADDING];
    
    
    
    /// Applies specified function to all central elements while also passing
    /// previous and next elements as parameters.
    /// @param count         count of central elements to apply function to
    /// @param prevOffset    offset of first central element
    /// @param midOffset     offset of first central element's predecessor
    /// @param nextOffset    offset of first central element's successor
    /// @param function      the function itself
    template <typename FUNC>
    __device__ void horizontalStep(const int count, const int prevOffset, 
                                   const int midOffset, const int nextOffset,
                                   const FUNC & function) {
      // number of unchecked iterations
      const int STEPS = count / SIZE_X;
      
      // items remaining after last unchecked iteration
      const int finalCount = count % SIZE_X; 
      
      // offset of items processed in last (checked) iteration
      const int finalOffset = count - finalCount;  
      
      // all threads perform fixed number of iterations ...
      for(int i = 0; i < STEPS; i++) {
      // for(int i = 0; i < 3; i++) {
        const T previous = data[prevOffset + i * SIZE_X + threadIdx.x];
        const T next     = data[nextOffset + i * SIZE_X + threadIdx.x];
        T & center       = data[midOffset  + i * SIZE_X + threadIdx.x];
        // function(previous, center, (nextOffset + i*SIZE_X+threadIdx.x));
        function(previous, center, next);// the real one
      }
      
      // ... but not all threads participate on final iteration
      if(threadIdx.x < finalCount) {
        const T previous = data[prevOffset + finalOffset + threadIdx.x];
        const T next     = data[nextOffset + finalOffset + threadIdx.x];
        T & center = data[midOffset + finalOffset + threadIdx.x];
        // function(previous, center, (nextOffset+finalOffset+threadIdx.x));
        // kaixi
        function(previous, center, next);//the real one
      }
    }

  public:
    
    /// Gets offset of the column with given index. Central columns have 
    /// indices from 0 to NUM_LINES - 1, left boundary columns have negative 
    /// indices and right boundary columns indices start with NUM_LINES.
    /// @param columnIndex  index of column to get pointer to
    /// @return  offset of the first item of column with specified index
    __device__ int getColumnOffset(int columnIndex) {
      columnIndex += BOUNDARY_X;             // skip boundary
      return columnIndex / 2                 // select right column
          + (columnIndex & 1) * ODD_OFFSET;  // select odd or even buffer
    }
    
    
    /// Provides access to data of the transform buffer.
    /// @param index  index of the item to work with
    /// @return reference to item at given index
    __device__ T & operator[] (const int index) {
      return data[index];
    }
    
            
    /// Applies specified function to all horizontally even elements in 
    /// specified lines. (Including even elements in boundaries except 
    /// first even element in first left boundary.) SIZE_X threads participate 
    /// and synchronization is needed before result can be used.
    /// @param firstLine  index of first line
    /// @param numLines   count of lines
    /// @param func       function to be applied on all even elements
    ///                   parameters: previous (odd) element, the even
    ///                   element itself and finally next (odd) element
    template <typename FUNC>
    __device__ void forEachHorizontalEven(const int firstLine,
                                          const int numLines,
                                          const FUNC & func) {
      // number of even elemens to apply function to
      const int count = numLines * VERTICAL_STRIDE - 1;
      // offset of first even element
      const int centerOffset = firstLine * VERTICAL_STRIDE + 1;
      // offset of odd predecessor of first even element
      const int prevOffset = firstLine * VERTICAL_STRIDE + ODD_OFFSET;
      // offset of odd successor of first even element
      const int nextOffset = prevOffset + 1;
      
      // call generic horizontal step function
      horizontalStep(count, prevOffset, centerOffset, nextOffset, func);
    }
    
    
    /// Applies given function to all horizontally odd elements in specified
    /// lines. (Including odd elements in boundaries except last odd element
    /// in last right boundary.) SIZE_X threads participate and synchronization
    /// is needed before result can be used.
    /// @param firstLine  index of first line
    /// @param numLines   count of lines
    /// @param func       function to be applied on all odd elements
    ///                   parameters: previous (even) element, the odd
    ///                   element itself and finally next (even) element
    template <typename FUNC>
    __device__ void forEachHorizontalOdd(const int firstLine,
                                         const int numLines,
                                         const FUNC & func) {
      // numbet of odd elements to apply function to
      const int count = numLines * VERTICAL_STRIDE - 1;
      // offset of even predecessor of first odd element
      const int prevOffset = firstLine * VERTICAL_STRIDE;
      // offset of first odd element
      const int centerOffset = prevOffset + ODD_OFFSET;
      // offset of even successor of first odd element
      const int nextOffset = prevOffset + 1;
      
      // call generic horizontal step function
      horizontalStep(count, prevOffset, centerOffset, nextOffset, func);
    }
    
    
    /// Applies specified function to all even elements (except element #0)
    /// of given column. Each thread takes care of one column, so there's 
    /// no need for synchronization.
    /// @param columnOffset  offset of thread's column
    /// @param f             function to be applied on all even elements
    ///                      parameters: previous (odd) element, the even
    ///                      element itself and finally next (odd) element
    template <typename F>
    __device__ void forEachVerticalEven(const int columnOffset, const F & f) {
      if(SIZE_Y > 3) { // makes no sense otherwise
        const int steps = SIZE_Y / 2 - 1;
        for(int i = 0; i < steps; i++) {
          const int row = 2 + i * 2;
          const T prev = data[columnOffset + (row - 1) * VERTICAL_STRIDE];
          const T next = data[columnOffset + (row + 1) * VERTICAL_STRIDE];
          f(prev, data[columnOffset + row * VERTICAL_STRIDE] , next);
		  
		  //--------------- FOR TEST -----------------
/*		__syncthreads();
		if ((blockIdx.x * blockDim.x + threadIdx.x) == 0){
			diffOut[2500]++;
			diffOut[diffOut[2500]] = 2;//data[columnOffset + row * VERTICAL_STRIDE];
		}	
		__syncthreads();
*/		  //--------------- FOR TEST -----------------
		  
		  
        }
      }
    }
    
    
    /// Applies specified function to all odd elements of given column.
    /// Each thread takes care of one column, so there's no need for
    /// synchronization.
    /// @param columnOffset  offset of thread's column
    /// @param f             function to be applied on all odd elements
    ///                      parameters: previous (even) element, the odd
    ///                      element itself and finally next (even) element
    template <typename F>
    __device__ void forEachVerticalOdd(const int columnOffset, const F & f) {
      const int steps = (SIZE_Y - 1) / 2;
      for(int i = 0; i < steps; i++) {
        const int row = i * 2 + 1;
        const T prev = data[columnOffset + (row - 1) * VERTICAL_STRIDE];
        const T next = data[columnOffset + (row + 1) * VERTICAL_STRIDE];

		f(prev, data[columnOffset + row * VERTICAL_STRIDE], next);
		
		
		  //--------------- FOR TEST -----------------
/*		__syncthreads();
		if ((blockIdx.x * blockDim.x + threadIdx.x) == 0){
			diffOut[2500]++;
			diffOut[diffOut[2500]] = 1; //data[columnOffset + row * VERTICAL_STRIDE];
		}

		__syncthreads();
*/		  //--------------- FOR TEST -----------------
      }
    }
    
    
    
    /// Scales elements at specified lines.
    /// @param evenScale  scaling factor for horizontally even elements
    /// @param oddScale   scaling factor for horizontally odd elements
    /// @param numLines   number of lines, whose elements should be scaled
    /// @param firstLine  index of first line to scale elements in
    __device__ void scaleHorizontal(const T evenScale, const T oddScale,
                                    const int firstLine, const int numLines) {
      const int offset = firstLine * VERTICAL_STRIDE;
      const int count = numLines * VERTICAL_STRIDE;
      const int steps = count / SIZE_X;
      const int finalCount = count % SIZE_X;
      const int finalOffset = count - finalCount;
      
      // run iterations, whete all threads participate
      for(int i = 0; i < steps; i++) {
        data[threadIdx.x + i * SIZE_X + offset] *= evenScale;
        data[threadIdx.x + i * SIZE_X + offset + ODD_OFFSET] *= oddScale;
      }
      
      // some threads also finish remaining unscaled items
      if(threadIdx.x < finalCount) {
        data[threadIdx.x + finalOffset + offset] *= evenScale;
        data[threadIdx.x + finalOffset + offset + ODD_OFFSET] *= oddScale;
      }
    }
    
    
    /// Scales elements in specified column.
    /// @param evenScale     scaling factor for vertically even elements
    /// @param oddScale      scaling factor for vertically odd elements
    /// @param columnOffset  offset of the column to work with
    /// @param numLines      number of lines, whose elements should be scaled
    /// @param firstLine     index of first line to scale elements in
    __device__ void scaleVertical(const T evenScale, const T oddScale,
                                  const int columnOffset, const int numLines,
                                  const int firstLine) {
      for(int i = firstLine; i < (numLines + firstLine); i++) {
        if(i & 1) {
          data[columnOffset + i * VERTICAL_STRIDE] *= oddScale;
        } else {
          data[columnOffset + i * VERTICAL_STRIDE] *= evenScale;
        }
      }
    }
	
	
	//****************For Test(Feb23), test inter parameters*************
	__device__ int getVERTICAL_STRIDE(){
		return VERTICAL_STRIDE;
	}
	__device__ int getSHM_BANKS(){
		return SHM_BANKS;
	}
	__device__ int  getBuffersize(){		
		return BUFFER_SIZE;
	}
	__device__ int getPADDING(){
		return PADDING;
	}
	__device__ int getODD_OFFSET(){
		return ODD_OFFSET;
	}


    //****************For Test(Feb23), test inter parameters*************
	
	
  };  // end of class TransformBuffer


} // namespace dwt_cuda


#endif	// TRANSFORM_BUFFER_H

