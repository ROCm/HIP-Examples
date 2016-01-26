///
/// @file:   io.h
/// @brief   Manages loading and saving lineary stored bands and input images.
/// @author  Martin Jirman (207962@mail.muni.cz)
/// @date    2011-01-20 22:38
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


#ifndef IO_H
#define	IO_H


#include "common.h"

namespace dwt_cuda {

  
  /// Base for all IO classes - manages mirroring.
  class DWTIO {
  protected:
    /// Handles mirroring of image at edges in a DWT correct way.
    /// @param d      a position in the image (will be replaced by mirrored d)
    /// @param sizeD  size of the image along the dimension of 'd'
    __device__ static void mirror(int & d, const int & sizeD) {
      // TODO: enable multiple mirroring:
//      if(sizeD > 1) {
//        if(d < 0) {
//          const int underflow = -1 - d;
//          const int phase = (underflow / (sizeD - 1)) & 1;
//          const int remainder = underflow % (sizeD - 1);
//          if(phase == 0) {
//            d = remainder + 1;
//          } else {
//            d = sizeD - 2 - remainder;
//          }
//        } else if(d >= sizeD) {
//          const int overflow = d - sizeD;
//          const int phase = (overflow / (sizeD - 1)) & 1;
//          const int remainder = overflow % (sizeD - 1);
//          if(phase == 0) {
//            d = sizeD - 2 - remainder;
//          } else {
//            d = remainder + 1;
//          }
//        }
//      } else {
//        d = 0;
//      }
  //for test the mirror's use Feb 17
      if(d >= sizeD) {
        d = 2 * sizeD - 2 - d;
      } else if(d < 0) {
        d = -d;
      }
    }
  };


  /// Base class for pixel loader and writer - manages computing start index,
  /// stride and end of image for loading column of pixels.
  /// @tparam T        type of image pixels
  /// @tparam CHECKED  true = be prepared to image boundary, false = don't care
  template <typename T, bool CHECKED>
  class VerticalDWTPixelIO : protected DWTIO {
  protected:
    int end;         ///< index of bottom neightbor of last pixel of column
    int stride;      ///< increment of pointer to get to next pixel

    /// Initializes pixel IO - sets end index and a position of first pixel.
    /// @param sizeX   width of the image
    /// @param sizeY   height of the image
    /// @param firstX  x-coordinate of first pixel to use
    /// @param firstY  y-coordinate of first pixel to use
    /// @return index of pixel at position [x, y] in the image
    __device__ int initialize(const int sizeX, const int sizeY,
                              int firstX, int firstY) {
      // initialize all pointers and stride
      end = CHECKED ? (sizeY * sizeX + firstX) : 0;
      stride = sizeX;
      return firstX + sizeX * firstY;
    }
  };



  /// Writes reverse transformed pixels directly into output image.
  /// @tparam T        type of output pixels
  /// @tparam CHECKED  true = be prepared to image boundary, false = don't care
  template <typename T, bool CHECKED>
  class VerticalDWTPixelWriter : VerticalDWTPixelIO<T, CHECKED> {
  private:
    int next;   // index of the next pixel to be loaded

  public:
    /// Initializes writer - sets output buffer and a position of first pixel.
    /// @param sizeX   width of the image
    /// @param sizeY   height of the image
    /// @param firstX  x-coordinate of first pixel to write into
    /// @param firstY  y-coordinate of first pixel to write into
    __device__ void init(const int sizeX, const int sizeY, 
                         int firstX, int firstY) {
      if(firstX < sizeX) {
        next = this->initialize(sizeX, sizeY, firstX, firstY);
      } else {
        this->end = 0;
        this->stride = 0;
        next = 0;
      }
    }

    /// Writes given value at next position and advances internal pointer while
    /// correctly handling mirroring.
    /// @param output  output image to write pixel into
    /// @param value   value of the pixel to be written
    __device__ void writeInto(T * const output, const T & value) {
      if((!CHECKED) || (next != this->end)) {
        output[next] = value;
        next += this->stride;
      }
    }
  };


  
  /// Loads pixels from input image.
  /// @tparam T        type of image input pixels
  /// @tparam CHECKED  true = be prepared to image boundary, false = don't care
  template <typename T, bool CHECKED>
  class VerticalDWTPixelLoader
          : protected VerticalDWTPixelIO<const T, CHECKED> {
  private:
    int last;  ///< index of last loaded pixel
  public:
  

  //******************* FOR TEST **********************
  __device__ int getlast(){
		return last;
	}
  __device__ int getend(){
		return this->end;
	}
  __device__ int getstride(){
		return this->stride;
	}
  __device__ void setend(int a){
      this->end=a;
	}
	//******************* FOR TEST **********************
  
  
  
    /// Initializes loader - sets input size and a position of first pixel.
    /// @param sizeX   width of the image
    /// @param sizeY   height of the image
    /// @param firstX  x-coordinate of first pixel to load
    /// @param firstY  y-coordinate of first pixel to load
    __device__ void init(const int sizeX, const int sizeY,
                         int firstX, int firstY) {
      // correctly mirror x coordinate
      this->mirror(firstX, sizeX);
      
      // 'last' always points to already loaded pixel (subtract sizeX = stride)
      last = this->initialize(sizeX, sizeY, firstX, firstY) - sizeX;
      //last = (FirstX + sizeX * FirstY) - sizeX
    }
    
    /// Sets all fields to zeros, for compiler not to complain about
    /// uninitialized stuff.
    __device__ void clear() {
      this->end = 0;
      this->stride = 0;
      this->last = 0;
    }

    /// Gets another pixel and advancees internal pointer to following one.
    /// @param input  input image to load next pixel from
    /// @return next pixel from given image
    __device__ T loadFrom(const T * const input) {
      last += this->stride;
      if(CHECKED && (last == this->end)) {
        last -= 2 * this->stride;
        this->stride = -this->stride; // reverse loader's direction
      }
      // avoid reading from negative indices if loader is checked
      // return (CHECKED && (last < 0)) ? 0 : input[last];  // TODO: use this checked variant later
      return input[last];
      // return this->end;
      // return last;
      // return this->stride;
    }
  };



  /// Base for band write and loader. Manages computing strides and pointers
  /// to first and last pixels in a linearly-stored-bands correct way.
  /// @tparam T        type of band coefficients
  /// @tparam CHECKED  true = be prepared to image boundary, false = don't care
  template <typename T, bool CHECKED>
  class VerticalDWTBandIO : protected DWTIO {
  protected:
    /// index of bottom neighbor of last pixel of loaded column
    int end;
    
    /// increment of index to get from highpass band to the lowpass one
    int strideHighToLow;
    
    /// increment of index to get from the lowpass band to the highpass one
    int strideLowToHigh;

    /// Initializes IO - sets size of image and a position of first pixel.
    /// @param imageSizeX   width of the image
    /// @param imageSizeY   height of the image
    /// @param firstX       x-coordinate of first pixel to use
    ///                     (Parity determines vertically low or high band.)
    /// @param firstY       y-coordinate of first pixel to use
    ///                     (Parity determines horizontally low or high band.)
    /// @return index of first item specified by firstX and firstY
    __device__ int initialize(const int imageSizeX, const int imageSizeY,
                              int firstX, int firstY) {
      // index of first pixel (topmost one) of the column with index firstX
      int columnOffset = firstX / 2;
      
      // difference between indices of two vertically neighboring pixels
      // in the same band
      int verticalStride;
      
      // resolve index of first pixel according to horizontal parity
      if(firstX & 1) {
        // first pixel in one of right bands
        verticalStride = imageSizeX / 2;
        columnOffset += divRndUp(imageSizeX, 2) * divRndUp(imageSizeY, 2);
        strideLowToHigh = (imageSizeX * imageSizeY) / 2;
      } else {
        // first pixel in one of left bands
        verticalStride = imageSizeX / 2 + (imageSizeX & 1);
        strideLowToHigh = divRndUp(imageSizeY, 2)  * imageSizeX;
      }
      
      // set the other stride
      strideHighToLow = verticalStride - strideLowToHigh;

      // compute index of coefficient which indicates end of image
      if(CHECKED) {
        end = columnOffset                            // right column
                + (imageSizeY / 2) * verticalStride   // right row
                + (imageSizeY & 1) * strideLowToHigh; // possibly in high band
      } else {
        end = 0;
      }


	//***********for test**************
	//	end = CHECKED;
	//***********for test**************
	
	
      // finally, return index of the first item
      return columnOffset                        // right column
              + (firstY / 2) * verticalStride    // right row
              + (firstY & 1) * strideLowToHigh;  // possibly in high band
    }
  };




  /// Directly loads coefficients from four consecutively stored transformed
  /// bands.
  /// @tparam T        type of input band coefficients
  /// @tparam CHECKED  true = be prepared to image boundary, false = don't care
  template <typename T, bool CHECKED>
  class VerticalDWTBandLoader : public VerticalDWTBandIO<const T, CHECKED> {
  private:
    int last;  ///< index of last loaded pixel

    /// Checks internal index and possibly reverses direction of loader.
    /// (Handles mirroring at the bottom of the image.)
    /// @param input   input image to load next coefficient from
    /// @param stride  stride to use now (one of two loader's strides)
    /// @return loaded coefficient
    __device__ T updateAndLoad(const T * const input, const int & stride) {
      last += stride;
      if(CHECKED && (last == this->end)) {
        // undo last two updates of index (to get to previous mirrored item)
        last -= (this->strideLowToHigh + this->strideHighToLow);

        // swap and reverse strides (to move up in the loaded column now)
        const int temp = this->strideLowToHigh;
        this->strideLowToHigh = -this->strideHighToLow;
        this->strideHighToLow = -temp;
      }
      // avoid reading from negative indices if loader is checked
      // return (CHECKED && (last < 0)) ? 0 : input[last];  // TODO: use this checked variant later
      return input[last];
    }
  public:

    /// Initializes loader - sets input size and a position of first pixel.
    /// @param imageSizeX   width of the image
    /// @param imageSizeY   height of the image
    /// @param firstX       x-coordinate of first pixel to load
    ///                     (Parity determines vertically low or high band.)
    /// @param firstY       y-coordinate of first pixel to load
    ///                     (Parity determines horizontally low or high band.)
    __device__ void init(const int imageSizeX, const int imageSizeY,
                         int firstX, const int firstY) {
      this->mirror(firstX, imageSizeX);
      last = this->initialize(imageSizeX, imageSizeY, firstX, firstY);
      
      // adjust to point to previous item
      last -= (firstY & 1) ? this->strideLowToHigh : this->strideHighToLow; 
    }
    
    /// Sets all fields to zeros, for compiler not to complain about
    /// uninitialized stuff.
    __device__ void clear() {
      this->end = 0;
      this->strideHighToLow = 0;
      this->strideLowToHigh = 0;
      this->last = 0;
    }

    /// Gets another coefficient from lowpass band and advances internal index.
    /// Call this method first if position of first pixel passed to init
    /// was in high band.
    /// @param input   input image to load next coefficient from
    /// @return next coefficient from the lowpass band of the given image
    __device__ T loadLowFrom(const T * const input) {
      return updateAndLoad(input, this->strideHighToLow);
    }

    /// Gets another coefficient from the highpass band and advances index.
    /// Call this method first if position of first pixel passed to init
    /// was in high band.
    /// @param input   input image to load next coefficient from
    /// @return next coefficient from the highbass band of the given image
    __device__ T loadHighFrom(const T * const input) {
      return updateAndLoad(input, this->strideLowToHigh);
    }

  };




  /// Directly saves coefficients into four transformed bands.
  /// @tparam T        type of output band coefficients
  /// @tparam CHECKED  true = be prepared to image boundary, false = don't care
  template <typename T, bool CHECKED>
  class VerticalDWTBandWriter : public VerticalDWTBandIO<T, CHECKED> {
  private:
    int next;  ///< index of last loaded pixel

    /// Checks internal index and possibly stops the writer.
    /// (Handles mirroring at edges of the image.)
    /// @param output  output buffer
    /// @param item    item to put into the output
    /// @param stride  increment of the pointer to get to next output index
    __device__ int saveAndUpdate(T * const output, const T & item,
                                  const int & stride) {
//	if(blockIdx.x == 0 && blockIdx.y == 11 && threadIdx.x == 0){		//test, Mar 20					  
      if((!CHECKED) || (next != this->end)) {
        output[next] = item;
        next += stride;
      } 
//	}
      // if((!CHECKED) || (next != this->end)) { //the real one
        // output[next] = item;
        // next += stride;  //stride has been test
      // } 
	return next;
    }
  public:

    /// Initializes writer - sets output size and a position of first pixel.
    /// @param output       output image
    /// @param imageSizeX   width of the image
    /// @param imageSizeY   height of the image
    /// @param firstX       x-coordinate of first pixel to write
    ///                     (Parity determines vertically low or high band.)
    /// @param firstY       y-coordinate of first pixel to write
    ///                     (Parity determines horizontally low or high band.)
    __device__ void init(const int imageSizeX, const int imageSizeY,
                         const int firstX, const int firstY) {
      if (firstX < imageSizeX) {
        next = this->initialize(imageSizeX, imageSizeY, firstX, firstY);
      } else {
        clear();
      }
    }
    
    /// Sets all fields to zeros, for compiler not to complain about
    /// uninitialized stuff.
    __device__ void clear() {
      this->end = 0;
      this->strideHighToLow = 0;
      this->strideLowToHigh = 0;
      this->next = 0;
    }

    /// Writes another coefficient into the band which was specified using
    /// init's firstX and firstY parameters and advances internal pointer.
    /// Call this method first if position of first pixel passed to init
    /// was in lowpass band.
    /// @param output  output image
    /// @param low     lowpass coefficient to save into the lowpass band
    __device__ int writeLowInto(T * const output, const T & primary) {
      return saveAndUpdate(output, primary, this->strideLowToHigh);
    }

    /// Writes another coefficient from the other band and advances pointer.
    /// Call this method first if position of first pixel passed to init
    /// was in highpass band.
    /// @param output  output image
    /// @param high    highpass coefficient to save into the highpass band
    __device__ int writeHighInto(T * const output, const T & other) {
      return saveAndUpdate(output, other, this->strideHighToLow);
    }

	//*******Add three functions to get private values*******
	__device__ int getnext(){
		return next;
	}
	
	__device__ int getend(){
		return this->end;
	}
	
	__device__ int getstrideHighToLow(){
		return this->strideHighToLow;
	}
	
	__device__ int getstrideLowToHigh(){
		return this->strideLowToHigh;
	}
	
	//*******Add three functions to get private values*******
  };
  
  
  
} // namespace dwt_cuda


#endif	// IO_H

