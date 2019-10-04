//
// CUDA benchmark for measuring effective memory bandwidth for strided array access
//
// Author: Karl Rupp,  me@karlrupp.net
// License: MIT/X11 license, see file LICENSE.txt
//

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>

#include "hip/hip_runtime.h"

#include "benchmark-utils.hpp"


inline void cuda_last_error_check()
{
  hipError_t error_code = hipGetLastError();

  if (hipSuccess != error_code)
  {
    std::stringstream ss;
    ss << "CUDA Runtime API error " << error_code << ": " << hipGetErrorString( error_code ) << std::endl;
    throw std::runtime_error(ss.str());
  }
}


// Kernel for the benchmark
template<typename NumericT>
__global__ void elementwise_add(hipLaunchParm lp,
                                const NumericT * x,
                                const NumericT * y,
                                      NumericT * z,
                                unsigned int stride,
                                unsigned int size)
{
  for (unsigned int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
                    i < size;
                    i += hipGridDim_x * hipBlockDim_x)
    z[i*stride] = x[i*stride] + y[i*stride];
}


int main(int argc, char **argv)
{
  typedef float       NumericT;

  hipDeviceProp_t prop;
  hipError_t err = hipGetDeviceProperties(&prop, 0); if (err != hipSuccess) throw std::runtime_error("Failed to get CUDA device name");
  std::cout << "# Using device: " << prop.name << std::endl;

  // Set up work vectors
  std::size_t N =  1000000;

  std::vector<NumericT> host_x(32*N);
  NumericT *x, *y, *z;

  err = hipMalloc(&x, sizeof(NumericT) * 32 * N); if (err != hipSuccess) throw std::runtime_error("Failed to allocate CUDA memory for x");
  err = hipMalloc(&y, sizeof(NumericT) * 32 * N); if (err != hipSuccess) throw std::runtime_error("Failed to allocate CUDA memory for y");
  err = hipMalloc(&z, sizeof(NumericT) * 32 * N); if (err != hipSuccess) throw std::runtime_error("Failed to allocate CUDA memory for z");


  // Warmup calculation:
  hipLaunchKernel(elementwise_add, dim3(256), dim3(256), 0, 0, x, y, z,
                                static_cast<unsigned int>(1),
                                static_cast<unsigned int>(N));
  cuda_last_error_check();

  // Benchmark runs
  Timer timer;
  std::cout << "# stride     time       GB/sec" << std::endl;
  for (std::size_t stride = 1; stride <= 32 ; ++stride)
  {
    hipDeviceSynchronize();
    timer.start();

    // repeat calculation several times, then average
    for (std::size_t num_runs = 0; num_runs < 20; ++num_runs)
    {
      hipLaunchKernel(elementwise_add, dim3(256), dim3(256), 0, 0, x, y, z,
                                    static_cast<unsigned int>(stride),
                                    static_cast<unsigned int>(N));
      cuda_last_error_check();
    }
    hipDeviceSynchronize();
    double exec_time = timer.get();

    std::cout << "   " << stride << "        " << exec_time << "        " << 20 * 3.0 * sizeof(NumericT) * N / exec_time * 1e-9 << std::endl;
  }

  return EXIT_SUCCESS;
}

