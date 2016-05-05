//
// OpenCL benchmark for measuring effective memory bandwidth for strided array access
//
// Author: Karl Rupp,  me@karlrupp.net
// License: MIT/X11 license, see file LICENSE.txt
//

#include <iostream>
#include <string>
#include <vector>
#include <sstream>

#include "benchmark-utils.hpp"

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

// OpenCL error checking
#define ERROR_CHECKER_CASE(ERRORCODE)  case ERRORCODE: throw std::runtime_error("#ERRORCODE");
static void checkError(cl_int err)
{
  if (err != CL_SUCCESS)
  {
    switch (err)
    {
      ERROR_CHECKER_CASE(CL_DEVICE_NOT_FOUND);
      ERROR_CHECKER_CASE(CL_DEVICE_NOT_AVAILABLE);
      ERROR_CHECKER_CASE(CL_COMPILER_NOT_AVAILABLE);
      ERROR_CHECKER_CASE(CL_MEM_OBJECT_ALLOCATION_FAILURE);
      ERROR_CHECKER_CASE(CL_OUT_OF_RESOURCES);
      ERROR_CHECKER_CASE(CL_OUT_OF_HOST_MEMORY);
      ERROR_CHECKER_CASE(CL_PROFILING_INFO_NOT_AVAILABLE);
      ERROR_CHECKER_CASE(CL_MEM_COPY_OVERLAP);
      ERROR_CHECKER_CASE(CL_IMAGE_FORMAT_MISMATCH);
      ERROR_CHECKER_CASE(CL_IMAGE_FORMAT_NOT_SUPPORTED);
      ERROR_CHECKER_CASE(CL_BUILD_PROGRAM_FAILURE);
      ERROR_CHECKER_CASE(CL_MAP_FAILURE);

      ERROR_CHECKER_CASE(CL_INVALID_VALUE);
      ERROR_CHECKER_CASE(CL_INVALID_DEVICE_TYPE);
      ERROR_CHECKER_CASE(CL_INVALID_PLATFORM);
      ERROR_CHECKER_CASE(CL_INVALID_DEVICE);
      ERROR_CHECKER_CASE(CL_INVALID_CONTEXT);
      ERROR_CHECKER_CASE(CL_INVALID_QUEUE_PROPERTIES);
      ERROR_CHECKER_CASE(CL_INVALID_COMMAND_QUEUE);
      ERROR_CHECKER_CASE(CL_INVALID_HOST_PTR);
      ERROR_CHECKER_CASE(CL_INVALID_MEM_OBJECT);
      ERROR_CHECKER_CASE(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
      ERROR_CHECKER_CASE(CL_INVALID_IMAGE_SIZE);
      ERROR_CHECKER_CASE(CL_INVALID_SAMPLER);
      ERROR_CHECKER_CASE(CL_INVALID_BINARY);
      ERROR_CHECKER_CASE(CL_INVALID_BUILD_OPTIONS);
      ERROR_CHECKER_CASE(CL_INVALID_PROGRAM);
      ERROR_CHECKER_CASE(CL_INVALID_PROGRAM_EXECUTABLE);
      ERROR_CHECKER_CASE(CL_INVALID_KERNEL_NAME);
      ERROR_CHECKER_CASE(CL_INVALID_KERNEL_DEFINITION);
      ERROR_CHECKER_CASE(CL_INVALID_KERNEL);
      ERROR_CHECKER_CASE(CL_INVALID_ARG_INDEX);
      ERROR_CHECKER_CASE(CL_INVALID_ARG_VALUE);
      ERROR_CHECKER_CASE(CL_INVALID_ARG_SIZE);
      ERROR_CHECKER_CASE(CL_INVALID_KERNEL_ARGS);
      ERROR_CHECKER_CASE(CL_INVALID_WORK_DIMENSION);
      ERROR_CHECKER_CASE(CL_INVALID_WORK_GROUP_SIZE);
      ERROR_CHECKER_CASE(CL_INVALID_WORK_ITEM_SIZE);
      ERROR_CHECKER_CASE(CL_INVALID_GLOBAL_OFFSET);
      ERROR_CHECKER_CASE(CL_INVALID_EVENT_WAIT_LIST);
      ERROR_CHECKER_CASE(CL_INVALID_EVENT);
      ERROR_CHECKER_CASE(CL_INVALID_OPERATION);
      ERROR_CHECKER_CASE(CL_INVALID_GL_OBJECT);
      ERROR_CHECKER_CASE(CL_INVALID_BUFFER_SIZE);
      ERROR_CHECKER_CASE(CL_INVALID_MIP_LEVEL);
      ERROR_CHECKER_CASE(CL_INVALID_GLOBAL_WORK_SIZE);
        
      default: throw std::runtime_error("Unknown error. Maybe OpenCL SDK not properly installed?");
    }
  }
}

#define ERR_CHECK(err) checkError(err);



// Kernel for the benchmark
static const char * benchmark_program =
"__kernel void elementwise_add(\n"
"          __global const float * x,\n"
"          __global const float * y, \n"
"          __global float * z,\n"
"          unsigned int stride,\n"
"          unsigned int size) \n"
"{ \n"
"  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))\n"
"    z[i*stride] = x[i*stride] + y[i*stride];\n"
"};\n";

int main(int argc, char **argv)
{
  typedef float       NumericT;

  /////////////////////////// Part 1: Initialize OpenCL ///////////////////////////////////
    
  //
  // Query platform:
  //
  cl_uint num_platforms;
  cl_platform_id platform_ids[42];   //no more than 42 platforms supported...
  cl_int err = clGetPlatformIDs(42, platform_ids, &num_platforms); ERR_CHECK(err);

  std::cout << "# Platforms found: " << num_platforms << std::endl;
  for (cl_uint i=0; i<num_platforms; ++i)
  {
    char buffer[1024];
    cl_int err;
    err = clGetPlatformInfo(platform_ids[i], CL_PLATFORM_VENDOR, 1024 * sizeof(char), buffer, NULL); ERR_CHECK(err);
    
    std::stringstream ss;
    ss << "# (" << i << ") " << buffer << ": ";

    err = clGetPlatformInfo(platform_ids[i], CL_PLATFORM_VERSION, 1024 * sizeof(char), buffer, NULL); ERR_CHECK(err);

    ss << buffer;

    std::cout << ss.str() << std::endl;
  }

  std::size_t platform_index = 0;
  if (num_platforms > 1)
  {
    std::cout << "# Enter platform index to use: ";
    std::cin >> platform_index;
    platform_index = std::min<std::size_t>(platform_index, num_platforms - 1);
    std::cout << "#" << std::endl;
  }
  
  //
  // Query devices:
  //
  cl_device_id device_ids[42];
  cl_uint num_devices;
  err = clGetDeviceIDs(platform_ids[platform_index], CL_DEVICE_TYPE_ALL, 42, device_ids, &num_devices); ERR_CHECK(err);
  std::cout << "# Devices found: " << num_devices << std::endl;
  for (cl_uint i=0; i<num_devices; ++i)
  {
    char buffer[1024]; 
    cl_int err;          
    err = clGetDeviceInfo(device_ids[i], CL_DEVICE_NAME, sizeof(char)*1024, &buffer, NULL); ERR_CHECK(err);
    
    std::cout << "# (" << i << ") " << buffer << std::endl;
  }

  std::size_t device_index = 0;
  if (num_devices > 1)
  {
    std::cout << "# Enter index of device to use: ";
    std::cin >> device_index;
    device_index = std::min<std::size_t>(device_index, num_devices - 1);
    std::cout << "#" << std::endl;
  }

  // now set up a context containing the selected device:
  cl_context my_context = clCreateContext(0, 1, &(device_ids[device_index]), NULL, NULL, &err); ERR_CHECK(err);
   
  // create a command queue for the device:
  cl_command_queue queue = clCreateCommandQueue(my_context, device_ids[device_index], 0, &err); ERR_CHECK(err);

  
  cl_program my_program = clCreateProgramWithSource(my_context, 1, &benchmark_program, NULL, &err); ERR_CHECK(err);
  err = clBuildProgram(my_program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    char buffer[8192];
    cl_build_status status;
    std::cout << "Build Scalar: Err = " << err;
    err = clGetProgramBuildInfo(my_program, device_ids[device_index], CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &status, NULL); ERR_CHECK(err);
    err = clGetProgramBuildInfo(my_program, device_ids[device_index], CL_PROGRAM_BUILD_LOG,    sizeof(char)*8192, &buffer, NULL); ERR_CHECK(err);
    std::cout << " Status = " << status << std::endl;
    std::cout << "Log: " << buffer << std::endl;
    std::cout << "Sources: " << benchmark_program << std::endl;
  }
  cl_kernel my_kernel = clCreateKernel(my_program, "elementwise_add", &err); ERR_CHECK(err);

  /////////////////////////// Part 2: Run benchmark ///////////////////////////////////
    

  // Set up work vectors
  cl_uint N = 1000000;
  std::vector<NumericT> host_x(32*N);
  cl_mem x = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 32*N*sizeof(NumericT), &(host_x[0]), &err); ERR_CHECK(err);
  cl_mem y = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 32*N*sizeof(NumericT), &(host_x[0]), &err); ERR_CHECK(err);
  cl_mem z = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 32*N*sizeof(NumericT), &(host_x[0]), &err); ERR_CHECK(err);

  // Warmup calculation:
  size_t localsize = 256;
  size_t globalsize = 256 * localsize;
  cl_uint stride = 1;
  err = clSetKernelArg(my_kernel, 0, sizeof(cl_mem), &x); ERR_CHECK(err);
  err = clSetKernelArg(my_kernel, 1, sizeof(cl_mem), &y); ERR_CHECK(err);
  err = clSetKernelArg(my_kernel, 2, sizeof(cl_mem), &z); ERR_CHECK(err);
  err = clSetKernelArg(my_kernel, 3, sizeof(cl_uint), &stride); ERR_CHECK(err);
  err = clSetKernelArg(my_kernel, 4, sizeof(cl_uint), &N); ERR_CHECK(err);
  err = clEnqueueNDRangeKernel(queue, my_kernel, 1, NULL, &globalsize, &localsize, 0, NULL, NULL); ERR_CHECK(err);

  // Benchmark runs
  Timer timer;
  char device_name[1024];
  err = clGetDeviceInfo(device_ids[device_index], CL_DEVICE_NAME, 1024, device_name, NULL); ERR_CHECK(err);
  std::cout << "# Using device: " << device_name << std::endl;
  std::cout << "# stride     time       GB/sec" << std::endl;
  for (; stride <= 32; ++stride)
  {
    err = clFinish(queue); ERR_CHECK(err);
    err = clSetKernelArg(my_kernel, 3, sizeof(cl_uint), &stride); ERR_CHECK(err);

    // repeat calculation several times, then average
    timer.start();
    for (std::size_t num_runs = 0; num_runs < 20; ++num_runs)
    {
      err = clEnqueueNDRangeKernel(queue, my_kernel, 1, NULL, &globalsize, &localsize, 0, NULL, NULL); ERR_CHECK(err);
    }
    err = clFinish(queue); ERR_CHECK(err);
    double exec_time = timer.get();

    std::cout << "     " << stride << "        " << exec_time << "        " <<  20.0 * 3.0 * sizeof(NumericT) * N / exec_time * 1e-9 << std::endl;
  }

  return EXIT_SUCCESS;
}

