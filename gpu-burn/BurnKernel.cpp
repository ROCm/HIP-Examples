/*
 * Public domain.  No warranty.
 * Ville Timonen 2013
 * edited by Timmy Liu for HIP API 01/2016
 */

#include <iostream>
#include <thread>
#include "hip/hip_runtime.h"

#include "common.h"
#include "BurnKernel.h"

#include "rocblas.h"

#define EPSILOND 0.0000001f
// ---------------------------------------------------------------------------
namespace gpuburn {

constexpr int BurnKernel::cRandSeed;
constexpr float BurnKernel::cUseMem;
constexpr uint32_t BurnKernel::cRowSize;
constexpr uint32_t BurnKernel::cMatrixSize;
constexpr uint32_t BurnKernel::cBlockSize;
constexpr float BurnKernel::cAlpha;
constexpr float BurnKernel::cBeta;

BurnKernel::BurnKernel(int hipDevice)
    : mHipDevice(hipDevice), mRunKernel(false),
    mDeviceAdata(NULL), mDeviceBdata(NULL), mDeviceCdata(NULL)
{

    err_num = 0;

}

BurnKernel::~BurnKernel()
{
    if (mBurnThread){
        mBurnThread->join();
    }

    if (mDeviceAdata){
        hipFree(mDeviceAdata);
    }

    if (mDeviceBdata){
        hipFree(mDeviceBdata);
    }

    if (mDeviceCdata){
        hipFree(mDeviceCdata);
    }
}

// ---------------------------------------------------------------------------



extern "C" __global__ void hip_compare_kernel(double *C, int *faultyElems, size_t iters) 
{
        //column major NN
        size_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;
        size_t dim_x = gridDim.x * blockDim.x;

        size_t myIdx = idx_y * dim_x + idx_x;


	size_t iterStep = hipBlockDim_x*hipBlockDim_y*hipGridDim_x*hipGridDim_y;

        int myFaulty = 0;
	for (size_t i = 1; i < iters; ++i){
	        if(fabs(C[myIdx] - C[myIdx + iterStep]) > EPSILOND){
			myFaulty++;
		}
        }
	atomicAdd(faultyElems, myFaulty);
}


int BurnKernel::Init()
{
    int hipDevice = bindHipDevice();

    std::string msg = "Init Burn Thread for device (" + std::to_string(hipDevice) + ")\n";
    std::cout << msg;

    srand(cRandSeed);
    for (int i = 0; i < cMatrixSize; ++i) {
        mHostAdata[i] = (rand() % 1000000)/100000.0;
        mHostBdata[i] = (rand() % 1000000)/100000.0;
    }


    size_t freeMem = getAvailableMemory() * cUseMem;
    //size_t matrixSizeBytes = sizeof(float)*cMatrixSize;
    size_t matrixSizeBytes = sizeof(double)*cMatrixSize;
    mNumIterations = (freeMem - (matrixSizeBytes*2))/matrixSizeBytes;

    checkError(hipMalloc((void**)&mDeviceAdata, matrixSizeBytes), "Alloc A");
    checkError(hipMalloc((void**)&mDeviceBdata, matrixSizeBytes), "Alloc B");
    checkError(hipMalloc((void**)&mDeviceCdata, matrixSizeBytes*mNumIterations), "Alloc C");

    //rocky added for acc check:
    checkError(hipMalloc(&d_faultyElemData, sizeof(int)), "faulty data");

    checkError(hipMemcpy(mDeviceAdata, mHostAdata, matrixSizeBytes, hipMemcpyHostToDevice), "A -> device");
    checkError(hipMemcpy(mDeviceBdata, mHostBdata, matrixSizeBytes, hipMemcpyHostToDevice), "B -> device");
    checkError(hipMemset(mDeviceCdata, 0, matrixSizeBytes*mNumIterations), "C memset");

    return 0;
}

int BurnKernel::startBurn()
{
    mRunKernel = true;

    mBurnThread = make_unique<std::thread>(&BurnKernel::threadMain, this);
    return 0;
}

int BurnKernel::threadMain()
{
    int err = 0;
    int hipDevice = bindHipDevice();
    std::string msg = "Burn Thread using device (" + std::to_string(hipDevice) + ")\n";
    std::cout << msg;

    while (mRunKernel) {
        err = runComputeKernel();
    }

    return err;
}

int BurnKernel::stopBurn()
{
    int hipDevice = bindHipDevice();

    std::string msg = "Stopping burn thread on device (" + std::to_string(hipDevice) + ")\n";
    std::cout << msg;

    mRunKernel = false;
    return 0;
}

int BurnKernel::bindHipDevice()
{
    int hipDevice = -1;
    hipSetDevice(mHipDevice);
    hipGetDevice(&hipDevice);
    return hipDevice;
}

int BurnKernel::runComputeKernel()
{
    int err = 0;


    for (int i = 0; mRunKernel && i < mNumIterations; ++i) {

	double alpha = 1.1;
	double beta = 0.0;

    	rocblas_handle handle;
  	rocblas_create_handle(&handle);
        rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_transpose, cRowSize, cRowSize, cRowSize, &alpha, mDeviceAdata, cRowSize, mDeviceBdata,cRowSize , &beta, mDeviceCdata + i*cMatrixSize, cRowSize);
    }

    checkError(hipDeviceSynchronize(), "Sync"); // rocky added to fix seg fault

    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_compare_kernel),dim3(cRowSize/cBlockSize, cRowSize/cBlockSize, 1),dim3(cBlockSize,cBlockSize,1), 0, 0, mDeviceCdata, d_faultyElemData, mNumIterations);


    int *d_faultyElemsHost;
    checkError(hipMemcpy(d_faultyElemsHost, d_faultyElemData, sizeof(int), hipMemcpyDeviceToHost), "Read faultyelemdata");

    err_num += *d_faultyElemsHost;

    checkError(hipDeviceSynchronize(), "Sync");

    return err;
}

int BurnKernel::get_err_num(){
    return err_num;
}


size_t BurnKernel::getAvailableMemory()
{
    size_t freeMem, totalMem;
    checkError(hipMemGetInfo(&freeMem, &totalMem));
    return freeMem;
}

}; //namespace gpuburn
