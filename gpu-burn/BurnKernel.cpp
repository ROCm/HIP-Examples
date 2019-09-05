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
}

BurnKernel::~BurnKernel()
{
    if (mBurnThread)
        mBurnThread->join();

    if (mDeviceAdata)
        hipFree(mDeviceAdata);

    if (mDeviceBdata)
        hipFree(mDeviceBdata);

    if (mDeviceCdata)
        hipFree(mDeviceCdata);
}

// ---------------------------------------------------------------------------

extern "C" __global__ void hip_sgemm_kernel(const int M,
                                            const int N, const int K,
                                            const float alpha,
                                            float *A, const int lda, float *B,
                                            const int ldb, const float beta,
                                            float *C, const int ldc)
{
        //column major NN
        size_t idx_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        size_t idx_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
        size_t dim_x = hipGridDim_x * hipBlockDim_x;
        size_t myIdx = idx_y * dim_x + idx_x;

        float local_c = beta * C[myIdx];

        for(int k = 0; k < K; k++) {
          local_c += alpha * A[ idx_y + k * K] * B[ idx_x * K + k];
        }

        C[myIdx] = local_c;
}

// ---------------------------------------------------------------------------

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
    size_t matrixSizeBytes = sizeof(float)*cMatrixSize;
    mNumIterations = (freeMem - (matrixSizeBytes*2))/matrixSizeBytes;

    checkError(hipMalloc((void**)&mDeviceAdata, matrixSizeBytes), "Alloc A");
    checkError(hipMalloc((void**)&mDeviceBdata, matrixSizeBytes), "Alloc B");
    checkError(hipMalloc((void**)&mDeviceCdata, matrixSizeBytes*mNumIterations), "Alloc C");

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
        hipLaunchKernelGGL(
            /* Launch params */
            hip_sgemm_kernel,
            dim3(cRowSize/cBlockSize, cRowSize/cBlockSize, 1),
            dim3(cBlockSize,cBlockSize,1), 0, 0,
            /* Kernel params */
            cRowSize, cRowSize, cRowSize, cAlpha,
            mDeviceAdata, cRowSize,
            mDeviceBdata, cRowSize,
            cBeta,
            mDeviceCdata + i*cMatrixSize,
            cRowSize);
    }
    checkError(hipDeviceSynchronize(), "Sync");

    return err;
}

size_t BurnKernel::getAvailableMemory()
{
    size_t freeMem, totalMem;
    checkError(hipMemGetInfo(&freeMem, &totalMem));
    return freeMem;
}

}; //namespace gpuburn
