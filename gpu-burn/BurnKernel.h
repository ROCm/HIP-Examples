/*
 * Public domain.  No warranty.
 * Ville Timonen 2013
 * edited by Timmy Liu for HIP API 01/2016
 */

#ifndef GPUBURN_BURNKERNEL_H_
#define GPUBURN_BURNKERNEL_H_

#include <thread>

// ---------------------------------------------------------------------------
namespace gpuburn {

/**
 * The Gpu class abstracts interactions with the hardware
 */
class BurnKernel {
    public:
        BurnKernel(int hipDevice);
        ~BurnKernel();

        int mHipDevice;

        int Init();

        /**
         * Run a stress workload on mHipDevice
         */
        int startBurn();

        /**
         * Stop the stress workload
         */
        int stopBurn();

    private:
        static constexpr int cRandSeed = 10;
        static constexpr float cUseMem = 0.80;
        static constexpr uint32_t cRowSize = 512;
        static constexpr uint32_t cMatrixSize = cRowSize * cRowSize;
        static constexpr uint32_t cBlockSize = 16;
        static constexpr float cAlpha = 1.0f;
        static constexpr float cBeta = 0.0f;

        float mHostAdata[cMatrixSize];
        float mHostBdata[cMatrixSize];

        float* mDeviceAdata;
        float* mDeviceBdata;
        float* mDeviceCdata;

        bool mRunKernel;
        int mNumIterations;

        std::unique_ptr<std::thread> mBurnThread;

        int bindHipDevice();
        int threadMain();
        int runComputeKernel();
        size_t getAvailableMemory();

};

}; // namespace gpuburn

// ---------------------------------------------------------------------------

#endif // GPUBURN_BURNKERNEL_H_
