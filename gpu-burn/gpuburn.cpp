/*
 * Public domain.  No warranty.
 * Ville Timonen 2013
 * edited by Timmy Liu for HIP API 01/2016
 */
#include <iostream>
#include <sstream>
#include <fstream>
#include <memory>
#include <vector>
#include <unistd.h>
#include <sys/stat.h>
#include "hip/hip_runtime.h"

#include "common.h"
#include "AmdGpuMonitor.h"
#include "BurnKernel.h"

// ---------------------------------------------------------------------------
using namespace gpuburn;

std::vector<std::unique_ptr<BurnKernel>> genBurnKernels()
{
    int deviceCount = 0;
    std::vector<std::unique_ptr<BurnKernel>> kernels;

    try {
        checkError(hipGetDeviceCount(&deviceCount));
        std::cout<<"Total no. of GPUs found: "<<deviceCount<<std::endl;
    } catch (std::string e) {
        std::cerr << "Error: couldn't find any HIP devices\n";
    }

    for (int i =0; i < deviceCount; ++i) {
        try {
            std::unique_ptr<BurnKernel> kernel(new BurnKernel(i));
            kernel->Init();
            kernels.push_back(std::move(kernel));
        } catch (std::string e) {
            std::cerr << e;
            std::cerr << "Error: failed to initialize hip device " << i << "\n";
        }
    }

    return kernels;
}

std::vector<std::unique_ptr<GpuMonitor>> genGpuMonitors()
{
    int deviceCount = 0;
    std::vector<std::unique_ptr<GpuMonitor>> monitors;

    for (int i = 0; true; i++) {
            struct stat dirInfo;
            std::string hwmonDir = "/sys/class/hwmon/hwmon" + std::to_string(i);

            if (stat(hwmonDir.c_str(), &dirInfo))
                break;

            std::string hwmonName;
            std::ifstream hwmon(hwmonDir + "/name");

            if (!hwmon.good())
                continue;

            hwmon >> hwmonName;
            if (hwmonName == "amdgpu") {
                GpuMonitor* monitor = new AmdGpuMonitor(i, "/sys/class/hwmon/hwmon" + std::to_string(i));
                std::unique_ptr<GpuMonitor> uniq_monitor(monitor);
                monitors.push_back(std::move(uniq_monitor));
            }
    }

    return monitors;
}

int doBurn(int burnSec) {
    std::vector<std::unique_ptr<BurnKernel>> burnKernels = genBurnKernels();
    std::vector<std::unique_ptr<GpuMonitor>> gpuMonitors = genGpuMonitors();

    if (burnKernels.size() == 0)
        return -ENOENT;

    for (auto& kernel : burnKernels) {
        kernel->startBurn();
    }

    for (; burnSec > 0; --burnSec) {
        std::ostringstream msg;
        msg << "Temps: ";
        for (auto& monitor : gpuMonitors) {
            msg << "[GPU" << monitor->getId() << ": " << monitor->getTemperature() << " C] ";
        }
        msg << burnSec << "s\n";
        std::cout << msg.str();
        sleep(1);
    }

    for (auto& kernel : burnKernels) {
        kernel->stopBurn();
    }

    return 0;
}

int main(int argc, char **argv) {
    int opt;
    int burnSec = 10;

    while ((opt = getopt (argc, argv, "ht:")) != -1)
        switch (opt)
        {
            case 't':
                burnSec = atoi(optarg);
                break;
            case 'h':
            default:
                std::cerr << "Usage: " << argv[0] << " [-t sec]\n";
                return -EINVAL;
        }

    return doBurn(burnSec);
}


// ---------------------------------------------------------------------------
