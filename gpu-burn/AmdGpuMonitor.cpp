/*
 * Public domain.  No warranty.
 * Ville Timonen 2013
 * edited by Timmy Liu for HIP API 01/2016
 */

#include <fstream>
#include <iostream>
#include "AmdGpuMonitor.h"

// ---------------------------------------------------------------------------
namespace gpuburn {

AmdGpuMonitor::AmdGpuMonitor(int id, std::string hwmon)
    : GpuMonitor(id), mHwmonPath(hwmon)
{
}

AmdGpuMonitor::~AmdGpuMonitor()
{
}

// ---------------------------------------------------------------------------

float AmdGpuMonitor::getTemperature()
{
    float gpuTemp = -1;

    std::ifstream tempFile((mHwmonPath + "/temp1_input").c_str());
    if (tempFile.is_open()) {
        tempFile >> gpuTemp;
        tempFile.close();
    }

    // Hwmon exposes temperatures in milliCelcius
    return gpuTemp / 1000.0f;
}

}; //namespace gpuburn
