/*
 * Public domain.  No warranty.
 * Ville Timonen 2013
 * edited by Timmy Liu for HIP API 01/2016
 */

#ifndef GPUBURN_AMDGPUMONITOR_H_
#define GPUBURN_AMDGPUMONITOR_H_

#include <string>
#include "GpuMonitor.h"

// ---------------------------------------------------------------------------
namespace gpuburn {

class AmdGpuMonitor : public GpuMonitor {
    public:
        /**
         * Initialize an AmdGpuMonitor instance
         *
         * @hwmonPath is the kernel hwmon resource associated to this GPU
         */
        AmdGpuMonitor(int id, std::string hwmonPath);
        virtual ~AmdGpuMonitor();

        virtual float getTemperature();

    private:
        std::string mHwmonPath;
};

}; // namespace gpuburn

// ---------------------------------------------------------------------------

#endif // GPUBURN_AMDGPUMONITOR_H_
