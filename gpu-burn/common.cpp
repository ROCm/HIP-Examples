/*
 * Public domain.  No warranty.
 * Ville Timonen 2013
 * edited by Timmy Liu for HIP API 01/2016
 */

#include <hip_runtime.h>
#include "common.h"

// ---------------------------------------------------------------------------
namespace gpuburn {

#define HIP_ERROR(var, code, str) \
    case code: \
        var = str; \
        break;
#define HIP_ERROR_SIMPLE(var, code) HIP_ERROR(var, code, #code)
int checkError(int err, std::string desc)
{
    if (err == hipSuccess)
        return 0;

    std::string errStr = "Unknown error: " + std::to_string(err);
    switch(err) {
        HIP_ERROR_SIMPLE(errStr, hipErrorMemoryAllocation)
        HIP_ERROR_SIMPLE(errStr, hipErrorLaunchOutOfResources)
        HIP_ERROR_SIMPLE(errStr, hipErrorInvalidValue)
        HIP_ERROR_SIMPLE(errStr, hipErrorInvalidResourceHandle)
        HIP_ERROR_SIMPLE(errStr, hipErrorInvalidDevice)
        HIP_ERROR_SIMPLE(errStr, hipErrorNoDevice)
        HIP_ERROR_SIMPLE(errStr, hipErrorNotReady)
        HIP_ERROR_SIMPLE(errStr, hipErrorUnknown)
        HIP_ERROR_SIMPLE(errStr, hipErrorTbd)
    }

    std::string errorMessage = "";
    if (desc == "")
        throw "Error: " + errStr + "\n";
    else
        throw "Error in \"" + desc + "\": " + errStr + "\n";

    return err;
}

}; // namespace common

// ---------------------------------------------------------------------------
