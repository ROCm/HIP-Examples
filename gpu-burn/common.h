/*
 * Public domain.  No warranty.
 * Ville Timonen 2013
 * edited by Timmy Liu for HIP API 01/2016
 */

#ifndef GPUBURN_COMMON_H_
#define GPUBURN_COMMON_H_

// ---------------------------------------------------------------------------
namespace gpuburn {

/**
 * c++11 doesn't support make_unique, which is very convenient
 *  Refer to: https://herbsutter.com/gotw/_102/
 */
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

int checkError(hipError_t err, std::string desc = "");

}; // namespace common

// ---------------------------------------------------------------------------

#endif // GPUBURN_COMMON_H_
