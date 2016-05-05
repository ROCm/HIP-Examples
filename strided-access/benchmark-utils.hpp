#ifndef BENCHMARK_UTILS_HPP_
#define BENCHMARK_UTILS_HPP_

/*
* Copyright (c) 2016, Karl Rupp
*
* License: MIT/X11, see file LICENSE.txt
*/

#include <stdexcept>

#ifdef _WIN32

#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#undef min
#undef max

class Timer
{
public:

  Timer() { QueryPerformanceFrequency(&freq); }

  void start() { QueryPerformanceCounter((LARGE_INTEGER*) &start_time); }

  double get() const
  {
    LARGE_INTEGER  end_time;
    QueryPerformanceCounter((LARGE_INTEGER*) &end_time);
    return (static_cast<double>(end_time.QuadPart) - static_cast<double>(start_time.QuadPart)) / static_cast<double>(freq.QuadPart);
  }

private:
  LARGE_INTEGER freq;
  LARGE_INTEGER start_time;
};

#else

#include <sys/time.h>

class Timer
{
public:

  Timer() : ts(0)
  {}

  void start()
  {
    struct timeval tval;
    gettimeofday(&tval, NULL);
    ts = tval.tv_sec * 1000000 + tval.tv_usec;
  }

  double get() const
  {
    struct timeval tval;
    gettimeofday(&tval, NULL);
    double end_time = tval.tv_sec * 1000000 + tval.tv_usec;

    return static_cast<double>(end_time-ts) / 1000000.0;
  }

private:
  double ts;
};


#endif

#endif
