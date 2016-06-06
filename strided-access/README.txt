###
### README for measuring effective memory bandwidth for strided array access
### by Karl Rupp
### 
### Supplements blog post:
### https://www.karlrupp.net/2016/02/strided-memory-access-on-cpus-gpus-and-mic
###

# License

The code is provided under a permissive MIT/X11-style license.
See file LICENSE.txt for details.

The results and plotting scripts in folder results/ are provided under the
Creative Commons Attribution 4.0 International (CC BY 4.0)
license, see results/LICENSE.txt


# Build

To build the executable, use (or adjust) one of the following commands to your environment:

HIP:
 $> /opt/rocm/hip/bin/hipcc -std=c++11 -O3 -o hip benchmark-hip.cpp 

CUDA:
 $> nvcc benchmark-cuda.cu -arch=sm_20 -I$VIENNACLPATH

OpenCL:
 $> g++ benchmark-opencl.cpp -I. -lOpenCL -L/usr/local/cuda/lib64/
 (If OpenCL is available system-wide, you may be able to omit the -L flag)

OpenMP:
 $> g++ benchmark-openmp.cpp benchmark-openmp2.cpp -I. -O3 -fopenmp
for CPUs or
 $> icc benchmark-openmp.cpp benchmark-openmp2.cpp -O3 -fopenmp -mmic
for Xeon Phi


# Run

To run the respective benchmark, issue
 $> ./a.out


# Plot

Have a look at the results/ folder, where the data and gnuplot commands are located.
Replot via
 $> gnuplot plot.gnuplot
(produces strided-access.eps)

Convert to .pdf via
 $> epstopdf strided-access.eps
and to .png using ImageMagick, e.g.
 $> convert -density 300 strided-access.eps -resize 1150x strided-access.png



