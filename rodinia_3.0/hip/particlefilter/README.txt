Particle Filter Changelog
Donnie Newell
11Nov2011

OpenCL particlefilter 

Implementation Notes:
  The OpenCL version of particle filter is very close to the Cuda version. The main 
differences are that OpenCL does not support 1D texture memory, so the float version of the 
particle filter uses 2D images and translate 1D indexes into 2D coordinates when accessing 
texture memory. 

Additionally, the -cl-fast-math-relaxed flag was added to the compilation 
of the kernel, in an attempt to apply the same optimization that is in the CUDA Makefile 
for fast math operations.

********************************************************************************************
CUDA particlefilter

Changes:
  It was discovered that kernels were crashing in the second iteration in the float version 
of particle filter. This was due to incorrectly accessing texture memory that was not bound 
in the sum kernel. The appropriate binding of texture memory was implemented before the sum 
and this corrected the kernel crashes. As a result, the float version of the particle filter 
now takes longer to execute than before. This is because when a kernel crashes, all 
subsequent CUDA API calls fail and immediately return, resulting in a shorter execution 
time. 

Also, since kernel calls are asynchronous, a call to cudaThreadSynchronize was added 
before the timer call signifying the end of execution, in order to improve the accuracy of 
the times reported in the output.  

*******************************************************************************************
     
