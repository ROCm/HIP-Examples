#!/bin/bash

#usage hipcmd.sh [MAKE_OPTIONS]

HIP_PATH=../../../..
HIPIFY="$HIP_PATH/bin/hipify -print-stats"

echo "==================================================================================="
$HIPIFY -no-translate-math cusrc/main.c > main.c

$HIPIFY cusrc/kernel/kernel_gpu_cuda.cu  > kernel/kernel_gpu_cuda.cu
$HIPIFY cusrc/kernel/kernel_gpu_cuda_2.cu > kernel/kernel_gpu_cuda_2.cu
$HIPIFY cusrc/kernel/kernel_gpu_cuda_wrapper.cu > kernel/kernel_gpu_cuda_wrapper.cu
$HIPIFY cusrc/kernel/kernel_gpu_cuda_wrapper.h > kernel/kernel_gpu_cuda_wrapper.h
$HIPIFY cusrc/kernel/kernel_gpu_cuda_wrapper_2.cu > kernel/kernel_gpu_cuda_wrapper_2.cu
$HIPIFY cusrc/kernel/kernel_gpu_cuda_wrapper_2.h > kernel/kernel_gpu_cuda_wrapper_2.h

$HIPIFY cusrc/util/cuda/cuda.cu > util/cuda/cuda.cu
echo "==================================================================================="
echo

make $@

