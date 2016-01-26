#!/bin/bash

#usage hipcmd.sh [MAKE_OPTIONS]

HIP_PATH=../../../../
HIPIFY="$HIP_PATH/bin/hipify -print-stats"



echo "==================================================================================="
$HIPIFY  cusrc/backprop_cuda.cu > backprop_cuda.cu
$HIPIFY  cusrc/backprop_cuda_kernel.cu > backprop_cuda_kernel.cu
echo "==================================================================================="
echo


make

