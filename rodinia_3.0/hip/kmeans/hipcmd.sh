#!/bin/bash

#usage hipcmd.sh [MAKE_OPTIONS]

HIP_PATH=../../../../
HIPIFY="$HIP_PATH/bin/hipify -print-stats"



echo "==================================================================================="
$HIPIFY  cusrc/kmeans_cuda.cu > kmeans_cuda.cu
$HIPIFY  cusrc/kmeans_cuda_kernel.cu > kmeans_cuda_kernel.cu
echo "==================================================================================="
echo


make

