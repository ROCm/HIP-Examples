#!/bin/bash

#usage hipcmd.sh [MAKE_OPTIONS]

HIP_PATH=../../../../
HIPIFY="$HIP_PATH/bin/hipify -print-stats"



echo "==================================================================================="
$HIPIFY  cusrc/ex_particle_CUDA_float_seq.cu > ex_particle_CUDA_float_seq.cu
$HIPIFY  cusrc/ex_particle_CUDA_naive_seq.cu > ex_particle_CUDA_naive_seq.cu
echo "==================================================================================="
echo


make

