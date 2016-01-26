#!/bin/bash

#usage hipcmd.sh [MAKE_OPTIONS]

HIP_PATH=../../../../
HIPIFY="$HIP_PATH/bin/hipify -print-stats"



echo "==================================================================================="
$HIPIFY  cusrc/srad_v1/compress_kernel.cu  > srad_v1/compress_kernel.cu
$HIPIFY  cusrc/srad_v1/define.c            > srad_v1/define.c
$HIPIFY  cusrc/srad_v1/device.c            > srad_v1/device.c
$HIPIFY  cusrc/srad_v1/extract_kernel.cu   > srad_v1/extract_kernel.cu
$HIPIFY  cusrc/srad_v1/graphics.c          > srad_v1/graphics.c
$HIPIFY  cusrc/srad_v1/main.cu             > srad_v1/main.cu
$HIPIFY  cusrc/srad_v1/prepare_kernel.cu   > srad_v1/prepare_kernel.cu
$HIPIFY  cusrc/srad_v1/reduce_kernel.cu    > srad_v1/reduce_kernel.cu
$HIPIFY  cusrc/srad_v1/resize.c            > srad_v1/resize.c
$HIPIFY  cusrc/srad_v1/srad2_kernel.cu     > srad_v1/srad2_kernel.cu
$HIPIFY  cusrc/srad_v1/srad_kernel.cu      > srad_v1/srad_kernel.cu
$HIPIFY  cusrc/srad_v1/timer.c             > srad_v1/timer.c

$HIPIFY  cusrc/srad_v2/srad.cu             > srad_v2/srad.cu
$HIPIFY  cusrc/srad_v2/srad.h              > srad_v2/srad.h
$HIPIFY  cusrc/srad_v2/srad_kernel.cu      > srad_v2/srad_kernel.cu
echo "==================================================================================="
echo

