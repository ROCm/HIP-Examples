#!/bin/bash

#usage hipcmd.sh [MAKE_OPTIONS]

HIP_PATH=../../../../
HIPIFY="$HIP_PATH/bin/hipify -print-stats"



echo "==================================================================================="
$HIPIFY  cusrc/main.cu > main.cu
$HIPIFY  cusrc/kernel.cu > kernel.cu
$HIPIFY  cusrc/setdevice.cu > setdevice.cu
echo "==================================================================================="
echo


make

