#!/bin/bash

#usage hipcmd.sh [MAKE_OPTIONS]

HIP_PATH=../../../../
HIPIFY="$HIP_PATH/bin/hipify -stats"



echo "==================================================================================="
$HIPIFY  cusrc/common.h                    > common.h
$HIPIFY  cusrc/components.cu               > components.cu
$HIPIFY  cusrc/components.h                > components.h
$HIPIFY  cusrc/dwt.cu                      > dwt.cu
$HIPIFY  cusrc/dwt.h                       > dwt.h
$HIPIFY  cusrc/main.cu                     > main.cu
$HIPIFY  cusrc/dwt_cuda/common.cu          > dwt_cuda/common.cu
$HIPIFY  cusrc/dwt_cuda/common.h           > dwt_cuda/common.h
$HIPIFY  cusrc/dwt_cuda/dwt.h              > dwt_cuda/dwt.h
$HIPIFY  cusrc/dwt_cuda/fdwt53.cu          > dwt_cuda/fdwt53.cu
$HIPIFY  cusrc/dwt_cuda/fdwt97.cu          > dwt_cuda/fdwt97.cu
$HIPIFY  cusrc/dwt_cuda/io.h               > dwt_cuda/io.h
$HIPIFY  cusrc/dwt_cuda/rdwt53.cu          > dwt_cuda/rdwt53.cu
$HIPIFY  cusrc/dwt_cuda/rdwt97.cu          > dwt_cuda/rdwt97.cu
$HIPIFY  cusrc/dwt_cuda/transform_buffer.h > dwt_cuda/transform_buffer.h
echo "==================================================================================="
echo


make

