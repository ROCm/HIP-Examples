#!/bin/bash

if [ -z  "$HIP_PATH" ]
then

if [ -d /opt/rocm/hip ]
then
    HIP_PATH=/opt/rocm/hip
else
    echo "Please install rocm package"
fi

fi 

echo "Using HIP_PATH=$HIP_PATH"

if [ -f "rtm8_hip" ]
then
    rm rtm8_hip
fi

GCC_VER=4.8

echo "hipcc -I /usr/include/x86_64-linux-gnu -I /usr/include/x86_64-linux-gnu/c++/$GCC_VER -I /usr/include/c++/$GCC_VER -std=c++11 -O3 -o rtm8_hip rtm8.cpp"
$HIP_PATH/bin/hipcc -I /usr/include/x86_64-linux-gnu -I /usr/include/x86_64-linux-gnu/c++/$GCC_VER -I /usr/include/c++/$GCC_VER -std=c++11 -O3 -o rtm8_hip rtm8.cpp

