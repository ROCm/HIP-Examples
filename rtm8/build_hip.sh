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

echo "hipcc -std=c++11 -O3 -o rtm8_hip rtm8.cpp"
$HIP_PATH/bin/hipcc -std=c++11 -O3 -o rtm8_hip rtm8.cpp

