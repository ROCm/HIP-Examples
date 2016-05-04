#!/bin/bash
if [ -f "rtm8_cuda" ]
then
    rm rtm8_cuda
fi
echo "nvcc -O3 rtm8.cpp -o rtm8_cuda"
nvcc -O3 rtm8.cu -o rtm8_cuda
