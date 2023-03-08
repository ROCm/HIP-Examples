#!/bin/bash

if [ -f "rtm8_hip" ]
then
    rm rtm8_hip
fi

echo "hipcc -std=c++11 -O3 -o rtm8_hip rtm8.cpp"
/opt/rocm/bin/hipcc -std=c++11 -O3 -o rtm8_hip rtm8.cpp

