#!/bin/bash

nvcc -arch=sm_35 -I../ -DSHMOO -o nbody-soa nbody-soa.cu

K=1024
for i in {1..10}; do
    echo "$(pwd)/nbody-soa" $K
    ./nbody-soa $K
    K=$(($K*2))
done
