#!/bin/bash

nvcc -arch=sm_35 -I../ -DSHMOO -o nbody-orig nbody-orig.cu

K=1024
for i in {1..10}; do
    echo "$(pwd)/nbody-orig" $K
    ./nbody-orig $K
    K=$(($K*2))
done
