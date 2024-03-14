#!/bin/bash

nvcc -arch=sm_35 -ftz=true -I../ -DSHMOO -o nbody-unroll nbody-unroll.cu

K=1024
for i in {1..10}; do
    echo "$(pwd)/nbody-unroll" $K
    ./nbody-unroll $K
    K=$(($K*2))
done
