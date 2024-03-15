#!/bin/bash

nvcc -arch=sm_35 -ftz=true -I../ -DSHMOO -o nbody-block nbody-block.cu

K=1024
for i in {1..10}; do
    echo "$(pwd)/nbody-block" $K
    ./nbody-block $K
    K=$(($K*2))
done
