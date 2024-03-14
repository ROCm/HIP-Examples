#!/bin/bash

nvcc -arch=sm_35 -ftz=true -I../ -DSHMOO -o nbody-ftz nbody-soa.cu

K=1024
for i in {1..10}; do
    echo "$(pwd)/nbody-ftz" $K
    ./nbody-ftz $K
    K=$(($K*2))
done
