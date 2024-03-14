#!/bin/bash

gcc -std=c99 -O3 -fopenmp -DSHMOO -o nbody nbody.c -lm

K=1024
for i in {1..10}; do
    echo "$(pwd)/nbody" $K
    ./nbody $K
    K=$(($K*2))
done
