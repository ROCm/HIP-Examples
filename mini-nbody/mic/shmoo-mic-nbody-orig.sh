#!/bin/bash

icc -std=c99 -openmp -mmic -DSHMOO -o nbody-orig-mic ../nbody-orig.c

K=1024
for i in {1..10}; do
    echo "$(pwd)/nbody-orig-mic" $K
    ./nbody-orig-mic $K
    K=$(($K*2))
done
