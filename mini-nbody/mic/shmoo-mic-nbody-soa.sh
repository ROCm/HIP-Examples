#!/bin/bash

icc -std=c99 -openmp -mmic -DSHMOO -I../ -o nbody-soa-mic nbody-soa.c

K=1024
for i in {1..10}; do
    echo "$(pwd)/nbody-soa-mic" $K
    ./nbody-soa-mic $K
    K=$(($K*2))
done
