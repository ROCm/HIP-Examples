#!/bin/bash

icc -std=c99 -openmp -mmic -fimf-domain-exclusion=8 -DSHMOO -I../ -o nbody-block-mic nbody-block.c

K=1024
for i in {1..10}; do
    echo "$(pwd)/nbody-block-mic" $K
    ./nbody-block-mic $K
    K=$(($K*2))
done
