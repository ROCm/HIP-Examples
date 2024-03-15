#!/bin/bash

icc -std=c99 -openmp -mmic -fimf-domain-exclusion=8 -DSHMOO -I../ -o nbody-align-mic nbody-align.c

K=1024
for i in {1..10}; do
    echo "$(pwd)/nbody-align-mic" $K
    ./nbody-align-mic $K
    K=$(($K*2))
done
