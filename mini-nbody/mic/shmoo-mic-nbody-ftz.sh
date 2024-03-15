#!/bin/bash

icc -std=c99 -openmp -mmic -fimf-domain-exclusion=8 -DSHMOO -I../ -o nbody-ftz-mic nbody-soa.c

K=1024
for i in {1..10}; do
    echo "$(pwd)/nbody-ftz-mic" $K
    ./nbody-ftz-mic $K
    K=$(($K*2))
done
