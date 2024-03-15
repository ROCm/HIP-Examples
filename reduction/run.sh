#!/bin/bash

K=$((1024*1024*4))
for i in {1..8}; do
    echo
    echo "$(pwd)/reduction" $K
    ./reduction $K
    K=$(($K*2))
done
