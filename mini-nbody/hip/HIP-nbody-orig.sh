#!/bin/bash

if ! [ -f nbody-orig.cpp ]; then
    echo "Hipify the original cuda source code to hip compatible code"
    hipify nbody-orig.cu > nbody-orig.cpp
fi

HIPCC="$(command -v hipcc)"
if ! [ -x "${HIPCC}" ]; then
  if [ -z  "${HIP_PATH}" ]; then
    if [ -d /opt/rocm/hip ]; then
      HIP_PATH=/opt/rocm/hip
    else
      HIP_PATH=/opt/rocm
    fi
    HIPCC="${HIP_PATH}/bin/hipcc"
  fi
fi

echo hipcc -I../ -DSHMOO -o nbody-orig nbody-orig.cpp
"${HIPCC}" -I../ -DSHMOO -o nbody-orig nbody-orig.cpp

K=1024
for i in {1..10}; do
    echo "$(pwd)/nbody-orig" $K
    ./nbody-orig $K
    K=$(($K*2))
done
