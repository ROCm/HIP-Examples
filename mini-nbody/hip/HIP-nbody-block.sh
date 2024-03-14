#!/bin/bash

if ! [ -f nbody-block.cpp ]; then
    echo "Hipify the blocked cuda source code to hip compatible code"
    # Manually add the first argument onto the kernel argument list
    hipify nbody-block.cu | sed -e 's/^void bodyForce\(Body /void bodyForce(hipLaunchParm lp, Body/' > nbody-block.cpp
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

echo hipcc -I../ -DSHMOO -o nbody-block nbody-block.cpp
"${HIPCC}" -I../ -DSHMOO -o nbody-block nbody-block.cpp

K=1024
for i in {1..8}; do
    echo "$(pwd)/nbody-block" $K
    ./nbody-block $K
    K=$(($K*2))
done
