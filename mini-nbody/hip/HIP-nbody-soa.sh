#!/bin/bash

if ! [ -f nbody-soa.cpp ]; then
    echo "Hipify the original cuda source code to hip compatible code"
    # Manually add the first argument onto the kernel argument list
    hipify nbody-soa.cu | sed -e 's/^void bodyForce\(Body /void bodyForce(hipLaunchParm lp, Body/' > nbody-soa.cpp
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

echo hipcc -I../ -DSHMOO -o nbody-soa nbody-soa.cpp
"${HIPCC}" -I../ -DSHMOO -o nbody-soa nbody-soa.cpp

K=1024
for i in {1..8}; do
    echo "$(pwd)/nbody-soa" $K
    ./nbody-soa $K
    K=$(($K*2))
done
