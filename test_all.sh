#!/bin/bash

: ${HIP_PLATFORM:="hcc"}

# vector_add
echo
echo "==== vectorAdd ===="
cd vectorAdd
make clean
make
cd ..

# gpu-burn
echo
echo "==== gpu-burn ===="
cd gpu-burn
make clean
make
./build/gpuburn-hip -t 5
cd ..

# strided-access
echo
echo "==== strided-access ===="
cd strided-access
make clean
make
./strided-access
cd ..


# rtm8
echo
echo "==== rtm8 ===="
cd rtm8
./build_hip.sh
./rtm8_hip
cd ..

# reduction
echo
echo "==== reduction ===="
cd reduction
make clean
make
./run.sh
cd ..

# mini-nbody
echo
echo "==== mini-nbody ===="
cd mini-nbody/hip
./HIP-nbody-orig.sh
./HIP-nbody-soa.sh
./HIP-nbody-block.sh
cd ../..

# add4
echo
echo "==== add4 ===="
cd add4
./buildit.sh
./runhip.sh
cd ..

# cuda-stream
echo
echo "==== cuda-stream ===="
cd cuda-stream
make clean
make
./stream
cd ..

#---
echo
: ${HIP_MAKEJ:="4"}
echo "==== Rodinia ===="
cd rodinia_3.0/hip
make clean -j ${HIP_MAKEJ}
make test 
cd ..



