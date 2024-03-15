#!/bin/bash

: ${HIP_PLATFORM:="hcc"}

echo
echo "==== vectorAdd ===="
(
cd vectorAdd
make clean
make test
)

echo
echo "==== gpu-burn ===="
(
cd gpu-burn
make clean
make && ./build/gpuburn-hip -t 5
)

echo
echo "==== strided-access ===="
(
cd strided-access
make clean
make test
)


echo
echo "==== rtm8 ===="
(
cd rtm8
make clean
make test
)

echo
echo "==== reduction ===="
(
cd reduction
make clean
make reduction
./run.sh
)

echo
echo "==== mini-nbody ===="
(
cd mini-nbody/hip
./HIP-nbody-orig.sh
./HIP-nbody-soa.sh
./HIP-nbody-block.sh
)

echo
echo "==== add4 ===="
(
cd add4
make clean
make test
)

echo
echo "==== cuda-stream ===="
(
cd cuda-stream
make clean
make test
)

echo
echo "==== OpenMP Hello World ===="
(
cd openmp-helloworld
mkdir -p build
cd build
cmake ..
make
./test_openmp_helloworld
)

echo
echo "==== HIP-Examples-Applications ===="
(
cd HIP-Examples-Applications
for SUBDIR in BinomialOption BitonicSort dct dwtHaar1D FastWalshTransform FloydWarshall HelloWorld Histogram MatrixMultiplication PrefixSum RecursiveGaussian SimpleConvolution; do
    echo
    echo "==== HIP-Examples-Applications/${SUBDIR} ===="
    (
    cd "${SUBDIR}"
    make clean
    make test
    )
done
)
