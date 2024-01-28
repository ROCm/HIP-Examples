#!/bin/bash

: ${HIP_PLATFORM:="hcc"}

# Look for hipcc
if [ -z "$HIP_PATH" ]; then
    # User path first
    which_hipcc=`which hipcc`
    if [ $? = 0 ]; then
	D=`dirname $which_hipcc`
	# strip the assumed /bin
	HIP_PATH=${D%/*}
    else
	if [ -x /opt/rocm/hip/bin/hipcc ]; then
	    HIP_PATH=/opt/rocm/hip
	elif [ -x /opt/rocm/bin/hipcc ]; then
	    HIP_PATH=/opt/rocm
	elif [ -x /usr/bin/hipcc ]; then
	    HIP_PATH=/usr
	fi
    fi
    if [ ! -z "$HIP_PATH" ]; then
	echo "Setting HIP_PATH to $HIP_PATH"
	sleep 3
	export HIP_PATH
    fi
fi

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

# openmp-helloworld
echo
echo "==== OpenMP Hello World ===="
cd openmp-helloworld
mkdir -p build
cd build
cmake ..
make
./test_openmp_helloworld
cd ../..

