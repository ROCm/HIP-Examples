#CUDAPATH=/usr/local/cuda
HIP_PATH=/opt/rocm/hip#this will change from system to system
# Have this point to an old enough gcc (for nvcc)
GCCPATH=/usr

#NVCC=${CUDAPATH}/bin/nvcc
HIPCC=${HIP_PATH}/bin/hipcc
CCPATH=${GCCPATH}/bin

drv:
	PATH=.:${CCPATH}:${PATH} ${HIPCC} -arch=compute_20 -I${HIP_PATH}/include -c compare.cpp
	${HIPCC} -arch=compute_20 -I${HIP_PATH}/include -c hipsgemm.cpp
	${HIPCC} -O3 -I${HIP_PATH}/include -I/opt/hcc/include -c gpu_burn-drv-hipFIJI.cpp
	${HIPCC} -lm -o gpu_burn-hip gpu_burn-drv-hipFIJI.o compare.o hipsgemm.o -O3
	#${HIPCC} -lm -o gpu_burn-hip gpu_burn-drv-hipFIJI.o compare.o ${HIP_PATH}/src/hip_hcc.o -O3
