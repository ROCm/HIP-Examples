HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
    HIP_PATH=../..
endif
HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --platform)
HIPCC=${HIP_PATH}/bin/hipcc

ifeq (${HIP_PLATFORM}, nvcc)
    HIPCC_FLAGS = -arch=compute_20
endif
ifeq (${HIP_PLATFORM}, hcc)
    HIPCC_FLAGS = -stdlib=libc++
endif

drv:
	${HIPCC} ${HIPCC_FLAGS} -I${HIP_PATH}/include -c compare.cpp
	${HIPCC} ${HIPCC_FLAGS} -I${HIP_PATH}/include -c hipsgemm.cpp
	${HIPCC} ${HIPCC_FLAGS} -O3 -I${HIP_PATH}/include -I/opt/hcc/include -c gpu_burn-drv-hipFIJI.cpp
	${HIPCC} ${HIPCC_FLAGS} -lm -o gpu_burn-hip gpu_burn-drv-hipFIJI.o compare.o hipsgemm.o -O3

clean:
	rm -f *.o gpu_burn-hip
