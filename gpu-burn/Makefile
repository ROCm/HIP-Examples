HIP_PATH ?= /opt/rocm/hip
HCC_PATH ?= /opt/rocm/hcc
HIP_PLATFORM = $(shell $(HIP_PATH)/bin/hipconfig --platform)
HIP_INCLUDE = -I${HIP_PATH}/include -I${HCC_PATH}/include
BUILD_DIR ?= build

HIPCC = ${HIP_PATH}/bin/hipcc
CPPFLAGS =
LDFLAGS = -lm

ifeq (${HIP_PLATFORM}, nvcc)
    CPPFLAGS += -arch=compute_20
endif

GPUBURN_SRC = $(wildcard *.cpp)
GPUBURN_OBJ = $(addprefix ${BUILD_DIR}/,$(subst .cpp,.o, $(GPUBURN_SRC)))
GPUBURN_BIN = ${BUILD_DIR}/gpuburn-hip

.PHONY: all clean run itburn

all: ${GPUBURN_BIN}

${GPUBURN_BIN}: ${GPUBURN_OBJ}
	${HIPCC} ${LDFLAGS} -o ${GPUBURN_BIN} ${GPUBURN_OBJ}

${BUILD_DIR}/%.o: %.cpp Makefile
	mkdir -p ${BUILD_DIR}
	${HIPCC} ${HIP_INCLUDE} ${CPPFLAGS} -c -o $@ $<  

run: itburn
itburn:
	HCC_LAZYINIT=ON ${GPUBURN_BIN}

clean:
	rm -rf ${BUILD_DIR}
