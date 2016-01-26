include ../../common/make.config

CUDA_SDK_PATH := $(SDK_DIR)

# Determine the correct version of the cutil library
CUTIL_LIB = # -lcutil
ifeq ($(shell uname -m), x86_64)
     ifeq ($(shell if test -e $(SDK_DIR)/lib/libcutil_x86_64.a; then echo T; else echo F; fi), T)
        CUTIL_LIB = #-lcutil_x86_64
     endif
endif


all: euler3d  #euler3d_double  pre_euler3d  pre_euler3d_double 

euler3d: euler3d.cu
	nvcc $(KERNEL_DIM) -O2 -Xptxas -v --gpu-architecture=compute_20 --gpu-code=compute_20 euler3d.cu -o euler3d -I$(CUDA_SDK_PATH)/common/inc  -L$(CUDA_SDK_PATH)/lib $(CUTIL_LIB)

euler3d_double: euler3d_double.cu
	nvcc -Xptxas -v -O3 --gpu-architecture=compute_20 --gpu-code=compute_20 euler3d_double.cu -o euler3d_double -I$(CUDA_SDK_PATH)/common/inc  -L$(CUDA_SDK_PATH)/lib $(CUTIL_LIB)


pre_euler3d: pre_euler3d.cu
	nvcc -Xptxas -v -O3 --gpu-architecture=compute_20 --gpu-code=compute_20 pre_euler3d.cu -o pre_euler3d -I$(CUDA_SDK_PATH)/common/inc  -L$(CUDA_SDK_PATH)/lib $(CUTIL_LIB)

pre_euler3d_double: pre_euler3d_double.cu
	nvcc -Xptxas -v -O3 --gpu-architecture=compute_20 --gpu-code=compute_20 pre_euler3d_double.cu -o pre_euler3d_double -I$(CUDA_SDK_PATH)/common/inc  -L$(CUDA_SDK_PATH)/lib $(CUTIL_LIB)


clean:
	rm -f euler3d euler3d_double pre_euler3d pre_euler3d_double *.linkinfo
