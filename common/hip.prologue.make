# This file is designed to be included at beginning of Makefile.
#
# It should not include any concrete makefile steps, so "make" still runs the first step in the Makefile.
#

#------
##Provide default if not already set:
HIP_PLATFORM=$(shell /opt/rocm/bin/hipconfig --compiler)

# CUDA toolkit installation path
CUDA_DIR?=/usr/local/cuda-7.5
# CUDA toolkit libraries
CUDA_LIB_DIR := $(CUDA_DIR)/lib
ifeq ($(shell uname -m), x86_64)
  ifeq ($(shell if test -d $(CUDA_DIR)/lib64; then echo T; else echo F; fi), T)
    CUDA_LIB_DIR := $(CUDA_DIR)/lib64
  endif
endif

# Some samples mix openmp with gpu acceleration.
# Those unfortunately have to be compiled with gcc, not clang.
# nvcc (7.5) can handle openmp though. 
# use OMPCC and OMP_FLAGS

HIPCC=/opt/rocm/bin/hipcc
HIPLD=/opt/rocm/bin/hipcc

#-- 
# Set up automatic make of HIP cpp depenendencies

HIPCC_FLAGS += -I../../common
# 'make dbg=1' enables HIPCC debugging and no opt switch.
ifeq ($(dbg),1)
	HIPCC_FLAGS += -g
	OMP_FLAGS += -g
else ifeq ($(opt),0)
	HIPCC_FLAGS += -O0
	OMP_FLAGS += -O0
else ifeq ($(opt),3)
	HIPCC_FLAGS += -O3
	OMP_FLAGS += -O3
else
	HIPCC_FLAGS += -O2
	OMP_FLAGS += -O2
endif

ifeq ($(HIP_PLATFORM), nvcc)
OMPCC = gcc
OMP_FLAGS = $(HIPCC_FLAGS)
HIP_DEPS = 

else ifeq ($(HIP_PLATFORM), hcc)
OMPCC = gcc
OMP_FLAGS += -fopenmp

# Add dependencies to make hip_cc.o and other support files.
#HIP_DEPS = $(HIP_SOURCES:.cpp=.o)
%.o:: %.cpp
	    $(HIPCC) $(HIPCC_FLAGS) $< -c -o $@
endif








#------
#
#---
# Rule for automatic HIPIFY call - assumes original cuda files are stored in local 'cusrc' directory.  See kmeans.
#%.cu : cusrc/%.cu
#	$(HIPIFY)  $< > $@
#%.cuh : cusrc/%.cuh
#	$(HIPIFY)  $< > $@


KCFLAGS += $(OPT) -I/opt/rocm/include  -I/opt/rocm/hip/include -I$(GRID_LAUNCH_PATH) -I$(AM_PATH)/include

%.o:: %.cpp
	$(HIPCC) $(HIPCC_FLAGS) $< -c -o $@


