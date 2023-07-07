# This file is designed to be included at beginning of Makefile, right after setting HIP_PATH.
# Note: define $HIP_PATH before including this file.
# HIP_PATH should be relevant to the parent makefile
#
# It should not include any concrete makefile steps, so "make" still runs the first step in the Makefile.
#

#------
##Provide default if not already set:
HIP_PATH?=../..
HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --compiler)

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

HIPCC=$(HIP_PATH)/bin/hipcc
HIPLD=$(HIP_PATH)/bin/hipcc

#-- 
# Set up automatic make of HIP cpp depenendencies
# TODO - this can be removed when HIP has a proper make structure.
#HIP_SOURCES = $(HIP_PATH)/src/hip_hcc.cpp

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
#HIP_DEPS = $(HIP_SOURCES:.cpp=.o)
OMPCC = gcc
OMP_FLAGS += -fopenmp

# Add dependencies to make hip_cc.o and other support files.
HSA_PATH ?= /opt/hsa
#HIP_SOURCES = $(HIP_PATH)/src/hip_hcc.cpp
#HIP_DEPS = $(HIP_SOURCES:.cpp=.o)
#$(HIP_DEPS): HIPCC_FLAGS += -I$(HSA_PATH)/include
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


KCFLAGS += $(OPT) -I$(HSA_PATH)/include  -I$(HIP_PATH)/include -I$(GRID_LAUNCH_PATH) -I$(AM_PATH)/include

%.o:: %.cpp
	$(HIPCC) $(HIPCC_FLAGS) $< -c -o $@


