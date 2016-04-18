HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../../../..
endif
HIP_EXAMPLES ?=../../../..
include ../../../common/make.config

# C compiler
CC = gcc
CC_FLAGS = -g


ifeq ($(HIP_PLATFORM), nvcc)
#Use textures for features:
HIPCC += -DUSE_TEXTURES
#Use constant buffer for c_clusters var:
#HIPCC += -DUSE_CONSTANT_BUFFER
endif

$(TESTNAME): sradGPU.o sradCPU.o
	$(HIPLD) $(CC_FLAGS) sradGPU.o -o sradGPU -lm
	$(HIPLD) $(CC_FLAGS) sradCPU.o -o sradCPU -lm

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c


sradGPU.o: srad.cu 
	$(HIPCC) $(HIPCC_FLAGS) -DGPU -c srad.cu -o sradGPU.o

sradCPU.o: srad.cu 
	$(HIPCC) $(HIPCC_FLAGS) -DCPU -c srad.cu -o sradCPU.o

clean: 
	rm -f *.o *~ sradGPU sradCPU srad.linkinfo
