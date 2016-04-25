HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../../../..
endif
HIP_EXAMPLES ?=../../../..
include ../../../common/make.config

# C compiler
CC = gcc
CC_FLAGS = -g

HIPCC_FLAGS += -I../../../common

$(TESTNAME): main.o
	$(HIPLD) $(CC_FLAGS) main.o -o srad -lm

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

main.o: main.cu 
	$(HIPCC) $(HIPCC_FLAGS) -c main.cu

clean: 
	rm -f *.o *~ srad srad.linkinfo isa.txt image_out.pgm *.perf
