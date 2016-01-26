HIP_PATH ?=../../../../..
include ../../../common/make.config

# C compiler
CC = gcc
CC_FLAGS = -g

HIPCC_FLAGS += -I../../../common

$(TESTNAME): main.o
	$(HIPLD) $(CC_FLAGS) main.o -o srad -lm

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c


main.o: main.cu $(HIP_DEPS)
	$(HIPCC) $(HIPCC_FLAGS) -c main.cu

clean: hip_clean
	rm -f *.o *~ srad srad.linkinfo isa.txt image_out.pgm *.perf

include ../../../../common/hip.epilogue.make
