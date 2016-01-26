include ../../common/make.config

# C compiler
 CC = gcc
 CC_FLAGS = -g -O2 -Icommon


$(TESTNAME): common/common.o lud_kernel.hip.o lud.hip.o
	$(HIPLD) $(CC_FLAGS)  common/common.o lud_kernel.hip.o lud.hip.o  -o lud_cuda -lm

%.o: %.[ch]
	        $(CC) $(CC_FLAGS) $< -c

lud.hip.o: lud.hip.cu
	$(HIPCC)  $(HIPCC_FLAGS) -c lud.hip.cu

lud_kernel.hip.o: lud_kernel.hip.cu
	$(HIPCC)  $(HIPCC_FLAGS) -c lud_kernel.hip.cu

clean:
	        rm -f *.o lud_cuda  *.linkinfo *.perf

include ../../../common/hip.epilogue.make
