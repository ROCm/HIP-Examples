include ../../common/make.config

# C compiler
 CC = gcc
 CC_FLAGS = -g -O2 -Icommon


$(TESTNAME): common/common.o lud_kernel.hip.o lud.hip.o
	$(HIPLD) $(CC_FLAGS) $^ -o $@ -lm

common/common.o: common/common.c
	        $(CC) $(CC_FLAGS) -c $< -o $@

lud.hip.o: lud.hip.cu
	$(HIPCC)  $(HIPCC_FLAGS) -c $< -o $@

lud_kernel.hip.o: lud_kernel.hip.cu
	$(HIPCC)  $(HIPCC_FLAGS) -c $< -o $@

clean:
	        rm -f *.o common/*.o lud *.linkinfo *.perf
