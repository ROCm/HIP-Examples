include ../../common/make.config

# C compiler
CC = gcc
CC_FLAGS = -g

$(TESTNAME): gaussian.o
	$(HIPLD) gaussian.o -lm -o gaussian

gaussian.o: gaussian.cu 
	$(HIPCC)  $(HIPCC_FLAGS) $(KERNEL_DIM) gaussian.cu -c -o gaussian.o

clean:
	rm -f *.o gaussian *.perf
