include ../../common/make.config

# C compiler
CC = gcc
CC_FLAGS = -g

$(TESTNAME): bfs.hip.o
	$(HIPLD) $(CC_FLAGS) bfs.hip.o -lm -o bfs

bfs.hip.o: bfs.hip.cu 
	$(HIPCC)  $(HIPCC_FLAGS) $(KERNEL_DIM) bfs.hip.cu -c -o bfs.hip.o

clean:
	rm -f *.o bfs result.txt *.perf
