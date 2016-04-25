include ../../common/make.config

CUDA_LIB =	-lcuda -lcudart
CUDA_FLAG = -arch sm_20

C_C = gcc

$(TESTNAME):	./nn.o 
	$(HIPLD)	./nn.o \
			-lm \
                       -o $(TESTNAME)

nn.o:	./nn_hip.cu
	$(HIPCC)  $(HIPCC_FLAGS)	$(KERNEL_DIM) ./nn_hip.cu \
			-c \
			-o ./nn.o

clean: 
	rm -f	*.o  $(TESTNAME) *.perf
