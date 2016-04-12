include ../../common/make.config

OUTPUT = -DOUTPUT

# C compiler
CC = gcc

ifeq ($(HIP_PLATFORM), nvcc)
CC_FLAGS = -L/usr/local/cuda/lib64 -lcuda -lcudart 
endif

# link objects (binaries) together
$(TESTNAME):          main.o \
			./kernel/kernel_gpu_cuda_wrapper.o \
			./util/num/num.o \
			./util/timer/timer.o 
#                        ./util/device/device.o
	$(HIPLD)  main.o \
			./kernel/kernel_gpu_cuda_wrapper.o \
			./util/num/num.o \
			./util/timer/timer.o \
			$(CC_FLAGS) -lm \
			-o lavaMD
#                        ./util/device/device.o 
# compile function files into objects (binaries)
main.o:         main.h \
                        main.c \
                        ./kernel/kernel_gpu_cuda_wrapper.h \
                        ./kernel/kernel_gpu_cuda_wrapper.cu \
                        ./util/num/num.h \
                        ./util/num/num.c \
                        ./util/timer/timer.h \
                        ./util/timer/timer.c 
#                        ./util/device/device.h \
#                        ./util/device/device.cu
	$(CC) $(OUTPUT) main.c \
                        -c \
                        -o main.o \
                        -O3

./kernel/kernel_gpu_cuda_wrapper.o:     ./kernel/kernel_gpu_cuda_wrapper.h \
                                                                        ./kernel/kernel_gpu_cuda_wrapper.cu
	$(HIPCC) $(HIPCC_FLAGS) -c  ./kernel/kernel_gpu_cuda_wrapper.cu -o ./kernel/kernel_gpu_cuda_wrapper.o 

./util/num/num.o:       ./util/num/num.h \
                                        ./util/num/num.c
	$(CC)                  ./util/num/num.c \
                                        -c \
                                        -o ./util/num/num.o \
                                        -O3

./util/timer/timer.o:   ./util/timer/timer.h \
                                                ./util/timer/timer.c
	$(CC)                          ./util/timer/timer.c \
                                                -c \
                                                -o ./util/timer/timer.o \
                                                -O3

# delete all object and executable files
clean:
	rm -f       *.o *.perf \
                ./kernel/*.o \
                ./util/num/*.o \
                ./util/timer/*.o \
                ./util/device/*.o \
                lavaMD \
				result.txt
