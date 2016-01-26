include ../../common/make.config

ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

OCL_INC_DIR = -I$(OPENCL_INC)
OCL_LIB_DIR = -L$(OPENCL_LIB)

C_C = gcc
OCL_LIB = -lOpenCL

./a.out:	./main.o \
			./kernel/kernel_gpu_opencl_wrapper.o \
			./util/avi/avilib.o \
			./util/avi/avimod.o \
			./util/file/file.o \
			./util/timer/timer.o \
			./util/opencl/opencl.o
	$(C_C)	./main.o \
			./kernel/kernel_gpu_opencl_wrapper.o \
			./util/avi/avilib.o \
			./util/avi/avimod.o \
			./util/file/file.o \
			./util/timer/timer.o \
			./util/opencl/opencl.o \
			-lm \
			$(OCL_LIB_DIR) \
			$(OCL_LIB) -o heartwall

# ======================================================================================================================================================150
#	MAIN FUNCTION
# ======================================================================================================================================================150

./main.o:	./main.h \
			./main.c
	$(C_C)	$(OUTPUT) $(KERNEL_DIM) ./main.c \
			-c \
			-o ./main.o \
			-O3

# ======================================================================================================================================================150
#	KERNELS
# ======================================================================================================================================================150

./kernel/kernel_gpu_opencl_wrapper.o:	./kernel/kernel_gpu_opencl_wrapper.h \
										./kernel/kernel_gpu_opencl_wrapper.c
	$(C_C) $(KERNEL_DIM)	./kernel/kernel_gpu_opencl_wrapper.c \
			-c \
			-o ./kernel/kernel_gpu_opencl_wrapper.o \
			-O3 \
			$(OCL_INC_DIR)

# ======================================================================================================================================================150
#	UTILITIES
# ======================================================================================================================================================150

./util/avi/avilib.o:./util/avi/avilib.h \
					./util/avi/avilib.c
	$(C_C)	./util/avi/avilib.c \
			-c \
			-o ./util/avi/avilib.o \
			-O3

./util/avi/avimod.o:./util/avi/avilib.h \
					./util/avi/avimod.c
	$(C_C)	./util/avi/avimod.c \
			-c \
			-o ./util/avi/avimod.o \
			-O3

./util/file/file.o:	./util/file/file.h \
					./util/file/file.c
	$(C_C)	./util/file/file.c \
			-c \
			-o ./util/file/file.o \
			-O3

./util/timer/timer.o:	./util/timer/timer.h \
						./util/timer/timer.c
	$(C_C)	./util/timer/timer.c \
			-c \
			-o ./util/timer/timer.o \
			-O3

./util/opencl/opencl.o:	./util/opencl/opencl.h \
						./util/opencl/opencl.c
	$(C_C)	./util/opencl/opencl.c \
			-c \
			-o ./util/opencl/opencl.o \
			-O3 \
			$(OCL_INC_DIR)

clean:
	rm	*.o heartwall \
		./kernel/*.o \
		./util/avi/*.o \
		./util/file/*.o \
		./util/timer/*.o \
		./util/opencl/*.o
