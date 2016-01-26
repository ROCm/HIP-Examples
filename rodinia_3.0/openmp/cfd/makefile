CUDA_SDK_PATH = /af10/sc5nf/NVIDIA_CUDA_SDK

OMP_NUM_THREADS = 8

all: euler3d_cpu euler3d_cpu_double pre_euler3d_cpu pre_euler3d_cpu_double

#euler3d: euler3d.cu
#	nvcc -Xptxas -v -O3 --gpu-architecture=compute_13 --gpu-code=compute_13 euler3d.cu -o euler3d -I$(CUDA_SDK_PATH)/common/inc  -L$(CUDA_SDK_PATH)/lib  -lcutil

#euler3d_double: euler3d_double.cu
#	nvcc -Xptxas -v -O3 --gpu-architecture=compute_13 --gpu-code=compute_13 euler3d_double.cu -o euler3d_double -I$(CUDA_SDK_PATH)/common/inc  -L$(CUDA_SDK_PATH)/lib  -lcutil

euler3d_cpu: euler3d_cpu.cpp
	g++ -O3 -Dblock_length=$(OMP_NUM_THREADS) -fopenmp euler3d_cpu.cpp -o euler3d_cpu
euler3d_cpu_double: euler3d_cpu_double.cpp
	g++ -O3 -Dblock_length=$(OMP_NUM_THREADS) -fopenmp euler3d_cpu_double.cpp -o euler3d_cpu_double

#pre_euler3d: pre_euler3d.cu
#	nvcc -Xptxas -v -O3 --gpu-architecture=compute_13 --gpu-code=compute_13 pre_euler3d.cu -o pre_euler3d -I$(CUDA_SDK_PATH)/common/inc  -L$(CUDA_SDK_PATH)/lib  -lcutil

#pre_euler3d_double: pre_euler3d_double.cu
#	nvcc -Xptxas -v -O3 --gpu-architecture=compute_13 --gpu-code=compute_13 pre_euler3d_double.cu -o pre_euler3d_double -I$(CUDA_SDK_PATH)/common/inc  -L$(CUDA_SDK_PATH)/lib  -lcutil

pre_euler3d_cpu: pre_euler3d_cpu.cpp
	g++ -O3 -Dblock_length=$(OMP_NUM_THREADS) -fopenmp pre_euler3d_cpu.cpp -o pre_euler3d_cpu

pre_euler3d_cpu_double: pre_euler3d_cpu_double.cpp
	g++ -O3 -Dblock_length=$(OMP_NUM_THREADS) -fopenmp pre_euler3d_cpu_double.cpp -o pre_euler3d_cpu_double


clean:
#	rm -f euler3d
	rm -f euler3d_cpu
#	rm -f euler3d_double
	rm -f euler3d_cpu_double

#	rm -f pre_euler3d
	rm -f pre_euler3d_cpu
#	rm -f pre_euler3d_double
	rm -f pre_euler3d_cpu_double
