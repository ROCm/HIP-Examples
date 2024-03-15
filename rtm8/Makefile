HIP_PATH?= $(wildcard /opt/rocm)
HIPCC=$(HIP_PATH)/bin/hipcc

CXXFLAGS += -std=c++11 -O3

rtm8_hip: rtm8.cpp
	${HIPCC} ${CXXFLAGS} -o $@ $^

rtm8_cuda: rtm8.cu
	nvcc -O3 rtm8.cu -o rtm8_cuda


FC:=gfortran
FCFLAGS:=-02

mysecond.o: mysecond.c
	$(CC) $(CFLAGS) -DUNDERSCORE -c -o mysecond.o mysecond.c

rtm8_fortran: rtm8.f mysecond.o
	$(FC) $(FCFLAGS) -o $@ rtm8.f mysecond.o

.PHONY: test
test: run_hip

.PHONY: run_hip
run_hip: rtm8_hip
	$(shell pwd)/rtm8_hip

.PHONY: run_cuda
run_cuda: rtm8_cuda
	$(shell pwd)/rtm8_cuda

.PHONY: run_fortran
run_fortran: rtm8_fortran
	$(shell pwd)/rtm8_fortran

.PHONY: clean
clean:
	rm -vf rtm8_hip rtm8_cuda rtm8_fortran *.o
