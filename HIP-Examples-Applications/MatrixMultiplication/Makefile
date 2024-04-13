HIP_PATH?= $(wildcard /opt/rocm)
HIPCC=$(HIP_PATH)/bin/hipcc

MatrixMultiplication: MatrixMultiplication.cpp ../include/HIPUtil.hpp ../include/SDKUtil.hpp
	$(HIPCC) -o $@ MatrixMultiplication.cpp

.PHONY: test
test: MatrixMultiplication
	$(shell pwd)/MatrixMultiplication

.PHONY: clean
clean:
	rm -vf MatrixMultiplication *.o
