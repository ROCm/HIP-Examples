HIP_PATH?= $(wildcard /opt/rocm)
HIPCC=$(HIP_PATH)/bin/hipcc

RecursiveGaussian: RecursiveGaussian.cpp RecursiveGaussian.hpp ../include/HIPUtil.hpp ../include/SDKUtil.hpp ../include/SDKBitMap.hpp 
	$(HIPCC) -o $@ RecursiveGaussian.cpp

.PHONY: test
test: RecursiveGaussian
	$(shell pwd)/RecursiveGaussian

.PHONY: clean
clean:
	rm -vf RecursiveGaussian *.o
