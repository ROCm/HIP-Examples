HIP_PATH?= $(wildcard /opt/rocm)
HIPCC=$(HIP_PATH)/bin/hipcc

BitonicSort: BitonicSort.cpp ../include/HIPUtil.hpp ../include/SDKUtil.hpp
	$(HIPCC) -o $@ BitonicSort.cpp

.PHONY: test
test: BitonicSort
	$(shell pwd)/BitonicSort

.PHONY: clean
clean:
	rm -vf BitonicSort *.o
