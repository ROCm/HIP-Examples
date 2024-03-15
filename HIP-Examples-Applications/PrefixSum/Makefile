HIP_PATH?= $(wildcard /opt/rocm)
HIPCC=$(HIP_PATH)/bin/hipcc

PrefixSum: PrefixSum.cpp ../include/HIPUtil.hpp ../include/SDKUtil.hpp
	$(HIPCC) -o $@ PrefixSum.cpp

.PHONY: test
test: PrefixSum
	$(shell pwd)/PrefixSum

.PHONY: clean
clean:
	rm -vf PrefixSum *.o
