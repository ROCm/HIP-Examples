HIP_PATH?= $(wildcard /opt/rocm)
HIPCC=$(HIP_PATH)/bin/hipcc

BinomialOption: BinomialOption.cpp ../include/HIPUtil.hpp ../include/SDKUtil.hpp
	$(HIPCC) -o $@ BinomialOption.cpp

.PHONY: test
test: BinomialOption
	$(shell pwd)/BinomialOption

.PHONY: clean
clean:
	rm -vf BinomialOption *.o
