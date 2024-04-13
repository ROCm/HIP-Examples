HIP_PATH?= $(wildcard /opt/rocm)
HIPCC=$(HIP_PATH)/bin/hipcc

Histogram: Histogram.cpp Histogram.hpp ../include/HIPUtil.hpp ../include/SDKUtil.hpp
	$(HIPCC) -o $@ Histogram.cpp

.PHONY: test
test: Histogram
	$(shell pwd)/Histogram

.PHONY: clean
clean:
	rm -vf Histogram *.o
