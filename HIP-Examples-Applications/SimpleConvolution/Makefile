HIP_PATH?= $(wildcard /opt/rocm)
HIPCC=$(HIP_PATH)/bin/hipcc

SimpleConvolution: SimpleConvolution.cpp SimpleConvolution.hpp FilterCoeff.h ../include/HIPUtil.hpp ../include/SDKUtil.hpp
	$(HIPCC) -o $@ SimpleConvolution.cpp

.PHONY: test
test: SimpleConvolution
	$(shell pwd)/SimpleConvolution

.PHONY: clean
clean:
	rm -vf SimpleConvolution *.o
