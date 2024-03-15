HIP_PATH?= $(wildcard /opt/rocm)
HIPCC=$(HIP_PATH)/bin/hipcc

FastWalshTransform: FastWalshTransform.cpp ../include/HIPUtil.hpp ../include/SDKUtil.hpp
	$(HIPCC) -o $@ FastWalshTransform.cpp

.PHONY: test
test: FastWalshTransform
	$(shell pwd)/FastWalshTransform

.PHONY: clean
clean:
	rm -vf FastWalshTransform *.o
