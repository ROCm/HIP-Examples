HIP_PATH?= $(wildcard /opt/rocm)
HIPCC=$(HIP_PATH)/bin/hipcc

dct: dct.cpp ../include/HIPUtil.hpp ../include/SDKUtil.hpp
	$(HIPCC) -o $@ dct.cpp

.PHONY: test
test: dct
	$(shell pwd)/dct

.PHONY: clean
clean:
	rm -vf dct *.o
