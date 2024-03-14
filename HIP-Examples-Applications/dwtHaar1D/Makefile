HIP_PATH?= $(wildcard /opt/rocm)
HIPCC=$(HIP_PATH)/bin/hipcc

dwtHaar1D: dwtHaar1D.cpp ../include/HIPUtil.hpp ../include/SDKUtil.hpp
	$(HIPCC) -o $@ dwtHaar1D.cpp

.PHONY: test
test: dwtHaar1D
	$(shell pwd)/dwtHaar1D

.PHONY: clean
clean:
	rm -vf dwtHaar1D *.o
