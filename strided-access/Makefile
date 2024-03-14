HIP_PATH?= $(wildcard /opt/rocm)
HIPCC=$(HIP_PATH)/bin/hipcc

CXXFLAGS += -std=c++11 -O3

strided-access: benchmark-hip.cpp benchmark-utils.hpp
	${HIPCC} ${CXXFLAGS} -o $@ benchmark-hip.cpp

.PHONY: test
test: strided-access
	$(shell pwd)/strided-access

.PHONY: clean
clean:
	rm -vf strided-access *.o
