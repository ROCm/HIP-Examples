HIP_PATH?= $(wildcard /opt/rocm)
HIPCC=$(HIP_PATH)/bin/hipcc

CXXFLAGS += -std=c++11 -O2

vectoradd: vectoradd_hip.cpp
	$(HIPCC) ${CXXFLAGS} -o $@ $^

.PHONY: test
test: vectoradd
	$(shell pwd)/vectoradd

.PHONY: clean
clean:
	rm -vf vectoradd *.o
