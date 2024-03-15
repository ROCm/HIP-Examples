HIP_PATH?= $(wildcard /opt/rocm)
HIPCC=$(HIP_PATH)/bin/hipcc

CXXFLAGS += -std=c++11 -O3

reduction: reduction.cpp
	${HIPCC} ${CXXFLAGS} -o $@ $^

.PHONY: clean
clean:
	rm -vf reduction *.o
