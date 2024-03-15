HIP_PATH?= $(wildcard /opt/rocm)
HIPCC=$(HIP_PATH)/bin/hipcc

CXXFLAGS += -std=c++11 -O3

stream: stream.cpp
	${HIPCC} ${CXXFLAGS} -o $@ $^

.PHONY: test
test: stream
	$(shell pwd)/stream

.PHONY: clean
clean:
	rm -vf stream *.o
