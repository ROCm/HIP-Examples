HIP_PATH?= $(wildcard /opt/rocm)
HIPCC=$(HIP_PATH)/bin/hipcc

CXXFLAGS:=-fopenmp

openmp_helloworld: openmp_helloworld.cpp
	$(HIPCC) $(CXXFLAGS) -o $@ $^

.PHONY: test
test: openmp_helloworld
	$(shell pwd)/openmp_helloworld

.PHONY: clean
clean:
	rm -f openmp_helloworld *.o
