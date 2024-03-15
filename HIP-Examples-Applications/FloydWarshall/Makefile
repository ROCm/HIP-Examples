HIP_PATH?= $(wildcard /opt/rocm)
HIPCC=$(HIP_PATH)/bin/hipcc

FloydWarshall: FloydWarshall.cpp ../include/HIPUtil.hpp ../include/SDKUtil.hpp
	$(HIPCC) -o $@ FloydWarshall.cpp

.PHONY: test
test: FloydWarshall
	$(shell pwd)/FloydWarshall

.PHONY: clean
clean:
	rm -vf FloydWarshall *.o
