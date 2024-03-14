HIP_PATH?= $(wildcard /opt/rocm)
HIPCC=$(HIP_PATH)/bin/hipcc

CXXFLAGS += -std=c++11 -O3

.PHONY: all
all:  gpu-stream-hip

common.o: common.cpp common.h Makefile

hip-stream.o : hip-stream.cpp
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

gpu-stream-hip: hip-stream.o common.o Makefile
	$(HIPCC) $(CXXFLAGS) common.o $< -lm -o $@

.PHONY: test
test: gpu-stream-hip
	$(shell pwd)/gpu-stream-hip
	$(shell pwd)/gpu-stream-hip --groups 256 --groupSize 256
	$(shell pwd)/gpu-stream-hip --float
	$(shell pwd)/gpu-stream-hip --float --groups 256 --groupSize 256

.PHONY: clean
clean:
	rm -vf gpu-stream-hip *.o
