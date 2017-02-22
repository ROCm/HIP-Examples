CXXFLAGS += -std=c++11 -O3

all:  gpu-stream-hip

common.o: common.cpp common.h Makefile

HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc

hip-stream.o : hip-stream.cpp
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

gpu-stream-hip: hip-stream.o common.o Makefile
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	$(HIPCC) $(CXXFLAGS) common.o $< -lm -o $@
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif


.PHONY: clean

clean:
	rm -f   gpu-stream-hip *.o

