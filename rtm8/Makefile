HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc

CXXFLAGS += -std=c++11 -O3

rtm8: rtm8.cpp
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^ 
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

.PHONY: clean

clean:
	rm -f rtm8 *.o

