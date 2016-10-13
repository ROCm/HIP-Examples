HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc

#### GCC system includes workaround ####
GCC_VER ?= 4.8

GCC_CUR_VER = $(shell gcc -dumpversion)
GPP_CUR_VER = $(shell g++ -dumpversion)

GCC_CUR = 0
GPP_CUR = 1

ifeq ($(findstring $(GCC_VER),$(GCC_CUR_VER)),$(GCC_VER))
GCC_CUR = GCC_VER
endif

ifeq ($(findstring $(GCC_VER),$(GPP_CUR_VER)),$(GCC_VER))
GPP_CUR = GCC_VER
endif

ifeq ($(GCC_CUR), $(GPP_CUR))
    CXXFLAGS += -I /usr/include/x86_64-linux-gnu -I /usr/include/x86_64-linux-gnu/c++/$(GCC_VER) -I /usr/include/c++/$(GCC_VER)
else
    $(warning )
    $(warning ***************************************************)
    $(warning *** The supported version of gcc and g++ is $(GCC_VER) ***)
    $(warning ***    Current default version of gcc is $(GCC_CUR_VER)    ***)
    $(warning ***    Current default version of g++ is $(GPP_CUR_VER)    ***)
    $(warning ***************************************************)
    $(warning )
endif
#### GCC system includes workaround ####

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

