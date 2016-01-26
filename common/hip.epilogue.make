# THis file is intended to be included at end of Makefiles.
HIP_SOURCES = $(HIP_PATH)/src/hip_hcc.cpp
HIP_OBJECTS = $(HIP_SOURCES:.cpp=.o)
# It may define additional make steps
#Assume HIP_PATH already defined.
#Assume HCC compiler already set up in HCC,HCC_CFLAGS, HCC_LDFLAGS


.DUMMY: hip_clean



hip_clean:
	rm -rf $(AM_LIB) $(HIP_OBJECTS)

hip_deps: $(HIP_DEPS)

%.o:: %.cpp
	$(HIPCC) $(HIPCC_FLAGS) $< -c -o $@



$(HIP_PATH)/src/hip_hcc.o:  $(wildcard $(HIP_PATH)/include/*.h) $(wildcard $(HIP_PATH)/src/*.cpp)
