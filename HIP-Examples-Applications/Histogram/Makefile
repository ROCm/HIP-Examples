HIP_PATH?= $(wildcard /opt/rocm)
HIPCC=$(HIP_PATH)/bin/hipcc

SOURCES = Histogram.cpp
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE=./Histogram

.PHONY: test


all: $(EXECUTABLE) test

CXXFLAGS =-g
CXX=$(HIPCC)


$(EXECUTABLE): $(OBJECTS)
	$(HIPCC) $(OBJECTS) -o $@


test: $(EXECUTABLE)
	$(EXECUTABLE)


clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
	rm -f $(HIP_PATH)src/*.o
