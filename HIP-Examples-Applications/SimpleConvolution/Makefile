HIPCC=/opt/rocm/bin/hipcc

SOURCES = SimpleConvolution.cpp
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE=./SimpleConvolution

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
