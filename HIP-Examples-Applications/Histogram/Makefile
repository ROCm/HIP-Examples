HIPCC=/opt/rocm/bin/hipcc

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
