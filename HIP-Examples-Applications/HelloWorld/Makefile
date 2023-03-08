HIPCC=/opt/rocm/bin/hipcc

TARGET=hcc

SOURCES = HelloWorld.cpp
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE=./HelloWorld

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
