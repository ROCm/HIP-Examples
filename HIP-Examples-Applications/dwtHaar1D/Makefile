HIPCC=/opt/rocm/bin/hipcc

SOURCES = dwtHaar1D.cpp
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE=./dwtHaar1D

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
