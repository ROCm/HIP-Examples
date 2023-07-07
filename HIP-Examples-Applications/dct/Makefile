HIPCC=/opt/rocm/bin/hipcc

SOURCES = dct.cpp
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE=./dct

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
