HIPCC=/opt/rocm/bin/hipcc

SOURCES = FastWalshTransform.cpp
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE=./FastWalshTransform

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
