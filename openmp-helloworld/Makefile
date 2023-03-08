HIPCC=/opt/rocm/bin/hipcc
CXX=$(HIPCC)
CXXFLAGS =-fopenmp

SOURCES = openmp_helloworld.cpp

EXECUTABLE=./openmp_helloworld.exe

.PHONY: test


all: $(EXECUTABLE) test


$(EXECUTABLE):
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $@


test: $(EXECUTABLE)
	$(EXECUTABLE)


clean:
	rm -f $(EXECUTABLE) *.o

