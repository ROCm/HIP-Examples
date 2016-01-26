include ../../common/make.config
CXX = g++
DEBUGFLAGS = -g
# You will need to modify the following 2 lines
# by inserting the correct path to the OpenCL
# lib and inc directories.
CXXFLAGS = -I$(OPENCL_INC) -c $(DEBUGFLAGS)
LINKFLAGS = -L$(OPENCL_LIB)

# link objects(binaries) together
a.out: main.o OpenCL.o
	$(CXX) main.o OpenCL.o $(LINKFLAGS) -o pathfinder

# compile main function file into object (binary)
main.o: main.cpp OpenCL.cpp kernels.cl
	$(CXX) $(CXXFLAGS) main.cpp OpenCL.cpp

clean:
	clear
	@rm -f pathfinder *.o *.out *~ *\#
