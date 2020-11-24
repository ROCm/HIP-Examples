# Simple OpenMP hello world example written directly to the HIP interface.

## Requirements
* Installed ROCm 3.9 or newer. See  [ROCm Installation Guide](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html).


## How to run this code:

### Using Make:
* To build and run: `make`.
* To clean the environment: `make clean`.


### Using CMake:
* To build: `mkdir -p build; cd build; cmake ..; make`
* To run the test: `./test_openmp_helloworld`
* To clean the build environment: `make clean`

**Note:** You may override `AMDGPU_TARGETS` in the HIP config file by modifying the CMakeLists.txt.

