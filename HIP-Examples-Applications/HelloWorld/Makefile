HIP_PATH?= $(wildcard /opt/rocm)
HIPCC=$(HIP_PATH)/bin/hipcc

HelloWorld: HelloWorld.cpp
	$(HIPCC) -o $@ HelloWorld.cpp

.PHONY: test
test: HelloWorld
	$(shell pwd)/HelloWorld

.PHONY: clean
clean:
	rm -vf HelloWorld *.o
