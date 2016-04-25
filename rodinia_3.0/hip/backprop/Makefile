include ../../common/make.config

printenv:
	@echo 'OMPCC = "$(OMPCC)"'
	@echo 'OMP_FLAGS = "$(OMP_FLAGS)"'
	@echo 'HIP_PATH = "$(HIP_PATH)"'
	@echo 'HIPCC_FLAGS = "$(HIPCC_FLAGS)"'
	@echo 'CUDA_LIB_DIR = "$(CUDA_LIB_DIR)"'
	
$(TESTNAME): backprop.o facetrain.o imagenet.o backprop_cuda.o
	$(HIPLD) backprop.o facetrain.o imagenet.o backprop_cuda.o -o backprop -lm

%.o: %.[ch]
	$(HIPCC) $(HIPCC_FLAGS) $< -c

facetrain.o: facetrain.c backprop.h
	$(OMPCC) $(OMP_FLAGS) facetrain.c -c
	
backprop.o: backprop.c backprop.h
	$(OMPCC) $(OMP_FLAGS) backprop.c -c
	
backprop_cuda.o: backprop_cuda.cu backprop_cuda_kernel.cu backprop.h $(HIP_DEPS)
	$(HIPCC) $(HIPCC_FLAGS) -c backprop_cuda.cu

imagenet.o: imagenet.c backprop.h
	$(OMPCC) $(OMP_FLAGS) imagenet.c -c
	
clean:
	rm -f *.o *~ backprop out_hip.txt backprop_cuda.linkinfo *.perf
