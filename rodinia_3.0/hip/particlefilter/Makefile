include ../../common/make.config

ifeq ($(HIP_PLATFORM), nvcc)
ifeq ($(dbg),1)
#	HIPCC_FLAGS += -arch=sm_13
else
#	HIPCC_FLAGS += -use_fast_math -arch=
	HIPCC_FLAGS += -use_fast_math
endif
endif

$(TESTNAME): naive float

	
naive: ex_particle_CUDA_naive_seq.o
	$(HIPLD) ex_particle_CUDA_naive_seq.o -o particlefilter_naive -lm

float: ex_particle_CUDA_float_seq.o
	$(HIPLD) ex_particle_CUDA_float_seq.o -o particlefilter_float -lm
	
%.o: %.[ch]
	$(HIPCC) $(HIPCC_FLAGS) $< -c

ex_particle_CUDA_naive_seq.o: ex_particle_CUDA_naive_seq.cu
	$(HIPCC) $(HIPCC_FLAGS) -c ex_particle_CUDA_naive_seq.cu

ex_particle_CUDA_float_seq.o: ex_particle_CUDA_float_seq.cu
	$(HIPCC) $(HIPCC_FLAGS) -c ex_particle_CUDA_float_seq.cu

clean:
	rm -f *.o *~ *.linkinfo particlefilter_naive particlefilter_float run*.out run*.diff
