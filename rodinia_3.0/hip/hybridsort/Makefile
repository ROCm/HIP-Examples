include ../../common/make.config

ifdef VERIFY
override VERIFY = -DVERIFY
endif

ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

ifdef TIMER
override TIMER = -DTIMER
endif

bucketsort.hip.o: bucketsort.hip.cu bucketsort.cuh bucketsort_kernel.hip.cu histogram1024_kernel.hip.cu 
	$(HIPCC)  $(HIPCC_FLAGS) $(KERNEL_DIM)  $(VERIFY) $(OUTPUT) $(TIMER) $(HISTO_WG_SIZE_0) $(BUCKET_WG_SIZE_0) $(BUCKET_WG_SIZE_1) $(MERGE_WG_SIZE_0) $(MERGE_WG_SIZE_1) bucketsort.hip.cu -c -o bucketsort.hip.o
mergesort.hip.o: mergesort.hip.cu mergesort.cuh mergesort_kernel.cu
	 $(HIPCC)  $(HIPCC_FLAGS) $(KERNEL_DIM)  $(VERIFY) $(OUTPUT) $(TIMER) $(HISTO_WG_SIZE_0) $(BUCKET_WG_SIZE_0) $(BUCKET_WG_SIZE_1) $(MERGE_WG_SIZE_0) $(MERGE_WG_SIZE_1) mergesort.hip.cu -c -o mergesort.hip.o
main.hip.o: main.hip.cu
	$(HIPCC)  $(HIPCC_FLAGS) $(KERNEL_DIM)  $(VERIFY) $(OUTPUT) $(TIMER) $(HISTO_WG_SIZE_0) $(BUCKET_WG_SIZE_0) $(BUCKET_WG_SIZE_1) $(MERGE_WG_SIZE_0) $(MERGE_WG_SIZE_1) main.hip.cu -c -o main.hip.o
$(TESTNAME): bucketsort.hip.o mergesort.hip.o main.hip.o 
	$(HIPLD) $(CC_FLAGS)  $(VERIFY) $(OUTPUT) $(TIMER) $(HISTO_WG_SIZE_0) $(BUCKET_WG_SIZE_0) $(BUCKET_WG_SIZE_1) $(MERGE_WG_SIZE_0) $(MERGE_WG_SIZE_1)  bucketsort.hip.o mergesort.hip.o main.hip.o -o hybridsort -lm

clean:
	rm -f *.o hybridsort hybridoutput.txt
