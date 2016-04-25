include ../../common/make.config

$(TESTNAME): hotspot.o
	$(HIPLD) hotspot.o -o hotspot -lm

hotspot.o: hotspot.cu $(HIP_DEPS)
	$(HIPCC) $(HIPCC_FLAGS) -c hotspot.cu

clean:
	rm -f *.o *~ hotspot *.linkinfo *.perf output.out
