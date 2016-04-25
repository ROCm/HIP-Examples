include ../../common/make.config

# link objects(binaries) together
$(TESTNAME): main.o ./AVI/avilib.o ./AVI/avimod.o 
	$(HIPLD) main.o ./AVI/avilib.o ./AVI/avimod.o -lm -o heartwall

%.o: %.[ch]
	$(HIPCC) $(HIPCC_FLAGS) $< -c
	
main.o: main.cu kernel.cu define.c 
	$(HIPCC) $(HIPCC_FLAGS) -DOUTPUT $(KERNEL_DIM) main.cu -I./AVI -c

./AVI/avilib.o ./AVI/avimod.o:
	cd AVI; make;

# delete all object files
clean: 
	rm -f *.o AVI/*.o heartwall *.linkinfo
