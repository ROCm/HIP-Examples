include ../../common/make.config

# C compiler
CC = gcc
CC_FLAGS = -g

$(TESTNAME): euler3d.o
	$(HIPLD) $(CC_FLAGS) euler3d.o -o euler3d -lm

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

euler3d.o: euler3d.cu
	$(HIPCC) $(HIPCC_FLAGS) -c euler3d.cu

clean:
	rm -f *.o *~ euler3d density density_energy momentum isa.txt *.out *.diff *.perf
