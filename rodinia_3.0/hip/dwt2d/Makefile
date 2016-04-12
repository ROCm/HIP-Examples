include ../../common/make.config

# C compiler
CC = gcc
CC_FLAGS = -g

HIPCC_FLAGS += -DOUTPUT

ifeq ($(HIP_PLATFORM), nvcc)
HIPCC_FLAGS += -arch sm_20
endif

ifeq ($(HIP_PLATFORM), hcc)
HIPCC_FLAGS += -I.
endif

$(TESTNAME): main.o dwt.o components.o fdwt53.o fdwt97.o common.o rdwt53.o rdwt97.o
	$(HIPLD) $(CC_FLAGS) main.o dwt.o components.o fdwt53.o fdwt97.o common.o rdwt53.o rdwt97.o -o dwt2d -lm

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

main.o: main.cu 
	$(HIPCC) $(HIPCC_FLAGS) -c main.cu

dwt.o: dwt.cu 
	$(HIPCC) $(HIPCC_FLAGS) -c dwt.cu

components.o: components.cu 
	$(HIPCC) $(HIPCC_FLAGS) -c components.cu

fdwt53.o: dwt_cuda/fdwt53.cu 
	$(HIPCC) $(HIPCC_FLAGS) -c dwt_cuda/fdwt53.cu

fdwt97.o: dwt_cuda/fdwt97.cu 
	$(HIPCC) $(HIPCC_FLAGS) -c dwt_cuda/fdwt97.cu

rdwt53.o: dwt_cuda/rdwt53.cu 
	$(HIPCC) $(HIPCC_FLAGS) -c dwt_cuda/rdwt53.cu

rdwt97.o: dwt_cuda/rdwt97.cu 
	$(HIPCC) $(HIPCC_FLAGS) -c dwt_cuda/rdwt97.cu

common.o: dwt_cuda/common.cu 
	$(HIPCC) $(HIPCC_FLAGS) -c dwt_cuda/common.cu

clean:
	rm -f *.o *~ dwt2d isa.txt *.bmp.dwt.* *.out *.diff *.perf
