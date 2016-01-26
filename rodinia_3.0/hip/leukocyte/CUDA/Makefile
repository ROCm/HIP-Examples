HIP_PATH= ../../../../..
include ../../../common/make.config

OUTPUT = -DOUTPUT

MATRIX_DIR = ../meschach_lib

# C compiler
CC = gcc
CC_FLAGS = -g -O3 -Wall -I$(MATRIX_DIR)

ifeq ($(HIP_PLATFORM), nvcc)
CC_FLAGS+=-DMEMCPYTOSYMBOL
HIPCC_FLAGS+=-DMEMCPYTOSYMBOL
endif

$(TESTNAME): detect_main.o avilib.o find_ellipse.o find_ellipse_kernel.o track_ellipse.o track_ellipse_kernel.o misc_math.o $(MATRIX_DIR)/meschach.a

	$(HIPLD)  avilib.o find_ellipse.o find_ellipse_kernel.o track_ellipse.o track_ellipse_kernel.o misc_math.o detect_main.o -o leukocyte $(MATRIX_DIR)/meschach.a -lm


%.o: %.[ch]
	$(CC) $(OUTPUT) $(CC_FLAGS) $< -c

detect_main.o: detect_main.c find_ellipse.h track_ellipse.h avilib.h
	$(CC) $(CC_FLAGS) detect_main.c -c

find_ellipse_kernel.o: find_ellipse_kernel.cu find_ellipse_kernel.h
	$(HIPCC)  $(HIPCC_FLAGS) -c find_ellipse_kernel.cu 

track_ellipse_kernel.o: track_ellipse_kernel.cu track_ellipse_kernel.h
	$(HIPCC)  $(HIPCC_FLAGS) -I$(MATRIX_DIR) -c track_ellipse_kernel.cu 


find_ellipse.o: avilib.h

track_ellipse.o: find_ellipse.h avilib.h

$(MATRIX_DIR)/meschach.a:
	cd $(MATRIX_DIR); ./configure --with-all; make all; make clean

clean:
	rm -f *.o *~ leukocyte *.linkinfo

include ../../../../common/hip.epilogue.make
