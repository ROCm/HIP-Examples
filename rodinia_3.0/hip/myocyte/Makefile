# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n

# link objects(binaries) together

include ../../common/make.config

$(TESTNAME):		main.hip.o
	$(HIPLD) main.hip.o  -o myocyte -lm

# compile main function file into object (binary)
main.hip.o:	define.c \
				main.hip.cu \
				work.hip.cu \
				solver.hip.cu \
				embedded_fehlberg_7_8.hip.cu \
				master.hip.cu \
				kernel.hip.cu \
				kernel_ecc.hip.cu \
				kernel_cam.hip.cu \
				kernel_fin.hip.cu \
				work_2.hip.cu \
				solver_2.hip.cu \
				embedded_fehlberg_7_8_2.hip.cu \
				kernel_2.hip.cu \
				kernel_ecc_2.hip.cu \
				kernel_cam_2.hip.cu \
				kernel_fin_2.hip.cu \
				file.c \
				timer.c
	$(HIPCC)  $(HIPCC_FLAGS) -c main.hip.cu  

# delete all object files
clean: 
	rm -f *.o *.out myocyte output.txt *.perf
