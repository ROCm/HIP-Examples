# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n

# link objects(binaries) together
a.out:	main.o
	gcc	main.o \
			-lm -fopenmp -o srad

# compile main function file into object (binary)
main.o: 	main.c \
				define.c \
				graphics.c
	gcc	main.c \
			-c -O3 -fopenmp

# delete all object files
clean:
	rm *.o srad
