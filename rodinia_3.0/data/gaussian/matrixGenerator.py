#!/usr/bin/python

# This script generates matrices to solve a set of equations with n variables and n unknowns.
# For each iteration, an n x n matrix and a 1 x n vector are created to describe the set of
# equations in the form:
#
#              a0x + b0y + c0z + d0w = e0
#              a1x + b1y + c1z + d1w = e1
#              a2x + b2y + c2z + d2w = e2
#              a3x + b3y + c3z + d3w = e3
#
# where in this case n=4.
#
# The files that are produced contain the dimension in line one, followed by
# the n x n coefficient matrix, the 1 x n vector, and the 1 x n solution.
# Each output file has the name matrixN.txt, where N is the value of n (e.g., matrix4.txt)
#
# The n x n matrix values and solution vector are confined to the values -0.9 to 0.9 
# (one decimal place), but the 1 x n vector can have two decimal place values.
# 
# usage:
# ./matrixGenerator a b c
#
# where a is the start value for n, b is the end value for n, and c is the step
#
# For example:
#
# ./matrixGenerator 16 256 4
#
# produces matrix16.txt, matrix20.txt, ... , matrix252.txt, matrix256.txt
#
# If there are no arguments, a is assumed to be 4 and only matrix4.txt is produced
# If only a is present, only that matrix is produced
# If a and b are present but not c, a step value of 1 is assumed

import random
import sys

a = 4
b = 4
c = 1

# parse command line
try:
    a = int(sys.argv[1])
    b = a+1
    b = int(sys.argv[2])+1
    c = int(sys.argv[3])
except IndexError:
    pass
except ValueError:
    pass

for sqSize in range(a,b,c):
	#size = 100
	#size = sqSize*sqSize
	size = sqSize;
	print size
	solnVec = []
	matrix = []
	bVector = []
	
	filename = "matrix"+str(size)+".txt"
	#f = open("matrix100.txt",'w')
	f = open(filename,'w')
	f.write(str(size)+"\n\n")
	for i in range(size):
		#solnVec.append(random.randint(-size,size)/float(size))
		solnVec.append(random.randint(-10,10)/float(10))
	
	for i in range(size):
		matrixRow = []
		for j in range(size):
			#matrixRow.append(random.randint(-size,size)/float(size))
			matrixRow.append(random.randint(-10,10)/float(10))
		matrix.append(matrixRow)
	
	for i in matrix:
		linResult = 0
		for j in range(size):
			f.write(str(i[j])+"\t")
			linResult+=i[j]*solnVec[j]
		bVector.append(linResult)
		f.write("\n")
	
	f.write("\n")
	for i in bVector:
		f.write(str(i)+"\t")
	
	f.write("\n\n");
	for i in solnVec:
		f.write(str(i)+"\t")
	f.write("\n\n")
	
	f.close()
