#!/bin/bash
if [ -f "rtm8_fortran" ]
then
    rm rtm8_fortran
fi
gfortran  -c rtm8.f
gcc -c -DUNDERSCORE mysecond.c
gfortran  -o rtm8_fortran rtm8.o mysecond.o
