#!/bin/bash

: ${HIP_PLATFORM:="hcc"}

# vector_add
echo
echo "==== vectorAdd ===="
cd vectorAdd
make clean
make
cd ..


#---
echo
: ${HIP_MAKEJ:="4"}
echo "==== Rodinia ===="
cd rodinia_3.0/hip
make clean -j ${HIP_MAKEJ}
make test 
cd ..



