#Hipify the original cuda source code to hip compatible code
#hipify nbody-orig.cu > nbody-orig.cpp

#compile the hipified source code into executable 
if [ -f nbody-orig ]
then
    rm nbody-orig
fi

echo hipcc -I../ -DSHMOO nbody-orig.cpp -o nbody-orig
/opt/rocm/bin/hipcc -I../ -DSHMOO nbody-orig.cpp -o nbody-orig

#To print our more details, remove  flag
#hipcc -I../  nbody-orig.cpp -o nbody-orig

#execute the program

EXE=nbody-orig
K=1024
for i in {1..10}
do
    echo ./$EXE $K
    ./$EXE $K
    K=$(($K*2))
done
