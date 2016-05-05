#Hipify the unrolled cuda source code to hip compatible code
#hipify nbody-unroll.cu > nbody-unroll.cpp
#Manually add the first argument onto the kernel argument list
#void bodyForce(Body *p, float dt, int n) //before modification
#void bodyForce(hipLaunchParm lp, Body *p, float dt, int n) //after modification

#compile the hipified source code into executable 
if [ -f nbody-unroll ]
then
    rm nbody-unroll
fi
if [ -d /opt/rocm/hip ]
then
    HIP_PATH=/opt/rocm/hip
else
    echo "Please install rocm package"
fi

echo hipcc -I../ -DSHMOO nbody-unroll.cpp -o nbody-unroll
$HIP_PATH/bin/hipcc -I../ -DSHMOO nbody-unroll.cpp -o nbody-unroll

#To print our more details, remove DSHMOO flag
#hipcc -I../  nbody-unroll.cpp -o nbody-unroll

#execute the program
EXE=nbody-unroll
K=1024
for i in {1..8}
do
    echo ./$EXE $K
    ./$EXE $K
    K=$(($K*2))
done

