#Hipify the soa cuda source code to hip compatible code
#hipify nbody-soa.cu > nbody-soa.cpp
#Manually add the first argument onto the kernel argument list
#void bodyForce(Body *p, float dt, int n) //before modification
#void bodyForce(hipLaunchParm lp, Body *p, float dt, int n) //after modification

#compile the hipified source code into executable 
if [ -f nbody-soa ]
then
    rm nbody-soa
fi

echo hipcc -I../ -DSHMOO nbody-soa.cpp -o nbody-soa
/opt/rocm/bin/hipcc -I../ -DSHMOO nbody-soa.cpp -o nbody-soa

#To print our more details, remove DSHMOO flag
#hipcc -I../  nbody-soa.cpp -o nbody-soa

#execute the program
EXE=nbody-soa
K=1024
for i in {1..8}
do
    echo ./$EXE $K
    ./$EXE $K
    K=$(($K*2))
done
