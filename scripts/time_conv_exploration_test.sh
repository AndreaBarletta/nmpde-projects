#!/bin/bash

exe=$1
if [ -z "$exe" ]; then
    echo "Usage: $0 <path_to_executable>"
    exit 1
fi

T_min=1
T_max=5
mesh_list=(10 20 50 100 200)

for mesh in "${mesh_list[@]}"; do
    mkdir -p results/m${mesh}
    echo "Running tests for mesh size ${mesh}"
    for T in $(seq $T_min $T_max); do
        dt=$((T+2))
        # run test, extract from stdout the L2 error at final time
        mkdir -p results/m${mesh}/Test_T${T}
        echo -e "\t Running tests for T=2e-${T} and dt=1e-${T}"
        mpirun -n 8 $exe 2e-$T 1e-$T ../mesh/mesh-square-${mesh}.msh | tail -200 | grep "U - L2 error" | tail -10 > results/m${mesh}/Test_T${T}/result_2e-${T}_1e-${T}.txt  
        echo -e "\t Running tests for T=2e-${T} and dt=0.5e-${T}"
        mpirun -n 8 $exe 2e-$T 0.5e-$dt ../mesh/mesh-square-${mesh}.msh | tail -200 | grep "U - L2 error" | tail -10 > results/m${mesh}/Test_T${T}/result_2e-${T}_05e-${dt}.txt  
        echo -e "\t Running tests for T=2e-${T} and dt=0.25e-${T}"
        mpirun -n 8 $exe 2e-$T 0.25e-$dt ../mesh/mesh-square-${mesh}.msh | tail -200 | grep "U - L2 error" | tail -10 > results/m${mesh}/Test_T${T}/result_2e-${T}_025e-${dt}.txt  
        echo -e "\t Running tests for T=2e-${T} and dt=0.125e-${T}"
        mpirun -n 8 $exe 2e-$T 0.125e-$dt ../mesh/mesh-square-${mesh}.msh | tail -200 | grep "U - L2 error" | tail -10 > results/m${mesh}/Test_T${T}/result_2e-${T}_0125e-${dt}.txt  
        tail -1 results/m${mesh}/Test_T${T}/result_2e-${T}_1e-${T}.txt >> results/m${mesh}/Test_T${T}/summary.txt
        tail -1 results/m${mesh}/Test_T${T}/result_2e-${T}_05e-${dt}.txt >> results/m${mesh}/Test_T${T}/summary.txt
        tail -1 results/m${mesh}/Test_T${T}/result_2e-${T}_025e-${dt}.txt >> results/m${mesh}/Test_T${T}/summary.txt
        tail -1 results/m${mesh}/Test_T${T}/result_2e-${T}_0125e-${dt}.txt >> results/m${mesh}/Test_T${T}/summary.txt
    done
done