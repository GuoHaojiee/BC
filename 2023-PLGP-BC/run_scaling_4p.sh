#!/bin/bash
#BSUB -J bc_scaling_4p
#BSUB -W 01:00
#BSUB -n 4
#BSUB -q normal 
#BSUB -R "span[ptile=2]"
#BSUB -m "polus-c3-ib polus-c4-ib"
#BSUB -gpu "num=2:mode=shared"
#BSUB -o scaling_4p_%J.out
#BSUB -e scaling_4p_%J.err

module load SpectrumMPI

for graph in rmat-12 rmat-14 random-12 random-14; do
    echo ""
    echo "############# 图：${graph}  (nproc=4) #############"
    if [ ! -f "${graph}" ]; then continue; fi
    
    for i in {1..3}; do
        echo "--- Run #$i ---"
        mpiexec -n 4 ./solution_mpi -in ${graph} -out ${graph}-4p.res
    done

    if [ -f "${graph}.ans" ] && [ -f "${graph}-4p.res" ]; then
        ./validation -ans ${graph}.ans -res ${graph}-4p.res
    fi
done