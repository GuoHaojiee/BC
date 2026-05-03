#!/bin/bash
#BSUB -J bc_scaling_2p
#BSUB -W 01:00
#BSUB -n 2
#BSUB -q normal 
#BSUB -R "span[ptile=1]"
#BSUB -m "polus-c3-ib polus-c4-ib"
#BSUB -gpu "num=2:mode=shared"
#BSUB -o scaling_2p_%J.out
#BSUB -e scaling_2p_%J.err

module load SpectrumMPI

for graph in rmat-12 rmat-14 random-12 random-14; do
    echo ""
    echo "############# 图：${graph}  (nproc=2) #############"
    if [ ! -f "${graph}" ]; then continue; fi
    
    for i in {1..3}; do
        echo "--- Run #$i ---"
        mpiexec -n 2 ./solution_mpi -in ${graph} -out ${graph}-2p.res
    done

    if [ -f "${graph}.ans" ] && [ -f "${graph}-2p.res" ]; then
        ./validation -ans ${graph}.ans -res ${graph}-2p.res
    fi
done