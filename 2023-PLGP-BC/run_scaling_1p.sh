#!/bin/bash
#BSUB -J bc_scaling_1p
#BSUB -W 01:00
#BSUB -q normal 
#BSUB -n 1
#BSUB -m "polus-c3-ib"
#BSUB -gpu "num=1:mode=shared"
#BSUB -o scaling_1p_%J.out
#BSUB -e scaling_1p_%J.err

module load SpectrumMPI

for graph in rmat-12 rmat-14 random-12 random-14; do
    echo ""
    echo "############# 图：${graph}  (nproc=1) #############"
    if [ ! -f "${graph}" ]; then continue; fi
    
    # 循环运行 3 次
    for i in {1..3}; do
        echo "--- Run #$i ---"
        mpiexec -n 1 ./solution_mpi -in ${graph} -out ${graph}-1p.res
    done

    if [ -f "${graph}.ans" ] && [ -f "${graph}-1p.res" ]; then
        ./validation -ans ${graph}.ans -res ${graph}-1p.res
    fi
done