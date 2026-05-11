#!/bin/bash
#BSUB -J bc_scale_2p_1node_r14
#BSUB -W 00:30
#BSUB -n 2
#BSUB -R "span[ptile=2]"
#BSUB -m "polus-c3-ib"
#BSUB -gpu "num=2:mode=shared"
#BSUB -o scaling_2p_1node_r14_%J.out
#BSUB -e scaling_2p_1node_r14_%J.err


module load SpectrumMPI

NPROC=2
GRAPH=rmat-14
echo "=== 强扩展性：nproc=${NPROC} / 1 节点 / 2 GPU (ptile=2) ==="
echo "图：${GRAPH}，开始：$(date)"

for i in 1 2 3; do
    echo "--- Run $i ---"
    mpiexec -n ${NPROC} ./solution_mpi -in ${GRAPH} -out ${GRAPH}-${NPROC}p-1node-r${i}.res
done

echo "SCALING nproc=${NPROC} graph=${GRAPH} (single node) done at $(date)"