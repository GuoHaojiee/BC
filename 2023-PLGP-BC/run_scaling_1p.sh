#!/bin/bash
#BSUB -J bc_scaling_1p
#BSUB -W 00:30
#BSUB -n 1
#BSUB -m "polus-c3-ib"
#BSUB -gpu "num=1:mode=shared"
#BSUB -o scaling_1p_%J.out
#BSUB -e scaling_1p_%J.err

module load SpectrumMPI

echo "================================================================"
echo "  强扩展性测试：nproc=1   (c3 单节点单 GPU，串行基线)"
echo "  开始时间：$(date)"
echo "  主机：$(hostname)"
echo "================================================================"

for graph in rmat-12 rmat-14 random-12 random-14; do
    echo ""
    echo "############# 图：${graph}  (nproc=1) #############"
    if [ ! -f "${graph}" ]; then
        echo "!!! 文件 ${graph} 不存在，跳过 !!!"
        continue
    fi
    mpiexec -n 1 ./solution_mpi -in ${graph} -out ${graph}-1p.res
    if [ -f "${graph}.ans" ] && [ -f "${graph}-1p.res" ]; then
        echo "--- validation [${graph}, 1p] ---"
        ./validation -ans ${graph}.ans -res ${graph}-1p.res
    fi
done

echo ""
echo "================================================================"
echo "  完成时间：$(date)"
echo "================================================================"