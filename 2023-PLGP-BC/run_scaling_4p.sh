#!/bin/bash
#BSUB -J bc_scaling_4p
#BSUB -W 00:30
#BSUB -n 4
#BSUB -R "span[ptile=2]"
#BSUB -m "polus-c3-ib polus-c4-ib"
#BSUB -gpu "num=2:mode=shared"
#BSUB -o scaling_4p_%J.out
#BSUB -e scaling_4p_%J.err

module load SpectrumMPI

echo "================================================================"
echo "  强扩展性测试：nproc=4   (c3+c4 各 2 进程，满载 4 个 P100)"
echo "  开始时间：$(date)"
echo "  主机：$(hostname)"
echo "================================================================"

for graph in rmat-12 rmat-14 random-12 random-14; do
    echo ""
    echo "############# 图：${graph}  (nproc=4) #############"
    if [ ! -f "${graph}" ]; then
        echo "!!! 文件 ${graph} 不存在，跳过 !!!"
        continue
    fi
    mpiexec -n 4 ./solution_mpi -in ${graph} -out ${graph}-4p.res
    if [ -f "${graph}.ans" ] && [ -f "${graph}-4p.res" ]; then
        echo "--- validation [${graph}, 4p] ---"
        ./validation -ans ${graph}.ans -res ${graph}-4p.res
    fi
done

echo ""
echo "================================================================"
echo "  完成时间：$(date)"
echo "================================================================"