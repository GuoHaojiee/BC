#!/bin/bash
#BSUB -J bc_scaling_2p
#BSUB -W 00:30
#BSUB -n 2
#BSUB -R "span[ptile=1]"
#BSUB -m "polus-c3-ib polus-c4-ib"
#BSUB -gpu "num=2:mode=shared"
#BSUB -o scaling_2p_%J.out
#BSUB -e scaling_2p_%J.err

module load SpectrumMPI

echo "================================================================"
echo "  强扩展性测试：nproc=2   (c3+c4 各 1 进程，强制跨节点 IB)"
echo "  开始时间：$(date)"
echo "  主机：$(hostname)"
echo "================================================================"

for graph in rmat-12 rmat-14 random-12 random-14; do
    echo ""
    echo "############# 图：${graph}  (nproc=2) #############"
    if [ ! -f "${graph}" ]; then
        echo "!!! 文件 ${graph} 不存在，跳过 !!!"
        continue
    fi
    mpiexec -n 2 ./solution_mpi -in ${graph} -out ${graph}-2p.res
    if [ -f "${graph}.ans" ] && [ -f "${graph}-2p.res" ]; then
        echo "--- validation [${graph}, 2p] ---"
        ./validation -ans ${graph}.ans -res ${graph}-2p.res
    fi
done

echo ""
echo "================================================================"
echo "  完成时间：$(date)"
echo "================================================================"