#!/bin/bash
# scaling_collect.sh
# 等 3 个 bsub 任务都跑完后，运行此脚本提取关键指标。
# 用法: ./scaling_collect.sh
# 输出 CSV: graph,nproc,memory_MB,time_s

echo "graph,nproc,memory_MB,time_s"
for nproc in 1 2 4; do
    for f in scaling_${nproc}p_*.out; do
        [ -f "$f" ] || continue
        awk -v np="$nproc" '
            /############# 图/ {
                line = $0
                sub(/.*图：/, "", line)
                sub(/[ \t]+\(nproc=.*/, "", line)
                graph = line
                mem = ""
                t = ""
            }
            /\[Memory\] core static/ {
                line = $0
                sub(/.*= /, "", line)
                sub(/ MB.*/, "", line)
                mem = line
            }
            /Time = .* sec/ {
                line = $0
                sub(/.*Time = /, "", line)
                sub(/ sec.*/, "", line)
                t = line
                if (graph != "" && mem != "" && t != "") {
                    printf("%s,%s,%s,%s\n", graph, np, mem, t)
                    graph = ""; mem = ""; t = ""
                }
            }
        ' "$f"
    done
done