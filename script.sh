#!/bin/bash

# Conda 초기화
eval "$(conda shell.bash hook)"

# 환경 활성화
conda activate wetie_dl_wsl

# 경로 이동 (lecture3_ex.py 있는 폴더로)
cd /mnt/e/Project/위티\ 딥러닝세션/wetie_dl

# 반복 실행
for bs in 128 256 512; do
    for lr in 0.01 0.001; do
        for ep in 10 20; do
            echo "▶️ 실행 중: batchSize=$bs, learningRate=$lr, epoch=$ep"
            python lecture3_ex.py --batchSize $bs --learningRate $lr --epoch $ep
        done
    done
done
