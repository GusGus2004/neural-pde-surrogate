#!/bin/bash
# FNO (w32, 4 layers, 16 modes) with the same recipe as the ResNet and U-Net runs:
# 2000 one-step, then 600 pushforward k=3. Conserved and positive variants.
# Run from the repo root: bash experiments/run_fno.sh
set -e
for variant in cons pos; do
    flag=$([ "$variant" = pos ] && echo "--positive" || true)
    python -u -m surrogate.train --name fno_${variant} --model fno --width 32 --depth 4 --modes 16 $flag \
        --steps 2000 --batch 32 --eval-every 200 | tee runs/fno_${variant}.txt
    python -u -m surrogate.train --name fno_${variant}_pf3 --init runs/fno_${variant} \
        --unroll 3 --mode pushforward --steps 600 --lr 3e-4 --warmup 20 --eval-every 100 | tee runs/fno_${variant}_pf3.txt
    python -u -m surrogate.rollout runs/fno_${variant}_pf3 --which last | tee runs/fno_${variant}_pf3_rollout.txt
done