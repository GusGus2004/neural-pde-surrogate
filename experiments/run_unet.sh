#!/bin/bash
# U-Net (w16, 3 levels) with the same recipe as the ResNet runs: 2000 one-step, then 600 pushforward k=3.
# Conserved and positive variants. Run from the repo root: bash experiments/run_unet.sh
set -e
for variant in cons pos; do
    flag=$([ "$variant" = pos ] && echo "--positive" || true)
    python -u -m surrogate.train --name unet_${variant} --model unet --width 16 --levels 3 $flag \
        --steps 2000 --batch 32 --eval-every 200 | tee runs/unet_${variant}.txt
    python -u -m surrogate.train --name unet_${variant}_pf3 --init runs/unet_${variant} \
        --unroll 3 --mode pushforward --steps 600 --lr 3e-4 --warmup 20 --eval-every 100 | tee runs/unet_${variant}_pf3.txt
    python -u -m surrogate.rollout runs/unet_${variant}_pf3 --which last | tee runs/unet_${variant}_pf3_rollout.txt
done