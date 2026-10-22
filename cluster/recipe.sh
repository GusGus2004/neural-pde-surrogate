#!/bin/bash
# The standard training recipe for one model, end to end (same everywhere: Mac, cluster, any machine):
#   1. one-step training from scratch      -> runs/<name>
#   2. pushforward fine-tune (k = 3)        -> runs/<name>_pf3
#   3. rollout evaluation of the fine-tune  -> runs/<name>_pf3/rollout_val.npz
# Each stage is skipped if its output already exists, so a job that was killed can simply be resubmitted.
#
#   bash cluster/recipe.sh NAME [train.py model options...]
#   bash cluster/recipe.sh resnet_w64_d8_s0 --model resnet --width 64 --depth 8 --seed 0
#
# Lengths and paths come from environment variables (defaults = the Mac recipe used so far):
#   STEPS=2000 PF_STEPS=600 BATCH=32 WARMUP=100 EVAL_EVERY=200 DATA=data/packed RUNS=runs
set -euo pipefail
NAME=$1; shift
STEPS=${STEPS:-2000}; PF_STEPS=${PF_STEPS:-600}; BATCH=${BATCH:-32}; WARMUP=${WARMUP:-100}
EVAL_EVERY=${EVAL_EVERY:-200}; DATA=${DATA:-data/packed}; RUNS=${RUNS:-runs}
mkdir -p "$RUNS"
echo "=== $NAME on $(hostname) at $(date) | steps $STEPS + pf $PF_STEPS | batch $BATCH | args: $*"

if [ ! -f "$RUNS/$NAME/last.eqx" ]; then
    python -u -m surrogate.train --name "$NAME" "$@" --steps "$STEPS" --batch "$BATCH" --warmup "$WARMUP" \
        --eval-every "$EVAL_EVERY" --data "$DATA" --runs "$RUNS" | tee "$RUNS/$NAME.txt"
else echo "--- $NAME: already trained, skipping"; fi

if [ ! -f "$RUNS/${NAME}_pf3/last.eqx" ]; then
    python -u -m surrogate.train --name "${NAME}_pf3" --init "$RUNS/$NAME" --unroll 3 --mode pushforward \
        --steps "$PF_STEPS" --batch "$BATCH" --lr 3e-4 --warmup 20 --eval-every "$EVAL_EVERY" \
        --data "$DATA" --runs "$RUNS" | tee "$RUNS/${NAME}_pf3.txt"
else echo "--- ${NAME}_pf3: already fine-tuned, skipping"; fi

if [ ! -f "$RUNS/${NAME}_pf3/rollout_val.npz" ]; then
    python -u -m surrogate.rollout "$RUNS/${NAME}_pf3" --which last --data "$DATA" | tee "$RUNS/${NAME}_pf3_rollout.txt"
else echo "--- ${NAME}_pf3: already evaluated, skipping"; fi
echo "=== $NAME done at $(date)"
