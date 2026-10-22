#!/bin/bash
# One GPU job per line of a job list written by cluster/sweep.py, as a SLURM job array:
#   sbatch --array=0-8%4 cluster/array.sbatch cluster/jobs/<sweep>.txt
# Array task i runs line i+1 of the list. %4 = at most 4 jobs at once.
#
# EDIT the account line (your project's GPU account, ends in -GPU) and check the time limit for your
# service level before the first submission.
#SBATCH -J npde
#SBATCH -A CHANGEME-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
JOBLIST=$1
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$JOBLIST")
[ -n "$LINE" ] || { echo "no line $((SLURM_ARRAY_TASK_ID + 1)) in $JOBLIST"; exit 1; }

# Environment (see cluster/SETUP.md). Adjust if you installed it elsewhere.
source ~/miniforge3/etc/profile.d/conda.sh
conda activate npde-gpu
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9      # JAX may use 90% of the GPU's memory

cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import jax; print('JAX devices:', jax.devices())"
echo "running: $LINE"
eval "$LINE"
