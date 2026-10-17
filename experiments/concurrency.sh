#!/bin/bash
# Run 1, 2 and 4 independent single-threaded jobs at once, at a small and a large grid,
# and report each job's throughput. If per-job speed drops much more at 512^2 than at 128^2,
# jobs are competing for memory bandwidth.
#
# Run from the repo root:  bash experiments/concurrency.sh

EXE=solver/build/euler2d
export OMP_NUM_THREADS=1

for n in 128 512; do
    tend=$([ "$n" = 128 ] && echo 0.2 || echo 0.05)
    for jobs in 1 2 4; do
        echo "N = $n, $jobs job(s) at once:"
        for k in $(seq "$jobs"); do
            "$EXE" --test kh --n "$n" --tend "$tend" --nout 1 --out "solver/runs_conc/job$k" \
                | grep -o "[0-9.]* million cell-updates per second" | sed 's/^/    /' &
        done
        wait     # wait for all jobs in this batch to finish before starting next
    done
done