"""OpenMP strong-scaling study: runs euler2d with 1, 2, 4, 8 threads at several grid sizes.

Each run is a short Kelvin-Helmholtz simulation; throughput is read from euler2d's "Done:" line.

  python experiments/scaling.py                  # default sizes and thread counts
  python experiments/scaling.py --quick          # small sizes only, for a fast check
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

EXE = Path("solver/build/euler2d")
THREADS = [1, 2, 4, 8]
SIZES = [64, 128, 256] if "--quick" in sys.argv else [128, 256, 512]
TEND = {64: 0.4, 128: 0.2, 256: 0.1, 512: 0.05}       # keeps every run to few seconds or less


def run(n, threads):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    cmd = [str(EXE), "--test", "kh", "--n", str(n), "--tend", str(TEND[n]), "--nout", "1", "--out", "runs_scaling"]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True).stdout
    m = re.search(r"in ([\d.]+) s \(([\d.]+) million", out)
    return float(m.group(1)), float(m.group(2))


results = {}
print(f"{'N':>5} {'threads':>8} {'time (s)':>9} {'Mcells/s':>9} {'speed-up':>9} {'efficiency':>11}")
for n in SIZES:
    base = None
    for t in THREADS:
        secs, rate = run(n, t)
        base = base or rate
        results[(n, t)] = rate
        print(f"{n:5d} {t:8d} {secs:9.2f} {rate:9.2f} {rate / base:9.2f} {rate / base / t:10.0%}")

fig, ax = plt.subplots(figsize=(6, 4.5))
for n in SIZES:
    ax.plot(THREADS, [results[(n, t)] / results[(n, 1)] for t in THREADS], "o-", label=f"{n} x {n}")
ax.plot(THREADS, THREADS, "k--", lw=0.8, label="ideal")
ax.set_xticks(THREADS)
ax.set_xlabel("OpenMP threads")
ax.set_ylabel("speed-up over 1 thread")
ax.set_title("euler2d strong scaling (Kelvin-Helmholtz)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
out = Path("outputs") / "scaling.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")