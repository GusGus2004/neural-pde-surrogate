"""Check two runs produced identical snapshots (e.g. serial vs parallel).

  python tests/compare_runs.py RUN_DIR_A RUN_DIR_B
"""
import sys
from pathlib import Path

import h5py
import numpy as np

a_dir, b_dir = Path(sys.argv[1]), Path(sys.argv[2])
snaps = sorted(p.name for p in a_dir.glob("snap_*.h5"))
assert snaps and snaps == sorted(p.name for p in b_dir.glob("snap_*.h5")), "different snapshot files"

worst = 0.0
for name in snaps:
    with h5py.File(a_dir / name) as fa, h5py.File(b_dir / name) as fb:
        for key in ("rho", "u", "v", "p"):
            worst = max(worst, float(np.abs(fa[key][:] - fb[key][:]).max()))
print(f"{len(snaps)} snapshots compared, max difference = {worst:.3e}")
print("IDENTICAL" if worst == 0.0 else "DIFFERENT")
sys.exit(0 if worst == 0.0 else 1)