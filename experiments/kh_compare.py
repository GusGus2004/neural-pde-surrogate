"""Final density of several Kelvin-Helmholtz runs side by side.

  python experiments/kh_compare.py 16 32 64            # resolutions to compare
"""
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt

NS = [int(a) for a in sys.argv[1:]] or [16, 32, 64, 128, 256]
RUNS = Path("solver/runs")

fig, axes = plt.subplots(1, len(NS), figsize=(4 * len(NS), 4.2), squeeze=False)
for ax, n in zip(axes[0], NS):
    snap = sorted((RUNS / f"kh_hllc_vanleer_N{n}").glob("snap_*.h5"))[-1]   # final snapshot
    with h5py.File(snap, "r") as f:
        rho, t = f["rho"][:], f.attrs["t"]
    ax.imshow(rho, origin="lower", extent=[0, 1, 0, 1], cmap="viridis", vmin=1, vmax=2)
    ax.set_title(f"{n} x {n}, t = {t:.2f}")
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()

out = Path("outputs") / f"kh_compare_{'_'.join(map(str, NS))}.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=120)
print(f"Saved {out}")