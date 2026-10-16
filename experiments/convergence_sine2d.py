"""Convergence study for the 2D diagonal advection test (euler2d --test sine2d).

Exact solution at t = 1: rho = 1 + 0.2 sin(2 pi (x + y)), identical to initial condition.
"""
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

NS = [32, 64, 128, 256]
LIMITERS = ["none", "vanleer"]
RUNS = Path(sys.argv[1] if len(sys.argv) > 1 else "solver/runs")


def errors(run_dir):
    snap = sorted(run_dir.glob("snap_*.h5"))[-1]          # final
    with h5py.File(snap, "r") as f:
        rho, x, y, t = f["rho"][:], f["x"][:], f["y"][:], f.attrs["t"]
    X, Y = np.meshgrid(x, y)                               # shape (ny, nx), like rho
    exact = 1.0 + 0.2 * np.sin(2 * np.pi * (X + Y - 2 * t))
    err = np.abs(rho - exact)
    dA = (x[1] - x[0]) * (y[1] - y[0])
    return err.sum() * dA, err.max()


fig, ax = plt.subplots(figsize=(6, 4.5))
for lim in LIMITERS:
    l1, linf = np.array([errors(RUNS / f"sine2d_hllc_{lim}_N{n}") for n in NS]).T
    print(f"\n{lim}")
    print(f"{'N':>6} {'L1':>11} {'order':>6} {'Linf':>11} {'order':>6}")
    for k, n in enumerate(NS):
        o1 = "" if k == 0 else f"{np.log2(l1[k-1] / l1[k]):6.2f}"
        oi = "" if k == 0 else f"{np.log2(linf[k-1] / linf[k]):6.2f}"
        print(f"{n:6d} {l1[k]:11.3e} {o1:>6} {linf[k]:11.3e} {oi:>6}")
    ax.loglog(NS, l1, "o-", label=f"HLLC + {lim}")

n = np.array(NS, dtype=float)
ax.loglog(n, 0.3 * (n / n[0]) ** -1.0, "k--", lw=0.8, label="order 1")
ax.loglog(n, 0.01 * (n / n[0]) ** -2.0, "k:", lw=0.8, label="order 2")
ax.set_xticks(NS)
ax.set_xticklabels([str(v) for v in NS])
ax.minorticks_off()
ax.set_xlabel("N (cells per side)")
ax.set_ylabel("L1 density error")
ax.set_title("2D diagonal advection, t = 1")
ax.legend()
fig.tight_layout()
out = Path("outputs") / "convergence_sine2d.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")