"""Convergence study on the smooth advected sine wave.

Exact solution: rho(x, t) = 1 + 0.2 sin(2 pi (x - t)), u = 1, p = 1.
After t = 1 wave has gone once round periodic domain, so exact
solution equals the initial condition.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

T_END = 1.0
NS = [50, 100, 200, 400, 800]
LIMITERS = ["none", "minmod", "vanleer"]
DATA_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "solver")


def errors(path):
    x, rho, _, _ = np.loadtxt(path, delimiter=",", skiprows=1, unpack=True)
    exact = 1.0 + 0.2 * np.sin(2 * np.pi * (x - T_END))
    err = np.abs(rho - exact)
    dx = x[1] - x[0]
    return np.sum(err) * dx, err.max()


fig, ax = plt.subplots(figsize=(6, 4.5))
for lim in LIMITERS:
    l1, linf = np.array([errors(DATA_DIR / f"sine_hllc_{lim}_N{n}.csv") for n in NS]).T
    print(f"\n{lim}")
    print(f"{'N':>6} {'L1':>11} {'order':>6} {'Linf':>11} {'order':>6}")
    for k, n in enumerate(NS):
        o1 = "" if k == 0 else f"{np.log2(l1[k-1] / l1[k]):6.2f}"
        oi = "" if k == 0 else f"{np.log2(linf[k-1] / linf[k]):6.2f}"
        print(f"{n:6d} {l1[k]:11.3e} {o1:>6} {linf[k]:11.3e} {oi:>6}")
    ax.loglog(NS, l1, "o-", label=f"HLLC + {lim}")

n = np.array(NS, dtype=float)
ax.loglog(n, 0.5 * (n / n[0]) ** -1.0, "k--", lw=0.8, label="order 1")
ax.loglog(n, 0.02 * (n / n[0]) ** -2.0, "k:", lw=0.8, label="order 2")
ax.set_xticks(NS)
ax.set_xticklabels([str(n) for n in NS])
ax.minorticks_off()
ax.set_xlabel("N (cells)")
ax.set_ylabel("L1 density error")
ax.set_title("Smooth advection test, t = 1")
ax.legend()
fig.tight_layout()
out = Path("outputs") / "convergence_sine.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")