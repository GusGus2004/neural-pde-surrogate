"""Checks trajectory file written by `euler2d --traj`.

  python tests/check_trajectory.py TRAJ.h5 [SNAPSHOT_RUN_DIR]

Shapes and frame times; each resolution is block average of finest one;
that mass/momentum/energy totals agree across resolutions (and stay constant in time for periodic
problems); positivity of density and pressure. If snapshot run of the same problem is given, also
checks that the finest level matches it.
"""
import sys
from pathlib import Path

import h5py
import numpy as np

GAMMA = 1.4
failures = 0


def check(name, ok, value=""):
    global failures
    print(f"{name:<60} {'PASS' if ok else 'FAIL'}  {value}")
    failures += not ok


def block_average(U, f):
    """Average f x f blocks over the last two axes."""
    *lead, ny, nx = U.shape
    return U.reshape(*lead, ny // f, f, nx // f, f).mean(axis=(-3, -1))


path = Path(sys.argv[1])
with h5py.File(path, "r") as fh:
    a = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in fh.attrs.items()}
    t = fh["t"][:]
    levels = {int(k[2:]): fh[k][:].astype(np.float64) for k in fh if k.startswith("U_")}
sizes = sorted(levels, reverse=True)
finest = levels[sizes[0]]
nframes = len(t)
area = (a["x_max"] - a["x_min"]) * (a["y_max"] - a["y_min"])
print(f"{path.name}: test = {a['test']}, levels {sizes}, {nframes} frames, t_end = {a['t_end']}")

# Shapes and times
for n in sizes:
    U = levels[n]
    check(f"U_{n} has shape (frames, 4, ny, nx)", U.ndim == 4 and U.shape[:2] == (nframes, 4) and U.shape[3] == n,
          str(U.shape))
check("frame times equally spaced from 0 to t_end",
      np.allclose(t, np.linspace(0, a["t_end"], nframes), rtol=0, atol=1e-12))

# Each coarse level is the block average of finest
for n in sizes[1:]:
    f = sizes[0] // n
    diff = np.abs(block_average(finest, f) - levels[n]).max() / np.abs(finest).max()
    check(f"U_{n} = {f}x{f} block average of U_{sizes[0]} (relative)", diff < 1e-6, f"{diff:.2e}")

# Totals: same at every resolution, and constant in time if periodic
totals = {n: levels[n].mean(axis=(-2, -1)) * area for n in sizes}        # (frames, 4)
for n in sizes[1:]:
    d = np.abs(totals[n] - totals[sizes[0]]).max() / np.abs(totals[sizes[0]]).max()
    check(f"totals of U_{n} equal those of U_{sizes[0]} (relative)", d < 1e-6, f"{d:.2e}")
if a["bc_x"] == "periodic" and a["bc_y"] == "periodic":
    tot = totals[sizes[0]]
    drift = np.abs(tot - tot[0]).max(axis=0) / np.abs(tot[0]).clip(1e-12)
    for q, name in enumerate(["mass", "x-momentum", "y-momentum", "energy"]):
        if abs(tot[0, q]) > 1e-12:
            check(f"periodic: total {name} constant in time (relative)", drift[q] < 1e-6, f"{drift[q]:.2e}")

# Positivity
for n in sizes:
    rho, mx, my, E = np.moveaxis(levels[n], 1, 0)
    p = (GAMMA - 1) * (E - 0.5 * (mx**2 + my**2) / rho)
    check(f"U_{n}: density and pressure positive", (rho > 0).all() and (p > 0).all(),
          f"min rho {rho.min():.3g}, min p {p.min():.3g}")

if len(sys.argv) > 2:
    snaps = sorted(Path(sys.argv[2]).glob("snap_*.h5"))
    worst = 0.0
    for k, sp in enumerate(snaps):
        with h5py.File(sp, "r") as fs:
            rho, u, v, p = (fs[q][:] for q in ("rho", "u", "v", "p"))
        U = np.stack([rho, rho * u, rho * v, p / (GAMMA - 1) + 0.5 * rho * (u**2 + v**2)])
        worst = max(worst, np.abs(U - finest[k]).max() / np.abs(U).max())
    check(f"U_{sizes[0]} matches the {len(snaps)} snapshots (relative, float32)",
          len(snaps) == nframes and worst < 1e-6, f"{worst:.2e}")

print(f"\n{'FAILED' if failures else 'ALL PASSED'} ({failures} failure{'s' if failures != 1 else ''})")
sys.exit(1 if failures else 0)