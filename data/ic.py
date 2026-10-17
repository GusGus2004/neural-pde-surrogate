"""Initial conditions for euler2d (read with `euler2d --init FILE`).

File format same as euler2d's snapshots: datasets rho, u, v, p of shape (ny, nx), plus
attributes describing the grid. Extra attributes (e.g. sampled parameters) are kept as a record.

  python data/ic.py lax_liu3 256 ic_lax_liu3.h5      # write the Lax-Liu configuration 3 test case
"""
import sys

import h5py
import numpy as np


def cell_centres(n, lo=0.0, hi=1.0):
    dx = (hi - lo) / n
    return lo + (np.arange(n) + 0.5) * dx


def write_ic(path, rho, u, v, p, *, x_range=(0.0, 1.0), y_range=(0.0, 1.0),
             bc=("periodic", "periodic"), t_end=None, test="init", attrs=None):
    """Write an initial condition. rho, u, v, p are arrays of shape (ny, nx)."""
    rho, u, v, p = (np.asarray(a, dtype=np.float64) for a in (rho, u, v, p))
    ny, nx = rho.shape
    assert u.shape == v.shape == p.shape == (ny, nx), "all fields must have shape (ny, nx)"
    assert (rho > 0).all() and (p > 0).all(), "density and pressure must be positive"
    with h5py.File(path, "w") as f:
        for name, arr in (("rho", rho), ("u", u), ("v", v), ("p", p)):
            f[name] = arr
        f["x"] = cell_centres(nx, *x_range)
        f["y"] = cell_centres(ny, *y_range)
        a = f.attrs
        a["nx"], a["ny"] = np.int32(nx), np.int32(ny)
        a["x_min"], a["x_max"] = map(float, x_range)
        a["y_min"], a["y_max"] = map(float, y_range)
        a["bc_x"], a["bc_y"] = bc
        a["test"] = test
        if t_end is not None:
            a["t_end"] = float(t_end)
        for key, val in (attrs or {}).items():
            a[key] = val


def lax_liu3(n):
    """Lax & Liu (1998) configuration 3, identical to euler2d --test lax_liu3."""
    x = cell_centres(n)
    X, Y = np.meshgrid(x, x)                      # shape (ny, nx)
    right, top = X > 0.5, Y > 0.5
    #                 top right            top left             bottom left           bottom right
    states = {(1, 1): (1.5, 0.0, 0.0, 1.5), (0, 1): (0.5323, 1.206, 0.0, 0.3),
              (0, 0): (0.138, 1.206, 1.206, 0.029), (1, 0): (0.5323, 0.0, 1.206, 0.3)}
    rho, u, v, p = (np.zeros((n, n)) for _ in range(4))
    for (r, t), (rr, uu, vv, pp) in states.items():
        mask = (right == bool(r)) & (top == bool(t))
        rho[mask], u[mask], v[mask], p[mask] = rr, uu, vv, pp
    return rho, u, v, p


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "lax_liu3":
        sys.exit(__doc__)
    n, path = int(sys.argv[2]), sys.argv[3]
    write_ic(path, *lax_liu3(n), bc=("transmissive", "transmissive"), t_end=0.3, test="lax_liu3_file")
    print(f"Wrote {path}")