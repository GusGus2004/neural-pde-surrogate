"""reads the file written by test_io2d and checks every value, shape and attribute"""
import sys

import h5py
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "io_test.h5"
failures = 0


def check(name, ok):
    global failures
    print(f"{name:<52} {'PASS' if ok else 'FAIL'}")
    failures += not ok


with h5py.File(path, "r") as f:
    nx, ny = 6, 4
    j, i = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")   # arrays of shape (ny, nx)

    check("rho has shape (ny, nx) = (4, 6)", f["rho"].shape == (ny, nx))
    check("rho[j, i] == 1 + i", np.allclose(f["rho"][:], 1.0 + i))
    check("u[j, i] == 0.1 i", np.allclose(f["u"][:], 0.1 * i))
    check("v[j, i] == 0.01 j", np.allclose(f["v"][:], 0.01 * j))
    check("p[j, i] == 2 + j", np.allclose(f["p"][:], 2.0 + j))
    check("no ghost-cell values (no negatives)", (f["rho"][:] > 0).all() and (f["p"][:] > 0).all())
    check("x = cell centres 0.25 .. 2.75", np.allclose(f["x"][:], 0.25 + 0.5 * np.arange(nx)))
    check("y = cell centres -0.75 .. 0.75", np.allclose(f["y"][:], -0.75 + 0.5 * np.arange(ny)))

    a = f.attrs
    check("attrs: t = 0.125, step = 42", a["t"] == 0.125 and a["step"] == 42)
    check("attrs: gamma = 1.4, cfl = 0.5", a["gamma"] == 1.4 and a["cfl"] == 0.5)
    check("attrs: nx = 6, ny = 4", a["nx"] == 6 and a["ny"] == 4)
    check("attrs: domain [0, 3] x [-1, 1]",
          (a["x_min"], a["x_max"], a["y_min"], a["y_max"]) == (0.0, 3.0, -1.0, 1.0))

    def s(key):  # HDF5 strings come back as bytes
        val = a[key]
        return val.decode() if isinstance(val, bytes) else str(val)

    check("attrs: bc_x = periodic, bc_y = transmissive", s("bc_x") == "periodic" and s("bc_y") == "transmissive")
    check("attrs: test/scheme/limiter",
          (s("test"), s("scheme"), s("limiter")) == ("io_test", "hllc", "vanleer"))

print(f"\n{'FAILED' if failures else 'ALL PASSED'} ({failures} failure{'s' if failures != 1 else ''})")
sys.exit(1 if failures else 0)