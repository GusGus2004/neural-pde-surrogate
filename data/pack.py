"""Pack raw trajectories into one file per (split, resolution), with normalisation statistics.

  python data/splits.py          # first: assign splits, write data/packed/manifest.json
  python data/pack.py            # then: write data/packed/<split>_<n>.h5 and stats_<n>.json

Each packed file contains
  U        (N, 51, 4, n, n) float32   conserved variables rho, rho*u, rho*v, E
  t        (N, 51)          float64   frame times (KH windows start later than 0)
  family   (N,)             string
  index    (N,)             int
so every sample can be traced back to its raw file. Frames are stored in chunks of one frame, so a
data loader can read any (trajectory, frame) pair without decompressing anything else.

Normalisation statistics come from the TRAIN split only, so nothing about val/test/ood leaks into
training. Every trajectory is also checked for positivity and conservation while it is copied.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parent.parent
GAMMA = 1.4
SPLITS = ["train", "val", "test", "ood"]
VARS = ["rho", "rho_u", "rho_v", "E"]


class RunningStats:
    """Accumulates per-channel mean and standard deviation over many arrays, in float64."""

    def __init__(self, channels):
        self.n = 0
        self.s = np.zeros(channels)
        self.ss = np.zeros(channels)

    def add(self, x, axis):             
        x = x.astype(np.float64)
        self.n += x.size // x.shape[1]
        self.s += x.sum(axis=axis)
        self.ss += (x**2).sum(axis=axis)

    def result(self):
        mean = self.s / self.n
        return mean, np.sqrt(np.maximum(self.ss / self.n - mean**2, 0.0))


def pressure(U):
    rho, mx, my, E = U[:, 0], U[:, 1], U[:, 2], U[:, 3]
    return (GAMMA - 1) * (E - 0.5 * (mx**2 + my**2) / rho)


def pack(split, n, entries, out_dir, stats=None):
    path = out_dir / f"{split}_{n}.h5"
    N, F = len(entries), 51
    worst_drift, min_rho, min_p = 0.0, np.inf, np.inf
    with h5py.File(path, "w") as f:
        U = f.create_dataset("U", (N, F, 4, n, n), dtype="float32", chunks=(1, 1, 4, n, n),
                             compression="gzip", compression_opts=1, shuffle=True)
        T = f.create_dataset("t", (N, F), dtype="float64")
        f["family"] = np.array([e["family"] for e in entries], dtype=h5py.string_dtype())
        f["index"] = np.array([e["index"] for e in entries], dtype=np.int32)
        for k, e in enumerate(entries):
            with h5py.File(REPO / e["path"], "r") as src:
                u = src[f"U_{n}"][:]                        # (51, 4, n, n)
                T[k] = src["t"][:]
            U[k] = u
            # Checks: positivity, and conservation (periodic box: totals constant in time)
            min_rho = min(min_rho, float(u[:, 0].min()))
            min_p = min(min_p, float(pressure(u).min()))
            totals = u.astype(np.float64).mean(axis=(2, 3))  # (51, 4)
            for q in (0, 3):                                 # mass and energy are never ~0
                worst_drift = max(worst_drift, float(np.abs(totals[:, q] - totals[0, q]).max() / abs(totals[0, q])))
            if stats is not None:
                stats["U"].add(u, axis=(0, 2, 3))
                stats["log_rho"].add(np.log(u[:, :1]), axis=(0, 2, 3))
                stats["dU"].add(np.diff(u, axis=0), axis=(0, 2, 3))   # one-frame increments
        f.attrs["variables"] = ", ".join(VARS)
        f.attrs["split"], f.attrs["n"], f.attrs["frames"], f.attrs["dt"] = split, n, F, 0.005
    size_gb = path.stat().st_size / 1e9
    print(f"  {path.name:<14} {N:5d} trajectories  {size_gb:6.2f} GB   min rho {min_rho:.3g}  "
          f"min p {min_p:.3g}  max mass/energy drift {worst_drift:.1e}")
    return min_rho > 0 and min_p > 0 and worst_drift < 1e-5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", default=str(REPO / "data" / "packed"))
    ap.add_argument("--sizes", default="128,64")
    args = ap.parse_args()

    out_dir = Path(args.packed)
    manifest = json.loads((out_dir / "manifest.json").read_text())["trajectories"]
    ok = True
    for n in [int(s) for s in args.sizes.split(",")]:
        print(f"Resolution {n}^2:")
        stats = {"U": RunningStats(4), "log_rho": RunningStats(1), "dU": RunningStats(4)}
        for split in SPLITS:
            entries = sorted((e for e in manifest if e["split"] == split), key=lambda e: (e["family"], e["index"]))
            if entries:
                ok &= pack(split, n, entries, out_dir, stats if split == "train" else None)
        mean, std = stats["U"].result()
        lmean, lstd = stats["log_rho"].result()
        dmean, dstd = stats["dU"].result()
        result = {"source": "train split only", "n": n, "variables": VARS,
                  "mean": mean.tolist(), "std": std.tolist(),
                  "log_rho_mean": float(lmean[0]), "log_rho_std": float(lstd[0]),
                  "delta_mean": dmean.tolist(), "delta_std": dstd.tolist()}
        (out_dir / f"stats_{n}.json").write_text(json.dumps(result, indent=1))
        print(f"  stats_{n}.json: mean {np.round(mean, 3)}  std {np.round(std, 3)}\n")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED: see the lines above")


if __name__ == "__main__":
    main()