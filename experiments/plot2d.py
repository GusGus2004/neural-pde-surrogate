"""Plot 2D snapshots written by euler2d.

  python experiments/plot2d.py RUN_DIR             # last snapshot: rho, u, v, p
  python experiments/plot2d.py RUN_DIR --all       # density at every snapshot, in a grid
  python experiments/plot2d.py RUN_DIR --show      # also open the figure window
"""
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    with h5py.File(path, "r") as f:
        data = {k: f[k][:] for k in ("rho", "u", "v", "p", "x", "y")}
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
    return data, attrs


def extent(d):
    dx, dy = d["x"][1] - d["x"][0], d["y"][1] - d["y"][0]
    return [d["x"][0] - dx / 2, d["x"][-1] + dx / 2, d["y"][0] - dy / 2, d["y"][-1] + dy / 2]


def plot_field(ax, d, name, contours=0):
    im = ax.imshow(d[name], origin="lower", extent=extent(d), cmap="viridis", aspect="equal")
    if contours:
        ax.contour(d["x"], d["y"], d[name], levels=contours, colors="k", linewidths=0.4)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run_dir = Path(args[0])
    snaps = sorted(run_dir.glob("snap_*.h5"))
    if not snaps:
        sys.exit(f"No snap_*.h5 files in {run_dir}")
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    if "--all" in sys.argv:
        ncol = 4
        nrow = int(np.ceil(len(snaps) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.0 * nrow), squeeze=False)
        for ax, path in zip(axes.flat, snaps):
            d, a = load(path)
            ax.imshow(d["rho"], origin="lower", extent=extent(d), cmap="viridis")
            ax.set_title(f"t = {a['t']:.3f}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        for ax in axes.flat[len(snaps):]:
            ax.axis("off")
        fig.suptitle(f"{run_dir.name}: density", fontsize=11)
        out = out_dir / f"{run_dir.name}_all.png"
    else:
        d, a = load(snaps[-1])
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        for ax, name in zip(axes.flat, ("rho", "u", "v", "p")):
            plot_field(ax, d, name, contours=30 if name == "rho" else 0)
            ax.set_title(name)
        fig.suptitle(f"{a['test']}, {a['nx']} x {a['ny']}, {a['scheme']} + {a['limiter']}, "
                     f"t = {a['t']:.3f}", fontsize=11)
        out = out_dir / f"{run_dir.name}.png"

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")
    if "--show" in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()