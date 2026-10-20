"""Visual check of packed dataset read exactly as training code will read it.

  python data/gallery.py                 # 2 random trajectories per split
  python data/gallery.py --per-split 4 --seed 1

Writes outputs/gallery.png (density at frames 0, 25, 50 at 128^2, and frame 50 at 64^2, for random
trajectories from each split) and outputs/value_ranges.png (distributions of density and pressure per
split, which shows how far the held-out set sits from the training data).
"""
import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
GAMMA = 1.4
SPLITS = ["train", "val", "test", "ood"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", default=str(REPO / "data" / "packed"))
    ap.add_argument("--per-split", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    packed, rng = Path(args.packed), np.random.default_rng(args.seed)
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    # ---- gallery: random trajectories from every split ----
    rows = []
    for split in SPLITS:
        with h5py.File(packed / f"{split}_128.h5", "r") as f:
            picks = rng.choice(len(f["index"]), size=min(args.per_split, len(f["index"])), replace=False)
            for k in sorted(picks):
                rows.append((split, int(k), f["family"][k].decode(), int(f["index"][k]),
                             f["U"][k, [0, 25, 50], 0], f["t"][k]))     # density at three frames

    cols = ["frame 0 (128²)", "frame 25 (128²)", "frame 50 (128²)", "frame 50 (64²)"]
    fig, axes = plt.subplots(len(rows), 4, figsize=(9.6, 2.5 * len(rows)), squeeze=False)
    for r, (split, k, family, index, rho, t) in enumerate(rows):
        with h5py.File(packed / f"{split}_64.h5", "r") as f:
            rho64 = f["U"][k, 50, 0]
            assert f["family"][k].decode() == family and int(f["index"][k]) == index, "64² and 128² files disagree"
        vmin, vmax = rho.min(), rho.max()                              # same colour scale across a row
        for c, img in enumerate([rho[0], rho[1], rho[2], rho64]):
            ax = axes[r, c]
            ax.imshow(img, origin="lower", extent=[0, 1, 0, 1], cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=9)
        axes[r, 0].set_ylabel(f"{split}\n{family} #{index}\nt = {t[0]:.2f}..{t[-1]:.2f}", fontsize=8)
    fig.suptitle("Packed dataset: density of random trajectories", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_dir / "gallery.png", dpi=110)
    print(f"Saved {out_dir / 'gallery.png'}")

    # ---- value ranges per split (sampled frames, for speed) ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for split in SPLITS:
        with h5py.File(packed / f"{split}_64.h5", "r") as f:
            N = len(f["index"])
            ks = np.sort(rng.choice(N, size=min(40, N), replace=False))
            U = np.stack([f["U"][k, ::10] for k in ks]).astype(np.float64)   # every 10th frame
        rho = U[:, :, 0]
        p = (GAMMA - 1) * (U[:, :, 3] - 0.5 * (U[:, :, 1]**2 + U[:, :, 2]**2) / rho)
        bins = np.linspace(-3.5, 1.5, 80)
        axes[0].hist(np.log10(rho).ravel(), bins=bins, density=True, histtype="step", lw=1.5, label=split)
        axes[1].hist(np.log10(p).ravel(), bins=bins, density=True, histtype="step", lw=1.5, label=split)
    for ax, name in zip(axes, ["log10 density", "log10 pressure"]):
        ax.set_xlabel(name); ax.set_yscale("log"); ax.legend(fontsize=8)
    axes[0].set_ylabel("fraction of cells (density)")
    fig.suptitle("Value distributions per split (64², every 10th frame of up to 40 trajectories)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "value_ranges.png", dpi=110)
    print(f"Saved {out_dir / 'value_ranges.png'}")


if __name__ == "__main__":
    main()