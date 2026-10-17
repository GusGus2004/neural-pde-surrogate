"""Growth of Kelvin-Helmholtz instability: amplitude of the seeded mode against time

Uses diagnostic of McNally, Lyra & Passy (2012), eqs. 4-7: project vertical velocity v onto
sin(4 pi x) and cos(4 pi x), weighted by exp(-4 pi |distance to the nearest shear layer|), so that
only seeded wavelength near the interfaces is measured

  python experiments/kh_growth.py RUN_DIR [RUN_DIR ...]
"""
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def mode_amplitude(path):
    with h5py.File(path, "r") as f:
        v, x, y, t = f["v"][:], f["x"][:], f["y"][:], f.attrs["t"]
    X, Y = np.meshgrid(x, y)                                 
    dist = np.where(Y < 0.5, np.abs(Y - 0.25), np.abs((1.0 - Y) - 0.25))
    w = np.exp(-4 * np.pi * dist)                           
    s = np.sum(v * np.sin(4 * np.pi * X) * w)
    c = np.sum(v * np.cos(4 * np.pi * X) * w)
    d = np.sum(w)
    return t, 2 * np.sqrt((s / d) ** 2 + (c / d) ** 2)


def main():
    run_dirs = [Path(a) for a in sys.argv[1:]]
    if not run_dirs:
        sys.exit(__doc__)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for run_dir in run_dirs:
        snaps = sorted(run_dir.glob("snap_*.h5"))
        t, amp = np.array([mode_amplitude(p) for p in snaps]).T
        ax.semilogy(t, amp, "o-", ms=3, label=run_dir.name)
        lin = (t >= 0.6) & (t <= 1.2)
        rate = np.polyfit(t[lin], np.log(amp[lin]), 1)[0]
        print(f"{run_dir.name}: amplitude {amp[0]:.4f} -> {amp.max():.4f} (max at t = {t[amp.argmax()]:.2f}), "
              f"growth rate in 0.6 <= t <= 1.2: {rate:.2f}")
    ax.set_xlabel("t")
    ax.set_ylabel("mode amplitude (McNally et al. 2012)")
    ax.set_title("Kelvin-Helmholtz: growth of the seeded mode")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = Path("outputs") / "kh_growth.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()