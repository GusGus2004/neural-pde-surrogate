"""Run few random samples of a family and show how they evolve

  python data/preview.py riemann 8                # 8 samples, indices 0..7, solver at 256^2
  python data/preview.py riemann 8 --n 128 --start 100
"""
import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from families import sample   # noqa: E402
from ic import write_ic       # noqa: E402

EXE = Path(__file__).parent.parent / "solver" / "build" / "euler2d"

ap = argparse.ArgumentParser()
ap.add_argument("family")
ap.add_argument("count", type=int)
ap.add_argument("--n", type=int, default=256, help="solver resolution")
ap.add_argument("--start", type=int, default=0, help="first index")
args = ap.parse_args()

frames_to_show = [0, 25, 50]
fig, axes = plt.subplots(args.count, len(frames_to_show), figsize=(2.4 * len(frames_to_show), 2.4 * args.count),
                         squeeze=False)
with tempfile.TemporaryDirectory() as tmp:
    for row, index in enumerate(range(args.start, args.start + args.count)):
        rho, u, v, p, t_end, params = sample(args.family, index, args.n)
        ic_path, traj_path = Path(tmp) / "ic.h5", Path(tmp) / "traj.h5"
        write_ic(ic_path, rho, u, v, p, t_end=t_end, test=args.family)
        t0 = time.time()
        t_start = params.get("t_start", 0.0)          
        res = subprocess.run([str(EXE), "--init", str(ic_path), "--nout", "50", "--traj", str(traj_path),
                              "--save", str(args.n // 2), "--tstart", str(t_start)],
                             capture_output=True, text=True)
        secs = time.time() - t0
        if res.returncode != 0:
            print(f"index {index}: FAILED\n{res.stderr.strip()}")
            for ax in axes[row]:
                ax.text(0.5, 0.5, "failed", ha="center"); ax.axis("off")
            continue
        steps = res.stdout.split("Done: ")[1].split(" steps")[0]
        with h5py.File(traj_path) as f:
            U = f[f"U_{args.n // 2}"][:]
        rho_all = U[:, 0]
        print(f"index {index}: {steps:>5} steps, {secs:5.1f} s, density {rho_all.min():.3f} .. {rho_all.max():.3f}")
        for ax, k in zip(axes[row], frames_to_show):
            ax.imshow(rho_all[k], origin="lower", extent=[0, 1, 0, 1], cmap="viridis")
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"frame {k}", fontsize=9)
        axes[row, 0].set_ylabel(f"#{index}  (t = {t_start:.2f}..{t_end:.2f})", fontsize=8)

fig.suptitle(f"{args.family}: density, solver {args.n}², stored {args.n // 2}²", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.98))  
out = Path("outputs") / f"preview_{args.family}.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=110)
print(f"Saved {out}")