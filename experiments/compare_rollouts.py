"""Compare rollout evaluations of several runs on one figure, plus a summary table.

  python experiments/compare_rollouts.py runs/ft_pf3 runs/pos_pf3 runs/unet_cons_pf3 runs/unet_pos_pf3
  python experiments/compare_rollouts.py runs/a runs/b --labels "ResNet" "U-Net" --out outputs/fig.png

Reads runs/<name>/rollout_<split>.npz written by surrogate/rollout.py (run that first, for every run).
Nothing is recomputed, so this takes a second.

Figure (4 panels, error = RMSE / train std averaged over the 4 variables):
  1. median error vs rollout step, with persistence (dotted grey) as the do-nothing baseline
  2. fraction of trajectories that have blown up (error has exceeded 2) by each step
  3. median error per family at each step (blast solid, riemann dashed)
  4. median |total-mass drift| vs step (the solver conserves mass to ~1e-8)

The table's 'blown' column uses the same definition as rollout.py (error > 2 AT step 50), so it matches the
numbers printed there; the curve in panel 2 counts a trajectory from the first step its error exceeds 2.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("runs", nargs="+", help="run directories, e.g. runs/ft_pf3")
p.add_argument("--labels", nargs="+", help="legend labels (default: run names)")
p.add_argument("--split", default="val")
p.add_argument("--out", default=None, help="output image (default outputs/compare_<split>.png)")
args = p.parse_args()

labels = args.labels or [Path(r).name for r in args.runs]
if len(labels) != len(args.runs):
    raise SystemExit("need one label per run")

data = []
for run in args.runs:
    path = Path(run) / f"rollout_{args.split}.npz"
    if not path.exists():
        raise SystemExit(f"{path} not found: run  python -m surrogate.rollout {run} --which last  first")
    data.append(dict(np.load(path)))

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
(ax_med, ax_blown), (ax_fam, ax_mass) = axes
colours = plt.cm.tab10.colors
steps = None

print(f"\n{'run':<22}{'blown':>8}{'non-phys':>10}{'med t10':>9}{'med t50':>9}{'mean t50':>10}{'mass drift t50':>16}")
for d, label, c in zip(data, labels, colours):
    e = d["rmse"].mean(axis=2)                                   # (N, T): error per trajectory and step
    n, T = e.shape
    steps = np.arange(1, T + 1)
    blown_by = np.maximum.accumulate(~(e <= 2), axis=1)          # has the trajectory blown up by step t? (NaN = blown)
    mass = np.abs(d["drift"][:, :, 0])                           # (N, T)
    fam = d["family"].astype(str)

    ax_med.semilogy(steps, np.median(e, axis=0), color=c, label=label)
    ax_blown.plot(steps, blown_by.mean(axis=0), color=c, label=label)
    for f, ls in (("blast", "-"), ("riemann", "--")):
        sel = fam == f
        if sel.any():
            ax_fam.semilogy(steps, np.median(e[sel], axis=0), ls, color=c, label=f"{label} ({f})")
    ax_mass.semilogy(steps, np.maximum(np.median(mass, axis=0), 1e-12), color=c, label=label)

    nonphys = (d["bad_cells"].sum(axis=1) > 0).sum()
    print(f"{label:<22}{(~(e[:, -1] <= 2)).sum():>5}/{n:<2}{nonphys:>7}/{n:<2}{np.median(e[:, 9]):>9.3f}"
          f"{np.median(e[:, -1]):>9.3f}{np.mean(e[:, -1]):>10.3f}{np.median(mass[:, -1]):>16.2e}")

persist = data[0]["rmse_persist"].mean(axis=2)
ax_med.semilogy(steps, np.median(persist, axis=0), ":", color="grey", label="persistence")

ax_med.set(title="Median error over trajectories", ylabel="RMSE / train std")
ax_blown.set(title="Fraction blown up (error has exceeded 2)", ylabel="fraction", ylim=(-0.01, None))
ax_fam.set(title="Median error by family (blast solid, riemann dashed)", ylabel="RMSE / train std")
ax_mass.set(title="Median |total mass drift|", ylabel="relative drift")
for ax in axes.ravel():
    ax.set_xlabel("rollout step (dt = 0.005)")
    ax.grid(alpha=0.3, which="both")
ax_med.legend(fontsize=8)
ax_blown.legend(fontsize=8)
ax_fam.legend(fontsize=7, ncol=2)
ax_mass.legend(fontsize=8)
fig.suptitle(f"Rollout comparison on the {args.split} split ({len(data[0]['rmse'])} trajectories)")
fig.tight_layout()

out = Path(args.out or f"outputs/compare_{args.split}.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130)
print(f"\nsaved {out}")