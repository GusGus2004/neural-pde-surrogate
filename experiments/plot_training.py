"""Plot training curves from one or more runs: python experiments/plot_training.py runs/<name> [runs/<name2> ...]

Left: train and validation loss (log scale). Right: validation loss relative to persistence
(1 = no better than "nothing changes"). Saves outputs/training_<name>.png (named after the first run).
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

runs = [Path(r) for r in sys.argv[1:]]
if not runs:
    sys.exit(__doc__)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
for run in runs:
    rows = list(csv.DictReader(open(run / "log.csv")))
    step = [int(r["step"]) for r in rows]
    line, = ax1.semilogy(step, [float(r["val_loss"]) for r in rows], label=f"{run.name} val")
    ax1.semilogy(step, [float(r["train_loss"]) for r in rows], "--", color=line.get_color(),
                 label=f"{run.name} train")
    ax2.semilogy(step, [float(r["val_ratio"]) for r in rows], color=line.get_color(), label=run.name)
ax1.set(xlabel="step", ylabel="MSE (normalised)", title="Loss")
ax2.axhline(1, color="grey", lw=0.8, ls=":")
ax2.set(xlabel="step", ylabel="val MSE / persistence MSE", title="Relative to persistence")
for ax in (ax1, ax2):
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
fig.tight_layout()
Path("outputs").mkdir(exist_ok=True)
path = Path("outputs") / f"training_{runs[0].name}.png"
fig.savefig(path, dpi=130)
print("saved", path)