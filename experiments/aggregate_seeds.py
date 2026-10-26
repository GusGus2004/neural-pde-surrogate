"""Average rollout results over random seeds: each architecture as mean +- spread, with error bands.

  python experiments/aggregate_seeds.py runs/seeds_small_*_pf3
  python experiments/aggregate_seeds.py runs/seeds_small_*_pf3 --refs runs/solver_32 runs/solver_64 \
      --labels FNO ResNet U-Net --out outputs/seeds_small_agg.png

Runs are grouped by name with the seed removed: seeds_small_resnet_small_s0_pf3, ..._s1_pf3, ..._s2_pf3
-> one group "seeds_small_resnet_small". Groups are listed in alphabetical order (--labels follows that order).
Each run needs runs/<name>/rollout_<split>.npz (from surrogate/rollout.py); the one-step validation error is
read from runs/<name>/log.csv if present.

Per group it reports mean +- standard deviation over seeds (plus each seed's value), for:
  blown       trajectories with error > 2 at the last step (out of N)
  non-phys    trajectories with any negative density/pressure or NaN
  med t10/50  median error over trajectories at step 10 / at the last step
  mean t50    mean error at the last step (sensitive to blow-ups)
  mass t50    median |relative total-mass drift| at the last step
  one-step    final one-step validation error / persistence (from training)
--refs adds single runs (e.g. the solver baselines) as reference lines/rows.

Figure: median error vs step (line = mean over seeds, band = min..max over seeds), and fraction blown up.
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run_metrics(run, split):
    d = np.load(Path(run) / f"rollout_{split}.npz")
    e = d["rmse"].mean(axis=2)                                        # (N, T)
    out = {
        "curve": np.median(e, axis=0),
        "blown_curve": np.maximum.accumulate(~(e <= 2), axis=1).mean(axis=0),
        "blown": float((~(e[:, -1] <= 2)).sum()),
        "non-phys": float((d["bad_cells"].sum(axis=1) > 0).sum()),
        "med t10": float(np.median(e[:, 9])),
        "med t50": float(np.median(e[:, -1])),
        "mean t50": float(np.mean(e[:, -1])),
        "mass t50": float(np.median(np.abs(d["drift"][:, -1, 0]))),
        "n": e.shape[0],
        "persist": np.median(d["rmse_persist"].mean(axis=2), axis=0),
    }
    log = Path(run) / "log.csv"
    if log.exists():
        rows = list(csv.DictReader(open(log)))
        out["one-step"] = float(rows[-1]["val_ratio"]) if rows else np.nan
    else:
        out["one-step"] = np.nan
    return out


COLUMNS = ["blown", "non-phys", "med t10", "med t50", "mean t50", "mass t50", "one-step"]
FORMATS = {"blown": ".1f", "non-phys": ".1f", "med t10": ".3f", "med t50": ".3f", "mean t50": ".3f",
           "mass t50": ".1e", "one-step": ".4f"}


def fmt(vals, key):
    vals = np.asarray(vals, float)
    f = FORMATS[key]
    if len(vals) == 0 or np.all(np.isnan(vals)):
        return "-"
    if len(vals) == 1:
        return format(vals[0], f)
    return f"{format(np.nanmean(vals), f)} ± {format(np.nanstd(vals, ddof=1), f)}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run directories, several seeds per architecture")
    p.add_argument("--labels", nargs="+", help="names for the groups, in alphabetical group order")
    p.add_argument("--refs", nargs="*", default=[], help="single reference runs, e.g. runs/solver_64")
    p.add_argument("--split", default="val")
    p.add_argument("--out", default=None, help="figure path (default outputs/seeds_<split>.png)")
    args = p.parse_args()

    groups = defaultdict(list)
    for run in args.runs:
        name = Path(run).name
        m = re.match(r"(.*)_s(\d+)(_pf3)?$", name)
        if not m:
            raise SystemExit(f"cannot find a seed (_s<number>) in the run name {name}")
        groups[m.group(1)].append((int(m.group(2)), run))
    names = sorted(groups)
    labels = args.labels or names
    if len(labels) != len(names):
        raise SystemExit(f"{len(names)} groups found ({', '.join(names)}) but {len(labels)} labels given")

    fig, (ax_med, ax_blown) = plt.subplots(1, 2, figsize=(14, 5))
    colours = plt.cm.tab10.colors
    header = f"{'':22}{'seeds':>6}" + "".join(f"{c:>18}" for c in COLUMNS)
    print("\n" + header)
    md = ["| | seeds | " + " | ".join(COLUMNS) + " |", "|---" * (len(COLUMNS) + 2) + "|"]
    persist = None

    for name, label, c in zip(names, labels, colours):
        runs = sorted(groups[name])
        ms = [run_metrics(r, args.split) for _, r in runs]
        persist = ms[0]["persist"]
        cells = [fmt([m[k] for m in ms], k) for k in COLUMNS]
        print(f"{label:22}{len(ms):>6}" + "".join(f"{x:>18}" for x in cells))
        print(f"{'':22}{'':>6}  seeds {[s for s, _ in runs]}: med t50 "
              + ", ".join(f"{m['med t50']:.3f}" for m in ms) + "; blown "
              + ", ".join(f"{m['blown']:.0f}" for m in ms))
        md.append(f"| {label} | {len(ms)} | " + " | ".join(cells) + " |")
        curves = np.stack([m["curve"] for m in ms])
        t = np.arange(1, curves.shape[1] + 1)
        ax_med.semilogy(t, curves.mean(axis=0), color=c, label=f"{label} ({len(ms)} seeds)")
        ax_med.fill_between(t, curves.min(axis=0), curves.max(axis=0), color=c, alpha=0.2)
        bc = np.stack([m["blown_curve"] for m in ms])
        ax_blown.plot(t, bc.mean(axis=0), color=c, label=label)
        ax_blown.fill_between(t, bc.min(axis=0), bc.max(axis=0), color=c, alpha=0.2)

    for ref, ls in zip(args.refs, ["--", "-.", ":", (0, (5, 1))]):
        m = run_metrics(ref, args.split)
        cells = [fmt([m[k]], k) for k in COLUMNS]
        label = Path(ref).name
        print(f"{label:22}{'ref':>6}" + "".join(f"{x:>18}" for x in cells))
        md.append(f"| {label} | ref | " + " | ".join(cells) + " |")
        t = np.arange(1, len(m["curve"]) + 1)
        ax_med.semilogy(t, m["curve"], ls, color="black", label=label)

    if persist is not None:
        ax_med.semilogy(np.arange(1, len(persist) + 1), persist, ":", color="grey", label="persistence")
    ax_med.set(title="Median error (line: mean over seeds, band: min..max)", ylabel="RMSE / train std")
    ax_blown.set(title="Fraction blown up (mean over seeds, band: min..max)", ylabel="fraction")
    for ax in (ax_med, ax_blown):
        ax.set_xlabel("rollout step (dt = 0.005)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(args.out or f"outputs/seeds_{args.split}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)

    print("\nmean ± standard deviation over seeds. Markdown, for results.md / README:\n")
    print("\n".join(md))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
