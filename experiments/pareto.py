"""Accuracy vs cost: every surrogate and every solver resolution as one point on the same plot.

  python experiments/pareto.py --models runs/ft_pf3 runs/unet_cons_pf3 runs/fno_cons_pf3 \
      --labels ResNet U-Net FNO --solvers runs/solver_32 runs/solver_64 runs/solver_128

Accuracy: median error at the last rollout step (RMSE / train std, averaged over variables), read from
runs/<name>/rollout_<split>.npz -- so run surrogate/rollout.py and experiments/solver_baseline.py first.

Cost: wall-clock time to produce one 50-step trajectory.
  - solver (--solvers): the solver's own compute time on ONE core (measured by solver_baseline.py)
  - solver (--solvers-parallel): throughput with many cores at once = total wall time / trajectories, from
    solver_baseline.py --workers N (the fair comparison with a batched GPU)
  - surrogate: measured here, for a full 50-step rollout, after compiling, best of --repeats:
      batch 1   one trajectory at a time (like running the solver once)
      batch B   B trajectories at once, time divided by B (how a network is used efficiently)
    JAX uses all CPU cores, the solver only one; the solver gets ~3x faster with 4 threads
    (experiments/results.md), so the comparison is generous to the surrogates. Say so when reporting.

Writes outputs/pareto_<split>.png and runs/<name>/timing_<device>.json for each surrogate.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from surrogate.data import PairDataset             # noqa: E402
from surrogate.rollout import rollout              # noqa: E402
from surrogate.train import load_model             # noqa: E402


def median_last(run, split):
    d = np.load(Path(run) / f"rollout_{split}.npz")
    e = d["rmse"].mean(axis=2)
    return float(np.median(e[:, -1])), d


def time_rollout(model, x0, steps, repeats):
    """Best-of-`repeats` wall time of a full rollout from x0 (B, 4, n, n), after one compile run."""
    jax.block_until_ready(rollout(model, x0, steps))               # compile (not timed)
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(rollout(model, x0, steps))
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=[], help="surrogate run directories")
    p.add_argument("--labels", nargs="+", help="labels for --models (default: run names)")
    p.add_argument("--solvers", nargs="+", default=[], help="solver_baseline.py output dirs (one-core time)")
    p.add_argument("--solvers-parallel", nargs="+", default=[],
                   help="solver_baseline.py --workers N output dirs (throughput of N cores)")
    p.add_argument("--split", default="val")
    p.add_argument("--data", default="data/packed")
    p.add_argument("--batch", type=int, default=80, help="trajectories per batched rollout")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--which", default="last")
    args = p.parse_args()
    labels = args.labels or [Path(r).name for r in args.models]
    device = str(jax.devices()[0])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    rows = []

    # Solver points, joined into a line: the solver's own accuracy-vs-cost curve
    for dirs, parallel, style in ((args.solvers, False, "k-o"), (args.solvers_parallel, True, "k--s")):
        if not dirs:
            continue
        pts = []
        for s in dirs:
            err, d = median_last(s, args.split)
            n = re.match(r"solver_(\d+)", Path(s).name).group(1)
            if parallel:
                cost, workers = float(d["throughput_seconds"]), int(d["workers"])
                what = f"{workers} cores"
            else:
                cost, what = float(np.mean(d["solver_seconds"])), "1 core"
            pts.append((cost, err, f"{n}²"))
            rows.append((f"solver {n}^2 ({what})", cost, None, err))
        pts.sort()
        ax.plot([c for c, _, _ in pts], [e for _, e, _ in pts], style, label=f"solver ({what})", zorder=3,
                mfc="white" if parallel else "black")
        for c, e, name in pts:
            ax.annotate(name, (c, e), textcoords="offset points", xytext=(6, 6), fontsize=9)

    # Surrogate points: batch 1 (hollow) and batched (filled), joined by a line
    if args.models:
        ds = PairDataset(args.data, args.split, 64, horizon=1)
        steps = ds.n_frames - 1
        x_all = np.stack([ds.trajectory(k)[0] for k in range(ds.n_traj)])     # true first frames, normalised
        B = min(args.batch, ds.n_traj)
        for run, label, c in zip(args.models, labels, plt.cm.tab10.colors):
            err, _ = median_last(run, args.split)
            model, cfg = load_model(run, args.which)
            t1 = time_rollout(model, jnp.asarray(x_all[:1]), steps, args.repeats)
            tB = time_rollout(model, jnp.asarray(x_all[:B]), steps, args.repeats) / B
            (Path(run) / f"timing_{device.replace(':', '')}.json").write_text(json.dumps(
                {"device": device, "steps": steps, "batch1_s": t1, f"batch{B}_s_per_traj": tB}, indent=2))
            rows.append((label, t1, tB, err))
            ax.plot([t1, tB], [err, err], "-", color=c, alpha=0.6)
            ax.plot(t1, err, "o", mfc="none", color=c, ms=8)
            ax.plot(tB, err, "o", color=c, ms=8, label=label)
            print(f"  timed {label}", flush=True)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"time per 50-step trajectory [s]  ({device})")
    ax.set_ylabel("median error at step 50 (RMSE / train std)")
    ax.set_title(f"Accuracy vs cost ({args.split})  --  surrogates: hollow = batch 1, filled = batch {args.batch}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = Path("outputs") / f"pareto_{args.split}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130)

    print(f"\n{'':26}{'s/traj batch 1':>16}{'s/traj batched':>16}{'median err t50':>16}")
    for name, t1, tB, err in rows:
        print(f"{name:26}{t1:>16.3f}{'' if tB is None else f'{tB:.4f}':>16}{err:>16.3f}")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
