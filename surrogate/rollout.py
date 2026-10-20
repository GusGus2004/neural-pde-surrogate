"""Autoregressive rollout evaluation: start from first frame, then feed the model its own predictions.

  python -m surrogate.rollout runs/<name>                 # all validation trajectories
  python -m surrogate.rollout runs/<name> --split test    # or another split (ood = Kelvin-Helmholtz)

For every trajectory and every time step t = 1..50 it records, in physical units unless stated:
  - RMSE per variable, divided by that variable's train std (so variables are comparable; 1 = "as big as
    the typical variation of the whole dataset")
  - the same for persistence (frame 0 held fixed), as baseline
  - drift of total mass, x/y momentum and energy, relative to the total absolute amount at t=0
    (the solver conserves these to ~1e-8)
  - number of cells with negative density or pressure (the solver never produces any)

Writes runs/<name>/rollout_<split>.npz and two figures in outputs/.
"""
import argparse
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from surrogate.data import PairDataset
from surrogate.train import load_model

GAMMA = 1.4
VARS = ["rho", "rho_u", "rho_v", "E"]


@eqx.filter_jit
def rollout(model, x0, steps):
    """x0: (B, 4, n, n) normalised. Returns (B, steps, 4, n, n): predictions for frames 1..steps."""
    def body(x, _):
        x_next = jax.vmap(model)(x)
        return x_next, x_next
    _, preds = jax.lax.scan(body, x0, None, length=steps)
    return jnp.swapaxes(preds, 0, 1)


def pressure(U):
    """U: (..., 4, n, n) conserved -> pressure (..., n, n)."""
    rho, mx, my, E = U[..., 0, :, :], U[..., 1, :, :], U[..., 2, :, :], U[..., 3, :, :]
    return (GAMMA - 1) * (E - 0.5 * (mx ** 2 + my ** 2) / rho)


def metrics(pred, true, std):
    """pred, true: (T, 4, n, n) physical. Returns dict of (T, ...) arrays."""
    rmse = np.sqrt(np.mean((pred - true) ** 2, axis=(-2, -1))) / std               # (T, 4)
    totals_p, totals_t = pred.sum(axis=(-2, -1)), true.sum(axis=(-2, -1))         # (T, 4)
    # Divide by the total ABSOLUTE amount (= the total for mass and energy); net momentum can be ~0
    drift = (totals_p - totals_t[0]) / np.abs(true[0]).sum(axis=(-2, -1))          # (T, 4)
    with np.errstate(invalid="ignore", divide="ignore"):
        bad = ((pred[:, 0] <= 0) | ~(pressure(pred) > 0)).sum(axis=(-2, -1))       # (T,)
    return {"rmse": rmse, "drift": drift, "bad_cells": bad}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="runs/<name>")
    p.add_argument("--split", default="val")
    p.add_argument("--which", default="best", help="best or last checkpoint")
    p.add_argument("--data", default="data/packed")
    p.add_argument("--batch", type=int, default=16, help="trajectories rolled out at once")
    args = p.parse_args()

    run = Path(args.run)
    model, cfg = load_model(run, args.which)
    ds = PairDataset(args.data, args.split, cfg["n"], horizon=1)
    norm, steps = ds.norm, ds.n_frames - 1
    print(f"{run.name}: rolling out {ds.n_traj} {args.split} trajectories x {steps} steps")

    out = {k: [] for k in ["rmse", "drift", "bad_cells", "rmse_persist"]}
    examples = {}
    for start in range(0, ds.n_traj, args.batch):
        ks = range(start, min(start + args.batch, ds.n_traj))
        true_n = np.stack([ds.trajectory(k) for k in ks])                          # (B, 51, 4, n, n) normalised
        pred_n = np.asarray(rollout(model, jnp.asarray(true_n[:, 0]), steps))     # (B, 50, 4, n, n)
        true, pred = norm.decode(true_n[:, 1:]), norm.decode(pred_n)
        for b, k in enumerate(ks):
            m = metrics(pred[b], true[b], norm.std)
            for key in ("rmse", "drift", "bad_cells"):
                out[key].append(m[key])
            persist = np.broadcast_to(norm.decode(true_n[b, :1]), true[b].shape)
            out["rmse_persist"].append(metrics(persist, true[b], norm.std)["rmse"])
            if ds.family[k] not in examples:                                        # first of each family
                examples[ds.family[k]] = (k, true[b], pred[b])
        print(f"  {ks[-1] + 1}/{ds.n_traj}", flush=True)

    res = {k: np.stack(v) for k, v in out.items()}                                 # (N, T, ...)
    np.savez(run / f"rollout_{args.split}.npz", family=ds.family, index=ds.index, **res)

    fams = sorted(set(ds.family))
    print(f"\nRMSE / train std, averaged over variables and trajectories (persistence in brackets)")
    print(f"{'family':<10}" + "".join(f"{'t=' + str(t):>20}" for t in (1, 10, 25, 50)))
    for fam in fams + ["all"]:
        sel = np.ones(len(ds.family), bool) if fam == "all" else ds.family == fam
        r, rp = res["rmse"][sel].mean(axis=(0, 2)), res["rmse_persist"][sel].mean(axis=(0, 2))
        print(f"{fam:<10}" + "".join(f"{r[t - 1]:>11.4f} ({rp[t - 1]:.3f})" for t in (1, 10, 25, 50)))
    # Mean is dominated by any trajectory that blows up, so also report the median and
    # how many trajectories blew up (error > 2, i.e. twice the typical variation of the whole dataset)
    e = res["rmse"].mean(axis=2)                                                   # (N, T)
    print(f"{'median':<10}" + "".join(f"{np.median(e[:, t - 1]):>20.4f}" for t in (1, 10, 25, 50)))
    blown = ~(e[:, -1] <= 2)                                                       # NaN counts as blown
    print(f"\nblown up by t=50 (error > 2): {blown.sum()}/{len(e)}  ("
          + ", ".join(f"{fam} {blown[ds.family == fam].sum()}/{(ds.family == fam).sum()}" for fam in fams) + ")")
    drift = np.abs(res["drift"][:, -1]).max(axis=0)
    print("\nmax |relative drift| at t=50:  " + "  ".join(f"{v} {d:.1e}" for v, d in zip(VARS, drift)))
    nbad = res["bad_cells"]
    print(f"trajectories with any negative rho/p or NaN: {(nbad.sum(axis=1) > 0).sum()} / {len(nbad)}"
          + (f"  (first at step {np.argmax(nbad.sum(axis=0) > 0) + 1})" if nbad.any() else ""))

    # Figure 1: error growth
    Path("outputs").mkdir(exist_ok=True)
    t = np.arange(1, steps + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for fam, c in zip(fams, plt.cm.tab10.colors):
        sel = ds.family == fam
        axes[0].semilogy(t, res["rmse"][sel].mean(axis=(0, 2)), color=c, label=f"{fam} model")
        axes[0].semilogy(t, res["rmse_persist"][sel].mean(axis=(0, 2)), "--", color=c, label=f"{fam} persistence")
        axes[2].plot(t, np.abs(res["drift"][sel]).max(axis=(0, 2)), color=c, label=fam)
    allr = res["rmse"].mean(axis=0)                                                # (T, 4)
    for v in range(4):
        axes[1].semilogy(t, allr[:, v], label=VARS[v])
    axes[0].set(title="Rollout error (mean over trajectories)", ylabel="RMSE / train std")
    axes[1].set(title="Per variable (all families)", ylabel="RMSE / train std")
    axes[2].set(title="Conservation: max |relative drift|", ylabel="relative drift")
    axes[2].set_yscale("symlog", linthresh=1e-6)
    for ax in axes:
        ax.set_xlabel("rollout step (dt = 0.005)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle(f"{run.name} on {args.split}")
    fig.tight_layout()
    fig.savefig(f"outputs/rollout_{run.name}_{args.split}.png", dpi=130)

    # Figure 2: density snapshots, truth vs prediction vs error, for one trajectory per family
    frames = [10, 25, 50]
    fig, axes = plt.subplots(3 * len(examples), len(frames), figsize=(3.2 * len(frames), 3.0 * 3 * len(examples)))
    for i, (fam, (k, true, pred)) in enumerate(examples.items()):
        lo, hi = true[:, 0].min(), true[:, 0].max()
        for j, f in enumerate(frames):
            err = pred[f - 1, 0] - true[f - 1, 0]
            e = np.abs(err).max() or 1
            for r, (img, kw, label) in enumerate([(true[f - 1, 0], dict(vmin=lo, vmax=hi, cmap="viridis"), "true"),
                                                  (pred[f - 1, 0], dict(vmin=lo, vmax=hi, cmap="viridis"), "model"),
                                                  (err, dict(vmin=-e, vmax=e, cmap="RdBu_r"), "error")]):
                ax = axes[3 * i + r, j]
                im = ax.imshow(img, origin="lower", **kw)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{fam} #{ds.index[k]} {label}  t={f}", fontsize=8)
                if r == 2:
                    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(f"outputs/rollout_{run.name}_{args.split}_snapshots.png", dpi=110)
    print(f"\nsaved outputs/rollout_{run.name}_{args.split}.png and ..._snapshots.png; metrics in {run}/rollout_{args.split}.npz")


if __name__ == "__main__":
    main()