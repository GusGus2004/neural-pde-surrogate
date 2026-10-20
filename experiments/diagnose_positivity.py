"""Where do rollouts go non-physical, and does that predict which trajectories blow up?

  python experiments/diagnose_positivity.py runs/ft_pf3 --which last

Rolls out every validation trajectory (same as surrogate/rollout.py) and, per trajectory, records:
  - blown: mean RMSE/std at t=50 > 2 (same definition as rollout.py)
  - first step with a non-physical cell (rho <= 0, p <= 0 or NaN), and how many bad cells at steps 1, 5, 10
  - the smallest TRUE density and pressure anywhere in the trajectory (is there near-vacuum?)
  - at the bad cells in steps 1-5: the TRUE density and pressure there (are the bad cells the near-vacuum ones?),
    whether it was the density or the pressure that went negative, and the TRUE kinetic fraction of the energy
    there. p = (gamma-1) (E - kinetic), so where kinetic ~ E a small error in E or momentum makes p negative.
"""
import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from surrogate.data import PairDataset               
from surrogate.rollout import pressure, rollout       
from surrogate.train import load_model                

p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("--which", default="last")
p.add_argument("--split", default="val")
p.add_argument("--data", default="data/packed")
args = p.parse_args()

model, cfg = load_model(args.run, args.which)
ds = PairDataset(args.data, args.split, cfg["n"], horizon=1)
norm, steps = ds.norm, ds.n_frames - 1

rows = []
true_rho_at_bad, true_p_at_bad, true_p_all = [], [], []
kin_at_bad, kin_all, n_rho_bad, n_p_bad = [], [], 0, 0
for start in range(0, ds.n_traj, 16):
    ks = range(start, min(start + 16, ds.n_traj))
    true_n = np.stack([ds.trajectory(k) for k in ks])
    pred = norm.decode(np.asarray(rollout(model, jnp.asarray(true_n[:, 0]), steps)))   
    true = norm.decode(true_n[:, 1:])
    for b, k in enumerate(ks):
        rho_p, p_p = pred[b, :, 0], pressure(pred[b])
        rho_t, p_t = true[b, :, 0], pressure(true[b])
        with np.errstate(invalid="ignore"):
            bad = ~(rho_p > 0) | ~(p_p > 0)                                     
        nbad = bad.sum(axis=(1, 2))
        err = np.sqrt(np.mean((pred[b] - true[b]) ** 2, axis=(-2, -1))) / norm.std 
        early = bad[:5]
        with np.errstate(invalid="ignore"):
            n_rho_bad += int((~(rho_p[:5] > 0)).sum())
            n_p_bad += int((~(p_p[:5] > 0) & (rho_p[:5] > 0)).sum())
        kin = 1 - p_t[:5] / (0.4 * true[b, :5, 3])                                   
        kin_at_bad.append(kin[early])
        kin_all.append(kin.ravel())
        true_rho_at_bad.append(rho_t[:5][early])
        true_p_at_bad.append(p_t[:5][early])
        true_p_all.append(p_t[:5].ravel())
        rows.append(dict(k=k, family=ds.family[k], index=ds.index[k],
                         blown=not (err[-1].mean() <= 2), err50=err[-1].mean(),
                         first_bad=int(np.argmax(nbad > 0)) + 1 if nbad.any() else None,
                         bad1=int(nbad[0]), bad5=int(nbad[4]), bad10=int(nbad[9]),
                         min_rho=float(rho_t.min()), min_p=float(p_t.min())))
    print(f"  {ks[-1] + 1}/{ds.n_traj}", flush=True)

blown = np.array([r["blown"] for r in rows])
print(f"\n{blown.sum()} blown up, {(~blown).sum()} stable\n")
print(f"{'':28}{'blown':>12}{'stable':>12}")


def compare(label, key, fn=np.nanmedian):
    vals = np.array([np.nan if r[key] is None else r[key] for r in rows], float)
    pick = lambda sel: fn(vals[sel]) if np.isfinite(vals[sel]).any() else np.nan
    print(f"{label:28}{pick(blown):>12.4g}{pick(~blown):>12.4g}")


compare("median bad cells at step 1", "bad1")
compare("median bad cells at step 5", "bad5")
compare("median bad cells at step 10", "bad10")
compare("median first bad step", "first_bad")
compare("median true min density", "min_rho")
compare("median true min pressure", "min_p")
never = [r for r in rows if r["first_bad"] is None]
print(f"\ntrajectories that never go non-physical: {len(never)} "
      f"({sum(r['blown'] for r in never)} of them blow up)")

rb, pb, pall = np.concatenate(true_rho_at_bad), np.concatenate(true_p_at_bad), np.concatenate(true_p_all)
if len(pb):
    print(f"\nat non-physical cells in steps 1-5 ({len(pb):,} cell-steps), the TRUE values were:")
    print(f"  density:  median {np.median(rb):.4f}   90th pct {np.percentile(rb, 90):.4f}")
    print(f"  pressure: median {np.median(pb):.4f}   90th pct {np.percentile(pb, 90):.4f}")
    print(f"  (pressure over ALL cells, steps 1-5: median {np.median(pall):.4f}, "
          f"1st pct {np.percentile(pall, 1):.4f})")
    for thr in (0.01, 0.05, 0.1):
        print(f"  fraction of bad cells with true p < {thr}: {(pb < thr).mean():.0%}   "
              f"(of all cells: {(pall < thr).mean():.2%})")
    kb, ka = np.concatenate(kin_at_bad), np.concatenate(kin_all)
    print(f"  what went negative: density {n_rho_bad:,} cell-steps, pressure only {n_p_bad:,}")
    print(f"  kinetic share of true energy: median {np.median(kb):.2f} at bad cells, {np.median(ka):.2f} over all cells")

def ranks(a):
    return np.argsort(np.argsort(a))


for key in ("bad5", "min_p"):
    a = np.array([r[key] for r in rows], float)
    e = np.array([r["err50"] for r in rows])
    print(f"Spearman correlation of {key:5} with error at t=50: {np.corrcoef(ranks(a), ranks(e))[0, 1]:+.2f}")

print("\nblown-up trajectories:")
print(f"{'family':>8}{'index':>7}{'err t50':>10}{'1st bad':>9}{'bad@5':>7}{'true min rho':>14}{'true min p':>12}")
for r in sorted((r for r in rows if r["blown"]), key=lambda r: r["first_bad"] or 99):
    print(f"{r['family']:>8}{r['index']:>7}{r['err50']:>10.2f}{str(r['first_bad']):>9}{r['bad5']:>7}"
          f"{r['min_rho']:>14.4f}{r['min_p']:>12.4f}")