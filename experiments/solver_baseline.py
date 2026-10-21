"""The numerical solver as a baseline model

  python experiments/solver_baseline.py                    # val split, solver at 32^2, 64^2, 128^2
  python experiments/solver_baseline.py --sizes 64 --split test

For every trajectory it takes true frame 0, runs euler2d from it
up to t = 0.25 with the same 50 output frames, and scores the result like surrogate/rollout.py,
against the same ground truth (solved at 512^2, block-averaged to 64^2). Same scheme as the dataset
(HLLC + van Leer, CFL 0.5), so the only difference from the ground truth is the resolution.

Initial conditions: 128^2 runs start from the stored 128^2 frame 0; 64^2 runs from the 64^2 frame 0;
32^2 runs from the 64^2 frame 0 block-averaged to 32^2. Results are compared on the 64^2 grid: 128^2
output is block-averaged to 64^2 by the solver itself; 32^2 output is copied into 2x2 blocks (each 32^2
cell value is the average over its area, so this is the natural piecewise-constant reading).

"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.ic import write_ic                            # noqa: E402
from surrogate.data import PairDataset                  # noqa: E402
from surrogate.rollout import GAMMA, metrics            # noqa: E402

REPO = Path(__file__).resolve().parent.parent
EXE = REPO / "solver" / "build" / "euler2d"


def block_average(U, f):
    """(..., n, n) -> (..., n/f, n/f) by averaging f x f blocks"""
    *lead, n, m = U.shape
    return U.reshape(*lead, n // f, f, m // f, f).mean(axis=(-3, -1))


def to_primitive(U):
    """Conserved (4, n, n) -> rho, u, v, p (float64)"""
    U = U.astype(np.float64)
    rho = U[0]
    u, v = U[1] / rho, U[2] / rho
    p = (GAMMA - 1) * (U[3] - 0.5 * rho * (u ** 2 + v ** 2))
    return rho, u, v, p


def run_solver(U0, t_end, frames, save):
    """Run euler2d from conserved state U0 (4, n, n). Returns (trajectory (frames+1, 4, s, s), solver seconds,
    wall seconds, steps), where s = save."""
    with tempfile.TemporaryDirectory() as tmp:
        ic, traj = Path(tmp) / "ic.h5", Path(tmp) / "traj.h5"
        write_ic(ic, *to_primitive(U0), t_end=t_end, test="solver_baseline")
        cmd = [str(EXE), "--init", str(ic), "--nout", str(frames), "--traj", str(traj), "--save", str(save)]
        t0 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="1"))
        wall = time.time() - t0
        if res.returncode != 0:
            raise RuntimeError(f"euler2d failed:\n{res.stdout}\n{res.stderr}")
        m = re.search(r"Done: (\d+) steps in ([\d.]+) s", res.stdout)
        with h5py.File(traj, "r") as f:
            return f[f"U_{save}"][:], float(m.group(2)), wall, int(m.group(1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="val")
    p.add_argument("--sizes", default="32,64,128", help="solver resolutions")
    p.add_argument("--data", default="data/packed")
    p.add_argument("--runs", default="runs")
    args = p.parse_args()
    if not EXE.exists():
        sys.exit(f"{EXE} not found: build the solver first")

    truth_ds = PairDataset(args.data, args.split, 64, horizon=1)
    fine = None
    norm, frames = truth_ds.norm, truth_ds.n_frames - 1

    for size in (int(s) for s in args.sizes.split(",")):
        if size == 128 and fine is None:
            fine = PairDataset(args.data, args.split, 128, horizon=1, in_memory=False)
        out = {k: [] for k in ("rmse", "drift", "bad_cells", "rmse_persist")}
        solver_s, wall_s, steps_all = [], [], []
        print(f"solver at {size}^2 on {truth_ds.n_traj} {args.split} trajectories", flush=True)
        for k in range(truth_ds.n_traj):
            true = truth_ds.trajectory(k, normalised=False)                   # (51, 4, 64, 64), from 512^2
            if size == 128:
                U0, save = fine.trajectory(k, normalised=False)[0], 64       # solver block-averages to 64
            elif size == 64:
                U0, save = true[0], 64
            elif size == 32:
                U0, save = block_average(true[0], 2), 32
            else:
                raise SystemExit("sizes must be 32, 64 or 128")
            t_end = 0.25                                                      # all val/test families: 0..0.25
            traj, s_solver, s_wall, steps = run_solver(U0, t_end, frames, save)
            if save == 32:
                traj = np.repeat(np.repeat(traj, 2, axis=-2), 2, axis=-1)     # piecewise constant on 64^2
            pred, tr = traj[1:].astype(np.float32), true[1:]
            m = metrics(pred, tr, norm.std)
            for key in ("rmse", "drift", "bad_cells"):
                out[key].append(m[key])
            out["rmse_persist"].append(metrics(np.broadcast_to(true[:1], tr.shape), tr, norm.std)["rmse"])
            solver_s.append(s_solver)
            wall_s.append(s_wall)
            steps_all.append(steps)
            if (k + 1) % 16 == 0 or k + 1 == truth_ds.n_traj:
                print(f"  {k + 1}/{truth_ds.n_traj}", flush=True)

        res = {k: np.stack(v) for k, v in out.items()}
        run_dir = Path(args.runs) / f"solver_{size}"
        run_dir.mkdir(parents=True, exist_ok=True)
        np.savez(run_dir / f"rollout_{args.split}.npz", family=truth_ds.family, index=truth_ds.index,
                 solver_seconds=np.array(solver_s), wall_seconds=np.array(wall_s), steps=np.array(steps_all), **res)
        e = res["rmse"].mean(axis=2)
        print(f"  median error  t=1 {np.median(e[:, 0]):.4f}  t=10 {np.median(e[:, 9]):.4f}  "
              f"t=25 {np.median(e[:, 24]):.4f}  t=50 {np.median(e[:, -1]):.4f}   (mean t=50 {e[:, -1].mean():.4f})")
        print(f"  time per trajectory (1 thread): solver {np.mean(solver_s):.3f} s, incl. process + I/O "
              f"{np.mean(wall_s):.3f} s; {np.mean(steps_all):.0f} time steps")
        print(f"  saved {run_dir}/rollout_{args.split}.npz\n", flush=True)


if __name__ == "__main__":
    main()