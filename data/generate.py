"""Generate dataset trajectories: (family, index) -> one finished trajectory file.

  python data/generate.py --family riemann --indices 0-7            # local batch
  python data/generate.py --family blast --indices 42               # a single run
  python data/generate.py --family kh --indices 0-3 --n 256         # cheaper for testing

Each run: sample the initial condition -> run euler2d at --n with --save sizes -> add the sampled
parameters and provenance (git hash, time, machine) to the file -> move it into place.

Output: <out>/<family>/<index>.h5 (e.g. data/raw/riemann/000017.h5).File only appears once the run
has completed, so existing files are skipped and interrupted batch can simply be restarted.
"""
import argparse
import datetime
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from families import sample  
from ic import write_ic       

EXE = REPO / "solver" / "build" / "euler2d"
FRAMES = 50                   


def parse_indices(text):
    """'0-7' -> [0..7], '3' -> [3], '0-3,10,12-13' -> [0, 1, 2, 3, 10, 12, 13]."""
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def git_hash():
    """Commit of the code that produced the data; '-dirty' if there were uncommitted changes."""
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                           text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def generate_one(family, index, n, save, out_dir, threads, force=False):
    """Run one trajectory. Returns 'done', 'skipped' or 'failed'."""
    final = out_dir / family / f"{index:06d}.h5"
    if final.exists() and not force:
        return "skipped"
    final.parent.mkdir(parents=True, exist_ok=True)

    rho, u, v, p, t_end, params = sample(family, index, n)
    t_start = params.get("t_start", 0.0)

    with tempfile.TemporaryDirectory(dir=final.parent) as tmp:
        ic_path, traj_path = Path(tmp) / "ic.h5", Path(tmp) / "traj.h5"
        write_ic(ic_path, rho, u, v, p, t_end=t_end, test=family)
        env = dict(os.environ, OMP_NUM_THREADS=str(threads))
        cmd = [str(EXE), "--init", str(ic_path), "--nout", str(FRAMES), "--tstart", repr(t_start),
               "--traj", str(traj_path), "--save", save]
        t0 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
        secs = time.time() - t0

        if res.returncode != 0:
            log = final.with_suffix(".failed.txt")
            log.write_text(f"command: {' '.join(cmd)}\n\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}")
            print(f"  {family} {index:6d}: FAILED after {secs:.0f} s (details in {log.name})")
            return "failed"

        # Add sampled params to the solver's output
        with h5py.File(traj_path, "r+") as f:
            f.attrs["family"] = family
            f.attrs["index"] = index
            for key, val in params.items():
                if key not in ("family", "index"):
                    f.attrs[f"param_{key}"] = val
            f.attrs["git_hash"] = git_hash()
            f.attrs["generated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
            f.attrs["host"] = platform.node()
            f.attrs["runtime_s"] = secs
            f.attrs["threads"] = threads
            steps = int(f.attrs["steps"])
        os.replace(traj_path, final)          # file appears only once complete
        old_log = final.with_suffix(".failed.txt")
        if old_log.exists():
            old_log.unlink()                 

    print(f"  {family} {index:6d}: {steps:6d} steps, {secs:6.1f} s -> {final}")
    return "done"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True)
    ap.add_argument("--indices", required=True, help="e.g. 0-99, or 5, or 0-9,20-29")
    ap.add_argument("--n", type=int, default=512, help="solver resolution (default 512)")
    ap.add_argument("--save", default="128,64", help="stored resolutions (default 128,64)")
    ap.add_argument("--out", default=str(REPO / "data" / "raw"))
    ap.add_argument("--threads", type=int, default=4, help="OpenMP threads per run (default 4)")
    ap.add_argument("--force", action="store_true", help="regenerate files that already exist")
    args = ap.parse_args()

    if not EXE.exists():
        sys.exit(f"Solver not found at {EXE}: build it first (cmake --build solver/build)")
    indices = parse_indices(args.indices)
    print(f"{args.family}: {len(indices)} run(s) at {args.n}^2, saving {args.save}, {args.threads} thread(s)")
    counts = {"done": 0, "skipped": 0, "failed": 0}
    t0 = time.time()
    for i in indices:
        counts[generate_one(args.family, i, args.n, args.save, Path(args.out), args.threads, args.force)] += 1
    print(f"Finished in {time.time() - t0:.0f} s: {counts['done']} done, {counts['skipped']} skipped, "
          f"{counts['failed']} failed")
    sys.exit(1 if counts["failed"] else 0)


if __name__ == "__main__":
    main()