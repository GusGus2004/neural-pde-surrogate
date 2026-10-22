"""Define a sweep of training runs and turn it into a SLURM job array (or run it locally).

  python cluster/sweep.py smoke --local        # 2 tiny jobs, run here, one after the other (a pipeline test)
  python cluster/sweep.py seeds_small          # write cluster/jobs/seeds_small.txt and print the sbatch command
  python cluster/sweep.py scale --submit       # ... and submit it

Every job is one line: environment settings + `bash cluster/recipe.sh NAME model-options`. Jobs whose final
output (runs/<name>_pf3/rollout_val.npz) already exists are left out, so re-running this after some jobs
failed or timed out only resubmits what is missing.
"""
import argparse
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    # the small, Mac-sized models used so far
    "resnet_small": "--model resnet --width 32 --depth 4",
    "unet_small": "--model unet --width 16 --levels 3",
    "fno_small": "--model fno --width 32 --depth 4 --modes 16",
    # larger models for GPUs
    "resnet_large": "--model resnet --width 64 --depth 8",
    "unet_large": "--model unet --width 32 --levels 3",
    "fno_large": "--model fno --width 64 --depth 4 --modes 16",
}

SWEEPS = {
    # Pipeline test: tiny model, a few steps. Run with --local, on the cluster in an interactive GPU session.
    "smoke": dict(models=["resnet_small", "fno_small"], seeds=[0],
                  env=dict(STEPS=10, PF_STEPS=5, EVAL_EVERY=5, WARMUP=2)),
    # The Mac results again, with 3 seeds each: how much of the differences seen so far is seed noise?
    "seeds_small": dict(models=["resnet_small", "unet_small", "fno_small"], seeds=[0, 1, 2],
                        env=dict(STEPS=2000, PF_STEPS=600, EVAL_EVERY=200, WARMUP=100)),
    # Bigger models, 10x longer training: how far does scale close the ~3x gap to the 64^2 solver?
    "scale": dict(models=["resnet_large", "unet_large", "fno_large"], seeds=[0, 1, 2],
                  env=dict(STEPS=20000, PF_STEPS=3000, EVAL_EVERY=1000, WARMUP=500)),
}


def jobs_for(sweep, runs, include_done, data=None):
    spec = SWEEPS[sweep]
    extra = f"DATA={shlex.quote(data)} " if data else ""
    lines, skipped = [], []
    for model in spec["models"]:
        for seed in spec["seeds"]:
            name = f"{sweep}_{model}_s{seed}"
            if not include_done and (runs / f"{name}_pf3" / "rollout_val.npz").exists():
                skipped.append(name)
                continue
            env = " ".join(f"{k}={v}" for k, v in spec["env"].items())
            lines.append(f"{env} {extra}RUNS={shlex.quote(str(runs))} bash cluster/recipe.sh {name} "
                         f"{MODELS[model]} --seed {seed}")
    return lines, skipped


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep", choices=sorted(SWEEPS))
    p.add_argument("--runs", default="runs", help="where run folders go (relative to the repo root)")
    p.add_argument("--data", default=None, help="packed data folder (default: data/packed)")
    p.add_argument("--max-parallel", type=int, default=4, help="at most this many array jobs at once")
    p.add_argument("--local", action="store_true", help="run the jobs here, one after the other")
    p.add_argument("--submit", action="store_true", help="submit the job array with sbatch")
    p.add_argument("--all", action="store_true", help="include jobs that already finished")
    args = p.parse_args()

    lines, skipped = jobs_for(args.sweep, Path(args.runs), args.all, args.data)
    if skipped:
        print(f"already finished, left out: {', '.join(skipped)}")
    if not lines:
        print("nothing to do")
        return
    joblist = ROOT / "cluster" / "jobs" / f"{args.sweep}.txt"
    joblist.parent.mkdir(exist_ok=True)
    joblist.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} job(s) written to {joblist.relative_to(ROOT)}:")
    for line in lines:
        print("  " + line)

    if args.local:
        for i, line in enumerate(lines):
            print(f"\n##### local job {i + 1}/{len(lines)}")
            subprocess.run(line, shell=True, check=True, cwd=ROOT)
        return
    (ROOT / "logs").mkdir(exist_ok=True)
    cmd = ["sbatch", f"--array=0-{len(lines) - 1}%{args.max_parallel}", "cluster/array.sbatch",
           str(joblist.relative_to(ROOT))]
    print("\nsubmit with:\n  " + " ".join(cmd))
    if args.submit:
        subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
