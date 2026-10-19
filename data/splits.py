"""Assign every raw trajectory to split and check, writing data/packed/manifest.json.

  python data/splits.py                     # reads data/raw, writes data/packed/manifest.json

Splits by whole trajectory and depend only on (family, index):
  - training families (riemann, blast): index % 10 in 0-7 -> train, 8 -> val, 9 -> test
  - held-out families (kh):             everything -> ood
Assignment never changes when more indices are generated later, and exact 80/10/10 split
holds for any range of indices.

Every file is also checked (frames, shapes, frame spacing, stored resolutions) so problems are
caught here rather than halfway through training.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parent.parent
TRAINING_FAMILIES = {"riemann", "blast"}
HELD_OUT_FAMILIES = {"kh"}
EXPECTED_FRAMES = 51
EXPECTED_DT = 0.005


def split_for(family, index):
    if family in HELD_OUT_FAMILIES:
        return "ood"
    if family not in TRAINING_FAMILIES:
        raise ValueError(f"family '{family}' is neither a training nor a held-out family")
    r = index % 10
    return "train" if r < 8 else ("val" if r == 8 else "test")


def as_str(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def inspect(path):
    """Read the metadata of one trajectory and return (entry, list of problems)."""
    problems = []
    with h5py.File(path, "r") as f:
        a = f.attrs
        t = f["t"][:]
        sizes = sorted((int(k[2:]) for k in f if k.startswith("U_")), reverse=True)
        for n in sizes:
            shape = f[f"U_{n}"].shape
            if shape != (len(t), 4, n, n):
                problems.append(f"U_{n} has shape {shape}")
        entry = {
            "family": as_str(a["family"]),
            "index": int(a["index"]),
            "path": str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path),
            "sizes": sizes,
            "t_start": float(t[0]),
            "solver_n": int(a["solver_nx"]),
            "steps": int(a["steps"]),
            "runtime_s": round(float(a["runtime_s"]), 1),
            "git_hash": as_str(a["git_hash"]),
        }
    if len(t) != EXPECTED_FRAMES:
        problems.append(f"{len(t)} frames, expected {EXPECTED_FRAMES}")
    if not np.allclose(np.diff(t), EXPECTED_DT, rtol=0, atol=1e-9):
        problems.append(f"frame spacing {np.diff(t).min():.6f}..{np.diff(t).max():.6f}, expected {EXPECTED_DT}")
    return entry, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(REPO / "data" / "raw"))
    ap.add_argument("--out", default=str(REPO / "data" / "packed"))
    args = ap.parse_args()

    files = sorted(Path(args.raw).glob("*/*.h5"))
    if not files:
        raise SystemExit(f"No trajectory files found under {args.raw}")

    entries, bad = [], []
    for path in files:
        entry, problems = inspect(path)
        if problems:
            bad.append((path, problems))
            continue
        entry["split"] = split_for(entry["family"], entry["index"])
        entries.append(entry)

    # Summary
    counts = defaultdict(Counter)
    for e in entries:
        counts[e["family"]][e["split"]] += 1
    splits = ["train", "val", "test", "ood"]
    print(f"{'family':<10}" + "".join(f"{s:>8}" for s in splits) + f"{'total':>8}")
    for fam in sorted(counts):
        row = counts[fam]
        print(f"{fam:<10}" + "".join(f"{row[s]:>8}" for s in splits) + f"{sum(row.values()):>8}")
    totals = Counter(e["split"] for e in entries)
    print(f"{'all':<10}" + "".join(f"{totals[s]:>8}" for s in splits) + f"{len(entries):>8}")

    sizes = {tuple(e["sizes"]) for e in entries}
    solver_ns = {e["solver_n"] for e in entries}
    hashes = Counter(e["git_hash"] for e in entries)
    print(f"\nstored resolutions: {sorted(sizes)}   solver resolution: {sorted(solver_ns)}")
    print("git commits: " + ", ".join(f"{h} ({c})" for h, c in hashes.most_common()))
    warnings = []
    if len(sizes) > 1:
        warnings.append("trajectories have different stored resolutions")
    if len(solver_ns) > 1:
        warnings.append("trajectories were solved at different resolutions")
    if any(h.endswith("-dirty") or h == "unknown" for h in hashes):
        warnings.append("some files come from uncommitted ('-dirty') or unknown code")
    for w in warnings:
        print(f"WARNING: {w}")
    for path, problems in bad:
        print(f"SKIPPED {path}: {'; '.join(problems)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"split_rule": "training families: index % 10 -> 0-7 train, 8 val, 9 test; held-out: ood",
                "trajectories": entries}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nWrote {out / 'manifest.json'} ({len(entries)} trajectories, {len(bad)} skipped)")


if __name__ == "__main__":
    main()
