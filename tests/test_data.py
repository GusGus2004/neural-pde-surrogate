"""Checks for surrogate/data.py. Run from the repo root: python tests/test_data.py [packed_dir]"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from surrogate.data import PairDataset   # noqa: E402

packed = sys.argv[1] if len(sys.argv) > 1 else "data/packed"
failures = 0


def check(name, ok, value=""):
    global failures
    print(f"{name:<62} {'PASS' if ok else 'FAIL'}  {value}")
    failures += not ok


ds = PairDataset(packed, "train", 64, horizon=1)
F = ds.n_frames
check("one pair per (trajectory, frame 0..49)", len(ds) == ds.n_traj * (F - 1), f"{len(ds)} pairs")

# x and y line up: y of (k, t) is x of (k, t+1), and pairs never cross trajectories
x0, y0 = ds.get([(0, 10)])
x1, _ = ds.get([(0, 11)])
check("target of (k, t) == input of (k, t+1)", np.array_equal(y0[0, 0], x1[0]))
check("no pair starts at the last frame", ds.pairs[:, 1].max() == F - 2)

# Normalisation round trip and train-set statistics
raw = ds.trajectory(3, normalised=False)
check("encode/decode round trip", np.allclose(ds.norm.decode(ds.norm.encode(raw)), raw, rtol=1e-5, atol=1e-5))
xs = np.concatenate([x for x, _ in ds.batches(64, rng=np.random.default_rng(0))])
m, s = xs.mean(axis=(0, 2, 3)), xs.std(axis=(0, 2, 3))
check("normalised train inputs: mean ~ 0 per variable", np.abs(m).max() < 0.05, np.round(m, 3))
check("normalised train inputs: std ~ 1 per variable", np.abs(s - 1).max() < 0.05, np.round(s, 3))

# Shapes, batching, reproducibility
x, y = next(ds.batches(8, rng=np.random.default_rng(1)))
check("batch shapes (8,4,64,64) and (8,1,4,64,64)", x.shape == (8, 4, 64, 64) and y.shape == (8, 1, 4, 64, 64))
check("float32", x.dtype == np.float32 and y.dtype == np.float32)
xa, _ = next(ds.batches(8, rng=np.random.default_rng(1)))
check("same rng seed -> same batch", np.array_equal(x, xa))

# Multi-step targets
ds3 = PairDataset(packed, "train", 64, horizon=3)
x, y = ds3.get([(2, 5)])
traj = ds3.trajectory(2)
check("horizon 3: targets are frames t+1..t+3", np.array_equal(y[0], traj[6:9]) and np.array_equal(x[0], traj[5]))
check("horizon 3: last start frame is 47", ds3.pairs[:, 1].max() == F - 4)

# Disk mode gives identical batches
dsd = PairDataset(packed, "train", 64, horizon=1, in_memory=False)
xd, yd = dsd.get(ds.pairs[[0, 7, 123 % len(ds)]])
xm, ym = ds.get(ds.pairs[[0, 7, 123 % len(ds)]])
check("disk mode == memory mode", np.array_equal(xd, xm) and np.array_equal(yd, ym))

# Throughput (rough)
t0 = time.time(); nb = 0
for _ in ds.batches(32, rng=np.random.default_rng(2)):
    nb += 1
    if nb == 50: break
mem_rate = 50 * 32 / (time.time() - t0)
t0 = time.time(); nb = 0
for _ in dsd.batches(32, rng=np.random.default_rng(2)):
    nb += 1
    if nb == 10: break
disk_rate = 10 * 32 / (time.time() - t0)
print(f"\nthroughput: in memory {mem_rate:,.0f} pairs/s, from disk {disk_rate:,.0f} pairs/s")

print(f"\n{'FAILED' if failures else 'ALL PASSED'} ({failures} failure{'s' if failures != 1 else ''})")
sys.exit(1 if failures else 0)