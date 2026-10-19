"""Data loading for training surrogates: (state at frame t) -> (states at frames t+1 .. t+horizon).

Reads packed files written by data/pack.py. Framework-agnostic: returns NumPy arrays, which the
training code moves to the accelerator (jax.device_put). Everything is normalised with statistics from
the TRAIN split only (data/packed/stats_<n>.json).

  ds = PairDataset("data/packed", split="train", n=64, horizon=1)
  for x, y in ds.batches(batch_size=32, rng=np.random.default_rng(0)):
      ...   # x: (B, 4, n, n), y: (B, horizon, 4, n, n), both normalised float32
"""
import json
from pathlib import Path

import h5py
import numpy as np


class Normaliser:
    """Per-variable affine normalisation of conserved variables: (U - mean) / std.

    Uses train-split statistics only. Works on any array whose variable axis is `axis`.
    """

    def __init__(self, stats_path):
        s = json.loads(Path(stats_path).read_text())
        self.mean = np.asarray(s["mean"], dtype=np.float32)
        self.std = np.asarray(s["std"], dtype=np.float32)
        self.delta_std = np.asarray(s["delta_std"], dtype=np.float32)   

    def _shape(self, v, ndim, axis):
        shape = [1] * ndim
        shape[axis] = 4
        return v.reshape(shape)

    def encode(self, U, axis=-3):
        return (U - self._shape(self.mean, U.ndim, axis)) / self._shape(self.std, U.ndim, axis)

    def decode(self, Z, axis=-3):
        return Z * self._shape(self.std, Z.ndim, axis) + self._shape(self.mean, Z.ndim, axis)


class PairDataset:
    """All (trajectory, start frame) pairs of one split, at one resolution.

    horizon = number of future frames returned as targets (1 for one-step training; more for
    multi-step losses or the pushforward trick). A pair never crosses from one trajectory into the next.

    in_memory=True loads the whole split into RAM (fast; 64^2 train is ~2 GB). in_memory=False reads
    each batch from disk (slower, but works for 128^2 on a laptop).
    """

    def __init__(self, packed_dir, split, n, horizon=1, in_memory=True):
        packed_dir = Path(packed_dir)
        self.path = packed_dir / f"{split}_{n}.h5"
        self.split, self.n, self.horizon, self.in_memory = split, n, horizon, in_memory
        self.norm = Normaliser(packed_dir / f"stats_{n}.json")
        with h5py.File(self.path, "r") as f:
            self.n_traj, self.n_frames = f["U"].shape[:2]
            self.family = f["family"][:].astype(str)
            self.index = f["index"][:]
            self.U = f["U"][:] if in_memory else None
        if horizon < 1 or horizon >= self.n_frames:
            raise ValueError(f"horizon must be between 1 and {self.n_frames - 1}")
        # Every valid (trajectory, start frame): the start frame needs `horizon` frames after it
        starts = np.arange(self.n_frames - horizon)
        self.pairs = np.stack(np.meshgrid(np.arange(self.n_traj), starts, indexing="ij"), -1).reshape(-1, 2)

    def __len__(self):
        return len(self.pairs)

    def _windows(self, pairs):
        """Raw (unnormalised) frames t .. t+horizon for each (trajectory, t), shape (B, 1+horizon, 4, n, n)."""
        h = self.horizon
        if self.in_memory:
            return np.stack([self.U[k, t:t + h + 1] for k, t in pairs])
        with h5py.File(self.path, "r") as f:
            U = f["U"]
            return np.stack([U[k, t:t + h + 1] for k, t in pairs])

    def get(self, pairs):
        """Normalised (x, y) for an array of (trajectory, start frame) pairs."""
        w = self.norm.encode(self._windows(np.asarray(pairs)).astype(np.float32))
        return w[:, 0], w[:, 1:]

    def batches(self, batch_size, rng=None, shuffle=True, drop_last=True):
        """One pass over all pairs, in random order if shuffle (pass an rng for reproducibility)."""
        order = np.arange(len(self.pairs))
        if shuffle:
            (rng or np.random.default_rng()).shuffle(order)
        stop = len(order) - (len(order) % batch_size if drop_last else 0)
        for i in range(0, stop, batch_size):
            yield self.get(self.pairs[order[i:i + batch_size]])

    def trajectory(self, k, normalised=True):
        """A whole trajectory, shape (frames, 4, n, n): for rollouts and evaluation."""
        if self.in_memory:
            U = self.U[k]
        else:
            with h5py.File(self.path, "r") as f:
                U = f["U"][k]
        U = U.astype(np.float32)
        return self.norm.encode(U) if normalised else U