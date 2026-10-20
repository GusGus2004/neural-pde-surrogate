"""Surrogate models (Equinox). Each model maps one normalised state (4, n, n) to the next state (4, n, n).

Batching is done outside with jax.vmap: the model code only
thinks about a single sample, and vmap adds the batch axis.

All convolutions use PERIODIC padding, matching the periodic boundaries of the dataset. A model built
this way is exactly translation-equivariant on the torus: shifting the input shifts the output.
"""
import equinox as eqx
import jax
import jax.numpy as jnp

import numpy as np


class PeriodicConv(eqx.Module):
    """k x k convolution with wrap-around padding, so the domain is a torus."""
    conv: eqx.nn.Conv2d
    pad: int = eqx.field(static=True)

    def __init__(self, c_in, c_out, kernel_size, *, key):
        self.conv = eqx.nn.Conv2d(c_in, c_out, kernel_size, padding=0, key=key)
        self.pad = kernel_size // 2

    def __call__(self, x):                       # x: (c_in, n, n)
        p = self.pad
        if p:
            x = jnp.pad(x, ((0, 0), (p, p), (p, p)), mode="wrap")
        return self.conv(x)


class ResBlock(eqx.Module):
    """x + conv(gelu(conv(gelu(x)))): a pre-activation residual block."""
    conv1: PeriodicConv
    conv2: PeriodicConv

    def __init__(self, width, kernel_size, *, key):
        k1, k2 = jax.random.split(key)
        self.conv1 = PeriodicConv(width, width, kernel_size, key=k1)
        self.conv2 = PeriodicConv(width, width, kernel_size, key=k2)

    def __call__(self, x):
        h = self.conv1(jax.nn.gelu(x))
        h = self.conv2(jax.nn.gelu(h))
        return x + h


class ResNet(eqx.Module):
    """Lift (1x1 conv) -> `depth` residual blocks -> project (1x1 conv), predicting the CHANGE over one frame.

    next = x + delta_scale * net(x)

    delta_scale (per variable) is the typical size of a one-frame change in normalised units
    (delta_std / std from the train statistics), so the network's raw output is O(1). The final layer
    is zero-initialised, so an untrained model returns its input unchanged.
    """
    lift: PeriodicConv
    blocks: list
    project: PeriodicConv
    delta_scale: tuple = eqx.field(static=True)   # fixed constants, not trained

    def __init__(self, *, width=64, depth=8, kernel_size=3, delta_scale=None, key, channels=4):
        keys = jax.random.split(key, depth + 2)
        self.lift = PeriodicConv(channels, width, 1, key=keys[0])
        self.blocks = [ResBlock(width, kernel_size, key=k) for k in keys[1:-1]]
        project = PeriodicConv(width, channels, 1, key=keys[-1])
        zero = lambda m: (m.conv.weight, m.conv.bias)
        self.project = eqx.tree_at(zero, project, (jnp.zeros_like(project.conv.weight),
                                                   jnp.zeros_like(project.conv.bias)))
        self.delta_scale = (1.0,) * channels if delta_scale is None else tuple(float(d) for d in delta_scale)

    def __call__(self, x):                       # x: (4, n, n), normalised
        h = self.lift(x)
        for block in self.blocks:
            h = block(h)
        scale = jnp.asarray(self.delta_scale, x.dtype)[:, None, None]
        return x + scale * self.project(jax.nn.gelu(h))


def count_params(model):
    """Number of trainable scalars (floating-point array leaves)."""
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))
    return sum(x.size for x in leaves)



# ---------------------------------------------------------------------------------------------------------
# Positivity-preserving wrapper
# ---------------------------------------------------------------------------------------------------------
GAMMA = 1.4
TINY = 1e-6   


def conserved_to_logprim(U):
    """(rho, rho u, rho v, E) -> (log rho, u, v, log p). U: (4, n, n) physical."""
    rho = jnp.maximum(U[0], TINY)
    u, v = U[1] / rho, U[2] / rho
    p = (GAMMA - 1) * (U[3] - 0.5 * rho * (u ** 2 + v ** 2))
    return jnp.stack([jnp.log(rho), u, v, jnp.log(jnp.maximum(p, TINY))])


def logprim_to_conserved(W):
    """(log rho, u, v, log p) -> (rho, rho u, rho v, E). Density, pressure and energy are positive by construction."""
    rho, u, v, p = jnp.exp(W[0]), W[1], W[2], jnp.exp(W[3])
    return jnp.stack([rho, rho * u, rho * v, p / (GAMMA - 1) + 0.5 * rho * (u ** 2 + v ** 2)])


class Positive(eqx.Module):
    """Wraps a core model so that it works in (log rho, u, v, log p) and can never output rho <= 0 or p <= 0.

    The input and output are the usual NORMALISED CONSERVED variables, so the loss, the data loader and the
    rollout code are unchanged. Inside:

      x (normalised conserved) -> U (physical) -> W = (log rho, u, v, log p) -> z = standardised W
        -> core(z) -> W' -> U' = conserved(W') -> normalised conserved output

    `core` is a residual model (e.g. ResNet) whose delta_scale is the typical one-frame change of z, so with
    its zero-initialised last layer the wrapped model starts as the identity, as before.
    """
    core: eqx.Module
    u_mean: tuple = eqx.field(static=True)
    u_std: tuple = eqx.field(static=True)
    w_mean: tuple = eqx.field(static=True)    # standardisation of (log rho, u, v, log p) (train stats)
    w_std: tuple = eqx.field(static=True)

    def __call__(self, x):
        c = lambda t: jnp.asarray(t, x.dtype)[:, None, None]
        U = x * c(self.u_std) + c(self.u_mean)
        z = (conserved_to_logprim(U) - c(self.w_mean)) / c(self.w_std)
        W = self.core(z) * c(self.w_std) + c(self.w_mean)
        return (logprim_to_conserved(W) - c(self.u_mean)) / c(self.u_std)


def logprim_stats(ds, n_frames=2000, seed=0):
    """Mean and std of (log rho, u, v, log p), and std of their one-frame change, from training pairs.

    ds: a PairDataset of the TRAIN split. Returns plain lists so they can be stored in config.json.
    """
    rng = np.random.default_rng(seed)
    pairs = ds.pairs[rng.choice(len(ds), min(n_frames, len(ds)), replace=False)]
    x, y = ds.get(pairs)
    to_w = jax.vmap(lambda z: conserved_to_logprim(ds.norm.decode(z)))
    w0, w1 = np.asarray(to_w(jnp.asarray(x))), np.asarray(to_w(jnp.asarray(y[:, 0])))
    mean, std = w0.mean(axis=(0, 2, 3)), w0.std(axis=(0, 2, 3))
    delta_std = (w1 - w0).std(axis=(0, 2, 3))
    return {"w_mean": mean.tolist(), "w_std": std.tolist(), "w_delta_scale": (delta_std / std).tolist()}