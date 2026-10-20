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

class ConvBlock(eqx.Module):
    """Two periodic 3x3 convolutions with a residual connection: skip(x) + conv(gelu(conv(gelu(x)))).
    """
    conv1: PeriodicConv
    conv2: PeriodicConv
    skip: PeriodicConv | None

    def __init__(self, c_in, c_out, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.conv1 = PeriodicConv(c_in, c_out, 3, key=k1)
        self.conv2 = PeriodicConv(c_out, c_out, 3, key=k2)
        self.skip = None if c_in == c_out else PeriodicConv(c_in, c_out, 1, key=k3)

    def __call__(self, x):
        h = self.conv2(jax.nn.gelu(self.conv1(jax.nn.gelu(x))))
        return (x if self.skip is None else self.skip(x)) + h


def downsample(x):
    """2x2 average pooling: (c, n, n) -> (c, n/2, n/2). Averaging, like the dataset's own coarsening."""
    c, n, m = x.shape
    return x.reshape(c, n // 2, 2, m // 2, 2).mean(axis=(2, 4))


def upsample(x):
    """Nearest-neighbour upsampling: (c, n, n) -> (c, 2n, 2n)."""
    return jnp.repeat(jnp.repeat(x, 2, axis=1), 2, axis=2)


class UNet(eqx.Module):
    """U-Net with periodic convolutions, predicting the change over one frame (like ResNet).

    levels = number of halvings: with n = 64 and levels = 3 the coarsest grid is 8 x 8, where a single
    3x3 convolution already spans most of the domain. That is the point of a U-Net here: information
    travels across the whole domain in a few layers, whereas the ResNet only sees +-(2 x depth) cells.

    Channels double at each coarser level: width, 2 width, 4 width, ... Each level has one ConvBlock on
    the way down and one on the way up; the way up concatenates the skip connection from the way down.

    Translation equivariance: exact only for shifts that are multiples of 2**levels (the pooling grid).
    """
    lift: PeriodicConv
    down: list
    bottleneck: ConvBlock
    up: list
    project: PeriodicConv
    delta_scale: tuple = eqx.field(static=True)

    def __init__(self, *, width=16, levels=3, delta_scale=None, key, channels=4):
        keys = iter(jax.random.split(key, 2 * levels + 3))
        ch = [width * 2 ** l for l in range(levels + 1)]                  # 16, 32, 64, 128
        self.lift = PeriodicConv(channels, ch[0], 1, key=next(keys))
        self.down = [ConvBlock(ch[max(l - 1, 0)], ch[l], key=next(keys)) for l in range(levels)]
        self.bottleneck = ConvBlock(ch[levels - 1], ch[levels], key=next(keys))
        self.up = [ConvBlock(ch[l + 1] + ch[l], ch[l], key=next(keys)) for l in range(levels)]
        project = PeriodicConv(ch[0], channels, 1, key=next(keys))
        zero = lambda m: (m.conv.weight, m.conv.bias)
        self.project = eqx.tree_at(zero, project, (jnp.zeros_like(project.conv.weight),
                                                   jnp.zeros_like(project.conv.bias)))
        self.delta_scale = (1.0,) * channels if delta_scale is None else tuple(float(d) for d in delta_scale)

    def __call__(self, x):                     
        h = self.lift(x)
        skips = []
        for block in self.down:
            h = block(h)
            skips.append(h)
            h = downsample(h)
        h = self.bottleneck(h)
        for block, skip in zip(reversed(self.up), reversed(skips)):
            h = block(jnp.concatenate([upsample(h), skip], axis=0))
        scale = jnp.asarray(self.delta_scale, x.dtype)[:, None, None]
        return x + scale * self.project(jax.nn.gelu(h))

class SpectralConv2d(eqx.Module):
    """The core of the FNO: a convolution done as a multiplication in Fourier space, keeping only low modes.

      x (c_in, n, n) --rfft2--> X (c_in, n, n/2+1) --keep |kx| < modes, ky < modes, multiply by learned
      complex weights, mixing channels--> Y (c_out, ...) --zero all other modes, irfft2--> y (c_out, n, n)

    Because the weights live on Fourier modes, not grid cells, the same layer works at any resolution n
    (as long as n/2 > modes).
    rfft2 only stores ky >= 0 (the input is real, so negative ky are complex conjugates); kx runs over both
    signs, so two weight blocks are needed: kx in [0, modes) and kx in [-modes, 0).
    Weights are stored as separate real and imaginary float arrays.
    """
    w_re: jax.Array          # (2, c_in, c_out, modes, modes): [0] for kx >= 0, [1] for kx < 0
    w_im: jax.Array
    modes: int = eqx.field(static=True)

    def __init__(self, c_in, c_out, modes, *, key):
        k1, k2 = jax.random.split(key)
        scale = 1.0 / (c_in * c_out)                                    # Li et al. (2021)
        shape = (2, c_in, c_out, modes, modes)
        self.w_re = scale * jax.random.uniform(k1, shape)
        self.w_im = scale * jax.random.uniform(k2, shape)
        self.modes = modes

    def __call__(self, x):                      
        c_in, n, _ = x.shape
        m = self.modes
        X = jnp.fft.rfft2(x)                                             # (c_in, n, n//2 + 1) complex
        W = self.w_re + 1j * self.w_im
        mix = lambda Xb, Wb: jnp.einsum("ixy,ioxy->oxy", Xb, Wb)        
        Y = jnp.zeros((W.shape[2], n, n // 2 + 1), dtype=X.dtype)
        Y = Y.at[:, :m, :m].set(mix(X[:, :m, :m], W[0]))                 # kx = 0 .. m-1
        Y = Y.at[:, -m:, :m].set(mix(X[:, -m:, :m], W[1]))               # kx = -m .. -1
        return jnp.fft.irfft2(Y, s=(n, n))


class FNOBlock(eqx.Module):
    """One Fourier layer: gelu(SpectralConv(h) + W h), where W is a 1x1 convolution.

    The spectral path mixes information across the WHOLE domain but only at low frequencies; the 1x1 path
    acts cell by cell at all frequencies, which is what lets the network still represent sharp features.
    """
    spectral: SpectralConv2d
    pointwise: PeriodicConv

    def __init__(self, width, modes, *, key):
        k1, k2 = jax.random.split(key)
        self.spectral = SpectralConv2d(width, width, modes, key=k1)
        self.pointwise = PeriodicConv(width, width, 1, key=k2)

    def __call__(self, h):
        return jax.nn.gelu(self.spectral(h) + self.pointwise(h))


class FNO(eqx.Module):
    """Fourier Neural Operator (Li et al., ICLR 2021), predicting the CHANGE over one frame like the others.

    lift (1x1 conv, 4 -> width) -> `depth` FNOBlocks -> project (1x1 conv width -> 128, gelu, 1x1 128 -> 4)

    No grid coordinates are appended to the input (the original paper appends x, y). On a periodic domain
    the physics has no preferred position, and leaving them out keeps the model exactly translation-
    equivariant, like the ResNet. The last layer is zero-initialised, so the untrained model is the identity.
    """
    lift: PeriodicConv
    blocks: list
    project1: PeriodicConv
    project2: PeriodicConv
    delta_scale: tuple = eqx.field(static=True)

    def __init__(self, *, width=32, depth=4, modes=12, delta_scale=None, key, channels=4, hidden=128):
        keys = jax.random.split(key, depth + 3)
        self.lift = PeriodicConv(channels, width, 1, key=keys[0])
        self.blocks = [FNOBlock(width, modes, key=k) for k in keys[1:depth + 1]]
        self.project1 = PeriodicConv(width, hidden, 1, key=keys[-2])
        project2 = PeriodicConv(hidden, channels, 1, key=keys[-1])
        zero = lambda m: (m.conv.weight, m.conv.bias)
        self.project2 = eqx.tree_at(zero, project2, (jnp.zeros_like(project2.conv.weight),
                                                     jnp.zeros_like(project2.conv.bias)))
        self.delta_scale = (1.0,) * channels if delta_scale is None else tuple(float(d) for d in delta_scale)

    def __call__(self, x):                       
        h = self.lift(x)
        for block in self.blocks:
            h = block(h)
        scale = jnp.asarray(self.delta_scale, x.dtype)[:, None, None]
        return x + scale * self.project2(jax.nn.gelu(self.project1(h)))
    
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