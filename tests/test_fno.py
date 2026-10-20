"""Checks for the FNO in surrogate/models.py. Run from the repo root: python tests/test_fno.py [packed_dir]

The spectral layer is tested against what the maths says it must do:
  - it equals a slow, explicit loop implementation (channel mixing and mode indexing are right)
  - a single Fourier mode inside the kept range comes out as that same mode only
  - a mode outside the kept range is removed completely
  - it is linear
  - on a smooth (band-limited) input it gives the same answer at 64^2 and 128^2 (resolution independence)
"""
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from surrogate.data import PairDataset                                 # noqa: E402
from surrogate.models import FNO, SpectralConv2d, count_params         # noqa: E402
from surrogate.train import build_model, loss_onestep, make_step       # noqa: E402

packed = sys.argv[1] if len(sys.argv) > 1 else "data/packed"
failures = 0


def check(name, ok, value=""):
    global failures
    print(f"{name:<66} {'PASS' if ok else 'FAIL'}  {value}")
    failures += not ok


def wave(n, kx, ky, phase=0.3):
    """cos(2 pi (kx i + ky j) / n + phase) on an n x n grid, as a (1, n, n) array. kx runs along axis 1."""
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return jnp.asarray(np.cos(2 * np.pi * (kx * i + ky * j) / n + phase)[None], dtype=jnp.float32)


def spectrum(y):
    """|FFT| of a (1, n, n) array, as a numpy array (n, n)."""
    return np.abs(np.fft.fft2(np.asarray(y[0])))


# explicit, slow reference implementation
m, n = 4, 16
layer = SpectralConv2d(2, 3, m, key=jax.random.PRNGKey(0))
x = jax.random.normal(jax.random.PRNGKey(1), (2, n, n))
X = np.fft.rfft2(np.asarray(x))
W = np.asarray(layer.w_re) + 1j * np.asarray(layer.w_im)
Y = np.zeros((3, n, n // 2 + 1), complex)
for o in range(3):
    for i in range(2):
        Y[o, :m, :m] += X[i, :m, :m] * W[0, i, o]
        Y[o, -m:, :m] += X[i, -m:, :m] * W[1, i, o]
ref = np.fft.irfft2(Y, s=(n, n))
err = np.abs(np.asarray(layer(x)) - ref).max()
check("spectral conv == explicit loop reference", err < 1e-5, f"max diff {err:.1e}")

# mode selec
m, n = 8, 64
layer = SpectralConv2d(1, 1, m, key=jax.random.PRNGKey(2))
for kx, ky in [(3, 5), (-6, 2)]:                                      
    S = spectrum(layer(wave(n, kx, ky)))
    keep = np.zeros_like(S, bool)
    keep[kx % n, ky % n] = keep[-kx % n, -ky % n] = True               
    leak = S[~keep].max() / S[keep].max()
    check(f"mode ({kx:+d},{ky:+d}) inside range: output is that mode only", leak < 1e-5, f"leakage {leak:.1e}")
for kx, ky in [(20, 3), (2, 12)]:                                      
    out = float(jnp.abs(layer(wave(n, kx, ky))).max())
    check(f"mode ({kx:+d},{ky:+d}) outside range: removed", out < 1e-6, f"max |output| {out:.1e}")

# linear
a, b = wave(n, 2, 3), wave(n, 5, 1, phase=1.1)
lin = float(jnp.abs(layer(2.0 * a - 3.0 * b) - (2.0 * layer(a) - 3.0 * layer(b))).max())
check("linear: L(2a - 3b) == 2 L(a) - 3 L(b)", lin < 1e-5, f"max diff {lin:.1e}")

# res indep
smooth64 = wave(64, 3, 2) + 0.5 * wave(64, -1, 5, phase=2.0)
smooth128 = wave(128, 3, 2) + 0.5 * wave(128, -1, 5, phase=2.0)
y64, y128 = layer(smooth64), layer(smooth128)
res = float(jnp.abs(y128[:, ::2, ::2] - y64).max() / jnp.abs(y64).max())
check("same weights, same answer at 64^2 and 128^2 (smooth input)", res < 1e-4, f"rel diff {res:.1e}")

# fno full
ds = PairDataset(packed, "train", 64, horizon=1)
scale = ds.norm.delta_std / ds.norm.std
xb, yb = ds.get(ds.pairs[np.random.default_rng(0).choice(len(ds), 4, replace=False)])
xb, yb = jnp.asarray(xb), jnp.asarray(yb[:, 0])
model = FNO(width=32, depth=4, modes=16, delta_scale=scale, key=jax.random.PRNGKey(0))
out = jax.vmap(model)(xb)
check("FNO output shape == input shape", out.shape == xb.shape, out.shape)
n_params = count_params(model)
check("parameter count (w32, d4, 16 modes) = 4,203,428", n_params == 4_203_428, f"{n_params:,}")
check("untrained FNO returns its input (zero last layer)", bool(jnp.array_equal(out, xb)))

w = model.project2.conv.weight
noisy = eqx.tree_at(lambda m: m.project2.conv.weight, model, 0.1 * jax.random.normal(jax.random.PRNGKey(3), w.shape))
for sx, sy in [(1, 3), (-7, 20)]:
    s = lambda q: jnp.roll(q, (sx, sy), axis=(-2, -1))
    e = float(jnp.abs(s(noisy(xb[0])) - noisy(s(xb[0]))).max())
    check(f"translation-equivariant for ANY shift, e.g. ({sx:+d},{sy:+d})", e < 1e-4, f"max diff {e:.1e}")

big = jnp.asarray(np.kron(np.asarray(xb[0]), np.ones((1, 2, 2))))   # same state on a 128^2 grid
check("runs on a 128^2 grid with the same weights", noisy(big).shape == (4, 128, 128))

loss = lambda m: jnp.mean((jax.vmap(m)(xb) - yb) ** 2)
g = eqx.filter_grad(loss)(noisy)
leaves = jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_array))
check("every parameter array gets a non-zero gradient", all(float(jnp.abs(l).max()) > 0 for l in leaves),
      f"{len(leaves)} arrays")

check("build_model('fno') works",
      isinstance(build_model(dict(model="fno", width=32, depth=4, modes=16), scale, jax.random.PRNGKey(0)), FNO))

opt = optax.adam(1e-3)
step = make_step(opt, loss_onestep)
mdl, st = model, opt.init(eqx.filter(model, eqx.is_array))
l0 = float(loss(mdl))
for _ in range(100):
    mdl, st, _ = step(mdl, st, xb, yb[:, None])
l1 = float(loss(mdl))
check("100 Adam steps on one batch at least halve the loss", l1 < 0.5 * l0, f"{l0:.2e} -> {l1:.2e}")

print(f"\n{'FAILED' if failures else 'ALL PASSED'} ({failures} failure{'s' if failures != 1 else ''})")
sys.exit(1 if failures else 0)