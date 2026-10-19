"""Checks for surrogate/models.py. Run from the repo root: python tests/test_models.py [packed_dir]"""
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from surrogate.data import PairDataset            # noqa: E402
from surrogate.models import ResNet, count_params  # noqa: E402

packed = sys.argv[1] if len(sys.argv) > 1 else "data/packed"
failures = 0


def check(name, ok, value=""):
    global failures
    print(f"{name:<62} {'PASS' if ok else 'FAIL'}  {value}")
    failures += not ok


ds = PairDataset(packed, "train", 64, horizon=1)
delta_scale = ds.norm.delta_std / ds.norm.std
print("delta_scale (typical one-frame change, normalised units):", np.round(delta_scale, 4), "\n")

model = ResNet(width=64, depth=8, delta_scale=delta_scale, key=jax.random.PRNGKey(0))
x, y = ds.get(ds.pairs[np.random.default_rng(0).choice(len(ds), 4, replace=False)])
x, y = jnp.asarray(x), jnp.asarray(y[:, 0])

# Shapes and size
out = jax.vmap(model)(x)
check("output shape == input shape (4, 4, 64, 64)", out.shape == x.shape, out.shape)
n_params = count_params(model)
check("parameter count as expected (591,428)", n_params == 591_428, f"{n_params:,}")

check("untrained model returns its input", bool(jnp.array_equal(out, x)))

# Periodic padding => exact translation equivariance on the torus
w = model.project.conv.weight
noisy = eqx.tree_at(lambda m: m.project.conv.weight, model,
                    0.1 * jax.random.normal(jax.random.PRNGKey(1), w.shape))
shift = lambda z: jnp.roll(z, (7, -13), axis=(-2, -1))
a = shift(noisy(x[0]))
b = noisy(shift(x[0]))
check("roll(model(x)) == model(roll(x))  (periodic padding)", bool(jnp.allclose(a, b, atol=1e-5)),
      f"max diff {float(jnp.abs(a - b).max()):.1e}")

# Gradients reach every layer once the last layer is non-zero
loss = lambda m, x, y: jnp.mean((jax.vmap(m)(x) - y) ** 2)
grads = eqx.filter_grad(loss)(noisy, x, y)
gnorms = [float(jnp.linalg.norm(g)) for g in jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))]
check("every parameter array gets a non-zero gradient", all(g > 0 for g in gnorms), f"{len(gnorms)} arrays")

# A few optimiser steps on one fixed batch reduce the loss
opt = optax.adam(1e-3)
m = model
state = opt.init(eqx.filter(m, eqx.is_array))


@eqx.filter_jit
def step(m, state, x, y):
    l, g = eqx.filter_value_and_grad(loss)(m, x, y)
    updates, state = opt.update(g, state, m)
    return eqx.apply_updates(m, updates), state, l


l0 = float(loss(m, x, y))
for _ in range(30):
    m, state, l = step(m, state, x, y)
l1 = float(loss(m, x, y))
check("30 Adam steps on one batch reduce the loss", l1 < 0.5 * l0, f"{l0:.2e} -> {l1:.2e}")

xb, yb = ds.get(ds.pairs[:32])
xb, yb = jnp.asarray(xb), jnp.asarray(yb[:, 0])
m, state, l = step(m, state, xb, yb)             
jax.block_until_ready(l)
t0 = time.time()
for _ in range(5):
    m, state, l = step(m, state, xb, yb)
jax.block_until_ready(l)
dt = (time.time() - t0) / 5
print(f"\ntraining step, batch 32 at 64^2: {dt * 1e3:.0f} ms  ({32 / dt:,.0f} pairs/s) on {jax.devices()[0]}")

print(f"\n{'FAILED' if failures else 'ALL PASSED'} ({failures} failure{'s' if failures != 1 else ''})")
sys.exit(1 if failures else 0)