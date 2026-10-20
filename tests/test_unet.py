"""Checks for the U-Net in surrogate/models.py. Run from the repo root: python tests/test_unet.py [packed_dir]"""
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from surrogate.data import PairDataset                                         
from surrogate.models import ResNet, UNet, count_params, downsample, upsample  
from surrogate.train import build_model, loss_onestep, make_step               

packed = sys.argv[1] if len(sys.argv) > 1 else "data/packed"
failures = 0


def check(name, ok, value=""):
    global failures
    print(f"{name:<66} {'PASS' if ok else 'FAIL'}  {value}")
    failures += not ok


ds = PairDataset(packed, "train", 64, horizon=1)
scale = ds.norm.delta_std / ds.norm.std
x, y = ds.get(ds.pairs[np.random.default_rng(0).choice(len(ds), 4, replace=False)])
x, y = jnp.asarray(x), jnp.asarray(y[:, 0])

z = jax.random.normal(jax.random.PRNGKey(0), (3, 64, 64))
check("downsample keeps the domain average", bool(jnp.allclose(downsample(z).mean(), z.mean(), atol=1e-6)))
check("downsample(upsample(z)) == z", bool(jnp.allclose(downsample(upsample(z)), z, atol=1e-6)))

model = UNet(width=16, levels=3, delta_scale=scale, key=jax.random.PRNGKey(0))
out = jax.vmap(model)(x)
check("output shape == input shape", out.shape == x.shape, out.shape)
n_params = count_params(model)
check("parameter count (w16, 3 levels) = 516,516", n_params == 516_516, f"{n_params:,}")
check("untrained model returns its input (zero last layer)", bool(jnp.array_equal(out, x)))


w = model.project.conv.weight
noisy = eqx.tree_at(lambda m: m.project.conv.weight, model, 0.1 * jax.random.normal(jax.random.PRNGKey(1), w.shape))
def equiv_err(sx, sy):
    s = lambda q: jnp.roll(q, (sx, sy), axis=(-2, -1))
    return float(jnp.abs(s(noisy(x[0])) - noisy(s(x[0]))).max())
e8, e1 = equiv_err(8, -16), equiv_err(1, 3)
check("shift by multiples of 8: exactly equivariant", e8 < 1e-5, f"max diff {e8:.1e}")
print(f"  (shift by (1, 3): max diff {e1:.1e} -- not exact, because the pooling grid is fixed; expected)")


loss = lambda m: jnp.mean((jax.vmap(m)(x) - y) ** 2)
g = eqx.filter_grad(loss)(noisy)
leaves = jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_array))
check("every parameter array gets a non-zero gradient", all(float(jnp.abs(l).max()) > 0 for l in leaves),
      f"{len(leaves)} arrays")

cfg = dict(model="unet", width=16, levels=3, n=64)
check("build_model('unet') works", isinstance(build_model(cfg, scale, jax.random.PRNGKey(0)), UNet))

# 100 Adam steps on one batch reduce the loss (the U-Net starts slower than the ResNet, then catches up)
opt = optax.adam(1e-3)
step = make_step(opt, loss_onestep)
m, st = model, opt.init(eqx.filter(model, eqx.is_array))
l0 = float(loss(m))
for _ in range(100):
    m, st, _ = step(m, st, x, y[:, None])
l1 = float(loss(m))
check("100 Adam steps on one batch at least halve the loss", l1 < 0.5 * l0, f"{l0:.2e} -> {l1:.2e}")

# Cost per training step, against the ResNet used so far (batch 32)
xb, yb = ds.get(ds.pairs[:32])
xb, yb = jnp.asarray(xb), jnp.asarray(yb)
for name, mdl in [("ResNet w32 d4", ResNet(width=32, depth=4, delta_scale=scale, key=jax.random.PRNGKey(0))),
                  ("UNet w16 L3", model)]:
    s_, o_ = make_step(opt, loss_onestep), opt.init(eqx.filter(mdl, eqx.is_array))
    mdl, o_, l_ = s_(mdl, o_, xb, yb)
    jax.block_until_ready(l_)
    t0 = time.time()
    for _ in range(3):
        mdl, o_, l_ = s_(mdl, o_, xb, yb)
    jax.block_until_ready(l_)
    print(f"  {name:14} {count_params(mdl):>9,} params   {(time.time() - t0) / 3:.2f} s/step")

print(f"\n{'FAILED' if failures else 'ALL PASSED'} ({failures} failure{'s' if failures != 1 else ''})")
sys.exit(1 if failures else 0)