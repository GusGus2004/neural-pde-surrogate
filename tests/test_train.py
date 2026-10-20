"""Checks for unrolled training losses in surrogate/train.py.
Run from repo root: python tests/test_train.py [packed_dir]"""
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from surrogate.data import PairDataset                                                    # noqa: E402
from surrogate.models import ResNet                                                       # noqa: E402
from surrogate.train import apply_n, loss_multistep, loss_onestep, loss_pushforward, mse  # noqa: E402

packed = sys.argv[1] if len(sys.argv) > 1 else "data/packed"
failures = 0


def check(name, ok, value=""):
    global failures
    print(f"{name:<66} {'PASS' if ok else 'FAIL'}  {value}")
    failures += not ok


ds = PairDataset(packed, "train", 64, horizon=3)
x, y = ds.get(ds.pairs[np.random.default_rng(0).choice(len(ds), 4, replace=False)])
x, y = jnp.asarray(x), jnp.asarray(y)                                     # y: (4, 3, 4, 64, 64)
model = ResNet(width=16, depth=2, delta_scale=ds.norm.delta_std / ds.norm.std, key=jax.random.PRNGKey(0))
# give the zero-initialised last layer random weights, so the model is not the identity
w = model.project.conv.weight
model = eqx.tree_at(lambda m: m.project.conv.weight, model, 0.1 * jax.random.normal(jax.random.PRNGKey(1), w.shape))
close = lambda a, b: bool(jnp.allclose(a, b, rtol=1e-5, atol=1e-7))

# With k = 1 all three losses are the same thing
y1 = y[:, :1]
l1 = loss_onestep(model, x, y1)
check("k=1: multistep loss == one-step loss", close(loss_multistep(model, x, y1), l1))
check("k=1: pushforward loss == one-step loss", close(loss_pushforward(model, x, y1), l1))

# Multistep = mean of errors after 1, 2, 3 self-fed steps
manual = jnp.mean(jnp.stack([jnp.mean((apply_n(model, x, j + 1) - y[:, j]) ** 2) for j in range(3)]))
check("k=3: multistep == mean over steps of self-fed errors", close(loss_multistep(model, x, y), manual))

# Pushforward value = one step from the twice-pushed state to frame t+3
pushed = apply_n(model, x, 2)
check("k=3: pushforward == error of one step from 2x-pushed input", close(loss_pushforward(model, x, y),
                                                                        mse(model, pushed, y[:, 2])))

# Pushforward gradient must NOT flow through the pushing: same as treating the pushed state as fixed data
g_pf = eqx.filter_grad(loss_pushforward)(model, x, y)
g_fixed = eqx.filter_grad(lambda m: mse(m, jax.lax.stop_gradient(pushed), y[:, 2]))(model)
g_through = eqx.filter_grad(lambda m: mse(m, apply_n(m, x, 2), y[:, 2]))(model)
flat = lambda g: jnp.concatenate([a.ravel() for a in jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_array))])
check("pushforward gradient == gradient with the pushed input held fixed", close(flat(g_pf), flat(g_fixed)))
diff = float(jnp.abs(flat(g_pf) - flat(g_through)).max())
check("... and differs from backpropagating through the chain", diff > 1e-6, f"max diff {diff:.1e}")

# Targets line up with what the model is asked to predict
traj = ds.trajectory(int(ds.pairs[0, 0]))
xa, ya = ds.get(ds.pairs[:1])
t0 = int(ds.pairs[0, 1])
check("horizon 3: y[:, j] is frame t+1+j", all(np.array_equal(ya[0, j], traj[t0 + 1 + j]) for j in range(3)))

print(f"\n{'FAILED' if failures else 'ALL PASSED'} ({failures} failure{'s' if failures != 1 else ''})")
sys.exit(1 if failures else 0)