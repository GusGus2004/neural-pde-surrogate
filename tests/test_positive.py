"""Checks for the positivity-preserving wrapper (surrogate/models.py: Positive).
Run from the repo root: python tests/test_positive.py [packed_dir]"""
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from surrogate.data import PairDataset                                                       
from surrogate.models import conserved_to_logprim, logprim_stats, logprim_to_conserved        
from surrogate.rollout import pressure                                                   
from surrogate.train import build_model                                                       

packed = sys.argv[1] if len(sys.argv) > 1 else "data/packed"
failures = 0


def check(name, ok, value=""):
    global failures
    print(f"{name:<66} {'PASS' if ok else 'FAIL'}  {value}")
    failures += not ok


ds = PairDataset(packed, "train", 64, horizon=1)
x, y = ds.get(ds.pairs[np.random.default_rng(0).choice(len(ds), 8, replace=False)])
x = jnp.asarray(x)

U = ds.norm.decode(np.asarray(x))
back = np.asarray(jax.vmap(lambda u: logprim_to_conserved(conserved_to_logprim(u)))(jnp.asarray(U)))
rel = np.abs(back - U).max() / np.abs(U).max()
check("conserved -> (log rho, u, v, log p) -> conserved round trip", rel < 1e-5, f"max rel err {rel:.1e}")

stats = logprim_stats(ds, n_frames=500)
print("  w_mean", np.round(stats["w_mean"], 3), " w_std", np.round(stats["w_std"], 3),
      " w_delta_scale", np.round(stats["w_delta_scale"], 3))
check("statistics are finite and std > 0", all(np.isfinite(v).all() for v in stats.values())
      and min(stats["w_std"]) > 0)

cfg = dict(model="resnet", width=16, depth=2, positive=True, u_mean=ds.norm.mean.tolist(),
           u_std=ds.norm.std.tolist(), **stats)
model = build_model(cfg, None, jax.random.PRNGKey(0))


out = jax.vmap(model)(x)
err = float(jnp.abs(out - x).max())
check("untrained positive model returns its input", err < 1e-4, f"max diff {err:.1e} (normalised)")

w = model.core.project.conv.weight
wild = eqx.tree_at(lambda m: m.core.project.conv.weight, model, 3.0 * jax.random.normal(jax.random.PRNGKey(1), w.shape))
U_out = np.asarray(ds.norm.decode(np.asarray(jax.vmap(wild)(x))))
rho_min, p_min = U_out[:, 0].min(), pressure(U_out).min()
check("random large weights: density > 0 everywhere", rho_min > 0, f"min {rho_min:.2e}")
check("random large weights: pressure > 0 everywhere (to float32 round-off)",
      p_min > -1e-6 * np.abs(U_out[:, 3]).max(), f"min {p_min:.2e}")
check("random large weights: no NaN/inf", bool(np.isfinite(U_out).all()))

shift = lambda z: jnp.roll(z, (5, -9), axis=(-2, -1))
a, b = shift(wild(x[0])), wild(shift(x[0]))
check("roll(model(x)) == model(roll(x))", bool(jnp.allclose(a, b, atol=1e-5)))


yy = jnp.asarray(y[:, 0])
g = eqx.filter_grad(lambda m: jnp.mean((jax.vmap(m)(x) - yy) ** 2))(wild)
leaves = jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_array))
check("gradients finite and non-zero", all(bool(jnp.isfinite(l).all()) for l in leaves)
      and all(float(jnp.abs(l).max()) > 0 for l in leaves), f"{len(leaves)} arrays")

print(f"\n{'FAILED' if failures else 'ALL PASSED'} ({failures} failure{'s' if failures != 1 else ''})")
sys.exit(1 if failures else 0)