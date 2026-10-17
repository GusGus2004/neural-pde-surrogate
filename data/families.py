"""Random init-condition families for the dataset

Every sample is fully determined by (family, index): the same pair always gives the same initial
condition, on any machine. That makes each dataset run reproducible and lets a cluster job array
regenerate any single run from its index.

  rho, u, v, p, t_end, params = sample("riemann", index=17, n=512)
"""
import zlib

import numpy as np

from ic import cell_centres


def rng_for(family, index):
    """A random generator seeded only by the family name and the index."""
    family_id = zlib.crc32(family.encode())          # stable across Python versions and machines
    return np.random.default_rng([family_id, index])


def log_uniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


RIEMANN = dict(
    rho=(0.1, 2.0),      # density range (sampled log-uniformly)
    p=(0.1, 2.0),        # pressure range (log-uniformly)
    vel=1.0,             # each velocity component uniform in [-vel, vel]
    split=(0.3, 0.7),    # position of the quadrant corner, in each direction
    t_end=0.25,
)


def riemann(rng, n):
    """Four constant quadrants with random states and a random corner position, periodic box.

    Because the box is periodic, the edges x = 0 and y = 0 are also interfaces, so every sample
    contains two vertical and two horizontal discontinuities.
    """
    c = RIEMANN
    xs, ys = rng.uniform(*c["split"]), rng.uniform(*c["split"])
    # states[k] = (rho, u, v, p) for quadrants k = 0 bottom-left, 1 bottom-right, 2 top-left, 3 top-right
    states = np.array([[log_uniform(rng, *c["rho"]), rng.uniform(-c["vel"], c["vel"]),
                        rng.uniform(-c["vel"], c["vel"]), log_uniform(rng, *c["p"])] for _ in range(4)])
    x = cell_centres(n)
    X, Y = np.meshgrid(x, x)                                      # shape (ny, nx)
    quadrant = (X > xs).astype(int) + 2 * (Y > ys).astype(int)    # 0..3, matching the order above
    rho, u, v, p = (states[quadrant, q] for q in range(4))
    params = {"split_x": xs, "split_y": ys, "states": states}
    return rho, u, v, p, c["t_end"], params


FAMILIES = {"riemann": riemann}


def sample(family, index, n):
    if family not in FAMILIES:
        raise ValueError(f"unknown family '{family}' (have: {', '.join(FAMILIES)})")
    rng = rng_for(family, index)
    rho, u, v, p, t_end, params = FAMILIES[family](rng, n)
    params = {"family": family, "index": int(index), **params}
    return rho, u, v, p, t_end, params