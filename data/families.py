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

# blast

BLAST = dict(
    rho0=(0.5, 2.0),          # background density (log-uniform)
    p0=(0.1, 0.5),            # background pressure (log-uniform)
    n_blasts=(1, 3),          # number of blasts (incl)
    radius=(0.04, 0.12),      # blast radius
    p_ratio=(5.0, 40.0),      # blast pressure / background pressure (log-uniform)
    rho_ratio=(0.5, 2.0),     # blast density / background density (log-uniform)
    n_bubbles=(0, 2),         # number of density bubbles (inclusive), in pressure balance
    bubble_radius=(0.05, 0.15),
    bubble_ratio=(0.3, 3.0),  # bubble density / background density (log-uniform)
    edge=0.005,               # width of the smoothed edges (physical units, same at every resolution)
    t_end=0.25,               # same as riemann: every family must share the frame spacing (0.25 / 50 = 0.005)
)


def periodic_distance(X, Y, cx, cy):
    """Distance to (cx, cy) on the unit periodic box (the shortest way round)."""
    dx = (X - cx + 0.5) % 1.0 - 0.5
    dy = (Y - cy + 0.5) % 1.0 - 0.5
    return np.sqrt(dx**2 + dy**2)


def smooth_disc(r, radius, width):
    """1 inside the disc, 0 outside, with a smooth tanh edge so the circle is not a staircase."""
    return 0.5 * (1.0 - np.tanh((r - radius) / width))


def blast(rng, n):
    """Quiet background with 1-3 circular high-pressure regions and 0-2 density bubbles, periodic box.

    The bubbles are in pressure balance with the background, so they only move once a blast wave
    reaches them: shock-bubble interaction, a classic source of vorticity.
    """
    c = BLAST
    rho0, p0 = log_uniform(rng, *c["rho0"]), log_uniform(rng, *c["p0"])
    x = cell_centres(n)
    X, Y = np.meshgrid(x, x)
    rho = np.full((n, n), rho0)
    p = np.full((n, n), p0)

    blasts = []
    for _ in range(rng.integers(c["n_blasts"][0], c["n_blasts"][1] + 1)):
        cx, cy = rng.uniform(0, 1, size=2)
        R = rng.uniform(*c["radius"])
        pr, rr = log_uniform(rng, *c["p_ratio"]), log_uniform(rng, *c["rho_ratio"])
        w = smooth_disc(periodic_distance(X, Y, cx, cy), R, c["edge"])
        p += w * (pr - 1.0) * p0
        rho += w * (rr - 1.0) * rho0
        blasts.append([cx, cy, R, pr, rr])

    bubbles = []
    for _ in range(rng.integers(c["n_bubbles"][0], c["n_bubbles"][1] + 1)):
        cx, cy = rng.uniform(0, 1, size=2)
        R = rng.uniform(*c["bubble_radius"])
        br = log_uniform(rng, *c["bubble_ratio"])
        w = smooth_disc(periodic_distance(X, Y, cx, cy), R, c["edge"])
        rho *= 1.0 + w * (br - 1.0)          # change density only: pressure balance is kept
        bubbles.append([cx, cy, R, br])

    u = np.zeros((n, n))
    v = np.zeros((n, n))
    params = {"rho0": rho0, "p0": p0,
              "blasts": np.array(blasts).reshape(-1, 5),    
              "bubbles": np.array(bubbles).reshape(-1, 4)}  
    return rho, u, v, p, c["t_end"], params


FAMILIES = {"riemann": riemann, "blast": blast}


def sample(family, index, n):
    if family not in FAMILIES:
        raise ValueError(f"unknown family '{family}' (have: {', '.join(FAMILIES)})")
    rng = rng_for(family, index)
    rho, u, v, p, t_end, params = FAMILIES[family](rng, n)
    params = {"family": family, "index": int(index), **params}
    return rho, u, v, p, t_end, params