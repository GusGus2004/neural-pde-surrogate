"""Plot 1D Euler solution against exact Riemann solution (Toro ch. 4)"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

G = 1.4


def f_K(p, rho, pK, a):
    if p > pK:
        A = 2.0 / ((G + 1) * rho)
        B = (G - 1) / (G + 1) * pK
        return (p - pK) * np.sqrt(A / (p + B))
    return 2 * a / (G - 1) * ((p / pK) ** ((G - 1) / (2 * G)) - 1)


def exact_riemann(left, right, x, x0, t):
    rhoL, uL, pL = left
    rhoR, uR, pR = right
    aL, aR = np.sqrt(G * pL / rhoL), np.sqrt(G * pR / rhoR)

    func = lambda p: f_K(p, rhoL, pL, aL) + f_K(p, rhoR, pR, aR) + (uR - uL)
    ps = brentq(func, 1e-10, 10 * max(pL, pR))
    us = 0.5 * (uL + uR) + 0.5 * (f_K(ps, rhoR, pR, aR) - f_K(ps, rhoL, pL, aL))

    gm = (G - 1) / (G + 1)
    out = np.zeros((len(x), 3))
    for k, xi in enumerate(x):
        S = (xi - x0) / t
        if S <= us:
            if ps > pL:  # left shock
                SL = uL - aL * np.sqrt((G + 1) / (2 * G) * ps / pL + (G - 1) / (2 * G))
                out[k] = (left if S <= SL else
                          (rhoL * (ps / pL + gm) / (gm * ps / pL + 1), us, ps))
            else:  # left rarefaction
                SHL, STL = uL - aL, us - aL * (ps / pL) ** ((G - 1) / (2 * G))
                if S <= SHL:
                    out[k] = left
                elif S > STL:
                    out[k] = (rhoL * (ps / pL) ** (1 / G), us, ps)
                else:  # inside the fan
                    c = 2 / (G + 1) * (aL + (G - 1) / 2 * (uL - S))
                    out[k] = (rhoL * (c / aL) ** (2 / (G - 1)),
                              2 / (G + 1) * (aL + (G - 1) / 2 * uL + S),
                              pL * (c / aL) ** (2 * G / (G - 1)))
        else:
            if ps > pR:  # right shock
                SR = uR + aR * np.sqrt((G + 1) / (2 * G) * ps / pR + (G - 1) / (2 * G))
                out[k] = (right if S >= SR else
                          (rhoR * (ps / pR + gm) / (gm * ps / pR + 1), us, ps))
            else:  # right rarefaction
                SHR, STR = uR + aR, us + aR * (ps / pR) ** ((G - 1) / (2 * G))
                if S >= SHR:
                    out[k] = right
                elif S <= STR:
                    out[k] = (rhoR * (ps / pR) ** (1 / G), us, ps)
                else:  # inside the fan
                    c = 2 / (G + 1) * (aR - (G - 1) / 2 * (uR - S))
                    out[k] = (rhoR * (c / aR) ** (2 / (G - 1)),
                              2 / (G + 1) * (-aR + (G - 1) / 2 * uR + S),
                              pR * (c / aR) ** (2 * G / (G - 1)))
    return out, ps, us


def main():
    csv = Path(sys.argv[1] if len(sys.argv) > 1 else "solver/sod.csv")
    x, rho, u, p = np.loadtxt(csv, delimiter=",", skiprows=1, unpack=True)

    x_fine = np.linspace(0, 1, 2000)
    exact, ps, us = exact_riemann((1.0, 0.0, 1.0), (0.125, 0.0, 0.1), x_fine, 0.5, 0.2)
    exact_at_cells, _, _ = exact_riemann((1.0, 0.0, 1.0), (0.125, 0.0, 0.1), x, 0.5, 0.2)

    dx = x[1] - x[0]
    l1 = np.sum(np.abs(rho - exact_at_cells[:, 0])) * dx
    print(f"p* = {ps:.5f}, u* = {us:.5f}")
    print(f"L1 density error = {l1:.4e}  (N = {len(x)})")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, num, k, name in zip(axes, (rho, u, p), range(3), ("density", "velocity", "pressure")):
        ax.plot(x_fine, exact[:, k], "k-", lw=1, label="exact")
        ax.plot(x, num, "o", ms=2.5, label="numerical")
        ax.set_title(name)
        ax.set_xlabel("x")
    axes[0].legend()
    fig.suptitle(f"Sod shock tube, t = 0.2, N = {len(x)}")
    fig.tight_layout()

    out = Path("outputs") / f"{csv.stem}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()