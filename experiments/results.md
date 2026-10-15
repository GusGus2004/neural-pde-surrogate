# Results log

All runs: 1D Euler, γ = 1.4, transmissive boundaries unless stated.
Errors are L1 in density against the exact solution.

## 1. Sod shock tube (t = 0.2)

### Rusanov, first order (forward Euler, CFL 0.8)

| N    | L1 error  | Order |
|------|-----------|-------|
| 100  | 2.279e-2  | –     |
| 200  | 1.561e-2  | 0.55  |
| 400  | 1.032e-2  | 0.60  |
| 800  | 6.682e-3  | 0.63  |
| 1600 | 4.268e-3  | 0.65  |
| 3200 | 2.726e-3  | 0.65  |

Order ≈ 0.65, well below 1: the discontinuities dominate the error.
Exact solver check: p* = 0.30313, u* = 0.92745 (matches Toro).

### Riemann solver comparison, first order, N = 800 (forward Euler, CFL 0.8)

| Scheme  | L1 error |
|---------|----------|
| Rusanov | 6.68e-3  |
| HLLC    | 3.96e-3  |

HLLC is 41% lower and beats Rusanov at N = 1600. The contact is much sharper;
the shock is about the same. In the HLLC run, the window x ∈ [0.6, 0.8] around
the contact still contributes 1.7e-3 of the total.

### MUSCL + HLLC, N = 200 (SSP-RK2, CFL 0.8)

| Limiter   | L1 error |
|-----------|----------|
| none      | 1.34e-2  |
| minmod    | 4.46e-3  |
| van Leer  | 3.38e-3  |
| unlimited | crashed  |

- Van Leer at N = 200 beats first-order HLLC at N = 800.
- Contact goes from ~20 cells wide to ~3–4 cells with van Leer.
- Small undershoot just right of the contact with van Leer (x ≈ 0.70).
- Unlimited: overshoots at the initial jump drive pressure negative within a few
  steps (cell 101: rho = 0.0049, p = -3.6). Now caught by the positivity check.

## 2. Smooth advection (sine wave), t = 1, periodic

ρ = 1 + 0.2 sin(2πx), u = 1, p = 1. HLLC, SSP-RK2, CFL 0.5.

| N   | none L1  | order | minmod L1 | order | van Leer L1 | order |
|-----|----------|-------|-----------|-------|-------------|-------|
| 50  | 4.153e-2 | –     | 6.373e-3  | –     | 2.884e-3    | –     |
| 100 | 2.281e-2 | 0.86  | 1.835e-3  | 1.80  | 7.559e-4    | 1.93  |
| 200 | 1.197e-2 | 0.93  | 5.039e-4  | 1.86  | 1.875e-4    | 2.01  |
| 400 | 6.131e-3 | 0.96  | 1.355e-4  | 1.89  | 4.620e-5    | 2.02  |
| 800 | 3.103e-3 | 0.98  | 3.613e-5  | 1.91  | 1.131e-5    | 2.03  |

- Confirms first order without a slope and second order with van Leer.
- L∞ order drops to ~1.3 (minmod) and ~1.4 (van Leer): the limiter clips the
  slope to zero at the wave's peak and trough (clipping at smooth extrema).
- Van Leer at N = 50 is as accurate as first order at N = 800.
- Mass drift ~1e-15 with periodic boundaries.

Figure: `outputs/convergence_sine.png`