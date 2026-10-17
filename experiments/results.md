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


## 3. 2D solver

### Diagonal advection, t = 1, periodic (HLLC, SSP-RK2, CFL 0.5)

ρ = 1 + 0.2 sin(2π(x + y)), u = v = 1, p = 1.

| N   | none L1  | order | van Leer L1 | order | van Leer L∞ order |
|-----|----------|-------|-------------|-------|-------------------|
| 32  | 9.024e-2 | –     | 1.088e-2    | –     | –                 |
| 64  | 5.862e-2 | 0.62  | 3.203e-3    | 1.76  | 1.41              |
| 128 | 3.379e-2 | 0.79  | 8.531e-4    | 1.91  | 1.41              |
| 256 | 1.820e-2 | 0.89  | 2.130e-4    | 2.00  | 1.41              |

- Second order in 2D with van Leer; first order still approaching 1 (diagonal
  wave has wavelength 1/√2, so coarse grids are pre-asymptotic).
- L∞ order 1.41: clipping at smooth extrema, as in 1D.

### Lax–Liu configuration 3, 256², t = 0.3 (HLLC + van Leer)

- Matches published structure; symmetric about x = y to ~3e-14.
- Faint stationary lines at x, y ≈ 0.5 in the top-right quadrant: weak waves from
  the 4-significant-figure initial data, not a solver bug.

### Performance (single core, MacBook Air)

- Lax–Liu 3, 256²: 682 steps in 7.66 s → 5.84 million cell-updates/s.

### Kelvin–Helmholtz (McNally et al. 2012 smooth setup), HLLC + van Leer, t = 2

| N   | growth rate (0.6 ≤ t ≤ 1.2) | final mode amplitude |
|-----|-----------------------------|----------------------|
| 64  | 2.41                        | 0.193                |
| 128 | 2.71                        | 0.225                |
| 256 | 2.77                        | 0.228                |

- Converges: differences 0.30 then 0.06 (ratio ≈ 5, consistent with ~2nd order).
  Richardson-extrapolated growth rate ≈ 2.785; 256² within 0.5%.
- Coarse grids under-predict growth (numerical diffusion damps the shear layer).
- Large-scale structure identical at all resolutions; only spiral detail sharpens.
- Single-core performance: 256², 4511 steps in 50 s → 5.9 M cell-updates/s.

### OpenMP strong scaling (MacBook Air, 4 performance + 4 efficiency cores)

Short KH runs; speed-up relative to 1 thread. Parallel results bit-for-bit identical to serial.

| N   | 1 thread (M cells/s) | 2      | 4      | 8      |
|-----|----------------------|--------|--------|--------|
| 128 | 5.38                 | 1.96×  | 3.12×  | 2.67×  |
| 256 | 5.99                 | 1.95×  | 3.08×  | 2.25×  |
| 512 | 6.06                 | 1.97×  | 2.68×  | 2.57×  |

- Near-perfect scaling to 2 threads: parallelisation itself is sound.
- 8 threads slower than 4: static row split + slower efficiency cores; P-core threads wait
  at every loop's end for E-core threads. Use OMP_NUM_THREADS=4 on this machine.
- 512² scales worst at 4 threads: working set (~52 MB) exceeds the shared cache, so the
  update becomes memory-bandwidth-bound (hypothesis; to confirm with profiling).

  ### Concurrent single-threaded jobs (memory-bandwidth test)

| N   | 1 job  | 2 jobs (each) | 4 jobs (each) | 4 jobs total |
|-----|--------|---------------|---------------|--------------|
| 128 | 4.73*  | 5.56          | ~4.5          | 18.0 M/s     |
| 512 | 6.09   | 5.89          | 3.33          | 13.3 M/s     |

*first run, likely clock ramp-up.

- 512²: per-job throughput drops 45% at 4 jobs vs ~19% at 128² → memory-bandwidth-bound
  at large grids (working set ~52 MB/job). Thermal throttling ruled out by the 128² control.
- At 512², 1 job × 4 threads (16.3 M/s) beats 4 jobs × 1 thread (13.3 M/s): threads share
  one working set. At 128² the reverse holds (18.0 vs 16.8).