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


  ### Per-phase profile, 512², KH (1 vs 4 threads)

| Phase                 | 1 thread (s) | 4 threads (s) | speed-up |
|-----------------------|--------------|---------------|----------|
| x: slopes + fluxes    | 4.276        | 1.908         | 2.24×    |
| y: face fluxes        | 3.000        | 1.325         | 2.26×    |
| y: slopes             | 0.847        | 0.385         | 2.20×    |
| y: add G into L       | 0.540        | 0.251         | 2.15×    |
| compute_dt            | 0.279        | 0.135         | 2.07×    |
| RK updates            | 0.406        | 0.260         | 1.56×    |
| U → W                 | 0.194        | 0.136         | 1.43×    |
| total                 | 9.55         | 4.42          | 2.16×    |

- Flux computation (Riemann solver) is ~75% of run time.
- Compute-heavy and streaming phases slow down almost equally at 4 threads → the main
  limit is not memory bandwidth but thread placement/clock on macOS (no thread pinning,
  P/E cores, fanless). Pure-streaming phases show an extra bandwidth penalty but are ~6% of time.
- Run-to-run variance ~25% at 4 threads on the Mac; use medians of repeats, and the
  cluster (pinned threads) for definitive scaling numbers.
- Optimisation targets: inline flux/limiter (templates instead of function pointers);
  avoid primitive↔conserved round trips at faces.

  ## 4. Dataset generation (local batch v0)

MacBook Air, 512² solver, saved at 128² and 64², 4 OpenMP threads, 51 frames per trajectory.

| Family  | Runs | Total time | Per run | 
|---------|------|------------|---------|
| riemann | 400  | 4.5 h      | 41 s    |
| blast   | 400  | 2.9 h      | 26 s    |
| kh      | 50   | 1.3 h      | 93 s    |

850 trajectories, 0 failures, 5.1 GB. Indices 0–399 / 0–399 / 0–49; the cluster run will continue from there.


# Dataset log

## v0 — local batch (MacBook Air), 18–19 Oct 2026

- riemann 0–399, blast 0–399, kh 0–49 (850 trajectories, 0 failures, 5.1 GB)
- solver 512², stored 128² and 64², 51 frames, Δt = 0.005, HLLC + van Leer, CFL 0.5, 4 threads
- splits (data/splits.py): 640 train / 80 val / 80 test / 50 ood

- packed (data/pack.py): 128² and 64², 5.6 GB; all checks passed (max mass/energy drift 4.5e-8);
  min rho 0.003, min p 0.0009 (near-vacuum cells from rarefactions/blast cavities)

**Provenance note:** all files record `git_hash = ee12184-dirty`. The uncommitted changes at
generation time were only `docs/figures/*.png` (staged README figures) and `experiments/results.md`,
verified with `git diff --stat ee12184`. No solver or data code differed from `ee12184`, so the data
is exactly what that commit produces.

# Phase 2:

## Model Cost:

- ResNet (w=64, d=8, 591k params): ~4.8 GFLOP forward per 64^2 sample, ~15 GFLOP fwd+bwd.
Batch-32 step: ~460 GFLOP. JAX CPU conv ~110 GFLOP/s (matmul 215) -> 6.0 s/step measured on the M-series Air.
=> compute-bound on CPU; Mac for development (w=32, d=4), GPUs for training.

## ResNet 1st Run:

- ResNet w=32 d=4 (74k params), Mac CPU, 2000 steps x batch 32 (2 epochs), lr 1e-3 cosine, 1934 s:
one-step val MSE 2.19e-3 (normalised) = 0.037 x persistence; train 2.04e-3 (no overfitting).
Overfit check (train traj 0 only, 300 steps): 0.060 x persistence.

- Rollout (val, 80 traj, from frame 0), RMSE/std mean over vars [persistence]:
t=1 0.041 [0.143]  t=10 0.198 [0.699]  t=25 0.400 [0.912]  t=50 6.69 [1.007]
-> beats persistence ~2-4x up to t=25, then diverges. Mass drift up to 1e2 at t=50.
Negative rho/p in 74/80 trajectories, from step 1. One-step error (0.037 x persistence) hides the instability.

- Median RMSE/std at t=50: 0.587 (persistence ~1.0); 20/80 trajectories blow up (error > 2), onsets spread
over steps 26-50 (8/40 blast, 12/40 riemann). Mean (6.7) dominated by the blow-ups.

- JAX CPU: lax.scan over conv layers in the backward pass was 18x slower than an unrolled Python loop
(65 s vs 3.5 s per step, w32 d4, k=3). Using the Python loop for small k. Re-check on GPU.

- Unrolled fine-tuning from resnet_w32_d4_mac, matched wall-clock (~15 min each), val rollouts, last checkpoint:
                     blown-up  median t50  mean t50  mass drift  t=1
baseline              20/80     0.587       6.69     1.0e2      0.041
+ one-step 1000 st    21/80     0.518      50.0      6.2e2      0.038
+ multistep k=3 300   19/80     0.548      20.1      1.9e2      0.042
+ pushforward k=3 600  7/80     0.404       0.86     6.3        0.045
Pushforward vs control: Fisher p = 0.006. Multistep vs control: p = 0.86. Single seed each.
More one-step training improves one-step error but not stability. Negative rho/p from step 1 persists (66/80).

- Positivity diagnosis (ft_pf3, val): all 7 blow-ups non-physical by step 5 (4 by step 2); 14 trajectories never
non-physical, 0 blow up. Bad cells at step 5: median 31 (blown) vs 1 (stable); Spearman(bad@5, err50) = +0.51.
Mostly density (2533 cell-steps) not pressure (406). Bad cells: true rho ~0.11 but p ~0.95 (hot rarefied blast
interiors / rarefactions), not near-vacuum (0% with p < 0.05). Kinetic share 0.05 -> not cancellation in p.

- Positive (log rho, u, v, log p) wrapper, same recipe (2000 one-step + 600 pushforward k=3), val rollouts:
                  blown-up  non-phys  mean t50  median t50  median t10  one-step   max mass drift
ft_pf3 (cons.)     7/80      66/80     0.86      0.404       0.151      0.038      6.3
pos_pf3 (log)      0/80       0/80     0.62      0.528       0.267      0.056      0.70
Positivity removes ALL blow-ups (diagnosis confirmed causally) but costs typical accuracy (~30% median t50,
~1.8x at t10; blasts worst) and one-step accuracy already before fine-tuning (0.049 vs 0.037 at 2000 steps).
Mass not conserved (up to 70%). Candidate middle way: conserved model + mass-conserving positivity limiter.

## UNET Runs:

- U-Net w16 L3 (516k params, 0.54 s/step vs ResNet 0.85 on Mac), same recipe (2000 one-step + 600 pf k=3):
                     blown  non-phys  med t10  med t50  mean t50  one-step  mass drift
ResNet cons + pf     7/80   66/80     0.151    0.404    0.86      0.038     6.3
ResNet pos  + pf     0/80    0/80     0.267    0.528    0.62      0.056     0.70
U-Net  cons + pf     0/80   79/80     0.208    0.416    0.51      0.077     0.54
U-Net  pos  + pf     1/80    0/80     0.248    0.608    0.70      0.097     0.38
- U-Net cons: 0 blow-ups despite 79/80 non-physical (vs ResNet 7/80, Fisher p = 0.014): negative cells are
  not sufficient for blow-up; they were the trigger for the local ResNet only.
- U-Net: 2x worse one-step, worse early, but slower error growth -> equal median, best mean at t50.
- Positivity costs accuracy in both architectures. U-Net had ~half the compute at equal steps. Single seed.

## FNO:

- FNO w32 d4 m16 (4.2M params, ~ResNet cost/step), same recipe, val rollouts:
                blown  non-phys  med t10  med t50  mean t50  median mass drift t50
ResNet          7/80   66/80     0.151    0.404    0.863     4.5e-2
U-Net           0/80   79/80     0.208    0.416    0.507     3.7e-2
FNO             3/80   71/80     0.203    0.454    0.634     4.3e-2
ResNet pos      0/80    0/80     0.267    0.528    0.621     3.6e-2
U-Net pos       1/80    0/80     0.248    0.608    0.695     4.7e-2
FNO pos         1/80    0/80     0.288    0.571    0.680     6.9e-2
- Medians within ~10% (single seed): no architecture clearly best at this scale. Global models (U-Net, FNO)
  blew up less than the local ResNet. FNO slightly worse on blasts (sharp curved shocks) -- to be tested.
- Positivity costs accuracy in all 3 architectures. Mass drift ~4-7% median for all models.
- FNO likely under-trained (4.2M params, 2600 steps); no claim of FNO inferiority.

- JAX CPU: lax.scan rollouts of conv models 23x slower than a Python loop over a jitted step
(ResNet w32 d4: 8.8 vs 0.39 s per 50-step trajectory); identical outputs. Fixed in rollout.py.

# Phase 3:
## Surrogates vs Solver:

- Accuracy vs cost (val, Mac CPU; solver 1 thread, surrogates all cores, best of 3, batch 80):
                 s/traj b1  s/traj b80  median err t50
solver 32^2        0.042       -           0.243
solver 64^2        0.239       -           0.137
solver 128^2       1.432       -           0.073
ResNet             0.336     0.287         0.404
U-Net              0.288     0.231         0.416
FNO                0.345     0.203         0.454
ResNet pos         0.323     0.290         0.528
All surrogates dominated by the solver: same cost as 64^2 solver with ~3x its error; 32^2 solver is both
cheaper (6x) and more accurate. Batching on CPU gives only 1.2-1.7x. Crossing the solver line needs ~3x
better accuracy at fixed cost (target median t50 < ~0.13), or ~10x lower cost at fixed accuracy.

## CAMBRIDGE GPUs (Lovelace):

lovelace (NVIDIA A30, JAX 0.6.2), training step batch 32 at 64^2:
ResNet w64 d8: 46 ms (Mac CPU 5971 ms, ~130x; ~10 TFLOP/s ~ A30 fp32 peak).
ResNet w32 d4 / U-Net w16 L3: ~20 ms (~40x / ~25x): small models are overhead-bound on the GPU.


- seeds_small on lovelace (A30): Mac recipe, 3 seeds per model, val rollouts (mean ± sd over seeds):
| | seeds | blown | non-phys | med t10 | med t50 | mean t50 | mass t50 | one-step |
| FNO    | 3 | 1.0 ± 1.7 | 71.3 ± 1.5 | 0.213 ± 0.014 | 0.496 ± 0.037 | 0.585 ± 0.047 | 4.7e-02 ± 1.6e-02 | 0.0762 ± 0.0036 |
| ResNet | 3 | 2.3 ± 2.5 | 67.3 ± 2.3 | 0.152 ± 0.002 | 0.385 ± 0.028 | 0.611 ± 0.236 | 3.5e-02 ± 9.3e-03 | 0.0380 ± 0.0008 |
| U-Net  | 3 | 0.0 ± 0.0 | 78.3 ± 0.6 | 0.202 ± 0.005 | 0.433 ± 0.005 | 0.515 ± 0.013 | 3.2e-02 ± 8.7e-03 | 0.0717 ± 0.0042 |
Seed 0 reproduces the Mac runs (medians 0.454 / 0.405 / 0.431 vs 0.454 / 0.404 / 0.416).
- ResNet best median at every seed (0.354-0.405 vs U-Net 0.431-0.439, FNO 0.454-0.522).
- U-Net 0 blow-ups in all seeds; ResNet 5/2/0, FNO 3/0/0 -> single-seed blow-up counts are unreliable.
  Pooled over seeds: ResNet 7/240 vs U-Net 0/240 (Fisher p = 0.015, trajectories shared across seeds).
- Trade-off: local ResNet most accurate but occasionally unstable; U-Net slightly less accurate, robust.
- Whole sweep: 29 min on one A30 (each run ~50 min on the Mac CPU).


## Sweep Scale:

- scale sweep on lovelace (A30): large models, 20000 one-step + 3000 pushforward steps, 3 seeds, val rollouts:
| ResNet large (w64 d8)   | 3 | blown 0.0 ± 0.0 | non-phys 27.3 ± 3.1 | med t10 0.042 ± 0.000 | med t50 0.119 ± 0.003 | mean t50 0.179 ± 0.004 | mass 4.3e-03 ± 7.8e-04 | one-step 0.0060 ± 0.0000 |
| U-Net large (w32 L3)    | 3 | blown 0.0 ± 0.0 | non-phys 38.3 ± 2.3 | med t10 0.076 ± 0.003 | med t50 0.188 ± 0.003 | mean t50 0.270 ± 0.005 | mass 7.5e-03 ± 1.5e-03 | one-step 0.0140 ± 0.0001 |
| FNO large (w64 d4 m16)  | 3 | blown 0.0 ± 0.0 | non-phys 53.0 ± 2.0 | med t10 0.113 ± 0.005 | med t50 0.300 ± 0.010 | mean t50 0.380 ± 0.004 | mass 2.0e-02 ± 1.5e-03 | one-step 0.0312 ± 0.0002 |
- Large ResNet median t50 0.119 < 64^2 solver 0.137 (all 3 seeds); at t10 0.042 vs 0.106. Mean t50 0.179 > solver 0.165.
- Scale: 3.2x (ResNet), 2.3x (U-Net), 1.7x (FNO) better medians; 0 blow-ups in all 9 runs; fewer non-physical.
- Ranking unchanged; FNO gains least from scale. Model size and training length changed together (confounded).
- Still to do: cost of the large models (A30 and CPU); final numbers on the untouched test split.

## Costs:

Cost on lovelace A30 (JAX 0.6.2), 50-step rollout, s per trajectory (batch 1 / batch 80), seed-0 model:
ResNet large 0.066 / 0.018 (med t50 0.121); U-Net large 0.063 / 0.0070 (0.187);
FNO large 0.054 / 0.0072 (0.311); ResNet small 0.033 / 0.0046 (0.405).
Solver, 1 Mac CPU core: 32^2 0.042 s (0.243), 64^2 0.24 s (0.137), 128^2 1.43 s (0.073); ~0.35 s at 0.12 (interp.).
=> ResNet large: more accurate than 64^2 solver and ~13x faster (~19x at matched accuracy) vs ONE CPU core.
   vs 32 cores in parallel (ideal scaling) roughly break-even -- to be measured on lovelace's own CPUs.
   Training cost (~20 GPU-min + ~9 h of 512^2 data generation) and growing error beyond 50 steps not included.