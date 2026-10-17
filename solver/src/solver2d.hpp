#pragma once
#include "grid2d.hpp"
#include "riemann2d.hpp"
#include "limiters.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace e2d {

// lim slope of each primitive variable
inline Primitive limited_slope(const Primitive& Wm, const Primitive& W0, const Primitive& Wp, Limiter lim) {
    return {lim(W0.rho - Wm.rho, Wp.rho - W0.rho),
            lim(W0.u   - Wm.u,   Wp.u   - W0.u),
            lim(W0.v   - Wm.v,   Wp.v   - W0.v),
            lim(W0.p   - Wm.p,   Wp.p   - W0.p)};
}

// val at a cell edge: W + s * dW / 2, with s = +1 for the right/top edge and -1 for the left/bottom.
inline Primitive edge(const Primitive& W, const Primitive& dW, double s) {
    return {W.rho + 0.5 * s * dW.rho, W.u + 0.5 * s * dW.u, W.v + 0.5 * s * dW.v, W.p + 0.5 * s * dW.p};
}

// 2-order finite-volume solver: MUSCL reconstruction, Riemann solver at every face, SSP-RK2.
// All work arrays are allocated once in the constructor and reused every step.
// Loops are parallelised over rows (j) with OpenMP; results are bit-for-bit identical for any thread count.
class Solver {
public:
    Solver(const Grid& grid, FluxFn flux, Limiter lim, double cfl)
        : g(grid), flux(flux), lim(lim), cfl(cfl),
          W(grid.size()), dW(grid.size()), L(grid.size()), U1(grid.size()),
          G(size_t(grid.nx) * (grid.ny + 1)) {}

    // Largest stable time step (2D CFL condition). Stops program on non-physical state.
    double compute_dt(const std::vector<Conserved>& U) const {
        double sx = 0.0, sy = 0.0;
        bool bad = false;
        #pragma omp parallel for reduction(max : sx, sy) reduction(|| : bad)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                Primitive Wc = to_primitive(U[g.idx(i, j)]);
                if (!(Wc.rho > 0.0) || !(Wc.p > 0.0)) { bad = true; continue; }
                double a = sound_speed(Wc);
                sx = std::max(sx, std::abs(Wc.u) + a);
                sy = std::max(sy, std::abs(Wc.v) + a);
            }
        if (bad) report_bad_cell(U);
        return cfl / (sx / g.dx + sy / g.dy);
    }

    // SSP-RK2 step: U <- U(t + dt).
    void step(std::vector<Conserved>& U, double dt) {
        compute_rhs(U);                                   // L = L(U)
        #pragma omp parallel for
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                int k = g.idx(i, j);
                U1[k] = U[k] + dt * L[k];
            }
        compute_rhs(U1);                                  // L = L(U1)
        #pragma omp parallel for
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                int k = g.idx(i, j);
                U[k] = 0.5 * (U[k] + U1[k] + dt * L[k]);
            }
    }

private:
    Grid g;
    FluxFn flux;
    Limiter lim;
    double cfl;
    std::vector<Primitive> W, dW;   // primitives and slopes (slopes reused for x, then y)
    std::vector<Conserved> L, U1;   // rate of change dU/dt, and the intermediate RK stage
    std::vector<Conserved> G;       // y-face fluxes, nx * (ny + 1)

    // srial search for  first non-physical cell so  error message same for any thread count
    void report_bad_cell(const std::vector<Conserved>& U) const {
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                Primitive Wc = to_primitive(U[g.idx(i, j)]);
                if (!(Wc.rho > 0.0) || !(Wc.p > 0.0)) {
                    std::cerr << "Non-physical state in cell (" << i << ", " << j << "): rho = "
                              << Wc.rho << ", p = " << Wc.p << '\n';
                    std::exit(1);
                }
            }
    }

    // L = -(F_right - F_left)/dx - (G_top - G_bottom)/dy every physical cell
    void compute_rhs(std::vector<Conserved>& U) {
        apply_bcs(g, U);
        #pragma omp parallel for
        for (int k = 0; k < g.size(); ++k) W[k] = to_primitive(U[k]);

        // ---- x-direction ----
        // Each row j only touches its own cells so rows can run on different threads
        #pragma omp parallel for
        for (int j = 0; j < g.ny; ++j) {
            for (int i = 0; i < g.nx; ++i) L[g.idx(i, j)] = Conserved{};
            for (int i = -1; i <= g.nx; ++i)
                dW[g.idx(i, j)] = limited_slope(W[g.idx(i - 1, j)], W[g.idx(i, j)], W[g.idx(i + 1, j)], lim);
            // Face i sits between cells i-1 and i
            for (int i = 0; i <= g.nx; ++i) {
                int kl = g.idx(i - 1, j), kr = g.idx(i, j);
                Conserved F = flux(to_conserved(edge(W[kl], dW[kl], +1.0)),
                                   to_conserved(edge(W[kr], dW[kr], -1.0)));
                if (i > 0)    L[kl] = L[kl] - (1.0 / g.dx) * F;   // leaves cell i-1 through its right face
                if (i < g.nx) L[kr] = L[kr] + (1.0 / g.dx) * F;   // enters cell i through its left face
            }
        }

        // ---- y-direction ----
        // A y-face touches two different rows, so two threads could update the same cell at once
        // Instead: (1) compute every face flux into G, (2) let each row collect its own two fluxes
        #pragma omp parallel for
        for (int j = -1; j <= g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
                dW[g.idx(i, j)] = limited_slope(W[g.idx(i, j - 1)], W[g.idx(i, j)], W[g.idx(i, j + 1)], lim);
        #pragma omp parallel for
        for (int j = 0; j <= g.ny; ++j)                    // face j sits between cells j-1 and j
            for (int i = 0; i < g.nx; ++i) {
                int kb = g.idx(i, j - 1), kt = g.idx(i, j);
                G[size_t(j) * g.nx + i] = flux_y_face(flux, to_conserved(edge(W[kb], dW[kb], +1.0)),
                                                            to_conserved(edge(W[kt], dW[kt], -1.0)));
            }
        #pragma omp parallel for
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                int k = g.idx(i, j);
                L[k] = L[k] + (1.0 / g.dy) * G[size_t(j) * g.nx + i];         // in through the bottom face
                L[k] = L[k] - (1.0 / g.dy) * G[size_t(j + 1) * g.nx + i];     // out through the top face
            }
    }
};

}