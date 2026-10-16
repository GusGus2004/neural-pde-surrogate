// Checks for 2D solver: free-stream pres, x/y symmetry, conservation
#include "solver2d.hpp"
#include <cmath>
#include <cstdio>

using namespace e2d;

int failures = 0;

void check(const char* name, bool ok, double value) {
    std::printf("%-58s %s  (%.3g)\n", name, ok ? "PASS" : "FAIL", value);
    if (!ok) ++failures;
}

double total(const Grid& g, const std::vector<Conserved>& U, double Conserved::*field) {
    double s = 0.0;
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) s += U[g.idx(i, j)].*field;
    return s * g.dx * g.dy;
}

// run to time t_end return the number of steps taken
int run(Solver& solver, std::vector<Conserved>& U, double t_end) {
    double t = 0.0;
    int steps = 0;
    while (t < t_end) {
        double dt = std::min(solver.compute_dt(U), t_end - t);
        solver.step(U, dt);
        t += dt;
        ++steps;
    }
    return steps;
}

int main() {
    // 1. Free-stream pres: uniform moving state must stay uniform.
    {
        Grid g = make_grid(16, 12, 0.0, 1.0, 0.0, 1.0, BC::Periodic, BC::Periodic);
        std::vector<Conserved> U(g.size());
        const Conserved U0 = to_conserved({1.3, 0.7, -0.4, 2.0});
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) U[g.idx(i, j)] = U0;
        Solver s(g, hllc_flux, van_leer, 0.5);
        run(s, U, 0.5);
        double err = 0.0;
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const Conserved& a = U[g.idx(i, j)];
                err = std::max({err, std::abs(a.rho - U0.rho), std::abs(a.mx - U0.mx),
                                     std::abs(a.my - U0.my), std::abs(a.E - U0.E)});
            }
        check("uniform flow stays uniform (max change)", err < 1e-13, err);
    }

    // 2. Sod along x on a 200 x 4 grid, along y on a 4 x 200 grid
    const int N = 200, M = 4;
    Grid gx = make_grid(N, M, 0.0, 1.0, 0.0, double(M) / N, BC::Transmissive, BC::Transmissive);
    Grid gy = make_grid(M, N, 0.0, double(M) / N, 0.0, 1.0, BC::Transmissive, BC::Transmissive);
    std::vector<Conserved> Ux(gx.size()), Uy(gy.size());
    const Primitive left{1.0, 0.0, 0.0, 1.0}, right{0.125, 0.0, 0.0, 0.1};
    for (int j = 0; j < gx.ny; ++j)
        for (int i = 0; i < gx.nx; ++i)
            Ux[gx.idx(i, j)] = to_conserved(gx.xc(i) < 0.5 ? left : right);
    for (int j = 0; j < gy.ny; ++j)
        for (int i = 0; i < gy.nx; ++i)
            Uy[gy.idx(i, j)] = to_conserved(gy.yc(j) < 0.5 ? left : right);

    double mass0 = total(gx, Ux, &Conserved::rho), energy0 = total(gx, Ux, &Conserved::E);
    Solver sx(gx, hllc_flux, van_leer, 0.5), sy(gy, hllc_flux, van_leer, 0.5);
    int nsx = run(sx, Ux, 0.2), nsy = run(sy, Uy, 0.2);
    check("same number of steps along x and along y", nsx == nsy, nsx);

    // 2a. Every row of x-run is identical (has no y-dependence)
    double row_diff = 0.0;
    for (int j = 1; j < M; ++j)
        for (int i = 0; i < N; ++i)
            row_diff = std::max(row_diff, std::abs(Ux[gx.idx(i, j)].rho - Ux[gx.idx(i, 0)].rho));
    check("x-run: all rows identical", row_diff == 0.0, row_diff);

    // 2b. no y-momentum is created
    double my_max = 0.0;
    for (int j = 0; j < M; ++j)
        for (int i = 0; i < N; ++i) my_max = std::max(my_max, std::abs(Ux[gx.idx(i, j)].my));
    check("x-run: y-momentum stays exactly zero", my_max == 0.0, my_max);

    // 2c. y-run is the x-run transposed, with the momenta swapped
    double sym = 0.0;
    for (int j = 0; j < M; ++j)
        for (int i = 0; i < N; ++i) {
            const Conserved& a = Ux[gx.idx(i, j)];
            const Conserved& b = Uy[gy.idx(j, i)];
            sym = std::max({sym, std::abs(a.rho - b.rho), std::abs(a.mx - b.my),
                                 std::abs(a.my - b.mx), std::abs(a.E - b.E)});
        }
    check("y-run equals x-run transposed (max difference)", sym == 0.0, sym);

    // 2d. Mass and energy conserved (no wave reaches boundary by t = 0.2)
    double dm = std::abs(total(gx, Ux, &Conserved::rho) - mass0);
    double dE = std::abs(total(gx, Ux, &Conserved::E) - energy0);
    check("x-run: mass conserved", dm < 1e-13, dm);
    check("x-run: energy conserved", dE < 1e-13, dE);

    // 2e. Plateau values between the contact and the shock match exact solution
    //     (rho* right of contact = 0.26557, p* = 0.30313, u* = 0.92745)
    Primitive Wc = to_primitive(Ux[gx.idx(int(0.78 * N), 0)]);
    check("x-run: rho at x = 0.78 within 1e-3 of 0.26557", std::abs(Wc.rho - 0.26557) < 1e-3, Wc.rho);
    check("x-run: p   at x = 0.78 within 1e-3 of 0.30313", std::abs(Wc.p - 0.30313) < 1e-3, Wc.p);
    check("x-run: u   at x = 0.78 within 1e-3 of 0.92745", std::abs(Wc.u - 0.92745) < 1e-3, Wc.u);

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}