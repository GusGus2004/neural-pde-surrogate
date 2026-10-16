// Checks for the 2D state and Riemann solvers. Run after any change to headers.
#include "riemann.hpp"      // 1D for comp
#include "riemann2d.hpp"
#include <cmath>
#include <cstdio>

int failures = 0;

void check(const char* name, double a, double b, double tol = 1e-13) {
    bool ok = std::abs(a - b) <= tol * (1.0 + std::abs(b));
    std::printf("%-46s %s  (%.15g vs %.15g)\n", name, ok ? "PASS" : "FAIL", a, b);
    if (!ok) ++failures;
}

int main() {
    //few left/right pairs: Sod, a moving shock, two colliding streams
    const double states[3][6] = {{1.0, 0.0, 1.0, 0.125, 0.0, 0.1},
                                 {1.0, 0.75, 1.0, 0.125, 0.0, 0.1},
                                 {1.0, 2.0, 0.4, 1.0, -2.0, 0.4}};
    for (const auto& s : states) {
        // 1. with v = 0, the 2D x-flux must equal the 1D flux exactly.
        e2d::Conserved L = e2d::to_conserved({s[0], s[1], 0.0, s[2]});
        e2d::Conserved R = e2d::to_conserved({s[3], s[4], 0.0, s[5]});
        Conserved L1 = to_conserved({s[0], s[1], s[2]});
        Conserved R1 = to_conserved({s[3], s[4], s[5]});
        e2d::Conserved F2 = e2d::hllc_flux(L, R);
        Conserved F1 = hllc_flux(L1, R1);
        check("HLLC 2D (v=0) vs 1D: mass",     F2.rho, F1.rho);
        check("HLLC 2D (v=0) vs 1D: x-mom",    F2.mx,  F1.mom);
        check("HLLC 2D (v=0) vs 1D: energy",   F2.E,   F1.E);
        check("HLLC 2D (v=0): y-mom flux is 0", F2.my, 0.0);

        // 2. along y must give the same flux, with momenta swapped.
        e2d::Conserved Ly = e2d::to_conserved({s[0], 0.0, s[1], s[2]});
        e2d::Conserved Ry = e2d::to_conserved({s[3], 0.0, s[4], s[5]});
        e2d::Conserved Fy = e2d::flux_y_face(e2d::hllc_flux, Ly, Ry);
        check("HLLC y-face vs x-face: mass",   Fy.rho, F2.rho);
        check("HLLC y-face vs x-face: y-mom",  Fy.my,  F2.mx);
        check("HLLC y-face vs x-face: energy", Fy.E,   F2.E);
    }

    // 3. Consistency: equal left and right states give the physical flux.
    e2d::Conserved U = e2d::to_conserved({1.3, 0.4, -0.7, 2.1});
    e2d::Conserved F = e2d::hllc_flux(U, U), Fp = e2d::flux_x(U);
    check("HLLC(U, U) == F(U): x-mom", F.mx, Fp.mx);
    check("HLLC(U, U) == F(U): y-mom", F.my, Fp.my);
    check("HLLC(U, U) == F(U): energy", F.E, Fp.E);

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}