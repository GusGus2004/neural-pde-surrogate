#include "state.hpp"
#include "riemann.hpp"
#include "reconstruct.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

int N = 200;  // physical cells (default and can be overridden from the command)
constexpr int    NG    = 2;    // ghost cells per side -> change to 2 for MUSCL
constexpr double X_MIN = 0.0, X_MAX = 1.0, X0 = 0.5;
double T_END = 0.2;
constexpr double CFL   = 0.5;
bool PERIODIC = false;
constexpr double PI = 3.14159265358979323846;

using FluxFn = Conserved (*)(const Conserved&, const Conserved&);

double cell_centre(int i, double dx) { return X_MIN + (i - NG + 0.5) * dx; }

void apply_bcs(std::vector<Conserved>& U) {
    for (int g = 0; g < NG; ++g) {
        if (PERIODIC) {                     // wrap around: ghosts copy the opposite end
            U[g]          = U[N + g];
            U[NG + N + g] = U[NG + g];
        } else {                            // transmissive: ghosts copy the nearest cell
            U[g]          = U[NG];
            U[NG + N + g] = U[NG + N - 1];
        }
    }
}
double compute_dt(const std::vector<Conserved>& U, double dx) {
    double s_max = 0.0;
    for (int i = NG; i<NG+N; ++i)
    {
        Primitive W = to_primitive(U[i]);
        if (!(W.rho > 0.0) || !(W.p > 0.0)) {
            std::cerr << "Non-physical state in cell " << i - NG
                      << ": rho = " << W.rho << ", p = " << W.p << '\n';
            std::exit(1);
        }
        double s = std::abs(W.u)+sound_speed(W);
        s_max = std::max(s_max,s);
    }
    return CFL * dx / s_max;
}

double total_mass(const std::vector<Conserved>& U, double dx) {
    double m = 0.0;
    for (int i = NG; i < NG + N; ++i) m += U[i].rho * dx;
    return m;
}

void write_csv(const std::vector<Conserved>& U, double dx, const std::string& path) {
    std::ofstream out(path);
    out.precision(12);
    out << "x,rho,u,p\n";
    for (int i = NG; i < NG + N; ++i) {
        Primitive W = to_primitive(U[i]);
        out << cell_centre(i, dx) << ',' << W.rho << ',' << W.u << ',' << W.p << '\n';
    }
}

void compute_fluxes(std::vector<Conserved>& U, std::vector<Conserved>& F,
                    FluxFn flux, Limiter lim) {
    apply_bcs(U);
    const int M = N + 2 * NG;
    std::vector<Primitive> W(M), dW(M);
    for (int i = 0; i < M; ++i) W[i] = to_primitive(U[i]);
    for (int i = 1; i < M - 1; ++i) dW[i] = limited_slope(W[i - 1], W[i], W[i + 1], lim);

    for (int j = 0; j <= N; ++j) {
        int iL = NG + j - 1, iR = NG + j;   // cells either side of face j
        Primitive WL{W[iL].rho + 0.5 * dW[iL].rho,   // right edge of the left cell
                     W[iL].u   + 0.5 * dW[iL].u,
                     W[iL].p   + 0.5 * dW[iL].p};
        Primitive WR{W[iR].rho - 0.5 * dW[iR].rho,   
                     W[iR].u   - 0.5 * dW[iR].u,
                     W[iR].p   - 0.5 * dW[iR].p};
        F[j] = flux(to_conserved(WL), to_conserved(WR));
    }
}

void euler_step(const std::vector<Conserved>& U_in, const std::vector<Conserved>& F,
                double dt, double dx, std::vector<Conserved>& U_out) {
    for (int i = NG; i < NG + N; ++i) {
        int j = i - NG;
        U_out[i] = U_in[i] - (dt / dx) * (F[j + 1] - F[j]);
    }
}

Primitive initial_state(double x, const std::string& test) {
    if (test == "sine") return {1.0 + 0.2 * std::sin(2.0 * PI * x), 1.0, 1.0};
    return x < X0 ? Primitive{1.0, 0.0, 1.0} : Primitive{0.125, 0.0, 0.1};   // Sod
}


int main(int argc, char* argv[]) {
    if (argc > 1) N = std::atoi(argv[1]);
    if (N <= 0) {
        std::cerr << "Usage: ./build/euler [N]   (N = number of cells, > 0)\n";
        return 1;
    }

    const std::string scheme = argc > 2 ? argv[2] : "rusanov";
    FluxFn flux = nullptr;
    if      (scheme == "rusanov") flux = rusanov_flux;
    else if (scheme == "hllc")    flux = hllc_flux;
    else {
        std::cerr << "Unknown scheme '" << scheme << "' (use rusanov or hllc)\n";
        return 1;
    }

    const std::string lim_name = argc > 3 ? argv[3] : "none";
    Limiter lim = nullptr;
    if      (lim_name == "none")      lim = no_slope;
    else if (lim_name == "minmod")    lim = minmod;
    else if (lim_name == "vanleer")   lim = van_leer;
    else if (lim_name == "unlimited") lim = unlimited;
    else {
        std::cerr << "Unknown limiter '" << lim_name
                  << "' (use none, minmod, vanleer or unlimited)\n";
        return 1;
    }

    const std::string test = argc > 4 ? argv[4] : "sod";
    if (test == "sine") { PERIODIC = true; T_END = 1.0; }
    else if (test != "sod") {
        std::cerr << "Unknown test '" << test << "' (use sod or sine)\n";
        return 1;
    }

    const double dx = (X_MAX - X_MIN) / N;
    std::vector<Conserved> U(N + 2 * NG);
    std::vector<Conserved> F(N + 1);  // F[j] = flux through the left face of physical cell j

    for (int i = NG; i < NG + N; ++i)
        U[i] = to_conserved(initial_state(cell_centre(i, dx), test));

    const double mass0 = total_mass(U, dx);
    double t = 0.0;
    int step = 0;

    std::vector<Conserved> U1(N + 2 * NG), U2(N + 2 * NG);

    while (t < T_END) {
        apply_bcs(U);
        double dt = compute_dt(U, dx);
        if (!(dt > 0.0) || !std::isfinite(dt)) {
            std::cerr << "Bad dt = " << dt << " at step " << step << '\n';
            return 1;
        }
        dt = std::min(dt, T_END - t);

        compute_fluxes(U, F, flux, lim);
        euler_step(U, F, dt, dx, U1);
        compute_fluxes(U1, F, flux, lim);
        euler_step(U1, F, dt, dx, U2);
        for (int i = NG; i < NG + N; ++i)
            U[i] = 0.5 * (U[i] + U2[i]);

        t += dt;
        ++step;
        if (step % 50 == 0)
            std::cout << "step " << step << "  t = " << t
                      << "  mass drift = " << total_mass(U, dx) - mass0 << '\n';
    }

    const std::string path = test + "_" + scheme + "_" + lim_name
                           + "_N" + std::to_string(N) + ".csv";
    write_csv(U, dx, path);
    std::cout << "Done: N = " << N << ", " << step << " steps, t = " << t
              << ", wrote " << path << '\n';
    return 0;
}