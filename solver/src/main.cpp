#include "state.hpp"
#include "riemann.hpp"

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
constexpr double T_END = 0.2;
constexpr double CFL   = 0.8;

double cell_centre(int i, double dx) { return X_MIN + (i - NG + 0.5) * dx; }

void apply_transmissive_bcs(std::vector<Conserved>& U) {
    for (int g = 0; g < NG; ++g) {
        U[g]          = U[NG];
        U[NG + N + g] = U[NG + N - 1];
    }
}

double compute_dt(const std::vector<Conserved>& U, double dx) {
    double s_max = 0.0;
    for (int i = NG; i<NG+N; ++i)
    {
        Primitive W = to_primitive(U[i]);
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
    out << "x,rho,u,p\n";
    for (int i = NG; i < NG + N; ++i) {
        Primitive W = to_primitive(U[i]);
        out << cell_centre(i, dx) << ',' << W.rho << ',' << W.u << ',' << W.p << '\n';
    }
}

int main(int argc, char* argv[]) {
    if (argc > 1) N = std::atoi(argv[1]);
    if (N <= 0) {
        std::cerr << "Usage: ./build/euler [N]   (N = number of cells, > 0)\n";
        return 1;
    }

    const std::string scheme = argc > 2 ? argv[2] : "rusanov";
    using FluxFn = Conserved (*)(const Conserved&, const Conserved&);
    FluxFn flux = nullptr;
    if      (scheme == "rusanov") flux = rusanov_flux;
    else if (scheme == "hllc")    flux = hllc_flux;
    else {
        std::cerr << "Unknown scheme '" << scheme << "' (use rusanov or hllc)\n";
        return 1;
    }
    const double dx = (X_MAX - X_MIN) / N;
    std::vector<Conserved> U(N + 2 * NG);
    std::vector<Conserved> F(N + 1);  // F[j] = flux through the left face of physical cell j

    const Primitive left {1.0,   0.0, 1.0};
    const Primitive right{0.125, 0.0, 0.1};
    for (int i = NG; i < NG + N; ++i)
        U[i] = to_conserved(cell_centre(i, dx) < X0 ? left : right);

    const double mass0 = total_mass(U, dx);
    double t = 0.0;
    int step = 0;

    while (t < T_END) {
        apply_transmissive_bcs(U);

        double dt = compute_dt(U, dx);
        if (!(dt > 0.0) || !std::isfinite(dt)) {
            std::cerr << "Bad dt = " << dt << " at step " << step
                      << " (is compute_dt implemented?)\n";
            return 1;
        }
        dt = std::min(dt, T_END - t);

        for (int j = 0; j <= N; ++j)
            F[j] = flux(U[NG + j - 1], U[NG + j]);

        for (int i = NG; i<NG+N;++i)
        {
            int j = i-NG;
            U[i] = U[i] - (dt/dx)*(F[j+1]-F[j]);
        }
        
        t += dt;
        ++step;
        if (step % 50 == 0)
            std::cout << "step " << step << "  t = " << t
                      << "  mass drift = " << total_mass(U, dx) - mass0 << '\n';
    }

    const std::string path = "sod_" + scheme + "_N" + std::to_string(N) + ".csv";
    write_csv(U, dx, path);
    std::cout << "Done: N = " << N << ", " << step << " steps, t = " << t
              << ", wrote " << path << '\n';
    return 0;
}