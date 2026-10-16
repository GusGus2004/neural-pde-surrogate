// 2D Euler solver: sets up test problem, runs writes HDF5 snapshots.
//
// Usage: ./build/euler2d --test lax_liu3 --n 256 --scheme hllc --limiter vanleer --nout 10 --out runs
#include "solver2d.hpp"
#include "io_hdf5.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <cmath>

using namespace e2d;

std::map<std::string, std::string> parse_args(int argc, char* argv[]) {
    std::map<std::string, std::string> a = {
        {"test", "lax_liu3"}, {"n", "256"}, {"scheme", "hllc"}, {"limiter", "vanleer"},
        {"cfl", "0.5"}, {"nout", "10"}, {"out", "runs"}};
    for (int k = 1; k < argc; ++k) {
        std::string key = argv[k];
        if (key.rfind("--", 0) != 0 || k + 1 >= argc) {
            std::cerr << "Expected '--key value', got '" << key << "'\n";
            std::exit(1);
        }
        key = key.substr(2);
        if (!a.count(key)) {
            std::cerr << "Unknown option --" << key << '\n';
            std::exit(1);
        }
        a[key] = argv[++k];
    }
    return a;
}


struct Problem {
    Grid grid;
    double t_end = 0.0;
};

// sets up grid and fills U with the initial condition for the named test
Problem setup(const std::string& test, int n, std::vector<Conserved>& U) {
    Problem pr;
    if (test == "sod_x" || test == "sod_y") {
        const int m = 4;
        bool along_x = (test == "sod_x");
        pr.grid = along_x ? make_grid(n, m, 0.0, 1.0, 0.0, double(m) / n, BC::Transmissive, BC::Transmissive)
                          : make_grid(m, n, 0.0, double(m) / n, 0.0, 1.0, BC::Transmissive, BC::Transmissive);
        pr.t_end = 0.2;
        U.assign(pr.grid.size(), Conserved{});
        const Primitive left{1.0, 0.0, 0.0, 1.0}, right{0.125, 0.0, 0.0, 0.1};
        const Grid& g = pr.grid;
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                double s = along_x ? g.xc(i) : g.yc(j);
                U[g.idx(i, j)] = to_conserved(s < 0.5 ? left : right);
            }
    } else if (test == "lax_liu3") {
        // Lax & Liu configuration 3: four quadrants, four shocks meeting at the centre
        pr.grid = make_grid(n, n, 0.0, 1.0, 0.0, 1.0, BC::Transmissive, BC::Transmissive);
        pr.t_end = 0.3;
        U.assign(pr.grid.size(), Conserved{});
        const Primitive q1{1.5,    0.0,   0.0,   1.5  };   // top right
        const Primitive q2{0.5323, 1.206, 0.0,   0.3  };   // top left
        const Primitive q3{0.138,  1.206, 1.206, 0.029};   // bottom left
        const Primitive q4{0.5323, 0.0,   1.206, 0.3  };   // bottom right
        const Grid& g = pr.grid;
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                bool right = g.xc(i) > 0.5, top = g.yc(j) > 0.5;
                const Primitive& W = top ? (right ? q1 : q2) : (right ? q4 : q3);
                U[g.idx(i, j)] = to_conserved(W);
            }
    }   else if (test == "sine2d") {
        // smooth wave advected diagonally across periodic box
        pr.grid = make_grid(n, n, 0.0, 1.0, 0.0, 1.0, BC::Periodic, BC::Periodic);
        pr.t_end = 1.0;
        U.assign(pr.grid.size(), Conserved{});
        const double PI = 3.14159265358979323846;
        const Grid& g = pr.grid;
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                double rho = 1.0 + 0.2 * std::sin(2.0 * PI * (g.xc(i) + g.yc(j)));
                U[g.idx(i, j)] = to_conserved({rho, 1.0, 1.0, 1.0});
            }
        }
    else {
        std::cerr << "Unknown test '" << test << "' (use sod_x, sod_y, lax_liu3 or sine2d)\n";
        std::exit(1);
    }
    return pr;
}


int main(int argc, char* argv[]) {
    auto a = parse_args(argc, argv);
    const std::string test = a["test"], scheme = a["scheme"], lim_name = a["limiter"];
    const int n = std::atoi(a["n"].c_str()), nout = std::atoi(a["nout"].c_str());
    const double cfl = std::atof(a["cfl"].c_str());
    if (n <= 0 || nout <= 0 || !(cfl > 0.0)) { std::cerr << "n, nout and cfl must be positive\n"; return 1; }

    FluxFn flux = nullptr;
    if      (scheme == "rusanov") flux = rusanov_flux;
    else if (scheme == "hllc")    flux = hllc_flux;
    else { std::cerr << "Unknown scheme '" << scheme << "' (use rusanov or hllc)\n"; return 1; }

    Limiter lim = nullptr;
    if      (lim_name == "none")      lim = no_slope;
    else if (lim_name == "minmod")    lim = minmod;
    else if (lim_name == "vanleer")   lim = van_leer;
    else if (lim_name == "unlimited") lim = unlimited;
    else { std::cerr << "Unknown limiter '" << lim_name << "'\n"; return 1; }

    std::vector<Conserved> U;
    Problem pr = setup(test, n, U);
    const Grid& g = pr.grid;
    Solver solver(g, flux, lim, cfl);

    const std::string dir = a["out"] + "/" + test + "_" + scheme + "_" + lim_name + "_N" + std::to_string(n);
    std::filesystem::create_directories(dir);
    RunInfo info{test, scheme, lim_name, cfl, 0};
    auto snap_path = [&](int k) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "/snap_%04d.h5", k);
        return dir + buf;
    };

    std::printf("%s: %d x %d cells, t_end = %g, %s + %s, CFL %g\n",
                test.c_str(), g.nx, g.ny, pr.t_end, scheme.c_str(), lim_name.c_str(), cfl);

    double t = 0.0;
    int step = 0;
    write_snapshot(snap_path(0), g, U, t, info);
    auto t0 = std::chrono::steady_clock::now();

    for (int k = 1; k <= nout; ++k) {
        const double t_next = pr.t_end * k / nout;    
        while (t < t_next) {
            double dt = std::min(solver.compute_dt(U), t_next - t);
            solver.step(U, dt);
            t += dt;
            ++step;
        }
        info.step = step;
        write_snapshot(snap_path(k), g, U, t, info);
        std::printf("  snapshot %2d/%d  t = %.4f  step %d\n", k, nout, t, step);
    }

    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double mcells = double(g.nx) * g.ny * step / secs / 1e6;
    std::printf("Done: %d steps in %.2f s (%.2f million cell-updates per second). Output in %s/\n",
                step, secs, mcells, dir.c_str());
    return 0;
}