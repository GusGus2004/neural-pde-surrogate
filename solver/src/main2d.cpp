// 2D Euler solver: sets up test problem, runs writes HDF5 snapshots.
//
// Usage: ./build/euler2d --test lax_liu3 --n 256 --scheme hllc --limiter vanleer --nout 10 --out runs
#include "solver2d.hpp"
#include "io_hdf5.hpp"
#include "trajectory.hpp"

#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <cmath>

using namespace e2d;

std::map<std::string, std::string> parse_args(int argc, char* argv[]) {
    std::map<std::string, std::string> a = {
        {"test", "lax_liu3"}, {"n", "256"}, {"scheme", "hllc"}, {"limiter", "vanleer"},
        {"cfl", "0.5"}, {"nout", "10"}, {"out", "runs"}, {"tend", "default"}, {"profile", "no"}, {"init", ""},
        {"traj", ""}, {"save", ""}};
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

        else if (test == "kh") {
        // Kelvin-Helmholtz instability smooth version of McNally Lyra & Passy (2012).
        // Two shear layers (at y = 1/4 and y = 3/4) separate a dense strip moving left from lighter
        // fluid moving right, small vertical velocity wiggle seeds the instability.
        pr.grid = make_grid(n, n, 0.0, 1.0, 0.0, 1.0, BC::Periodic, BC::Periodic);
        pr.t_end = 2.0;
        U.assign(pr.grid.size(), Conserved{});
        const double PI = 3.14159265358979323846;
        const double rho1 = 1.0, rho2 = 2.0, rhom = 0.5 * (rho1 - rho2);
        const double u1 = 0.5, u2 = -0.5, um = 0.5 * (u1 - u2);
        const double L = 0.025;                       // thickness of the shear layers
        const Grid& g = pr.grid;
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                double x = g.xc(i), y = g.yc(j), rho, u;
                if (y < 0.25) {
                    rho = rho1 - rhom * std::exp((y - 0.25) / L);
                    u   = u1   - um   * std::exp((y - 0.25) / L);
                } else if (y < 0.5) {
                    rho = rho2 + rhom * std::exp((0.25 - y) / L);
                    u   = u2   + um   * std::exp((0.25 - y) / L);
                } else if (y < 0.75) {
                    rho = rho2 + rhom * std::exp((y - 0.75) / L);
                    u   = u2   + um   * std::exp((y - 0.75) / L);
                } else {
                    rho = rho1 - rhom * std::exp((0.75 - y) / L);
                    u   = u1   - um   * std::exp((0.75 - y) / L);
                }
                double v = 0.01 * std::sin(4.0 * PI * x);     // seed perturbation
                U[g.idx(i, j)] = to_conserved({rho, u, v, 2.5});
            }
        }
    else {
        std::cerr << "Unknown test '" << test << "' (use sod_x, sod_y, lax_liu3, sine2d or kh)\n";
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
    Problem pr;
    std::string run_name = test;
    if (!a["init"].empty()) {
        // Initial condition from a file: grid, boundaries and state all come from the file
        InitInfo ii;
        try {
            ii = read_initial_condition(a["init"], pr.grid, U);
        } catch (const std::exception& e) {
            std::cerr << "Error reading " << a["init"] << ": " << e.what() << '\n';
            return 1;
        }
        pr.t_end = ii.t_end;
        run_name = ii.test;
        if (pr.t_end <= 0.0 && a["tend"] == "default") {
            std::cerr << "The initial-condition file has no t_end; give one with --tend\n";
            return 1;
        }
    } else {
        pr = setup(test, n, U);
    }
    if (a["tend"] != "default") pr.t_end = std::atof(a["tend"].c_str());   // override test's end time
    if (!(pr.t_end > 0.0)) { std::cerr << "tend must be positive\n"; return 1; }

    const Grid& g = pr.grid;
    Solver solver(g, flux, lim, cfl);

        RunInfo info{run_name, scheme, lim_name, cfl, 0};
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    std::printf("%s: %d x %d cells, t_end = %g, %s + %s, CFL %g, %d thread(s)\n",
                run_name.c_str(), g.nx, g.ny, pr.t_end, scheme.c_str(), lim_name.c_str(), cfl, threads);

    // Two output modes:
    //   --traj FILE  one trajectory file (all frames, conserved variables, float32), at the --save sizes
    //   otherwise    one snapshot file per frame (primitive variables, float64) in a run folder
    const bool traj_mode = !a["traj"].empty();
    std::unique_ptr<TrajectoryWriter> traj;
    std::string dir;
    if (traj_mode) {
        try {
            traj = std::make_unique<TrajectoryWriter>(
                g, a["save"].empty() ? std::vector<int>{g.nx} : parse_sizes(a["save"]));
        } catch (const std::exception& e) {
            std::cerr << "Bad --save: " << e.what() << '\n';
            return 1;
        }
    } else {
        dir = a["out"] + "/" + run_name + "_" + scheme + "_" + lim_name + "_N" + std::to_string(g.nx);
        std::filesystem::create_directories(dir);
    }
    auto snap_path = [&](int k) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "/snap_%04d.h5", k);
        return dir + buf;
    };
    auto output = [&](int k, double t) {
        if (traj_mode) traj->add(U, t);
        else           write_snapshot(snap_path(k), g, U, t, info);
    };

    double t = 0.0;
    int step = 0;
    output(0, t);
    auto t0 = std::chrono::steady_clock::now();

    for (int k = 1; k <= nout; ++k) {
        const double t_next = pr.t_end * k / nout;     // frames at equally spaced times
        while (t < t_next) {
            double dt = std::min(solver.compute_dt(U), t_next - t);   // land exactly on t_next
            solver.step(U, dt);
            t += dt;
            ++step;
        }
        info.step = step;
        output(k, t);
        if (!traj_mode) std::printf("  snapshot %2d/%d  t = %.4f  step %d\n", k, nout, t, step);
    }

    if (traj_mode) {
        try {
            traj->write(a["traj"], info, pr.t_end);
        } catch (const std::exception& e) {
            std::cerr << "Error writing trajectory: " << e.what() << '\n';
            return 1;
        }
    }

    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double mcells = double(g.nx) * g.ny * step / secs / 1e6;
    std::printf("Done: %d steps in %.2f s (%.2f million cell-updates per second). Output in %s\n",
                step, secs, mcells, traj_mode ? a["traj"].c_str() : (dir + "/").c_str());
    if (a["profile"] == "yes") solver.profile().print(step, long(g.nx) * g.ny);
    return 0;
}