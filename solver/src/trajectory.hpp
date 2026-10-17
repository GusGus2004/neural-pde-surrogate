#pragma once
#include "grid2d.hpp"
#include "io_hdf5.hpp"
#include <hdf5.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace e2d {

// Collects whole simulation in memory at one or more storage resolutions and writes as
// single HDF5 file: one float32 dataset U_<n> of shape (frames, 4, ny_c, nx_c) per resolution,
// holding the conserved variables (rho, rho*u, rho*v, E), plus the frame times t.
//
// Vals are block averages of conserved variables, so totals of mass, momentum
// and energy are same at every resolution (up to float32 rounding)
class TrajectoryWriter {
public:
    TrajectoryWriter(const Grid& grid, const std::vector<int>& sizes) : g(grid) {
        for (int m : sizes) {
            if (m <= 0 || g.nx % m != 0)
                throw std::runtime_error("save size " + std::to_string(m) + " does not divide nx = " + std::to_string(g.nx));
            int f = g.nx / m;                                 // coarsening factor
            if (g.ny % f != 0)
                throw std::runtime_error("save size " + std::to_string(m) + " does not divide ny evenly");
            levels.push_back({m, g.ny / f, f, {}});
        }
    }

    // add the current state as a new frame at every storage resolution
    void add(const std::vector<Conserved>& U, double t) {
        times.push_back(t);
        for (Level& L : levels) {
            const int nxc = L.nx, nyc = L.ny, f = L.factor;
            const double inv = 1.0 / (double(f) * f);
            size_t base = L.data.size();
            L.data.resize(base + size_t(4) * nyc * nxc);
            float* out = L.data.data() + base;
            for (int jc = 0; jc < nyc; ++jc)
                for (int ic = 0; ic < nxc; ++ic) {
                    double s[4] = {0.0, 0.0, 0.0, 0.0};
                    for (int j = jc * f; j < (jc + 1) * f; ++j)       // sum over the f x f fine cells
                        for (int i = ic * f; i < (ic + 1) * f; ++i) {
                            const Conserved& c = U[g.idx(i, j)];
                            s[0] += c.rho; s[1] += c.mx; s[2] += c.my; s[3] += c.E;
                        }
                    for (int q = 0; q < 4; ++q)                      // layout: (variable, y, x)
                        out[(size_t(q) * nyc + jc) * nxc + ic] = float(s[q] * inv);
                }
        }
    }

    void write(const std::string& path, const RunInfo& info, double t_end, double t_start = 0.0) const {
        hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (file < 0) throw std::runtime_error("could not create " + path + " (does the folder exist?)");

        detail::write_dataset(file, "t", times, {hsize_t(times.size())});
        for (const Level& L : levels) {
            hsize_t dims[4] = {hsize_t(times.size()), 4, hsize_t(L.ny), hsize_t(L.nx)};
            hid_t space = H5Screate_simple(4, dims, nullptr);
            // Store in chunks of one frame, compressed: shocks leave large flat regions that compress well.
            hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
            hsize_t chunk[4] = {1, 4, hsize_t(L.ny), hsize_t(L.nx)};
            H5Pset_chunk(plist, 4, chunk);
            H5Pset_shuffle(plist);
            H5Pset_deflate(plist, 4);
            std::string name = "U_" + std::to_string(L.nx);
            hid_t dset = H5Dcreate2(file, name.c_str(), H5T_NATIVE_FLOAT, space, H5P_DEFAULT, plist, H5P_DEFAULT);
            if (dset < 0) throw std::runtime_error("could not create dataset " + name);
            H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, L.data.data());
            H5Dclose(dset); H5Pclose(plist); H5Sclose(space);
        }

        detail::write_attr(file, "variables", std::string("rho, rho*u, rho*v, E"));
        detail::write_attr(file, "t_start", t_start);
        detail::write_attr(file, "t_end", t_end);
        detail::write_attr(file, "frames", int(times.size()));
        detail::write_attr(file, "steps", info.step);
        detail::write_attr(file, "gamma", GAMMA);
        detail::write_attr(file, "cfl", info.cfl);
        detail::write_attr(file, "solver_nx", g.nx);
        detail::write_attr(file, "solver_ny", g.ny);
        detail::write_attr(file, "x_min", g.x_min); detail::write_attr(file, "x_max", g.x_max);
        detail::write_attr(file, "y_min", g.y_min); detail::write_attr(file, "y_max", g.y_max);
        detail::write_attr(file, "bc_x", std::string(g.bc_x == BC::Periodic ? "periodic" : "transmissive"));
        detail::write_attr(file, "bc_y", std::string(g.bc_y == BC::Periodic ? "periodic" : "transmissive"));
        detail::write_attr(file, "test", info.test);
        detail::write_attr(file, "scheme", info.scheme);
        detail::write_attr(file, "limiter", info.limiter);
        H5Fclose(file);
    }

private:
    struct Level {
        int nx, ny, factor;
        std::vector<float> data;    // frames x 4 x ny x nx, filled frame by frame
    };
    Grid g;
    std::vector<Level> levels;
    std::vector<double> times;
};

// Parse "128,64" into {128, 64}.
inline std::vector<int> parse_sizes(const std::string& s) {
    std::vector<int> out;
    size_t start = 0;
    while (start < s.size()) {
        size_t end = s.find(',', start);
        if (end == std::string::npos) end = s.size();
        out.push_back(std::stoi(s.substr(start, end - start)));
        start = end + 1;
    }
    return out;
}

}  // namespace e2d