#pragma once
#include "grid2d.hpp"
#include <hdf5.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace e2d {

// info about run, stored as attributes so every snapshot documents itself
struct RunInfo {
    std::string test, scheme, limiter;
    double cfl = 0.0;
    int step = 0;
};

namespace detail {

inline void write_attr(hid_t obj, const char* name, double value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr  = H5Acreate2(obj, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_DOUBLE, &value);
    H5Aclose(attr); H5Sclose(space);
}

inline void write_attr(hid_t obj, const char* name, int value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr  = H5Acreate2(obj, name, H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_INT, &value);
    H5Aclose(attr); H5Sclose(space);
}

inline void write_attr(hid_t obj, const char* name, const std::string& value) {
    hid_t type = H5Tcopy(H5T_C_S1);                       // fixed-length string type
    H5Tset_size(type, value.empty() ? 1 : value.size());
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr  = H5Acreate2(obj, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, type, value.empty() ? "" : value.c_str());
    H5Aclose(attr); H5Sclose(space); H5Tclose(type);
}

// Write 1D or 2D array of doubles as dataset. dims = {ny, nx} for 2D, {n} for 1D.
inline void write_dataset(hid_t file, const char* name, const std::vector<double>& data,
                          const std::vector<hsize_t>& dims) {
    hid_t space = H5Screate_simple(int(dims.size()), dims.data(), nullptr);
    hid_t dset  = H5Dcreate2(file, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0) throw std::runtime_error(std::string("could not create dataset ") + name);
    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    H5Dclose(dset); H5Sclose(space);
}

} 

// Write the physical cells (no ghosts) as primitive variables rho, u, v, p, each with shape (ny, nx)
// anz the cell-centre coordinates x (nx) and y (ny). Grid, time and run settings go in att
inline void write_snapshot(const std::string& path, const Grid& g, const std::vector<Conserved>& U,
                           double t, const RunInfo& info) {
    const size_t n = size_t(g.nx) * g.ny;
    std::vector<double> rho(n), u(n), v(n), p(n), x(g.nx), y(g.ny);
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            Primitive W = to_primitive(U[g.idx(i, j)]);
            size_t k = size_t(j) * g.nx + i;                // row-major, no ghosts
            rho[k] = W.rho; u[k] = W.u; v[k] = W.v; p[k] = W.p;
        }
    for (int i = 0; i < g.nx; ++i) x[i] = g.xc(i);
    for (int j = 0; j < g.ny; ++j) y[j] = g.yc(j);

    hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) throw std::runtime_error("could not create " + path + " (does the folder exist?)");

    const std::vector<hsize_t> dims2{hsize_t(g.ny), hsize_t(g.nx)};
    detail::write_dataset(file, "rho", rho, dims2);
    detail::write_dataset(file, "u",   u,   dims2);
    detail::write_dataset(file, "v",   v,   dims2);
    detail::write_dataset(file, "p",   p,   dims2);
    detail::write_dataset(file, "x",   x,   {hsize_t(g.nx)});
    detail::write_dataset(file, "y",   y,   {hsize_t(g.ny)});

    detail::write_attr(file, "t", t);
    detail::write_attr(file, "step", info.step);
    detail::write_attr(file, "gamma", GAMMA);
    detail::write_attr(file, "cfl", info.cfl);
    detail::write_attr(file, "nx", g.nx);
    detail::write_attr(file, "ny", g.ny);
    detail::write_attr(file, "x_min", g.x_min); detail::write_attr(file, "x_max", g.x_max);
    detail::write_attr(file, "y_min", g.y_min); detail::write_attr(file, "y_max", g.y_max);
    detail::write_attr(file, "bc_x", std::string(g.bc_x == BC::Periodic ? "periodic" : "transmissive"));
    detail::write_attr(file, "bc_y", std::string(g.bc_y == BC::Periodic ? "periodic" : "transmissive"));
    detail::write_attr(file, "test", info.test);
    detail::write_attr(file, "scheme", info.scheme);
    detail::write_attr(file, "limiter", info.limiter);

    H5Fclose(file);
}

}