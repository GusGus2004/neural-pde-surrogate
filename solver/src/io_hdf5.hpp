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

namespace detail {

inline bool has_attr(hid_t obj, const char* name) { return H5Aexists(obj, name) > 0; }

inline double read_attr_double(hid_t obj, const char* name) {
    if (!has_attr(obj, name)) throw std::runtime_error(std::string("missing attribute ") + name);
    double value = 0.0;
    hid_t attr = H5Aopen(obj, name, H5P_DEFAULT);
    H5Aread(attr, H5T_NATIVE_DOUBLE, &value);       // HDF5 converts from int or float
    H5Aclose(attr);
    return value;
}

inline int read_attr_int(hid_t obj, const char* name) {
    if (!has_attr(obj, name)) throw std::runtime_error(std::string("missing attribute ") + name);
    int value = 0;
    hid_t attr = H5Aopen(obj, name, H5P_DEFAULT);
    H5Aread(attr, H5T_NATIVE_INT, &value);
    H5Aclose(attr);
    return value;
}

// read string attribute written either by C++ writer (fixed-length) or by h5py (variable-length)
inline std::string read_attr_string(hid_t obj, const char* name) {
    if (!has_attr(obj, name)) throw std::runtime_error(std::string("missing attribute ") + name);
    hid_t attr = H5Aopen(obj, name, H5P_DEFAULT);
    hid_t type = H5Aget_type(attr);
    std::string out;
    if (H5Tis_variable_str(type) > 0) {
        hid_t mem = H5Tcopy(H5T_C_S1);
        H5Tset_size(mem, H5T_VARIABLE);
        H5Tset_cset(mem, H5Tget_cset(type));
        char* buf = nullptr;
        H5Aread(attr, mem, &buf);
        if (buf) out = buf;
        H5free_memory(buf);
        H5Tclose(mem);
    } else {
        std::vector<char> buf(H5Tget_size(type) + 1, '\0');
        H5Aread(attr, type, buf.data());
        out = buf.data();
    }
    H5Tclose(type);
    H5Aclose(attr);
    return out;
}

inline std::vector<double> read_dataset(hid_t file, const char* name, size_t expected) {
    hid_t dset = H5Dopen2(file, name, H5P_DEFAULT);
    if (dset < 0) throw std::runtime_error(std::string("missing dataset ") + name);
    hid_t space = H5Dget_space(dset);
    hssize_t n = H5Sget_simple_extent_npoints(space);
    if (n != hssize_t(expected))
        throw std::runtime_error(std::string("dataset ") + name + " has the wrong size");
    std::vector<double> data(expected);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    H5Sclose(space); H5Dclose(dset);
    return data;
}

inline BC parse_bc(const std::string& s) {
    if (s == "periodic") return BC::Periodic;
    if (s == "transmissive") return BC::Transmissive;
    throw std::runtime_error("unknown boundary condition '" + s + "'");
}

}

// initial-condition file describes besides the state itself
struct InitInfo {
    std::string test = "init";
    double t_end = -1.0;          // negative if the file does not specify one
};

// Read an initial condition in the same format as our snapshots: datasets rho, u, v, p with shape (ny, nx),
// attributes nx, ny, x_min, x_max, y_min, y_max, bc_x, bc_y, and t_end and test
// Sets up grid and fills physical cells of U snapshot file is therefore valid
// init condition which lets a run be restarted from any snapshot
inline InitInfo read_initial_condition(const std::string& path, Grid& g, std::vector<Conserved>& U) {
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);   // report errors ourselves
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) throw std::runtime_error("could not open " + path + " (does the file exist?)");

    g = make_grid(detail::read_attr_int(file, "nx"), detail::read_attr_int(file, "ny"),
                  detail::read_attr_double(file, "x_min"), detail::read_attr_double(file, "x_max"),
                  detail::read_attr_double(file, "y_min"), detail::read_attr_double(file, "y_max"),
                  detail::parse_bc(detail::read_attr_string(file, "bc_x")),
                  detail::parse_bc(detail::read_attr_string(file, "bc_y")));

    const size_t n = size_t(g.nx) * g.ny;
    auto rho = detail::read_dataset(file, "rho", n);
    auto u   = detail::read_dataset(file, "u",   n);
    auto v   = detail::read_dataset(file, "v",   n);
    auto p   = detail::read_dataset(file, "p",   n);

    InitInfo info;
    if (detail::has_attr(file, "t_end")) info.t_end = detail::read_attr_double(file, "t_end");
    if (detail::has_attr(file, "test"))  info.test  = detail::read_attr_string(file, "test");
    H5Fclose(file);

    U.assign(g.size(), Conserved{});
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            size_t k = size_t(j) * g.nx + i;
            if (!(rho[k] > 0.0) || !(p[k] > 0.0))
                throw std::runtime_error("initial condition has non-positive density or pressure");
            U[g.idx(i, j)] = to_conserved({rho[k], u[k], v[k], p[k]});
        }
    return info;
}
}