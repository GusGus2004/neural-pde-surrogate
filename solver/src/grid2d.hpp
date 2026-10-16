#pragma once
#include "state2d.hpp"
#include <vector>

namespace e2d {

enum class BC { Transmissive, Periodic };

// uniform 2D grid with ng ghost cells every side.
// physical cells have indices i = 0..nx-1, j = 0..ny-1.
// Ghost cells have indices -ng..-1 and nx..nx+ng-1 (same in j).
struct Grid {
    int nx = 0, ny = 0;
    int ng = 2;
    double x_min = 0.0, x_max = 1.0, y_min = 0.0, y_max = 1.0;
    double dx = 0.0, dy = 0.0;
    BC bc_x = BC::Transmissive, bc_y = BC::Transmissive;

    int nx_tot() const { return nx + 2 * ng; }       // row length including ghosts
    int ny_tot() const { return ny + 2 * ng; }
    int size()   const { return nx_tot() * ny_tot(); }

    // Position of cell (i, j) in the flat array. Rows fixed j
    int idx(int i, int j) const { return (j + ng) * nx_tot() + (i + ng); }

    double xc(int i) const { return x_min + (i + 0.5) * dx; }   // cell centres
    double yc(int j) const { return y_min + (j + 0.5) * dy; }
};

inline Grid make_grid(int nx, int ny, double x_min, double x_max, double y_min, double y_max,
                      BC bc_x, BC bc_y) {
    Grid g;
    g.nx = nx; g.ny = ny;
    g.x_min = x_min; g.x_max = x_max; g.y_min = y_min; g.y_max = y_max;
    g.dx = (x_max - x_min) / nx;
    g.dy = (y_max - y_min) / ny;
    g.bc_x = bc_x; g.bc_y = bc_y;
    return g;
}

// Fill all ghost cells from the physical cells.
// x-ghosts first (physical rows only), then y-ghosts for whole rows, so corners get filled
inline void apply_bcs(const Grid& g, std::vector<Conserved>& U) {
    for (int j = 0; j < g.ny; ++j) {
        for (int k = 1; k <= g.ng; ++k) {
            if (g.bc_x == BC::Periodic) {
                U[g.idx(-k, j)]           = U[g.idx(g.nx - k, j)];
                U[g.idx(g.nx - 1 + k, j)] = U[g.idx(k - 1, j)];
            } else {
                U[g.idx(-k, j)]           = U[g.idx(0, j)];
                U[g.idx(g.nx - 1 + k, j)] = U[g.idx(g.nx - 1, j)];
            }
        }
    }
    for (int i = -g.ng; i < g.nx + g.ng; ++i) {
        for (int k = 1; k <= g.ng; ++k) {
            if (g.bc_y == BC::Periodic) {
                U[g.idx(i, -k)]           = U[g.idx(i, g.ny - k)];
                U[g.idx(i, g.ny - 1 + k)] = U[g.idx(i, k - 1)];
            } else {
                U[g.idx(i, -k)]           = U[g.idx(i, 0)];
                U[g.idx(i, g.ny - 1 + k)] = U[g.idx(i, g.ny - 1)];
            }
        }
    }
}

} 