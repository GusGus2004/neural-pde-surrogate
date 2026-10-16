// Writes snapshot with known values; tests/check_snapshot.py then reads it back and checks
#include "io_hdf5.hpp"
#include <cstdio>

using namespace e2d;

int main() {
    // non-square, non-unit domain so that swapping x/y or nx/ny is caught
    Grid g = make_grid(6, 4, 0.0, 3.0, -1.0, 1.0, BC::Periodic, BC::Transmissive);
    std::vector<Conserved> U(g.size());
    // fll ghost cells 
    for (auto& c : U) c = to_conserved({-1.0, 0.0, 0.0, -1.0});
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i)
            U[g.idx(i, j)] = to_conserved({1.0 + i, 0.1 * i, 0.01 * j, 2.0 + j});

    RunInfo info{"io_test", "hllc", "vanleer", 0.5, 42};
    write_snapshot("io_test.h5", g, U, 0.125, info);
    std::printf("Wrote io_test.h5\n");
    return 0;
}