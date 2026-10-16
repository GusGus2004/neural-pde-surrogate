// Checks for 2D grid: indexing and ghost-cell filling.
#include "grid2d.hpp"
#include <cmath>
#include <cstdio>

using namespace e2d;

int failures = 0;

void check(const char* name, bool ok) {
    std::printf("%-55s %s\n", name, ok ? "PASS" : "FAIL");
    if (!ok) ++failures;
}

// Tag each physical cell with its (i, j) can see where ghost values were copied from
void fill_tags(const Grid& g, std::vector<Conserved>& U) {
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i)
            U[g.idx(i, j)] = {1.0, double(i), double(j), 0.0};  
}

bool is_copy_of(const Grid& g, const std::vector<Conserved>& U, int i, int j, int i_src, int j_src) {
    const Conserved& a = U[g.idx(i, j)];
    return a.mx == double(i_src) && a.my == double(j_src);
}

int main() {
    //small non-square grid, so mixing up nx and ny shows up
    Grid g = make_grid(5, 3, 0.0, 1.0, 0.0, 0.6, BC::Transmissive, BC::Transmissive);
    std::vector<Conserved> U(g.size());

    // indxing
    check("size = (5+4) * (3+4) = 63", g.size() == 63);
    check("idx(-2, -2) = 0 (first ghost corner)", g.idx(-2, -2) == 0);
    check("idx(6, 4) = 62 (last ghost corner)", g.idx(6, 4) == 62);
    check("idx(1, 0) = idx(0, 0) + 1 (rows contiguous)", g.idx(1, 0) == g.idx(0, 0) + 1);
    check("dx = 0.2, dy = 0.2", std::abs(g.dx - 0.2) < 1e-15 && std::abs(g.dy - 0.2) < 1e-15);
    check("xc(0) = 0.1, yc(2) = 0.5", std::abs(g.xc(0) - 0.1) < 1e-15 && std::abs(g.yc(2) - 0.5) < 1e-15);

    // transmissive: ghosts copy the nearest physical cell
    fill_tags(g, U);
    apply_bcs(g, U);
    check("transmissive: left ghosts copy column 0",   is_copy_of(g, U, -1, 1, 0, 1) && is_copy_of(g, U, -2, 1, 0, 1));
    check("transmissive: right ghosts copy column 4",  is_copy_of(g, U, 5, 2, 4, 2) && is_copy_of(g, U, 6, 2, 4, 2));
    check("transmissive: bottom ghosts copy row 0",    is_copy_of(g, U, 3, -1, 3, 0) && is_copy_of(g, U, 3, -2, 3, 0));
    check("transmissive: top ghosts copy row 2",       is_copy_of(g, U, 3, 3, 3, 2) && is_copy_of(g, U, 3, 4, 3, 2));
    check("transmissive: corner copies corner cell",   is_copy_of(g, U, -2, -2, 0, 0) && is_copy_of(g, U, 6, 4, 4, 2));

    // periodic: ghosts copy opposite end
    g.bc_x = BC::Periodic; g.bc_y = BC::Periodic;
    fill_tags(g, U);
    apply_bcs(g, U);
    check("periodic: left ghosts copy columns 4, 3",   is_copy_of(g, U, -1, 1, 4, 1) && is_copy_of(g, U, -2, 1, 3, 1));
    check("periodic: right ghosts copy columns 0, 1",  is_copy_of(g, U, 5, 1, 0, 1) && is_copy_of(g, U, 6, 1, 1, 1));
    check("periodic: bottom ghosts copy rows 2, 1",    is_copy_of(g, U, 2, -1, 2, 2) && is_copy_of(g, U, 2, -2, 2, 1));
    check("periodic: top ghosts copy rows 0, 1",       is_copy_of(g, U, 2, 3, 2, 0) && is_copy_of(g, U, 2, 4, 2, 1));
    check("periodic: corner wraps in both directions", is_copy_of(g, U, -1, -1, 4, 2) && is_copy_of(g, U, 5, 3, 0, 0));

    // mixed: periodic in x, transmissive in y
    g.bc_x = BC::Periodic; g.bc_y = BC::Transmissive;
    fill_tags(g, U);
    apply_bcs(g, U);
    check("mixed: x wraps, y copies nearest row",      is_copy_of(g, U, -1, 1, 4, 1) && is_copy_of(g, U, 2, 3, 2, 2));

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}