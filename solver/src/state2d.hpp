#pragma once
#include <cmath>

// 2D Euler: everything lives in namespace e2d so the names don't clash with 1D.
namespace e2d {

constexpr double GAMMA = 1.4;

struct Conserved { double rho = 0.0, mx = 0.0, my = 0.0, E = 0.0; };  // (rho, rho*u, rho*v, E)
struct Primitive { double rho = 0.0, u = 0.0, v = 0.0, p = 0.0; };

inline Conserved operator+(Conserved a, Conserved b) { return {a.rho + b.rho, a.mx + b.mx, a.my + b.my, a.E + b.E}; }
inline Conserved operator-(Conserved a, Conserved b) { return {a.rho - b.rho, a.mx - b.mx, a.my - b.my, a.E - b.E}; }
inline Conserved operator*(double s, Conserved a)    { return {s * a.rho, s * a.mx, s * a.my, s * a.E}; }

inline Conserved to_conserved(const Primitive& W) {
    return {W.rho, W.rho * W.u, W.rho * W.v,
            W.p / (GAMMA - 1.0) + 0.5 * W.rho * (W.u * W.u + W.v * W.v)};
}

inline Primitive to_primitive(const Conserved& U) {
    double u = U.mx / U.rho, v = U.my / U.rho;
    double p = (GAMMA - 1.0) * (U.E - 0.5 * U.rho * (u * u + v * v));
    return {U.rho, u, v, p};
}

inline double sound_speed(const Primitive& W) { return std::sqrt(GAMMA * W.p / W.rho); }

// physical flux in the x-direction, y-direction flux obtained by rotation
inline Conserved flux_x(const Conserved& U) {
    Primitive W = to_primitive(U);
    return {U.mx,
            U.mx * W.u + W.p,
            U.my * W.u,
            W.u * (U.E + W.p)};
}

// swapp the two momentum components: turns y-direction problem into an x-direction one.
inline Conserved swap_xy(Conserved U) { return {U.rho, U.my, U.mx, U.E}; }

}