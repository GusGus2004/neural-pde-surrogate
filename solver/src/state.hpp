#pragma once
#include <cmath>

constexpr double GAMMA = 1.4;

struct Conserved { double rho = 0.0, mom = 0.0, E = 0.0; };  // (rho, rho*u, E)
struct Primitive { double rho = 0.0, u = 0.0, p = 0.0; };

inline Conserved operator+(Conserved a, Conserved b) { return {a.rho + b.rho, a.mom + b.mom, a.E + b.E}; }
inline Conserved operator-(Conserved a, Conserved b) { return {a.rho - b.rho, a.mom - b.mom, a.E - b.E}; }
inline Conserved operator*(double s, Conserved a)    { return {s * a.rho, s * a.mom, s * a.E}; }

inline Conserved to_conserved(const Primitive& W) {
    return {W.rho,
            W.rho * W.u,
            W.p / (GAMMA - 1.0) + 0.5 * W.rho * W.u * W.u};
}

inline Primitive to_primitive(const Conserved& U) {
    double u = U.mom / U.rho;
    double p = (GAMMA - 1.0) * (U.E - 0.5 * U.rho * u * u);
    return {U.rho, u, p};
}

inline double sound_speed(const Primitive& W) {
    return std::sqrt(GAMMA*W.p/W.rho);
}

// TODO: physical flux F(U) = (rho*u, rho*u^2 + p, u*(E + p))
inline Conserved physical_flux(const Conserved& U) {
    Primitive W = to_primitive(U);
    return {U.mom,
    U.mom*W.u + W.p,
    W.u*(U.E +W.p)};
}