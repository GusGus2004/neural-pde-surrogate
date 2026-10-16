#pragma once
#include "state2d.hpp"
#include <algorithm>
#include <cmath>

namespace e2d {

using FluxFn = Conserved (*)(const Conserved&, const Conserved&);

// rusanov flux normal to x-face.
inline Conserved rusanov_flux(const Conserved& UL, const Conserved& UR) {
    Primitive WL = to_primitive(UL), WR = to_primitive(UR);
    double s_max = std::max(std::abs(WL.u) + sound_speed(WL), std::abs(WR.u) + sound_speed(WR));
    return 0.5 * (flux_x(UL) + flux_x(UR)) - 0.5 * s_max * (UR - UL);
}

// HLLC star state (Toro eq. 10.39) tangential velocity v is carried unchanged across contact
inline Conserved hllc_star_state(const Conserved& U, const Primitive& W, double S, double S_star) {
    double factor = W.rho * (S - W.u) / (S - S_star);
    return {factor,
            factor * S_star,
            factor * W.v,
            factor * (U.E / W.rho + (S_star - W.u) * (S_star + W.p / (W.rho * (S - W.u))))};
}

// HLLC flux normal to x-face, Davis wave-speed estimate
inline Conserved hllc_flux(const Conserved& UL, const Conserved& UR) {
    Primitive WL = to_primitive(UL), WR = to_primitive(UR);
    double aL = sound_speed(WL), aR = sound_speed(WR);
    double SL = std::min(WL.u - aL, WR.u - aR);
    double SR = std::max(WL.u + aL, WR.u + aR);
    double S_star = (WR.p - WL.p + WL.rho * WL.u * (SL - WL.u) - WR.rho * WR.u * (SR - WR.u))
                  / (WL.rho * (SL - WL.u) - WR.rho * (SR - WR.u));

    if (0.0 <= SL)     return flux_x(UL);
    if (0.0 <= S_star) return flux_x(UL) + SL * (hllc_star_state(UL, WL, SL, S_star) - UL);
    if (0.0 <= SR)     return flux_x(UR) + SR * (hllc_star_state(UR, WR, SR, S_star) - UR);
    return flux_x(UR);
}

// Flux through a y-face: rotate so y becomes x, solve the 1D problem, rotate back. - same as state2d
inline Conserved flux_y_face(FluxFn flux, const Conserved& UL, const Conserved& UR) {
    return swap_xy(flux(swap_xy(UL), swap_xy(UR)));
}

}