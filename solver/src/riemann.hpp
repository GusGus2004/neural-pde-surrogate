#pragma once
#include "state.hpp"
#include <algorithm>
#include <cmath>


inline Conserved rusanov_flux(const Conserved& UL, const Conserved& UR) {
    Primitive WL = to_primitive(UL);
    Primitive WR = to_primitive(UR);

    double s_max = std::max(std::abs(WL.u) + sound_speed(WL),
                            std::abs(WR.u) + sound_speed(WR));

    return 0.5 * (physical_flux(UL) + physical_flux(UR))
         - 0.5 * s_max * (UR - UL);
}

inline Conserved hllc_star_state(const Conserved& U, const Primitive& W,
                                 double S, double S_star) {
    double factor = W.rho * (S - W.u) / (S - S_star);
    return {factor,
            factor * S_star,
            factor * (U.E / W.rho
                      + (S_star - W.u) * (S_star + W.p / (W.rho * (S - W.u))))};
}

inline Conserved hllc_flux(const Conserved& UL, const Conserved& UR) {
    Primitive WL = to_primitive(UL);
    Primitive WR = to_primitive(UR);
    double aL = sound_speed(WL), aR = sound_speed(WR);

    double SL = std::min(WL.u - aL, WR.u - aR);
    double SR = std::max(WL.u + aL, WR.u + aR);

    double S_star = (WR.p - WL.p
                     + WL.rho * WL.u * (SL - WL.u)
                     - WR.rho * WR.u * (SR - WR.u))
                  / (WL.rho * (SL - WL.u) - WR.rho * (SR - WR.u));

    if (0.0 <= SL)     return physical_flux(UL);
    if (0.0 <= S_star) return physical_flux(UL) + SL * (hllc_star_state(UL, WL, SL, S_star) - UL);
    if (0.0 <= SR)     return physical_flux(UR) + SR * (hllc_star_state(UR, WR, SR, S_star) - UR);
    return physical_flux(UR);
}