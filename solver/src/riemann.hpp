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