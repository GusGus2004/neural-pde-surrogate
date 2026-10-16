#pragma once
#include "state.hpp"
#include "limiters.hpp"
#include <cmath>


inline Primitive limited_slope(const Primitive& Wm, const Primitive& W0,
                               const Primitive& Wp, Limiter lim) {
    return {lim(W0.rho - Wm.rho, Wp.rho - W0.rho),
            lim(W0.u   - Wm.u,   Wp.u   - W0.u),
            lim(W0.p   - Wm.p,   Wp.p   - W0.p)};
}