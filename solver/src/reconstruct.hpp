#pragma once
#include "state.hpp"
#include <cmath>


using Limiter = double (*)(double, double);

inline double no_slope(double, double) { return 0.0; }               
inline double unlimited(double a, double b) { return 0.5 * (a + b); }

inline double minmod(double a, double b) {
    if (a * b <= 0.0) return 0.0;                    
    return (std::abs(a) < std::abs(b)) ? a : b;      
}

inline double van_leer(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    return 2.0 * a * b / (a + b);                    
}

inline Primitive limited_slope(const Primitive& Wm, const Primitive& W0,
                               const Primitive& Wp, Limiter lim) {
    return {lim(W0.rho - Wm.rho, Wp.rho - W0.rho),
            lim(W0.u   - Wm.u,   Wp.u   - W0.u),
            lim(W0.p   - Wm.p,   Wp.p   - W0.p)};
}