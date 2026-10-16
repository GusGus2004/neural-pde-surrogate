#pragma once
#include <cmath>

// limiter takes jumps to the left (a) and right (b) of a cell and returns a slope.
using Limiter = double (*)(double, double);

inline double no_slope(double, double) { return 0.0; }              
inline double unlimited(double a, double b) { return 0.5 * (a + b); }

inline double minmod(double a, double b) {
    if (a * b <= 0.0) return 0.0;                   // local extreme   
    return (std::abs(a) < std::abs(b)) ? a : b;    
}

inline double van_leer(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    return 2.0 * a * b / (a + b);                     // harmonic mean: less diffusive
}