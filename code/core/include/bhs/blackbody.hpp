#pragma once
// Black-body colour: Planck spectrum integrated with the CIE 1931 functions (analytic
// multi-lobe fit of Wyman, Sloan & Shirley 2013) -> XYZ -> linear sRGB. Absolute radiance
// (not normalised): hotter = brighter. The exposure is left to the caller.
// Redshift: the observed spectrum is a black body at T_obs = g * T_emit (I_nu/nu^3 invariant).
#include <algorithm>
#include <cmath>

#include "vec3.hpp"

namespace bhs::blackbody {

inline double lobe(double l, double mu, double s1, double s2) {
  const double t = (l - mu) / (l < mu ? s1 : s2);
  return std::exp(-0.5 * t * t);
}

inline Vec3 rgb(double T) {
  if (T < 100.0) return {0, 0, 0};
  double X = 0, Y = 0, Z = 0;
  for (double l = 380.0; l <= 780.0; l += 5.0) {
    const double planck = std::pow(l, -5.0) / (std::exp(1.438776877e7 / (l * T)) - 1.0);
    const double xb = 1.056 * lobe(l, 599.8, 37.9, 31.0) + 0.362 * lobe(l, 442.0, 16.0, 26.7) - 0.065 * lobe(l, 501.1, 20.4, 26.2);
    const double yb = 0.821 * lobe(l, 568.8, 46.9, 40.5) + 0.286 * lobe(l, 530.9, 16.3, 31.1);
    const double zb = 1.217 * lobe(l, 437.0, 11.8, 36.0) + 0.681 * lobe(l, 459.0, 26.0, 13.8);
    X += planck * xb; Y += planck * yb; Z += planck * zb;
  }
  const double k = 1e21;  // arbitrary scale (lambda in nm): leaves values around 1
  X *= k; Y *= k; Z *= k;
  return {std::max(0.0, 3.2406 * X - 1.5372 * Y - 0.4986 * Z),
          std::max(0.0, -0.9689 * X + 1.8758 * Y + 0.0415 * Z),
          std::max(0.0, 0.0557 * X - 0.2040 * Y + 1.0570 * Z)};
}

}  // namespace bhs::blackbody
