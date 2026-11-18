#pragma once
// Schwarzschild-de Sitter (Kottler), v2.1c: f(r) = 1 - 2M/r - Lambda r^2/3, 0 < Lambda M^2 < 1/9. M and Lambda in the same system of units (G = c = 1).
// Sources: Perlick, Tsupko & Bisnovatyi-Kogan 2018 (arXiv:1804.04898; partly read by a subagent) and own derivation:
//  - orbit: (du/dphi)^2 = 1/b^2 + Lambda/3 - u^2 + 2 M u^3  (u = 1/r, b = L/E): the Binet equation d^2u/dphi^2 = -u + 3 M u^2 does NOT depend on Lambda;
//    only the constant 1/b^2 + Lambda/3 changes. Hence, in the (r, phi) plane, the SdS ray with impact parameter b is the Schwarzschild one with b_eff, 1/b_eff^2 = 1/b^2 + Lambda/3;
//  - photon sphere r = 3M (independent of Lambda); shadow b_c^2 = 27 M^2 / (1 - 9 Lambda M^2);
//  - static observer at r_O (between the horizons): sin^2(theta) = 27 M^2 f(r_O) / [r_O^2 (1 - 9 Lambda M^2)] (= b_c^2 f / r_O^2).
#include <algorithm>
#include <cmath>

namespace bhs::sds {

inline double f(double r, double M, double Lambda) { return 1.0 - 2.0 * M / r - Lambda * r * r / 3.0; }

// horizons: r_h (black hole) and r_c (cosmological), positive roots of Lambda r^3/3 - r + 2M = 0 (trigonometric method; 2M < r_h < 3M < r_c)
inline bool horizons(double M, double Lambda, double& r_h, double& r_c) {
  const double x = -3.0 * M * std::sqrt(Lambda);
  if (!(Lambda > 0.0) || x <= -1.0) return false;           // Lambda M^2 < 1/9
  const double th = std::acos(x) / 3.0, k = 2.0 / std::sqrt(Lambda);
  double r[3];
  for (int i = 0; i < 3; ++i) r[i] = k * std::cos(th - 2.0 * M_PI * i / 3.0);
  std::sort(r, r + 3);
  r_h = r[1]; r_c = r[2];                                    // r[0] < 0
  return r_h > 0.0;
}
inline double shadow_b(double M, double Lambda) { return std::sqrt(27.0) * M / std::sqrt(1.0 - 9.0 * Lambda * M * M); }
inline double static_shadow_sin(double M, double Lambda, double r_O) {
  return std::sqrt(27.0 * M * M * f(r_O, M, Lambda) / (r_O * r_O * (1.0 - 9.0 * Lambda * M * M)));
}

}  // namespace bhs::sds
