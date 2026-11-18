#pragma once
// Photon rings (Schwarzschild): order-n images of the disk.
//
// Observer on the axis (top view): the disk plane cuts the ray plane at phi = pi/2 + k pi
// (k = 0, 1, ...). A ray with total sweep phi_total crosses the plane
//     n = floor((phi_total - pi/2)/pi) + 1   times.
// n = 1: direct image; n = 2: second-order image (the back face of the disk); n >= 3: photon
// ring. Near the critical impact parameter b_c, phi_total ~ -ln(b - b_c) + const, so the
// boundaries b_k (phi_total = pi/2 + k pi) approach b_c like exp(-pi k):
//     (b_{k+1} - b_c) / (b_k - b_c) -> exp(-gamma),  gamma = pi (Lyapunov exponent, Schwarzschild).
#include <cmath>

#include "schwarzschild.hpp"

namespace bhs::rings {

struct Crossings {
  int n = 0;
  schw::Fate fate = schw::Fate::Orbiting;
};

// Counts the plane crossings (any ray) of a ray coming from r0 with impact parameter b,
// moving away from the observer inwards (inward). phi_first = angle of the 1st crossing.
inline Crossings count_crossings(double r0, double b, double phi_first, double dphi = 1e-3, double max_phi = 200.0) {
  const double u0 = 1.0 / r0;
  const double w2 = 1.0 / (b * b) - u0 * u0 * (1.0 - 2.0 * schw::kM * u0);
  schw::Ray s{u0, std::sqrt(std::max(0.0, w2))};
  double phi = 0.0, next = phi_first;
  Crossings c;
  while (phi < max_phi) {
    const schw::Ray n = schw::step_rk4(s, dphi);
    const double phi_new = phi + dphi;
    while (next <= phi_new) { ++c.n; next += M_PI; }
    phi = phi_new;
    if (n.u >= schw::kHorizonU) { c.fate = schw::Fate::Captured; return c; }
    if (n.w < 0 && n.u <= 0) { c.fate = schw::Fate::Escaped; return c; }
    s = n;
  }
  return c;
}

// b (>= b_c) for which the total sweep is pi/2 + k pi (outer boundary of the order-k image).
inline double ring_boundary(double r0, int k, double dphi = 1e-3) {
  const double target = M_PI / 2 + k * M_PI;
  auto phi_total = [&](double b) {
    const auto t = schw::trace_from(r0, b, true, dphi, 400.0);
    return t.fate == schw::Fate::Escaped ? t.phi_end : 1e9;  // orbiting/captured: "infinite" sweep
  };
  double lo = schw::critical_b() * (1.0 + 1e-14), hi = schw::critical_b() * 1.5;  // phi(lo) > target > phi(hi)
  for (int i = 0; i < 100; ++i) {
    const double mid = 0.5 * (lo + hi);
    (phi_total(mid) > target ? lo : hi) = mid;
  }
  return 0.5 * (lo + hi);
}

}  // namespace bhs::rings
