#pragma once
// Null geodesics (light) in the Schwarzschild equatorial plane, units G = c = M = 1.
//
// With u = 1/r and phi as the independent variable, the light trajectory obeys the
// relativistic Binet equation:   d^2u/dphi^2 + u = 3 M u^2
// (Newton would give u'' + u = 0: a straight line. The 3Mu^2 term is the curvature of spacetime.)
// First integral: (du/dphi)^2 = 1/b^2 - u^2 (1 - 2Mu), with b = L/E the impact parameter.
#include <algorithm>
#include <cmath>
#include <vector>

#include "vec2.hpp"

namespace bhs::schw {

constexpr double kM = 1.0;
constexpr double kHorizonU = 1.0 / (2.0 * kM);  // r = 2M
inline double critical_b() { return std::sqrt(27.0) * kM; }  // shadow radius

inline double binet_accel(double u) { return -u + 3.0 * kM * u * u; }

struct Ray {
  double u, w;  // u = 1/r, w = du/dphi
};

// RK4 em phi (u' = w, w' = -u + 3Mu^2).
inline Ray step_rk4(Ray s, double h) {
  const double k1u = s.w, k1w = binet_accel(s.u);
  const double k2u = s.w + 0.5 * h * k1w, k2w = binet_accel(s.u + 0.5 * h * k1u);
  const double k3u = s.w + 0.5 * h * k2w, k3w = binet_accel(s.u + 0.5 * h * k2u);
  const double k4u = s.w + h * k3w, k4w = binet_accel(s.u + h * k3u);
  return {s.u + (h / 6.0) * (k1u + 2 * k2u + 2 * k3u + k4u),
          s.w + (h / 6.0) * (k1w + 2 * k2w + 2 * k3w + k4w)};
}

enum class Fate { Escaped, Captured, Orbiting };  // Orbiting: still going round at max_phi

struct TraceResult {
  Fate fate;
  double phi_end;      // total swept angle (only if Escaped)
  double deflection;   // phi_end - pi (only if Escaped); straight line = 0
};

// Ray coming from infinity (u = 0) with impact parameter b > 0.
// `path` (optional) receives Cartesian points: the ray enters from the left (-x),
// with y = +b far from the hole; the hole is at the origin.
inline TraceResult trace(double b, double dphi = 1e-3, double max_phi = 400.0,
                         std::vector<Vec2>* path = nullptr) {
  Ray s{0.0, 1.0 / b};
  double phi = 0.0;
  auto emit = [&](const Ray& r, double ph) {
    if (path && r.u > 1.0 / 200.0) {
      const double rad = 1.0 / r.u;
      path->push_back({-rad * std::cos(ph), rad * std::sin(ph)});
    }
  };
  emit(s, phi);
  while (phi < max_phi) {
    const Ray n = step_rk4(s, dphi);
    phi += dphi;
    if (n.u >= kHorizonU) {
      emit({kHorizonU, 0}, phi);
      return {Fate::Captured, 0, 0};
    }
    if (n.w < 0 && n.u <= 0) {  // back at infinity: interpolates the phi of the crossing
      const double frac = s.u / (s.u - n.u);
      const double phi_end = phi - dphi + frac * dphi;
      return {Fate::Escaped, phi_end, phi_end - M_PI};
    }
    s = n;
    emit(s, phi);
  }
  return {Fate::Orbiting, 0, 0};
}

// Ray leaving a static observer at radius r0 with impact parameter b.
// `inward`: the ray direction has an inward radial component (u grows at the start).
// phi_end is the angle swept (in the ray plane) up to infinity, only if Escaped.
// Used to trace backwards: the direction the light CAME from.
inline TraceResult trace_from(double r0, double b, bool inward, double dphi = 5e-3,
                              double max_phi = 100.0) {
  const double u0 = 1.0 / r0;
  const double w2 = 1.0 / (b * b) - u0 * u0 * (1.0 - 2.0 * kM * u0);
  Ray s{u0, (inward ? 1.0 : -1.0) * std::sqrt(std::max(0.0, w2))};
  double phi = 0.0;
  while (phi < max_phi) {
    const Ray n = step_rk4(s, dphi);
    phi += dphi;
    if (n.u >= kHorizonU) return {Fate::Captured, 0, 0};
    if (n.w < 0 && n.u <= 0) {
      const double frac = s.u / (s.u - n.u);
      const double phi_end = phi - dphi + frac * dphi;
      return {Fate::Escaped, phi_end, phi_end - M_PI};
    }
    s = n;
  }
  return {Fate::Orbiting, 0, 0};
}

struct DiskHit {
  bool hit = false;
  double r = 0;    // radius of the crossing
  double phi = 0;  // angle in the ray plane (since the observer)
};
struct DiskTrace {
  Fate fate;
  double phi_end;  // only if Escaped
  DiskHit disk;    // first crossing of the disk plane with r_in < r < r_out
};

// Like trace_from, but also looks for the FIRST crossing of the disk plane (opaque disk).
// The disk plane cuts the ray plane at phi = phi_first + k*pi (k = 0, 1, ...), with
// phi_first in (0, pi]. At each crossing u is interpolated and r_in < r < r_out is tested.
// If it hits, the ray ends there (fate stays as it was: the caller uses disk.hit first).
inline DiskTrace trace_with_disk(double r0, double b, bool inward, double phi_first, double r_in,
                                 double r_out, double dphi = 5e-3, double max_phi = 100.0) {
  const double u0 = 1.0 / r0;
  const double w2 = 1.0 / (b * b) - u0 * u0 * (1.0 - 2.0 * kM * u0);
  Ray s{u0, (inward ? 1.0 : -1.0) * std::sqrt(std::max(0.0, w2))};
  double phi = 0.0, next_cross = phi_first;
  while (phi < max_phi) {
    const Ray n = step_rk4(s, dphi);
    const double phi_new = phi + dphi;
    while (next_cross <= phi_new) {  // crossing(s) of the disk plane in this step
      const double frac = (next_cross - phi) / dphi;
      const double uc = s.u + frac * (n.u - s.u);
      if (uc > 0) {
        const double rc = 1.0 / uc;
        if (rc > r_in && rc < r_out) return {Fate::Captured, 0, {true, rc, next_cross}};
      }
      next_cross += M_PI;
    }
    phi = phi_new;
    if (n.u >= kHorizonU) return {Fate::Captured, 0, {}};
    if (n.w < 0 && n.u <= 0) {
      const double frac = s.u / (s.u - n.u);
      return {Fate::Escaped, phi - dphi + frac * dphi, {}};
    }
    s = n;
  }
  return {Fate::Orbiting, 0, {}};
}

}  // namespace bhs::schw
