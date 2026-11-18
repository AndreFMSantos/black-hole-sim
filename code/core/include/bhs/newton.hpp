#pragma once
#include "vec2.hpp"

namespace bhs {

// Units: GM = 1 by default. Central body at the origin.
constexpr double kGM = 1.0;

inline Vec2 gravity(Vec2 pos, double gm = kGM) {
  const double r = norm(pos);
  return (-gm / (r * r * r)) * pos;  // a = -GM r_vec / |r|^3
}

struct State {
  Vec2 x, v;
};

// Specific energy (per unit mass): v^2/2 - GM/r. Conserved.
inline double energy(const State& s, double gm = kGM) {
  return 0.5 * dot(s.v, s.v) - gm / norm(s.x);
}
// Specific angular momentum (z): x*vy - y*vx. Conserved (central force).
inline double angular_momentum(const State& s) { return cross(s.x, s.v); }

// Circular orbit of radius r: v = sqrt(GM/r), period T = 2*pi*sqrt(r^3/GM).
inline State circular_orbit(double r, double gm = kGM) {
  return {{r, 0}, {0, std::sqrt(gm / r)}};
}

}  // namespace bhs
