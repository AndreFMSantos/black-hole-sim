#pragma once
// Orbital camera around the black hole (static observer at r = dist, M = 1).
// c_hat (hole -> observer) = (sin th cos ph, sin th sin ph, cos th). With ph = 0 it is the
// v0.4 camera (observer in the x-z plane). The (right, up) basis rotates with the azimuth around z.
#include <algorithm>
#include <cmath>

#include "vec3.hpp"

namespace bhs::camera {

constexpr double kMinDist = 10.0, kMaxDist = 400.0;
constexpr double kMinTheta = 0.5 * M_PI / 180.0, kMaxTheta = 179.5 * M_PI / 180.0;  // avoids the pole

struct Orbit {
  double theta = 80.0 * M_PI / 180.0;  // polar, from +z
  double phi = 0.0;                    // azimuth
  double dist = 40.0;

  void clamp() {
    theta = std::clamp(theta, kMinTheta, kMaxTheta);
    dist = std::clamp(dist, kMinDist, kMaxDist);
    phi = std::remainder(phi, 2.0 * M_PI);
  }
  Vec3 c_hat() const {
    return {std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)};
  }
  // Same convention as v0.4: fwd = -c_hat, right = fwd x z, up = right x fwd.
  void basis(Vec3& fwd, Vec3& right, Vec3& up) const {
    fwd = -1.0 * c_hat();
    right = normalize(cross(fwd, Vec3{0, 0, 1}));
    up = cross(right, fwd);
  }
};

}  // namespace bhs::camera
