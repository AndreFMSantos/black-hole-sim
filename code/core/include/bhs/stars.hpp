#pragma once
// Procedural celestial background: stars + coordinate grid (az/el every 15 degrees).
// The grid is what makes gravitational lensing legible before there are real images.
#include <algorithm>
#include <cmath>
#include <cstdint>

#include "vec3.hpp"

namespace bhs::stars {

inline uint32_t hash(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t h = a * 0x9E3779B1u ^ b * 0x85EBCA6Bu ^ c * 0xC2B2AE35u;
  h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
  return h;
}
inline double unit(uint32_t h) { return (h & 0xFFFFFF) / 16777216.0; }

// Unit direction v -> linear RGB colour (>= 0).
inline Vec3 sample(Vec3 v, bool grid = true) {
  Vec3 col{0.004, 0.005, 0.012};  // almost black sky, slightly blue

  // stars: cube projection, N x N cells per face, ~6% of the cells have a star
  const double ax = std::abs(v.x), ay = std::abs(v.y), az = std::abs(v.z);
  int face; double u, w, m;
  if (ax >= ay && ax >= az) { face = v.x > 0 ? 0 : 1; m = ax; u = v.y; w = v.z; }
  else if (ay >= az)        { face = v.y > 0 ? 2 : 3; m = ay; u = v.x; w = v.z; }
  else                      { face = v.z > 0 ? 4 : 5; m = az; u = v.x; w = v.y; }
  u /= m; w /= m;  // [-1, 1]
  constexpr int N = 220;
  const double fu = (u + 1) * 0.5 * N, fw = (w + 1) * 0.5 * N;
  const int cu = std::clamp(static_cast<int>(fu), 0, N - 1), cw = std::clamp(static_cast<int>(fw), 0, N - 1);
  const uint32_t h = hash(face, cu, cw);
  if (unit(h) < 0.06) {
    const double su = cu + 0.2 + 0.6 * unit(hash(h, 1, 2)), sw = cw + 0.2 + 0.6 * unit(hash(h, 3, 4));
    const double d2 = (fu - su) * (fu - su) + (fw - sw) * (fw - sw);
    const double mag = 0.25 + 0.75 * unit(hash(h, 5, 6));
    const double bright = mag * mag * std::exp(-d2 / 0.018);
    const double t = unit(hash(h, 7, 8));  // temperature: bluish <-> orange
    col = col + bright * Vec3{0.8 + 0.4 * t, 0.9, 1.3 - 0.5 * t};
  }

  if (grid) {
    const double az_deg = std::atan2(v.y, v.x) * 180.0 / M_PI;
    const double el_deg = std::asin(std::clamp(v.z, -1.0, 1.0)) * 180.0 / M_PI;
    auto near_line = [](double deg, double step, double width) {
      const double d = std::abs(deg - step * std::round(deg / step));
      return d < width;
    };
    const double cosel = std::max(0.2, std::sqrt(1 - v.z * v.z));
    if (near_line(az_deg, 15.0, 0.12 / cosel)) col = col + Vec3{0.05, 0.10, 0.16};
    if (near_line(el_deg, 15.0, 0.12)) col = col + Vec3{0.05, 0.10, 0.16};
    if (near_line(el_deg, 90.0, 0.12) || (std::abs(el_deg) < 0.12)) col = col + Vec3{0.22, 0.12, 0.04};  // equator
  }
  return col;
}

}  // namespace bhs::stars
