#pragma once
// HDR sky map (NASA SVS "Deep Star Maps 2020"): equirectangular projection (plate carree),
// ICRF/J2000, linear values stored in half (16 bit) RGB.
//
// Convention (the same as the map's and the one we use in the simulation world):
//   world +x axis = right ascension (RA) 0h ; +y = RA 6h (90 degrees) ; +z = north celestial pole.
//   RA = atan2(y, x) in [0, 2pi) ; declination dec = asin(z).
//   u = 0.5 - RA / (2 pi)   (RA 0h at the CENTRE of the image; RA grows to the LEFT)  -> wrap in u
//   v = 0.5 - dec / pi      (top of the image v = 0 = dec +90 degrees)                -> clamp in v
// Pixel coordinates: x = u * w - 0.5 (texel centres at integers), y = v * h - 0.5.
// So +x lands at the centre (0.5, 0.5), +y at u = 0.25, -x at u = 0 (= 1), +z on the top row.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "vec3.hpp"

namespace bhs::sky {

struct Skymap {
  int w = 0, h = 0;
  std::vector<uint16_t> rgb;  // w*h*3 halves, rows from top to bottom
  bool empty() const { return w == 0 || h == 0; }
};

inline float half_to_float(uint16_t hv) {
  const uint32_t s = (hv >> 15) & 1u, e = (hv >> 10) & 0x1Fu, m = hv & 0x3FFu;
  uint32_t f;
  if (e == 0) {
    if (m == 0) f = s << 31;
    else {  // subnormal: normalise
      int ee = -1;
      uint32_t mm = m;
      do { ++ee; mm <<= 1; } while (!(mm & 0x400u));
      f = (s << 31) | (uint32_t(127 - 15 - ee) << 23) | ((mm & 0x3FFu) << 13);
    }
  } else if (e == 31) f = (s << 31) | 0x7F800000u | (m << 13);
  else f = (s << 31) | ((e + 112u) << 23) | (m << 13);
  float out;
  std::memcpy(&out, &f, 4);
  return out;
}

// direction (any norm) -> (u, v); u in [0, 1), v in [0, 1]
inline void dir_to_uv(Vec3 d, double& u, double& v) {
  const Vec3 n = normalize(d);
  double ra = std::atan2(n.y, n.x);
  if (ra < 0) ra += 2.0 * M_PI;
  u = 0.5 - ra / (2.0 * M_PI);
  if (u < 0) u += 1.0;
  if (u >= 1.0) u -= 1.0;
  v = 0.5 - std::asin(std::clamp(n.z, -1.0, 1.0)) / M_PI;
}

inline Vec3 uv_to_dir(double u, double v) {
  const double ra = (0.5 - u) * 2.0 * M_PI, dec = (0.5 - v) * M_PI;
  return {std::cos(dec) * std::cos(ra), std::cos(dec) * std::sin(ra), std::sin(dec)};
}

inline Vec3 texel(const Skymap& m, int x, int y) {
  x %= m.w; if (x < 0) x += m.w;  // wrap em u
  y = std::clamp(y, 0, m.h - 1);   // clamp em v
  const uint16_t* p = &m.rgb[(size_t(y) * m.w + x) * 3];
  return {half_to_float(p[0]), half_to_float(p[1]), half_to_float(p[2])};
}

// Bilinear sampling (no mipmaps) in a world direction.
inline Vec3 sample(const Skymap& m, Vec3 dir) {
  double u, v;
  dir_to_uv(dir, u, v);
  const double fx = u * m.w - 0.5, fy = v * m.h - 0.5;
  const double x0 = std::floor(fx), y0 = std::floor(fy);
  const double tx = fx - x0, ty = fy - y0;
  const int ix = int(x0), iy = int(y0);
  return (1 - ty) * ((1 - tx) * texel(m, ix, iy) + tx * texel(m, ix + 1, iy)) +
         ty * ((1 - tx) * texel(m, ix, iy + 1) + tx * texel(m, ix + 1, iy + 1));
}

}  // namespace bhs::sky
