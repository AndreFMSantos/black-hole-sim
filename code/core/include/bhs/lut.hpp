#pragma once
// Black-body colour LUT (absolute radiance) log-spaced in T, in a format ready for a
// 1D texture (RGB32F) and with the same linear interpolation used by v0.4 on the CPU.
#include <algorithm>
#include <cmath>
#include <vector>

#include "blackbody.hpp"
#include "disk_emission.hpp"

namespace bhs::lut {

constexpr int kN = 1024;
constexpr double kT0 = 500.0, kT1 = 100000.0;

inline double temperature_at(int i) { return kT0 * std::pow(kT1 / kT0, double(i) / (kN - 1)); }

// kN * 3 floats (r, g, b per entry).
// fcol = spectral hardening factor (1 = pure black body, same as v0.8; see disk_emission.hpp).
inline std::vector<float> build(double fcol = 1.0) {
  std::vector<float> d(size_t(kN) * 3);
  for (int i = 0; i < kN; ++i) {
    const Vec3 c = emission::rgb_hardened(temperature_at(i), fcol);
    d[3 * i] = float(c.x); d[3 * i + 1] = float(c.y); d[3 * i + 2] = float(c.z);
  }
  return d;
}

// CPU reference (same as v0.4): linear interpolation in log T. The GPU does the same with
// texture() + GL_LINEAR at (x + 0.5)/kN.
inline Vec3 lookup(const std::vector<float>& d, double T) {
  if (T <= kT0) return {0, 0, 0};
  const double x = std::log(std::min(T, kT1) / kT0) / std::log(kT1 / kT0) * (kN - 1);
  const int i = std::min(kN - 2, int(x));
  const double f = x - i;
  auto at = [&](int k) { return Vec3{d[3 * k], d[3 * k + 1], d[3 * k + 2]}; };
  return (1 - f) * at(i) + f * at(i + 1);
}

}  // namespace bhs::lut
