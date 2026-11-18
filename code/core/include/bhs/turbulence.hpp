#pragma once
// Disk turbulence and hot spot (v0.9c), pure functions (the oracle of what the kernel does in float).
//
// Density: lognormal factor exp(a n - a^2 var / 2) with n = 3D fbm in a frame that ROTATES with the Kerr Omega(r)
// (phi' = phi - Omega(r) t): differential rotation; the cells have the size scale * H(r) (angle scale*h, ln r and mu
// scaled by scale*h; (cos phi', sin phi') guarantees continuity in phi). Hot spot: a Gaussian that orbits at r_h with
// Omega(r_h) (period 2 pi / Omega) and raises the emissivity: T -> T (1 + amp G)^(1/4), i.e. the bolometric
// radiance ~ T^4 rises by 1 + amp G; flare = Gaussian pulse in time on top of amp.
#include <cmath>
#include <cstdint>

namespace bhs::turb {

inline uint32_t hash3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t h = a * 0x9E3779B1u ^ b * 0x85EBCA6Bu ^ c * 0xC2B2AE35u;
  h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
  return h;
}
// 3D value noise in [0, 1] (same as render.cu's vnoise)
inline double vnoise(double x, double y, double z) {
  const int xi = int(std::floor(x)), yi = int(std::floor(y)), zi = int(std::floor(z));
  const double fx = x - xi, fy = y - yi, fz = z - zi;
  auto sm = [](double t) { return t * t * (3 - 2 * t); };
  double r = 0;
  for (int dz = 0; dz < 2; ++dz)
    for (int dy = 0; dy < 2; ++dy)
      for (int dx = 0; dx < 2; ++dx) {
        const double hv = double(hash3(uint32_t(xi + dx), uint32_t(yi + dy), uint32_t(zi + dz)) & 0xFFFFFF) / 16777216.0;
        r += hv * (dx ? sm(fx) : 1 - sm(fx)) * (dy ? sm(fy) : 1 - sm(fy)) * (dz ? sm(fz) : 1 - sm(fz));
      }
  return r;
}

constexpr int kOct = 3;
constexpr double kVar = 0.0584;  // variance of fbm, measured with tools/turb_var.cpp (2e6 points: 0.05844; g++ -O2 -std=c++20 -Icore/include tools/turb_var.cpp)

// fractional Brownian motion: 3 octaves, weights 1, 1/2, 1/4; distinct offsets per octave; result in [-1, 1]
inline double fbm(double x, double y, double z) {
  double s = 0, wsum = 0, w = 1.0, f = 1.0;
  for (int k = 0; k < kOct; ++k) {
    s += w * (2.0 * vnoise(f * x + 17.31 * k, f * y + 5.77 * k, f * z + 11.13 * k) - 1.0);
    wsum += w; w *= 0.5; f *= 2.0;
  }
  return s / wsum;
}

inline double omega(double r, double a) { return 1.0 / (std::pow(r, 1.5) + a); }

// n(r, phi, mu, t): noise coordinates in the rotating frame (see the header); hs = scale * H/r
inline double noise_n(double r, double phi, double mu, double t, double a, double hs) {
  const double pp = phi - omega(r, a) * t, inv = 1.0 / hs;
  return fbm(std::cos(pp) * inv, std::sin(pp) * inv, (std::log(r) + 1.7 * mu) * inv);
}
inline double density_factor(double r, double phi, double mu, double t, double a, double h, double amp, double scale) {
  if (amp <= 0.0) return 1.0;
  return std::exp(amp * noise_n(r, phi, mu, t, a, scale * h) - 0.5 * amp * amp * kVar);
}

struct Spot {
  double r = 10.0, phi0 = 0.0, size = 1.5, amp = 0.0;
  double fl_t = 0.0, fl_w = 1.0, fl_boost = 0.0;
};
inline double spot_phase(const Spot& s, double a, double t) { return s.phi0 + omega(s.r, a) * t; }
inline double spot_amp_t(const Spot& s, double t) {
  if (s.fl_boost == 0.0) return s.amp;
  const double d = (t - s.fl_t) / s.fl_w;
  return s.amp * (1.0 + s.fl_boost * std::exp(-0.5 * d * d));
}
// Gaussian (in Euclidean distance, M = 1) around the spot centre: d^2 = r^2 + r_h^2 - 2 r r_h cos(phi - phi_h) + z^2, z = r mu
inline double spot_gauss(const Spot& s, double a, double r, double phi, double mu, double t) {
  const double ph = spot_phase(s, a, t), z = r * mu;
  const double d2 = r * r + s.r * s.r - 2.0 * r * s.r * std::cos(phi - ph) + z * z;
  return std::exp(-0.5 * d2 / (s.size * s.size));
}
inline double spot_T_factor(const Spot& s, double a, double r, double phi, double mu, double t) {
  const double A = spot_amp_t(s, t);
  if (A <= 0.0) return 1.0;
  return std::pow(1.0 + A * spot_gauss(s, a, r, phi, mu, t), 0.25);
}

}  // namespace bhs::turb
