#pragma once
// Thin accretion disk, in the equatorial plane, on Keplerian circular orbits (Schwarzschild, M = 1).
#include <cmath>

namespace bhs::disk {

constexpr double kIsco = 6.0;  // ISCO de Schwarzschild = 6M

// Shakura-Sunyaev profile with zero torque at the inner edge:
//   T^4 ~ r^-3 (1 - sqrt(r_in/r))     (Newtonian form; no relativistic factors, v0.4)
// normalised to be 1 at the maximum (r = 49/36 r_in). Falls to 0 at r_in, ~ r^(-3/4) far away.
inline double temperature(double r, double r_in = kIsco) {
  auto f = [&](double x) { return std::pow(x, -3.0) * (1.0 - std::sqrt(r_in / x)); };
  if (r <= r_in) return 0.0;
  return std::pow(f(r) / f(49.0 / 36.0 * r_in), 0.25);
}

// Shift factor g = nu_obs / nu_emit for an emitter on a circular orbit of radius r and a
// static observer at radius D. lz_over_e = L_z/E of the photon (angular momentum about the disk
// axis, per unit energy, with the sign of the photon travelling FROM the emitter TO the observer).
//   g = sqrt(1 - 3M/r) / ( sqrt(1 - 2M/D) * (1 - Omega L_z/E) ),   Omega = sqrt(M / r^3)
inline double redshift(double r, double D, double lz_over_e) {
  const double omega = std::sqrt(1.0 / (r * r * r));
  return std::sqrt(1.0 - 3.0 / r) / (std::sqrt(1.0 - 2.0 / D) * (1.0 - omega * lz_over_e));
}

}  // namespace bhs::disk
