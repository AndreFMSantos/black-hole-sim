#pragma once
// Observer moving relative to the ZAMO: aberration and Doppler (local special relativity).
//
// In the local frame (e_r, e_theta, e_phi) of a ZAMO, the observer moves with velocity beta
// (vector, |beta| < 1, in units of c). A photon has propagation direction n' in the observer
// frame (unit) and n in the ZAMO's. With 4-momentum p = E (1, n):
//     E  = gamma E' (1 + beta . n')          p_par = gamma (p'_par + beta E')       p_perp = p'_perp
//     E' = gamma E  (1 - beta . n)           (inverse transformation)
// `dop` = E'/E (energy in the observer / energy in the ZAMO): > 1 if the observer moves towards the photon.
#include <cmath>

#include "kerr.hpp"
#include "vec3.hpp"

namespace bhs::lorentz {

struct Photon {
  Vec3 n;      // propagation direction, unit
  double dop;  // E_destination / E_origin... see each function
};

// n' (observer) -> n (ZAMO); dop = E'/E_zamo
inline Photon observer_to_zamo(Vec3 n_prime, Vec3 beta) {
  const double b2 = dot(beta, beta);
  if (b2 < 1e-30) return {n_prime, 1.0};
  const double gamma = 1.0 / std::sqrt(1.0 - b2);
  const double bn = dot(beta, n_prime);
  const double E = gamma * (1.0 + bn);                 // with E' = 1
  const Vec3 p = n_prime + (((gamma - 1.0) * bn / b2) + gamma) * beta;  // p' = n', E' = 1
  return {normalize(p), 1.0 / E};
}

// n (ZAMO) -> n' (observer); dop = E'/E_zamo
inline Photon zamo_to_observer(Vec3 n, Vec3 beta) {
  const double b2 = dot(beta, beta);
  if (b2 < 1e-30) return {n, 1.0};
  const double gamma = 1.0 / std::sqrt(1.0 - b2);
  const double bn = dot(beta, n);
  const double Ep = gamma * (1.0 - bn);                // with E = 1
  const Vec3 p = n + (((gamma - 1.0) * bn / b2) - gamma) * beta;
  return {normalize(p), Ep};
}

// Velocity (in c) of a prograde equatorial circular geodesic of radius r, relative to the ZAMO:
//   beta = varpi (Omega - omega) / alpha     (a = 0: sqrt(M/(r - 2M)))
inline double circular_beta(double a, double r, double theta) {
  const double s = std::sin(theta), mu = std::cos(theta);
  const double Sigma = r * r + a * a * mu * mu, Delta = kerr::delta(r, a);
  const double A = std::pow(r * r + a * a, 2) - a * a * Delta * s * s;
  const double alpha = std::sqrt(Sigma * Delta / A), omega = 2.0 * a * r / A, varpi = std::sqrt(A / Sigma) * s;
  const double Omega = 1.0 / (std::pow(r, 1.5) + a);
  return varpi * (Omega - omega) / alpha;
}

// Radial free fall from rest at infinity, measured by a static observer (Schwarzschild):
//   beta = sqrt(2M/r)   (for a != 0 it is an approximation: the ZAMO is not static)
inline double infall_beta(double r) { return std::sqrt(2.0 / r); }

// Exact radial free fall from rest at infinity in Kerr (E = 1, L = 0, Q = 0, so p_theta = 0 at any theta), measured by the ZAMO at (r, theta):
//   u^r = -sqrt(R)/Sigma with R = 2 r (r^2 + a^2),  ZAMO energy E_Z = 1/alpha,  beta = u^(r)/E_Z = sqrt(R/A) = sqrt(1 - Delta Sigma / A),  A = (r^2 + a^2)^2 - a^2 Delta sin^2(theta)
// (a = 0: sqrt(2/r)). Valid for every r > r+: beta -> 1 only at the horizon. Check: gamma = 1/sqrt(1 - beta^2) = 1/alpha = E_Z (energy conservation).
inline double infall_beta_kerr(double a, double r, double theta) {
  const double s = std::sin(theta), mu = std::cos(theta);
  const double Sigma = r * r + a * a * mu * mu, Delta = kerr::delta(r, a);
  const double A = std::pow(r * r + a * a, 2) - a * a * Delta * s * s;
  return std::sqrt(std::max(0.0, 1.0 - Delta * Sigma / A));
}

}  // namespace bhs::lorentz
