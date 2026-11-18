#pragma once
// Lyapunov exponent of the equatorial circular photon orbit in Kerr.
// Near the double root of R(r), in Mino time: dr/dtau = sqrt(R) ~ kappa (r - r_ph), kappa = sqrt(R''/2);
// the azimuth advances at dphi/dtau = Phi(r_ph). A half turn (delta phi = pi) multiplies the deviation by
//     e^{gamma},   gamma = pi * kappa / |Phi|           (Schwarzschild: gamma = pi)
// Successive photon rings shrink by e^{-gamma} (Johnson et al. 2020, Gralla-Holz-Wald 2019).
#include <cmath>

#include "kerr.hpp"

namespace bhs::kerr {

inline double d2R(double r, const Consts& c) {
  const double A = r * r + c.a * c.a - c.a * c.L, K = (c.L - c.a) * (c.L - c.a) + c.Q;
  return 4.0 * A + 8.0 * r * r - 2.0 * K;
}

inline double lyapunov_equatorial(double a, bool prograde) {
  const double r = photon_orbit(a, prograde);
  const double L = (a == 0.0) ? (prograde ? std::sqrt(27.0) : -std::sqrt(27.0)) : xi_photon(r, a);
  const Consts c{a, L, 0.0};
  const double kappa = std::sqrt(0.5 * d2R(r, c));
  const double phi = std::abs(dphi_dtau(r, 0.0, c));
  return M_PI * kappa / phi;
}

// n_phi (azimuthal component of the photon direction in the ZAMO) that produces a given L = L_z/E (E = 1) for a
// ZAMO observer at (r, theta): inverse of  L = varpi E_loc n_phi,  E_loc = 1/(alpha + omega varpi n_phi).
inline double nphi_for_L(double a, double r, double theta, double L) {
  const double s = std::sin(theta), mu = std::cos(theta);
  const double Sigma = r * r + a * a * mu * mu, Delta = delta(r, a);
  const double A = std::pow(r * r + a * a, 2) - a * a * Delta * s * s;
  const double alpha = std::sqrt(Sigma * Delta / A), omega = 2.0 * a * r / A, varpi = std::sqrt(A / Sigma) * s;
  return L * alpha / (varpi * (1.0 - omega * L));
}

}  // namespace bhs::kerr
