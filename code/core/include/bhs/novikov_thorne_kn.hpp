#pragma once
// Novikov-Thorne / Page-Thorne thin disk in Kerr-Newman (M = 1; a >= 0, q2 = Q^2, a^2 + q2 <= 1), equatorial circular orbits of NEUTRAL particles,
// inner edge at the ISCO, zero torque (v2.1e). Generalises nt:: (Kerr) without Kerr closed formulas:
//   equatorial metric: N = 2 r - q2;  g_tt = -1 + N/r^2,  g_tphi = -a N/r^2,  g_phiphi = r^2 + a^2 + a^2 N/r^2   (derived from the Boyer-Lindquist form of KN;
//   det of the (t, phi) block = -Delta, Delta = r^2 - 2 r + a^2 + q2; g_rr = r^2/Delta; g_zz = 1; hence e^{nu+psi+mu} = r as in Kerr, verified)
//   Omega_pm = [ -g_tphi,r +- sqrt(g_tphi,r^2 - g_tt,r g_phiphi,r) ] / g_phiphi,r  (Page & Thorne 1974, eq. 7; same form in Hassani 2026 eq. 16)
//   u^t = 1/sqrt(-g_tt - 2 Omega g_tphi - Omega^2 g_phiphi),  E = -(g_tt + Omega g_tphi) u^t,  L = (g_tphi + Omega g_phiphi) u^t
//   ISCO = minimum of E(r) (the outermost): dE/dr = dL/dr = 0 (Page-Thorne eq. 13); no closed formula in KN (Pugliese et al. 2013 only give bounds).
//   flux (Page-Thorne 11b + 12): F = 1/(4 pi r) (-Omega_,r) (E - Omega L)^-2 Int_{r_isco}^{r} (E - Omega L) L_,r dr'  per unit Mdot.
// Independent verification: energy conservation Int 4 pi r E F dr = 1 - E_isco (valid in asymptotically flat spacetimes) and the Kerr (q2 = 0) and RN (a = 0) limits.
#include <algorithm>
#include <cmath>
#include <vector>

#include "kerr.hpp"
#include "novikov_thorne.hpp"

namespace bhs::ntk {

struct EqMetric { double gtt, gtp, gpp, dgtt, dgtp, dgpp; };
inline EqMetric eq_metric(double r, double a, double q2) {
  const double N = 2.0 * r - q2, r2 = r * r, dN_r2 = 2.0 * (q2 - r) / (r2 * r);  // d(N/r^2)/dr
  return {-1.0 + N / r2, -a * N / r2, r2 + a * a + a * a * N / r2, dN_r2, -a * dN_r2, 2.0 * r + a * a * dN_r2};
}

struct Circ { double Omega, ut, E, L; bool ok; };
inline Circ circ(double r, double a, double q2, bool prograde = true) {
  const auto m = eq_metric(r, a, q2);
  const double disc = m.dgtp * m.dgtp - m.dgtt * m.dgpp;
  if (!(disc >= 0.0)) return {0, 0, 0, 0, false};
  const double sq = std::sqrt(disc), Om = (-m.dgtp + (prograde ? sq : -sq)) / m.dgpp;
  const double den = -m.gtt - 2.0 * Om * m.gtp - Om * Om * m.gpp;
  if (!(den > 0.0)) return {Om, 0, 0, 0, false};
  const double ut = 1.0 / std::sqrt(den);
  return {Om, ut, -(m.gtt + Om * m.gtp) * ut, (m.gtp + Om * m.gpp) * ut, true};
}

// 5-point central derivative (error ~ (h/r)^4 ~ 1e-12 with h = 1e-3 r)
template <class F>
inline double deriv(F f, double r) {
  const double h = 1e-3 * r;
  return (-f(r + 2 * h) + 8 * f(r + h) - 8 * f(r - h) + f(r - 2 * h)) / (12 * h);
}

// Prograde ISCO: the outermost radius where dE/dr changes sign (E grows with r outside the ISCO). At extremes (a^2 + q2 = 1) the ISCO tends to the horizon (= 1).
inline double isco(double a, double q2, bool prograde = true) {
  const double rh = kerr::horizon(a, q2);
  auto dE = [&](double r) {
    const auto c0 = circ(r, a, q2, prograde);
    if (!c0.ok) return -1.0;  // below the circular photon orbit: treat as "inside"
    return deriv([&](double x) { return circ(x, a, q2, prograde).E; }, r);
  };
  const double lo_floor = std::max(rh * (1.0 + 1e-6), 1e-3);
  double hi = 30.0, lo = hi;
  for (double r = 30.0; r > lo_floor; r *= 0.995) {
    if (dE(r) <= 0.0) { lo = r; hi = r / 0.995; break; }
    lo = r;
  }
  if (!(dE(lo) <= 0.0)) return lo_floor;  // no minimum before the horizon
  for (int i = 0; i < 200; ++i) { const double m = 0.5 * (lo + hi); (dE(m) <= 0.0 ? lo : hi) = m; if (hi - lo < 1e-13 * hi) break; }
  return 0.5 * (lo + hi);
}

inline double flux(double r, double a, double q2, double rin) {
  if (r <= rin) return 0.0;
  auto Eo = [&](double x) { const auto c = circ(x, a, q2); return c.E - c.Omega * c.L; };
  auto dL = [&](double x) { return deriv([&](double y) { return circ(y, a, q2).L; }, x); };
  auto g = [&](double x) { return Eo(x) * dL(x); };
  double sum = 0, lo = rin;
  const int pieces = std::max(1, int(std::ceil(std::log(r / rin) / std::log(1.6))));
  for (int k = 1; k <= pieces; ++k) {
    const double hi = (k == pieces) ? r : rin * std::pow(r / rin, double(k) / pieces);
    sum += nt::integrate<24>(g, lo, hi);
    lo = hi;
  }
  const double dOm = deriv([&](double y) { return circ(y, a, q2).Omega; }, r), eo = Eo(r);
  return (1.0 / (4.0 * M_PI * r)) * (-dOm) / (eo * eo) * sum;
}

// Total luminosity at infinity per unit Mdot (expected: 1 - E_isco)
inline double luminosity(double a, double q2, double r_max = 1e8) {
  const double rin = isco(a, q2);
  const double x0 = std::log(rin), x1 = std::log(r_max);
  double L = 0;
  const int pieces = 80;
  for (int k = 0; k < pieces; ++k) {
    const double lo = x0 + (x1 - x0) * k / pieces, hi = x0 + (x1 - x0) * (k + 1) / pieces;
    L += nt::integrate<16>([&](double x) { const double r = std::exp(x); return 4.0 * M_PI * r * r * circ(r, a, q2).E * flux(r, a, q2, rin); }, lo, hi);
  }
  return L + 1.5 / r_max;
}

// g = nu_obs / nu_emit: emitter on a prograde circular orbit (radius r), KN ZAMO observer at (r_obs, theta_obs); L = L_z/E of the photon
inline double redshift(double a, double q2, double r, double L, double r_obs, double theta_obs) {
  const double s = std::sin(theta_obs), mu = std::cos(theta_obs);
  const double Sigma = r_obs * r_obs + a * a * mu * mu, Delta = kerr::delta(r_obs, a, q2);
  const double A = std::pow(r_obs * r_obs + a * a, 2) - a * a * Delta * s * s;
  const double alpha = std::sqrt(Sigma * Delta / A), omega = a * (2.0 * r_obs - q2) / A;
  const double E_obs = (1.0 - omega * L) / alpha;
  const auto c = circ(r, a, q2);
  return E_obs / (c.ut * (1.0 - c.Omega * L));
}

// Normalised temperature profile (T ~ F^(1/4)) in ln r, like nt::Profile
class Profile {
 public:
  Profile(double a, double q2, double r_out = 1e3, int n = 2048) : rin_(isco(a, q2)), rout_(r_out), tab_(n) {
    const double lmin = std::log(rin_), lmax = std::log(r_out);
    double fmax = 0;
    for (int i = 0; i < n; ++i) {
      const double r = std::exp(lmin + (lmax - lmin) * i / (n - 1));
      tab_[i] = std::max(0.0, flux(r, a, q2, rin_));  // rounding can make the first entries (r ~ r_in) slightly negative -> pow(neg, 0.25) = NaN (27 % of (a, Q) pairs; 2026-11-16 sweep)
      if (tab_[i] > fmax) { fmax = tab_[i]; rpeak_ = r; }
    }
    for (auto& f : tab_) f = std::pow(f / fmax, 0.25);
  }
  double r_peak() const { return rpeak_; }
  double r_in() const { return rin_; }
  const std::vector<double>& table() const { return tab_; }

 private:
  double rin_, rout_, rpeak_ = 0;
  std::vector<double> tab_;
};

}  // namespace bhs::ntk
