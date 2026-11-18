#pragma once
// Novikov-Thorne / Page-Thorne thin disk around a Kerr black hole (M = 1, a >= 0,
// prograde equatorial circular orbits, inner edge at the ISCO, zero torque at the inner edge).
//
// The radiative flux (energy emitted per unit proper area and proper time, per unit
// of Mdot) is computed from the DEFINITION, with no memorised closed formula:
//     F(r) = [ 1 / (4 pi sqrt(-g)) ] * ( -Omega_{,r} ) / (E - Omega L)^2 * Int_{r_in}^{r} (E - Omega L) L_{,r} dr'
// with sqrt(-g) = r in the Kerr equatorial plane (Page-Thorne coordinates) and E, L, Omega the specific energy,
// angular momentum and angular velocity of the circular orbit. The independent
// check is energy conservation: Int 4 pi r E F dr = 1 - E_isco.
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "kerr.hpp"

namespace bhs::nt {

inline double denom(double r, double a) { return std::pow(r, 0.75) * std::sqrt(std::pow(r, 1.5) - 3.0 * std::sqrt(r) + 2.0 * a); }
inline double energy(double r, double a) { return (std::pow(r, 1.5) - 2.0 * std::sqrt(r) + a) / denom(r, a); }
inline double ang_mom(double r, double a) { return (r * r - 2.0 * a * std::sqrt(r) + a * a) / denom(r, a); }
inline double omega(double r, double a) { return 1.0 / (std::pow(r, 1.5) + a); }
inline double domega(double r, double a) { return -1.5 * std::sqrt(r) / std::pow(std::pow(r, 1.5) + a, 2.0); }
inline double dang_mom(double r, double a) {  // numerical derivative (central, relative): error ~1e-10
  const double h = 1e-5 * r;
  return (ang_mom(r + h, a) - ang_mom(r - h, a)) / (2.0 * h);
}

// Gauss-Legendre nodes and weights in [-1, 1] (Newton on P_n)
template <int N>
inline const std::array<std::array<double, 2>, N>& gauss_legendre() {
  static const auto table = [] {
    std::array<std::array<double, 2>, N> t{};
    for (int i = 0; i < N; ++i) {
      double x = std::cos(M_PI * (i + 0.75) / (N + 0.5)), dp = 1;
      for (int it = 0; it < 100; ++it) {
        double p0 = 1, p1 = x;
        for (int k = 2; k <= N; ++k) { const double p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k; p0 = p1; p1 = p2; }
        dp = N * (x * p1 - p0) / (x * x - 1);
        const double dx = p1 / dp;
        x -= dx;
        if (std::abs(dx) < 1e-15) break;
      }
      t[i] = {x, 2.0 / ((1 - x * x) * dp * dp)};
    }
    return t;
  }();
  return table;
}

// Int_{lo}^{hi} f(x) dx with N-point GL
template <int N, class F>
inline double integrate(F f, double lo, double hi) {
  const double m = 0.5 * (hi + lo), h = 0.5 * (hi - lo);
  double s = 0;
  for (auto& [x, w] : gauss_legendre<N>()) s += w * f(m + h * x);
  return s * h;
}

inline double flux(double r, double a) {
  const double rin = kerr::isco(a);
  if (r <= rin) return 0.0;
  auto g = [&](double x) { return (energy(x, a) - omega(x, a) * ang_mom(x, a)) * dang_mom(x, a); };
  // the integrand is smooth but varies a lot in scale: split [r_in, r] into geometric segments
  double sum = 0, lo = rin;
  const int pieces = std::max(1, int(std::ceil(std::log(r / rin) / std::log(1.6))));
  for (int k = 1; k <= pieces; ++k) {
    const double hi = (k == pieces) ? r : rin * std::pow(r / rin, double(k) / pieces);
    sum += integrate<24>(g, lo, hi);
    lo = hi;
  }
  const double eo = energy(r, a) - omega(r, a) * ang_mom(r, a);
  return (1.0 / (4.0 * M_PI * r)) * (-domega(r, a)) / (eo * eo) * sum;
}

// Total luminosity at infinity per unit Mdot: Int 4 pi r E F dr (expected: 1 - E_isco).
inline double luminosity(double a, double r_max = 1e8) {
  const double rin = kerr::isco(a);
  const double x0 = std::log(rin), x1 = std::log(r_max);
  double L = 0;
  const int pieces = 80;
  for (int k = 0; k < pieces; ++k) {
    const double lo = x0 + (x1 - x0) * k / pieces, hi = x0 + (x1 - x0) * (k + 1) / pieces;
    L += integrate<16>([&](double x) { const double r = std::exp(x); return 4.0 * M_PI * r * r * energy(r, a) * flux(r, a); }, lo, hi);
  }
  return L + 1.5 / r_max;  // Newtonian tail: Int_{R}^{inf} 4 pi r F dr = 3/(2R)
}

// Normalised temperature profile (T ~ F^(1/4)), tabulated in ln r for fast use and for the GPU.
class Profile {
 public:
  explicit Profile(double a, double r_out = 1e3, int n = 2048) : rin_(kerr::isco(a)), rout_(r_out), tab_(n) {
    const double lmin = std::log(rin_), lmax = std::log(r_out);
    double fmax = 0;
    for (int i = 0; i < n; ++i) {
      const double r = std::exp(lmin + (lmax - lmin) * i / (n - 1));
      tab_[i] = std::max(0.0, flux(r, a));  // rounding near r_in can give a tiny negative flux -> pow(neg, 0.25) = NaN (11 % of the spins; 2026-11-16 sweep)
      if (tab_[i] > fmax) { fmax = tab_[i]; rpeak_ = r; }
    }
    for (auto& f : tab_) f = std::pow(f / fmax, 0.25);
  }
  double temperature(double r) const {
    if (r <= rin_) return 0.0;
    if (r >= rout_) return tab_.back() * std::pow(r / rout_, -0.75);
    const double lmin = std::log(rin_), lmax = std::log(rout_);
    const double x = (std::log(r) - lmin) / (lmax - lmin) * (tab_.size() - 1);
    const int i = std::min<int>(int(tab_.size()) - 2, int(x));
    const double f = x - i;
    return (1 - f) * tab_[i] + f * tab_[i + 1];
  }
  double r_peak() const { return rpeak_; }
  double r_in() const { return rin_; }
  const std::vector<double>& table() const { return tab_; }

 private:
  double rin_, rout_, rpeak_ = 0;
  std::vector<double> tab_;
};

}  // namespace bhs::nt
