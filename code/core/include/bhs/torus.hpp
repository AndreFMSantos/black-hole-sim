#pragma once
// Fishbone-Moncrief torus (1976): analytic hydrostatic equilibrium of a polytropic fluid with constant specific angular momentum
// l = -u_phi/u_t around a Kerr black hole (M = 1, Boyer-Lindquist coordinates). The simulator's first non-thin "DiskModel"
// (v2.2b): analytic pressure, density and temperature, the oracle of the future GRMHD-simulation reader.
//
// Fluid: u_mu = (u_t, 0, 0, u_phi), u_phi = -l u_t  =>  u_t^2 (g^tt - 2 l g^tphi + l^2 g^phiphi) = -1.
// Enthalpy: h = |u_t|_in / |u_t|(r, theta)  (constant l, isentropic; h = 1 at the inner edge, h > 1 inside the torus).
// Polytropic P = K rho^Gamma:  h = 1 + Gamma/(Gamma-1) P/rho  =>  P/rho = (Gamma-1)(h-1)/Gamma,  rho ~ (h-1)^(1/(Gamma-1)).
// The density is normalised to 1 at the pressure maximum (r_max, equator); l = l_Keplerian(r_max), so that is where the maximum is.
#include <cmath>

namespace bhs::torus {

struct Metric { double gtt, gtp, gpp; };  // the (t, phi) block of the Kerr metric (BL)

inline Metric metric(double a, double r, double th) {
  const double s2 = std::sin(th) * std::sin(th), Sig = r * r + a * a * std::cos(th) * std::cos(th);
  return {-(1.0 - 2.0 * r / Sig), -2.0 * a * r * s2 / Sig, (r * r + a * a + 2.0 * a * a * r * s2 / Sig) * s2};
}

// specific angular momentum of the prograde equatorial circular orbit of radius r (Fishbone-Moncrief)
inline double ell_kepler(double r, double a) {
  return (r * r - 2.0 * a * std::sqrt(r) + a * a) / (std::pow(r, 1.5) - 2.0 * std::sqrt(r) + a);
}

struct U4 { double ut, uphi; };

struct Torus {
  double a = 0.9, r_in = 6, r_max = 12, gamma = 4.0 / 3.0;
  double ell = 0, ut_in = 0, h_max = 1;

  // |u_t| (> 0) at the point; 0 if the fluid with this l is not timelike there (outside the domain)
  double abs_ut(double r, double th) const {
    const Metric m = metric(a, r, th);
    const double det = m.gtt * m.gpp - m.gtp * m.gtp;
    const double itt = m.gpp / det, itp = -m.gtp / det, ipp = m.gtt / det;  // inverse of the 2x2 block
    const double D = itt - 2.0 * ell * itp + ell * ell * ipp;
    return D < 0.0 ? 1.0 / std::sqrt(-D) : 0.0;
  }
  double enthalpy(double r, double th) const {
    const double u = abs_ut(r, th);
    return u > 0.0 ? ut_in / u : 0.0;
  }
  double density(double r, double th) const {
    const double h = enthalpy(r, th);
    return h > 1.0 ? std::pow((h - 1.0) / (h_max - 1.0), 1.0 / (gamma - 1.0)) : 0.0;
  }
  // P / (rho c^2) = ion temperature in units of m_p c^2 (ideal gas)
  double theta_ion(double r, double th) const {
    const double h = enthalpy(r, th);
    return h > 1.0 ? (gamma - 1.0) / gamma * (h - 1.0) : 0.0;
  }
  // contravariant 4-velocity (u^t, u^phi) with u^t > 0
  U4 four_velocity(double r, double th) const {
    const Metric m = metric(a, r, th);
    const double det = m.gtt * m.gpp - m.gtp * m.gtp;
    const double itt = m.gpp / det, itp = -m.gtp / det, ipp = m.gtt / det;
    const double ut_low = -abs_ut(r, th);
    return {ut_low * (itt - ell * itp), ut_low * (itp - ell * ipp)};
  }
};

inline bool make(double a, double r_in, double r_max, double gamma, Torus& t) {
  if (!(gamma > 1.0) || !(r_in > 1.0 + std::sqrt(1.0 - a * a) + 0.05) || !(r_in < r_max)) return false;
  t = Torus{};
  t.a = a; t.r_in = r_in; t.r_max = r_max; t.gamma = gamma;
  t.ell = ell_kepler(r_max, a);
  t.ut_in = t.abs_ut(r_in, M_PI / 2);
  const double um = t.abs_ut(r_max, M_PI / 2);
  if (t.ut_in <= 0.0 || um <= 0.0) return false;
  t.h_max = t.ut_in / um;
  // BOUND torus: |u_t| < 1 at the inner edge (energy per unit mass < 1), otherwise h > 1 also at infinity and the torus has no
  // outer surface (measured: a = 0.9, r_max = 12: r_in = 6 -> 1.0041 open, r_in = 7 -> 0.9814, outer edge at r ~ 46)
  return t.h_max > 1.0 && t.ut_in < 1.0;
}

}  // namespace bhs::torus
