#pragma once
// Null geodesics in Kerr spacetime (M = 1, spin a in [0,1), rotation axis = +z) and, with `q2 = Q^2 > 0`, Kerr-Newman (v2.1a/b):
// only Delta(r) = r^2 - 2r + a^2 + q2 and the ZAMO angular velocity omega = a (2r - q2)/A change; the Carter equations keep their form. q2 = 0 (default)
// recovers Kerr exactly (adding 0.0 changes nothing). With a^2 + q2 > 1 there is no horizon (naked singularity): see Options::r_min.
// Boyer-Lindquist coordinates (t, r, theta, phi); photons with E = 1; constants L (=L_z) and Q (Carter).
//
// In "Mino time" tau (d tau = d lambda / Sigma) the equations separate:
//   (dr/dtau)^2   = R(r)  = [(r^2+a^2) - aL]^2 - Delta [(L-a)^2 + Q]
//   (dmu/dtau)^2  = M(mu) = (1-mu^2)(Q + a^2 mu^2) - L^2 mu^2,      mu = cos(theta)
//   dphi/dtau     = L/(1-mu^2) - a + a (r^2+a^2 - aL)/Delta
// To integrate without the sign change at the turning points the smooth 2nd-order form is used:
//   d^2r/dtau^2 = R'(r)/2 ,   d^2mu/dtau^2 = M'(mu)/2     (mu instead of theta: no singularity on the axis)
// After each step the velocities are reprojected onto (dr/dtau)^2 = R, (dmu/dtau)^2 = M.
// The ray is traced BACKWARDS (from the observer to the sky): integrate in tau' = -tau.
#include <algorithm>
#include <array>
#include <cmath>

namespace bhs::kerr {

// outer horizon radius; 0 if it does not exist (a^2 + q2 > 1: naked singularity)
inline double horizon(double a, double q2 = 0.0) { const double d = 1.0 - a * a - q2; return d < 0.0 ? 0.0 : 1.0 + std::sqrt(d); }
// Reissner-Nordstrom (a = 0), q2 = Q^2/M^2: photon sphere (only exists for q2 <= 9/8), shadow and ISCO (root of the cubic r^3 - 6r^2 + 9 q2 r - 4 q2^2 = 0)
inline double rn_photon_sphere(double q2) { return 0.5 * (3.0 + std::sqrt(9.0 - 8.0 * q2)); }
inline double rn_shadow_b(double q2) { const double r = rn_photon_sphere(q2); return r * r / std::sqrt(r * r - 2.0 * r + q2); }
inline double rn_isco(double q2) {
  double r = 6.0;  // Newton from r = 6 (q2 = 0); the relevant root is the largest
  for (int i = 0; i < 100; ++i) {
    const double f = ((r - 6.0) * r + 9.0 * q2) * r - 4.0 * q2 * q2, df = (3.0 * r - 12.0) * r + 9.0 * q2;
    const double dr = f / df; r -= dr;
    if (std::fabs(dr) < 1e-15) break;
  }
  return r;
}

// ISCO (Bardeen, Press, Teukolsky 1972)
inline double isco(double a, bool prograde = true) {
  const double z1 = 1.0 + std::cbrt(1.0 - a * a) * (std::cbrt(1.0 + a) + std::cbrt(1.0 - a));
  const double z2 = std::sqrt(3.0 * a * a + z1 * z1);
  const double s = std::sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2));
  return prograde ? 3.0 + z2 - s : 3.0 + z2 + s;
}

// radius of the equatorial circular photon orbit
inline double photon_orbit(double a, bool prograde = true) {
  return 2.0 * (1.0 + std::cos((2.0 / 3.0) * std::acos(prograde ? -a : a)));
}
// L/E of a spherical photon orbit of radius r (Bardeen); on the equator (Q = 0) it is L = xi(r_ph)
inline double xi_photon(double r, double a) { return (r * r * (3.0 - r) - a * a * (r + 1.0)) / (a * (r - 1.0)); }

struct Consts {
  double a, L, Q;
  double q2 = 0.0;  // charge^2 (Kerr-Newman); Q above is the Carter constant
};

inline double delta(double r, double a, double q2 = 0.0) { return r * r - 2.0 * r + a * a + q2; }
inline double R(double r, const Consts& c) {
  const double A = r * r + c.a * c.a - c.a * c.L;
  return A * A - delta(r, c.a, c.q2) * ((c.L - c.a) * (c.L - c.a) + c.Q);
}
inline double dR(double r, const Consts& c) {
  return 4.0 * r * (r * r + c.a * c.a - c.a * c.L) - (2.0 * r - 2.0) * ((c.L - c.a) * (c.L - c.a) + c.Q);
}
inline double Mmu(double mu, const Consts& c) {
  return (1.0 - mu * mu) * (c.Q + c.a * c.a * mu * mu) - c.L * c.L * mu * mu;
}
inline double dMmu(double mu, const Consts& c) {
  return 2.0 * (c.a * c.a - c.Q - c.L * c.L) * mu - 4.0 * c.a * c.a * mu * mu * mu;
}
// dt/dtau (Mino time, E = 1): (r^2 + a^2)(r^2 + a^2 - aL)/Delta + a (L - a sin^2 theta)
inline double dt_dtau(double r, double mu, const Consts& c) {
  const double A = r * r + c.a * c.a, P = A - c.a * c.L;
  return A * P / delta(r, c.a, c.q2) + c.a * (c.L - c.a * (1.0 - mu * mu));
}
inline double dphi_dtau(double r, double mu, const Consts& c) {
  const double om = std::max(1e-14, 1.0 - mu * mu);
  return c.L / om - c.a + c.a * (r * r + c.a * c.a - c.a * c.L) / delta(r, c.a, c.q2);
}

// ZAMO observer (zero angular momentum) at (r, theta); n = propagation direction of the photon in the
// local frame (e_r, e_theta, e_phi), unit. Returns the constants and covariant momenta (E = 1).
struct LocalInit {
  double L, Q, pr, ptheta, E_loc;
};
inline LocalInit init_from_zamo(double a, double r, double theta, double nr, double nth, double nph, double q2 = 0.0) {
  const double s = std::sin(theta), mu = std::cos(theta);
  const double Sigma = r * r + a * a * mu * mu, Delta = delta(r, a, q2);
  const double A = std::pow(r * r + a * a, 2) - a * a * Delta * s * s;
  const double alpha = std::sqrt(Sigma * Delta / A), omega = a * (2.0 * r - q2) / A;
  const double varpi = std::sqrt(A / Sigma) * s;
  const double E_loc = 1.0 / (alpha + omega * varpi * nph);  // E = alpha E_loc + omega L = 1
  LocalInit o;
  o.E_loc = E_loc;
  o.L = varpi * E_loc * nph;
  o.pr = E_loc * nr * std::sqrt(Sigma / Delta);
  o.ptheta = E_loc * nth * std::sqrt(Sigma);
  o.Q = o.ptheta * o.ptheta + mu * mu * (o.L * o.L / std::max(1e-14, 1.0 - mu * mu) - a * a);
  return o;
}

// ---- Dormand-Prince 5(4), adaptive step ----
template <size_t N, class F>
double dp45_step(F f, const std::array<double, N>& y, double h, std::array<double, N>& y_new,
                 std::array<double, N>& k1, std::array<double, N>& k7, double atol, double rtol) {
  using V = std::array<double, N>;
  auto comb = [&](std::initializer_list<std::pair<double, const V*>> terms) {
    V r = y;
    for (auto& [c, k] : terms)
      for (size_t i = 0; i < N; ++i) r[i] += h * c * (*k)[i];
    return r;
  };
  k1 = f(y);
  const V k2 = f(comb({{1.0 / 5, &k1}}));
  const V k3 = f(comb({{3.0 / 40, &k1}, {9.0 / 40, &k2}}));
  const V k4 = f(comb({{44.0 / 45, &k1}, {-56.0 / 15, &k2}, {32.0 / 9, &k3}}));
  const V k5 = f(comb({{19372.0 / 6561, &k1}, {-25360.0 / 2187, &k2}, {64448.0 / 6561, &k3}, {-212.0 / 729, &k4}}));
  const V k6 = f(comb({{9017.0 / 3168, &k1}, {-355.0 / 33, &k2}, {46732.0 / 5247, &k3}, {49.0 / 176, &k4}, {-5103.0 / 18656, &k5}}));
  y_new = comb({{35.0 / 384, &k1}, {500.0 / 1113, &k3}, {125.0 / 192, &k4}, {-2187.0 / 6784, &k5}, {11.0 / 84, &k6}});
  k7 = f(y_new);
  double err = 0;
  for (size_t i = 0; i < N; ++i) {
    const double e = h * (71.0 / 57600 * k1[i] - 71.0 / 16695 * k3[i] + 71.0 / 1920 * k4[i] -
                          17253.0 / 339200 * k5[i] + 22.0 / 525 * k6[i] - 1.0 / 40 * k7[i]);
    err = std::max(err, std::abs(e) / (atol + rtol * std::max(std::abs(y[i]), std::abs(y_new[i]))));
  }
  return err;
}
inline double step_factor(double err) {
  if (err <= 0) return 5.0;
  return std::min(5.0, std::max(0.2, 0.9 * std::pow(err, -0.2)));
}

// ---- tracing ----
enum class Fate { Escaped, Captured, Orbiting };

struct Disk {
  double r_in = 0, r_out = 0;  // r_out = 0: no disk
  int skip = 0;                // ignores the first `skip` plane crossings inside the disk (semi-transparent gas: continue behind it)
};
struct Options {
  double atol = 1e-10, rtol = 1e-9;
  double r_far = 1e6;       // radius beyond which the ray has "escaped" (direction ~ asymptotic)
  int max_steps = 200000;
  // minimum |L|. With L ~ 0 the ray lies in a plane that contains the rotation axis and crosses the pole, where phi
  // (Boyer-Lindquist) is singular: with exact L = 0 phi does not jump by pi (final direction wrong by pi: vertical
  // line at the image centre) and with |L| ~ 1e-8 the jump is stiff (~1e5 steps). |L| >= l_min is enforced.
  // Measured (a = 0, y = 0.2, D = 40): 1e-5 is still stiff (1100 steps at tol 1e-8, 1.7e5 at 1e-10, wrong
  // result); 1e-4 and 1e-3 give the correct limit (2.494982) with ~450 steps. It stays at 1e-4: it shifts the local ray
  // direction by L/(varpi E) ~ 2.5e-6 rad (0.002 px) at theta = 80 degrees and 3e-4 rad (0.45 px, the worst case) with the camera at the pole.
  double l_min = 1e-4;
  // Pole crossing: near the axis sigma = 1 - mu^2 has a narrow minimum (width ~ sqrt(sigma/|sigma''|)) and
  // the phi integrand, L/sigma, a spike. The error estimated at the RK stages does not see a spike that falls between them:
  // with loose tolerances the step skips it and the jump of pi in phi is lost (measured, at tol 2e-5: final phi wrong
  // by pi, even in double). The step is limited to pole_kappa * sqrt(sigma/|sigma''|) when sigma < pole_sigma
  // (sigma'' = -(2M + mu M')): it costs O(log) steps per crossing. 0 turns it off.
  double pole_kappa = 0.5, pole_sigma = 0.2;
  double q2 = 0.0;     // charge^2 (Kerr-Newman)
  // without a horizon (naked singularity) the ray that reaches r < r_min is stopped and marked `singular` (fate = Captured); with a horizon r_min has no effect
  double r_min = 0.05;
};
struct Trace {
  Fate fate = Fate::Orbiting;
  double theta_end = 0, phi_end = 0;  // direction the light came from (only if Escaped)
  bool disk_hit = false;
  double disk_r = 0, disk_phi = 0;    // first crossing of the equatorial plane inside the disk
  double L = 0;                       // L_z/E of the physical photon
  double Q = 0;                       // Carter constant of the photon (E = 1); on the equator p_theta^2 = Q
  double t_disk = 0;                  // coordinate flight time of the light from the disk to the observer (>0): t_emission = t_obs - t_disk
  bool singular = false;              // stopped at r < Options::r_min without a horizon (naked singularity): not a physical prediction
  double min_sigma = 1.0;             // smallest sin^2(theta) = 1 - mu^2 along the ray (diagnostic: a ray that passes close to the pole, |L| ~ 0, is ill-conditioned in mu)
  double max_inv_err = 0;             // maximum violation of (dr/dtau)^2 = R and (dmu/dtau)^2 = M
  int steps = 0;
  // mu = cos(theta) near the pole has a spacing of 1e-16 while sigma = 1 - mu^2 can be 1e-10: the polar motion of a meridional ray (|L| ~ l_min) is then known to ~1e-7 relative, whatever the
  // tolerance. Measured (a = 0.9, theta_o = 60, L at the floor, min_sigma = 3e-10): error in the disk radius 6e-10 at atol 1e-10, 5e-5 at 1e-12, and no hit at all at 1e-14 (14 M steps).
  // Such a request cannot be honoured by this integrator; kerr_analytic.hpp (note 38) is the reference for these rays.
  bool precision_limited(double atol) const { return min_sigma < 1e-6 && atol < 1e-11; }
};

// Ray traced from the ZAMO observer at (r0, theta0, phi = 0). (nr, nth, nph): propagation direction
// of the PHOTON (= opposite to the viewing one) in the local frame, unit.
inline Trace trace(double a, double r0, double theta0, double nr, double nth, double nph, Disk disk = {},
                   Options opt = {}) {
  auto in = init_from_zamo(a, r0, theta0, nr, nth, nph, opt.q2);
  if (std::abs(in.L) < opt.l_min) in.L = std::copysign(opt.l_min, in.L == 0.0 ? 1.0 : in.L);
  const Consts c{a, in.L, in.Q, opt.q2};
  Trace out;
  int skipped = 0;
  out.L = in.L;
  out.Q = in.Q;
  const double r_plus = horizon(a, opt.q2);
  const bool naked = r_plus == 0.0;
  const double mu0 = std::cos(theta0), s0 = std::sin(theta0);
  // state in tau' = -tau: reversed velocities, phi' = -dphi/dtau
  using S = std::array<double, 6>;  // r, dr/dtau', mu, dmu/dtau', phi, t (t - t_obs: -flight time since the observer)
  S y{r0, -delta(r0, a, opt.q2) * in.pr, mu0, s0 * in.ptheta, 0.0, 0.0};
  auto f = [&](const S& s) {
    return S{s[1], 0.5 * dR(s[0], c), s[3], 0.5 * dMmu(s[2], c), -dphi_dtau(s[0], s[2], c), -dt_dtau(s[0], s[2], c)};
  };
  double h = 0.02 / r0;
  for (int step = 0; step < opt.max_steps; ++step) {
    if (opt.pole_kappa > 0) {
      const double sig = 1.0 - y[2] * y[2];
      if (sig < opt.pole_sigma) {
        const double curv = std::abs(2.0 * Mmu(y[2], c) + y[2] * dMmu(y[2], c));
        h = std::min(h, opt.pole_kappa * std::sqrt(std::max(sig, 1e-300) / std::max(curv, 1e-12)));
      }
    }
    S yn, k1, k7;
    const double err = dp45_step(f, y, h, yn, k1, k7, opt.atol, opt.rtol);
    if (err > 1.0) { h *= step_factor(err); continue; }
    ++out.steps;
    out.min_sigma = std::min(out.min_sigma, 1.0 - yn[2] * yn[2]);
    // invariants
    const double ir = std::abs(yn[1] * yn[1] - R(yn[0], c)) / (1.0 + yn[1] * yn[1] + std::abs(R(yn[0], c)));
    const double im = std::abs(yn[3] * yn[3] - Mmu(yn[2], c)) / (1.0 + yn[3] * yn[3] + std::abs(Mmu(yn[2], c)));
    out.max_inv_err = std::max({out.max_inv_err, ir, im});
    // Projects back onto the surface of the first integrals: the accumulated ABSOLUTE error at large r
    // (where R ~ r^4 is huge) would ruin the relative precision near the hole (small R).
    // Since R(r) and M(mu) are exact for the current r and mu, only the velocity is corrected (sign kept).
    const double yn1 = yn[1], yn3 = yn[3];
    yn[1] = std::copysign(std::sqrt(std::max(R(yn[0], c), 0.0)), yn1);
    yn[3] = std::copysign(std::sqrt(std::max(Mmu(yn[2], c), 0.0)), yn3);

    // crossing of the equatorial plane (mu changes sign) inside the disk
    if (disk.r_out > 0 && !out.disk_hit && (y[2] > 0) != (yn[2] > 0)) {
      // cubic Hermite of mu (value and derivative at both ends) -> root by bisection in s in [0,1]
      auto herm = [&](int i, int di, double s) {
        const double h00 = 2 * s * s * s - 3 * s * s + 1, h10 = s * s * s - 2 * s * s + s;
        const double h01 = -2 * s * s * s + 3 * s * s, h11 = s * s * s - s * s;
        return h00 * y[i] + h10 * h * (di < 0 ? 0.0 : k1[di]) + h01 * yn[i] + h11 * h * (di < 0 ? 0.0 : k7[di]);
      };
      double lo = 0, hi = 1;
      const bool pos0 = y[2] > 0;
      for (int it = 0; it < 60; ++it) {
        const double m = 0.5 * (lo + hi);
        ((herm(2, 2, m) > 0) == pos0 ? lo : hi) = m;
      }
      const double s = 0.5 * (lo + hi);
      const double rc = herm(0, 0, s), pc = herm(4, 4, s);
      if (rc > disk.r_in && rc < disk.r_out && skipped++ >= disk.skip) {
        out.disk_hit = true;
        out.disk_r = rc;
        out.disk_phi = pc;
        out.t_disk = -herm(5, 5, s);
        out.fate = Fate::Captured;  // opaque disk: the ray ends here
        return out;
      }
    }
    y = yn;
    if (naked ? y[0] < opt.r_min : y[0] < r_plus + 1e-3) { out.fate = Fate::Captured; out.singular = naked; return out; }
    if (y[0] > opt.r_far && y[1] > 0) {
      out.fate = Fate::Escaped;
      out.theta_end = std::acos(std::clamp(y[2], -1.0, 1.0));
      out.phi_end = y[4];
      return out;
    }
    h *= step_factor(err);
  }
  out.fate = Fate::Orbiting;
  return out;
}

// WARNING: Kerr only (q2 = 0): the circular orbit and the ZAMO assume Kerr.
// g = nu_obs / nu_emit of an emitter on a prograde equatorial circular orbit (radius r) for a
// ZAMO observer at (r_obs, theta_obs); L = L_z/E of the photon.
inline double redshift(double a, double r, double L, double r_obs, double theta_obs) {
  const double s = std::sin(theta_obs), mu = std::cos(theta_obs);
  const double Sigma = r_obs * r_obs + a * a * mu * mu, Delta = delta(r_obs, a);
  const double A = std::pow(r_obs * r_obs + a * a, 2) - a * a * Delta * s * s;
  const double alpha = std::sqrt(Sigma * Delta / A), omega = 2.0 * a * r_obs / A;
  const double E_obs = (1.0 - omega * L) / alpha;
  const double ut = (std::pow(r, 1.5) + a) / (std::pow(r, 0.75) * std::sqrt(std::pow(r, 1.5) - 3.0 * std::sqrt(r) + 2.0 * a));
  const double Omega = 1.0 / (std::pow(r, 1.5) + a);
  return E_obs / (ut * (1.0 - Omega * L));
}

}  // namespace bhs::kerr
