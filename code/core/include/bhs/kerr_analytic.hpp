#pragma once
// O4 (refinement phase): a second, independent oracle for null geodesics in Kerr, built on the closed-form structure of Gralla & Lupsasca,
// "The Null Geodesics of the Kerr Exterior", PRD 101, 044032 (2020), arXiv:1910.12881. See notes/physics/38-analytic-kerr-geodesics.md.
//
// What is closed form here and what is not (stated plainly, this is the spike's finding, not marketing):
//   * the polar motion is closed form. With mu = cos(theta), (dmu/ds)^2 = a^2 (u+ - mu^2)(mu^2 - u-), u+ > 0 > u-, and
//         mu(s) = sqrt(u+) sn(w | m),  w = w0 + a sqrt(-u-) s,  m = u+/u- < 0,
//     the polar part of phi is an incomplete elliptic integral of the third kind (Carlson RF, RJ), and the equatorial crossings are the zeros of sn (w = 2 j K);
//   * the radial Mino integrals  int g(u) du / sqrt(P(u))  with  u = 1/r,  P(u) = r^-4 R(r) = 1 + c2 u^2 + c3 u^3 + c4 u^4  are evaluated by double-exponential
//     quadrature, with the turning point u_t removed by u = u_t - xi^2 (the integrand is then analytic). They are NOT yet Legendre/Carlson closed forms.
// It is therefore independent of any step-by-step integrator (no ODE, no step control, no pole guard), which is what makes it an oracle; it is not (yet) a fast kernel.
//
// Conventions are those of kerr::trace: the ray is followed BACKWARDS from the observer, s >= 0 is the backward Mino time, E = 1, lambda = L, eta = Q,
// (dr/ds)^2 = R(r), (dmu/ds)^2 = M(mu), dphi/ds = -(dphi/dtau). Only Kerr (q2 = 0).
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>

#include "kerr.hpp"

namespace bhs::kerr_an {

// ---------------------------------------------------------------------------------------------------------------------------------------------------
// Carlson symmetric elliptic integrals (Carlson 1995; the iterations of Numerical Recipes 3rd ed. 6.12 with double-precision tolerances)
// ---------------------------------------------------------------------------------------------------------------------------------------------------
inline double carlson_rf(double x, double y, double z) {
  constexpr double ERRTOL = 0.0025, C1 = 1.0 / 24.0, C2 = 0.1, C3 = 3.0 / 44.0, C4 = 1.0 / 14.0;
  double xt = x, yt = y, zt = z, ave, dx, dy, dz;
  do {
    const double sx = std::sqrt(xt), sy = std::sqrt(yt), sz = std::sqrt(zt);
    const double al = sx * (sy + sz) + sy * sz;
    xt = 0.25 * (xt + al); yt = 0.25 * (yt + al); zt = 0.25 * (zt + al);
    ave = (xt + yt + zt) / 3.0;
    dx = (ave - xt) / ave; dy = (ave - yt) / ave; dz = (ave - zt) / ave;
  } while (std::max({std::fabs(dx), std::fabs(dy), std::fabs(dz)}) > ERRTOL);
  const double e2 = dx * dy - dz * dz, e3 = dx * dy * dz;
  return (1.0 + (C1 * e2 - C2 - C3 * e3) * e2 + C4 * e3) / std::sqrt(ave);
}
inline double carlson_rc(double x, double y) {  // y > 0
  constexpr double ERRTOL = 0.0012, C1 = 0.3, C2 = 1.0 / 7.0, C3 = 0.375, C4 = 9.0 / 22.0;
  double xt = x, yt = y, ave, s;
  do {
    const double al = 2.0 * std::sqrt(xt) * std::sqrt(yt) + yt;
    xt = 0.25 * (xt + al); yt = 0.25 * (yt + al);
    ave = (xt + yt + yt) / 3.0;
    s = (yt - ave) / ave;
  } while (std::fabs(s) > ERRTOL);
  return (1.0 + s * s * (C1 + s * (C2 + s * (C3 + s * C4)))) / std::sqrt(ave);
}
inline double carlson_rj(double x, double y, double z, double p) {  // p > 0
  constexpr double ERRTOL = 0.0015, C1 = 3.0 / 14.0, C2 = 1.0 / 3.0, C3 = 3.0 / 22.0, C4 = 3.0 / 26.0, C5 = 0.75 * C3, C6 = 1.5 * C4, C7 = 0.5 * C2, C8 = C3 + C3;
  double xt = x, yt = y, zt = z, pt = p, sum = 0.0, fac = 1.0, ave, dx, dy, dz, dp;
  do {
    const double sx = std::sqrt(xt), sy = std::sqrt(yt), sz = std::sqrt(zt);
    const double al = sx * (sy + sz) + sy * sz;
    const double t1 = pt * (sx + sy + sz) + sx * sy * sz, alpha = t1 * t1;
    const double t2 = pt + al, beta = pt * t2 * t2;
    sum += fac * carlson_rc(alpha, beta);
    fac *= 0.25;
    xt = 0.25 * (xt + al); yt = 0.25 * (yt + al); zt = 0.25 * (zt + al); pt = 0.25 * (pt + al);
    ave = 0.2 * (xt + yt + zt + pt + pt);
    dx = (ave - xt) / ave; dy = (ave - yt) / ave; dz = (ave - zt) / ave; dp = (ave - pt) / ave;
  } while (std::max({std::fabs(dx), std::fabs(dy), std::fabs(dz), std::fabs(dp)}) > ERRTOL);
  const double ea = dx * (dy + dz) + dy * dz, eb = dx * dy * dz, ec = dp * dp, ed = ea - 3.0 * ec, ee = eb + 2.0 * dp * (ea - ec);
  return 3.0 * sum + fac * (1.0 + ed * (-C1 + C5 * ed - C6 * ee) + eb * (C7 + dp * (-C8 + dp * C4)) + dp * ea * (C2 - dp * C3) - C2 * dp * ec) / (ave * std::sqrt(ave));
}

// Legendre forms, parameter m = k^2 (here m < 0 is the case that matters), amplitude |phi| <= pi/2
inline double ellip_F(double phi, double m) {
  const double s = std::sin(phi), c = std::cos(phi);
  return s * carlson_rf(c * c, 1.0 - m * s * s, 1.0);
}
inline double ellip_K(double m) { return carlson_rf(0.0, 1.0 - m, 1.0); }
inline double ellip_Pi(double n, double phi, double m) {  // n < 1
  const double s = std::sin(phi), c = std::cos(phi), s2 = s * s;
  return s * carlson_rf(c * c, 1.0 - m * s2, 1.0) + (n / 3.0) * s * s2 * carlson_rj(c * c, 1.0 - m * s2, 1.0, 1.0 - n * s2);
}
inline double ellip_Pi_complete(double n, double m) { return carlson_rf(0.0, 1.0 - m, 1.0) + (n / 3.0) * carlson_rj(0.0, 1.0 - m, 1.0, 1.0 - n); }

// Jacobi sn, cn, dn for 0 <= m < 1: arithmetic-geometric mean (descending Landen), after Numerical Recipes sncndn
inline void jacobi01(double u, double m, double& sn, double& cn, double& dn) {
  const double emmc = 1.0 - m;
  if (emmc <= 0.0) { sn = std::tanh(u); cn = dn = 1.0 / std::cosh(u); return; }
  constexpr double CA = 1e-13;
  double a = 1.0, b = emmc, c;
  std::array<double, 40> am{}, gm{};
  int l = 0;
  double bt = emmc;
  dn = 1.0;
  for (int i = 0; i < 38; ++i) {
    l = i; am[i] = a; bt = std::sqrt(bt); gm[i] = bt; c = 0.5 * (a + bt);
    if (std::fabs(a - bt) <= CA * a) { a = c; break; }
    bt *= a; a = c;
  }
  (void)b;
  c = a;  // the AGM
  const double phi = c * u;
  sn = std::sin(phi); cn = std::cos(phi);
  if (sn != 0.0) {
    double aa = cn / sn; c *= aa;
    for (int ii = l; ii >= 0; --ii) {
      const double bb = am[ii];
      aa *= c; c *= dn;
      dn = (gm[ii] + aa) / (bb + aa);
      aa = c / bb;
    }
    aa = 1.0 / std::sqrt(c * c + 1.0);
    sn = sn >= 0.0 ? aa : -aa;
    cn = c * sn;
  }
}
// m < 1 (including m < 0): for m = -p, sn(u|-p) = sd(v|k2)/sqrt(1+p), cn(u|-p) = cd(v|k2), dn(u|-p) = nd(v|k2), v = sqrt(1+p) u, k2 = p/(1+p)
inline void jacobi(double u, double m, double& sn, double& cn, double& dn) {
  if (m >= 0.0) { jacobi01(u, m, sn, cn, dn); return; }
  const double p = -m, q = std::sqrt(1.0 + p);
  double s, c, d;
  jacobi01(q * u, p / (1.0 + p), s, c, d);
  sn = s / (d * q); cn = c / d; dn = 1.0 / d;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------
// Double-exponential (tanh-sinh) quadrature on [a, b]. f(x, da, db): da = x - a and db = b - x, computed without cancellation, so an integrand with an
// inverse-square-root endpoint can use them.
// ---------------------------------------------------------------------------------------------------------------------------------------------------
template <class F>
double tanh_sinh(F&& f, double a, double b, double rtol = 1e-14, int max_level = 9) {
  const double half = 0.5 * (b - a), mid = 0.5 * (a + b);
  const double pi2 = 1.5707963267948966;
  auto node = [&](double t) {
    const double y = pi2 * std::sinh(t);
    const double ch = std::cosh(y), w = pi2 * std::cosh(t) / (ch * ch);
    const double e = std::exp(-2.0 * std::fabs(y));
    const double near = half * 2.0 * e / (1.0 + e), far = half * 2.0 / (1.0 + e);  // half (1 - |tanh y|), half (1 + |tanh y|)
    const double x = y >= 0 ? b - near : a + near;
    const double da = y >= 0 ? far : near, db = y >= 0 ? near : far;
    if (!(near > 0.0) || !std::isfinite(w)) return 0.0;
    const double fx = f(x, da, db);
    return std::isfinite(fx) ? fx * w * half : 0.0;
  };
  (void)mid;
  double h = 1.0, sum = node(0.0), prev = 0.0, tmax = 6.0;  // y = pi/2 sinh(6) = 317: the nearest node is ~e^-634 from the endpoint, so the truncated tail of a 1/sqrt singularity is ~1e-137
  for (double t = h; t <= tmax; t += h) sum += node(t) + node(-t);
  double I = h * sum;
  for (int level = 1; level <= max_level; ++level) {
    h *= 0.5;
    double add = 0.0;
    for (double t = h; t <= tmax; t += 2.0 * h) add += node(t) + node(-t);
    sum += add;
    prev = I; I = h * sum;
    if (level >= 3 && std::fabs(I - prev) <= rtol * std::fabs(I)) break;
  }
  return I;
}


// ---------------------------------------------------------------------------------------------------------------------------------------------------
// Polar motion, closed form
// ---------------------------------------------------------------------------------------------------------------------------------------------------
struct Angular {
  double a = 0, lam = 0, eta = 0;
  double up = 0, um = 0, m = 0, kw = 0, w0 = 0, K = 0, Pc = 0;
  double n3 = 0, m3 = 0, ifac = 1.0, sgn0 = 1.0;  // characteristic and parameter of the Pi in phi_lam, and its prefactor
  bool vortical = false;  // eta < 0: mu^2 stays in [u-, u+] with u- > 0 (the ray circles the axis and never crosses the equator)
  bool ok = false;
  // mu0 = cos(theta_o); vmu0 = dmu/ds at the observer (backward Mino time).
  //   eta > 0:  u- < 0 < u+,  mu = sqrt(u+) sn(w | m),  m = u+/u- < 0, zeros of sn = equatorial crossings;
  //   eta < 0:  0 < u- < u+,  mu = sgn(mu0) sqrt(u+) dn(w | k2),  k2 = 1 - u-/u+,  dw/ds = a sqrt(u+)   (dn^2 in [1 - k2, 1], (dn')^2 = (1 - dn^2)(dn^2 - 1 + k2))
  bool init(double a_, double lam_, double eta_, double mu0, double vmu0) {
    a = std::max(a_, 1e-9);  // the polar motion is a^2 (u+ - mu^2)(mu^2 - u-): a = 0 is the (smooth) limit m -> 0-, evaluated at a = 1e-9 (error ~1e-18)
    lam = lam_; eta = eta_;
    if (eta == 0.0) return ok = false;
    const double a2 = a * a;
    const double dth = 0.5 * (1.0 - (eta + lam * lam) / a2), root = std::sqrt(std::max(dth * dth + eta / a2, 0.0));
    vortical = eta < 0.0;
    if (!vortical) {
      // u+ u- = -eta / a^2: take the root without cancellation first
      if (dth < 0.0) { um = dth - root; up = -(eta / a2) / um; } else { up = dth + root; um = -(eta / a2) / up; }
      m = up / um;  // < 0
      kw = a * std::sqrt(-um);
      K = ellip_K(m);
      n3 = up; m3 = m; ifac = 1.0;
      Pc = lam == 0.0 ? 0.0 : ellip_Pi_complete(n3, m3);  // lam = 0 (a ray leaving the axis): u+ = 1, Pi is infinite and is multiplied by 0
      // Phase from (sn, cn) without cancelling: sn = mu0 / sqrt(u+), and cn^2 = (u+ - mu0^2)/u+ with u+ - mu0^2 = M(mu0) / (a^2 (mu0^2 - u-)) taken from the
      // initial VELOCITY (exact in the data) when mu0 is close to the turning latitude, where u+ - mu0^2 loses all its digits (and asin(1 - 1e-16) is off by 1.4e-8).
      const double Bd = mu0 * mu0 - um;  // > 0 (u- < 0)
      double Ad = up - mu0 * mu0;
      if (Ad < 1e-3 * up) Ad = vmu0 * vmu0 / (a2 * Bd);
      double dir = vmu0;
      if (dir == 0.0) dir = -mu0;  // at a turning latitude: head for the equator
      const double sn0 = std::clamp(mu0 / std::sqrt(up), -1.0, 1.0), cn0 = std::sqrt(std::max(Ad, 0.0) / up);
      // dn = sqrt(1 - m sn^2) > 0, so the sign of dmu/ds is the sign of cn; amplitude in [-pi/2, pi/2] after folding cn >= 0
      const double F0 = ellip_F(std::atan2(sn0, cn0), m);
      w0 = dir >= 0.0 ? F0 : 2.0 * K - F0;  // sn(w0) = mu0 and the sign of cn(w0) is the sign of dmu/ds
    } else {
      if (!(dth > 0.0) || !(dth * dth + eta / a2 >= 0.0)) return ok = false;  // no real turning latitudes
      up = dth + root; um = (-eta / a2) / up;  // both > 0
      if (!(up < 1.0)) return ok = false;
      m = 1.0 - um / up;  // k^2 in (0, 1)
      kw = a * std::sqrt(up);
      K = ellip_K(m);
      sgn0 = mu0 >= 0.0 ? 1.0 : -1.0;
      // 1 - mu^2 = 1 - u+ dn^2 = (1 - u+) (1 - n' sn^2),  n' = - u+ k2 / (1 - u+) < 0
      n3 = -up * m / (1.0 - up); m3 = m; ifac = 1.0 / (1.0 - up);
      Pc = lam == 0.0 ? 0.0 : ellip_Pi_complete(n3, m3);  // lam = 0 (a ray leaving the axis): u+ = 1, Pi is infinite and is multiplied by 0
      // dn(w0) = |mu0| / sqrt(u+); sn^2 = (u+ - mu0^2) / (m u+) and cn^2 = (mu0^2 - u-) / (m u+), the smaller of the two numerators from the velocity (see above)
      double Ad = up - mu0 * mu0, Bd = mu0 * mu0 - um;
      if (Ad < Bd) { if (Ad < 1e-3 * up) Ad = vmu0 * vmu0 / (a2 * Bd); } else if (Bd < 1e-3 * up) Bd = vmu0 * vmu0 / (a2 * Ad);
      const double x = std::fabs(mu0) / std::sqrt(up);
      const double sn0 = std::sqrt(std::max(Ad, 0.0) / (m * up)), cn0 = std::sqrt(std::max(Bd, 0.0) / (m * up));
      const double F0 = ellip_F(std::atan2(sn0, cn0), m);  // w0 = -F0 when |mu| is increasing (dn rising for w in (-K, 0))
      double growing = vmu0 * sgn0;                          // d|mu|/ds
      if (growing == 0.0) growing = x >= 0.5 * (1.0 + std::sqrt(1.0 - m)) ? -1.0 : 1.0;
      w0 = growing > 0.0 ? -F0 : F0;
    }
    return ok = true;
  }
  bool crosses_equator() const { return !vortical; }
  double w(double s) const { return w0 + kw * s; }
  double mu(double s) const {
    double sn, cn, dn;
    jacobi(w(s), m, sn, cn, dn);
    return vortical ? sgn0 * std::sqrt(up) * dn : std::sqrt(up) * sn;
  }
  // int_0^w dw' / (1 - n3 sn^2 w'): odd in w, increases by 2 Pi_c per period 2K (of sn^2 and of dn^2)
  double I3(double ww) const {
    const double j = std::nearbyint(ww / (2.0 * K)), wr = ww - 2.0 * K * j;
    double sn, cn, dn;
    jacobi(wr, m3, sn, cn, dn);
    // amplitude from (sn, cn) with cn >= 0 on [-K, K]: asin(sn) near sn = 1 is off by sqrt(2e-16) = 1.4e-8, and the integrand of Pi there is 1 / (1 - n) ~ 3000 for a ray near the pole
    return ellip_Pi(n3, std::atan2(sn, std::fabs(cn)), m3) + 2.0 * Pc * j;
  }
  // int_0^s lambda / (1 - mu^2) ds'
  double phi_lam(double s) const { return lam == 0.0 ? 0.0 : lam / kw * ifac * (I3(w(s)) - I3(w0)); }
  // backward Mino time of the j-th zero of mu (w = 2 j K) after the observer (eta > 0 only)
  double s_cross(long j) const { return (2.0 * K * double(j) - w0) / kw; }
  long first_cross_index() const { return long(std::floor(w0 / (2.0 * K))) + 1; }
};

// ---------------------------------------------------------------------------------------------------------------------------------------------------
// Radial motion in u = 1/r:   ds = |du| / sqrt(P(u)),   P(u) = 1 + c2 u^2 + c3 u^3 + c4 u^4  (= R(r) / r^4)
// ---------------------------------------------------------------------------------------------------------------------------------------------------
struct Radial {
  double a = 0, lam = 0, eta = 0, c2 = 0, c3 = 0, c4 = 0;
  double u0 = 0, uh = 0, ut = 0;
  // The turning point and the cubic C(u) = P(u)/(u - ut) are kept in long double: near the critical curve two roots of P nearly coincide, C(ut) = P'(ut) is then small and
  // a double-precision root (error 1e-16) gives C an error ~ 1e-16 / (root gap).
  long double utl = 0, b3 = 0, b2 = 0, b1 = 0, b0 = 0;
  // turn = 0: no turning point (escape to infinity if moving out, capture by the horizon if moving in);
  // turn = 1: moving in, turns at the root ut > u0 (P > 0 for u < ut: u = ut - xi^2) and escapes;
  // turn = 2: moving out, turns at the root ut < u0 (P > 0 for u > ut: u = ut + xi^2, a photon bent back by the hole) and falls in.
  int turn = 0;
  double sigma = -1.0;
  bool outward = false, turning = false, ok = false;
  double P(double u) const { return ((c4 * u + c3) * u + c2) * u * u + 1.0; }
  long double Pl(long double u) const { return ((c4 * u + c3) * u + c2) * u * u + 1.0L; }
  long double Cl(long double u) const { return ((b3 * u + b2) * u + b1) * u + b0; }  // P(u) = (u - ut) C(u)
  // all positive real roots of P in (0, umax], ascending (P(0) = 1 > 0). P' = u (4 c4 u^2 + 3 c3 u + 2 c2): the monotone pieces are known in closed form
  int roots(double umax, std::array<long double, 4>& out) const {
    std::array<double, 4> pts{0.0, 0.0, 0.0, umax};
    int n = 1;
    double q[2]; int nq = 0;
    if (c4 != 0.0) {
      const double disc = 9.0 * c3 * c3 - 32.0 * c4 * c2;
      if (disc >= 0.0) {
        const double sq = std::sqrt(disc), qq = -0.5 * (3.0 * c3 + (c3 >= 0.0 ? sq : -sq));
        q[nq++] = qq / (4.0 * c4);
        if (qq != 0.0) q[nq++] = 2.0 * c2 / qq;
      }
    } else if (c3 != 0.0) q[nq++] = -2.0 * c2 / (3.0 * c3);
    std::sort(q, q + nq);
    for (int i = 0; i < nq; ++i) if (q[i] > 0.0 && q[i] < umax) pts[n++] = q[i];
    pts[n++] = umax;
    int nr = 0;
    for (int i = 0; i + 1 < n; ++i) {
      long double lo = pts[i], hi = pts[i + 1];
      const long double plo = Pl(lo), phi = Pl(hi);
      if ((plo > 0.0L) == (phi > 0.0L) && phi != 0.0L) continue;  // no sign change on this monotone piece
      const bool rising = plo < phi;
      for (int it = 0; it < 300 && hi - lo > 1e-19L * hi; ++it) { const long double mid = 0.5L * (lo + hi); ((Pl(mid) > 0.0L) == rising ? hi : lo) = mid; }
      if (nr < 4) out[nr++] = 0.5L * (lo + hi);
    }
    return nr;
  }
  // r0, the radial velocity dr/ds at the observer (sign only matters), Kerr
  bool init(double a_, double lam_, double eta_, double r0, double vr0) {
    a = a_; lam = lam_; eta = eta_;
    const double Kc = eta + (lam - a) * (lam - a);
    c2 = a * a - lam * lam - eta; c3 = 2.0 * Kc; c4 = -a * a * eta;
    u0 = 1.0 / r0; outward = vr0 > 0.0;
    uh = 1.0 / kerr::horizon(a);
    std::array<long double, 4> rt{};
    const int nr = roots(uh, rt);
    bool has_lo = false, has_hi = false;
    long double lo = 0, hi = 0;
    for (int i = 0; i < nr; ++i) {
      if (rt[i] < u0) { lo = rt[i]; has_lo = true; }                // the largest root below the observer (outer turning point)
      else if (!has_hi) { hi = rt[i]; has_hi = true; }              // the smallest root above it (inner turning point)
    }
    turn = 0; turning = false;
    if (has_lo && has_hi) return ok = false;  // the observer sits between two turning points: a bound radial motion (not an escaping or captured ray)
    if (outward && has_lo) { turn = 2; sigma = 1.0; utl = lo; }
    else if (!outward && has_hi) { turn = 1; sigma = -1.0; utl = hi; }
    if (turn) {
      turning = true; ut = double(utl);
      b3 = c4; b2 = c3 + utl * b3; b1 = c2 + utl * b2; b0 = utl * b1;
    }
    return ok = true;
  }
  double u_of_xi(double xi) const { return double(utl + (long double)sigma * (long double)xi * xi); }
  // g(u) du / sqrt(P(u)) on [ua, ub] with no root of P in it
  template <class G> double A(G&& g, double ua, double ub) const {
    return tanh_sinh([&](double x, double, double) { return g(x) / std::sqrt(P(x)); }, ua, ub);
  }
  // 2 g(u) / sqrt(sigma C(u)) d(xi) with u = ut + sigma xi^2, xi in [xa, xb]: the turning point removed (the integrand is analytic)
  template <class G> double T(G&& g, double xa, double xb) const {
    return tanh_sinh([&](double xi, double, double) {
      const long double ul = utl + (long double)sigma * (long double)xi * xi;
      return 2.0 * g(double(ul)) / double(std::sqrt((long double)sigma * Cl(ul)));
    }, xa, xb);
  }
  // azimuthal radial part: dphi/ds contains  g_rad = -a + a (r^2 + a^2 - a lam) / Delta = a (2 r - a lam) / Delta = a u (2 - a lam u) / (1 - 2u + a^2 u^2)
  // 1/sqrt(P(u)) and the integrand of T in xi: the exact derivatives of the integrals A and T with respect to their limits (used by the Newton root search)
  double dA(double u) const { return 1.0 / std::sqrt(P(u)); }
  double dT(double xi) const { const long double ul = utl + (long double)sigma * (long double)xi * xi; return 2.0 / double(std::sqrt((long double)sigma * Cl(ul))); }
  double grad(double u) const { return a * u * (2.0 - a * lam * u) / (1.0 - 2.0 * u + a * a * u * u); }
};

struct Options {
  double r_far = 0.0;   // 0: the asymptotic direction (u = 0); > 0: the state at r = r_far, to compare with an integrator that stops there
  double l_min = 1e-4;  // as kerr::Options::l_min (the same |L| floor, so both solve the same ray)
  double eta_min = 1e-12;
};
struct Result {
  kerr::Trace t;
  bool supported = true;
  const char* why = "";
  double s_end = 0;   // backward Mino time to the end of the ray
  double s_hit = 0;   // ... to the disk hit, if any
  bool turned = false;
  int turn = 0;       // 0 none, 1 = moving in, turned at an inner root and escaped, 2 = moving out, bent back by the hole and fell in (see Radial)
  int crossings = 0;  // equatorial crossings found before the end of the ray
  int pr_sign = 0;    // at the disk hit: sign of dr along the path, in the direction the initial velocity points (+1 outward). For photons started at a source that is the physical
                      // direction of motion (the lamppost); for rays traced backwards from a camera it is the backward one
};

namespace detail {
template <class F>
double bisect(F&& f, double lo, double hi, double tol_rel = 1e-15) {  // f(lo) and f(hi) of opposite sign
  double flo = f(lo);
  for (int it = 0; it < 100 && std::fabs(hi - lo) > tol_rel * std::max(std::fabs(lo), std::fabs(hi)); ++it) {
    const double mid = 0.5 * (lo + hi), fm = f(mid);
    if ((fm > 0.0) == (flo > 0.0)) { lo = mid; flo = fm; } else hi = mid;
  }
  return 0.5 * (lo + hi);
}
// Root of a monotone f on [lo, hi] (opposite signs at the ends) by Newton steps with the exact derivative df, safeguarded by the bracket (bisection when a step leaves it or does not
// shrink it fast enough). Each f is a full quadrature, so the 50 evaluations of bisection become ~5.
template <class F, class DF>
double newton_bracket(F&& f, DF&& df, double lo, double hi, double tol_rel = 1e-15) {
  double flo = f(lo), x = 0.5 * (lo + hi);
  for (int it = 0; it < 100; ++it) {
    const double fx = f(x);
    if (fx == 0.0) return x;
    if ((fx > 0.0) == (flo > 0.0)) { lo = x; flo = fx; } else hi = x;
    const double d = df(x);
    double xn = x - fx / d;
    if (!(xn > lo && xn < hi) || !std::isfinite(xn)) xn = 0.5 * (lo + hi);
    if (std::fabs(xn - x) <= tol_rel * std::max(std::fabs(xn), std::fabs(x)) || std::fabs(hi - lo) <= tol_rel * std::max(std::fabs(lo), std::fabs(hi))) return xn;
    x = xn;
  }
  return x;
}
}  // namespace detail

// Same arguments and conventions as kerr::trace (Kerr only). Returns the fate, the sky direction of an escaping ray and the first equatorial crossing inside the disk.
inline Result trace_consts(double a, double r0, double mu0, double lam, double eta, double vr, double vmu, kerr::Disk disk, Options opt);

inline Result trace(double a, double r0, double theta0, double nr, double nth, double nph, kerr::Disk disk = {}, Options opt = {}) {
  const auto in = kerr::init_from_zamo(a, r0, theta0, nr, nth, nph);
  double lam = in.L;
  if (std::fabs(lam) < opt.l_min) lam = std::copysign(opt.l_min, lam == 0.0 ? 1.0 : lam);
  return trace_consts(a, r0, std::cos(theta0), lam, in.Q, -kerr::delta(r0, a) * in.pr, std::sin(theta0) * in.ptheta, disk, opt);  // vr, vmu = dr/ds, dmu/ds (backward Mino time)
}

// The same from the constants of motion and the initial state (mu0 = cos theta, vr = dr/ds, vmu = dmu/ds, backward Mino time). Used by trace() and by rays that do not
// start from a ZAMO observer (a source on the axis, lamppost.hpp). lam = 0 is allowed (phi_lam = 0), no |L| floor is applied here.
inline Result trace_consts(double a, double r0, double mu0, double lam, double eta, double vr, double vmu, kerr::Disk disk, Options opt) {
  Result res;
  res.t.L = lam; res.t.Q = eta;
  Angular ang; Radial rad;
  if (!(std::fabs(eta) > opt.eta_min * (1.0 + lam * lam))) { res.supported = false; res.why = "eta = 0 (the separatrix between ordinary and vortical rays)"; return res; }
  if (!ang.init(a, lam, eta, mu0, vmu)) { res.supported = false; res.why = "polar motion"; return res; }
  if (!rad.init(a, lam, eta, r0, vr)) { res.supported = false; res.why = "observer in a bound radial region"; return res; }
  const double u_end = opt.r_far > 0.0 ? 1.0 / opt.r_far : 0.0;
  auto g = [&](double u) { return rad.grad(u); };
  auto one = [](double) { return 1.0; };

  // length of the whole path (backward Mino time) and the radial azimuth integral over it
  double s1 = 0.0, xi0 = 0.0, xiE = 0.0, s_end = 0.0, phirad_end = 0.0, phirad_leg1 = 0.0;
  bool captured = false;
  if (rad.turn == 1) {                     // in, turn, out to u_end
    xi0 = std::sqrt(rad.ut - rad.u0); xiE = std::sqrt(std::max(rad.ut - u_end, 0.0));
    s1 = rad.T(one, 0.0, xi0);
    phirad_leg1 = rad.T(g, 0.0, xi0);
    s_end = s1 + rad.T(one, 0.0, xiE);
    phirad_end = phirad_leg1 + rad.T(g, 0.0, xiE);
    res.turned = true; res.turn = 1;
  } else if (rad.turn == 2) {              // out, turn (bent back), in to the horizon
    xi0 = std::sqrt(rad.u0 - rad.ut); xiE = std::sqrt(rad.uh - rad.ut);
    s1 = rad.T(one, 0.0, xi0);
    phirad_leg1 = rad.T(g, 0.0, xi0);
    s_end = s1 + rad.T(one, 0.0, xiE);
    captured = true;
    res.turned = true; res.turn = 2;
  } else if (rad.outward) {
    s_end = rad.A(one, u_end, rad.u0);
    phirad_end = rad.A(g, u_end, rad.u0);
  } else {
    captured = true;
    s_end = rad.A(one, rad.u0, rad.uh);
  }
  res.s_end = s_end;

  // equatorial crossings of the polar motion, in order of s; the first one inside the disk ends the ray (opaque disk, as in kerr::trace)
  int skipped = 0;
  if (disk.r_out > 0.0 && ang.crosses_equator()) {
    for (long j = ang.first_cross_index(); j < ang.first_cross_index() + 4000; ++j) {
      const double sj = ang.s_cross(j);
      if (sj <= 1e-13) continue;          // the observer itself on the equator
      if (sj >= s_end) break;
      ++res.crossings;
      double uj, phirad;
      if (rad.turning) {
        if (sj <= s1) {                    // leg 1: u0 -> ut, xi from xi0 down to 0
          const double xi = detail::newton_bracket([&](double x) { return rad.T(one, x, xi0) - sj; }, [&](double x) { return -rad.dT(x); }, 0.0, xi0);
          uj = rad.u_of_xi(xi); phirad = rad.T(g, xi, xi0);
        } else {                           // leg 2: ut -> u_end (escape) or the horizon, xi from 0 up
          const double xi = detail::newton_bracket([&](double x) { return rad.T(one, 0.0, x) - (sj - s1); }, [&](double x) { return rad.dT(x); }, 0.0, xiE);
          uj = rad.u_of_xi(xi); phirad = phirad_leg1 + rad.T(g, 0.0, xi);
        }
      } else if (rad.outward) {
        uj = detail::newton_bracket([&](double u) { return rad.A(one, u, rad.u0) - sj; }, [&](double u) { return -rad.dA(u); }, u_end, rad.u0);
        phirad = rad.A(g, uj, rad.u0);
      } else {
        uj = detail::newton_bracket([&](double u) { return rad.A(one, rad.u0, u) - sj; }, [&](double u) { return rad.dA(u); }, rad.u0, rad.uh);
        phirad = rad.A(g, rad.u0, uj);
      }
      const double rj = 1.0 / uj;
      if (rj > disk.r_in && rj < disk.r_out && skipped++ >= disk.skip) {
        res.t.disk_hit = true; res.t.disk_r = rj; res.s_hit = sj;
        res.pr_sign = rad.turning ? ((rad.turn == 1) == (sj <= s1) ? -1 : +1) : (rad.outward ? +1 : -1);  // turn 1: in then out; turn 2: out then in
        res.t.disk_phi = -(ang.phi_lam(sj) + phirad);
        res.t.fate = kerr::Fate::Captured;  // opaque disk
        return res;
      }
    }
  }
  if (captured) { res.t.fate = kerr::Fate::Captured; return res; }
  res.t.fate = kerr::Fate::Escaped;
  res.t.theta_end = std::acos(std::clamp(ang.mu(s_end), -1.0, 1.0));
  res.t.phi_end = -(ang.phi_lam(s_end) + phirad_end);
  return res;
}

}  // namespace bhs::kerr_an
