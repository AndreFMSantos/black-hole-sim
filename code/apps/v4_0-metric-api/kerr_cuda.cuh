#pragma once
// Kerr physics (v0.6a, core/include/bhs/kerr.hpp) ported to the device, templated on the scalar type T
// (double = reference; float = fast mode). Same equations, same adaptive Dormand-Prince 5(4) in
// Mino time, same reprojection of the velocities onto (dr/dtau)^2 = R and (dmu/dtau)^2 = M after every step.
// The constants (L, Q) and the initial state always come from init_from_zamo in double (once per ray).
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

#include "dop853_coeffs.cuh"

namespace bk {

struct Trace {
  int fate;            // 0 escaped, 1 captured, 2 orbiting (max_steps)
  int disk_hit;
  int steps;           // accepted steps
  double mu_end, phi_end;  // direction the light came from (only if fate == 0): mu = cos(theta)
  double disk_r, disk_phi;
  double L;
  double Q;            // Carter constant (E = 1); on the equator p_theta^2 = Q
  double vr = 0.0;     // dr/dtau' (reversed time) at the disk crossing: p_r has the sign of -vr (needed for gas that moves radially: the plunging region of the lamppost view)
  int rejects = 0;     // rejected steps (error > 1 or non-finite); R0.2 diagnostic, read through trace_batch
};

// v2.1a: q2 = Q^2 (Kerr-Newman): only changes Delta = r^2 - 2 r + a^2 + q2 (and the ZAMO angular velocity, below); q2 = 0 is the usual Kerr
// Non-contracting multiplications (nor sharing sub-expressions with the rest): the Lambda terms only exist if Lambda != 0 and must not alter the FMAs of the Kerr code
// (the byte-for-byte regression of the float cases failed when these terms used a shared r*r)
__device__ __forceinline__ float mulrn(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ double mulrn(double a, double b) { return __dmul_rn(a, b); }
template <class T>
__device__ __forceinline__ T lam_r4(T lam, T r) { return mulrn(mulrn(lam, mulrn(r, r)), mulrn(r, r)); }  // Lambda r^4
// v2.1c: lam = Lambda (Schwarzschild-de Sitter, only a = 0, q2 = 0): Delta = r^2 f(r) = r^2 - 2 r - Lambda r^4/3 (R = r^4 - Delta K is still the radial null function, E = 1)
__device__ __forceinline__ double delta_d(double r, double a, double q2 = 0.0, double lam = 0.0) {
  double d = r * r - 2.0 * r + a * a + q2;
  if (lam != 0.0) d -= lam_r4(lam, r) / 3.0;
  return d;
}

struct Init {
  double L, Q, pr, ptheta, E_loc;
};
// O6: the part of init_from_zamo that depends only on the observer (a, r, theta, q2, lam), not on the ray. It is computed once per launch on the
// device (same device functions, so the bits are those of the per-ray version) instead of once per ray: sin, cos, three sqrt and several divisions
// in FP64, which runs at 1/64 of the FP32 rate on GeForce cards (notes/physics/36-ncu-first-profile.md).
struct ZamoC {
  double s, mu, alpha, omega, varpi, rsd, sS, sig;  // sin/cos theta, lapse, frame dragging, varpi, sqrt(Sigma/Delta), sqrt(Sigma), max(1e-14, 1 - mu^2)
};
__device__ inline ZamoC zamo_const(double a, double r, double theta, double q2 = 0.0, double lam = 0.0) {
  ZamoC z;
  z.s = sin(theta); z.mu = cos(theta);
  const double s = z.s, mu = z.mu;
  const double Sigma = r * r + a * a * mu * mu, Delta = delta_d(r, a, q2, lam);
  const double rr = r * r + a * a;
  const double A = rr * rr - a * a * Delta * s * s;
  z.alpha = sqrt(Sigma * Delta / A); z.omega = a * (2.0 * r - q2) / A;
  z.varpi = sqrt(A / Sigma) * s;
  z.rsd = sqrt(Sigma / Delta); z.sS = sqrt(Sigma);
  z.sig = fmax(1e-14, 1.0 - mu * mu);
  return z;
}
__device__ __forceinline__ Init init_from_zamo(const ZamoC& z, double a, double nr, double nth, double nph) {
  const double mu = z.mu;
  Init o;
  o.E_loc = 1.0 / (z.alpha + z.omega * z.varpi * nph);
  o.L = z.varpi * o.E_loc * nph;
  o.pr = o.E_loc * nr * z.rsd;
  o.ptheta = o.E_loc * nth * z.sS;
  o.Q = o.ptheta * o.ptheta + mu * mu * (o.L * o.L / z.sig - a * a);
  return o;
}
__device__ inline Init init_from_zamo(double a, double r, double theta, double nr, double nth, double nph, double q2 = 0.0, double lam = 0.0) {
  return init_from_zamo(zamo_const(a, r, theta, q2, lam), a, nr, nth, nph);
}

// v4.0: the metric is a compile-time tag. GenericM keeps the old run-time tests (hooks, selftests); KerrM (q2 = lam = 0), KerrNewmanM (lam = 0) and
// SdSM (a = q2 = 0, Lambda term always on) remove the dead branches, so the Kerr code path carries no Lambda / charge test at all.
// The arithmetic of each live branch is the same expression as before (goldens: tests/golden).
struct GenericM { static constexpr int kind = -1; };
struct KerrM { static constexpr int kind = 0; };
struct KerrNewmanM { static constexpr int kind = 1; };
struct SdSM { static constexpr int kind = 2; };

template <class T, class M = GenericM>
struct Cn {
  T a, a2, L, Q, K, aL, q2, lam;  // lam = Lambda (0 outside SdS); K = (L-a)^2 + Q; q2 = Q_charge^2 (0 in Kerr)
};
template <class T, class M>
__device__ __forceinline__ T Rf(T r, const Cn<T, M>& c) {
  const T A = r * r + c.a2 - c.aL;
  T D = r * r - T(2) * r + c.a2 + c.q2;
  if constexpr (M::kind == 2) D -= lam_r4(c.lam, r) / T(3);
  else if constexpr (M::kind == -1) { if (c.lam != T(0)) D -= lam_r4(c.lam, r) / T(3); }
  return A * A - D * c.K;
}
template <class T, class M>
__device__ __forceinline__ T dRf(T r, const Cn<T, M>& c) {
  T Dp = T(2) * r - T(2);
  if constexpr (M::kind == 2) Dp -= mulrn(T(4) * c.lam, mulrn(r, mulrn(r, r))) / T(3);
  else if constexpr (M::kind == -1) { if (c.lam != T(0)) Dp -= mulrn(T(4) * c.lam, mulrn(r, mulrn(r, r))) / T(3); }
  return T(4) * r * (r * r + c.a2 - c.aL) - Dp * c.K;
}
template <class T, class M>
__device__ __forceinline__ T Mf(T mu, T sig, const Cn<T, M>& c) {  // sig = 1 - mu^2, kept separately (see deriv)
  return sig * (c.Q + c.a2 * mu * mu) - c.L * c.L * mu * mu;
}
template <class T, class M>
__device__ __forceinline__ T dMf(T mu, const Cn<T, M>& c) {
  return T(2) * (c.a2 - c.Q - c.L * c.L) * mu - T(4) * c.a2 * mu * mu * mu;
}
template <class T, class M>
__device__ __forceinline__ T dphif(T r, T om, const Cn<T, M>& c) {  // om = sigma = 1 - mu^2
  om = fmax(T(1e-14), om);
  // v2.7: Kerr-Newman/SdS Delta (q2 was missing here: wrong phi and division by zero at Kerr's r = r+). In Kerr (q2 = lam = 0) the old branch, bit for bit.
  T D;
  if constexpr (M::kind == 0) D = r * r - T(2) * r + c.a2;
  else if constexpr (M::kind == 1) D = r * r - T(2) * r + c.a2 + c.q2;
  else if constexpr (M::kind == 2) { D = r * r - T(2) * r + c.a2 + c.q2; D -= lam_r4(c.lam, r) / T(3); }
  else {
    if (c.q2 != T(0) || c.lam != T(0)) { D = r * r - T(2) * r + c.a2 + c.q2; if (c.lam != T(0)) D -= lam_r4(c.lam, r) / T(3); }
    else D = r * r - T(2) * r + c.a2;
  }
  return c.L / om - c.a + c.a * (r * r + c.a2 - c.aL) / D;
}

template <class T, class M>
__device__ __forceinline__ void deriv(const T* s, T* d, const Cn<T, M>& c) {
  // state: r, dr/dtau, mu, dmu/dtau, phi, sigma = 1 - mu^2.  sigma is redundant with mu, but near the poles
  // (mu ~ 1) 1 - mu^2 loses all its digits in float (mu has resolution 6e-8 and 1-mu^2 ~ 1e-4): the term
  // L/(1-mu^2) of dphi/dtau gets noisy and the step control does not converge. sigma evolves with
  // d sigma/dtau = -2 mu dmu/dtau and keeps relative precision near the pole.
  d[0] = s[1];
  d[1] = T(0.5) * dRf(s[0], c);
  d[2] = s[3];
  d[3] = T(0.5) * dMf(s[2], c);
  d[4] = -dphif(s[0], s[5], c);
  d[5] = T(-2) * s[2] * s[3];
}

template <class S, class F>
__device__ __forceinline__ F* tf_from(const S* y, F* tf) {
#pragma unroll
  for (int i = 0; i < 6; ++i) tf[i] = F(y[i]);
  return tf;
}

// One Dormand-Prince 5(4) step; returns the normalised error. k1 and k7 are kept for the Hermite.
// S = state type (accumulator), F = type of the derivatives and increments. S = F: uniform precision;
// S = double, F = float: "mixed" (float increments added to a double accumulator).
template <class S, class F, class M>
__device__ __forceinline__ F dp45_step(const Cn<F, M>& c, const S* y, S h, S* yn, F* k1, F* k7, F atol, F rtol, F atol_phi) {
  F k2[6], k3[6], k4[6], k5[6], k6[6], tf[6];
  S t[6];
  const F hf = F(h);
  deriv(tf_from(y, tf), k1, c);
#pragma unroll
  for (int i = 0; i < 6; ++i) { t[i] = y[i] + S(hf * (F(1.0 / 5) * k1[i])); tf[i] = F(t[i]); }
  deriv(tf, k2, c);
#pragma unroll
  for (int i = 0; i < 6; ++i) { t[i] = y[i] + S(hf * (F(3.0 / 40) * k1[i] + F(9.0 / 40) * k2[i])); tf[i] = F(t[i]); }
  deriv(tf, k3, c);
#pragma unroll
  for (int i = 0; i < 6; ++i) { t[i] = y[i] + S(hf * (F(44.0 / 45) * k1[i] + F(-56.0 / 15) * k2[i] + F(32.0 / 9) * k3[i])); tf[i] = F(t[i]); }
  deriv(tf, k4, c);
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    t[i] = y[i] + S(hf * (F(19372.0 / 6561) * k1[i] + F(-25360.0 / 2187) * k2[i] + F(64448.0 / 6561) * k3[i] + F(-212.0 / 729) * k4[i]));
    tf[i] = F(t[i]);
  }
  deriv(tf, k5, c);
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    t[i] = y[i] + S(hf * (F(9017.0 / 3168) * k1[i] + F(-355.0 / 33) * k2[i] + F(46732.0 / 5247) * k3[i] + F(49.0 / 176) * k4[i] +
                          F(-5103.0 / 18656) * k5[i]));
    tf[i] = F(t[i]);
  }
  deriv(tf, k6, c);
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    yn[i] = y[i] + S(hf * (F(35.0 / 384) * k1[i] + F(500.0 / 1113) * k3[i] + F(125.0 / 192) * k4[i] + F(-2187.0 / 6784) * k5[i] +
                           F(11.0 / 84) * k6[i]));
    tf[i] = F(yn[i]);
  }
  deriv(tf, k7, c);
  F err = 0;
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    const F e = hf * (F(71.0 / 57600) * k1[i] - F(71.0 / 16695) * k3[i] + F(71.0 / 1920) * k4[i] - F(17253.0 / 339200) * k5[i] +
                      F(22.0 / 525) * k6[i] - F(1.0 / 40) * k7[i]);
    const F ym = F(fmax(fabs(y[i]), fabs(yn[i])));
    // sigma = 1 - mu^2: purely RELATIVE and coarse control (5 %). Near the pole sigma goes through a very narrow
    // minimum ~ L^2 and the phi integrand (L/sigma) has a spike of width ~ L there: without this the step
    // skips the spike (the error estimated at the stages does not see it) and the jump of pi in phi is lost.
    const F w = i == 5 ? F(1e-30) + F(0.05) * ym : (i == 4 ? atol_phi : atol) + rtol * ym;
    err = fmax(err, fabs(e) / w);
  }
  return err;
}
// O3 (refinement phase, notes/physics/33-dop853.md): one DOP853 step (Hairer, order 8) for the pure-double thin-disk/sky path. k1 = f(y), kend = f(yn)
// (the same roles as DP45's k1/k7, so the cubic Hermite of the crossing search and the volume sampling are unchanged). Error: Hairer's combination of the
// order-5 and order-3 estimators, err_i = h e5^2 / sqrt(e5^2 + 0.01 e3^2) / w_i, applied per component with the same weights as dp45_step, max norm.
template <class M>
__device__ double dop853_step(const Cn<double, M>& c, const double* y, double h, double* yn, double* k1, double* kend, double atol, double rtol, double atol_phi) {
  using namespace dop853;
  double K[kStages][6], t[6];
  deriv(y, K[0], c);
#pragma unroll
  for (int s = 1; s < kStages; ++s) {
#pragma unroll
    for (int i = 0; i < 6; ++i) {
      double acc = 0;
#pragma unroll
      for (int j = 0; j < s; ++j) if (A[s][j] != 0.0) acc += A[s][j] * K[j][i];
      t[i] = y[i] + h * acc;
    }
    deriv(t, K[s], c);
  }
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    double acc = 0;
#pragma unroll
    for (int j = 0; j < kStages; ++j) if (B[j] != 0.0) acc += B[j] * K[j][i];
    yn[i] = y[i] + h * acc;
  }
  deriv(yn, kend, c);
  double err = 0;
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    double e5 = E5[kStages] * kend[i], e3 = E3[kStages] * kend[i];
#pragma unroll
    for (int j = 0; j < kStages; ++j) { if (E5[j] != 0.0) e5 += E5[j] * K[j][i]; if (E3[j] != 0.0) e3 += E3[j] * K[j][i]; }
    const double ym = fmax(fabs(y[i]), fabs(yn[i]));
    const double w = i == 5 ? 1e-30 + 0.05 * ym : (i == 4 ? atol_phi : atol) + rtol * ym;
    const double den = sqrt(e5 * e5 + 0.01 * e3 * e3);
    if (den > 0.0) err = fmax(err, fabs(h) * e5 * e5 / den / w);
    k1[i] = K[0][i];
  }
  return err;
}
__device__ __forceinline__ double step_factor8(double err) {  // order 8: err^(-1/8), Hairer's limits [1/3, 6]
  if (err <= 0.0) return 6.0;
  return fmin(6.0, fmax(1.0 / 3.0, 0.9 * pow(err, -0.125)));
}

template <class T>
__device__ __forceinline__ T step_factor(T err) {
  if (err <= T(0)) return T(5);
  return fmin(T(5), fmax(T(0.2), T(0.9) * pow(err, T(-0.2))));
}

// ---------------------------------------------------------------------------------------------------------------------
// O1 (refinement phase, notes/physics/30-analytic-escape-tail.md): analytic escape tail. An outgoing ray (dr/dtau > 0) of an
// E = 1 photon has no radial turning point left (one at most outside the horizon, and it is behind it), so from r_t to infinity
// its radial motion is fixed and Mino-time separability gives the rest exactly:
//   T     = int_{r_t}^inf dr / sqrt(R)               (Mino time still to go)
//   Phi_r = -int_{r_t}^inf g(r) dr / sqrt(R),  g = a (r^2 + a^2 - aL)/Delta - a = a (2r - q2 - aL)/Delta   (radial part of phi)
// With u = 1/r both become smooth integrals on the short interval [0, 1/r_t]:
//   dr/sqrt(R) = du / sqrt(P(u)),  P(u) = R u^4 = (1 + (a^2 - aL) u^2)^2 - K u^2 (1 - 2u + (a^2 + q2) u^2),  P(0) = 1
//   g dr/sqrt(R) = a u (2 - (q2 + aL) u) / (1 - 2u + (a^2 + q2) u^2) du / sqrt(P(u))
// done with 8-point Gauss-Legendre (the integrands are polynomial/rational and smooth there: ~machine precision).
// Then mu and the angular part of phi (dphi = -L/sigma dtau) advance by T with the angular equations alone (adaptive DP45 on
// (mu, dmu/dtau, phi, sigma), same tolerances and pole-crossing step cap as the main loop). Kerr and Kerr-Newman (Lambda = 0) only.
// This replaces the ~45 % of the steps spent between r ~ 1e3 and r_far = 1e6 and is MORE accurate: at finite r_far the position
// angles differ from the asymptotic direction by O(b / r_far).
template <class T>
__device__ __forceinline__ void gl8(int k, T& x, T& w) {  // Gauss-Legendre nodes/weights on [-1, 1] (k = 0..7)
  const double xs[4] = {0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363};
  const double ws[4] = {0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763};
  const int i = k < 4 ? k : k - 4;
  x = T(k < 4 ? -xs[i] : xs[i]); w = T(ws[i]);
}
template <class S, class F, class M>
__device__ void escape_tail(const Cn<S, M>& c, S* y, F at, F rt, F atp) {
  const S ut = S(1) / y[0];
  S Tm = 0, Phr = 0;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    S x, w;
    gl8<S>(k, x, w);
    const S u = S(0.5) * ut * (x + S(1)), u2 = u * u;
    const S A = S(1) + (c.a2 - c.aL) * u2, D = S(1) - S(2) * u + (c.a2 + c.q2) * u2;
    const S P = A * A - c.K * u2 * D;
    const S inv = S(1) / sqrt(fmax(P, S(1e-30)));
    Tm += w * inv;
    Phr += w * c.a * u * (S(2) - (c.q2 + c.aL) * u) / D * inv;
  }
  Tm *= S(0.5) * ut; Phr *= S(0.5) * ut;
  y[4] -= Phr;
  // angular part over the Mino time Tm: z = (mu, dmu/dtau, phi, sigma)
  auto f = [&](const S* z, S* d) {
    d[0] = z[1];
    d[1] = S(0.5) * dMf(z[0], c);
    d[2] = -c.L / fmax(S(1e-14), z[3]);
    d[3] = S(-2) * z[0] * z[1];
  };
  S z[4] = {y[2], y[3], y[4], y[5]};
  S tau = 0, h = Tm;
  for (int it = 0; it < 4000 && tau < Tm; ++it) {
    if (z[3] < S(0.2)) {
      const S curv = fabs(S(2) * Mf(z[0], z[3], c) + z[0] * dMf(z[0], c));
      h = fmin(h, S(0.5) * sqrt(fmax(z[3], S(1e-30)) / fmax(curv, S(1e-12))));
    }
    h = fmin(h, Tm - tau);
    S k1[4], k2[4], k3[4], k4[4], k5[4], k6[4], k7[4], t[4], zn[4];
    f(z, k1);
    for (int i = 0; i < 4; ++i) t[i] = z[i] + h * (S(1.0 / 5) * k1[i]);
    f(t, k2);
    for (int i = 0; i < 4; ++i) t[i] = z[i] + h * (S(3.0 / 40) * k1[i] + S(9.0 / 40) * k2[i]);
    f(t, k3);
    for (int i = 0; i < 4; ++i) t[i] = z[i] + h * (S(44.0 / 45) * k1[i] + S(-56.0 / 15) * k2[i] + S(32.0 / 9) * k3[i]);
    f(t, k4);
    for (int i = 0; i < 4; ++i) t[i] = z[i] + h * (S(19372.0 / 6561) * k1[i] + S(-25360.0 / 2187) * k2[i] + S(64448.0 / 6561) * k3[i] + S(-212.0 / 729) * k4[i]);
    f(t, k5);
    for (int i = 0; i < 4; ++i) t[i] = z[i] + h * (S(9017.0 / 3168) * k1[i] + S(-355.0 / 33) * k2[i] + S(46732.0 / 5247) * k3[i] + S(49.0 / 176) * k4[i] + S(-5103.0 / 18656) * k5[i]);
    f(t, k6);
    for (int i = 0; i < 4; ++i) zn[i] = z[i] + h * (S(35.0 / 384) * k1[i] + S(500.0 / 1113) * k3[i] + S(125.0 / 192) * k4[i] + S(-2187.0 / 6784) * k5[i] + S(11.0 / 84) * k6[i]);
    f(zn, k7);
    S err = 0;
    for (int i = 0; i < 4; ++i) {
      const S e = h * (S(71.0 / 57600) * k1[i] - S(71.0 / 16695) * k3[i] + S(71.0 / 1920) * k4[i] - S(17253.0 / 339200) * k5[i] + S(22.0 / 525) * k6[i] - S(1.0 / 40) * k7[i]);
      const S ym = fmax(fabs(z[i]), fabs(zn[i]));
      const S w = i == 3 ? S(1e-30) + S(0.05) * ym : (i == 2 ? S(atp) : S(at)) + S(rt) * ym;
      err = fmax(err, fabs(e) / w);
    }
    if (err > S(1) || !isfinite(err)) { h *= S(isfinite(err) ? fmax(S(0.2), S(0.9) * pow(err, S(-0.2))) : S(0.2)); continue; }
    tau += h;
    // same reprojection as the main loop: mu/sigma consistency and dmu/dtau onto M(mu)
    if (zn[3] < S(0.5)) zn[0] = copysign(sqrt(fmax(S(1) - zn[3], S(0))), zn[0]);
    else zn[3] = S(1) - zn[0] * zn[0];
    zn[1] = copysign(sqrt(fmax(Mf(zn[0], zn[3], c), S(0))), zn[1]);
    for (int i = 0; i < 4; ++i) z[i] = zn[i];
    h *= S(err <= S(0) ? S(5) : fmin(S(5), fmax(S(0.2), S(0.9) * pow(err, S(-0.2)))));
  }
  y[2] = z[0]; y[3] = z[1]; y[4] = z[2]; y[5] = z[3];
}

// Without volume (default): the trace only looks for the equatorial-plane crossing (thin disk). The `if constexpr (Vol::kActive)`
// guarantees that the code generated for this case is that of v0.9a2 (byte-for-byte regression).
struct NoVol { static constexpr bool kActive = false; __device__ void begin(double, double) {} };

// Ray tracing (equivalent to kerr::trace). PHOTON direction (nr, nth, nph) in the ZAMO frame.
// Vol (v0.9b): thick-disk functor; vol->sigma(r) = H/r, vol->add(r, mu, dMino) accumulates the radiative transport
// over dense samples (Hermite) of each accepted step; the ray ends (fate = 1) when the transmittance drops below 1e-3.
template <class S, class F, class Vol = NoVol, class M = GenericM>
__device__ void trace(double a, double r0, double theta0, double nr, double nth, double nph, double disk_rin, double disk_rout,
                      double rtol, double atol, double atol_phi, double l_min, double r_far, int max_steps, Trace& out, int skip = 0, Vol* vol = nullptr, double q2 = 0.0, double lam = 0.0, double r_tail = 0.0, int integ = 0, const ZamoC* zc = nullptr) {
  // zc: observer constants hoisted out of the per-ray code (O6); null = computed here (hooks, selftests)
  const ZamoC zl = zc ? *zc : zamo_const(a, r0, theta0, q2, lam);
  Init in = init_from_zamo(zl, a, nr, nth, nph);
  // L ~ 0: the ray plane contains the axis and the ray crosses the pole, where phi (Boyer-Lindquist) is singular:
  // with exact L = 0 phi does not jump by pi; with |L| ~ 1e-8 the jump is stiff (1e5 steps). |L| >= l_min is enforced.
  if (fabs(in.L) < l_min) in.L = copysign(l_min, in.L == 0.0 ? 1.0 : in.L);
  Cn<S, M> cS{S(a), S(a * a), S(in.L), S(in.Q), S((in.L - a) * (in.L - a) + in.Q), S(a * in.L), S(q2), S(lam)};  // projection always in the state type
  Cn<F, M> c;
  c.a = F(a); c.a2 = F(a * a); c.L = F(in.L); c.Q = F(in.Q);
  c.K = F((in.L - a) * (in.L - a) + in.Q);
  c.aL = F(a * in.L); c.q2 = F(q2); c.lam = F(lam);
  out = Trace{2, 0, 0, 0, 0, 0, 0, in.L, in.Q};
  int skipped = 0;
  if constexpr (Vol::kActive) vol->begin(in.L, in.Q);
  const double disc = 1.0 - a * a - q2;  // < 0: naked singularity (no horizon): rays reaching r_min = 0.05 are given as absorbed (black)
  double r_plus = disc >= 0.0 ? 1.0 + sqrt(disc) : 0.0, r_far_eff = r_far, r_c = 0.0;  // r_c: cosmological horizon (SdS only)
  if (lam > 0.0) {  // Schwarzschild-de Sitter (a = q2 = 0, 0 < Lambda < 1/9): the black-hole and cosmological horizons (roots of Lambda r^3/3 - r + 2 = 0, trigonometric method);
    const double th = acos(-3.0 * sqrt(lam)) / 3.0, k = 2.0 / sqrt(lam);  // the ray "escapes" on reaching r_c (0.9999 r_c): the (theta, phi) direction at that point is the sky one
    const double x0 = k * cos(th), x1 = k * cos(th - 2.0943951023931953), x2 = k * cos(th - 4.1887902047863905);
    double lo = fmin(x0, fmin(x1, x2)), hi = fmax(x0, fmax(x1, x2)), mid = x0 + x1 + x2 - lo - hi;
    r_plus = mid; r_far_eff = 0.9999 * hi; r_c = hi;
  }
  const S r_cap = S(disc >= 0.0 ? r_plus + 1e-3 : 0.05), rfar = S(r_far_eff);
  // a camera inside the black-hole horizon, or beyond the cosmological one, has no static frame: treat the ray as absorbed instead of integrating NaNs for max_steps (watchdog)
  if (disc >= 0.0 && r0 <= r_plus) { out.fate = 1; return; }
  if (lam > 0.0 && r0 >= r_far_eff) { out.fate = 1; return; }
  // directions that cannot be traced back to infinity with E = 1: the ZAMO energy E_loc = 1/(alpha + omega varpi n_phi) is NEGATIVE for a camera inside the ergoregion looking against the frame
  // dragging (the photon would have negative energy at infinity: it comes out of the past horizon), and it diverges (|E_loc| -> inf, L ~ E_loc, Q ~ E_loc^2) at the boundary of that set, where
  // R = A^2 - Delta K cancels catastrophically (float garbage already at E_loc ~ 10^3) and the step control never converges. Found by the sweep (KN a = 0.5, Q = 0.5, r = 1.7412, ergoregion
  // up to r = 1.86): 2 rays x 200000 steps = 12 s per frame. Both are black (captured), the physical answer for backward tracing.
  if (!(in.E_loc > 0.0 && in.E_loc < (sizeof(F) == 4 ? 1e3 : 1e6))) { out.fate = 1; return; }
  const double mu0 = zl.mu, s0 = zl.s;
  // dr/dtau = -Delta p_r: Delta WITH the Lambda term (v4.1 fix, found with O3: without it the SdS initial radial velocity was wrong since v2.1c; the per-step
  // reprojection onto sqrt(R) hid most of it, but the result depended on the step sizes: up to 6e-3 rad at tol 1e-12, notes/physics/33-dop853.md)
  S y[6] = {S(r0), S(-delta_d(r0, a, q2, lam) * in.pr), S(mu0), S(s0 * in.ptheta), S(0), S(s0 * s0)};
  S h = S(0.02 / r0);
  const F at = F(atol), rt = F(rtol), atp = F(atol_phi);
  const bool disk_on = disk_rout > 0;
  const bool tail_on = r_tail > 0.0 && lam == 0.0;  // O6: the double comparisons once, not in every step
  const S rtail_s = S(r_tail);
  int nan_rej = 0;
  for (int step = 0; step < max_steps; ++step) {
    // pole crossing (see kerr::Options::pole_kappa): the step cannot skip the narrow minimum of sigma = 1 - mu^2,
    // where the phi integrand (L/sigma) has a spike; limited to 0.5 * sqrt(sigma / |sigma''|), sigma'' = -(2M + mu M').
    if (y[5] < S(0.2)) {
      const S curv = fabs(S(2) * Mf(y[2], y[5], cS) + y[2] * dMf(y[2], cS));
      h = fmin(h, S(0.5) * sqrt(fmax(y[5], S(1e-30)) / fmax(curv, S(1e-12))));
    }
    if constexpr (Vol::kActive) {
      // near the layer the step cannot skip the Gaussian: |dmu| <= max(0.6 sigma, (|mu| - 4 sigma)/2) per step (approaching
      // from outside the distance to 4 sigma is at most halved); limit by the speed and by the curvature of mu(tau)
      // (in float: doubles are 1/64 of the throughput on GeForce cards and this runs in every step)
      const float sg = vol->sigma(float(y[0])), am = fabsf(fabsf(float(y[2])) - vol->mu_s());  // distance to the emitting surface |mu| = mu_s (E2; 0 = equatorial layer)
      const float allow = fmaxf(0.6f * sg, 0.5f * (am - 4.0f * sg)), rate = fabsf(float(y[3]));
      float hcap = sqrtf(2.0f * allow / (fabsf(0.5f * float(dMf(y[2], cS))) + 1e-12f));
      if (rate > 0.0f) hcap = fminf(hcap, allow / rate);
      h = fmin(h, S(hcap));
      h = fmin(h, S(vol->cap(float(y[0]))));  // the model's own step limit (torus: path length per step)
    }
    // SdS: an outgoing step must not jump over the cosmological horizon r_c (Delta = 0; beyond it r is timelike): a DOP853 step from r = 15.4 to 17.2 across
    // r_c = 16.2 was accepted by its error estimate (err 0.45) with a real error of 1e-6 rad. Each step covers at most half the remaining distance to r_c
    // (r'/r ~ const, so this costs a few steps), and the exit at 0.9999 r_c is reached with short steps.
    if constexpr (M::kind == 2 || M::kind == -1) { if (lam > 0.0 && y[1] > S(0)) h = fmin(h, S(0.5) * (S(r_c) - y[0]) / y[1]); }
    S yn[6];
    F k1[6], k7[6];
    // O3: DOP853 only where it pays (pure double, no volume: the volume sampling relies on short Hermite segments); integ = 1 selects it
    constexpr bool kCan853 = std::is_same<S, double>::value && std::is_same<F, double>::value && !Vol::kActive;
    bool use853 = false;
    F err;
    if constexpr (kCan853) {
      use853 = integ == 1;
      // calibration: DOP853 at tol/10 is no worse than DP45 at tol in every quantile (max, p99, median of the sky direction and of disk r/phi), and faster
      if (use853) err = dop853_step<M>(c, y, h, yn, k1, k7, 0.1 * at, 0.1 * rt, 0.1 * atp);
      else err = dp45_step<S, F>(c, y, h, yn, k1, k7, at, rt, atp);
    } else {
      err = dp45_step<S, F>(c, y, h, yn, k1, k7, at, rt, atp);
    }
    if (err > F(1) || !isfinite(err)) {  // v2.7: a NaN/inf error (repulsive core of naked KN, extremes) is NOT accepted as good: it is rejected and, if it persists, the ray ends (absorbed)
      ++out.rejects;
      if (!isfinite(err) && ++nan_rej > 40) { out.fate = 1; return; }
      if constexpr (kCan853) { if (use853) { h *= isfinite(err) ? step_factor8(err) : 1.0 / 3.0; continue; } }
      h *= S(step_factor(err)); continue;
    }
    ++out.steps;
    const S yn1 = yn[1], yn3 = yn[3];
    // mu and sigma = 1 - mu^2 are redundant: near the pole (sigma < 1/2) sigma rules (relative precision), otherwise mu
    if (yn[5] < S(0.5)) yn[2] = copysign(sqrt(fmax(S(1) - yn[5], S(0))), yn[2]);
    else yn[5] = S(1) - yn[2] * yn[2];
    // Velocity reprojection onto (dr/dtau)^2 = R(r) and (dmu/dtau)^2 = M(mu) (v0.6a: keeps the invariants), but only where it is well conditioned
    // (v4.1, found with O3: notes/physics/33-dop853.md). Near a turning point sqrt(R(r)) turns a position error dr into a velocity error ~sqrt(R' dr) and,
    // through the sign, a jump in Mino time: one SdS ray ending its step on the periapsis (dr/dtau = 8e-6) came out 1e-6 rad off at tol 1e-9, and in float
    // the worst rays of every grid were these (Kerr float max error 7.0e-4 -> 2.2e-5 rad). The projection is applied when the amplification is small,
    // y'^2 >= 100 |dR/dr| (atol + rtol |r|) (same for mu with M); otherwise the second-order equation carries the velocity through the turning point.
    // Threshold 100: 10 and 1000 measured within 2-4x of it; no change in the step count. The angular projection keeps a pole guard (below).
    {
      const S tolr = S(at) + S(rt) * fabs(yn[0]), tolm = S(at) + S(rt) * fabs(yn[2]);
      // R'(r_n) = 2 k7[1] and M'(mu_n) = 2 k7[3]: k7 = f(y_n) is already computed by the step (free; recomputing dRf/dMf cost up to 15 % in the float volume kernels)
      if (yn1 * yn1 >= S(200) * fabs(S(k7[1])) * tolr) yn[1] = copysign(sqrt(fmax(Rf(yn[0], cS), S(0))), yn1);
      // near the poles (sigma < 0.1) always: there the mu turning point coincides with sigma -> 0 and float rays along the axis got stuck without it (selftest_axis)
      if (yn3 * yn3 >= S(200) * fabs(S(k7[3])) * tolm || yn[5] < S(0.1)) yn[3] = copysign(sqrt(fmax(Mf(yn[2], yn[5], cS), S(0))), yn3);
    }

    if (!Vol::kActive && disk_on && !out.disk_hit && (y[2] > S(0)) != (yn[2] > S(0))) {
      S lo = 0, hi = 1;
      const bool pos0 = y[2] > S(0);
      const int iters = sizeof(S) == 8 ? 60 : 28;
      for (int it = 0; it < iters; ++it) {
        const S m = S(0.5) * (lo + hi);
        const S s = m, s2 = s * s, s3 = s2 * s;
        const S v = (S(2) * s3 - S(3) * s2 + S(1)) * y[2] + (s3 - S(2) * s2 + s) * h * k1[2] + (S(-2) * s3 + S(3) * s2) * yn[2] +
                    (s3 - s2) * h * k7[2];
        ((v > S(0)) == pos0 ? lo : hi) = m;
      }
      const S s = S(0.5) * (lo + hi), s2 = s * s, s3 = s2 * s;
      const S h00 = S(2) * s3 - S(3) * s2 + S(1), h10 = s3 - S(2) * s2 + s, h01 = S(-2) * s3 + S(3) * s2, h11 = s3 - s2;
      S rc = h00 * y[0] + h10 * h * k1[0] + h01 * yn[0] + h11 * h * k7[0];
      S pc = h00 * y[4] + h10 * h * k1[4] + h01 * yn[4] + h11 * h * k7[4];
      if constexpr (kCan853) {
        // O3: with DOP853 steps the cubic Hermite (O(h^4)) would limit the crossing; one Newton correction with real sub-steps from y:
        // z = step(y, s h), s -= mu(z) / (h dmu/dtau(z)), then the final sub-step gives r and phi at mu = 0 to the integrator's accuracy
        if (use853) {
          double z[6], d1[6], d2[6], sn = double(s);
          dop853_step<M>(c, y, sn * h, z, d1, d2, 0.1 * at, 0.1 * rt, 0.1 * atp);
          if (z[3] != 0.0) sn -= z[2] / (h * z[3]);
          sn = fmin(1.0, fmax(0.0, sn));
          dop853_step<M>(c, y, sn * h, z, d1, d2, 0.1 * at, 0.1 * rt, 0.1 * atp);
          rc = z[0]; pc = z[4];
        }
      }
      if (rc > S(disk_rin) && rc < S(disk_rout) && skipped++ >= skip) {
        out.disk_hit = 1; out.disk_r = double(rc); out.disk_phi = double(pc); out.vr = double(yn[1]);
        out.fate = 1;
        return;
      }
    }
    if constexpr (Vol::kActive) {
      auto hm = [&](int i, S s) {  // cubic Hermite of y[i] over the step (value and derivative at the ends)
        const S s2 = s * s, s3 = s2 * s;
        return (S(2) * s3 - S(3) * s2 + S(1)) * y[i] + (s3 - S(2) * s2 + s) * h * k1[i] + (S(-2) * s3 + S(3) * s2) * yn[i] + (s3 - s2) * h * k7[i];
      };
      const float lim = 4.0f * vol->sigma(float(y[0]));
      const float mmid = float(hm(2, S(0.5)));
      // slow light: dt/dtau (Mino, E = 1) = (r^2 + a^2)(r^2 + a^2 - aL)/Delta + a (L - a sigma), sigma = 1 - mu^2 (state y[5])
      const float af = float(a), a2f = af * af, Lf = float(in.L), aLf = af * Lf;
      const float q2f = float(q2), lamf = float(lam);
      auto tdot = [&](float r, float sg) { const float A = r * r + a2f; float D = r * r - 2.f * r + a2f + q2f; if (lamf != 0.f) D -= lam_r4(lamf, r) / 3.f; return A * (A - aLf) / D + af * (Lf - af * sg); };
      const float ms = vol->mu_s();
      const bool relevant = fminf(fminf(fabsf(fabsf(float(y[2])) - ms), fabsf(fabsf(float(yn[2])) - ms)), fabsf(fabsf(mmid) - ms)) < lim || (y[2] > S(0)) != (yn[2] > S(0)) ||
                            ((fabsf(float(y[2])) > ms) != (fabsf(float(yn[2])) > ms));
      if (relevant) {
        const int K = vol->nsub;
        const float dh = float(h) / K;
        for (int q = 0; q < K; ++q) {
          const S s = (S(q) + S(0.5)) / S(K);  // O6: was (q + 0.5) / K, a double division per sub-sample (int + double literal); q + 0.5 and K are exact in float, IEEE division is correctly rounded
          const float rr = float(hm(0, s));
          if (vol->slow) {  // flight time at the midpoint of the sub-sample: T += f dh / 2 before and after the add
            const float f = tdot(rr, float(hm(5, s)));
            vol->T += 0.5f * f * dh;
            vol->add(rr, float(hm(2, s)), float(hm(4, s)), dh, float(hm(1, s)), float(hm(3, s)));
            vol->T += 0.5f * f * dh;
          } else {
            vol->add(rr, float(hm(2, s)), float(hm(4, s)), dh, float(hm(1, s)), float(hm(3, s)));
          }
        }
        if (vol->opaque()) { out.fate = 1; return; }
      } else if (vol->slow) {  // Simpson: h/6 (f0 + 4 fm + f1)
        const float f0 = tdot(float(y[0]), float(y[5])), f1 = tdot(float(yn[0]), float(yn[5])), fm = tdot(float(hm(0, S(0.5))), float(hm(5, S(0.5))));
        vol->T += float(h) * (f0 + 4.f * fm + f1) * (1.f / 6.f);
      }
    }
    // Schwarzschild-de Sitter exit (found with O3, notes/physics/33-dop853.md): the sky direction is the position at r = r_far_eff = 0.9999 r_c. Taking the first
    // accepted state beyond it made the answer depend on the last step's overshoot (two tight references, DP45 and DOP853 at tol 1e-12, differed by 1e-2 rad).
    // The crossing of r = r_far_eff is found on the cubic Hermite of the step (and, with DOP853, refined by a Newton correction with real sub-steps).
    if constexpr (M::kind == 2 || M::kind == -1) {  // only SdS (and the run-time generic path) can have Lambda > 0
      if (lam > 0.0 && yn[0] > rfar && yn[1] > S(0) && y[0] <= rfar) {
        auto hm = [&](int i, S s) { const S s2 = s * s, s3 = s2 * s; return (S(2) * s3 - S(3) * s2 + S(1)) * y[i] + (s3 - S(2) * s2 + s) * h * k1[i] + (S(-2) * s3 + S(3) * s2) * yn[i] + (s3 - s2) * h * k7[i]; };
        S lo = 0, hi = 1;
        for (int it = 0; it < (sizeof(S) == 8 ? 60 : 28); ++it) { const S m = S(0.5) * (lo + hi); (hm(0, m) < rfar ? lo : hi) = m; }
        const S s = S(0.5) * (lo + hi);
        double mu_x = double(hm(2, s)), ph_x = double(hm(4, s));
        if constexpr (kCan853) {
          if (use853) {
            double z[6], d1[6], d2[6], sn = double(s);
            dop853_step<M>(c, y, sn * h, z, d1, d2, 0.1 * at, 0.1 * rt, 0.1 * atp);
            if (z[1] != 0.0) sn -= (z[0] - double(rfar)) / (h * z[1]);
            sn = fmin(1.0, fmax(0.0, sn));
            dop853_step<M>(c, y, sn * h, z, d1, d2, 0.1 * at, 0.1 * rt, 0.1 * atp);
            mu_x = z[2]; ph_x = z[4];
          }
        }
        out.fate = 0;
        out.mu_end = fmin(1.0, fmax(-1.0, mu_x));
        out.phi_end = ph_x;
        return;
      }
    }
    const S r_prev = y[0];
#pragma unroll
    for (int i = 0; i < 6; ++i) y[i] = yn[i];
    if (y[0] < r_cap) { out.fate = 1; return; }
    if constexpr (M::kind == 0 || M::kind == 1 || M::kind == -1) {  // O1: analytic tail (Kerr / Kerr-Newman; r_tail = 0 = off)
      if (tail_on && y[0] > rtail_s && y[1] > S(0)) {
        escape_tail<S, F, M>(cS, y, at, rt, atp);
        out.fate = 0;
        out.mu_end = fmin(1.0, fmax(-1.0, double(y[2])));
        out.phi_end = double(y[4]);
        return;
      }
    }
    if (y[0] > rfar && y[1] > S(0)) {
      out.fate = 0;
      out.mu_end = fmin(1.0, fmax(-1.0, double(y[2])));
      out.phi_end = double(y[4]);
      return;
    }
    // O2 (refinement phase, notes/physics/32-step-control.md): on the way out r(tau) runs into a pole at finite Mino time (r ~ 1/(tau_inf - tau)), so the
    // step that meets the tolerance shrinks like 1/r (measured: h r ~ 0.19 at tol 2e-5). The error-based factor alone always lags one step behind: in float every
    // second attempt was rejected (34 % of all attempts). Scaling the proposal by r_prev/r keeps h r instead of h. Acceptance (err <= 1) is unchanged. At tight
    // tolerances the shrink per step is small and the plain controller keeps up (double: 1.7 % rejections; the scaling would cost +2.6 % steps), hence rtol >= 1e-6.
    if constexpr (kCan853) { if (use853) { h *= step_factor8(err); if (y[1] > S(0)) h *= fmin(1.0, r_prev / y[0]); continue; } }  // O2 scaling: order-8 steps are long, the 1/r shrink matters at any tolerance
    h *= S(step_factor(err));
    if (y[1] > S(0) && rt >= F(1e-6)) h *= fmin(S(1), r_prev / y[0]);
  }
  out.fate = 2;
}

}  // namespace bk
