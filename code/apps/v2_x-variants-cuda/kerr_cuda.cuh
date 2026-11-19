#pragma once
// Kerr physics (v0.6a, core/include/bhs/kerr.hpp) ported to the device, templated on the scalar type T
// (double = reference; float = fast mode). Same equations, same adaptive Dormand-Prince 5(4) in
// Mino time, same reprojection of the velocities onto (dr/dtau)^2 = R and (dmu/dtau)^2 = M after every step.
// The constants (L, Q) and the initial state always come from init_from_zamo in double (once per ray).
#include <cuda_runtime.h>

#include <cstdint>

namespace bk {

struct Trace {
  int fate;            // 0 escaped, 1 captured, 2 orbiting (max_steps)
  int disk_hit;
  int steps;           // accepted steps
  double mu_end, phi_end;  // direction the light came from (only if fate == 0): mu = cos(theta)
  double disk_r, disk_phi;
  double L;
  double Q;            // Carter constant (E = 1); on the equator p_theta^2 = Q
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
__device__ inline Init init_from_zamo(double a, double r, double theta, double nr, double nth, double nph, double q2 = 0.0, double lam = 0.0) {
  const double s = sin(theta), mu = cos(theta);
  const double Sigma = r * r + a * a * mu * mu, Delta = delta_d(r, a, q2, lam);
  const double rr = r * r + a * a;
  const double A = rr * rr - a * a * Delta * s * s;
  const double alpha = sqrt(Sigma * Delta / A), omega = a * (2.0 * r - q2) / A;
  const double varpi = sqrt(A / Sigma) * s;
  Init o;
  o.E_loc = 1.0 / (alpha + omega * varpi * nph);
  o.L = varpi * o.E_loc * nph;
  o.pr = o.E_loc * nr * sqrt(Sigma / Delta);
  o.ptheta = o.E_loc * nth * sqrt(Sigma);
  o.Q = o.ptheta * o.ptheta + mu * mu * (o.L * o.L / fmax(1e-14, 1.0 - mu * mu) - a * a);
  return o;
}

template <class T>
struct Cn {
  T a, a2, L, Q, K, aL, q2, lam;  // lam = Lambda (0 outside SdS); K = (L-a)^2 + Q; q2 = Q_charge^2 (0 in Kerr)
};
template <class T>
__device__ __forceinline__ T Rf(T r, const Cn<T>& c) {
  const T A = r * r + c.a2 - c.aL;
  T D = r * r - T(2) * r + c.a2 + c.q2;
  if (c.lam != T(0)) D -= lam_r4(c.lam, r) / T(3);
  return A * A - D * c.K;
}
template <class T>
__device__ __forceinline__ T dRf(T r, const Cn<T>& c) {
  T Dp = T(2) * r - T(2);
  if (c.lam != T(0)) Dp -= mulrn(T(4) * c.lam, mulrn(r, mulrn(r, r))) / T(3);
  return T(4) * r * (r * r + c.a2 - c.aL) - Dp * c.K;
}
template <class T>
__device__ __forceinline__ T Mf(T mu, T sig, const Cn<T>& c) {  // sig = 1 - mu^2, kept separately (see deriv)
  return sig * (c.Q + c.a2 * mu * mu) - c.L * c.L * mu * mu;
}
template <class T>
__device__ __forceinline__ T dMf(T mu, const Cn<T>& c) {
  return T(2) * (c.a2 - c.Q - c.L * c.L) * mu - T(4) * c.a2 * mu * mu * mu;
}
template <class T>
__device__ __forceinline__ T dphif(T r, T om, const Cn<T>& c) {  // om = sigma = 1 - mu^2
  om = fmax(T(1e-14), om);
  // v2.7: Kerr-Newman/SdS Delta (q2 was missing here: wrong phi and division by zero at Kerr's r = r+). In Kerr (q2 = lam = 0) the old branch, bit for bit.
  T D;
  if (c.q2 != T(0) || c.lam != T(0)) { D = r * r - T(2) * r + c.a2 + c.q2; if (c.lam != T(0)) D -= lam_r4(c.lam, r) / T(3); }
  else D = r * r - T(2) * r + c.a2;
  return c.L / om - c.a + c.a * (r * r + c.a2 - c.aL) / D;
}

template <class T>
__device__ __forceinline__ void deriv(const T* s, T* d, const Cn<T>& c) {
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
template <class S, class F>
__device__ __forceinline__ F dp45_step(const Cn<F>& c, const S* y, S h, S* yn, F* k1, F* k7, F atol, F rtol, F atol_phi) {
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
template <class T>
__device__ __forceinline__ T step_factor(T err) {
  if (err <= T(0)) return T(5);
  return fmin(T(5), fmax(T(0.2), T(0.9) * pow(err, T(-0.2))));
}

// Without volume (default): the trace only looks for the equatorial-plane crossing (thin disk). The `if constexpr (Vol::kActive)`
// guarantees that the code generated for this case is that of v0.9a2 (byte-for-byte regression).
struct NoVol { static constexpr bool kActive = false; __device__ void begin(double, double) {} };

// Ray tracing (equivalent to kerr::trace). PHOTON direction (nr, nth, nph) in the ZAMO frame.
// Vol (v0.9b): thick-disk functor; vol->sigma(r) = H/r, vol->add(r, mu, dMino) accumulates the radiative transport
// over dense samples (Hermite) of each accepted step; the ray ends (fate = 1) when the transmittance drops below 1e-3.
template <class S, class F, class Vol = NoVol>
__device__ void trace(double a, double r0, double theta0, double nr, double nth, double nph, double disk_rin, double disk_rout,
                      double rtol, double atol, double atol_phi, double l_min, double r_far, int max_steps, Trace& out, int skip = 0, Vol* vol = nullptr, double q2 = 0.0, double lam = 0.0) {
  Init in = init_from_zamo(a, r0, theta0, nr, nth, nph, q2, lam);
  // L ~ 0: the ray plane contains the axis and the ray crosses the pole, where phi (Boyer-Lindquist) is singular:
  // with exact L = 0 phi does not jump by pi; with |L| ~ 1e-8 the jump is stiff (1e5 steps). |L| >= l_min is enforced.
  if (fabs(in.L) < l_min) in.L = copysign(l_min, in.L == 0.0 ? 1.0 : in.L);
  Cn<S> cS{S(a), S(a * a), S(in.L), S(in.Q), S((in.L - a) * (in.L - a) + in.Q), S(a * in.L), S(q2), S(lam)};  // projection always in the state type
  Cn<F> c;
  c.a = F(a); c.a2 = F(a * a); c.L = F(in.L); c.Q = F(in.Q);
  c.K = F((in.L - a) * (in.L - a) + in.Q);
  c.aL = F(a * in.L); c.q2 = F(q2); c.lam = F(lam);
  out = Trace{2, 0, 0, 0, 0, 0, 0, in.L, in.Q};
  int skipped = 0;
  if constexpr (Vol::kActive) vol->begin(in.L, in.Q);
  const double disc = 1.0 - a * a - q2;  // < 0: naked singularity (no horizon): rays reaching r_min = 0.05 are given as absorbed (black)
  double r_plus = disc >= 0.0 ? 1.0 + sqrt(disc) : 0.0, r_far_eff = r_far, r_c = r_far;
  if (lam > 0.0) {  // Schwarzschild-de Sitter (a = q2 = 0, 0 < Lambda < 1/9): the black-hole and cosmological horizons (roots of Lambda r^3/3 - r + 2 = 0, trigonometric method);
    const double th = acos(-3.0 * sqrt(lam)) / 3.0, k = 2.0 / sqrt(lam);  // the ray "escapes" on reaching r_c (0.9999 r_c): the (theta, phi) direction at that point is the sky one
    const double x0 = k * cos(th), x1 = k * cos(th - 2.0943951023931953), x2 = k * cos(th - 4.1887902047863905);
    double lo = fmin(x0, fmin(x1, x2)), hi = fmax(x0, fmax(x1, x2)), mid = x0 + x1 + x2 - lo - hi;
    r_plus = mid; r_c = hi; r_far_eff = 0.9999 * hi;
  }
  const S r_cap = S(disc >= 0.0 ? r_plus + 1e-3 : 0.05), rfar = S(r_far_eff);
  const double mu0 = cos(theta0), s0 = sin(theta0);
  S y[6] = {S(r0), S(-delta_d(r0, a, q2, lam) * in.pr), S(mu0), S(s0 * in.ptheta), S(0), S(s0 * s0)};
  S h = S(0.02 / r0);
  const F at = F(atol), rt = F(rtol), atp = F(atol_phi);
  const bool disk_on = disk_rout > 0;
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
      const float sg = vol->sigma(float(y[0])), am = fabsf(float(y[2]));
      const float allow = fmaxf(0.6f * sg, 0.5f * (am - 4.0f * sg)), rate = fabsf(float(y[3]));
      float hcap = sqrtf(2.0f * allow / (fabsf(0.5f * float(dMf(y[2], cS))) + 1e-12f));
      if (rate > 0.0f) hcap = fminf(hcap, allow / rate);
      h = fmin(h, S(hcap));
      h = fmin(h, S(vol->cap(float(y[0]))));  // the model's own step limit (torus: path length per step)
    }
    if (lam > 0.0 && y[1] > S(0)) h = fmin(h, S(0.5) * (S(r_c) - y[0]) / y[1]);
    S yn[6];
    F k1[6], k7[6];
    const F err = dp45_step<S, F>(c, y, h, yn, k1, k7, at, rt, atp);
    if (err > F(1) || !isfinite(err)) {  // v2.7: a NaN/inf error (repulsive core of naked KN, extremes) is NOT accepted as good: it is rejected and, if it persists, the ray ends (absorbed)
      if (!isfinite(err) && ++nan_rej > 40) { out.fate = 1; return; }
      h *= S(step_factor(err)); continue;
    }
    ++out.steps;
    const S yn1 = yn[1], yn3 = yn[3];
    // mu and sigma = 1 - mu^2 are redundant: near the pole (sigma < 1/2) sigma rules (relative precision), otherwise mu
    if (yn[5] < S(0.5)) yn[2] = copysign(sqrt(fmax(S(1) - yn[5], S(0))), yn[2]);
    else yn[5] = S(1) - yn[2] * yn[2];
    // Backport (v4.0 O3): conditional velocity reprojection (avoid ill-conditioned sqrt near turning points)
    {
      const S tolr = S(at) + S(rt) * fabs(yn[0]), tolm = S(at) + S(rt) * fabs(yn[2]);
      if (yn1 * yn1 >= S(200) * fabs(S(k7[1])) * tolr) yn[1] = copysign(sqrt(fmax(Rf(yn[0], cS), S(0))), yn1);
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
      const S rc = h00 * y[0] + h10 * h * k1[0] + h01 * yn[0] + h11 * h * k7[0];
      const S pc = h00 * y[4] + h10 * h * k1[4] + h01 * yn[4] + h11 * h * k7[4];
      if (rc > S(disk_rin) && rc < S(disk_rout) && skipped++ >= skip) {
        out.disk_hit = 1; out.disk_r = double(rc); out.disk_phi = double(pc);
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
      const bool relevant = fminf(fminf(fabsf(float(y[2])), fabsf(float(yn[2]))), fabsf(mmid)) < lim || (y[2] > S(0)) != (yn[2] > S(0));
      if (relevant) {
        const int K = vol->nsub;
        const float dh = float(h) / K;
        for (int q = 0; q < K; ++q) {
          const S s = S((q + 0.5) / K);
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
    if (lam > 0.0 && yn[0] > rfar && yn[1] > S(0) && y[0] <= rfar) {
      auto hm = [&](int i, S s) { const S s2 = s * s, s3 = s2 * s; return (S(2) * s3 - S(3) * s2 + S(1)) * y[i] + (s3 - S(2) * s2 + s) * h * k1[i] + (S(-2) * s3 + S(3) * s2) * yn[i] + (s3 - s2) * h * k7[i]; };
      S lo = 0, hi = 1;
      for (int it = 0; it < (sizeof(S) == 8 ? 60 : 28); ++it) { const S m = S(0.5) * (lo + hi); (hm(0, m) < rfar ? lo : hi) = m; }
      const S s = S(0.5) * (lo + hi);
      out.fate = 0; out.mu_end = fmin(1.0, fmax(-1.0, double(hm(2, s)))); out.phi_end = double(hm(4, s));
      return;
    }
#pragma unroll
    for (int i = 0; i < 6; ++i) y[i] = yn[i];
    if (y[0] < r_cap) { out.fate = 1; return; }
    if (y[0] > rfar && y[1] > S(0)) {
      out.fate = 0;
      out.mu_end = fmin(1.0, fmax(-1.0, double(y[2])));
      out.phi_end = double(y[4]);
      return;
    }
    h *= S(step_factor(err));
  }
  out.fate = 2;
}

}  // namespace bk
