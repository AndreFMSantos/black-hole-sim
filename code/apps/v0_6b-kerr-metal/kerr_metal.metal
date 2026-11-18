// Kerr ray tracer and post-processing in Metal Shading Language (Apple Silicon).
// Ported from apps/v0_6b-kerr-cuda/kerr_cuda.cuh and render.cu, with support for:
//   kPrec == 1: native float (FP32)
//   kPrec == 2: mixed (state in double-float df64, derivatives in float)
//   kPrec == 0: double emulated via double-float (df64, ~48 bits of mantissa)
#include <metal_stdlib>
#include "metal_shared.hpp"

using namespace metal;

constant int kPrec [[function_constant(0)]];
constant int kModel [[function_constant(1)]];

// ------------------------------------------------------------------ double-float (df64)
struct df64 {
  float hi;
  float lo;
};

inline df64 df_make(float a) { return df64{a, 0.0f}; }
inline df64 df_make(float2 v) { return df64{v.x, v.y}; }
inline float to_f(float a) { return a; }
inline float to_f(df64 a) { return a.hi + a.lo; }

inline df64 quick_two_sum(float a, float b) {
  float s = a + b;
  float e = b - (s - a);
  return df64{s, e};
}

inline df64 two_sum(float a, float b) {
  float s = a + b;
  float v = s - a;
  float e = (a - (s - v)) + (b - v);
  return df64{s, e};
}

inline df64 two_prod(float a, float b) {
  float p = a * b;
  float e = fma(a, b, -p);
  return df64{p, e};
}

inline df64 df_add(df64 a, df64 b) {
  df64 s = two_sum(a.hi, b.hi);
  df64 t = two_sum(a.lo, b.lo);
  s.lo += t.hi;
  s = quick_two_sum(s.hi, s.lo);
  s.lo += t.lo;
  return quick_two_sum(s.hi, s.lo);
}

inline df64 df_sub(df64 a, df64 b) {
  return df_add(a, df64{-b.hi, -b.lo});
}

inline df64 df_mul(df64 a, df64 b) {
  df64 p = two_prod(a.hi, b.hi);
  p.lo += a.hi * b.lo + a.lo * b.hi;
  return quick_two_sum(p.hi, p.lo);
}

inline df64 df_div(df64 a, df64 b) {
  float q1 = a.hi / b.hi;
  df64 r = df_sub(a, df_mul(df_make(q1), b));
  float q2 = r.hi / b.hi;
  r = df_sub(r, df_mul(df_make(q2), b));
  float q3 = r.hi / b.hi;
  df64 out = quick_two_sum(q1, q2);
  return df_add(out, df_make(q3));
}

inline df64 df_neg(df64 a) { return df64{-a.hi, -a.lo}; }
inline df64 df_abs(df64 a) { return a.hi < 0.0f ? df_neg(a) : a; }

inline df64 df_sqrt(df64 a) {
  if (a.hi <= 0.0f) return df64{0.0f, 0.0f};
  float x = 1.0f / sqrt(a.hi);
  float ax = a.hi * x;
  df64 ax2 = two_prod(ax, ax);
  float d = df_sub(a, ax2).hi;
  return two_sum(ax, 0.5f * d * x);
}

inline bool df_lt(df64 a, df64 b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }
inline bool df_gt(df64 a, df64 b) { return a.hi > b.hi || (a.hi == b.hi && a.lo > b.lo); }
inline df64 df_fmax(df64 a, df64 b) { return df_lt(a, b) ? b : a; }
inline df64 df_fmin(df64 a, df64 b) { return df_gt(a, b) ? b : a; }
inline df64 df_copysign(df64 a, df64 b) {
  df64 aa = df_abs(a);
  return b.hi < 0.0f ? df_neg(aa) : aa;
}

// Uniform wrappers for <S, F> templates
inline float t_add(float a, float b) { return a + b; }
inline float t_sub(float a, float b) { return a - b; }
inline float t_mul(float a, float b) { return a * b; }
inline float t_div(float a, float b) { return a / b; }
inline float t_neg(float a) { return -a; }
inline float t_abs(float a) { return fabs(a); }
inline float t_sqrt(float a) { return sqrt(a); }
inline float t_fmax(float a, float b) { return fmax(a, b); }
inline float t_fmin(float a, float b) { return fmin(a, b); }
inline float t_copysign(float a, float b) { return copysign(a, b); }
inline bool t_lt(float a, float b) { return a < b; }
inline bool t_gt(float a, float b) { return a > b; }
template <typename T> inline T t_from(float x);
template <> inline float t_from<float>(float x) { return x; }
template <> inline df64 t_from<df64>(float x) { return df_make(x); }
template <typename T> inline T t_from_df(df64 x);
template <> inline float t_from_df<float>(df64 x) { return x.hi + x.lo; }
template <> inline df64 t_from_df<df64>(df64 x) { return x; }

inline df64 t_add(df64 a, df64 b) { return df_add(a, b); }
inline df64 t_sub(df64 a, df64 b) { return df_sub(a, b); }
inline df64 t_mul(df64 a, df64 b) { return df_mul(a, b); }
inline df64 t_div(df64 a, df64 b) { return df_div(a, b); }
inline df64 t_neg(df64 a) { return df_neg(a); }
inline df64 t_abs(df64 a) { return df_abs(a); }
inline df64 t_sqrt(df64 a) { return df_sqrt(a); }
inline df64 t_fmax(df64 a, df64 b) { return df_fmax(a, b); }
inline df64 t_fmin(df64 a, df64 b) { return df_fmin(a, b); }
inline df64 t_copysign(df64 a, df64 b) { return df_copysign(a, b); }
inline bool t_lt(df64 a, df64 b) { return df_lt(a, b); }
inline bool t_gt(df64 a, df64 b) { return df_gt(a, b); }

// ------------------------------------------------------------------ Physical constants
constant float kT0 = 500.0f;
constant float kT1 = 100000.0f;
constant int kLutN = 1024;
constant int kTurbOct = 3;
constant float kTurbVar = 0.0584f;
constant float kSynchE = 4.80320471e-10f;
constant float kSynchMe = 9.1093837015e-28f;
constant float kSynchMp = 1.67262192369e-24f;
constant float kSynchC = 2.99792458e10f;
constant float kSynchKb = 1.380649e-16f;
constant float kSynchH = 6.62607015e-27f;

// ------------------------------------------------------------------ Kerr physics
inline df64 delta_df(df64 r, df64 a) {
  return df_add(df_sub(df_mul(r, r), df_mul(df_make(2.0f), r)), df_mul(a, a));
}

struct InitDF {
  df64 L, Q, pr, ptheta, E_loc;
};

inline InitDF init_from_zamo_df(df64 a, df64 r, df64 s, df64 mu, df64 nr, df64 nth, df64 nph) {
  df64 r2 = df_mul(r, r);
  df64 a2 = df_mul(a, a);
  df64 mu2 = df_mul(mu, mu);
  df64 s2 = df_mul(s, s);
  df64 Sigma = df_add(r2, df_mul(a2, mu2));
  df64 Delta = delta_df(r, a);
  df64 rr = df_add(r2, a2);
  df64 A = df_sub(df_mul(rr, rr), df_mul(df_mul(a2, Delta), s2));
  df64 alpha = df_sqrt(df_div(df_mul(Sigma, Delta), A));
  df64 omega = df_div(df_mul(df_make(2.0f), df_mul(a, r)), A);
  df64 varpi = df_mul(df_sqrt(df_div(A, Sigma)), s);
  InitDF o;
  o.E_loc = df_div(df_make(1.0f), df_add(alpha, df_mul(omega, df_mul(varpi, nph))));
  o.L = df_mul(varpi, df_mul(o.E_loc, nph));
  o.pr = df_mul(o.E_loc, df_mul(nr, df_sqrt(df_div(Sigma, Delta))));
  o.ptheta = df_mul(o.E_loc, df_mul(nth, df_sqrt(Sigma)));
  df64 L2 = df_mul(o.L, o.L);
  df64 sig_safe = df_fmax(df_make(1e-14f), s2);
  o.Q = df_add(df_mul(o.ptheta, o.ptheta), df_mul(mu2, df_sub(df_div(L2, sig_safe), a2)));
  return o;
}

template <typename T>
struct Cn {
  T a, a2, L, Q, K, aL;
};

template <typename T>
inline T Rf(T r, thread const Cn<T>& c) {
  T r2 = t_mul(r, r);
  T A = t_sub(t_add(r2, c.a2), c.aL);
  T del = t_add(t_sub(r2, t_mul(t_from<T>(2.0f), r)), c.a2);
  return t_sub(t_mul(A, A), t_mul(del, c.K));
}

template <typename T>
inline T dRf(T r, thread const Cn<T>& c) {
  T r2 = t_mul(r, r);
  T term1 = t_mul(t_mul(t_from<T>(4.0f), r), t_sub(t_add(r2, c.a2), c.aL));
  T term2 = t_mul(t_sub(t_mul(t_from<T>(2.0f), r), t_from<T>(2.0f)), c.K);
  return t_sub(term1, term2);
}

template <typename T>
inline T Mf(T mu, T sig, thread const Cn<T>& c) {
  T mu2 = t_mul(mu, mu);
  T L2 = t_mul(c.L, c.L);
  return t_sub(t_mul(sig, t_add(c.Q, t_mul(c.a2, mu2))), t_mul(L2, mu2));
}

template <typename T>
inline T dMf(T mu, thread const Cn<T>& c) {
  T L2 = t_mul(c.L, c.L);
  T c1 = t_mul(t_from<T>(2.0f), t_sub(t_sub(c.a2, c.Q), L2));
  T mu3 = t_mul(t_mul(mu, mu), mu);
  return t_sub(t_mul(c1, mu), t_mul(t_mul(t_from<T>(4.0f), c.a2), mu3));
}

template <typename T>
inline T dphif(T r, T om, thread const Cn<T>& c) {
  om = t_fmax(t_from<T>(1e-14f), om);
  T r2 = t_mul(r, r);
  T num = t_sub(t_add(r2, c.a2), c.aL);
  T den = t_add(t_sub(r2, t_mul(t_from<T>(2.0f), r)), c.a2);
  return t_add(t_sub(t_div(c.L, om), c.a), t_div(t_mul(c.a, num), den));
}

template <typename T>
inline void deriv(thread const T* s, thread T* d, thread const Cn<T>& c) {
  d[0] = s[1];
  d[1] = t_mul(t_from<T>(0.5f), dRf(s[0], c));
  d[2] = s[3];
  d[3] = t_mul(t_from<T>(0.5f), dMf(s[2], c));
  d[4] = t_neg(dphif(s[0], s[5], c));
  d[5] = t_mul(t_from<T>(-2.0f), t_mul(s[2], s[3]));
}

template <typename S, typename F>
inline F dp45_step(thread const Cn<F>& c, thread const S* y, S h, thread S* yn,
                   thread F* k1, thread F* k7, F atol, F rtol, F atol_phi) {
  F k2[6], k3[6], k4[6], k5[6], k6[6], tf[6];
  S t[6];
  const F hf = t_from<F>(to_f(h));
  for (int i = 0; i < 6; ++i) tf[i] = t_from<F>(to_f(y[i]));
  deriv(tf, k1, c);

  auto stage = [&](int i, F sum) {
    S inc = t_from<S>(to_f(t_mul(hf, sum)));
    t[i] = t_add(y[i], inc);
    tf[i] = t_from<F>(to_f(t[i]));
  };

  for (int i = 0; i < 6; ++i) stage(i, t_mul(t_from<F>(1.0f / 5.0f), k1[i]));
  deriv(tf, k2, c);

  for (int i = 0; i < 6; ++i) stage(i, t_add(t_mul(t_from<F>(3.0f / 40.0f), k1[i]), t_mul(t_from<F>(9.0f / 40.0f), k2[i])));
  deriv(tf, k3, c);

  for (int i = 0; i < 6; ++i) {
    F s = t_add(t_mul(t_from<F>(44.0f / 45.0f), k1[i]), t_mul(t_from<F>(-56.0f / 15.0f), k2[i]));
    stage(i, t_add(s, t_mul(t_from<F>(32.0f / 9.0f), k3[i])));
  }
  deriv(tf, k4, c);

  for (int i = 0; i < 6; ++i) {
    F s = t_add(t_mul(t_from<F>(19372.0f / 6561.0f), k1[i]), t_mul(t_from<F>(-25360.0f / 2187.0f), k2[i]));
    s = t_add(s, t_mul(t_from<F>(64448.0f / 6561.0f), k3[i]));
    stage(i, t_add(s, t_mul(t_from<F>(-212.0f / 729.0f), k4[i])));
  }
  deriv(tf, k5, c);

  for (int i = 0; i < 6; ++i) {
    F s = t_add(t_mul(t_from<F>(9017.0f / 3168.0f), k1[i]), t_mul(t_from<F>(-355.0f / 33.0f), k2[i]));
    s = t_add(s, t_mul(t_from<F>(46732.0f / 5247.0f), k3[i]));
    s = t_add(s, t_mul(t_from<F>(49.0f / 176.0f), k4[i]));
    stage(i, t_add(s, t_mul(t_from<F>(-5103.0f / 18656.0f), k5[i])));
  }
  deriv(tf, k6, c);

  for (int i = 0; i < 6; ++i) {
    F s = t_add(t_mul(t_from<F>(35.0f / 384.0f), k1[i]), t_mul(t_from<F>(500.0f / 1113.0f), k3[i]));
    s = t_add(s, t_mul(t_from<F>(125.0f / 192.0f), k4[i]));
    s = t_add(s, t_mul(t_from<F>(-2187.0f / 6784.0f), k5[i]));
    s = t_add(s, t_mul(t_from<F>(11.0f / 84.0f), k6[i]));
    yn[i] = t_add(y[i], t_from<S>(to_f(t_mul(hf, s))));
    tf[i] = t_from<F>(to_f(yn[i]));
  }
  deriv(tf, k7, c);

  float err = 0.0f;
  for (int i = 0; i < 6; ++i) {
    F e = t_mul(t_from<F>(71.0f / 57600.0f), k1[i]);
    e = t_add(e, t_mul(t_from<F>(-71.0f / 16695.0f), k3[i]));
    e = t_add(e, t_mul(t_from<F>(71.0f / 1920.0f), k4[i]));
    e = t_add(e, t_mul(t_from<F>(-17253.0f / 339200.0f), k5[i]));
    e = t_add(e, t_mul(t_from<F>(22.0f / 525.0f), k6[i]));
    e = t_add(e, t_mul(t_from<F>(-1.0f / 40.0f), k7[i]));
    float ef = fabs(to_f(t_mul(hf, e)));
    float ym = fmax(fabs(to_f(y[i])), fabs(to_f(yn[i])));
    float w = (i == 5) ? (1e-30f + 0.05f * ym) : ((i == 4 ? to_f(atol_phi) : to_f(atol)) + to_f(rtol) * ym);
    err = fmax(err, ef / w);
  }
  return t_from<F>(err);
}

inline float step_factor_f(float err) {
  if (err <= 0.0f) return 5.0f;
  return fmin(5.0f, fmax(0.2f, 0.9f * pow(err, -0.2f)));
}

// ------------------------------------------------------------------ Common helpers (LUT, noise, Lorentz, camera)
inline float3 lut_lookup(device const float* d, float T) {
  if (T <= kT0) return float3(0.0f);
  float x = log(fmin(T, kT1) / kT0) / log(kT1 / kT0) * float(kLutN - 1);
  int i = min(kLutN - 2, int(x));
  float f = x - float(i);
  return float3((1.0f - f) * d[3 * i] + f * d[3 * i + 3],
                (1.0f - f) * d[3 * i + 1] + f * d[3 * i + 4],
                (1.0f - f) * d[3 * i + 2] + f * d[3 * i + 5]);
}

inline uint hash3(uint a, uint b, uint c) {
  uint h = a * 0x9E3779B1u ^ b * 0x85EBCA6Bu ^ c * 0xC2B2AE35u;
  h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
  return h;
}

inline float vnoise_f(float x, float y, float z) {
  float fxf = floor(x), fyf = floor(y), fzf = floor(z);
  int xi = int(fxf), yi = int(fyf), zi = int(fzf);
  float fx = x - fxf, fy = y - fyf, fz = z - fzf;
  float sx = fx * fx * (3.0f - 2.0f * fx);
  float sy = fy * fy * (3.0f - 2.0f * fy);
  float sz = fz * fz * (3.0f - 2.0f * fz);
  float r = 0.0f;
  for (int dz = 0; dz < 2; ++dz)
    for (int dy = 0; dy < 2; ++dy)
      for (int dx = 0; dx < 2; ++dx) {
        float hv = float(hash3(uint(xi + dx), uint(yi + dy), uint(zi + dz)) & 0xFFFFFFu) * (1.0f / 16777216.0f);
        r += hv * (dx ? sx : 1.0f - sx) * (dy ? sy : 1.0f - sy) * (dz ? sz : 1.0f - sz);
      }
  return r;
}

inline float fbm_f(float x, float y, float z) {
  float s = 0.0f, wsum = 0.0f, w = 1.0f, f = 1.0f;
  for (int k = 0; k < kTurbOct; ++k) {
    s += w * (2.0f * vnoise_f(f * x + 17.31f * k, f * y + 5.77f * k, f * z + 11.13f * k) - 1.0f);
    wsum += w; w *= 0.5f; f *= 2.0f;
  }
  return s / wsum;
}

inline float turb_density_f(constant MetalKP& p, float r, float phi, float mu, float h, float omega, float t) {
  if (p.turb_amp <= 0.0f) return 1.0f;
  float ang = phi - omega * t;
  float sn = sin(ang), cs = cos(ang);
  float inv = 1.0f / (p.turb_scale * h);
  float n = fbm_f(cs * inv, sn * inv, (log(r) + 1.7f * mu) * inv);
  float a = p.turb_amp;
  return exp(a * n - 0.5f * a * a * kTurbVar);
}

inline float spot_factor_f(constant MetalKP& p, float r, float phi, float mu, float t) {
  if (p.spot_amp <= 0.0f) return 1.0f;
  float rh = p.spot_r, z = r * mu;
  float ph = p.spot_ph + p.spot_om * (t - p.disk_time);
  float amp = p.spot_amp;
  if (p.fl_boost != 0.0f) {
    float d = (t - p.fl_t) / p.fl_w;
    amp *= 1.0f + p.fl_boost * exp(-0.5f * d * d);
  }
  float d2 = r * r + rh * rh - 2.0f * r * rh * cos(phi - ph) + z * z;
  float s = p.spot_size;
  return pow(1.0f + amp * exp(-0.5f * d2 / (s * s)), 0.25f);
}

inline float redshift_f(float a, float r, float L, float r_obs, float th_obs) {
  float s = sin(th_obs), mu = cos(th_obs);
  float Sigma = r_obs * r_obs + a * a * mu * mu;
  float Delta = r_obs * r_obs - 2.0f * r_obs + a * a;
  float rr = r_obs * r_obs + a * a;
  float A = rr * rr - a * a * Delta * s * s;
  float alpha = sqrt(Sigma * Delta / A), omega = 2.0f * a * r_obs / A;
  float E_obs = (1.0f - omega * L) / alpha;
  float rs = sqrt(r), r32 = r * rs, r34 = rs * sqrt(rs);
  float ut = (r32 + a) / (r34 * sqrt(r32 - 3.0f * rs + 2.0f * a));
  float Omega = 1.0f / (r32 + a);
  return E_obs / (ut * (1.0f - Omega * L));
}

inline float disk_temp_f(float r, float rin, float fnorm) {
  if (r <= rin) return 0.0f;
  float f = pow(r, -3.0f) * (1.0f - sqrt(rin / r));
  return pow(f / fnorm, 0.25f);
}

inline float nt_temp_f(constant MetalKP& p, device const float* nt_tab, float r) {
  if (r <= p.nt_rin) return 0.0f;
  if (r >= p.nt_rout) return nt_tab[p.nt_n - 1] * pow(r / p.nt_rout, -0.75f);
  float lmin = log(p.nt_rin), lmax = log(p.nt_rout);
  float x = (log(r) - lmin) / (lmax - lmin) * float(p.nt_n - 1);
  int i = min(p.nt_n - 2, int(x));
  float f = x - float(i);
  return (1.0f - f) * nt_tab[i] + f * nt_tab[i + 1];
}

inline float emitter_mu_f(float a, float r, float L, float Q) {
  float rs = sqrt(r), r32 = r * rs, r34 = rs * sqrt(rs);
  float ut = (r32 + a) / (r34 * sqrt(r32 - 3.0f * rs + 2.0f * a));
  float Omega = 1.0f / (r32 + a);
  float e_emit = ut * (1.0f - Omega * L);
  return fmin(1.0f, sqrt(fmax(Q, 0.0f)) / (r * fabs(e_emit)));
}

inline float ss_f(float a, float b, float x) {
  float t = fmin(1.0f, fmax(0.0f, (x - a) / (b - a)));
  return t * t * (3.0f - 2.0f * t);
}

inline float limb_eddington_f(float mu) { return (1.0f + 2.06f * mu) / (1.0f + 2.06f * 2.0f / 3.0f); }

inline void observer_to_zamo_f(float3 np, float3 beta, thread float3& n, thread float& dop) {
  float b2 = dot(beta, beta);
  if (b2 < 1e-30f) { n = np; dop = 1.0f; return; }
  float gamma = 1.0f / sqrt(1.0f - b2);
  float bn = dot(beta, np);
  float E = gamma * (1.0f + bn);
  float k = ((gamma - 1.0f) * bn / b2) + gamma;
  n = normalize(np + k * beta);
  dop = 1.0f / E;
}

inline float3 camera_ray_f(constant MetalKP& p, float x, float y) {
  float inv = rsqrt(1.0f + x * x + y * y);
  float3 fwd = float3(p.fwd[0], p.fwd[1], p.fwd[2]);
  float3 right = float3(p.right[0], p.right[1], p.right[2]);
  float3 up = float3(p.up[0], p.up[1], p.up[2]);
  return -(fwd + x * right + y * up) * inv;
}

// ------------------------------------------------------------------ Synchrotron, Torus and GRMHD
inline float k2_f(float z) {
  float K0, K1;
  if (z <= 2.0f) {
    float lz = log(0.5f * z), y = (z / 3.75f) * (z / 3.75f), w = z * z * 0.25f;
    float i0 = 1.0f + y * (3.5156229f + y * (3.0899424f + y * (1.2067492f + y * (0.2659732f + y * (0.0360768f + y * 0.0045813f)))));
    float i1 = z * (0.5f + y * (0.87890594f + y * (0.51498869f + y * (0.15084934f + y * (0.02658733f + y * (0.00301532f + y * 0.00032411f))))));
    K0 = -lz * i0 + (-0.57721566f + w * (0.42278420f + w * (0.23069756f + w * (0.03488590f + w * (0.00262698f + w * (0.00010750f + w * 0.0000074f))))));
    K1 = lz * i1 + (1.0f / z) * (1.0f + w * (0.15443144f + w * (-0.67278579f + w * (-0.18156897f + w * (-0.01919402f + w * (-0.00110404f + w * -0.00004686f))))));
  } else {
    float u = 2.0f / z, e = exp(-z) / sqrt(z);
    K0 = e * (1.25331414f + u * (-0.07832358f + u * (0.02189568f + u * (-0.01062446f + u * (0.00587872f + u * (-0.00251540f + u * 0.00053208f))))));
    K1 = e * (1.25331414f + u * (0.23498619f + u * (-0.03655620f + u * (0.01504268f + u * (-0.00780353f + u * (0.00325614f + u * -0.00068245f))))));
  }
  return K0 + 2.0f * K1 / z;
}

inline float synch_j_f(int emis, float ne, float thE, float nuB, float sinB, float nu_em) {
  if (emis == 1) {
    float nu_s = (2.0f / 9.0f) * nuB * sinB * thE * thE;
    float X = nu_em / nu_s;
    float c = pow(X, 1.0f / 3.0f);
    if (!(c < 80.0f)) return 0.0f;
    float sX = sqrt(X);
    float t = sX + 1.8877486253633869f * pow(sX, 1.0f / 3.0f);
    return ne * (kSynchE * kSynchE * 1.41421356237f * M_PI_F / (3.0f * kSynchC)) * nu_s / k2_f(1.0f / thE) * t * t * exp(-c);
  }
  float x = nu_em / (1.5f * nuB * sinB * thE * thE);
  float cx = pow(x, 1.0f / 3.0f);
  float II = 2.5651f * (1.0f + 1.92f / cx + 0.9977f / (cx * cx)) * exp(-1.8899f * cx);
  return ne * (kSynchE * kSynchE / (2.0f * 1.73205080757f * kSynchC)) * nu_em / (thE * thE) * II;
}

inline float expm1_f(float x) {
  return fabs(x) < 1e-4f ? (x + 0.5f * x * x) : (exp(x) - 1.0f);
}

struct TorusPt { float rho, thE, B, sinB, j, alpha, g, e_emit; };

inline bool torus_emit_f(constant MetalKP& p, float r, float mu, float L, float eobs_dop, thread TorusPt& o) {
  float a = p.a, s2 = 1.0f - mu * mu;
  if (s2 < 1e-8f || r < p.t_rin * 0.7f) return false;
  float Sig = r * r + a * a * mu * mu;
  float gtt = -(1.0f - 2.0f * r / Sig), gtp = -2.0f * a * r * s2 / Sig, gpp = (r * r + a * a + 2.0f * a * a * r * s2 / Sig) * s2;
  float det = gtt * gpp - gtp * gtp, itt = gpp / det, itp = -gtp / det, ipp = gtt / det;
  float ell = p.t_ell;
  float D = itt - 2.0f * ell * itp + ell * ell * ipp;
  if (D >= 0.0f) return false;
  float ut_abs = 1.0f / sqrt(-D), h = p.t_utin / ut_abs;
  if (h <= 1.0f) return false;
  float rho = pow((h - 1.0f) / (p.t_hmax - 1.0f), 1.0f / (p.t_gamma - 1.0f));
  if (rho < 1e-8f) return false;
  o.rho = rho;
  float thI = ((p.t_gamma - 1.0f) / p.t_gamma) * (h - 1.0f);
  float thE = p.th_e_max > 0.0f ? p.th_e_max * (h - 1.0f) / (p.t_hmax - 1.0f) : thI * (kSynchMp / kSynchMe / p.ti_te);
  o.thE = thE;
  if (thE < p.the_min) return false;
  float ne = p.ne0 * rho;
  float Pn = kSynchMp * thI + kSynchMe * thE;
  float B = sqrt((8.0f * M_PI_F * kSynchC * kSynchC / p.pbeta) * ne * Pn);
  o.B = B;
  float ut_low = -ut_abs, ut_up = ut_low * (itt - ell * itp), uphi_up = ut_low * (itp - ell * ipp), Om = uphi_up / ut_up;
  float e_emit = fabs(ut_up * (1.0f - Om * L));
  o.e_emit = e_emit;
  float g = eobs_dop / e_emit;
  o.g = g;
  float Delta = r * r - 2.0f * r + a * a, A = (r * r + a * a) * (r * r + a * a) - a * a * Delta * s2;
  float alz = sqrt(Sig * Delta / A), omz = 2.0f * a * r / A, varpi = sqrt(A / Sig * s2);
  float nphi = (L / varpi) * alz / (1.0f - omz * L), v = varpi * (Om - omz) / alz;
  float cB = (nphi - v) / (1.0f - v * nphi);
  float sinB = sqrt(fmax(0.0f, 1.0f - cB * cB));
  o.sinB = sinB;
  if (sinB <= 0.0f) { o.j = 0.0f; o.alpha = 0.0f; return true; }
  float nu_em = p.nu_obs / g;
  float nuB = (kSynchE / (2.0f * M_PI_F * kSynchMe * kSynchC)) * B;
  float jF = synch_j_f(p.emis, ne, thE, nuB, sinB, nu_em);
  o.j = jF;
  float xp = (kSynchH / (kSynchMe * kSynchC * kSynchC)) * nu_em / thE;
  float nus = nu_em * 1e-10f;
  float Bn = (2.0f * kSynchH * 1e30f / (kSynchC * kSynchC)) * nus * nus * nus / expm1_f(xp);
  o.alpha = xp > 80.0f ? 0.0f : (jF / Bn);
  return true;
}

inline float3 radio_rgb_f(constant MetalKP& p, float I) {
  float Tb = I * kSynchC * kSynchC / (2.0f * kSynchKb * p.nu_obs * p.nu_obs);
  float x = fmin(1.0f, fmax(0.0f, (log10(fmax(Tb, 1.0f)) - p.tb_lo) / (p.tb_hi - p.tb_lo)));
  float r = fmin(1.0f, 2.0f * x), g = fmin(1.0f, fmax(0.0f, 2.0f * x - 0.5f)), b = fmin(1.0f, fmax(0.0f, 2.0f * x - 1.0f));
  return float3(pow(r, 2.2f), pow(g, 2.2f), pow(b, 2.2f));
}

// FMKS and GRMHD sampling
inline float f_theta(constant MetalKP& p, float X1, float X2, thread float* dth2) {
  float hs = p.g_hslope, pa = p.g_palpha, xt = p.g_pxt, pn = p.g_pnorm;
  float tg = 3.14159265f * X2 + 0.5f * (1.0f - hs) * sin(6.2831853f * X2), y = 2.0f * X2 - 1.0f;
  float yr = pow(fabs(y) / xt, pa);
  float tj = pn * y * (1.0f + yr / (pa + 1.0f)) + 1.57079633f, w = exp(p.g_ms * (p.g_startx1 - X1));
  if (dth2) {
    float dg = 3.14159265f + (1.0f - hs) * 3.14159265f * cos(6.2831853f * X2), dj = 2.0f * pn * (1.0f + yr);
    *dth2 = dg + w * (dj - dg);
  }
  return tg + w * (tj - tg);
}

inline bool f_inv_theta_exact(constant MetalKP& p, float X1, float th, thread float& X2) {
  float lo = 0.0f, hi = 1.0f, x = th * 0.31830989f;
  for (int it = 0; it < 40; ++it) {
    float d;
    float f = f_theta(p, X1, x, &d) - th;
    if (fabs(f) < 2e-7f || hi - lo < 1e-7f) break;
    if (f > 0.0f) hi = x; else lo = x;
    float xn = x - f / d;
    if (!(xn > lo && xn < hi)) xn = 0.5f * (lo + hi);
    x = xn;
  }
  X2 = x;
  return true;
}

inline bool f_inv_theta(constant MetalKP& p, float X1, float th, thread float& X2, texture2d<float, access::sample> g_theta_lut) {
  if (!p.has_theta_lut) return f_inv_theta_exact(p, X1, th, X2);
  constexpr sampler lut_smp(coord::normalized, address::clamp_to_edge, filter::linear);
  float s1 = (X1 - p.g_startx1) / (float(p.g_n1) * p.g_dx1);
  float x = g_theta_lut.sample(lut_smp, float2(s1, th * 0.31830989f)).x;
  for (int it = 0; it < 2; ++it) {
    float d;
    float f = f_theta(p, X1, x, &d) - th;
    x = fmin(1.0f, fmax(0.0f, x - f / fmax(d, 1e-6f)));
  }
  X2 = x;
  return true;
}

struct GrmPt { float rho, thI, absb, beta; float u[4], b[4]; };

inline bool grmhd_sample(constant MetalKP& p, float r, float mu, float phi_bl, thread GrmPt& o,
                         texture3d<float, access::sample> g_tex0, texture3d<float, access::sample> g_tex1,
                         texture3d<float, access::sample> g_tex2, texture2d<float, access::sample> g_theta_lut) {
  float X1 = log(r), s1 = (X1 - p.g_startx1) / (float(p.g_n1) * p.g_dx1);
  if (s1 <= 0.0f || s1 >= 1.0f) return false;
  float X2;
  f_inv_theta(p, X1, acos(fmin(1.0f, fmax(-1.0f, mu))), X2, g_theta_lut);
  float a = p.a, phi = phi_bl;
  if (a != 0.0f) {
    float rp = 1.0f + sqrt(fmax(0.0f, 1.0f - a * a)), rm = 1.0f - sqrt(fmax(0.0f, 1.0f - a * a));
    if (r > rp) phi += a / (rp - rm) * log((r - rp) / (r - rm));
  }
  float s3 = phi * 0.15915494f; s3 -= floor(s3);
  constexpr sampler grm_smp(coord::normalized, s_address::repeat, t_address::clamp_to_edge, r_address::clamp_to_edge, filter::linear);
  float4 t0 = g_tex0.sample(grm_smp, float3(s3, X2, s1));
  float4 t1 = g_tex1.sample(grm_smp, float3(s3, X2, s1));
  float4 t2 = g_tex2.sample(grm_smp, float3(s3, X2, s1));
  o.rho = t0.x; o.thI = t0.y; o.absb = t0.z; o.beta = t0.w;
  float th = acos(fmin(1.0f, fmax(-1.0f, mu))), s2 = sin(th) * sin(th), rho2 = r * r + a * a * mu * mu, z = 2.0f * r / rho2;
  float gtt = -1.0f + z, gtr = z, gtp = -a * s2 * z, grr = 1.0f + z, grp = -a * s2 * (1.0f + z), gthth = rho2, gpp = s2 * (rho2 + a * a * s2 * (1.0f + z));
  float ur = t1.y, uth = t1.z, uph = t1.w, ut = t1.x;
  float C = grr * ur * ur + gthth * uth * uth + gpp * uph * uph + 2.0f * grp * ur * uph + 1.0f, Bc = 2.0f * (gtr * ur + gtp * uph);
  for (int it = 0; it < 3; ++it) { float f = gtt * ut * ut + Bc * ut + C, df = 2.0f * gtt * ut + Bc; if (fabs(df) > 1e-12f) ut -= f / df; }
  o.u[0] = ut; o.u[1] = ur; o.u[2] = uth; o.u[3] = uph;
  float bt = t2.x, br = t2.y, bth = t2.z, bp = t2.w;
  float ub = gtt * ut * bt + gtr * (ut * br + ur * bt) + gtp * (ut * bp + uph * bt) + grr * ur * br + gthth * uth * bth + grp * (ur * bp + uph * br) + gpp * uph * bp;
  bt += ub * ut; br += ub * ur; bth += ub * uth; bp += ub * uph;
  float nb = gtt * bt * bt + 2.0f * gtr * bt * br + 2.0f * gtp * bt * bp + grr * br * br + gthth * bth * bth + 2.0f * grp * br * bp + gpp * bp * bp;
  float inv = nb > 1e-30f ? rsqrt(nb) : 0.0f;
  o.b[0] = bt * inv; o.b[1] = br * inv; o.b[2] = bth * inv; o.b[3] = bp * inv;
  return true;
}

// ------------------------------------------------------------------ Volumetric accumulator
struct NoVol {
  static constant bool kActive = false;
  int nsub = 1;
  bool slow = false;
  float T = 0.0f;
  void begin(float, float) {}
  float sigma(float) const { return 1.0f; }
  float cap(float) const { return 1e30f; }
  bool opaque() const { return false; }
  void add(float, float, float, float, float, float) {}
};

struct VolAcc {
  static constant bool kActive = true;
  constant MetalKP& p;
  device const float* lut;
  device const float* nt_tab;
  texture3d<float, access::sample> g_tex0, g_tex1, g_tex2;
  texture2d<float, access::sample> g_theta_lut;
  float dop, eobs = 1.0f, L = 0.0f, trans = 1.0f;
  float T = 0.0f, tref = 0.0f, T_half = -1.0f;
  bool slow = false;
  float3 I = float3(0.0f);
  float Irad = 0.0f;
  float Qc = 0.0f, null_err = 0.0f;
  int nsub;

  inline int mdl() const { return kModel >= 0 ? kModel : p.model; }

  VolAcc(constant MetalKP& pp, device const float* l, device const float* nt,
         texture3d<float, access::sample> gt0, texture3d<float, access::sample> gt1,
         texture3d<float, access::sample> gt2, texture2d<float, access::sample> gtl, float d)
      : p(pp), lut(l), nt_tab(nt), g_tex0(gt0), g_tex1(gt1), g_tex2(gt2), g_theta_lut(gtl),
        dop(d), slow(pp.slow != 0), nsub(pp.v_sub) {}

  void begin(float L_, float Q_) {
    float Lg = p.nodoppler ? 0.0f : L_;
    L = Lg;
    float s = p.theta_sin_df.x, mu = p.theta_cos_df.x;
    float Sigma = p.dist * p.dist + p.a * p.a * mu * mu, Delta = p.dist * p.dist - 2.0f * p.dist + p.a * p.a;
    float rr = p.dist * p.dist + p.a * p.a, A = rr * rr - p.a * p.a * Delta * s * s;
    eobs = (1.0f - 2.0f * p.a * p.dist / A * Lg) / sqrt(Sigma * Delta / A);
    Qc = Q_;
    tref = (p.dist + 2.0f * log(p.dist / 2.0f - 1.0f)) - (10.0f + 2.0f * log(4.0f));
  }

  float sigma(float r) const {
    return mdl() >= 1 ? 1.0f : (p.v_flare == 0.0f ? p.v_hr : p.v_hr * pow(r * 0.1f, p.v_flare));
  }

  float cap(float r) const {
    if (mdl() == 2) return r < p.g_rcap ? fmin(p.g_step * r, 2.0f) / (r * r + p.a * p.a) : 1e30f;
    return (mdl() == 1 && r < 4.0f * p.t_rmax) ? p.t_step / (r * r + p.a * p.a) : 1e30f;
  }

  bool opaque() const { return trans < 1e-3f; }

  void add_torus(float rf, float muf, float dM) {
    TorusPt o;
    if (!torus_emit_f(p, rf, muf, L, eobs * dop, o) || o.j <= 0.0f) return;
    float Sig = rf * rf + p.a * p.a * muf * muf;
    float ds = o.e_emit * Sig * dM * p.mcm;
    float dtau = o.alpha * ds, g = o.g;
    float frac = dtau > 1e-6f ? -expm1_f(-dtau) / dtau : 1.0f;
    Irad += trans * g * g * g * o.j * ds * frac;
    trans *= exp(-dtau);
    if (T_half < 0.0f && trans < 0.5f) T_half = T;
  }

  void add_grmhd(float rf, float muf, float phif, float dM, float y1, float y3) {
    GrmPt o;
    if (!grmhd_sample(p, rf, muf, phif, o, g_tex0, g_tex1, g_tex2, g_theta_lut)) return;
    if (!(o.rho > 0.0f)) return;
    float r = rf, mu = muf, a = p.a, Lp = L;
    float s = sqrt(fmax(1e-12f, 1.0f - mu * mu)), Delta = r * r - 2.0f * r + a * a;
    float Rr = (r * r + a * a - a * Lp) * (r * r + a * a - a * Lp) - Delta * ((Lp - a) * (Lp - a) + Qc);
    float Mm = (1.0f - mu * mu) * (Qc + a * a * mu * mu) - Lp * Lp * mu * mu;
    float prBL = (y1 > 0.0f ? -1.0f : 1.0f) * sqrt(fmax(Rr, 0.0f)) / Delta;
    float pth = (y3 >= 0.0f ? 1.0f : -1.0f) * sqrt(fmax(Mm, 0.0f) / fmax(1.0f - mu * mu, 1e-12f));
    float pt = -1.0f, pr = prBL + (2.0f * r - a * Lp) / Delta, pp = Lp;
    if (p.rcheck) {
      float S = r * r + a * a * mu * mu, s2 = s * s;
      float gtt_i = -(1.0f + 2.0f * r / S), gtr_i = 2.0f * r / S, grr_i = Delta / S, grp_i = a / S, gthth_i = 1.0f / S, gpp_i = 1.0f / (S * s2);
      float nul = gtt_i * pt * pt + 2.0f * gtr_i * pt * pr + grr_i * pr * pr + 2.0f * grp_i * pr * pp + gthth_i * pth * pth + gpp_i * pp * pp;
      null_err = fmax(null_err, fabs(nul));
    }
    float E = -(pt * o.u[0] + pr * o.u[1] + pth * o.u[2] + pp * o.u[3]);
    if (!(E > 1e-6f)) return;
    float pb = pt * o.b[0] + pr * o.b[1] + pth * o.b[2] + pp * o.b[3];
    float cB = fmin(1.0f, fmax(-1.0f, pb / E)), sinB = sqrt(fmax(0.0f, 1.0f - cB * cB));
    float g = eobs * dop / E, thI = o.thI;
    float thE;
    if (p.el_model == 2) thE = p.g_ce * thI;
    else if (p.el_model == 1) {
      float b2 = o.beta * o.beta;
      float tite = (p.g_rhigh * b2 + 1.0f) / (1.0f + b2);
      thE = thI * (kSynchMp / kSynchMe) / (1.0f + tite);
    } else thE = thI * (kSynchMp / kSynchMe) / (1.0f + p.ti_te);
    if (thE < p.the_min || sinB <= 0.0f) return;
    float ne = o.rho * (p.g_rho_unit / (kSynchMp + kSynchMe)), Bg = o.absb * p.g_b_unit;
    float nuB = (kSynchE / (2.0f * M_PI_F * kSynchMe * kSynchC)) * Bg, nu_em = p.nu_obs / g;
    float j = synch_j_f(p.emis, ne, thE, nuB, sinB, nu_em);
    if (!(j > 0.0f)) return;
    float xp = (kSynchH / (kSynchMe * kSynchC * kSynchC)) * nu_em / thE;
    float nus = nu_em * 1e-10f;
    float Bn = (2.0f * kSynchH * 1e30f / (kSynchC * kSynchC)) * nus * nus * nus / expm1_f(xp);
    float alpha = xp > 80.0f ? 0.0f : (j / Bn);
    float Sig = r * r + a * a * mu * mu;
    float ds = E * Sig * dM * p.mcm;
    float dtau = alpha * ds, frac = dtau > 1e-6f ? -expm1_f(-dtau) / dtau : 1.0f;
    Irad += trans * g * g * g * j * ds * frac;
    trans *= exp(-dtau);
  }

  void add(float r, float mu, float phi, float dM, float y1 = 0.0f, float y3 = 0.0f) {
    if (mdl() == 2) { add_grmhd(r, mu, phi, dM, y1, y3); return; }
    if (mdl() == 1) { add_torus(r, mu, dM); return; }
    if (r <= p.rin) return;
    float rout = p.rout, es = p.edge_soft;
    float w = 1.0f;
    if (es > 0.0f) {
      float t = fmin(1.0f, fmax(0.0f, (r - rout * (1.0f - es)) / (rout * 2.0f * es)));
      w = 1.0f - t * t * (3.0f - 2.0f * t);
    } else if (r >= rout) return;
    if (w <= 0.0f) return;
    float H = sigma(r) * r, z = r * mu;
    float alpha = w * p.v_tau / (2.50662827f * H) * exp(-0.5f * z * z / (H * H));
    if (alpha < 1e-9f) return;
    float rs = sqrt(r), r32 = r * rs, r34 = rs * sqrt(rs), af = p.a;
    float phd = phi + p.phi;
    float tem = slow ? (p.disk_time - (T - tref)) : p.disk_time;
    float tdf = turb_density_f(p, r, phd, mu, sigma(r), 1.0f / (r32 + af), tem);
    alpha *= tdf;
    float ut = (r32 + af) / (r34 * sqrt(r32 - 3.0f * rs + 2.0f * af));
    float e_emit = fabs(ut * (1.0f - L / (r32 + af)));
    float Sig = r * r + af * af * mu * mu;
    float dtau = alpha * e_emit * Sig * dM;
    float g = eobs * dop / e_emit;
    if (p.beaming != 1.0f) g = pow(g, p.beaming);
    float Tprof = p.nt ? nt_temp_f(p, nt_tab, r) : disk_temp_f(r, p.rin, p.fnorm);
    float Tobs = g * p.tpeak * Tprof * spot_factor_f(p, r, phd, mu, tem);
    float3 c = lut_lookup(lut, Tobs);
    float wgt = trans * (-expm1_f(-dtau)) * p.exposure;
    I += wgt * c;
    trans *= exp(-dtau);
    if (T_half < 0.0f && trans < 0.5f) T_half = T;
  }
};

// ------------------------------------------------------------------ Generic tracer
template <typename S, typename F, typename Vol>
inline void trace_kerr(constant MetalKP& p, df64 nr_d, df64 nth_d, df64 nph_d,
                       float disk_rin, float disk_rout, float rtol, float atol, float atol_phi,
                       thread MetalTrace& out, int skip, thread Vol* vol) {
  df64 a_d = df_make(p.a_df), r0_d = df_make(p.dist_df);
  df64 s0_d = df_make(p.theta_sin_df), mu0_d = df_make(p.theta_cos_df);
  InitDF in = init_from_zamo_df(a_d, r0_d, s0_d, mu0_d, nr_d, nth_d, nph_d);
  float l_min = p.l_min;
  if (fabs(to_f(in.L)) < l_min) {
    float sgn = (to_f(in.L) == 0.0f) ? 1.0f : to_f(in.L);
    in.L = df_make(copysign(l_min, sgn));
  }
  df64 a2_d = df_mul(a_d, a_d);
  df64 Lma_d = df_sub(in.L, a_d);
  df64 K_d = df_add(df_mul(Lma_d, Lma_d), in.Q);
  df64 aL_d = df_mul(a_d, in.L);

  Cn<S> cS{t_from_df<S>(a_d), t_from_df<S>(a2_d), t_from_df<S>(in.L), t_from_df<S>(in.Q), t_from_df<S>(K_d), t_from_df<S>(aL_d)};
  Cn<F> c{t_from_df<F>(a_d), t_from_df<F>(a2_d), t_from_df<F>(in.L), t_from_df<F>(in.Q), t_from_df<F>(K_d), t_from_df<F>(aL_d)};

  out = MetalTrace{2, 0, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f, to_f(in.L), to_f(in.Q)};
  int skipped = 0;
  if (Vol::kActive) vol->begin(to_f(in.L), to_f(in.Q));

  float a_f = to_f(a_d), r0_f = to_f(r0_d);
  float r_plus = 1.0f + sqrt(fmax(0.0f, 1.0f - a_f * a_f));
  S r_cap = t_from<S>(r_plus + 1e-3f), rfar = t_from<S>(p.r_far);
  df64 del0 = delta_df(r0_d, a_d);
  S y[6] = {
    t_from_df<S>(r0_d),
    t_from_df<S>(df_neg(df_mul(del0, in.pr))),
    t_from_df<S>(mu0_d),
    t_from_df<S>(df_mul(s0_d, in.ptheta)),
    t_from<S>(0.0f),
    t_from_df<S>(df_mul(s0_d, s0_d))
  };
  S h = t_from<S>(0.02f / r0_f);
  F at = t_from<F>(atol), rt = t_from<F>(rtol), atp = t_from<F>(atol_phi);
  bool disk_on = disk_rout > 0.0f;

  for (int step = 0; step < p.max_steps; ++step) {
    if (to_f(y[5]) < 0.2f) {
      float curv = fabs(to_f(t_add(t_mul(t_from<S>(2.0f), Mf(y[2], y[5], cS)), t_mul(y[2], dMf(y[2], cS)))));
      float hlim = 0.5f * sqrt(fmax(to_f(y[5]), 1e-30f) / fmax(curv, 1e-12f));
      if (hlim < to_f(h)) h = t_from<S>(hlim);
    }
    if (Vol::kActive) {
      float sg = vol->sigma(to_f(y[0])), am = fabs(to_f(y[2]));
      float allow = fmax(0.6f * sg, 0.5f * (am - 4.0f * sg)), rate = fabs(to_f(y[3]));
      float hcap = sqrt(2.0f * allow / (fabs(0.5f * to_f(dMf(y[2], cS))) + 1e-12f));
      if (rate > 0.0f) hcap = fmin(hcap, allow / rate);
      hcap = fmin(hcap, vol->cap(to_f(y[0])));
      if (hcap < to_f(h)) h = t_from<S>(hcap);
    }
    S yn[6];
    F k1[6], k7[6];
    F err = dp45_step<S, F>(c, y, h, yn, k1, k7, at, rt, atp);
    float err_f = to_f(err);
    if (err_f > 1.0f) { h = t_mul(h, t_from<S>(step_factor_f(err_f))); continue; }
    ++out.steps;
    S yn1 = yn[1], yn3 = yn[3];
    if (to_f(yn[5]) < 0.5f) {
      yn[2] = t_copysign(t_sqrt(t_fmax(t_sub(t_from<S>(1.0f), yn[5]), t_from<S>(0.0f))), yn[2]);
    } else {
      yn[5] = t_sub(t_from<S>(1.0f), t_mul(yn[2], yn[2]));
    }
    yn[1] = t_copysign(t_sqrt(t_fmax(Rf(yn[0], cS), t_from<S>(0.0f))), yn1);
    yn[3] = t_copysign(t_sqrt(t_fmax(Mf(yn[2], yn[5], cS), t_from<S>(0.0f))), yn3);

    if (!Vol::kActive && disk_on && !out.disk_hit && ((to_f(y[2]) > 0.0f) != (to_f(yn[2]) > 0.0f))) {
      S lo = t_from<S>(0.0f), hi = t_from<S>(1.0f);
      bool pos0 = to_f(y[2]) > 0.0f;
      int iters = (sizeof(S) > 4) ? 50 : 28;
      S y2 = y[2], yn2 = yn[2], k12 = t_from<S>(to_f(k1[2])), k72 = t_from<S>(to_f(k7[2]));
      S hk12 = t_mul(h, k12), hk72 = t_mul(h, k72);
      for (int it = 0; it < iters; ++it) {
        S m = t_mul(t_from<S>(0.5f), t_add(lo, hi));
        S s = m, s2 = t_mul(s, s), s3 = t_mul(s2, s);
        S h00 = t_add(t_sub(t_mul(t_from<S>(2.0f), s3), t_mul(t_from<S>(3.0f), s2)), t_from<S>(1.0f));
        S h10 = t_add(t_sub(s3, t_mul(t_from<S>(2.0f), s2)), s);
        S h01 = t_add(t_mul(t_from<S>(-2.0f), s3), t_mul(t_from<S>(3.0f), s2));
        S h11 = t_sub(s3, s2);
        S v = t_add(t_add(t_mul(h00, y2), t_mul(h10, hk12)), t_add(t_mul(h01, yn2), t_mul(h11, hk72)));
        if ((to_f(v) > 0.0f) == pos0) lo = m; else hi = m;
      }
      S s = t_mul(t_from<S>(0.5f), t_add(lo, hi)), s2 = t_mul(s, s), s3 = t_mul(s2, s);
      S h00 = t_add(t_sub(t_mul(t_from<S>(2.0f), s3), t_mul(t_from<S>(3.0f), s2)), t_from<S>(1.0f));
      S h10 = t_add(t_sub(s3, t_mul(t_from<S>(2.0f), s2)), s);
      S h01 = t_add(t_mul(t_from<S>(-2.0f), s3), t_mul(t_from<S>(3.0f), s2));
      S h11 = t_sub(s3, s2);
      float rc = to_f(t_add(t_add(t_mul(h00, y[0]), t_mul(h10, t_mul(h, t_from<S>(to_f(k1[0]))))),
                            t_add(t_mul(h01, yn[0]), t_mul(h11, t_mul(h, t_from<S>(to_f(k7[0])))))));
      float pc = to_f(t_add(t_add(t_mul(h00, y[4]), t_mul(h10, t_mul(h, t_from<S>(to_f(k1[4]))))),
                            t_add(t_mul(h01, yn[4]), t_mul(h11, t_mul(h, t_from<S>(to_f(k7[4])))))));
      if (rc > disk_rin && rc < disk_rout && skipped++ >= skip) {
        out.disk_hit = 1; out.disk_r = rc; out.disk_phi = pc; out.fate = 1;
        return;
      }
    }

    if (Vol::kActive) {
      float hf = to_f(h);
      auto hm = [&](int i, float s) {
        float s2 = s * s, s3 = s2 * s;
        return (2.0f * s3 - 3.0f * s2 + 1.0f) * to_f(y[i]) + (s3 - 2.0f * s2 + s) * hf * to_f(k1[i]) + (-2.0f * s3 + 3.0f * s2) * to_f(yn[i]) + (s3 - s2) * hf * to_f(k7[i]);
      };
      float lim = 4.0f * vol->sigma(to_f(y[0]));
      float mmid = hm(2, 0.5f);
      float af = p.a, a2f = af * af, Lf = to_f(in.L), aLf = af * Lf;
      auto tdot = [&](float r, float sg) {
        float A = r * r + a2f;
        return A * (A - aLf) / (r * r - 2.0f * r + a2f) + af * (Lf - af * sg);
      };
      bool relevant = fmin(fmin(fabs(to_f(y[2])), fabs(to_f(yn[2]))), fabs(mmid)) < lim || ((to_f(y[2]) > 0.0f) != (to_f(yn[2]) > 0.0f));
      if (relevant) {
        int K = vol->nsub;
        float dh = hf / float(K);
        for (int q = 0; q < K; ++q) {
          float s = (float(q) + 0.5f) / float(K);
          float rr = hm(0, s);
          if (vol->slow) {
            float f = tdot(rr, hm(5, s));
            vol->T += 0.5f * f * dh;
            vol->add(rr, hm(2, s), hm(4, s), dh, hm(1, s), hm(3, s));
            vol->T += 0.5f * f * dh;
          } else {
            vol->add(rr, hm(2, s), hm(4, s), dh, hm(1, s), hm(3, s));
          }
        }
        if (vol->opaque()) { out.fate = 1; return; }
      } else if (vol->slow) {
        float f0 = tdot(to_f(y[0]), to_f(y[5])), f1 = tdot(to_f(yn[0]), to_f(yn[5])), fm = tdot(hm(0, 0.5f), hm(5, 0.5f));
        vol->T += hf * (f0 + 4.0f * fm + f1) * (1.0f / 6.0f);
      }
    }

    for (int i = 0; i < 6; ++i) y[i] = yn[i];
    if (t_lt(y[0], r_cap)) { out.fate = 1; return; }
    if (t_gt(y[0], rfar) && to_f(y[1]) > 0.0f) {
      out.fate = 0;
      out.mu_end = fmin(1.0f, fmax(-1.0f, to_f(y[2])));
      out.phi_end = to_f(y[4]);
      return;
    }
    h = t_mul(h, t_from<S>(step_factor_f(err_f)));
  }
  out.fate = 2;
}

template <typename Vol>
inline void dispatch_trace(int prec, constant MetalKP& p, df64 nr, df64 nth, df64 nph,
                           float rout, float atol, thread MetalTrace& tr, int skip, thread Vol* vol) {
  float ap = p.phi_atol > 0.0f ? p.phi_atol : atol;
  if (prec == 0) trace_kerr<df64, df64, Vol>(p, nr, nth, nph, p.rin, rout, p.tol, atol, ap, tr, skip, vol);
  else if (prec == 1) trace_kerr<float, float, Vol>(p, nr, nth, nph, p.rin, rout, p.tol, atol, ap, tr, skip, vol);
  else trace_kerr<df64, float, Vol>(p, nr, nth, nph, p.rin, rout, p.tol, atol, ap, tr, skip, vol);
}

// ------------------------------------------------------------------ Kernel principal de render
kernel void kernel_trace(
    constant MetalKP& p [[buffer(0)]],
    device float4* accum [[buffer(1)]],
    device float4* alb [[buffer(2)]],
    device const float* lut [[buffer(3)]],
    device const float* nt_tab [[buffer(4)]],
    device atomic_uint* stats [[buffer(5)]],
    texture2d<float, access::sample> sky [[texture(0)]],
    texture3d<float, access::sample> g_tex0 [[texture(1)]],
    texture3d<float, access::sample> g_tex1 [[texture(2)]],
    texture3d<float, access::sample> g_tex2 [[texture(3)]],
    texture2d<float, access::sample> g_theta_lut [[texture(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint bidx [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint warp [[simdgroup_index_in_threadgroup]]) {
  uint lq = lane & 3u, lx = lq & 1u, ly = lq >> 1u, quad = lane >> 2u;
  uint tilesx = (uint(p.w) + 15u) / 16u;
  uint bx = bidx % tilesx, by = bidx / tilesx;
  int px = int(bx * 16u + (warp & 1u) * 8u + (quad & 3u) * 2u + lx);
  int py = int(by * 8u + (warp >> 1u) * 4u + (quad >> 2u) * 2u + ly);
  bool inside = px < p.w && py < p.h;

  float jx = 0.0f, jy = 0.0f;
  if (p.sample_k > 0) {
    uint hq = hash3(uint(px >> 1), uint(py >> 1), p.seed);
    float kx = float(p.sample_k) * 0.7548776662466927f + float(hq & 0xFFFFu) / 65536.0f;
    float ky = float(p.sample_k) * 0.5698402909980532f + float(hq >> 16) / 65536.0f;
    jx = kx - floor(kx) - 0.5f;
    jy = ky - floor(ky) - 0.5f;
  }

  float3 col = float3(0.0f);
  float3 base = float3(0.0f);
  float trans = 1.0f;
  float u = 0.0f, v = 0.0f;
  bool sky_ok = false;
  int steps = 0;

  if (inside) {
    float x = (2.0f * ((float(px) + 0.5f + jx) / float(p.w)) - 1.0f) * p.tan_half * p.aspect;
    float y = (1.0f - 2.0f * ((float(py) + 0.5f + jy) / float(p.h))) * p.tan_half;
    float rout = p.nodisk ? 0.0f : p.rout;
    float atol = p.tol * 0.1f;

    float3 nz;
    float dop;
    observer_to_zamo_f(camera_ray_f(p, x, y), float3(p.bx, p.by, p.bz), nz, dop);

    MetalTrace tr;
    bool soft = p.edge_soft > 0.0f || p.edge_noise > 0.0f;
    bool volmode = kModel >= 0;
    float rtrace = (soft && rout > 0.0f) ? rout * (1.0f + p.edge_noise) * (1.0f + p.edge_soft) : rout;

    if (volmode) {
      VolAcc vol(p, lut, nt_tab, g_tex0, g_tex1, g_tex2, g_theta_lut, dop);
      dispatch_trace(kPrec, p, df_make(nz.x), df_make(nz.y), df_make(nz.z), rout, atol, tr, 0, &vol);
      base = vol.I;
      trans = vol.opaque() ? 0.0f : vol.trans;
      col = base;
      if (kModel >= 1) { base = radio_rgb_f(p, vol.Irad); col = base; trans = 0.0f; }
    } else {
      NoVol nv;
      dispatch_trace(kPrec, p, df_make(nz.x), df_make(nz.y), df_make(nz.z), rtrace, atol, tr, 0, &nv);
    }
    steps = tr.steps;

    for (int layer = 0; !volmode && tr.disk_hit; ++layer) {
      float r = tr.disk_r;
      float wgt = 1.0f;
      if (soft) {
        float phe = tr.disk_phi + p.phi;
        float ne = 0.6f * vnoise_f(3.0f * cos(phe), 3.0f * sin(phe), 1.7f) + 0.4f * vnoise_f(8.0f * cos(phe), 8.0f * sin(phe), 5.3f);
        float re = rout * (1.0f + p.edge_noise * (2.0f * ne - 1.0f));
        wgt = p.edge_soft > 0.0f ? (1.0f - ss_f(re * (1.0f - p.edge_soft), re * (1.0f + p.edge_soft), r)) : (r < re ? 1.0f : 0.0f);
      }
      float g = p.nodoppler ? redshift_f(p.a, r, 0.0f, p.dist, p.theta) : redshift_f(p.a, r, tr.L, p.dist, p.theta) * dop;
      if (p.beaming != 1.0f) g = pow(g, p.beaming);
      float Tprof = p.nt ? nt_temp_f(p, nt_tab, r) : disk_temp_f(r, p.rin, p.fnorm);
      float T_obs = g * p.tpeak * Tprof;
      float psi = tr.disk_phi + p.phi - 2.2f * log(r / p.rin);
      float lr = log(r);
      float n = 0.55f * vnoise_f(3.0f * cos(psi), 3.0f * sin(psi), 9.0f * lr) + 0.45f * vnoise_f(9.0f * cos(psi), 9.0f * sin(psi), 28.0f * lr);
      float k = p.exposure * (0.55f + (1.15f - 0.55f) * n);
      if (p.limb) k *= limb_eddington_f(emitter_mu_f(p.a, r, tr.L, tr.Q));
      float3 c = lut_lookup(lut, T_obs);
      if (!soft) { col = k * c; break; }
      base += trans * wgt * k * c;
      trans *= (1.0f - wgt);
      if (trans < 1e-3f || layer >= 2) { trans = 0.0f; break; }
      NoVol nv;
      dispatch_trace(kPrec, p, df_make(nz.x), df_make(nz.y), df_make(nz.z), rtrace, atol, tr, layer + 1, &nv);
      steps += tr.steps;
    }
    if (soft && !volmode) col = base;
    if (!tr.disk_hit && trans > 0.0f && tr.fate == 0 && p.use_sky) {
      float ph = tr.phi_end + p.phi;
      float uu = 0.5f - ph / (2.0f * M_PI_F);
      uu -= floor(uu);
      u = uu;
      v = acos(tr.mu_end) / M_PI_F;
      sky_ok = true;
    }
  }

  simdgroup_barrier(mem_flags::mem_none);
  if (p.use_sky) {
    float uH = simd_shuffle_xor(u, 1), vH = simd_shuffle_xor(v, 1);
    float uV = simd_shuffle_xor(u, 2), vV = simd_shuffle_xor(v, 2);
    int okH = simd_shuffle_xor(int(sky_ok), 1), okV = simd_shuffle_xor(int(sky_ok), 2);
    if (sky_ok) {
      float sx = lx ? -1.0f : 1.0f, sy = ly ? -1.0f : 1.0f;
      auto wrap = [](float d) { return d - rint(d); };
      float2 gx = okH ? float2(sx * wrap(uH - u), sx * (vH - v)) : float2(0.0f);
      float2 gy = okV ? float2(sy * wrap(uV - u), sy * (vV - v)) : float2(0.0f);
      constexpr sampler sky_smp(coord::normalized, s_address::repeat, t_address::clamp_to_edge, filter::linear, mip_filter::linear, max_anisotropy(16));
      constexpr sampler raw_smp(coord::normalized, s_address::repeat, t_address::clamp_to_edge, filter::linear, mip_filter::nearest);
      float3 s = p.sky_raw ? sky.sample(raw_smp, float2(u, v), level(0.0f)).xyz
                           : sky.sample(sky_smp, float2(u, v), gradient2d(gx, gy)).xyz;
      col = (trans == 1.0f) ? (p.sky_exposure * s) : (base + trans * p.sky_exposure * s);
    }
  }

  if (p.has_stats) {
    int wm = simd_max(steps);
    int ws = simd_sum(steps);
    if (lane == 0u) {
      atomic_fetch_add_explicit(&stats[0], uint(ws), memory_order_relaxed);
      atomic_fetch_add_explicit(&stats[1], uint(wm) * 32u, memory_order_relaxed);
      atomic_fetch_max_explicit(&stats[2], uint(wm), memory_order_relaxed);
    }
  }

  if (!inside) return;
  float L = 0.2126f * col.x + 0.7152f * col.y + 0.0722f * col.z;
  if (p.clamp_lum > 0.0f && L > p.clamp_lum) {
    float s = p.clamp_lum / L;
    col *= s;
    L = p.clamp_lum;
  }
  uint idx = uint(py) * uint(p.w) + uint(px);
  if (p.has_alb) {
    float m = fmax(col.x, fmax(col.y, col.z));
    float k = m > 1e-6f ? (1.0f / m) : 0.0f;
    float4 al = alb[idx];
    al.xyz += col * k;
    alb[idx] = al;
  }
  if (p.ldr) {
    float t = 1.0f / (1.0f + L);
    col *= t;
  }
  float4 acc = accum[idx];
  acc.xyz += col;
  accum[idx] = acc;
}

// ------------------------------------------------------------------ Post-processing (resolve, bloom, glare, tone mapping, HUD)
inline float4 ld_clamp(device const float4* t, int w, int h, int x, int y) {
  return t[size_t(clamp(y, 0, h - 1)) * size_t(w) + size_t(clamp(x, 0, w - 1))];
}

inline float3 bilerp(device const float4* t, int w, int h, float fx, float fy) {
  float x0 = floor(fx), y0 = floor(fy), tx = fx - x0, ty = fy - y0;
  int ix = int(x0), iy = int(y0);
  float3 a = ld_clamp(t, w, h, ix, iy).xyz, b = ld_clamp(t, w, h, ix + 1, iy).xyz;
  float3 c = ld_clamp(t, w, h, ix, iy + 1).xyz, d = ld_clamp(t, w, h, ix + 1, iy + 1).xyz;
  return mix(mix(a, b, tx), mix(c, d, tx), ty);
}

inline float3 ldz(device const float4* t, int w, int h, float fx, float fy) {
  float x0 = floor(fx), y0 = floor(fy), tx = fx - x0, ty = fy - y0;
  float3 r = float3(0.0f);
  for (int j = 0; j < 2; ++j)
    for (int i = 0; i < 2; ++i) {
      int ix = int(x0) + i, iy = int(y0) + j;
      if (ix < 0 || iy < 0 || ix >= w || iy >= h) continue;
      float wt = (i ? tx : 1.0f - tx) * (j ? ty : 1.0f - ty);
      r += wt * t[size_t(iy) * size_t(w) + size_t(ix)].xyz;
    }
  return r;
}

inline float aces_f(float x) {
  return clamp((x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f), 0.0f, 1.0f);
}

struct AvgArgs { int n; float inv; };
kernel void k_avg(device const float4* acc [[buffer(0)]], device float4* hdr [[buffer(1)]],
                  constant AvgArgs& a [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= a.n) return;
  float4 v = acc[gid];
  hdr[gid] = float4(v.xyz * a.inv, 1.0f);
}

struct DownArgs { int sw, sh, dw, dh; };
kernel void k_down(device const float4* src [[buffer(0)]], device float4* dst [[buffer(1)]],
                   constant DownArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  int x = int(gid.x), y = int(gid.y);
  if (x >= a.dw || y >= a.dh) return;
  float4 s = ld_clamp(src, a.sw, a.sh, 2 * x, 2 * y) + ld_clamp(src, a.sw, a.sh, 2 * x + 1, 2 * y) +
             ld_clamp(src, a.sw, a.sh, 2 * x, 2 * y + 1) + ld_clamp(src, a.sw, a.sh, 2 * x + 1, 2 * y + 1);
  dst[size_t(y) * size_t(a.dw) + size_t(x)] = float4(0.25f * s.xyz, 1.0f);
}

struct BlurArgs { int w, h, dx, dy; };
kernel void k_blur(device const float4* src [[buffer(0)]], device float4* dst [[buffer(1)]],
                   constant BlurArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  int x = int(gid.x), y = int(gid.y);
  if (x >= a.w || y >= a.h) return;
  const float wt[5] = {1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f, 4.0f / 16.0f, 1.0f / 16.0f};
  float3 s = float3(0.0f);
  for (int k = -2; k <= 2; ++k) s += wt[k + 2] * ld_clamp(src, a.w, a.h, x + k * a.dx, y + k * a.dy).xyz;
  dst[size_t(y) * size_t(a.w) + size_t(x)] = float4(s, 1.0f);
}

struct BrightArgs { int w, h, gw, gh, cw, ch; float thr; };
kernel void k_bright(device const float4* hdr [[buffer(0)]], device const float4* coarse [[buffer(1)]],
                     device float4* g0 [[buffer(2)]], constant BrightArgs& a [[buffer(3)]],
                     uint2 gid [[thread_position_in_grid]]) {
  int x = int(gid.x), y = int(gid.y);
  if (x >= a.gw || y >= a.gh) return;
  float3 acc = float3(0.0f);
  for (int dy = 0; dy < 2; ++dy)
    for (int dx = 0; dx < 2; ++dx) {
      int px = 2 * x + dx, py = 2 * y + dy;
      float3 c = ld_clamp(hdr, a.w, a.h, px, py).xyz;
      float3 m = bilerp(coarse, a.cw, a.ch, (px + 0.5f) * a.cw / float(a.w) - 0.5f, (py + 0.5f) * a.ch / float(a.h) - 0.5f);
      float3 d = max(c - m, float3(0.0f));
      float L = 0.2126f * d.x + 0.7152f * d.y + 0.0722f * d.z;
      float k = L > a.thr ? (L - a.thr) / L : 0.0f;
      acc += k * d;
    }
  g0[size_t(y) * size_t(a.gw) + size_t(x)] = float4(0.25f * acc, 1.0f);
}

struct StreaksArgs { int gw, gh, taps; float tau; };
kernel void k_streaks(device const float4* g0 [[buffer(0)]], device float4* g1 [[buffer(1)]],
                      constant StreaksArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  int x = int(gid.x), y = int(gid.y);
  if (x >= a.gw || y >= a.gh) return;
  float3 acc = float3(0.0f);
  for (int dir = 0; dir < 3; ++dir) {
    float ang = float(dir) * 1.0471975512f, cx = cos(ang), cy = sin(ang);
    for (int sgn = -1; sgn <= 1; sgn += 2)
      for (int d = 1; d <= a.taps; ++d) {
        float wt = 0.5f * exp(-float(d) / a.tau);
        acc += wt * ldz(g0, a.gw, a.gh, float(x) + float(sgn * d) * cx, float(y) + float(sgn * d) * cy);
      }
  }
  g1[size_t(y) * size_t(a.gw) + size_t(x)] = float4(acc, 1.0f);
}

kernel void k_final(
    device const float4* hdr [[buffer(0)]],
    device uchar4* out [[buffer(1)]],
    device ushort4* out16 [[buffer(2)]],
    constant MetalPolish& po [[buffer(3)]],
    device const float4* lv0 [[buffer(4)]],
    device const float4* lv1 [[buffer(5)]],
    device const float4* lv2 [[buffer(6)]],
    device const float4* lv3 [[buffer(7)]],
    device const float4* lv4 [[buffer(8)]],
    device const float4* lv5 [[buffer(9)]],
    device const float4* g1 [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]]) {
  int x = int(gid.x), y = int(gid.y);
  if (x >= po.w || y >= po.h) return;
  float3 c = hdr[size_t(y) * size_t(po.w) + size_t(x)].xyz;
  float hx = 0.5f * po.w, hy = 0.5f * po.h, hd = sqrt(hx * hx + hy * hy);
  float rx = float(x) + 0.5f - hx, ry = float(y) + 0.5f - hy, rn = sqrt(rx * rx + ry * ry) / hd;
  if (po.chroma > 0.0f) {
    float kr = 1.0f + po.chroma * rn, kb = 1.0f - po.chroma * rn;
    c.x = bilerp(hdr, po.w, po.h, hx + rx * kr - 0.5f, hy + ry * kr - 0.5f).x;
    c.z = bilerp(hdr, po.w, po.h, hx + rx * kb - 0.5f, hy + ry * kb - 0.5f).z;
  }
  if (po.bloom > 0.0f && po.levels > 0) {
    device const float4* lvs[6] = {lv0, lv1, lv2, lv3, lv4, lv5};
    float3 sum = float3(0.0f);
    for (int l = 0; l < po.levels; ++l) {
      float fx = (float(x) + 0.5f) * po.lw[l] / float(po.w) - 0.5f;
      float fy = (float(y) + 0.5f) * po.lh[l] / float(po.h) - 0.5f;
      sum += bilerp(lvs[l], po.lw[l], po.lh[l], fx, fy);
    }
    c += (po.bloom / float(po.levels)) * sum;
  }
  if (po.glare > 0.0f && po.has_g1) {
    c += po.glare * ldz(g1, po.gw, po.gh, (float(x) + 0.5f) * po.gw / float(po.w) - 0.5f, (float(y) + 0.5f) * po.gh / float(po.h) - 0.5f);
  }
  if (po.vignette > 0.0f) {
    float t = clamp((rn - 0.35f) / 0.75f, 0.0f, 1.0f), s = t * t * (3.0f - 2.0f * t);
    c *= (1.0f - po.vignette * s);
  }
  if (!po.ldr) {
    if (po.tonemap == 0) {
      float L = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
      c *= (1.0f / (1.0f + L));
    } else {
      c = float3(aces_f(0.6f * c.x), aces_f(0.6f * c.y), aces_f(0.6f * c.z));
    }
  }
  float3 e = pow(clamp(c, 0.0f, 1.0f), float3(1.0f / 2.2f));
  if (po.grain > 0.0f) {
    uint h1 = hash3(uint(x), uint(y), po.seed ^ (uint(po.frame) * 0x9E3779B1u));
    float n = (float(h1 & 0xFFFFu) + float(h1 >> 16)) / 65536.0f - 1.0f;
    float lum = 0.2126f * e.x + 0.7152f * e.y + 0.0722f * e.z;
    e += po.grain * (0.35f + 0.65f * 4.0f * lum * (1.0f - lum)) * n;
  }
  float3 q8 = clamp(e, 0.0f, 1.0f) * 255.0f + 0.5f;
  float3 q16 = clamp(e, 0.0f, 1.0f) * 65535.0f + 0.5f;
  size_t idx = size_t(y) * size_t(po.w) + size_t(x);
  out[idx] = uchar4(uchar(q8.x), uchar(q8.y), uchar(q8.z), 255);
  out16[idx] = ushort4(ushort(q16.x), ushort(q16.y), ushort(q16.z), 65535);
}

struct HudArgs { int w, h, cols, rows, scale, x0, y0, pad, rw, rh; };
kernel void k_hud(device uchar4* out [[buffer(0)]], device const uchar* grid [[buffer(1)]],
                  constant uchar* font [[buffer(2)]], constant HudArgs& a [[buffer(3)]],
                  uint2 gid [[thread_position_in_grid]]) {
  int x = int(gid.x), y = int(gid.y);
  if (x >= a.rw || y >= a.rh) return;
  int ox = a.x0 + x, oy = a.y0 + y;
  if (ox >= a.w || oy >= a.h) return;
  uchar4 px = out[size_t(oy) * size_t(a.w) + size_t(ox)];
  px.x = uchar(float(px.x) * 0.35f);
  px.y = uchar(float(px.y) * 0.35f);
  px.z = uchar(float(px.z) * 0.35f);
  int gx = x - a.pad, gy = y - a.pad;
  if (gx >= 0 && gy >= 0) {
    int cx = gx / (7 * a.scale), cy = gy / (13 * a.scale);
    if (cx < a.cols && cy < a.rows) {
      int ch = int(grid[cy * a.cols + cx]);
      if (ch >= 32 && ch <= 126) {
        int ix = (gx / a.scale) % 7, iy = (gy / a.scale) % 13;
        if ((font[(ch - 32) * 13 + iy] >> (6 - ix)) & 1) {
          bool warn = grid[cy * a.cols] == '!';
          if (warn && cx == 0) { out[size_t(oy) * size_t(a.w) + size_t(ox)] = px; return; }
          px = warn ? uchar4(255, 90, 80, 255) : uchar4(235, 235, 235, 255);
        }
      }
    }
  }
  out[size_t(oy) * size_t(a.w) + size_t(ox)] = px;
}

// ------------------------------------------------------------------ Test / batch kernels
kernel void cam_kernel(constant MetalKP& p [[buffer(0)]], device const float* xy [[buffer(1)]],
                       device float* out [[buffer(2)]], constant int& n [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= n) return;
  float3 d = camera_ray_f(p, xy[2 * gid], xy[2 * gid + 1]);
  out[3 * gid] = d.x; out[3 * gid + 1] = d.y; out[3 * gid + 2] = d.z;
}

kernel void nt_kernel(constant MetalKP& p [[buffer(0)]], device const float* r [[buffer(1)]],
                      device const float* nt_tab [[buffer(2)]], device float* out [[buffer(3)]],
                      constant int& n [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= n) return;
  out[gid] = nt_temp_f(p, nt_tab, r[gid]);
}

kernel void mu_kernel(constant float& a [[buffer(0)]], device const float* rLQ [[buffer(1)]],
                      device float* out [[buffer(2)]], constant int& n [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= n) return;
  out[gid] = emitter_mu_f(a, rLQ[3 * gid], rLQ[3 * gid + 1], rLQ[3 * gid + 2]);
}

kernel void boost_kernel(device const float* np [[buffer(0)]], constant float3& beta [[buffer(1)]],
                         device float* out [[buffer(2)]], constant int& n [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= n) return;
  float3 nz; float dop;
  observer_to_zamo_f(float3(np[3 * gid], np[3 * gid + 1], np[3 * gid + 2]), beta, nz, dop);
  out[4 * gid] = nz.x; out[4 * gid + 1] = nz.y; out[4 * gid + 2] = nz.z; out[4 * gid + 3] = dop;
}

kernel void batch_kernel(constant MetalKP& p [[buffer(0)]], device const float* rays [[buffer(1)]],
                         device MetalTrace* out [[buffer(2)]], constant int& n [[buffer(3)]],
                         constant int& skip [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= n) return;
  float rout = p.nodisk ? 0.0f : p.rout;
  NoVol nv;
  MetalTrace tr;
  dispatch_trace(p.prec, p, df_make(rays[3 * gid]), df_make(rays[3 * gid + 1]), df_make(rays[3 * gid + 2]),
                 rout, p.tol * 0.1f, tr, skip, &nv);
  out[gid] = tr;
}

kernel void vol_flight_kernel(constant MetalKP& p [[buffer(0)]], device const float* rays [[buffer(1)]],
                              device const float* lut [[buffer(2)]], device const float* nt_tab [[buffer(3)]],
                              texture3d<float, access::sample> gt0 [[texture(0)]],
                              texture3d<float, access::sample> gt1 [[texture(1)]],
                              texture3d<float, access::sample> gt2 [[texture(2)]],
                              texture2d<float, access::sample> gtl [[texture(3)]],
                              device float* out [[buffer(4)]], constant int& n [[buffer(5)]],
                              uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= n) return;
  MetalTrace tr;
  VolAcc vol(p, lut, nt_tab, gt0, gt1, gt2, gtl, 1.0f);
  dispatch_trace(p.prec, p, df_make(rays[3 * gid]), df_make(rays[3 * gid + 1]), df_make(rays[3 * gid + 2]),
                 p.rout, p.tol * 0.1f, tr, 0, &vol);
  out[gid] = vol.T_half;
}

kernel void torus_kernel(constant MetalKP& p [[buffer(0)]], device const float* in [[buffer(1)]],
                         device MetalTorusPoint* out [[buffer(2)]], constant int& n [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= n) return;
  TorusPt o;
  bool ok = torus_emit_f(p, in[3 * gid], in[3 * gid + 1], in[3 * gid + 2], 1.0f, o);
  out[gid] = ok ? MetalTorusPoint{o.rho, o.thE, o.B, o.sinB, o.j, o.alpha, o.g, o.e_emit} : MetalTorusPoint{0, 0, 0, 0, 0, 0, 0, 0};
}

kernel void turb_kernel(constant MetalKP& p [[buffer(0)]], device const float* in [[buffer(1)]],
                        device float* out [[buffer(2)]], constant int& n [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= n) return;
  float r = in[3 * gid], phi = in[3 * gid + 1], mu = in[3 * gid + 2];
  float h = p.v_hr * (p.v_flare == 0.0f ? 1.0f : pow(r * 0.1f, p.v_flare));
  float r32 = r * sqrt(r);
  out[2 * gid] = turb_density_f(p, r, phi, mu, h, 1.0f / (r32 + p.a), p.disk_time);
  out[2 * gid + 1] = spot_factor_f(p, r, phi, mu, p.disk_time);
}

kernel void radio_kernel(constant MetalKP& p [[buffer(0)]], device const float* rays [[buffer(1)]],
                         device const float* lut [[buffer(2)]], device const float* nt_tab [[buffer(3)]],
                         texture3d<float, access::sample> gt0 [[texture(0)]],
                         texture3d<float, access::sample> gt1 [[texture(1)]],
                         texture3d<float, access::sample> gt2 [[texture(2)]],
                         texture2d<float, access::sample> gtl [[texture(3)]],
                         device float* out [[buffer(4)]], constant int& n [[buffer(5)]],
                         uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= n) return;
  MetalTrace tr;
  VolAcc vol(p, lut, nt_tab, gt0, gt1, gt2, gtl, 1.0f);
  dispatch_trace(p.prec, p, df_make(rays[3 * gid]), df_make(rays[3 * gid + 1]), df_make(rays[3 * gid + 2]),
                 p.rout, p.tol * 0.1f, tr, 0, &vol);
  out[gid] = vol.Irad;
}

kernel void radio_image_kernel(constant MetalKP& p [[buffer(0)]], device const float* lut [[buffer(1)]],
                               device const float* nt_tab [[buffer(2)]],
                               texture3d<float, access::sample> gt0 [[texture(0)]],
                               texture3d<float, access::sample> gt1 [[texture(1)]],
                               texture3d<float, access::sample> gt2 [[texture(2)]],
                               texture2d<float, access::sample> gtl [[texture(3)]],
                               device float* Iout [[buffer(3)]], device float* about [[buffer(4)]],
                               constant int& ss [[buffer(5)]], uint2 gid [[thread_position_in_grid]]) {
  int px = int(gid.x), py = int(gid.y);
  if (px >= p.w || py >= p.h) return;
  float acc = 0.0f;
  for (int sy = 0; sy < ss; ++sy)
    for (int sx = 0; sx < ss; ++sx) {
      float x = (2.0f * ((float(px) + (float(sx) + 0.5f) / float(ss)) / float(p.w)) - 1.0f) * p.tan_half * p.aspect;
      float y = (1.0f - 2.0f * ((float(py) + (float(sy) + 0.5f) / float(ss)) / float(p.h))) * p.tan_half;
      float3 nz; float dop;
      observer_to_zamo_f(camera_ray_f(p, x, y), float3(p.bx, p.by, p.bz), nz, dop);
      MetalTrace tr;
      VolAcc vol(p, lut, nt_tab, gt0, gt1, gt2, gtl, dop);
      dispatch_trace(p.prec, p, df_make(nz.x), df_make(nz.y), df_make(nz.z), p.rout, p.tol * 0.1f, tr, 0, &vol);
      acc += vol.Irad;
    }
  size_t idx = size_t(py) * size_t(p.w) + size_t(px);
  Iout[idx] = acc / float(ss * ss);
  float x = (2.0f * ((float(px) + 0.5f) / float(p.w)) - 1.0f) * p.tan_half * p.aspect;
  float y = (1.0f - 2.0f * ((float(py) + 0.5f) / float(p.h))) * p.tan_half;
  float3 nz; float dop;
  observer_to_zamo_f(camera_ray_f(p, x, y), float3(p.bx, p.by, p.bz), nz, dop);
  InitDF o = init_from_zamo_df(df_make(p.a_df), df_make(p.dist_df), df_make(p.theta_sin_df), df_make(p.theta_cos_df),
                               df_make(nz.x), df_make(nz.y), df_make(nz.z));
  float s = p.theta_sin_df.x;
  about[2 * idx] = -to_f(o.L) / s;
  about[2 * idx + 1] = -to_f(o.ptheta);
}

kernel void grm_sample_kernel(constant MetalKP& p [[buffer(0)]], device const float* in [[buffer(1)]],
                              device float* out [[buffer(2)]], constant int& n [[buffer(3)]],
                              texture3d<float, access::sample> gt0 [[texture(0)]],
                              texture3d<float, access::sample> gt1 [[texture(1)]],
                              texture3d<float, access::sample> gt2 [[texture(2)]],
                              texture2d<float, access::sample> gtl [[texture(3)]],
                              uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= n) return;
  GrmPt o;
  bool ok = grmhd_sample(p, in[3 * gid], in[3 * gid + 1], in[3 * gid + 2], o, gt0, gt1, gt2, gtl);
  device float* q = out + 12 * gid;
  if (!ok) { for (int k = 0; k < 12; ++k) q[k] = NAN; return; }
  q[0] = o.rho; q[1] = o.thI; q[2] = o.absb; q[3] = o.beta;
  for (int k = 0; k < 4; ++k) { q[4 + k] = o.u[k]; q[8 + k] = o.b[k]; }
}

kernel void grm_map_kernel(constant MetalKP& p [[buffer(0)]], device const float* in [[buffer(1)]],
                           device float* out [[buffer(2)]], constant int& n [[buffer(3)]],
                           texture2d<float, access::sample> gtl [[texture(0)]],
                           uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= n) return;
  float X1 = log(in[2 * gid]), X2;
  f_inv_theta(p, X1, acos(clamp(in[2 * gid + 1], -1.0f, 1.0f)), X2, gtl);
  out[2 * gid] = X1; out[2 * gid + 1] = X2;
}

kernel void grm_null_kernel(constant MetalKP& p [[buffer(0)]], device const float* rays [[buffer(1)]],
                            device const float* lut [[buffer(2)]], device const float* nt_tab [[buffer(3)]],
                            texture3d<float, access::sample> gt0 [[texture(0)]],
                            texture3d<float, access::sample> gt1 [[texture(1)]],
                            texture3d<float, access::sample> gt2 [[texture(2)]],
                            texture2d<float, access::sample> gtl [[texture(3)]],
                            device float* out [[buffer(4)]], constant int& n [[buffer(5)]],
                            uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= n) return;
  MetalTrace tr;
  VolAcc vol(p, lut, nt_tab, gt0, gt1, gt2, gtl, 1.0f);
  dispatch_trace(p.prec, p, df_make(rays[3 * gid]), df_make(rays[3 * gid + 1]), df_make(rays[3 * gid + 2]),
                 p.rout, p.tol * 0.1f, tr, 0, &vol);
  out[gid] = vol.null_err;
}
