// v0.6b — Kerr render in CUDA. One thread per pixel runs exactly the shade() of v0.6a (CPU).
// Ver notes/physics/06b-kerr-cuda.md.
#include <cuda_fp16.h>
#include <cuda_gl_interop.h>
#include <chrono>
#include <future>
#include <cuda_runtime.h>

#include <algorithm>
#include <bhs/exr_loader.hpp>
#include <bhs/kerr.hpp>
#include <bhs/lut.hpp>
#include <bhs/lamppost.hpp>
#include <bhs/turbulence.hpp>
#include <bhs/novikov_thorne_kn.hpp>
#include <bhs/synchrotron.hpp>
#include <bhs/torus.hpp>
#include <bhs/grmhd.hpp>
#include <bhs/novikov_thorne.hpp>
#include <bhs/skymap.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "denoise.hpp"
#include "hud_font.hpp"
#include "kerr_cuda.cuh"
#include "render.hpp"

namespace bhs::kcuda {
namespace {

#define CK(x)                                                                      \
  do {                                                                             \
    cudaError_t e_ = (x);                                                          \
    if (e_ != cudaSuccess) {                                                       \
      std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      std::exit(2);                                                                \
    }                                                                              \
  } while (0)

// parameters in kernel units
// O6c: float copies of the double parameters that the device code converts at run time (float(p.x), F(p.x)): a double -> float conversion runs on the FP64 pipe
// (1/64 rate on GeForce) and the compiler cannot hoist it out of the sample loops. Filled at the end of make_kp with the same host conversion (round to nearest).
struct PF {
  float a, arm_amp, arm_cot, arm_om, beaming, c_alpha, c_mu, c_rout, c_th, c_w, c_wmu, disk_time, edge_soft, exposure, fl_boost, fl_t, fl_w, fnorm, g_b_unit, g_ce, g_dx1, g_hslope, g_ms, g_palpha, g_pnorm, g_pxt, g_rcap, g_rhigh, g_scut, g_startx1, g_step, g_tau, hv_amp, hv_scale, mcm, nt_rin, nt_rout, nu_obs, phi, r_hr, r_pn, r_pt, r_rout, r_xi, rin, ring_amp, ring_freq, ring_gap, rout, s_alpha, s_rin, s_rout, spot_amp, spot_om, spot_ph, spot_r, spot_size, t_ell, t_rmax, t_step, t_utin, th_e_max, the_min, ti_te, tpeak, turb_amp, turb_scale, v_flare, v_hr, v_tau, wh_M, wh_a, wh_lc, wh_lmax, wh_rho;
};
template <class T> __device__ __forceinline__ T pcv(double d, float f) { if constexpr (sizeof(T) == 4) return f; else return d; }  // the parameter in the precision of the path
struct KP {
  PF f;
  const bk::ZamoC* zamo;  // O6: observer constants of the main kernel (null elsewhere: computed per ray)
  int w, h, use_sky, sky_raw, nodisk, nodoppler, noshift, uniform_disk, smooth_disk, max_steps, prec;
  double phi_atol, l_min;
  double a, dist, theta, phi, tan_half, aspect, rin, rout, exposure, sky_exposure, tol, r_far, r_tail;
  int integ;  // O3: 1 = DOP853 on the pure-double thin-disk path
  double tpeak, fnorm;
  // v0.7d
  int nt, nt_n, ldr, sample_k, limb;
  uint32_t seed;
  double nt_rin, nt_rout, bx, by, bz, clamp_lum, beaming;
  double turb_amp, turb_scale, disk_time, spot_r, spot_ph, spot_size, spot_amp, spot_om, fl_t, fl_w, fl_boost;  // v0.9c (spot_ph = phase at time disk_time, computed on the host; spot_amp = amp without flare)
  double arm_amp, arm_cot, arm_om; int arm_m; double ring_amp, ring_freq, ring_gap, hv_amp, hv_scale;  // v2.6: spiral arms, rings with gaps, variable thickness
  double q2, lam; double tex_amp, tex_fr, tex_fphi; int tex_oct;  // v2.4: filamentary texture of the thin disk
  int slow;  // v0.9c2: slow light
  int rcheck;  // v2.2f (test): check the null condition in KS
  cudaTextureObject_t g_theta_lut = 0;  // v2.2h: X2(X1, theta) LUT (linear filter) + 2 Newton steps instead of 40 iterations
  cudaTextureObject_t g_tex[3];  // v2.2f: (rho, theta_i, |b|, beta), u^mu KS, bhat^mu KS (hardware trilinear filter)
  cudaTextureObject_t g_tex2[3]; double g_tau, g_dtM, g_pat; int g_blend;  // v4.0: 2nd snapshot (t + dt) and the time fraction tau in [0,1]; g_blend = 0: static snapshot
  int g_n1, g_n2, g_n3, el_model;
  double g_startx1, g_dx1, g_hslope, g_ms, g_palpha, g_pxt, g_pnorm, g_rho_unit, g_b_unit, g_rhigh, g_ce, g_rcap, g_gam, g_scut, g_te_c;
  int model;  // v2.2b: 0 layer, 1 torus with synchrotron
  double t_ell, t_utin, t_hmax, t_gamma, t_rin, t_rmax, ne0, pbeta, ti_te, nu_obs, mcm, tb_lo, tb_hi, th_e_max, t_step, g_step, the_min;
  double r_pn, r_pt, r_hr, r_xi, r_rout;  // v4.0 parametric RIAF (model 3)
  int s_flow, s_nar; double s_alpha, s_rin, s_rout, s_lo, s_hi;  // E1 spherical inflow (model 4)
  double c_mu, c_wmu, c_th, c_w, c_alpha, c_rout, c_lo, c_hi;  // E2 dual-cone jet walls (model 5): c_mu = cos theta_c, c_wmu = w sin theta_c (width in mu)
  int emis;
  int vol, v_sub;                // v0.9b: thick disk (Gaussian layer + radiative transport)
  double v_hr, v_flare, v_tau;
  double edge_soft, edge_noise;  // v0.9a2: soft / irregular outer edge (0 = v0.8 hard cut)
  double fwd[3], right[3], up[3];  // v0.8b: camera basis in (e_r, e_theta, e_phi): d' = fwd + x right + y up, n' = -d'/|d'|
  const float* nt_tab; double nt_lmin, nt_scale;
  const float* lp_tab; int lp_n; double lp_lmin, lp_lmax, lp_slope, lp_lo, lp_hi, lp_risco, lp_E, lp_L, lp_gamma, lp_lscale; int lp_plunge;  // E4 lamppost: log10 eps_E on a uniform ln r grid (lp_n = 0: off)
  float4* accum;
  float4* alb;  // albedo accumulator (nullptr = off)
  const float* lut;
  cudaTextureObject_t sky, sky2;
  int wh, wh_ellis; double wh_rho, wh_a, wh_M, wh_lc, wh_lmax;  // v2.1d: wormhole (model in the main kernel)
  unsigned long long* stats;  // [0] sum of steps, [1] sum of per-warp maxima * 32, [2] maximum, [3] sanitised pixels (non-finite value -> black), [4] 1 + index of the last pixel whose ray hit max_steps, [5] how many
};

constexpr double kT0 = lut::kT0, kT1 = lut::kT1;

__constant__ double c_lut_scale;  // O6d: (kN - 1) / ln(kT1 / kT0), set once by the host (a constexpr cannot hold a log): saves a double log per lookup
__device__ double3 lut_lookup(const float* d, double T) {
  if (T <= kT0) return make_double3(0, 0, 0);
  const double x = log(fmin(T, kT1) / kT0) * c_lut_scale;
  const int i = min(lut::kN - 2, int(x));
  const double f = x - i;
  return make_double3((1 - f) * d[3 * i] + f * d[3 * i + 3], (1 - f) * d[3 * i + 1] + f * d[3 * i + 4],
                      (1 - f) * d[3 * i + 2] + f * d[3 * i + 5]);
}

__device__ uint32_t hash3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t h = a * 0x9E3779B1u ^ b * 0x85EBCA6Bu ^ c * 0xC2B2AE35u;
  h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
  return h;
}
__device__ double vnoise(double x, double y, double z) {
  const int xi = int(floor(x)), yi = int(floor(y)), zi = int(floor(z));
  const double fx = x - xi, fy = y - yi, fz = z - zi;
  auto sm = [](double t) { return t * t * (3 - 2 * t); };
  double r = 0;
  for (int dz = 0; dz < 2; ++dz)
    for (int dy = 0; dy < 2; ++dy)
      for (int dx = 0; dx < 2; ++dx) {
        const double hv = double(hash3(uint32_t(xi + dx), uint32_t(yi + dy), uint32_t(zi + dz)) & 0xFFFFFF) / 16777216.0;
        r += hv * (dx ? sm(fx) : 1 - sm(fx)) * (dy ? sm(fy) : 1 - sm(fy)) * (dz ? sm(fz) : 1 - sm(fz));
      }
  return r;
}

// kerr::redshift
// v2.1e: prograde circular orbit in Kerr-Newman (device equivalent of ntk::circ): Omega and u^t at (r, a, q2)
__device__ void circ_kn(double r, double a, double q2, double& Om, double& ut) {
  const double N = 2.0 * r - q2, r2 = r * r, dN = 2.0 * (q2 - r) / (r2 * r);
  const double gtt = -1.0 + N / r2, gtp = -a * N / r2, gpp = r2 + a * a + a * a * N / r2, dgtt = dN, dgtp = -a * dN, dgpp = 2.0 * r + a * a * dN;
  Om = (-dgtp + sqrt(fmax(0.0, dgtp * dgtp - dgtt * dgpp))) / dgpp;
  ut = 1.0 / sqrt(-gtt - 2.0 * Om * gtp - Om * Om * gpp);
}
__device__ double redshift(double a, double r, double L, double r_obs, double th_obs, double q2 = 0.0, const bk::ZamoC* zc = nullptr) {
  if (zc && q2 == 0.0) {  // O6d: the observer's alpha and omega are per-ray constants (hoisted by the 1-thread kernel, same expressions): no sin/cos/sqrt per hit
    const double E_obs = (1.0 - zc->omega * L) / zc->alpha;
    const double ut = (pow(r, 1.5) + a) / (pow(r, 0.75) * sqrt(pow(r, 1.5) - 3.0 * sqrt(r) + 2.0 * a));
    const double Omega = 1.0 / (pow(r, 1.5) + a);
    return E_obs / (ut * (1.0 - Omega * L));
  }
  const double s = sin(th_obs), mu = cos(th_obs);
  const double Sigma = r_obs * r_obs + a * a * mu * mu, Delta = bk::delta_d(r_obs, a, q2);
  const double rr = r_obs * r_obs + a * a;
  const double A = rr * rr - a * a * Delta * s * s;
  const double alpha = sqrt(Sigma * Delta / A), omega = q2 == 0.0 ? 2.0 * a * r_obs / A : a * (2.0 * r_obs - q2) / A;
  const double E_obs = (1.0 - omega * L) / alpha;
  if (q2 != 0.0) { double Om, ut; circ_kn(r, a, q2, Om, ut); return E_obs / (ut * (1.0 - Om * L)); }
  const double ut = (pow(r, 1.5) + a) / (pow(r, 0.75) * sqrt(pow(r, 1.5) - 3.0 * sqrt(r) + 2.0 * a));
  const double Omega = 1.0 / (pow(r, 1.5) + a);
  return E_obs / (ut * (1.0 - Omega * L));
}

// disk::temperature
__device__ double disk_temp(double r, double rin, double fnorm) {
  if (r <= rin) return 0.0;
  const double f = pow(r, -3.0) * (1.0 - sqrt(rin / r));
  return pow(f / fnorm, 0.25);
}

// nt::Profile::temperature: table in ln r between r_in (ISCO) and r_out, linear interpolation; r^-3/4 tail
__device__ double nt_temp(const KP& p, double r) {
  if (r <= p.nt_rin) return 0.0;
  if (r >= p.nt_rout) return double(p.nt_tab[p.nt_n - 1]) * pow(r / p.nt_rout, -0.75);
  const double x = (log(r) - p.nt_lmin) * p.nt_scale;  // O6d: (n - 1) / (ln r_out - ln r_in) and ln r_in computed once on the host
  const int i = min(p.nt_n - 2, int(x));
  const double f = x - i;
  return (1 - f) * double(p.nt_tab[i]) + f * double(p.nt_tab[i + 1]);
}

// disk_emission.hpp: mu in the gas frame and Chandrasekhar (1960, Table XXIV) electron-scattering limb darkening (flux conserved)
__device__ double emitter_mu_d(double a, double r, double L, double Q) {
  const double r32 = pow(r, 1.5);
  const double ut = (r32 + a) / (pow(r, 0.75) * sqrt(r32 - 3.0 * sqrt(r) + 2.0 * a));
  const double Omega = 1.0 / (r32 + a);
  const double e_emit = ut * (1.0 - Omega * L);
  return fmin(1.0, sqrt(fmax(Q, 0.0)) / (r * fabs(e_emit)));
}
__device__ double ss(double a, double b, double x) {
  const double t = fmin(1.0, fmax(0.0, (x - a) / (b - a)));
  return t * t * (3.0 - 2.0 * t);
}
__device__ double limb_eddington_d(double mu) { return (1.0 + 2.06 * mu) / (1.0 + 2.06 * 2.0 / 3.0); }

// lorentz::observer_to_zamo: n' (observer) -> n (ZAMO); dop = E'/E_zamo. beta = observer velocity in the ZAMO frame.
__device__ void observer_to_zamo(double3 np, double bx, double by, double bz, double3& n, double& dop) {
  const double b2 = bx * bx + by * by + bz * bz;
  if (b2 < 1e-30) { n = np; dop = 1.0; return; }
  const double gamma = 1.0 / sqrt(1.0 - b2);
  const double bn = bx * np.x + by * np.y + bz * np.z;
  const double E = gamma * (1.0 + bn);
  const double k = ((gamma - 1.0) * bn / b2) + gamma;
  double3 q = make_double3(np.x + k * bx, np.y + k * by, np.z + k * bz);
  const double inv = 1.0 / sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
  n = make_double3(q.x * inv, q.y * inv, q.z * inv);
  dop = 1.0 / E;
}

// Photon direction in the observer frame for pixel (x, y) (FOV tangents): n' = -d'/|d'|, d' = fwd + x right + y up.
// Orthonormal basis: |d'|^2 = 1 + x^2 + y^2 (the same expression as v0.6b; with the default basis, fwd = -e_r, right = +e_phi,
// up = -e_theta, the result is bit-for-bit (1, y, -x)/nrm).
__device__ __forceinline__ double3 camera_ray(const KP& p, double x, double y) {
  const double nrm = sqrt(1.0 + x * x + y * y);
  const double dx = p.fwd[0] + x * p.right[0] + y * p.up[0];
  const double dy = p.fwd[1] + x * p.right[1] + y * p.up[1];
  const double dz = p.fwd[2] + x * p.right[2] + y * p.up[2];
  return make_double3(-dx / nrm, -dy / nrm, -dz / nrm);
}

// v0.9b: accumulator of the radiative transport along the ray (see core/include/bhs/volume.hpp). Front -> back:
// I += trans B (1 - e^-dtau), trans *= e^-dtau, with dtau = alpha E_emit Sigma dMino (length in the gas frame
// = E_emit d(lambda), d(lambda) = Sigma d(Mino)). B = shifted black-body colour (LUT at T_obs = g T(r)), without aesthetic
// noise (the physical turbulence is left for v0.9c).
// v0.9c: 3D value noise and fbm in float (same as the core's turb::vnoise/fbm), turbulence in the frame rotating with Omega(r), hot spot
__device__ float vnoise_f(float x, float y, float z) {
  const float fxf = floorf(x), fyf = floorf(y), fzf = floorf(z);
  const int xi = int(fxf), yi = int(fyf), zi = int(fzf);
  const float fx = x - fxf, fy = y - fyf, fz = z - fzf;
  auto sm = [](float t) { return t * t * (3.f - 2.f * t); };
  const float sx = sm(fx), sy = sm(fy), sz = sm(fz);
  float r = 0.f;
  for (int dz = 0; dz < 2; ++dz)
    for (int dy = 0; dy < 2; ++dy)
      for (int dx = 0; dx < 2; ++dx) {
        const float hv = float(hash3(uint32_t(xi + dx), uint32_t(yi + dy), uint32_t(zi + dz)) & 0xFFFFFF) * (1.f / 16777216.f);
        r += hv * (dx ? sx : 1.f - sx) * (dy ? sy : 1.f - sy) * (dz ? sz : 1.f - sz);
      }
  return r;
}
__device__ float fbm_f(float x, float y, float z) {
  float s = 0.f, wsum = 0.f, w = 1.f, f = 1.f;
  for (int k = 0; k < turb::kOct; ++k) {
    s += w * (2.f * vnoise_f(f * x + 17.31f * k, f * y + 5.77f * k, f * z + 11.13f * k) - 1.f);
    wsum += w; w *= 0.5f; f *= 2.f;
  }
  return s / wsum;
}
// lognormal density factor at (r, phi, mu) at time p.disk_time; h = H/r at r; omega = Omega(r)
__device__ float turb_density_f(const KP& p, float r, float phi, float mu, float h, float omega, float t) {
  if (p.turb_amp <= 0.0) return 1.f;
  float sn, cs;
  sincosf(phi - omega * t, &sn, &cs);
  const float inv = 1.f / (p.f.turb_scale * h);
  const float n = fbm_f(cs * inv, sn * inv, (logf(r) + 1.7f * mu) * inv);
  const float a = p.f.turb_amp;
  return __expf(a * n - 0.5f * a * a * float(turb::kVar));
}
// ---- v2.1d: wormhole ds^2 = -dt^2 + dl^2 + r(l)^2 dOmega^2 (float equivalent of bhs::wh::trace, core/include/bhs/wormhole.hpp; validated against it in --selftest)
__device__ __forceinline__ float wh_r(const KP& p, float l) {
  const float rho = p.f.wh_rho;
  if (p.wh_ellis) return sqrtf(rho * rho + l * l);
  const float al = fabsf(l), a = p.f.wh_a;
  if (al <= a) return rho;
  const float M = p.f.wh_M, x = 2.f * (al - a) / (3.14159265f * M);
  return rho + M * (x * atanf(x) - 0.5f * log1pf(x * x));
}
__device__ __forceinline__ float wh_drdl(const KP& p, float l) {
  if (p.wh_ellis) return l / sqrtf(float(p.wh_rho * p.wh_rho) + l * l);
  const float al = fabsf(l), a = p.f.wh_a;
  if (al <= a) return 0.f;
  const float x = 2.f * (al - a) / (3.14159265f * p.f.wh_M);
  return copysignf(0.63661977f * atanf(x), l);
}
// viewing ray from the camera at rest at l_c with cos(alpha) = initial p_l; returns the final side (+1/-1) and the angle psi swept in the plane (with flat tail)
__device__ void wh_trace(const KP& p, float cos_alpha, float sin_alpha, int& side, float& psi_out) {
  const float lc = p.f.wh_lc, lmax = p.f.wh_lmax, rho = p.f.wh_rho;
  const float B = wh_r(p, lc) * sin_alpha, B2 = B * B;
  float l = lc, pl = cos_alpha, psi = 0.f;
  side = 0; psi_out = 0.f;
  auto f = [&](float ll, float ppl, float* d) { const float r = wh_r(p, ll); d[0] = ppl; d[1] = B2 * wh_drdl(p, ll) / (r * r * r); d[2] = B / (r * r); };
  for (int it = 0; it < 40000; ++it) {
    const float r = wh_r(p, l);
    if (fabsf(l) > lmax) { side = l > 0.f ? 1 : -1; psi_out = psi + asinf(fminf(1.f, B / r)); return; }
    const float h = 0.02f * fminf(r, 5.f * rho + 0.2f * fabsf(l)) + 1e-9f;
    float k1[3], k2[3], k3[3], k4[3];
    f(l, pl, k1);
    f(l + 0.5f * h * k1[0], pl + 0.5f * h * k1[1], k2);
    f(l + 0.5f * h * k2[0], pl + 0.5f * h * k2[1], k3);
    f(l + h * k3[0], pl + h * k3[1], k4);
    l += h / 6.f * (k1[0] + 2.f * k2[0] + 2.f * k3[0] + k4[0]);
    pl += h / 6.f * (k1[1] + 2.f * k2[1] + 2.f * k3[1] + k4[1]);
    psi += h / 6.f * (k1[2] + 2.f * k2[2] + 2.f * k3[2] + k4[2]);
  }
  side = l > 0.f ? 1 : -1; psi_out = psi;  // did not terminate: ray trapped near the turning radius
}
// v2.6: disk structure (aesthetic): logarithmic spiral arms rotating as a rigid pattern (density wave, angular velocity arm_om),
// radial rings with gaps (1D noise in ln r) and variable thickness (hmod >= 1, so that the layer's step limit, based on sigma(r), stays safe)
__device__ float disk_struct_f(const KP& p, float r, float phd, float tem) {
  float f = 1.f;
  if (p.arm_amp > 0.0) {
    const float psi = phd - p.f.arm_om * tem - logf(r) * p.f.arm_cot;
    f *= fmaxf(0.f, 1.f + p.f.arm_amp * cosf(float(p.arm_m) * psi));
  }
  if (p.ring_amp > 0.0) {
    const float lr = logf(r), fr = p.f.ring_freq;
    const float n = 0.7f * vnoise_f(fr * lr, 3.1f, 7.7f) + 0.3f * vnoise_f(2.3f * fr * lr, 9.3f, 1.9f);
    f *= __expf(p.f.ring_amp * (2.f * n - 1.f));
    const float g = p.f.ring_gap;
    if (g > 0.f && n < g) { const float s = n / g; f *= s * s; }
  }
  return f;
}
__device__ float disk_hmod_f(const KP& p, float r, float phd, float tem) {
  if (p.hv_amp <= 0.0) return 1.f;
  float sn, cs;
  sincosf(phd - tem / (r * sqrtf(r) + p.f.a), &sn, &cs);
  const float s = p.f.hv_scale;
  return __expf(p.f.hv_amp * vnoise_f(cs * s + 21.7f, sn * s + 3.3f, logf(r) * s * 1.5f));
}
// hot-spot T factor at (r, phi, mu): (1 + amp G)^(1/4); G Gaussian in the Euclidean distance to the centre (phase and amp already at the right time)
__device__ float spot_factor_f(const KP& p, float r, float phi, float mu, float t) {
  if (p.spot_amp <= 0.0) return 1.f;
  const float rh = p.f.spot_r, z = r * mu;
  // phase and flare at time t (= disk_time without slow light): the phase at time disk_time is exact (host, double), only the difference t - disk_time is corrected
  const float ph = p.f.spot_ph + p.f.spot_om * (t - p.f.disk_time);
  float amp = p.f.spot_amp;
  if (p.fl_boost != 0.0) { const float d = (t - p.f.fl_t) / p.f.fl_w; amp *= 1.f + p.f.fl_boost * __expf(-0.5f * d * d); }
  const float d2 = r * r + rh * rh - 2.f * r * rh * cosf(phi - ph) + z * z;
  const float s = p.f.spot_size;
  return powf(1.f + amp * __expf(-0.5f * d2 / (s * s)), 0.25f);
}

// ---- v2.2b: Fishbone-Moncrief torus + thermal synchrotron (core/include/bhs/torus.hpp and synchrotron.hpp; here in double, gas only) ----
struct TorusPt { double rho, thE, B, sinB, j, alpha, g, e_emit; };
__device__ double s_I(double x) {
  const double c = cbrt(x);
  return 2.5651 * (1.0 + 1.92 / c + 0.9977 / (c * c)) * exp(-1.8899 * c);
}
__device__ double s_planck(double nu, double T) {
  const double x = synch::kH * nu / (synch::kKb * T);
  return x > 700.0 ? 0.0 : 2.0 * synch::kH * nu * nu * nu / (synch::kC * synch::kC) / expm1(x);
}
// v2.2b3: K_2(z) (Abramowitz & Stegun 9.8.5-9.8.8: K_0, K_1 to ~1e-7; K_2 = K_0 + 2 K_1 / z), template F (float production, double reference)
__device__ __forceinline__ float ex_(float x) { return __expf(x); }
__device__ __forceinline__ double ex_(double x) { return exp(x); }
template <class F>
__device__ F k2_t(F z) {
  F K0, K1;
  if (z <= F(2)) {
    const F lz = log(F(0.5) * z), y = (z / F(3.75)) * (z / F(3.75)), w = z * z * F(0.25);
    const F i0 = F(1) + y * (F(3.5156229) + y * (F(3.0899424) + y * (F(1.2067492) + y * (F(0.2659732) + y * (F(0.0360768) + y * F(0.0045813))))));
    const F i1 = z * (F(0.5) + y * (F(0.87890594) + y * (F(0.51498869) + y * (F(0.15084934) + y * (F(0.02658733) + y * (F(0.00301532) + y * F(0.00032411)))))));
    K0 = -lz * i0 + (F(-0.57721566) + w * (F(0.42278420) + w * (F(0.23069756) + w * (F(0.03488590) + w * (F(0.00262698) + w * (F(0.00010750) + w * F(0.0000074)))))));
    K1 = lz * i1 + (F(1) / z) * (F(1) + w * (F(0.15443144) + w * (F(-0.67278579) + w * (F(-0.18156897) + w * (F(-0.01919402) + w * (F(-0.00110404) + w * F(-0.00004686)))))));
  } else {
    const F u = F(2) / z, e = ex_(-z) / sqrt(z);
    K0 = e * (F(1.25331414) + u * (F(-0.07832358) + u * (F(0.02189568) + u * (F(-0.01062446) + u * (F(0.00587872) + u * (F(-0.00251540) + u * F(0.00053208)))))));
    K1 = e * (F(1.25331414) + u * (F(0.23498619) + u * (F(-0.03655620) + u * (F(0.01504268) + u * (F(-0.00780353) + u * (F(0.00325614) + u * F(-0.00068245)))))));
  }
  return K0 + F(2) * K1 / z;
}
// j (cgs) of thermal synchrotron: emis = 0 Dexter 2016 (A12/A18), 1 Leung et al. 2011 (see core/include/bhs/synchrotron.hpp)
template <class F>
__device__ F synch_j(int emis, F ne, F thE, F nuB, F sinB, F nu_em) {
  if (emis == 1) {
    const F nu_s = F(2.0 / 9.0) * nuB * sinB * thE * thE, X = nu_em / nu_s, c = cbrt(X);
    if (!(c < F(80))) return F(0);
    const F t = sqrt(X) + F(1.8877486253633869) * cbrt(sqrt(X));  // 2^(11/12) X^(1/6)
    return ne * F(synch::kE * synch::kE * sqrt(2.0) * M_PI / (3.0 * synch::kC)) * nu_s / k2_t<F>(F(1) / thE) * t * t * ex_(-c);
  }
  const F x = nu_em / (F(1.5) * nuB * sinB * thE * thE), cx = cbrt(x);
  const F II = F(2.5651) * (F(1) + F(1.92) / cx + F(0.9977) / (cx * cx)) * ex_(F(-1.8899) * cx);
  return ne * F(synch::kE * synch::kE / (2.0 * sqrt(3.0) * synch::kC)) * nu_em / (thE * thE) * II;
}
// gas state and emission at (r, mu = cos theta) for a photon of L = L_z/E; eobs_dop = the ray's E_obs dop (g = eobs_dop / e_emit).
// F = float (production: FP64 is 1/64 of the throughput) or double (reference of the tests). The quantities with a large dynamic range (j, alpha, I) accumulate in double.
template <class F>
__device__ bool torus_emit(const KP& p, double r_, double mu_, double L_, double eobs_dop, TorusPt& o) {
  const F r = F(r_), mu = F(mu_), a = pcv<F>(p.a, p.f.a), L = F(L_), one = F(1);
  const F s2 = one - mu * mu;
  if (s2 < F(1e-8) || r < F(p.t_rin * 0.7)) return false;
  const F Sig = r * r + a * a * mu * mu;
  const F gtt = -(one - F(2) * r / Sig), gtp = -F(2) * a * r * s2 / Sig, gpp = (r * r + a * a + F(2) * a * a * r * s2 / Sig) * s2;
  const F det = gtt * gpp - gtp * gtp, itt = gpp / det, itp = -gtp / det, ipp = gtt / det;
  const F ell = pcv<F>(p.t_ell, p.f.t_ell);
  const F D = itt - F(2) * ell * itp + ell * ell * ipp;
  if (D >= F(0)) return false;
  const F ut_abs = one / sqrt(-D), h = pcv<F>(p.t_utin, p.f.t_utin) / ut_abs;
  if (h <= one) return false;
  const F rho = pow((h - one) / F(p.t_hmax - 1.0), F(1.0 / (p.t_gamma - 1.0)));
  if (rho < F(1e-8)) return false;
  o.rho = double(rho);
  const F thI = F((p.t_gamma - 1.0) / p.t_gamma) * (h - one);
  const F thE = p.th_e_max > 0.0 ? pcv<F>(p.th_e_max, p.f.th_e_max) * (h - one) / F(p.t_hmax - 1.0) : thI * F(synch::kMp / synch::kMe / p.ti_te);
  o.thE = double(thE);
  if (thE < pcv<F>(p.the_min, p.f.the_min)) return false;  // outside the validity of the fit (see synchrotron.hpp)
  const double ne = p.ne0 * o.rho;
  const F Pn = F(synch::kMp) * thI + F(synch::kMe) * thE;  // P / (ne c^2) = m_p Theta_i + m_e Theta_e  (g)
  const F B = sqrt(F(8.0 * M_PI * synch::kC * synch::kC / p.pbeta) * F(ne) * Pn);
  o.B = double(B);
  const F ut_low = -ut_abs, ut_up = ut_low * (itt - ell * itp), uphi_up = ut_low * (itp - ell * ipp), Om = uphi_up / ut_up;
  const F e_emit = fabs(ut_up * (one - Om * L));
  o.e_emit = double(e_emit);
  const F g = F(eobs_dop) / e_emit;
  o.g = double(g);
  // angle between the toroidal field and the photon in the fluid frame: boost by v (ZAMO) along e_phi: cos = (n_phi - v)/(1 - v n_phi)
  const F Delta = r * r - F(2) * r + a * a, A = (r * r + a * a) * (r * r + a * a) - a * a * Delta * s2;
  const F alz = sqrt(Sig * Delta / A), omz = F(2) * a * r / A, varpi = sqrt(A / Sig * s2);
  const F nphi = (L / varpi) * alz / (one - omz * L), v = varpi * (Om - omz) / alz;
  const F cB = (nphi - v) / (one - v * nphi);
  const F sinB = sqrt(fmax(F(0), one - cB * cB));
  o.sinB = double(sinB);
  if (sinB <= F(0)) { o.j = 0.0; o.alpha = 0.0; return true; }
  const F nu_em = pcv<F>(p.nu_obs, p.f.nu_obs) / g;
  const F nuB = F(synch::kE / (2.0 * M_PI * synch::kMe * synch::kC)) * B;
  // j in F (float: ne ~ 1e5, e^2 ~ 2e-19, nu ~ 2e11 -> j ~ 1e-14..1e-22, within the float range)
  const F jF = synch_j<F>(p.emis, F(ne), thE, nuB, sinB, nu_em);
  o.j = double(jF);
  const F xp = F(synch::kH / (synch::kMe * synch::kC * synch::kC)) * nu_em / thE;  // h nu / (k T_e) = h nu / (Theta_e m_e c^2)
  // B_nu = 2 h nu^3/c^2 / expm1(xp): 2h/c^2 ~ 1.5e-47 underflows in float (min 1.4e-45); with nu = 1e10 nus the coefficient becomes 1.5e-17
  const F nus = nu_em * F(1e-10);
  const F Bn = F(2.0 * synch::kH * 1e30 / (synch::kC * synch::kC)) * nus * nus * nus / expm1(xp);
  o.alpha = xp > F(80) ? 0.0 : double(jF / Bn);
  return true;
}

// v4.0 parametric RIAF: n_e = ne0 (r/6)^-pn exp(-z^2/2H^2) (H = hr r), Theta_e = theta_e_max (r/6)^-pt, B from the plasma beta (as the torus), fluid on circular orbits at
// xi Omega_K (Omega_K = 1/(r^1.5 + a), the same at all heights), toroidal field; ISCO (rin) < r < riaf_rout. The profiles and exponents are free parameters of the model.
template <class F>
__device__ bool riaf_emit(const KP& p, double r_, double mu_, double L_, double eobs_dop, TorusPt& o) {
  const F r = F(r_), mu = F(mu_), a = pcv<F>(p.a, p.f.a), L = F(L_), one = F(1);
  const F s2 = one - mu * mu;
  if (s2 < F(1e-8) || r < pcv<F>(p.rin, p.f.rin) || r > pcv<F>(p.r_rout, p.f.r_rout)) return false;
  const F Sig = r * r + a * a * mu * mu;
  const F gtt = -(one - F(2) * r / Sig), gtp = -F(2) * a * r * s2 / Sig, gpp = (r * r + a * a + F(2) * a * a * r * s2 / Sig) * s2;
  const F Om = pcv<F>(p.r_xi, p.f.r_xi) / (r * sqrt(r) + a);
  const F den = -(gtt + F(2) * gtp * Om + gpp * Om * Om);
  if (den <= F(0)) return false;
  const F ut_up = one / sqrt(den);
  const F x = r / F(6), z = r * mu, H = pcv<F>(p.r_hr, p.f.r_hr) * r;
  const F rho = pow(x, -pcv<F>(p.r_pn, p.f.r_pn)) * exp(-F(0.5) * z * z / (H * H));
  if (rho < F(1e-8)) return false;
  o.rho = double(rho);
  const F thE = pcv<F>(p.th_e_max, p.f.th_e_max) * pow(x, -pcv<F>(p.r_pt, p.f.r_pt));
  o.thE = double(thE);
  if (thE < pcv<F>(p.the_min, p.f.the_min)) return false;
  const F thI = thE * F(p.ti_te * synch::kMe / synch::kMp);
  const double ne = p.ne0 * o.rho;
  const F Pn = F(synch::kMp) * thI + F(synch::kMe) * thE;
  const F B = sqrt(F(8.0 * M_PI * synch::kC * synch::kC / p.pbeta) * F(ne) * Pn);
  o.B = double(B);
  const F e_emit = fabs(ut_up * (one - Om * L));
  o.e_emit = double(e_emit);
  const F g = F(eobs_dop) / e_emit;
  o.g = double(g);
  const F Delta = r * r - F(2) * r + a * a, A = (r * r + a * a) * (r * r + a * a) - a * a * Delta * s2;
  const F alz = sqrt(Sig * Delta / A), omz = F(2) * a * r / A, varpi = sqrt(A / Sig * s2);
  const F nphi = (L / varpi) * alz / (one - omz * L), v = varpi * (Om - omz) / alz;
  const F cB = (nphi - v) / (one - v * nphi);
  const F sinB = sqrt(fmax(F(0), one - cB * cB));
  o.sinB = double(sinB);
  if (sinB <= F(0)) { o.j = 0.0; o.alpha = 0.0; return true; }
  const F nu_em = pcv<F>(p.nu_obs, p.f.nu_obs) / g;
  const F nuB = F(synch::kE / (2.0 * M_PI * synch::kMe * synch::kC)) * B;
  const F jF = synch_j<F>(p.emis, F(ne), thE, nuB, sinB, nu_em);
  o.j = double(jF);
  const F xp = F(synch::kH / (synch::kMe * synch::kC * synch::kC)) * nu_em / thE;
  const F nus = nu_em * F(1e-10);
  const F Bn = F(2.0 * synch::kH * 1e30 / (synch::kC * synch::kC)) * nus * nus * nus / expm1(xp);
  o.alpha = xp > F(80) ? 0.0 : double(jF / Bn);
  return true;
}

// E1: the same "afmhot" false colour for the bolometric intensity of the spherical inflow, log10 I in [s_lo, s_hi] (code units, M = 1)
__device__ double3 sph_rgb(const KP& p, double I, bool cones = false) {
  const double lo = cones ? p.c_lo : p.s_lo, hi = cones ? p.c_hi : p.s_hi;
  const double x = fmin(1.0, fmax(0.0, (log10(fmax(I, 1e-300)) - lo) / (hi - lo)));
  const double r = fmin(1.0, 2.0 * x), g = fmin(1.0, fmax(0.0, 2.0 * x - 0.5)), b = fmin(1.0, fmax(0.0, 2.0 * x - 1.0));
  return make_double3(pow(r, 2.2), pow(g, 2.2), pow(b, 2.2));
}
// "afmhot" false colour of the brightness temperature (Rayleigh-Jeans) T_b = I c^2 / (2 k nu^2), in log10 between tb_lo and tb_hi; linear = rgb^2.2 (the pipeline does the gamma)
__device__ double3 radio_rgb(const KP& p, double I) {
  const double Tb = I * synch::kC * synch::kC / (2.0 * synch::kKb * p.nu_obs * p.nu_obs);
  const double x = fmin(1.0, fmax(0.0, (log10(fmax(Tb, 1.0)) - p.tb_lo) / (p.tb_hi - p.tb_lo)));
  const double r = fmin(1.0, 2.0 * x), g = fmin(1.0, fmax(0.0, 2.0 * x - 0.5)), b = fmin(1.0, fmax(0.0, 2.0 * x - 1.0));
  return make_double3(pow(r, 2.2), pow(g, 2.2), pow(b, 2.2));
}

// E4 plunging region: g = E_obs / E_gas of the camera photon (E = 1, L, Q) at a gas that falls on the geodesic with the E and L of the ISCO. E_gas = u^t - L u^phi - p_r u^r with p_r = -sign(vr) sqrt(R)/Delta
// (the state runs in reversed time: p^r = -dr/dtau'), R = (r^2 + a^2 - a L)^2 - Delta (Q + (L - a)^2); u^t, u^phi, u^r as in core/include/bhs/lamppost.hpp (gas()).
__device__ double redshift_plunge(const KP& p, double r, double L, double Q, double vr) {
  const double a = p.a, s = sin(p.theta), mu = cos(p.theta);
  const double Sigma = p.dist * p.dist + a * a * mu * mu, Delta = bk::delta_d(p.dist, a, 0.0), rr = p.dist * p.dist + a * a;
  const double A = rr * rr - a * a * Delta * s * s, alpha = sqrt(Sigma * Delta / A), omega = 2.0 * a * p.dist / A;
  const double E_obs = (1.0 - omega * L) / alpha;
  const double D = r * r - 2.0 * r + a * a, E = p.lp_E, Lg = p.lp_L;
  const double ut = ((r * r + a * a + 2.0 * a * a / r) * E - 2.0 * a * Lg / r) / D, uph = ((1.0 - 2.0 / r) * Lg + 2.0 * a * E / r) / D;
  const double Ag = E * (r * r + a * a) - a * Lg, Bg = Lg - a * E;
  const double ur = -sqrt(fmax(0.0, (Ag * Ag - D * (r * r + Bg * Bg)) / (r * r * r * r)));
  const double R = (r * r + a * a - a * L) * (r * r + a * a - a * L) - D * (Q + (L - a) * (L - a));
  const double pr = (vr > 0.0 ? -1.0 : 1.0) * sqrt(fmax(R, 0.0)) / D;
  return E_obs / (ut - L * uph - pr * ur);
}

// E4 X-ray lamppost: log10 eps_E(r) (normalised to a maximum of 0) from the table on the ln r grid; below the first radius the first value, beyond the last the power-law slope of the last segment
__device__ double lp_log10e(const KP& p, double r) {
  const double x = log(r);
  if (x <= p.lp_lmin) return double(p.lp_tab[0]);
  if (x >= p.lp_lmax) return double(p.lp_tab[p.lp_n - 1]) + p.lp_slope * (x - p.lp_lmax) * 0.43429448190325176;
  const double u = (x - p.lp_lmin) / (p.lp_lmax - p.lp_lmin) * (p.lp_n - 1);
  const int i = min(p.lp_n - 2, int(u));
  const double f = u - i;
  return (1 - f) * double(p.lp_tab[i]) + f * double(p.lp_tab[i + 1]);
}
// "afmhot" false colour of log10 of the bolometric reflected intensity
__device__ double3 xray_rgb(double lI, double lo, double hi) {
  const double x = fmin(1.0, fmax(0.0, (lI - lo) / (hi - lo)));
  const double r = fmin(1.0, 2.0 * x), g = fmin(1.0, fmax(0.0, 2.0 * x - 0.5)), b = fmin(1.0, fmax(0.0, 2.0 * x - 1.0));
  return make_double3(pow(r, 2.2), pow(g, 2.2), pow(b, 2.2));
}

// ---- v2.2f: GRMHD snapshot (FMKS) in 3D textures; sampling in (r, mu, phi_BL) ----
// FMKS theta(X1, X2) and inverse in float (the cell has dX2 = 1/128: 1e-6 of precision in X2 is more than enough)
__device__ float f_theta(const KP& p, float X1, float X2, float* dth2) {
  const float hs = p.f.g_hslope, pa = p.f.g_palpha, xt = p.f.g_pxt, pn = p.f.g_pnorm;
  const float tg = 3.14159265f * X2 + 0.5f * (1.f - hs) * sinf(6.2831853f * X2), y = 2.f * X2 - 1.f;
  const float yr = powf(fabsf(y) / xt, pa);  // even exponent (14): (y/xt)^pa = |y/xt|^pa
  const float tj = pn * y * (1.f + yr / (pa + 1.f)) + 1.57079633f, w = __expf(p.f.g_ms * (p.f.g_startx1 - X1));
  if (dth2) {
    const float dg = 3.14159265f + (1.f - hs) * 3.14159265f * cosf(6.2831853f * X2), dj = 2.f * pn * (1.f + yr);
    *dth2 = dg + w * (dj - dg);
  }
  return tg + w * (tj - tg);
}
// reference: Newton with bisection (used if there is no LUT and in the tests)
__device__ bool f_inv_theta_exact(const KP& p, float X1, float th, float& X2) {
  float lo = 0.f, hi = 1.f, x = th * 0.31830989f;
  for (int it = 0; it < 40; ++it) {  // 14 were not enough: near the pole dtheta/dX2 ~ 15 and Newton falls back to bisection (measured: 0.017 cells of error at i = 18, j = 125)
    float d; const float f = f_theta(p, X1, x, &d) - th;
    if (fabsf(f) < 2e-7f || hi - lo < 1e-7f) break;
    if (f > 0.f) hi = x; else lo = x;
    float xn = x - f / d;
    if (!(xn > lo && xn < hi)) xn = 0.5f * (lo + hi);
    x = xn;
  }
  X2 = x;
  return true;
}
// 2D LUT (X1, theta) -> X2 and 2 Newton steps (measured error ~2e-7 in X2, the cell is 7.8e-3)
__device__ __forceinline__ bool f_inv_theta(const KP& p, float X1, float th, float& X2) {
  if (!p.g_theta_lut) return f_inv_theta_exact(p, X1, th, X2);
  const float s1 = (X1 - p.f.g_startx1) / (float(p.g_n1) * p.f.g_dx1);
  float x = tex2D<float>(p.g_theta_lut, s1, th * 0.31830989f);
  for (int it = 0; it < 2; ++it) {
    float d; const float f = f_theta(p, X1, x, &d) - th;
    x = fminf(1.f, fmaxf(0.f, x - f / fmaxf(d, 1e-6f)));
  }
  X2 = x;
  return true;
}
struct GrmPt { float rho, thI, absb, beta; float u[4], b[4]; };
// fields interpolated at (r, mu = cos theta_KS, phi_BL); false if outside the grid. Reprojects u (u.u = -1) and b (b.u = 0, |b| = 1) at the interpolated point.
__device__ bool grmhd_sample(const KP& p, float r, float mu, float phi_bl, GrmPt& o, float* gks_out = nullptr) {
  const float X1 = logf(r), s1 = (X1 - p.f.g_startx1) / (float(p.g_n1) * p.f.g_dx1);
  if (s1 <= 0.f || s1 >= 1.f) return false;
  float X2; f_inv_theta(p, X1, acosf(fminf(1.f, fmaxf(-1.f, mu))), X2);
  const float a = p.f.a;
  float phi = phi_bl;
  if (a != 0.f) { const float rp = 1.f + sqrtf(fmaxf(0.f, 1.f - a * a)), rm = 1.f - sqrtf(fmaxf(0.f, 1.f - a * a)); if (r > rp) phi += a / (rp - rm) * logf((r - rp) / (r - rm)); }
  float s3 = phi * 0.15915494f; s3 -= floorf(s3);
  float4 t0, t1, t2;
  if (p.g_blend) {
    // v4.0 time interpolation between two snapshots dt apart that FOLLOWS THE ROTATION: a feature seen at phi in snapshot A has turned by pat Omega tau dt at time t,
    // so A is read at phi - pat Omega tau dt and B at phi + pat Omega (1 - tau) dt, then blended (Omega Keplerian; u^mu / b^mu are re-normalised below).
    // pat = 0.2 is MEASURED (notes/physics/22f-b): against the middle snapshot of triplets of the own MAD run the error is minimal there (-35 % vs no shift; the full Keplerian shift is
    // worse than no shift: the emitting pattern turns much slower than the circular orbits).
    const float tau = p.f.g_tau, om_dt = float(p.g_pat * p.g_dtM) / (r * sqrtf(r) + a) * 0.15915494f;
    float sa = s3 - tau * om_dt, sb = s3 + (1.f - tau) * om_dt; sa -= floorf(sa); sb -= floorf(sb);
    const float4 a0 = tex3D<float4>(p.g_tex[0], sa, X2, s1), a1 = tex3D<float4>(p.g_tex[1], sa, X2, s1), a2 = tex3D<float4>(p.g_tex[2], sa, X2, s1);
    const float4 b0 = tex3D<float4>(p.g_tex2[0], sb, X2, s1), b1 = tex3D<float4>(p.g_tex2[1], sb, X2, s1), b2 = tex3D<float4>(p.g_tex2[2], sb, X2, s1);
    auto lerp4 = [&](float4 x, float4 y) { return make_float4(x.x + tau * (y.x - x.x), x.y + tau * (y.y - x.y), x.z + tau * (y.z - x.z), x.w + tau * (y.w - x.w)); };
    t0 = lerp4(a0, b0); t1 = lerp4(a1, b1); t2 = lerp4(a2, b2);
  } else {
    t0 = tex3D<float4>(p.g_tex[0], s3, X2, s1); t1 = tex3D<float4>(p.g_tex[1], s3, X2, s1); t2 = tex3D<float4>(p.g_tex[2], s3, X2, s1);
  }
  o.rho = t0.x; o.thI = t0.y; o.absb = t0.z; o.beta = t0.w;
  // covariant KS metric at the point
  const float th = acosf(fminf(1.f, fmaxf(-1.f, mu))), s2 = sinf(th) * sinf(th), rho2 = r * r + a * a * mu * mu, z = 2.f * r / rho2;
  const float gtt = -1.f + z, gtr = z, gtp = -a * s2 * z, grr = 1.f + z, grp = -a * s2 * (1.f + z), gthth = rho2, gpp = s2 * (rho2 + a * a * s2 * (1.f + z));
  float ur = t1.y, uth = t1.z, uph = t1.w, ut = t1.x;
  const float C = grr * ur * ur + gthth * uth * uth + gpp * uph * uph + 2.f * grp * ur * uph + 1.f, Bc = 2.f * (gtr * ur + gtp * uph);
  for (int it = 0; it < 3; ++it) { const float f = gtt * ut * ut + Bc * ut + C, df = 2.f * gtt * ut + Bc; if (fabsf(df) > 1e-12f) ut -= f / df; }
  o.u[0] = ut; o.u[1] = ur; o.u[2] = uth; o.u[3] = uph;
  float bt = t2.x, br = t2.y, bth = t2.z, bp = t2.w;
  const float ub = gtt * ut * bt + gtr * (ut * br + ur * bt) + gtp * (ut * bp + uph * bt) + grr * ur * br + gthth * uth * bth + grp * (ur * bp + uph * br) + gpp * uph * bp;
  bt += ub * ut; br += ub * ur; bth += ub * uth; bp += ub * uph;
  const float nb = gtt * bt * bt + 2.f * gtr * bt * br + 2.f * gtp * bt * bp + grr * br * br + gthth * bth * bth + 2.f * grp * br * bp + gpp * bp * bp;
  const float inv = nb > 1e-30f ? rsqrtf(nb) : 0.f;
  o.b[0] = bt * inv; o.b[1] = br * inv; o.b[2] = bth * inv; o.b[3] = bp * inv;
  if (gks_out) { gks_out[0] = gtt; gks_out[1] = gtr; gks_out[2] = gtp; gks_out[3] = grr; gks_out[4] = grp; gks_out[5] = gthth; gks_out[6] = gpp; }
  return true;
}

// MODEL: -1 = chosen at run time (test hooks); 0 = layer, 1 = torus, 2 = GRMHD (specialised kernels: fewer registers)
template <int MODEL = -1>
struct VolAccT {
  static constexpr bool kActive = true;
  __device__ __forceinline__ int mdl() const { if constexpr (MODEL >= 0) return MODEL; else return p.model; }
  const KP& p;
  float dop, eobs = 1.f, L = 0.f, trans = 1.f;
  float T = 0.f, tref = 0.f, T_half = -1.f;  // v0.9c2: flight time accumulated since the camera; T_half = T when trans < 1/2 (test)
  bool slow = false;
  float3 I = make_float3(0.f, 0.f, 0.f);
  double Irad = 0.0;  // v2.2b: accumulated specific intensity (cgs) in torus mode
  float Qc = 0.f;  // Carter constant of the ray (E = 1)
  float null_err = 0.f;  // v2.2f (test): max |g^{mu nu} p_mu p_nu| / p_t^2 in Kerr-Schild along the ray
  int nsub;
  __device__ VolAccT(const KP& pp, double d) : p(pp), dop(float(d)), nsub(pp.v_sub), slow(pp.slow != 0) {}
  // E_obs = (1 - omega L)/alpha (ZAMO observer) only depends on the radius, not on the point: g = E_obs dop / e_emit
  __device__ void begin(double L_, double Q_) {
    const double Lg = p.nodoppler ? 0.0 : L_;
    L = float(Lg);
    const double s = sin(p.theta), mu = cos(p.theta), Sigma = p.dist * p.dist + p.a * p.a * mu * mu, Delta = bk::delta_d(p.dist, p.a);
    const double rr = p.dist * p.dist + p.a * p.a, A = rr * rr - p.a * p.a * Delta * s * s;
    eobs = float(((1.0 - 2.0 * p.a * p.dist / A * Lg) / sqrt(Sigma * Delta / A)));
    Qc = float(Q_);
    // time reference: the flight time of a radial ray from the camera to r = 10 in Schwarzschild, r*(D) - r*(10), r* = r + 2 ln(r/2 - 1)
    // (convention: t_obs = disk_time is the instant at which light emitted at r = 10 along the line of sight would arrive; it only fixes the origin)
    tref = float((p.dist + 2.0 * log(fmax(p.dist / 2.0 - 1.0, 1e-9))) - (10.0 + 2.0 * log(4.0)));  // v2.7: r < 2 (camera inside the Schwarzschild horizon, outside the high-spin Kerr one): log of a negative was NaN
  }
  __device__ float mu_s() const { return mdl() == 5 ? p.f.c_mu : 0.f; }  // E2: the emitting surface is |mu| = mu_s (0: the equatorial layer)
  __device__ float sigma(float r) const { return mdl() == 5 ? p.f.c_wmu : mdl() >= 1 ? 1.f : (p.v_flare == 0.0 ? p.f.v_hr : p.f.v_hr * powf(r * 0.1f, p.f.v_flare)); }
  // model step limit: torus = ~0.6 M of path per step (Sigma dMino ~ length) inside the gas region; layer = no limit
  __device__ float cap(float r) const {
    if (mdl() == 2) return r < p.f.g_rcap ? fminf(p.f.g_step * r, 2.f) / (r * r + float(p.a * p.a)) : 1e30f;  // ds ~ g_step r (the radial cell is 0.022 r)
    if (mdl() == 5) return r < 1.05f * p.f.c_rout ? 0.1f * r / (r * r + float(p.a * p.a)) : 1e30f;
    if (mdl() == 4) return r < 1.05f * p.f.s_rout ? 0.1f * r / (r * r + float(p.a * p.a)) : 1e30f;  // ds ~ r/10 (j ~ r^-alpha: smooth on the scale r)
    if (mdl() == 3) return r < 1.5f * p.f.r_rout ? fminf(0.5f, 0.25f * p.f.r_hr * r) / (r * r + float(p.a * p.a)) : 1e30f;  // ds ~ a quarter of the local scale height
    return (mdl() == 1 && r < 4.f * p.f.t_rmax) ? p.f.t_step / (r * r + float(p.a * p.a)) : 1e30f;
  }
  __device__ bool opaque() const { return trans < 1e-3f; }
  __device__ void add_torus(float rf, float muf, float dM) {
    const double r = rf, mu = muf;
    TorusPt o;
    const bool ok = mdl() == 3 ? riaf_emit<float>(p, r, mu, double(L), double(eobs) * double(dop), o) : torus_emit<float>(p, r, mu, double(L), double(eobs) * double(dop), o);
    if (!ok || o.j <= 0.0) return;
    const float Sig = rf * rf + float(p.a * p.a) * muf * muf;
    const float ds = float(o.e_emit) * Sig * dM * p.f.mcm;  // length in the gas frame (cm)
    const float dtau = float(o.alpha) * ds, g = float(o.g);
    const float frac = dtau > 1e-6f ? -expm1f(-dtau) / dtau : 1.f;
    Irad += double(trans * g * g * g * float(o.j) * ds * frac);
    trans *= __expf(-dtau);
    if (T_half < 0.f && trans < 0.5f) T_half = T;
  }
  // GRMHD mode: n_e, B, Theta_e of the snapshot; photon momentum in KS: p_t = -1, p_phi = L, p_theta = dtheta/dMino = y3/sin(theta), p_r^BL = -y1/Delta,
  // p_r^KS = p_r^BL + (2r - aL)/Delta (from t_KS = t_BL + int 2r/Delta dr, phi_KS = phi_BL + int a/Delta dr); y1 = dr/dtau', y3 = dmu/dtau' (tau' = -tau, ray traced backwards)
  __device__ void add_grmhd(float rf, float muf, float phif, float dM, float y1, float y3) {
    GrmPt o; float gk[7];
    if (!grmhd_sample(p, rf, muf, phif + p.f.phi, o, gk)) return;  // the ray's phi is relative to the camera: add its azimuth (the snapshot is not axisymmetric), as the thin disk does
    if (!(o.rho > 0.f)) return;
    if (p.g_scut > 0.0 && o.absb * o.absb > p.f.g_scut * o.rho) return;  // E3: sigma = b^2/rho above the cut (jet funnel, density at the floor): no emission
    const float r = rf, mu = muf, a = p.f.a, Lp = L;
    const float s = sqrtf(fmaxf(1e-12f, 1.f - mu * mu)), Delta = r * r - 2.f * r + a * a;
    // p_r^BL = +-sqrt(R)/Delta, p_theta = +-sqrt(M/(1 - mu^2)) with the constants L, Q (null by construction in BL; the signs come from the state y1, y3)
    const float Rr = (r * r + a * a - a * Lp) * (r * r + a * a - a * Lp) - Delta * ((Lp - a) * (Lp - a) + Qc), Mm = (1.f - mu * mu) * (Qc + a * a * mu * mu) - Lp * Lp * mu * mu;
    const float prBL = (y1 > 0.f ? -1.f : 1.f) * sqrtf(fmaxf(Rr, 0.f)) / Delta, pth = (y3 >= 0.f ? 1.f : -1.f) * sqrtf(fmaxf(Mm, 0.f) / fmaxf(1.f - mu * mu, 1e-12f));
    const float pt = -1.f, pr = prBL + (2.f * r - a * Lp) / Delta, pp = Lp;
    if (p.rcheck) {  // null: g^{mu nu} p_mu p_nu em KS
      const float S = r * r + a * a * mu * mu, s2 = s * s;
      const float gtt_i = -(1.f + 2.f * r / S), gtr_i = 2.f * r / S, grr_i = Delta / S, grp_i = a / S, gthth_i = 1.f / S, gpp_i = 1.f / (S * s2);
      const float nul = gtt_i * pt * pt + 2.f * gtr_i * pt * pr + grr_i * pr * pr + 2.f * grp_i * pr * pp + gthth_i * pth * pth + gpp_i * pp * pp;
      null_err = fmaxf(null_err, fabsf(nul));
    }
    const float E = -(pt * o.u[0] + pr * o.u[1] + pth * o.u[2] + pp * o.u[3]);
    if (!(E > 1e-6f)) return;
    const float pb = pt * o.b[0] + pr * o.b[1] + pth * o.b[2] + pp * o.b[3];
    const float cB = fminf(1.f, fmaxf(-1.f, pb / E)), sinB = sqrtf(fmaxf(0.f, 1.f - cB * cB));
    const float g = float(eobs) * float(dop) / E, thI = o.thI;
    float thE;
    if (p.el_model == 2) thE = p.f.g_ce * thI;
    else if (p.el_model == 1) {  // R_high, EHT convention (ipole; EHT GRRT formula list eq. 8): Theta_e = (m_p/m_e)(ge-1)(gp-1)/((gp-1) + (ge-1) R) u/rho, ge = 4/3, gp = 5/3
      const float b2 = o.beta * o.beta; const float R = (p.f.g_rhigh * b2 + 1.f) / (1.f + b2); thE = thI * float(p.g_te_c) / (2.f + R);
    } else if (p.el_model == 3) { const float b2 = o.beta * o.beta; const float tite = (p.f.g_rhigh * b2 + 1.f) / (1.f + b2); thE = thI * float(synch::kMp / synch::kMe) / (1.f + tite); }  // R_high, single-temperature split (before 2026-11-18)
    else thE = thI * float(synch::kMp / synch::kMe) / (1.f + p.f.ti_te);
    if (thE < p.f.the_min || sinB <= 0.f) return;
    // v2.2h: all the transport in float (FP64 is 1/64 of the rate on the RTX 5080); the intervals fit in float (j ~ 1e-14..1e-22, ds ~ 1e14 cm)
    const float ne = o.rho * float(p.g_rho_unit / (synch::kMp + synch::kMe)), Bg = o.absb * p.f.g_b_unit;
    const float nuB = float(synch::kE / (2.0 * M_PI * synch::kMe * synch::kC)) * Bg, nu_em = p.f.nu_obs / g;
    const float j = synch_j<float>(p.emis, ne, thE, nuB, sinB, nu_em);
    if (!(j > 0.f)) return;
    const float xp = float(synch::kH / (synch::kMe * synch::kC * synch::kC)) * nu_em / thE;
    const float nus = nu_em * 1e-10f;
    const float Bn = float(2.0 * synch::kH * 1e30 / (synch::kC * synch::kC)) * nus * nus * nus / expm1f(xp);
    const float alpha = xp > 80.f ? 0.f : j / Bn;
    const float Sig = r * r + a * a * mu * mu;
    const float ds = E * Sig * dM * p.f.mcm;
    const float dtau = alpha * ds, frac = dtau > 1e-6f ? -expm1f(-dtau) / dtau : 1.f;
    Irad += double(trans * g * g * g * j * ds * frac);
    trans *= __expf(-dtau);
  }
  // E1 spherical inflow (Narayan, Johnson & Gammie 2019): bolometric, optically thin, I += g^4 j ds. Photon (traced backwards, E = 1): k_t = -1, k_phi = L,
  // k_r = -+sqrt(R)/Delta (sign from the state y1, as in add_grmhd). Free fall (u_t = -1, u_phi = 0): u^t = A/(Sigma Delta), u^phi = 2 a r/(Sigma Delta),
  // u^r = -sqrt(2 r (r^2 + a^2))/Sigma (from u.u = -1: A - Sigma Delta = 2 r (r^2 + a^2)). Gas at rest: u^t = 1/sqrt(-g_tt), none inside the ergoregion.
  // photon energy in the frame of gas falling freely from infinity (u_t = -1, u_phi = u_theta = 0), E = -k.u; shared by E1 and E2
  __device__ float infall_E(float r, float mu, float y1) const {
    const float a = p.f.a, a2 = a * a, s2 = fmaxf(1.f - mu * mu, 0.f), Sig = r * r + a2 * mu * mu, Delta = r * r - 2.f * r + a2;
    const float rr = r * r + a2, A = rr * rr - a2 * Delta * s2;
    const float ut = A / (Sig * Delta), uph = 2.f * a * r / (Sig * Delta), ur = -sqrtf(2.f * r * rr) / Sig;
    const float Rr = (rr - a * L) * (rr - a * L) - Delta * ((L - a) * (L - a) + Qc);
    const float kr = (y1 > 0.f ? -1.f : 1.f) * sqrtf(fmaxf(Rr, 0.f)) / Delta;
    return ut - kr * ur - L * uph;
  }
  // E2 dual-cone jet walls (Papoutsis et al. 2023): j = r^-alpha [G(theta - theta_c) + G(theta - (pi - theta_c))], G(x) = exp(-(x/w)^2), free fall, I += g^3 j ds
  __device__ void add_cones(float r, float mu, float dM, float y1) {
    const float a = p.f.a, Delta = r * r - 2.f * r + a * a;
    if (!(Delta > 0.f) || r > p.f.c_rout) return;
    const float th = acosf(fminf(1.f, fmaxf(-1.f, mu))), w = p.f.c_w, tc = p.f.c_th;
    const float x1 = (th - tc) / w, x2 = (th - (3.14159265f - tc)) / w;
    const float G = __expf(-x1 * x1) + __expf(-x2 * x2);
    if (G < 1e-12f) return;
    const float E = infall_E(r, mu, y1);
    if (!(E > 1e-12f) || !isfinite(E)) return;
    const float j = powf(r, -p.f.c_alpha) * G, g = float(eobs) * float(dop) / E;
    const float Sig = r * r + a * a * mu * mu, ds = E * Sig * dM;
    Irad += double(g * g * g * j * ds);
  }
  __device__ void add_sph(float r, float mu, float dM, float y1) {
    if (r < p.f.s_rin || r > p.f.s_rout) return;
    const float a = p.f.a, a2 = a * a, Sig = r * r + a2 * mu * mu, Delta = r * r - 2.f * r + a2;
    if (!(Delta > 0.f)) return;
    float E;
    if (p.s_flow == 0) {
      E = infall_E(r, mu, y1);
    } else {
      const float mgtt = 1.f - 2.f * r / Sig;
      if (!(mgtt > 1e-6f)) return;
      E = 1.f / sqrtf(mgtt);
    }
    if (!(E > 1e-12f) || !isfinite(E)) return;
    const float j = p.s_nar ? 1.f / (157.91367f * r * r * r * r * sqrtf(fmaxf(1.f - 2.f / r, 1e-30f))) : powf(r, -p.f.s_alpha) * (1.f / 157.91367f);  // 16 pi^2 = 157.91367
    const float g = float(eobs) * float(dop) / E;
    const float ds = E * Sig * dM;  // proper length in the gas frame (affine dlambda = Sigma dMino, nu_gas = E)
    Irad += double(g * g * g * g * j * ds);
  }
  __device__ void add(float r, float mu, float phi, float dM, float y1 = 0.f, float y3 = 0.f) {
    if (mdl() == 2) { add_grmhd(r, mu, phi, dM, y1, y3); return; }
    if (mdl() == 4) { add_sph(r, mu, dM, y1); return; }
    if (mdl() == 5) { add_cones(r, mu, dM, y1); return; }
    if (mdl() == 1 || mdl() == 3) { add_torus(r, mu, dM); return; }
    if (r <= p.f.rin) return;
    const float rout = p.f.rout, es = p.f.edge_soft;
    float w = 1.f;
    if (es > 0.f) { const float t = fminf(1.f, fmaxf(0.f, (r - rout * (1.f - es)) / (rout * 2.f * es))); w = 1.f - t * t * (3.f - 2.f * t); }
    else if (r >= rout) return;
    if (w <= 0.f) return;
    const float phd0 = phi + p.f.phi, tem0 = slow ? p.f.disk_time - (T - tref) : p.f.disk_time;
    const float H = sigma(r) * r * disk_hmod_f(p, r, phd0, tem0), z = r * mu;
    float alpha = w * p.f.v_tau / (2.50662827f * H) * __expf(-0.5f * z * z / (H * H));
    if (alpha < 1e-9f) return;
    const float rs = sqrtf(r), r32 = r * rs, r34 = rs * sqrtf(rs), af = p.f.a;
    const float phd = phi + p.f.phi;  // azimuth in the disk (the camera is at phi = p.phi)
    const float tem = slow ? p.f.disk_time - (T - tref) : p.f.disk_time;  // emission time
    const float tdf = turb_density_f(p, r, phd, mu, sigma(r), 1.f / (r32 + af), tem);
    alpha *= tdf * disk_struct_f(p, r, phd, tem);
    const float den_ut = r32 - 3.f * rs + 2.f * af;
    if (!(den_ut > 0.f)) return;  // v2.7: inside the circular photon orbit u^t does not exist (NaN): no emission
    const float ut = (r32 + af) / (r34 * sqrtf(den_ut));
    const float e_emit = fabsf(ut * (1.f - L / (r32 + af)));
    if (!(e_emit > 1e-6f) || !isfinite(e_emit)) return;  // v2.7: infinite redshift (near the horizon): no light (avoids inf/NaN)
    const float Sig = r * r + af * af * mu * mu;
    const float dtau = alpha * e_emit * Sig * dM;
    float g = eobs * dop / e_emit;
    if (p.beaming != 1.0) g = powf(g, p.f.beaming);
    // NT temperature profile (table in ln r) or Newtonian, in float
    float Tprof;
    if (p.nt) {
      if (r >= p.f.nt_rout) Tprof = p.nt_tab[p.nt_n - 1] * powf(r / p.f.nt_rout, -0.75f);
      else {
        const float lmin = logf(p.f.nt_rin), lmax = logf(p.f.nt_rout), x = (logf(r) - lmin) / (lmax - lmin) * (p.nt_n - 1);
        const int i = min(p.nt_n - 2, int(x));
        const float f = x - i;
        Tprof = (1.f - f) * p.nt_tab[i] + f * p.nt_tab[i + 1];
      }
    } else {
      Tprof = powf(powf(r, -3.f) * (1.f - sqrtf(p.f.rin / r)) / p.f.fnorm, 0.25f);
    }
    const float Tobs = g * p.f.tpeak * Tprof * spot_factor_f(p, r, phd, mu, tem);  // (do not call it T: it shadows the accumulated flight time, member T)
    float3 c = make_float3(0.f, 0.f, 0.f);
    if (Tobs > float(kT0)) {  // colour LUT (linear interpolation in log T, like lut_lookup)
      const float xx = logf(fminf(Tobs, float(kT1)) / float(kT0)) / logf(float(kT1 / kT0)) * (lut::kN - 1);
      const int i = min(lut::kN - 2, int(xx));
      const float f = xx - i;
      c = make_float3((1.f - f) * p.lut[3 * i] + f * p.lut[3 * i + 3], (1.f - f) * p.lut[3 * i + 1] + f * p.lut[3 * i + 4], (1.f - f) * p.lut[3 * i + 2] + f * p.lut[3 * i + 5]);
    }
    const float wgt = trans * (-expm1f(-dtau)) * p.f.exposure;
    I.x += wgt * c.x; I.y += wgt * c.y; I.z += wgt * c.z;
    trans *= __expf(-dtau);
    if (T_half < 0.f && trans < 0.5f) T_half = T;
  }
};
using VolAcc = VolAccT<-1>;

// PREC: -1 = chosen at run time (hooks); 0 = double, 1 = float, 2 = mixed (double state, float derivatives) -- at compile time in the specialised kernels
template <int PREC = -1, class V = VolAcc>
__device__ __forceinline__ void trace_dispatch_vol(const KP& p, double nr, double nth, double nph, double rout, double atol, bk::Trace& tr, V& vol) {
  const double ap = p.phi_atol > 0 ? p.phi_atol : atol;
  const int pr = PREC >= 0 ? PREC : p.prec;
  if ((PREC < 0 || PREC == 0) && pr == 0) bk::trace<double, double, V, bk::KerrM>(p.a, p.dist, p.theta, nr, nth, nph, p.rin, rout, p.tol, atol, ap, p.l_min, p.r_far, p.max_steps, tr, 0, &vol, p.q2, p.lam, p.r_tail, 0, p.zamo);
  else if ((PREC < 0 || PREC == 1) && pr == 1) bk::trace<float, float, V, bk::KerrM>(p.a, p.dist, p.theta, nr, nth, nph, p.rin, rout, p.tol, atol, ap, p.l_min, p.r_far, p.max_steps, tr, 0, &vol, p.q2, p.lam, p.r_tail, 0, p.zamo);
  else if (PREC < 0 || PREC == 2) bk::trace<double, float, V, bk::KerrM>(p.a, p.dist, p.theta, nr, nth, nph, p.rin, rout, p.tol, atol, ap, p.l_min, p.r_far, p.max_steps, tr, 0, &vol, p.q2, p.lam, p.r_tail, 0, p.zamo);
}

template <int PREC, class M>
__device__ __forceinline__ void trace_dispatch_m(const KP& p, double nr, double nth, double nph, double rout, double atol, bk::Trace& tr, int skip = 0) {
  const double ap = p.phi_atol > 0 ? p.phi_atol : atol;
  const int pr = PREC >= 0 ? PREC : p.prec;
  if ((PREC < 0 || PREC == 0) && pr == 0) bk::trace<double, double, bk::NoVol, M>(p.a, p.dist, p.theta, nr, nth, nph, p.rin, rout, p.tol, atol, ap, p.l_min, p.r_far, p.max_steps, tr, skip, (bk::NoVol*)nullptr, p.q2, p.lam, p.r_tail, p.integ, p.zamo);
  else if ((PREC < 0 || PREC == 1) && pr == 1) bk::trace<float, float, bk::NoVol, M>(p.a, p.dist, p.theta, nr, nth, nph, p.rin, rout, p.tol, atol, ap, p.l_min, p.r_far, p.max_steps, tr, skip, (bk::NoVol*)nullptr, p.q2, p.lam, p.r_tail, p.integ, p.zamo);
  else if (PREC < 0 || PREC == 2) bk::trace<double, float, bk::NoVol, M>(p.a, p.dist, p.theta, nr, nth, nph, p.rin, rout, p.tol, atol, ap, p.l_min, p.r_far, p.max_steps, tr, skip, (bk::NoVol*)nullptr, p.q2, p.lam, p.r_tail, p.integ, p.zamo);
}

// v4.0: the metric is chosen once per ray and the whole trace is compiled for it (the Kerr path has no Lambda / charge test). Vol disks (layer, torus, GRMHD) are Kerr only.
template <int PREC = -1>
__device__ __forceinline__ void trace_dispatch(const KP& p, double nr, double nth, double nph, double rout, double atol, bk::Trace& tr, int skip = 0) {
  if (p.lam != 0.0) trace_dispatch_m<PREC, bk::SdSM>(p, nr, nth, nph, rout, atol, tr, skip);
  else if (p.q2 != 0.0) trace_dispatch_m<PREC, bk::KerrNewmanM>(p, nr, nth, nph, rout, atol, tr, skip);
  else trace_dispatch_m<PREC, bk::KerrM>(p, nr, nth, nph, rout, atol, tr, skip);
}

// Each thread = 1 pixel and 1 sub-pixel sample (R2 jitter per quad, sample index sample_k; sample 0 is the
// pixel centre). Adds the HDR colour into p.accum (RGBA32F); the resolve (average, bloom, tone mapping) is separate.
// 1D blocks; 4 consecutive threads form a 2x2 quad (horizontal neighbour = lane^1, vertical = lane^2) to
// derive the sky-uv gradient via __shfl_sync, like v0.5b's subgroup_quad. The jitter is the same in the 4
// threads of the quad (quad hash), so that the uv difference between neighbours remains the pixel's derivative.
// One warp (32) = 8 quads = an 8x4-pixel tile; a block of 128 = 4 warps = a 16x8 tile.
// PREC: precision (0 double, 1 float, 2 mixed); MODEL: -1 = thin disk (surface), 0 = layer with thickness, 1 = torus, 2 = GRMHD.
// The host chooses the instantiation (see launch_kernel): each one only contains the path it uses.

#include "interfaces.cuh"

template <int PREC, int MODEL, int METRIC = kMetricKerrFamily>
__global__ void kernel(const KP p) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int lq = lane & 3, lx = lq & 1, ly = lq >> 1, quad = lane >> 2;
  const int tilesx = (p.w + 15) / 16;
  const int bx = blockIdx.x % tilesx, by = blockIdx.x / tilesx;
  const int px = bx * 16 + (warp & 1) * 8 + (quad & 3) * 2 + lx;
  const int py = by * 8 + (warp >> 1) * 4 + (quad >> 2) * 2 + ly;
  const bool inside = px < p.w && py < p.h;

  double jx = 0, jy = 0;
  if (p.sample_k > 0) {
    const uint32_t hq = hash3(uint32_t(px >> 1), uint32_t(py >> 1), p.seed);
    const double kx = p.sample_k * 0.7548776662466927 + double(hq & 0xFFFF) / 65536.0;
    const double ky = p.sample_k * 0.5698402909980532 + double(hq >> 16) / 65536.0;
    jx = kx - floor(kx) - 0.5;
    jy = ky - floor(ky) - 0.5;
  }

  double3 col = make_double3(0, 0, 0);
  double3 base = make_double3(0, 0, 0);  // v0.9a2: emission of the disk layers already summed; trans = remaining transmittance
  double trans = 1.0;
  float u = 0.f, v = 0.f;
  bool sky_ok = false;
  int sky_id = 0;  // which sky the ray ends on (the wormhole has a 2nd one behind the throat)
  int steps = 0;
  if (inside) {
    const double x = (2.0 * ((px + 0.5 + jx) / p.w) - 1.0) * p.tan_half * p.aspect;
    const double y = (1.0 - 2.0 * ((py + 0.5 + jy) / p.h)) * p.tan_half;
    const double rout = p.nodisk ? 0.0 : p.rout;
    const double atol = p.tol * 0.1;
    // moving observer: n' (camera basis) in the observer frame -> n in the ZAMO (aberration)
    if constexpr (METRIC == kMetricWormhole) {
      double3 vdir; double dop;
      WormholeObserver::ray(p, x, y, vdir, dop);
      const SkyHit h = WormholeMetric::sky(p, vdir);
      u = h.u; v = h.v; sky_id = h.sky_id; sky_ok = p.use_sky; steps = 1;
    } else {
      double3 nz; double dop;
      KerrZamoObserver::ray(p, x, y, nz, dop);  // moving observer: n' (camera basis) in the observer frame -> n in the ZAMO (aberration)
      bk::Trace tr;
      if constexpr (MODEL >= 0) VolumeDisk<PREC, MODEL>::shade(p, nz, dop, rout, atol, tr, base, trans, col, steps);
      else ThinDisk<PREC>::shade(p, nz, dop, rout, atol, tr, base, trans, col, steps);
      if (!tr.disk_hit && trans > 0.0 && tr.fate == 0 && p.use_sky) {
        const double ph = tr.phi_end + p.phi;  // axisymmetry: the camera azimuth only adds
        double uu = 0.5 - ph / (2.0 * M_PI);
        uu -= floor(uu);
        u = float(uu);
        v = float(acos(tr.mu_end) / M_PI);
        sky_ok = true;
      }
    }
  }
  __syncwarp();
  if (p.use_sky) {  // uniform flow: all threads of the warp execute the shuffles
    const unsigned full = 0xffffffffu;
    const float uH = __shfl_xor_sync(full, u, 1), vH = __shfl_xor_sync(full, v, 1);
    const float uV = __shfl_xor_sync(full, u, 2), vV = __shfl_xor_sync(full, v, 2);
    const int sH = __shfl_xor_sync(full, sky_id, 1), sV = __shfl_xor_sync(full, sky_id, 2);
    const int okH = __shfl_xor_sync(full, int(sky_ok), 1) && sH == sky_id, okV = __shfl_xor_sync(full, int(sky_ok), 2) && sV == sky_id;  // neighbour from another sky: no gradient
    if (sky_ok) {
      const float sx = lx ? -1.f : 1.f, sy = ly ? -1.f : 1.f;  // the neighbour is at +1 if lx == 0, at -1 if lx == 1
      auto wrap = [](float d) { return d - rintf(d); };
      float2 gx = okH ? make_float2(sx * wrap(uH - u), sx * (vH - v)) : make_float2(0.f, 0.f);
      float2 gy = okV ? make_float2(sy * wrap(uV - u), sy * (vV - v)) : make_float2(0.f, 0.f);
      const cudaTextureObject_t tsel = (sky_id == 1 && p.sky2) ? p.sky2 : p.sky;
      const float4 s = p.sky_raw ? tex2DLod<float4>(tsel, u, v, 0.f) : tex2DGrad<float4>(tsel, u, v, gx, gy);
      col = trans == 1.0 ? make_double3(p.sky_exposure * s.x, p.sky_exposure * s.y, p.sky_exposure * s.z)
                         : make_double3(base.x + trans * p.sky_exposure * s.x, base.y + trans * p.sky_exposure * s.y, base.z + trans * p.sky_exposure * s.z);
    }
  }
  if (p.stats && inside && steps >= p.max_steps) { atomicAdd(&p.stats[5], 1ull); atomicMax(&p.stats[4], (unsigned long long)(py * p.w + px) + 1ull); }  // a ray that never terminated (diagnostic: the last such pixel)
  if (p.stats) {
    const int wm = __reduce_max_sync(0xffffffffu, steps);
    const int ws = __reduce_add_sync(0xffffffffu, steps);
    if (lane == 0) {
      atomicAdd(&p.stats[0], (unsigned long long)ws);
      atomicAdd(&p.stats[1], (unsigned long long)wm * 32ull);
      atomicMax(&p.stats[2], (unsigned long long)wm);
    }
  }
  if (!inside) return;
  // v2.7: last line of defence: a non-finite value (a single NaN contaminates the bloom pyramid and the glare and ruins the whole frame) becomes black; counted in stats[3] for the audit
  if (!(isfinite(col.x) && isfinite(col.y) && isfinite(col.z))) { col = make_double3(0, 0, 0); if (p.stats) atomicAdd(&p.stats[3], 1ull); }
  double L = 0.2126 * col.x + 0.7152 * col.y + 0.0722 * col.z;
  // per-sample luminance clamp (fireflies on the blue side of the Doppler); 0 = off
  if (p.clamp_lum > 0 && L > p.clamp_lum) { const double s = p.clamp_lum / L; col.x *= s; col.y *= s; col.z *= s; L = p.clamp_lum; }
  if (p.alb) {  // albedo guide for the denoiser: normalised colour (hue/saturation, no brightness); 0 where there is no light
    const double m = fmax(col.x, fmax(col.y, col.z));
    const float k = m > 1e-6 ? float(1.0 / m) : 0.f;
    float4& al = p.alb[size_t(py) * p.w + px];
    al.x += float(col.x) * k; al.y += float(col.y) * k; al.z += float(col.z) * k;
  }
  // "ldr" mode: Reinhard per sample before accumulating (= v0.6a/b and the CPU SSAA); otherwise accumulates HDR
  if (p.ldr) { const double t = 1.0 / (1.0 + L); col.x *= t; col.y *= t; col.z *= t; }
  float4& acc = p.accum[size_t(py) * p.w + px];
  acc.x += float(col.x); acc.y += float(col.y); acc.z += float(col.z);
}

// ---------------------------------------------------------------- resolve: media, bloom, tone mapping
constexpr int kBloomLevels = 6;

__global__ void k_avg(const float4* acc, float4* hdr, int n, float inv) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float4 a = acc[i];
  hdr[i] = make_float4(a.x * inv, a.y * inv, a.z * inv, 1.f);
}
__device__ __forceinline__ float4 ld(const float4* t, int w, int h, int x, int y) {
  return t[size_t(min(max(y, 0), h - 1)) * w + min(max(x, 0), w - 1)];
}
// 2x2 (box) reduction of the previous level
__global__ void k_down(const float4* src, int sw, int sh, float4* dst, int dw, int dh) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dw || y >= dh) return;
  float4 a = ld(src, sw, sh, 2 * x, 2 * y), b = ld(src, sw, sh, 2 * x + 1, 2 * y);
  float4 c = ld(src, sw, sh, 2 * x, 2 * y + 1), d = ld(src, sw, sh, 2 * x + 1, 2 * y + 1);
  dst[size_t(y) * dw + x] = make_float4(0.25f * (a.x + b.x + c.x + d.x), 0.25f * (a.y + b.y + c.y + d.y), 0.25f * (a.z + b.z + c.z + d.z), 1.f);
}
// separable Gaussian [1 4 6 4 1]/16
__global__ void k_blur(const float4* src, float4* dst, int w, int h, int dx, int dy) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  const float wt[5] = {1.f / 16, 4.f / 16, 6.f / 16, 4.f / 16, 1.f / 16};
  float3 s = make_float3(0, 0, 0);
  for (int k = -2; k <= 2; ++k) {
    const float4 t = ld(src, w, h, x + k * dx, y + k * dy);
    s.x += wt[k + 2] * t.x; s.y += wt[k + 2] * t.y; s.z += wt[k + 2] * t.z;
  }
  dst[size_t(y) * w + x] = make_float4(s.x, s.y, s.z, 1.f);
}
struct BloomPtrs {
  const float4* lv[kBloomLevels];
  int lw[kBloomLevels], lh[kBloomLevels];
};
__device__ float3 bilerp(const float4* t, int w, int h, float fx, float fy) {  // fx, fy in texels (centres at integers)
  const float x0 = floorf(fx), y0 = floorf(fy), tx = fx - x0, ty = fy - y0;
  const int ix = int(x0), iy = int(y0);
  const float4 a = ld(t, w, h, ix, iy), b = ld(t, w, h, ix + 1, iy), c = ld(t, w, h, ix, iy + 1), d = ld(t, w, h, ix + 1, iy + 1);
  return make_float3((1 - ty) * ((1 - tx) * a.x + tx * b.x) + ty * ((1 - tx) * c.x + tx * d.x),
                     (1 - ty) * ((1 - tx) * a.y + tx * b.y) + ty * ((1 - tx) * c.y + tx * d.y),
                     (1 - ty) * ((1 - tx) * a.z + tx * b.z) + ty * ((1 - tx) * c.z + tx * d.z));
}
// ACES, Narkowicz fit (2016); input already multiplied by the exposure
__device__ float aces(float x) { return fminf(1.f, fmaxf(0.f, (x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f))); }

// ---- glare: broad, star-shaped glow (3 streaks at 0/60/120 degrees, both directions = 6 points) from the very
// bright pixels. Step 1: extract the part above the threshold (soft, per luminance component) at half resolution.
__global__ void k_bright(const float4* hdr, int w, int h, float4* g0, int gw, int gh, float thr, float cap, const float4* coarse, int cw, int ch) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= gw || y >= gh) return;
  float3 acc = make_float3(0, 0, 0);
  for (int dy = 0; dy < 2; ++dy)
    for (int dx = 0; dx < 2; ++dx) {
      const int px = 2 * x + dx, py = 2 * y + dy;
      const float4 c = ld(hdr, w, h, px, py);
      // "top-hat": pixel minus the coarse level of the pyramid (1/8 of the resolution): only the compact points remain (stars,
      // ring core), not the broad bright areas of the disk
      const float3 m = bilerp(coarse, cw, ch, (px + 0.5f) * cw / w - 0.5f, (py + 0.5f) * ch / h - 0.5f);
      float3 d = make_float3(fmaxf(0.f, c.x - m.x), fmaxf(0.f, c.y - m.y), fmaxf(0.f, c.z - m.z));
      float L = 0.2126f * d.x + 0.7152f * d.y + 0.0722f * d.z;
      if (cap > 0.f && L > cap) { const float w = fmaxf(0.f, 2.f - L / cap); d.x *= w; d.y *= w; d.z *= w; L *= w; }  // source brighter than a star (photon ring, disk core): leaves the glare (smooth between cap and 2 cap)
      const float k = L > thr ? (L - thr) / L : 0.f;
      acc.x += k * d.x; acc.y += k * d.y; acc.z += k * d.z;
    }
  g0[size_t(y) * gw + x] = make_float4(0.25f * acc.x, 0.25f * acc.y, 0.25f * acc.z, 1.f);
}
__device__ __forceinline__ float3 ldz(const float4* t, int w, int h, float fx, float fy) {  // bilinear, zero outside
  const float x0 = floorf(fx), y0 = floorf(fy), tx = fx - x0, ty = fy - y0;
  float3 r = make_float3(0, 0, 0);
  for (int j = 0; j < 2; ++j)
    for (int i = 0; i < 2; ++i) {
      const int ix = int(x0) + i, iy = int(y0) + j;
      if (ix < 0 || iy < 0 || ix >= w || iy >= h) continue;
      const float wt = (i ? tx : 1 - tx) * (j ? ty : 1 - ty);
      const float4 c = t[size_t(iy) * w + ix];
      r.x += wt * c.x; r.y += wt * c.y; r.z += wt * c.z;
    }
  return r;
}
// Step 2: streaks with exponential decay exp(-d/tau); each point integrates 1/6 of the energy (normalised sum of weights).
__global__ void k_streaks(const float4* g0, int gw, int gh, float4* g1, float tau, int taps) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= gw || y >= gh) return;
  float3 acc = make_float3(0, 0, 0);
  for (int dir = 0; dir < 3; ++dir) {
    const float ang = dir * 1.0471975512f, cx = cosf(ang), cy = sinf(ang);
    for (int sgn = -1; sgn <= 1; sgn += 2)
      for (int d = 1; d <= taps; ++d) {
        const float wt = 0.5f * __expf(-d / tau);  // intensity next to the point ~ glare x excess (not normalised by energy)
        const float3 v = ldz(g0, gw, gh, x + sgn * d * cx, y + sgn * d * cy);
        acc.x += wt * v.x; acc.y += wt * v.y; acc.z += wt * v.z;
      }
  }
  g1[size_t(y) * gw + x] = make_float4(acc.x, acc.y, acc.z, 1.f);
}

struct Polish {  // only with --cinematic / explicit flags; everything at 0 = identical to v0.7d
  float glare, grain, vignette, chroma;
  int frame;
  uint32_t seed;
  const float4* g1;
  int gw, gh;
  int hdr; float hdr_nits, hdr_peak;  // v2.5: HDR10 output (PQ, BT.2020), see k_final
};

__global__ void k_final(const float4* hdr, uchar4* out, ushort4* out16, int w, int h, BloomPtrs bp, int levels, float bloom, int ldr, int tonemap, Polish po) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  float4 c4 = hdr[size_t(y) * w + x];
  float3 c = make_float3(c4.x, c4.y, c4.z);
  // radial distance normalised to the centre (0 centre, 1 corner)
  const float hx = 0.5f * w, hy = 0.5f * h, hd = sqrtf(hx * hx + hy * hy);
  const float rx = (x + 0.5f - hx), ry = (y + 0.5f - hy), rn = sqrtf(rx * rx + ry * ry) / hd;
  if (po.chroma > 0.f) {  // radial chromatic aberration: R sampled further out, B further in
    const float kr = 1.f + po.chroma * rn, kb = 1.f - po.chroma * rn;
    const float3 r = bilerp(hdr, w, h, hx + rx * kr - 0.5f, hy + ry * kr - 0.5f);
    const float3 b = bilerp(hdr, w, h, hx + rx * kb - 0.5f, hy + ry * kb - 0.5f);
    c.x = r.x; c.z = b.z;
  }
  if (bloom > 0.f && levels > 0) {
    float3 sum = make_float3(0, 0, 0);
    for (int l = 0; l < levels; ++l) {
      const float fx = (x + 0.5f) * bp.lw[l] / w - 0.5f, fy = (y + 0.5f) * bp.lh[l] / h - 0.5f;
      const float3 b = bilerp(bp.lv[l], bp.lw[l], bp.lh[l], fx, fy);
      sum.x += b.x; sum.y += b.y; sum.z += b.z;
    }
    const float k = bloom / levels;
    c.x += k * sum.x; c.y += k * sum.y; c.z += k * sum.z;
  }
  if (po.glare > 0.f && po.g1) {
    const float3 g = ldz(po.g1, po.gw, po.gh, (x + 0.5f) * po.gw / w - 0.5f, (y + 0.5f) * po.gh / h - 0.5f);
    c.x += po.glare * g.x; c.y += po.glare * g.y; c.z += po.glare * g.z;
  }
  if (po.vignette > 0.f) {  // soft: 1 at the centre, falls off from rn ~ 0.35
    const float t = fminf(1.f, fmaxf(0.f, (rn - 0.35f) / 0.75f)), s = t * t * (3.f - 2.f * t);
    const float v = 1.f - po.vignette * s;
    c.x *= v; c.y *= v; c.z *= v;
  }
  if (po.hdr) {  // v2.5: HDR10: linear light (Rec.709, 1 = hdr_nits cd/m2) -> soft luminance shoulder up to hdr_peak -> BT.2020 -> PQ (ST 2084); no display tone mapping
    float3 nt = make_float3(c.x * po.hdr_nits, c.y * po.hdr_nits, c.z * po.hdr_nits);
    const float Y = 0.2126f * nt.x + 0.7152f * nt.y + 0.0722f * nt.z, Y0 = 0.4f * po.hdr_peak, W = po.hdr_peak - Y0;
    if (Y > Y0) { const float s = (Y0 + W * (1.f - __expf(-(Y - Y0) / W))) / Y; nt.x *= s; nt.y *= s; nt.z *= s; }
    const float3 b2 = make_float3(0.6274f * nt.x + 0.3293f * nt.y + 0.0433f * nt.z, 0.0691f * nt.x + 0.9195f * nt.y + 0.0114f * nt.z, 0.0164f * nt.x + 0.0880f * nt.y + 0.8956f * nt.z);
    auto pq = [](float nits) {
      const float y = powf(fminf(1.f, fmaxf(0.f, nits * 1e-4f)), 0.1593017578125f);
      return powf((0.8359375f + 18.8515625f * y) / (1.f + 18.6875f * y), 78.84375f);
    };
    float3 e = make_float3(pq(b2.x), pq(b2.y), pq(b2.z));
    if (po.grain > 0.f) {
      const uint32_t h1 = hash3(uint32_t(x), uint32_t(y), po.seed ^ (uint32_t(po.frame) * 0x9E3779B1u));
      const float n = (float(h1 & 0xFFFF) + float(h1 >> 16)) / 65536.f - 1.f;
      const float amp = 0.5f * po.grain * (0.35f + 0.65f * 4.f * e.y * (1.f - e.y));
      e.x += amp * n; e.y += amp * n; e.z += amp * n;
    }
    auto q8 = [](float v) { return (unsigned char)(fminf(1.f, fmaxf(0.f, v)) * 255.f + 0.5f); };
    out[size_t(y) * w + x] = make_uchar4(q8(e.x), q8(e.y), q8(e.z), 255);
    if (out16) {
      auto q16 = [](float v) { return (unsigned short)(fminf(1.f, fmaxf(0.f, v)) * 65535.f + 0.5f); };
      out16[size_t(y) * w + x] = make_ushort4(q16(e.x), q16(e.y), q16(e.z), 65535);
    }
    return;
  }
  if (!ldr) {
    if (tonemap == 0) {  // Reinhard on the luminance (as in v0.6a/b, but after the average)
      const float L = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z, t = 1.f / (1.f + L);
      c.x *= t; c.y *= t; c.z *= t;
    } else {  // per-channel ACES, exposure 0.6 (Narkowicz fit convention)
      c.x = aces(0.6f * c.x); c.y = aces(0.6f * c.y); c.z = aces(0.6f * c.z);
    }
  }
  float3 e = make_float3(powf(fminf(1.f, fmaxf(0.f, c.x)), 1.f / 2.2f), powf(fminf(1.f, fmaxf(0.f, c.y)), 1.f / 2.2f), powf(fminf(1.f, fmaxf(0.f, c.z)), 1.f / 2.2f));
  if (po.grain > 0.f) {  // film grain AFTER tone mapping: triangular noise, larger amplitude in the mid-tones; fixed seed + frame
    const uint32_t h1 = hash3(uint32_t(x), uint32_t(y), po.seed ^ (uint32_t(po.frame) * 0x9E3779B1u));
    const float n = (float(h1 & 0xFFFF) + float(h1 >> 16)) / 65536.f - 1.f;  // [-1,1] triangular
    const float lum = 0.2126f * e.x + 0.7152f * e.y + 0.0722f * e.z;
    const float amp = po.grain * (0.35f + 0.65f * 4.f * lum * (1.f - lum));
    e.x += amp * n; e.y += amp * n; e.z += amp * n;
  }
  auto q = [](float v) { return (unsigned char)(fminf(1.f, fmaxf(0.f, v)) * 255.f + 0.5f); };
  out[size_t(y) * w + x] = make_uchar4(q(e.x), q(e.y), q(e.z), 255);
  if (out16) {
    auto q16 = [](float v) { return (unsigned short)(fminf(1.f, fmaxf(0.f, v)) * 65535.f + 0.5f); };
    out16[size_t(y) * w + x] = make_ushort4(q16(e.x), q16(e.y), q16(e.z), 65535);
  }
}

// ---- HUD: monospaced text (embedded 7x13 bitmap font, see tools/gen_hud_font.py) drawn on the final frame
__constant__ uint8_t c_font[95 * 13];
__global__ void k_hud(uchar4* out, int w, int h, const char* grid, int cols, int rows, int scale, int x0, int y0, int pad) {
  const int rw = cols * hud::kGlyphW * scale + 2 * pad, rh = rows * hud::kGlyphH * scale + 2 * pad;
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= rw || y >= rh) return;
  const int ox = x0 + x, oy = y0 + y;
  if (ox >= w || oy >= h) return;
  uchar4 px = out[size_t(oy) * w + ox];
  // translucent background
  px.x = (unsigned char)(px.x * 0.35f); px.y = (unsigned char)(px.y * 0.35f); px.z = (unsigned char)(px.z * 0.35f);
  const int gx = x - pad, gy = y - pad;
  if (gx >= 0 && gy >= 0) {
    const int cx = gx / (hud::kGlyphW * scale), cy = gy / (hud::kGlyphH * scale);
    if (cx < cols && cy < rows) {
      const int ch = grid[cy * cols + cx];
      if (ch >= 32 && ch <= 126) {
        const int ix = (gx / scale) % hud::kGlyphW, iy = (gy / scale) % hud::kGlyphH;
        if ((c_font[(ch - 32) * 13 + iy] >> (hud::kGlyphW - 1 - ix)) & 1) {
          const bool warn = grid[cy * cols] == '!';  // lines starting with '!' (warning) in red; the '!' is not drawn
          if (warn && cx == 0) { out[size_t(oy) * w + ox] = px; return; }
          px = warn ? make_uchar4(255, 90, 80, 255) : make_uchar4(235, 235, 235, 255);
        }
      }
    }
  }
  out[size_t(oy) * w + ox] = px;
}

// --- test tools
__global__ void cam_kernel(KP p, const double* xy, int n, double* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double3 d = camera_ray(p, xy[2 * i], xy[2 * i + 1]);
  out[3 * i] = d.x; out[3 * i + 1] = d.y; out[3 * i + 2] = d.z;
}
__global__ void finite_kernel(const float4* a, int n, int* bad) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && !(isfinite(a[i].x) && isfinite(a[i].y) && isfinite(a[i].z))) atomicAdd(bad, 1);
}
__global__ void nt_kernel(KP p, const double* r, int n, double* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = nt_temp(p, r[i]);
}
__global__ void mu_kernel(double a, const double* rLQ, int n, double* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = emitter_mu_d(a, rLQ[3 * i], rLQ[3 * i + 1], rLQ[3 * i + 2]);
}
__global__ void vol_flight_kernel(KP p, const double* rays, int n, double* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  bk::Trace tr;
  VolAcc vol(p, 1.0);
  trace_dispatch_vol(p, rays[3 * i], rays[3 * i + 1], rays[3 * i + 2], p.rout, p.tol * 0.1, tr, vol);
  out[i] = double(vol.T_half);
}
__global__ void grm_sample_kernel(KP p, const double* in, int n, float* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  GrmPt o;
  const bool ok = grmhd_sample(p, float(in[3 * i]), float(in[3 * i + 1]), float(in[3 * i + 2]), o);
  float* q = out + 12 * i;
  if (!ok) { for (int k = 0; k < 12; ++k) q[k] = nanf(""); return; }
  q[0] = o.rho; q[1] = o.thI; q[2] = o.absb; q[3] = o.beta;
  for (int k = 0; k < 4; ++k) { q[4 + k] = o.u[k]; q[8 + k] = o.b[k]; }
}
__global__ void grm_map_kernel(KP p, const double* in, int n, float* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float X1 = logf(float(in[2 * i])); float X2;
  f_inv_theta(p, X1, acosf(fminf(1.f, fmaxf(-1.f, float(in[2 * i + 1])))), X2);
  out[2 * i] = X1; out[2 * i + 1] = X2;
}
__global__ void grm_null_kernel(KP p, const double* rays, int n, double* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  bk::Trace tr;
  VolAcc vol(p, 1.0);
  trace_dispatch_vol(p, rays[3 * i], rays[3 * i + 1], rays[3 * i + 2], p.rout, p.tol * 0.1, tr, vol);
  out[i] = double(vol.null_err);
}
__global__ void radio_kernel(KP p, const double* rays, int n, double* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  bk::Trace tr;
  VolAcc vol(p, 1.0);
  trace_dispatch_vol(p, rays[3 * i], rays[3 * i + 1], rays[3 * i + 2], p.rout, p.tol * 0.1, tr, vol);
  out[i] = vol.Irad;
}
// v2.2b3: radio image in cgs intensity (regular ss x ss average per pixel) + celestial coordinates (alpha, beta) in M of the pixel centre
__global__ void radio_image_kernel(KP p, int ss, double* Iout, double* about) {
  const int px = blockIdx.x * blockDim.x + threadIdx.x, py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= p.w || py >= p.h) return;
  double acc = 0.0;
  for (int sy = 0; sy < ss; ++sy)
    for (int sx = 0; sx < ss; ++sx) {
      const double x = (2.0 * ((px + (sx + 0.5) / ss) / p.w) - 1.0) * p.tan_half * p.aspect;
      const double y = (1.0 - 2.0 * ((py + (sy + 0.5) / ss) / p.h)) * p.tan_half;
      double3 nz; double dop;
      observer_to_zamo(camera_ray(p, x, y), p.bx, p.by, p.bz, nz, dop);
      bk::Trace tr;
      VolAcc vol(p, dop);
      trace_dispatch_vol(p, nz.x, nz.y, nz.z, p.rout, p.tol * 0.1, tr, vol);
      acc += vol.Irad;
    }
  Iout[py * p.w + px] = acc / (ss * ss);
  const double x = (2.0 * ((px + 0.5) / p.w) - 1.0) * p.tan_half * p.aspect;
  const double y = (1.0 - 2.0 * ((py + 0.5) / p.h)) * p.tan_half;
  double3 nz; double dop;
  observer_to_zamo(camera_ray(p, x, y), p.bx, p.by, p.bz, nz, dop);
  const bk::Init o = bk::init_from_zamo(p.a, p.dist, p.theta, nz.x, nz.y, nz.z, p.q2, p.lam);
  const double s = sin(p.theta);
  about[2 * (py * p.w + px)] = -o.L / s;  // alpha = -L/(E sin theta_o); init_from_zamo already returns L, p_theta with E = 1
  about[2 * (py * p.w + px) + 1] = -o.ptheta;                          // beta = -p_theta (sign: theta grows downwards in the image); only |.| counts for the area
}
__global__ void torus_kernel(KP p, const double* in, int n, TorusPoint* out, bool fast) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  TorusPt o{};
  const bool ok = p.model == 3 ? (fast ? riaf_emit<float>(p, in[3 * i], in[3 * i + 1], in[3 * i + 2], 1.0, o) : riaf_emit<double>(p, in[3 * i], in[3 * i + 1], in[3 * i + 2], 1.0, o))
                               : (fast ? torus_emit<float>(p, in[3 * i], in[3 * i + 1], in[3 * i + 2], 1.0, o) : torus_emit<double>(p, in[3 * i], in[3 * i + 1], in[3 * i + 2], 1.0, o));
  out[i] = ok ? TorusPoint{o.rho, o.thE, o.B, o.sinB, o.j, o.alpha, o.g, o.e_emit} : TorusPoint{0, 0, 0, 0, 0, 0, 0, 0};
}
__global__ void ntkn_kernel(KP p, const double* in, int n, double* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[2 * i] = nt_temp(p, in[2 * i]);
  out[2 * i + 1] = redshift(p.a, in[2 * i], in[2 * i + 1], p.dist, p.theta, p.q2);
}
__global__ void wh_kernel(KP p, const double* alpha, int n, double* psi, int* side) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float ps; int sd;
  wh_trace(p, float(cos(alpha[i])), float(sin(alpha[i])), sd, ps);
  psi[i] = ps; side[i] = sd;
}
__global__ void turb_kernel(KP p, const double* in, int n, double* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float r = float(in[3 * i]), phi = float(in[3 * i + 1]), mu = float(in[3 * i + 2]);
  const float h = p.f.v_hr * (p.v_flare == 0.0 ? 1.f : powf(r * 0.1f, p.f.v_flare));
  const float r32 = r * sqrtf(r);
  out[2 * i] = turb_density_f(p, r, phi, mu, h, 1.f / (r32 + p.f.a), p.f.disk_time);
  out[2 * i + 1] = spot_factor_f(p, r, phi, mu, p.f.disk_time);
}
__global__ void boost_kernel(const double* np, double bx, double by, double bz, int n, double* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double3 nz; double dop;
  observer_to_zamo(make_double3(np[3 * i], np[3 * i + 1], np[3 * i + 2]), bx, by, bz, nz, dop);
  out[4 * i] = nz.x; out[4 * i + 1] = nz.y; out[4 * i + 2] = nz.z; out[4 * i + 3] = dop;
}

__global__ void batch_kernel(KP p, const double* rays, int n, bk::Trace* out, int skip) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double rout = p.nodisk ? 0.0 : p.rout;
  trace_dispatch(p, rays[3 * i], rays[3 * i + 1], rays[3 * i + 2], rout, p.tol * 0.1, out[i], skip);
}

// ------------------------------------------------------------------ host

// O5: the texture data of one snapshot in the layout of the 3 float4 (or half4) arrays, packed on the CPU. 11 of the 17.5 ms of a snapshot change were this
// strided gather on the render thread (BHS_TIMING); it is now done by the prefetch worker (or by 3 threads when there is no prefetch).
struct GrmStage {
  std::vector<float4> f4[3]; std::vector<ushort4> h4[3]; bool half = false, valid = false;
};
struct LpKey { double a = -1, h = -1, rin = -1, rtab = -1, rdisk = -1, gamma = -1; int ns = 0; };  // what a lamppost table depends on

struct State {
  float* lut = nullptr;
  std::vector<float> lutd, lut1;  // lutd: LUT of the current f_col; lut1: reference f = 1 (physical exposure)
  double lut_fcol = 1.0;
  bool grm_half = false;  // v2.2f-d: half4 textures
  cudaArray_t grm_arr[3] = {nullptr, nullptr, nullptr};  // v2.2f: GRMHD snapshot
  cudaTextureObject_t grm_tex[3] = {0, 0, 0};
  cudaArray_t grm_arr2[3] = {nullptr, nullptr, nullptr};  // v4.0: second snapshot (t + dt) for the time interpolation
  cudaTextureObject_t grm_tex2[3] = {0, 0, 0};
  bool grm_next = false;
  // O5b: a third texture set (the spare) receives the prefetched snapshot asynchronously, from pinned staging, on its own stream while the current frames render: the swap on the
  // render thread is then a pointer rotation instead of free + malloc3D + a 6.8 ms pageable memcpy3D. +226 MB VRAM (one set) and +226 MB pinned host memory.
  cudaArray_t grm_arr3[3] = {nullptr, nullptr, nullptr}; cudaTextureObject_t grm_tex3[3] = {0, 0, 0};
  cudaStream_t grm_stream = nullptr; cudaEvent_t grm_ev = nullptr, grm_free_ev = nullptr;
  void* grm_pin[3] = {nullptr, nullptr, nullptr}; size_t grm_pin_bytes = 0; int sp_n1 = 0, sp_n2 = 0, sp_n3 = 0; bool sp_half = false;  // the spare's grid / format
  bool pf_async = false;  // the prefetch in flight copies into the spare (else: it packs a GrmStage for the synchronous upload)
  std::future<bool> pf_fut; std::string pf_path, pf_err; grmhd::Field pf_field; bool pf_valid = false; GrmStage pf_stage;  // v4.0: next snapshot parsed on a worker thread
  double grm_tau = -1.0, grm_dtM = 2.0, grm_pat = 0.2;  // tau < 0: no blending; pat = pattern speed / Keplerian
  cudaArray_t grm_theta_arr = nullptr;
  cudaTextureObject_t grm_theta_tex = 0;
  grmhd::Header grm_h;
  bool grm_loaded = false;
  cudaMipmappedArray_t mm = nullptr, mm2 = nullptr;  // mm2/sky2: 2nd sky (wormhole)
  cudaTextureObject_t sky = 0, sky2 = 0;
  bool has_sky = false;
  uchar4* out = nullptr;
  ushort4* out16 = nullptr;  // v1.0: the same final frame at 16 bit per channel (10-bit offline encode pipeline)
  float4* accum = nullptr;  // sum of the samples (RGBA32F)
  float4* hdr = nullptr;    // media
  float4 *hdr2 = nullptr, *alb = nullptr, *alb_avg = nullptr;  // denoiser output; accumulated and averaged albedo
  const float4* hdr_last = nullptr;  // HDR buffer used in the last resolve
  bool den_ready = false, den_albedo = false, den_failed = false;
  float4* lv[kBloomLevels] = {};  // bloom pyramid
  float4* tmp = nullptr;
  float4 *g0 = nullptr, *g1 = nullptr;  // glare (half resolution)
  int glw = 0, glh = 0;
  int lw[kBloomLevels] = {}, lh[kBloomLevels] = {};
  int samples = 0;  // accumulated samples
  float* nt_tab = nullptr;
  int nt_n = 0;
  double nt_a = -1, nt_q2 = 0, nt_rin = 0, nt_rout = 0;
  float* lp_tab = nullptr;  // E4: lamppost table + cache key
  int lp_n = 0;
  std::future<lamppost::LogTable> lp_fut; bool lp_busy = false; LpKey lp_key, lp_pkey;  // current table's key; key of the background build
  double lp_lmin = 0, lp_lmax = 0, lp_slope = -3, lp_rgeo = 0;
  unsigned long long* stats = nullptr;
  int w = 0, h = 0, cw = 0, ch = 0;  // logical size and allocated capacity
  char* hud_buf = nullptr;
  int hud_cap = 0;
  cudaEvent_t e0 = nullptr, e1 = nullptr, e2 = nullptr, e3 = nullptr, e4 = nullptr, e5 = nullptr;
  Timing last{};
  cudaGraphicsResource_t gres = nullptr;
  int gw = 0, gh = 0;
} S;

uint16_t f2h(float f) { return __half_as_ushort(__float2half(f)); }
float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }

void ensure_nt(const Params& p, KP& k) {
  // Novikov-Thorne T(r) table in ln r (2048 points, nt::Profile::table()); rebuilt when a (or rout) changes
  const double rout_prof = std::max(2.0 * p.rout, 100.0);
  const double q2c = p.charge * p.charge;
  if (!S.nt_tab || std::abs(p.a - S.nt_a) > 1e-9 || std::abs(q2c - S.nt_q2) > 1e-12 || std::abs(rout_prof - S.nt_rout) > 1e-9) {
    std::vector<float> tab; double rin_t = 0;
    if (q2c > 0.0) { const ntk::Profile prof(p.a, q2c, rout_prof); tab.assign(prof.table().begin(), prof.table().end()); rin_t = prof.r_in(); }  // Kerr-Newman (v2.1e)
    else { const nt::Profile prof(p.a, rout_prof); tab.assign(prof.table().begin(), prof.table().end()); rin_t = prof.r_in(); }
    if (!S.nt_tab) CK(cudaMalloc(&S.nt_tab, 2048 * sizeof(float)));
    CK(cudaMemcpy(S.nt_tab, tab.data(), tab.size() * sizeof(float), cudaMemcpyHostToDevice));
    S.nt_n = int(tab.size()); S.nt_a = p.a; S.nt_q2 = q2c; S.nt_rout = rout_prof; S.nt_rin = rin_t;
  }
  k.nt_tab = S.nt_tab; k.nt_n = S.nt_n; k.nt_rin = S.nt_rin; k.nt_rout = S.nt_rout;
  if (S.nt_rin > 0.0 && S.nt_rout > S.nt_rin) { k.nt_lmin = std::log(S.nt_rin); k.nt_scale = double(S.nt_n - 1) / (std::log(S.nt_rout) - k.nt_lmin); }
}

// E4: table of the lamppost (log10 eps_E on a ln r grid), rebuilt when its key changes (~0.8 ms per photon on the analytic oracle)
bool lp_same(const LpKey& x, const LpKey& y) {
  auto eq = [](double u, double v, double t) { return std::abs(u - v) <= t; };
  return eq(x.a, y.a, 1e-12) && eq(x.h, y.h, 1e-12) && eq(x.rin, y.rin, 1e-12) && eq(x.rtab, y.rtab, 1e-9) && eq(x.rdisk, y.rdisk, 1e-9) && eq(x.gamma, y.gamma, 1e-12) && x.ns == y.ns;
}
lamppost::LogTable lp_build(const LpKey& q) { return lamppost::log_table(lamppost::emissivity(q.a, q.h, q.rin, q.rtab, q.ns, q.gamma, q.rdisk), 2048); }

void ensure_lp(const Params& p, KP& k) {
  const LpKey want{p.a, p.lp_h, k.rin, std::max(2.0 * p.rout, 100.0), p.rout, p.lp_gamma, p.lp_samples};
  auto install = [&](const lamppost::LogTable& lt, const LpKey& q) {
    if (lt.log10e.empty()) { std::fprintf(stderr, "lamppost: no ray from h = %g reaches the disk (a = %g, r_in = %g)\n", q.h, q.a, q.rin); std::exit(1); }
    if (!S.lp_tab) CK(cudaMalloc(&S.lp_tab, 2048 * sizeof(float)));
    CK(cudaMemcpy(S.lp_tab, lt.log10e.data(), lt.log10e.size() * sizeof(float), cudaMemcpyHostToDevice));
    S.lp_n = int(lt.log10e.size()); S.lp_key = q; S.lp_lmin = lt.lmin; S.lp_lmax = lt.lmax; S.lp_slope = lt.slope_out; S.lp_rgeo = lt.refl_frac;
    std::fprintf(stderr, "lamppost: a = %g, h = %g, Gamma = %g: geometric reflection fraction N_disk/N_inf = %.3f (disk %g..%g)\n", q.a, q.h, q.gamma, lt.refl_frac, q.rin, q.rdisk);
  };
  if (S.lp_busy && S.lp_fut.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {  // the background table is ready: install it
    lamppost::LogTable lt = S.lp_fut.get(); S.lp_busy = false;
    if (!lt.log10e.empty()) install(lt, S.lp_pkey);  // an empty table (no ray reaches the disk): keep the previous one
  }
  const bool stale = !S.lp_tab || !lp_same(want, S.lp_key);
  if (stale && S.lp_busy && lp_same(want, S.lp_pkey)) {
    // the table being built is the one we want: keep drawing the old one
  } else if (stale && p.lp_async && S.lp_tab && !S.lp_busy) {
    S.lp_pkey = want; S.lp_busy = true;
    S.lp_fut = std::async(std::launch::async, [want] { return lp_build(want); });
  } else if (stale && !(p.lp_async && S.lp_tab)) {
    if (S.lp_busy) { S.lp_fut.wait(); S.lp_busy = false; }  // synchronous request while a build runs: drop it, build the one asked for
    install(lp_build(want), want);
  }
  k.lp_tab = S.lp_tab; k.lp_n = S.lp_n; k.lp_lmin = S.lp_lmin; k.lp_lmax = S.lp_lmax; k.lp_slope = S.lp_slope; k.lp_lo = p.lp_lo; k.lp_hi = std::max(p.lp_hi, p.lp_lo + 0.5);
  k.lp_gamma = p.lp_gamma;  // observed intensity ~ g^(2 + Gamma) eps(r); Gamma = 2 is the bolometric view (g^4)
  k.lp_lscale = (p.lp_refl > 0.0 && S.lp_rgeo > 0.0) ? std::log10(p.lp_refl / S.lp_rgeo) : 0.0;  // reflection fraction asked / the geometric one
}

KP make_kp(const Params& p) {
  KP k{};
  static bool lut_scale_set = false;
  if (!lut_scale_set) { const double sc = double(lut::kN - 1) / std::log(kT1 / kT0); CK(cudaMemcpyToSymbol(c_lut_scale, &sc, sizeof sc)); lut_scale_set = true; }
  k.w = p.w; k.h = p.h; k.use_sky = p.use_sky && S.has_sky; k.sky_raw = p.sky_raw; k.nodisk = p.nodisk; k.nodoppler = p.nodoppler; k.noshift = p.noshift; k.uniform_disk = p.uniform_disk; k.smooth_disk = p.smooth_disk;
  k.max_steps = p.max_steps; k.prec = int(p.prec);
  k.a = p.a; k.dist = p.dist; k.theta = std::min(179.999, std::max(0.001, p.theta_deg)) * M_PI / 180.0;  // v2.7: camera exactly on the axis (theta = 0/180) is degenerate (phi undefined): limit 0.001 degrees
  k.phi = p.phi_deg * M_PI / 180.0;
  k.tan_half = std::tan(0.5 * p.fov_deg * M_PI / 180.0);
  k.aspect = double(p.w) / p.h;
  k.rin = p.rin < 0 ? (p.charge > 0 ? ntk::isco(p.a, p.charge * p.charge) : kerr::isco(p.a, true)) : p.rin;  // Kerr-Newman: ISCO = minimum of E(r) (ntk::isco)
  k.rout = p.rout;
  // exposure: black-body luminance at T_peak <-> 0.5 before tone mapping (as in v0.6a)
  if (p.fcol != S.lut_fcol) {  // new LUT for the hardened spectrum (host only + a 12 kB copy)
    S.lutd = lut::build(p.fcol); S.lut_fcol = p.fcol;
    CK(cudaMemcpy(S.lut, S.lutd.data(), S.lutd.size() * sizeof(float), cudaMemcpyHostToDevice));
  }
  // physical exposure: the same reference (f = 1) for any f_col, so that the hardening changes the brightness as in physics
  const Vec3 ref = lut::lookup(p.expose_physical ? S.lut1 : S.lutd, p.tpeak);
  k.exposure = p.exposure * 0.5 / (0.2126 * ref.x + 0.7152 * ref.y + 0.0722 * ref.z);
  k.sky_exposure = p.sky_exposure;
  k.phi_atol = p.phi_atol; k.l_min = p.l_min >= 0 ? p.l_min : 1e-4;
  k.tol = p.tol; k.r_far = p.r_far; k.integ = p.integ;
  // O1: the tail must start outside every emitter (10x margin; the torus model itself stops sampling at 4 rmax)
  k.r_tail = p.r_tail > 0 ? std::max({p.r_tail, 10.0 * p.rout, 40.0 * p.torus_rmax, 10.0 * p.riaf_rout, 10.0 * p.grmhd_rcap, p.model == 4 ? 10.0 * p.sph_rout : 0.0, p.model == 5 ? 10.0 * p.cone_rout : 0.0}) : 0.0; k.tpeak = p.tpeak;
  auto f = [&](double x) { return std::pow(x, -3.0) * (1.0 - std::sqrt(k.rin / x)); };
  k.fnorm = f(49.0 / 36.0 * k.rin);
  k.lut = S.lut; k.sky = S.sky; k.sky2 = p.sky2_off ? 0 : S.sky2;
  k.wh = p.wormhole; k.wh_rho = p.wh_rho; k.wh_a = p.wh_a; k.wh_M = p.wh_M; k.wh_lc = p.wh_lc; k.wh_lmax = p.wh_lmax; k.wh_ellis = p.wh_ellis;
  k.nt = p.nt; k.ldr = p.accum_ldr; k.seed = p.seed;
  k.bx = p.beta[0]; k.by = p.beta[1]; k.bz = p.beta[2];
  k.clamp_lum = p.clamp_lum; k.beaming = p.beaming; k.limb = p.limb;
  k.edge_soft = p.edge_soft; k.edge_noise = p.edge_noise;
  {  // v0.9c: spot phase and amplitude at the disk time (on the host, in double, reduced to [0, 2 pi))
    turb::Spot sp; sp.r = p.spot_r; sp.phi0 = p.spot_phi_deg * M_PI / 180.0; sp.amp = p.spot_amp; sp.fl_t = p.flare_t; sp.fl_w = p.flare_w; sp.fl_boost = p.flare_boost;
    k.spot_ph = std::fmod(turb::spot_phase(sp, p.a, p.disk_time), 2.0 * M_PI); k.spot_amp = p.spot_amp; k.spot_om = turb::omega(p.spot_r, p.a);
    k.fl_t = p.flare_t; k.fl_w = p.flare_w; k.fl_boost = p.flare_boost; k.slow = p.slowlight;
    k.arm_amp = p.arm_amp; k.arm_m = p.arm_m; k.arm_cot = 1.0 / std::tan(p.arm_pitch_deg * M_PI / 180.0); k.arm_om = turb::omega(p.arm_rpat, p.a); k.ring_amp = p.ring_amp; k.ring_freq = p.ring_freq; k.ring_gap = p.ring_gap; k.hv_amp = p.hv_amp; k.hv_scale = p.hv_scale;
    k.q2 = p.charge * p.charge; k.lam = p.lambda; k.tex_amp = p.tex_amp; k.tex_fr = p.tex_fr; k.tex_fphi = p.tex_fphi; k.tex_oct = p.tex_oct;
    k.spot_r = p.spot_r; k.spot_size = p.spot_size; k.turb_amp = p.turb; k.turb_scale = p.turb_scale; k.disk_time = p.disk_time;
  }
  k.emis = p.emission; k.the_min = p.theta_e_min; k.model = p.model; k.nu_obs = p.nu_ghz * 1e9; k.mcm = p.mbh * 1.476625e5; k.tb_lo = p.tb_lo; k.tb_hi = p.tb_hi;
  k.r_pn = p.riaf_pn; k.r_pt = p.riaf_pt; k.r_hr = p.riaf_hr; k.r_xi = p.riaf_xi; k.r_rout = p.riaf_rout;
  k.s_flow = p.sph_flow; k.s_nar = p.sph_narayan; k.s_alpha = p.sph_alpha; k.s_rout = p.sph_rout; k.s_lo = p.sph_lo; k.s_hi = p.sph_hi;
  { const double th = p.cone_theta_deg * M_PI / 180.0; k.c_th = th; k.c_mu = std::cos(th); k.c_w = p.cone_w; k.c_wmu = p.cone_w * std::sin(th); k.c_alpha = p.cone_alpha; k.c_rout = p.cone_rout; k.c_lo = p.cone_lo; k.c_hi = p.cone_hi; }
  k.s_rin = p.sph_rin > 0 ? p.sph_rin : p.sph_narayan ? 2.0 : 1.0 + std::sqrt(std::max(0.0, 1.0 - p.a * p.a));
  k.ne0 = p.ne0; k.pbeta = p.plasma_beta; k.ti_te = p.ti_te; k.th_e_max = p.theta_e_max; k.t_step = p.torus_step; k.t_gamma = p.torus_gamma; k.t_rin = p.torus_rin;
  if (p.model == 1) {
    torus::Torus tt;
    if (torus::make(p.a, p.torus_rin, p.torus_rmax, p.torus_gamma, tt)) { k.t_ell = tt.ell; k.t_utin = tt.ut_in; k.t_hmax = tt.h_max; }
    else { k.t_ell = 0; k.t_utin = 1; k.t_hmax = 1.0000001; }  // invalid: the CLI refuses before getting here
  }
  k.t_rmax = p.torus_rmax;
  k.rcheck = p.debug_null;
  if (p.model == 2 && S.grm_loaded) {  // v2.2f: the spin and the grid come from the dump
    const auto& h = S.grm_h;
    k.a = h.a;
    k.g_n1 = int(h.n1); k.g_n2 = int(h.n2); k.g_n3 = int(h.n3); k.g_startx1 = h.startx1; k.g_dx1 = h.dx1;
    k.g_hslope = h.hslope; k.g_ms = h.mks_smooth; k.g_palpha = h.poly_alpha; k.g_pxt = h.poly_xt; k.g_pnorm = h.fmks().poly_norm(); k.g_gam = h.gam; k.g_te_c = eht_theta_e_coef(h.gam);
    for (int i = 0; i < 3; ++i) k.g_tex[i] = S.grm_tex[i];
    k.g_blend = S.grm_next && S.grm_tau >= 0.0; k.g_tau = std::min(1.0, std::max(0.0, S.grm_tau)); k.g_dtM = S.grm_dtM; k.g_pat = S.grm_pat;
    for (int i = 0; i < 3; ++i) k.g_tex2[i] = S.grm_tex2[i];
    k.g_theta_lut = p.theta_lut ? S.grm_theta_tex : 0;
    k.g_rho_unit = p.munit / (k.mcm * k.mcm * k.mcm); k.g_b_unit = synch::kC * std::sqrt(4.0 * M_PI * k.g_rho_unit);
    k.g_step = p.grmhd_step; k.el_model = p.electrons; k.g_rhigh = p.rhigh; k.g_ce = p.th_e_ratio; k.g_rcap = p.grmhd_rcap; k.g_scut = p.grmhd_sigma_cut;
  }
  k.vol = p.hr > 0.0 || p.model >= 1; k.v_hr = p.hr; k.v_flare = p.flare; k.v_tau = p.tau; k.v_sub = std::max(1, p.vol_sub);
  for (int i = 0; i < 3; ++i) { k.fwd[i] = p.fwd[i]; k.right[i] = p.right[i]; k.up[i] = p.up[i]; }
  if (p.nt) ensure_nt(p, k);
  if (p.lp_h > 0.0 && !p.nodisk && p.model == 0 && p.hr == 0.0) {
    k.lp_plunge = p.lp_plunge;
    if (p.lp_plunge) {  // the disk reaches the horizon; E and L of the circular orbit at the ISCO are the plunging gas's constants (the ISCO itself, not --rin: as in lamppost::hit)
      const double ri = kerr::isco(p.a, true), sr = std::sqrt(ri), den = std::pow(ri, 0.75) * std::sqrt(ri * sr - 3.0 * sr + 2.0 * p.a);
      k.lp_risco = ri; k.lp_E = (ri * sr - 2.0 * sr + p.a) / den; k.lp_L = (ri * ri - 2.0 * p.a * sr + p.a * p.a) / den;
      k.rin = 1.0 + std::sqrt(std::max(0.0, 1.0 - p.a * p.a)) + 1e-3;
    }
    ensure_lp(p, k);
  }
  k.accum = S.accum;
  k.f = PF{float(k.a), float(k.arm_amp), float(k.arm_cot), float(k.arm_om), float(k.beaming), float(k.c_alpha), float(k.c_mu), float(k.c_rout), float(k.c_th), float(k.c_w), float(k.c_wmu), float(k.disk_time), float(k.edge_soft), float(k.exposure), float(k.fl_boost), float(k.fl_t), float(k.fl_w), float(k.fnorm), float(k.g_b_unit), float(k.g_ce), float(k.g_dx1), float(k.g_hslope), float(k.g_ms), float(k.g_palpha), float(k.g_pnorm), float(k.g_pxt), float(k.g_rcap), float(k.g_rhigh), float(k.g_scut), float(k.g_startx1), float(k.g_step), float(k.g_tau), float(k.hv_amp), float(k.hv_scale), float(k.mcm), float(k.nt_rin), float(k.nt_rout), float(k.nu_obs), float(k.phi), float(k.r_hr), float(k.r_pn), float(k.r_pt), float(k.r_rout), float(k.r_xi), float(k.rin), float(k.ring_amp), float(k.ring_freq), float(k.ring_gap), float(k.rout), float(k.s_alpha), float(k.s_rin), float(k.s_rout), float(k.spot_amp), float(k.spot_om), float(k.spot_ph), float(k.spot_r), float(k.spot_size), float(k.t_ell), float(k.t_rmax), float(k.t_step), float(k.t_utin), float(k.th_e_max), float(k.the_min), float(k.ti_te), float(k.tpeak), float(k.turb_amp), float(k.turb_scale), float(k.v_flare), float(k.v_hr), float(k.v_tau), float(k.wh_M), float(k.wh_a), float(k.wh_lc), float(k.wh_lmax), float(k.wh_rho)};
  return k;
}

}  // namespace

// builds a sky (EXR half RGB -> texture with mipmaps, 16x anisotropic filtering); used for the main sky and for the wormhole's 2nd sky
static bool build_sky(const std::string& sky_path, cudaMipmappedArray_t& mmo, cudaTextureObject_t& texo, std::string& err) {
  sky::Skymap m;
  if (!sky::load_exr_half(sky_path, m, &err)) return false;
  int levels = 1;
  for (int n = std::max(m.w, m.h); n > 1; n >>= 1) ++levels;
  cudaChannelFormatDesc cd = cudaCreateChannelDescHalf4();
  CK(cudaMallocMipmappedArray(&mmo, &cd, make_cudaExtent(m.w, m.h, 0), levels));
  // level 0: RGB half -> RGBA half; following levels: 2x2 mean in float (linear HDR values)
  std::vector<uint16_t> cur(size_t(m.w) * m.h * 4), nxt;
  for (size_t i = 0; i < size_t(m.w) * m.h; ++i) {
    cur[4 * i] = m.rgb[3 * i]; cur[4 * i + 1] = m.rgb[3 * i + 1]; cur[4 * i + 2] = m.rgb[3 * i + 2]; cur[4 * i + 3] = f2h(1.f);
  }
  m.rgb.clear(); m.rgb.shrink_to_fit();
  int cw = m.w, ch = m.h;
  for (int l = 0; l < levels; ++l) {
    cudaArray_t arr;
    CK(cudaGetMipmappedArrayLevel(&arr, mmo, l));
    CK(cudaMemcpy2DToArray(arr, 0, 0, cur.data(), size_t(cw) * 8, size_t(cw) * 8, ch, cudaMemcpyHostToDevice));
    if (l + 1 == levels) break;
    const int nw = std::max(1, cw / 2), nh = std::max(1, ch / 2);
    nxt.assign(size_t(nw) * nh * 4, 0);
    for (int y = 0; y < nh; ++y)
      for (int x = 0; x < nw; ++x)
        for (int c = 0; c < 4; ++c) {
          float s = 0;
          for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx) s += h2f(cur[(size_t(std::min(ch - 1, 2 * y + dy)) * cw + std::min(cw - 1, 2 * x + dx)) * 4 + c]);
          nxt[(size_t(y) * nw + x) * 4 + c] = f2h(0.25f * s);
        }
    cur.swap(nxt); cw = nw; ch = nh;
  }
  cudaResourceDesc rd{};
  rd.resType = cudaResourceTypeMipmappedArray;
  rd.res.mipmap.mipmap = mmo;
  cudaTextureDesc td{};
  td.addressMode[0] = cudaAddressModeWrap;   // u: right ascension, wraps around
  td.addressMode[1] = cudaAddressModeClamp;  // v: polos
  td.filterMode = cudaFilterModeLinear;
  td.mipmapFilterMode = cudaFilterModeLinear;
  td.readMode = cudaReadModeElementType;
  td.normalizedCoords = 1;
  td.maxAnisotropy = 16;
  td.minMipmapLevelClamp = 0.f;
  td.maxMipmapLevelClamp = float(levels - 1);
  CK(cudaCreateTextureObject(&texo, &rd, &td, nullptr));
  std::printf("sky: %s (%dx%d, %d levels, aniso 16x)\n", sky_path.c_str(), m.w, m.h, levels);
  return true;
}

bool init(const std::string& sky_path, std::string& err) {
  CK(cudaFree(nullptr));
  S.lutd = lut::build();
  S.lut1 = S.lutd; S.lut_fcol = 1.0;
  const std::vector<float>& lutd = S.lutd;
  CK(cudaMalloc(&S.lut, lutd.size() * sizeof(float)));
  CK(cudaMemcpy(S.lut, lutd.data(), lutd.size() * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&S.stats, 8 * sizeof(unsigned long long)));
  CK(cudaEventCreate(&S.e0));
  CK(cudaEventCreate(&S.e1));
  {
    uint8_t f[95 * 13];
    for (int i = 0; i < 95; ++i) for (int j = 0; j < 13; ++j) f[i * 13 + j] = hud::kFont[i][j];
    CK(cudaMemcpyToSymbol(c_font, f, sizeof f));
  }
  CK(cudaEventCreate(&S.e2));
  CK(cudaEventCreate(&S.e4));
  CK(cudaEventCreate(&S.e5));
  CK(cudaEventCreate(&S.e3));
  if (sky_path.empty()) return true;
  if (!build_sky(sky_path, S.mm, S.sky, err)) return false;
  S.has_sky = true;
  return true;
}

bool has_sky() { return S.has_sky; }
bool load_sky2(const std::string& path, std::string& err) { return build_sky(path, S.mm2, S.sky2, err); }
bool denoise_available() { return denoise::compiled(); }
const char* denoise_version() { return denoise::version(); }
bool denoise_enable(const Params& p, std::string& err) {
  if (!denoise::compiled()) { err = "OptiX not available: built without BHS_OPTIX (run code/tools/fetch_optix.sh and rebuild)"; return false; }
  set_size(p.w, p.h);
  if (!denoise::init(S.cw, S.ch, p.denoise_albedo, err)) return false;
  S.den_ready = true; S.den_albedo = p.denoise_albedo; S.den_failed = false;
  return true;
}
bool denoise_image(const std::vector<float>& in, int w, int h, std::vector<float>& out, std::string& err) {
  if (!denoise::init(w, h, false, err)) return false;
  const size_t bytes = size_t(w) * h * 16;
  void *di, *dout;
  CK(cudaMalloc(&di, bytes)); CK(cudaMalloc(&dout, bytes));
  CK(cudaMemcpy(di, in.data(), bytes, cudaMemcpyHostToDevice));
  const bool ok = denoise::run(di, nullptr, dout, w, h, nullptr, err);
  out.resize(in.size());
  if (ok) { CK(cudaDeviceSynchronize()); CK(cudaMemcpy(out.data(), dout, bytes, cudaMemcpyDeviceToHost)); }
  CK(cudaFree(di)); CK(cudaFree(dout));
  S.den_ready = false;  // the next resolve calls init again (the maximum size may have changed)
  return ok;
}
void download_hdr(std::vector<float>& rgba) {
  rgba.resize(size_t(S.w) * S.h * 4);
  CK(cudaMemcpy(rgba.data(), S.hdr_last ? S.hdr_last : S.hdr, rgba.size() * sizeof(float), cudaMemcpyDeviceToHost));
}

void set_size(int w, int h) {
  auto dims = [&] {
    S.w = w; S.h = h;
    int lw = w, lh = h;
    for (int l = 0; l < kBloomLevels; ++l) { lw = std::max(1, (lw + 1) / 2); lh = std::max(1, (lh + 1) / 2); S.lw[l] = lw; S.lh[l] = lh; }
    S.glw = std::max(1, (w + 1) / 2); S.glh = std::max(1, (h + 1) / 2);
  };
  if (S.accum && w <= S.cw && h <= S.ch) { dims(); return; }  // fits in the capacity: no reallocation (variable-scale rendering)
  if (S.out) { CK(cudaFree(S.out)); CK(cudaFree(S.out16)); CK(cudaFree(S.accum)); CK(cudaFree(S.hdr)); CK(cudaFree(S.tmp)); CK(cudaFree(S.g0)); CK(cudaFree(S.g1)); CK(cudaFree(S.hdr2)); CK(cudaFree(S.alb)); CK(cudaFree(S.alb_avg)); for (auto* l : S.lv) CK(cudaFree(l)); }
  S.cw = std::max(w, S.cw); S.ch = std::max(h, S.ch);
  const size_t n = size_t(S.cw) * S.ch;
  CK(cudaMalloc(&S.out, n * sizeof(uchar4)));
  CK(cudaMemset(S.out, 0, n * sizeof(uchar4)));
  CK(cudaMalloc(&S.out16, n * sizeof(ushort4)));
  CK(cudaMemset(S.out16, 0, n * sizeof(ushort4)));
  CK(cudaMalloc(&S.accum, n * sizeof(float4)));
  CK(cudaMemset(S.accum, 0, n * sizeof(float4)));
  CK(cudaMalloc(&S.hdr, n * sizeof(float4)));
  CK(cudaMalloc(&S.hdr2, n * sizeof(float4)));
  CK(cudaMalloc(&S.alb, n * sizeof(float4)));
  CK(cudaMemset(S.alb, 0, n * sizeof(float4)));
  CK(cudaMalloc(&S.alb_avg, n * sizeof(float4)));
  int lw = S.cw, lh = S.ch;
  size_t tmpn = 0;
  for (int l = 0; l < kBloomLevels; ++l) {
    lw = std::max(1, (lw + 1) / 2); lh = std::max(1, (lh + 1) / 2);
    CK(cudaMalloc(&S.lv[l], size_t(lw) * lh * sizeof(float4)));
    tmpn = std::max(tmpn, size_t(lw) * lh);
  }
  CK(cudaMalloc(&S.tmp, tmpn * sizeof(float4)));
  const size_t gn = size_t(std::max(1, (S.cw + 1) / 2)) * std::max(1, (S.ch + 1) / 2);
  CK(cudaMalloc(&S.g0, gn * sizeof(float4)));
  CK(cudaMalloc(&S.g1, gn * sizeof(float4)));
  dims();
  S.samples = 0;
}

void accum_reset() {
  if (S.accum) CK(cudaMemset(S.accum, 0, size_t(S.w) * S.h * sizeof(float4)));
  if (S.alb) CK(cudaMemset(S.alb, 0, size_t(S.w) * S.h * sizeof(float4)));
  S.samples = 0;
}
int accum_count() { return S.samples; }
int render_w() { return S.w; }
int render_h() { return S.h; }

// Chooses the specialised kernel (precision x model): each one only contains the path it uses, with far fewer registers than the single kernel.
__device__ bk::ZamoC g_zamo;
__global__ void zamo_kernel(KP p) { g_zamo = bk::zamo_const(p.a, p.dist, p.theta, p.q2, p.lam); }

template <int PREC>
static void launch_prec(const KP& kp0, bool vol, int blocks) {
  KP kp = kp0;
  { void* g = nullptr; CK(cudaGetSymbolAddress(&g, g_zamo)); kp.zamo = static_cast<const bk::ZamoC*>(g); }
  zamo_kernel<<<1, 1>>>(kp);
  if (!vol) { if (kp.wh) kernel<PREC, -1, kMetricWormhole><<<blocks, 128>>>(kp); else kernel<PREC, -1><<<blocks, 128>>>(kp); return; }
  switch (kp.model) {
    case 1: kernel<PREC, 1><<<blocks, 128>>>(kp); break;
    case 2: kernel<PREC, 2><<<blocks, 128>>>(kp); break;
    case 3: kernel<PREC, 3><<<blocks, 128>>>(kp); break;
    case 4: kernel<PREC, 4><<<blocks, 128>>>(kp); break;
    case 5: kernel<PREC, 5><<<blocks, 128>>>(kp); break;
    default: kernel<PREC, 0><<<blocks, 128>>>(kp); break;
  }
}
static void launch_kernel(const KP& kp, int blocks) {
  const bool vol = kp.vol && !kp.nodisk && kp.rout > 0.0;
  if (kp.prec == 0) launch_prec<0>(kp, vol, blocks);
  else if (kp.prec == 1) launch_prec<1>(kp, vol, blocks);
  else launch_prec<2>(kp, vol, blocks);
}

float accum_sample(const Params& p, int k, Stats* stats) {
  set_size(p.w, p.h);
  KP kp = make_kp(p);
  kp.sample_k = k + p.sample_offset;
  kp.alb = p.denoise && p.denoise_albedo ? S.alb : nullptr;
  kp.stats = nullptr;
  if (stats) {
    CK(cudaMemset(S.stats, 0, 8 * sizeof(unsigned long long)));
    kp.stats = S.stats;
  }
  const int tilesx = (p.w + 15) / 16, tilesy = (p.h + 7) / 8;
  CK(cudaEventRecord(S.e0));
  launch_kernel(kp, tilesx * tilesy);
  CK(cudaEventRecord(S.e1));
  CK(cudaEventSynchronize(S.e1));
  CK(cudaGetLastError());
  float ms = 0;
  CK(cudaEventElapsedTime(&ms, S.e0, S.e1));
  ++S.samples;
  if (stats) {
    unsigned long long h[8];
    CK(cudaMemcpy(h, S.stats, sizeof h, cudaMemcpyDeviceToHost));
    stats->sanitized = int(h[3]);
    stats->stuck_rays = int(h[5]); stats->stuck_pixel = h[4] ? int(h[4]) - 1 : -1;
    stats->steps_mean = double(h[0]) / (double(p.w) * p.h);
    stats->steps_max = int(h[2]);
    stats->warp_eff = h[1] ? double(h[0]) / double(h[1]) : 1.0;
  }
  S.last.trace_ms = ms;
  return ms;
}

// average of the samples -> bloom (Gaussian pyramid) -> tone mapping -> RGBA8
float resolve(const Params& p) {
  set_size(p.w, p.h);
  const int w = S.w, h = S.h, n = w * h;
  CK(cudaEventRecord(S.e0));
  k_avg<<<(n + 255) / 256, 256>>>(S.accum, S.hdr, n, 1.f / float(std::max(1, S.samples)));
  CK(cudaEventRecord(S.e1));
  const float4* hsrc = S.hdr;
  float dn_ms = 0;
  if (p.denoise && !S.den_failed) {  // OptiX: between the HDR average and the bloom/glare/tone mapping (state created once)
    std::string err;
    if (denoise::init(S.cw, S.ch, p.denoise_albedo, err)) { S.den_ready = true; S.den_albedo = p.denoise_albedo; }  // idempotent (creates only once)
    else { std::fprintf(stderr, "denoiser: %s (continuing without denoiser)\n", err.c_str()); S.den_failed = true; S.den_ready = false; }
    if (S.den_ready) {
      if (p.denoise_albedo) k_avg<<<(n + 255) / 256, 256>>>(S.alb, S.alb_avg, n, 1.f / float(std::max(1, S.samples)));
      CK(cudaEventRecord(S.e4));
      if (denoise::run(S.hdr, p.denoise_albedo ? S.alb_avg : nullptr, S.hdr2, w, h, nullptr, err)) hsrc = S.hdr2;
      else { std::fprintf(stderr, "denoiser: %s (continuing without denoiser)\n", err.c_str()); S.den_failed = true; }
      CK(cudaEventRecord(S.e5));
      CK(cudaEventSynchronize(S.e5));
      CK(cudaEventElapsedTime(&dn_ms, S.e4, S.e5));
    }
  }
  S.hdr_last = hsrc;
  S.last.denoise_ms = dn_ms;
  const bool bloom = p.bloom > 0;
  if (bloom || p.glare > 0) {
    const dim3 blk(16, 16);
    const float4* src = hsrc;
    int sw = w, sh = h;
    for (int l = 0; l < kBloomLevels; ++l) {
      const int dw = S.lw[l], dh = S.lh[l];
      const dim3 grid((dw + 15) / 16, (dh + 15) / 16);
      k_down<<<grid, blk>>>(src, sw, sh, S.lv[l], dw, dh);
      k_blur<<<grid, blk>>>(S.lv[l], S.tmp, dw, dh, 1, 0);
      k_blur<<<grid, blk>>>(S.tmp, S.lv[l], dw, dh, 0, 1);
      src = S.lv[l]; sw = dw; sh = dh;
    }
  }
  if (p.glare > 0) {
    const dim3 blk(16, 16), grid((S.glw + 15) / 16, (S.glh + 15) / 16);
    k_bright<<<grid, blk>>>(hsrc, w, h, S.g0, S.glw, S.glh, float(p.glare_thr), float(p.glare_cap), S.lv[2], S.lw[2], S.lh[2]);
    k_streaks<<<grid, blk>>>(S.g0, S.glw, S.glh, S.g1, float(p.glare_len), int(3 * p.glare_len));
  }
  CK(cudaEventRecord(S.e2));
  BloomPtrs bp{};
  for (int l = 0; l < kBloomLevels; ++l) { bp.lv[l] = S.lv[l]; bp.lw[l] = S.lw[l]; bp.lh[l] = S.lh[l]; }
  const dim3 blk(16, 16), grid((w + 15) / 16, (h + 15) / 16);
  Polish po{float(p.glare), float(p.grain), float(p.vignette), float(p.chroma), p.frame_index, 0xA5F1u, p.glare > 0 ? S.g1 : nullptr, S.glw, S.glh, int(p.hdr), float(p.hdr_nits), float(p.hdr_peak)};
  k_final<<<grid, blk>>>(hsrc, S.out, S.out16, w, h, bp, bloom ? kBloomLevels : 0, float(p.bloom), p.accum_ldr, int(p.tonemap), po);
  CK(cudaEventRecord(S.e3));
  CK(cudaEventSynchronize(S.e3));
  CK(cudaGetLastError());
  float a = 0, b = 0, c = 0;
  CK(cudaEventElapsedTime(&a, S.e0, S.e1));
  CK(cudaEventElapsedTime(&b, S.e1, S.e2));
  CK(cudaEventElapsedTime(&c, S.e2, S.e3));
  S.last.bloom_ms = b;
  S.last.resolve_ms = a + c;
  return a + b + c;
}

float render(const Params& p, Stats* stats) {
  set_size(p.w, p.h);
  accum_reset();
  float ms = 0;
  for (int k = 0; k < std::max(1, p.samples); ++k) ms += accum_sample(p, k, k == 0 ? stats : nullptr);
  ms += resolve(p);
  return ms;
}

Timing last_timing() { return S.last; }

void hud_draw(const std::string& text, int scale, bool top_right) {
  if (!S.out || text.empty()) return;
  std::vector<std::string> lines;
  size_t pos = 0, cols = 1;
  while (pos <= text.size()) {
    const size_t nl = text.find('\n', pos);
    std::string ln = text.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
    cols = std::max(cols, ln.size());
    lines.push_back(ln);
    if (nl == std::string::npos) break;
    pos = nl + 1;
  }
  const int rows = int(lines.size());
  std::string grid(size_t(cols) * rows, ' ');
  for (int r = 0; r < rows; ++r) for (size_t c = 0; c < lines[r].size(); ++c) grid[size_t(r) * cols + c] = lines[r][c];
  const int need = int(grid.size());
  if (need > S.hud_cap) { if (S.hud_buf) CK(cudaFree(S.hud_buf)); CK(cudaMalloc(&S.hud_buf, need)); S.hud_cap = need; }
  CK(cudaMemcpy(S.hud_buf, grid.data(), need, cudaMemcpyHostToDevice));
  scale = std::max(1, scale);
  const int pad = 4 * scale, rw = int(cols) * hud::kGlyphW * scale + 2 * pad, rh = rows * hud::kGlyphH * scale + 2 * pad;
  const dim3 blk(16, 16), grid2((rw + 15) / 16, (rh + 15) / 16);
  k_hud<<<grid2, blk>>>(S.out, S.w, S.h, S.hud_buf, int(cols), rows, scale, top_right ? std::max(0, S.w - rw - 12 * scale) : 12 * scale, 12 * scale, pad);  // top_right: the interactive panel sits at the top left
  CK(cudaGetLastError());
}

void download_rgb16(std::vector<uint16_t>& rgb) {
  std::vector<ushort4> px(size_t(S.w) * S.h);
  CK(cudaMemcpy(px.data(), S.out16, px.size() * sizeof(ushort4), cudaMemcpyDeviceToHost));
  rgb.resize(px.size() * 3);
  for (size_t i = 0; i < px.size(); ++i) { rgb[3 * i] = px[i].x; rgb[3 * i + 1] = px[i].y; rgb[3 * i + 2] = px[i].z; }
}

void download_rgb(std::vector<uint8_t>& rgb) {
  std::vector<uchar4> px(size_t(S.w) * S.h);
  CK(cudaMemcpy(px.data(), S.out, px.size() * sizeof(uchar4), cudaMemcpyDeviceToHost));
  rgb.resize(px.size() * 3);
  for (size_t i = 0; i < px.size(); ++i) { rgb[3 * i] = px[i].x; rgb[3 * i + 1] = px[i].y; rgb[3 * i + 2] = px[i].z; }
}

bool gl_register(unsigned tex, int w, int h, std::string& err) {
  gl_unregister();
  cudaError_t e = cudaGraphicsGLRegisterImage(&S.gres, tex, 0x0DE1 /*GL_TEXTURE_2D*/, cudaGraphicsRegisterFlagsWriteDiscard);
  if (e != cudaSuccess) { err = cudaGetErrorString(e); S.gres = nullptr; return false; }
  S.gw = w; S.gh = h;
  return true;
}
void gl_copy() {
  if (!S.gres) return;
  CK(cudaGraphicsMapResources(1, &S.gres));
  cudaArray_t arr;
  CK(cudaGraphicsSubResourceGetMappedArray(&arr, S.gres, 0, 0));
  CK(cudaMemcpy2DToArray(arr, 0, 0, S.out, size_t(S.w) * 4, size_t(S.w) * 4, S.h, cudaMemcpyDeviceToDevice));
  CK(cudaGraphicsUnmapResources(1, &S.gres));
}
void gl_unregister() {
  if (S.gres) { cudaGraphicsUnregisterResource(S.gres); S.gres = nullptr; }
}

void trace_batch(const Params& p, const std::vector<double>& rays, std::vector<RayOut>& out, int skip) {
  const int n = int(rays.size() / 3);
  KP k = make_kp(p);
  double* dr; bk::Trace* dt;
  CK(cudaMalloc(&dr, rays.size() * sizeof(double)));
  CK(cudaMalloc(&dt, n * sizeof(bk::Trace)));
  CK(cudaMemcpy(dr, rays.data(), rays.size() * sizeof(double), cudaMemcpyHostToDevice));
  batch_kernel<<<(n + 63) / 64, 64>>>(k, dr, n, dt, skip);
  CK(cudaGetLastError());
  std::vector<bk::Trace> h(n);
  CK(cudaMemcpy(h.data(), dt, n * sizeof(bk::Trace), cudaMemcpyDeviceToHost));
  CK(cudaFree(dr)); CK(cudaFree(dt));
  out.resize(n);
  for (int i = 0; i < n; ++i)
    out[i] = RayOut{h[i].fate, h[i].disk_hit, h[i].steps, h[i].mu_end, h[i].phi_end, h[i].disk_r, h[i].disk_phi, h[i].L, h[i].Q, h[i].rejects};
}

void nt_lookup(const Params& p, const std::vector<double>& r, std::vector<double>& out) {
  KP kp = make_kp(p);
  Params q = p; q.nt = true;
  kp = make_kp(q);
  double *dr, *dout;
  CK(cudaMalloc(&dr, r.size() * sizeof(double)));
  CK(cudaMalloc(&dout, r.size() * sizeof(double)));
  CK(cudaMemcpy(dr, r.data(), r.size() * sizeof(double), cudaMemcpyHostToDevice));
  nt_kernel<<<(int(r.size()) + 63) / 64, 64>>>(kp, dr, int(r.size()), dout);
  out.resize(r.size());
  CK(cudaMemcpy(out.data(), dout, r.size() * sizeof(double), cudaMemcpyDeviceToHost));
  CK(cudaFree(dr)); CK(cudaFree(dout));
}

void vol_flight_batch(const Params& p, const std::vector<double>& rays, std::vector<double>& out) {
  const int n = int(rays.size() / 3);
  KP k = make_kp(p);
  k.slow = 1;
  double *dr, *dout;
  CK(cudaMalloc(&dr, rays.size() * sizeof(double)));
  CK(cudaMalloc(&dout, n * sizeof(double)));
  CK(cudaMemcpy(dr, rays.data(), rays.size() * sizeof(double), cudaMemcpyHostToDevice));
  vol_flight_kernel<<<(n + 63) / 64, 64>>>(k, dr, n, dout);
  CK(cudaGetLastError());
  out.resize(n);
  CK(cudaMemcpy(out.data(), dout, n * sizeof(double), cudaMemcpyDeviceToHost));
  CK(cudaFree(dr)); CK(cudaFree(dout));
}

// uploads one snapshot (3 float4 / half4 3D textures) into the given slot
static void grm_pack(const grmhd::Field& f, bool half, GrmStage& st) {
  const size_t n = size_t(f.h.n1) * f.h.n2 * f.h.n3;
  st.half = half; st.valid = false;
  auto one = [&](int t) {
    if (half) { st.h4[t].resize(n); st.f4[t].clear(); } else { st.f4[t].resize(n); st.h4[t].clear(); }
    for (size_t c = 0; c < n; ++c) {
      const float* q = &f.data[c * grmhd::kNF + 4 * t];
      if (half) st.h4[t][c] = make_ushort4(f2h(q[0]), f2h(q[1]), f2h(q[2]), f2h(q[3])); else st.f4[t][c] = make_float4(q[0], q[1], q[2], q[3]);
    }
  };
  auto a1 = std::async(std::launch::async, one, 1), a2 = std::async(std::launch::async, one, 2);
  one(0); a1.get(); a2.get();
  st.valid = true;
}

// the same pack, into caller-provided (pinned) buffers
static void grm_pack_into(const grmhd::Field& f, bool half, void* const (&dst)[3]) {
  const size_t n = size_t(f.h.n1) * f.h.n2 * f.h.n3;
  auto one = [&](int t) {
    for (size_t c = 0; c < n; ++c) {
      const float* q = &f.data[c * grmhd::kNF + 4 * t];
      if (half) static_cast<ushort4*>(dst[t])[c] = make_ushort4(f2h(q[0]), f2h(q[1]), f2h(q[2]), f2h(q[3])); else static_cast<float4*>(dst[t])[c] = make_float4(q[0], q[1], q[2], q[3]);
    }
  };
  auto a1 = std::async(std::launch::async, one, 1), a2 = std::async(std::launch::async, one, 2);
  one(0); a1.get(); a2.get();
}
static void grm_make_tex(cudaArray_t arr, cudaTextureObject_t& tex) {
  cudaResourceDesc rd = {}; rd.resType = cudaResourceTypeArray; rd.res.array.array = arr;
  cudaTextureDesc td = {};
  td.normalizedCoords = 1; td.filterMode = cudaFilterModeLinear; td.readMode = cudaReadModeElementType;
  td.addressMode[0] = cudaAddressModeWrap; td.addressMode[1] = cudaAddressModeClamp; td.addressMode[2] = cudaAddressModeClamp;  // periodic phi
  CK(cudaCreateTextureObject(&tex, &rd, &td, nullptr));
}
// frees the spare set and the pinned staging (a new first snapshot may have another grid / format)
static void grm_free_spare() {
  if (S.pf_fut.valid()) S.pf_fut.wait();  // a worker may still be packing into the pinned buffers
  if (S.grm_stream) cudaStreamSynchronize(S.grm_stream);
  for (int t = 0; t < 3; ++t) {
    if (S.grm_tex3[t]) { cudaDestroyTextureObject(S.grm_tex3[t]); S.grm_tex3[t] = 0; }
    if (S.grm_arr3[t]) { cudaFreeArray(S.grm_arr3[t]); S.grm_arr3[t] = nullptr; }
    if (S.grm_pin[t]) { cudaFreeHost(S.grm_pin[t]); S.grm_pin[t] = nullptr; }
  }
  S.grm_pin_bytes = 0; S.sp_n1 = S.sp_n2 = S.sp_n3 = 0;
}
// allocates the spare set + pinned staging for the grid of the resident snapshot (once), on the render thread
static bool grm_ensure_spare() {
  const auto& h = S.grm_h; const bool half = S.grm_half;
  if (S.grm_arr3[0] && S.sp_n1 == h.n1 && S.sp_n2 == h.n2 && S.sp_n3 == h.n3 && S.sp_half == half) return true;
  grm_free_spare();
  if (!S.grm_stream) { CK(cudaStreamCreateWithFlags(&S.grm_stream, cudaStreamNonBlocking)); CK(cudaEventCreateWithFlags(&S.grm_ev, cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&S.grm_free_ev, cudaEventDisableTiming)); }
  const size_t n = size_t(h.n1) * h.n2 * h.n3, bytes = n * (half ? sizeof(ushort4) : sizeof(float4));
  const cudaChannelFormatDesc cd = half ? cudaCreateChannelDescHalf4() : cudaCreateChannelDesc<float4>();
  const cudaExtent ext = make_cudaExtent(h.n3, h.n2, h.n1);
  for (int t = 0; t < 3; ++t) {
    if (cudaMallocHost(&S.grm_pin[t], bytes) != cudaSuccess || cudaMalloc3DArray(&S.grm_arr3[t], &cd, ext) != cudaSuccess) { cudaGetLastError(); grm_free_spare(); return false; }  // no room: the synchronous path stays
    grm_make_tex(S.grm_arr3[t], S.grm_tex3[t]);
  }
  S.grm_pin_bytes = bytes; S.sp_n1 = h.n1; S.sp_n2 = h.n2; S.sp_n3 = h.n3; S.sp_half = half;
  return true;
}

static void grm_upload(const grmhd::Field& f, cudaArray_t (&arr)[3], cudaTextureObject_t (&tex)[3], GrmStage* pre = nullptr) {
  using clk = std::chrono::steady_clock; const auto u0 = clk::now(); double t_free = 0, t_pack = 0, t_alloc = 0, t_copy = 0, t_tex = 0;  // BHS_TIMING breakdown
  auto lap = [](clk::time_point& a) { const auto b = clk::now(); const double d = 1000.0 * std::chrono::duration<double>(b - a).count(); a = b; return d; };
  auto u1 = u0;
  for (int t = 0; t < 3; ++t) {
    if (tex[t]) { cudaDestroyTextureObject(tex[t]); tex[t] = 0; }
    if (arr[t]) { cudaFreeArray(arr[t]); arr[t] = nullptr; }
  }
  t_free = lap(u1);
  const auto& h = f.h;
  const bool half = S.grm_half;
  const cudaChannelFormatDesc cd = half ? cudaCreateChannelDescHalf4() : cudaCreateChannelDesc<float4>();
  const cudaExtent ext = make_cudaExtent(h.n3, h.n2, h.n1);  // x = phi (fast), y = theta, z = r
  GrmStage local;
  if (!(pre && pre->valid && pre->half == half)) { grm_pack(f, half, local); pre = &local; }  // not prefetched (or the half-float setting changed since): pack here, in 3 threads
  t_pack = lap(u1);
  for (int t = 0; t < 3; ++t) {
    CK(cudaMalloc3DArray(&arr[t], &cd, ext));
    t_alloc += lap(u1);
    cudaMemcpy3DParms mp = {};
    mp.srcPtr = half ? make_cudaPitchedPtr(pre->h4[t].data(), h.n3 * sizeof(ushort4), h.n3, h.n2) : make_cudaPitchedPtr(pre->f4[t].data(), h.n3 * sizeof(float4), h.n3, h.n2);
    mp.dstArray = arr[t]; mp.extent = ext; mp.kind = cudaMemcpyHostToDevice;
    CK(cudaMemcpy3D(&mp));
    t_copy += lap(u1);
    cudaResourceDesc rd = {}; rd.resType = cudaResourceTypeArray; rd.res.array.array = arr[t];
    cudaTextureDesc td = {};
    td.normalizedCoords = 1; td.filterMode = cudaFilterModeLinear; td.readMode = cudaReadModeElementType;
    td.addressMode[0] = cudaAddressModeWrap; td.addressMode[1] = cudaAddressModeClamp; td.addressMode[2] = cudaAddressModeClamp;  // periodic phi
    CK(cudaCreateTextureObject(&tex[t], &rd, &td, nullptr));
    t_tex += lap(u1);
  }
  if (std::getenv("BHS_TIMING")) std::printf("  [grmhd upload] total %.1f ms: free %.1f, repack %.1f, malloc3D %.1f, memcpy3D %.1f, texture %.1f\n", 1000.0 * std::chrono::duration<double>(clk::now() - u0).count(), t_free, t_pack, t_alloc, t_copy, t_tex);
}

// v4.0: the second snapshot (the next one in time); it must have the same grid. grmhd_set_blend(tau, dt) turns the time interpolation on.
bool grmhd_load_next(const std::string& path, std::string& err) {
  if (!S.grm_loaded) { err = "load the first snapshot before the next one"; return false; }
  grmhd::Field f;
  const auto t0 = std::chrono::steady_clock::now();
  bool prefetched = false;
  if (S.pf_valid && S.pf_path == path) {  // parsed in the background while the previous snapshot was playing: only the upload is left
    S.pf_valid = false; prefetched = true;
    if (!S.pf_fut.get()) { err = S.pf_err; return false; }
    f = std::move(S.pf_field);
  } else if (!grmhd::load_bgm(path, f, err)) return false;
  if (std::getenv("BHS_TIMING")) std::printf("  [grmhd] %s %.1f ms\n", prefetched ? "prefetched (wait)" : "parse", 1000.0 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  const auto& h = f.h;
  if (h.n1 != S.grm_h.n1 || h.n2 != S.grm_h.n2 || h.n3 != S.grm_h.n3 || h.startx1 != S.grm_h.startx1) { err = "the next snapshot has another grid"; return false; }
  if (prefetched && S.pf_async) {  // O5b: the snapshot is already in (or on its way to) the spare set: wait for the copy engine, rotate next <-> spare
    CK(cudaEventSynchronize(S.grm_ev));
    for (int t = 0; t < 3; ++t) { std::swap(S.grm_arr2[t], S.grm_arr3[t]); std::swap(S.grm_tex2[t], S.grm_tex3[t]); }
    CK(cudaEventRecord(S.grm_free_ev, 0));  // the spare is now the stale set: frames that read it (queued on the default stream) must finish before the next copy into it
    S.grm_next = true; S.pf_async = false;
    if (std::getenv("BHS_TIMING")) std::printf("  [grmhd] async upload: rotation %.1f ms\n", 1000.0 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return true;
  }
  grm_upload(f, S.grm_arr2, S.grm_tex2, prefetched ? &S.pf_stage : nullptr);
  if (prefetched) S.pf_stage = GrmStage{};  // free the host copy (only if it was the one used: a worker may be filling it for another path)
  S.grm_next = true;
  return true;
}
// parses `path` on a worker thread; the next grmhd_load_next(path) then only uploads (the file read and conversion, ~13 of ~28 ms, leave the render thread)
void grmhd_prefetch(const std::string& path) {
  if (S.pf_valid) { if (S.pf_path == path) return; S.pf_fut.wait(); S.pf_valid = false; if (S.grm_stream) cudaStreamSynchronize(S.grm_stream); }  // (copies of the abandoned prefetch read the pinned buffers)
  S.pf_path = path; S.pf_valid = true; S.pf_err.clear();
  const bool half = S.grm_half;
  S.pf_async = S.grm_loaded && !std::getenv("BHS_NO_ASYNC_UPLOAD") && grm_ensure_spare();
  if (S.pf_async) {  // O5b: pack into pinned memory and enqueue the copy into the spare set; nothing is left for the render thread but a rotation
    CK(cudaStreamWaitEvent(S.grm_stream, S.grm_free_ev, 0));  // the spare was the previous first snapshot: its last frame must be done (event recorded at the rotation)
    S.pf_fut = std::async(std::launch::async, [path, half]() {
      if (!grmhd::load_bgm(path, S.pf_field, S.pf_err)) return false;
      const auto& h = S.pf_field.h;
      if (h.n1 != S.sp_n1 || h.n2 != S.sp_n2 || h.n3 != S.sp_n3) return true;  // another grid: grmhd_load_next reports it
      grm_pack_into(S.pf_field, half, S.grm_pin);
      const cudaExtent ext = make_cudaExtent(h.n3, h.n2, h.n1);
      for (int t = 0; t < 3; ++t) {
        cudaMemcpy3DParms mp = {};
        mp.srcPtr = half ? make_cudaPitchedPtr(S.grm_pin[t], h.n3 * sizeof(ushort4), h.n3, h.n2) : make_cudaPitchedPtr(S.grm_pin[t], h.n3 * sizeof(float4), h.n3, h.n2);
        mp.dstArray = S.grm_arr3[t]; mp.extent = ext; mp.kind = cudaMemcpyHostToDevice;
        if (cudaMemcpy3DAsync(&mp, S.grm_stream) != cudaSuccess) { cudaGetLastError(); S.pf_err = "async upload failed"; return false; }
      }
      cudaEventRecord(S.grm_ev, S.grm_stream);
      return true;
    });
    return;
  }
  S.pf_fut = std::async(std::launch::async, [path, half]() { if (!grmhd::load_bgm(path, S.pf_field, S.pf_err)) return false; grm_pack(S.pf_field, half, S.pf_stage); return true; });
}
// the second snapshot becomes the first (no upload): playing forward costs one load per snapshot
void grmhd_advance() {
  if (!S.grm_next) return;
  for (int t = 0; t < 3; ++t) { std::swap(S.grm_arr[t], S.grm_arr2[t]); std::swap(S.grm_tex[t], S.grm_tex2[t]); }
  S.grm_next = false; S.grm_tau = -1.0;
}
void grmhd_set_blend(double tau, double dt_M, double pattern) { S.grm_tau = tau; S.grm_dtM = dt_M; S.grm_pat = pattern; }
bool grmhd_has_next() { return S.grm_next; }

bool grmhd_load(const std::string& path, std::string& err) {
  grmhd::Field f;
  if (!grmhd::load_bgm(path, f, err)) return false;
  const auto& h = f.h;
  // O5b: the spare set and its pinned staging stay (a prefetch in flight is keyed by its path and is still valid); grm_ensure_spare rebuilds them if the grid or the format changed
  grm_upload(f, S.grm_arr, S.grm_tex);
  S.grm_next = false; S.grm_tau = -1.0;  // a new first snapshot invalidates the second
  // X2(X1, theta) LUT: 256 (X1) x 512 (theta), built in double with the same inverse as the reader (core)
  {
    if (S.grm_theta_tex) { cudaDestroyTextureObject(S.grm_theta_tex); S.grm_theta_tex = 0; }
    if (S.grm_theta_arr) { cudaFreeArray(S.grm_theta_arr); S.grm_theta_arr = nullptr; }
    constexpr int W = 256, H = 512;
    const auto fm = h.fmks();
    std::vector<float> lut(size_t(W) * H);
    for (int j = 0; j < H; ++j)
      for (int i = 0; i < W; ++i) {
        const double th = (j + 0.5) / H * M_PI, X1 = h.startx1 + (i + 0.5) / W * double(h.n1) * h.dx1;
        double X2 = 0.5;
        fm.inv_theta(X1, th, X2);
        lut[size_t(j) * W + i] = float(X2);
      }
    const cudaChannelFormatDesc c2 = cudaCreateChannelDesc<float>();
    CK(cudaMallocArray(&S.grm_theta_arr, &c2, W, H));
    CK(cudaMemcpy2DToArray(S.grm_theta_arr, 0, 0, lut.data(), W * sizeof(float), W * sizeof(float), H, cudaMemcpyHostToDevice));
    cudaResourceDesc rd = {}; rd.resType = cudaResourceTypeArray; rd.res.array.array = S.grm_theta_arr;
    cudaTextureDesc td = {};
    td.normalizedCoords = 1; td.filterMode = cudaFilterModeLinear; td.readMode = cudaReadModeElementType;
    td.addressMode[0] = cudaAddressModeClamp; td.addressMode[1] = cudaAddressModeClamp;
    CK(cudaCreateTextureObject(&S.grm_theta_tex, &rd, &td, nullptr));
  }
  S.grm_h = h; S.grm_loaded = true;
  return true;
}
bool grmhd_loaded() { return S.grm_loaded; }
void grmhd_set_half(bool on) { S.grm_half = on; }
double grmhd_spin() { return S.grm_h.a; }

void grmhd_sample_batch(const Params& p, const std::vector<double>& rmuphi, std::vector<float>& out) {
  const int n = int(rmuphi.size() / 3);
  KP k = make_kp(p);
  double* din; float* dout;
  CK(cudaMalloc(&din, rmuphi.size() * sizeof(double)));
  CK(cudaMalloc(&dout, size_t(n) * 12 * sizeof(float)));
  CK(cudaMemcpy(din, rmuphi.data(), rmuphi.size() * sizeof(double), cudaMemcpyHostToDevice));
  grm_sample_kernel<<<(n + 63) / 64, 64>>>(k, din, n, dout);
  CK(cudaGetLastError());
  out.resize(size_t(n) * 12);
  CK(cudaMemcpy(out.data(), dout, size_t(n) * 12 * sizeof(float), cudaMemcpyDeviceToHost));
  CK(cudaFree(din)); CK(cudaFree(dout));
}

void grmhd_map_batch(const Params& p, const std::vector<double>& rmu, std::vector<float>& out) {
  const int n = int(rmu.size() / 2);
  KP k = make_kp(p);
  double* din; float* dout;
  CK(cudaMalloc(&din, rmu.size() * sizeof(double)));
  CK(cudaMalloc(&dout, size_t(n) * 2 * sizeof(float)));
  CK(cudaMemcpy(din, rmu.data(), rmu.size() * sizeof(double), cudaMemcpyHostToDevice));
  grm_map_kernel<<<(n + 63) / 64, 64>>>(k, din, n, dout);
  CK(cudaGetLastError());
  out.resize(size_t(n) * 2);
  CK(cudaMemcpy(out.data(), dout, size_t(n) * 2 * sizeof(float), cudaMemcpyDeviceToHost));
  CK(cudaFree(din)); CK(cudaFree(dout));
}

void grmhd_null_batch(const Params& p, const std::vector<double>& rays, std::vector<double>& out) {
  const int n = int(rays.size() / 3);
  Params q = p; q.debug_null = true;
  KP k = make_kp(q);
  double *dr, *dout;
  CK(cudaMalloc(&dr, rays.size() * sizeof(double)));
  CK(cudaMalloc(&dout, n * sizeof(double)));
  CK(cudaMemcpy(dr, rays.data(), rays.size() * sizeof(double), cudaMemcpyHostToDevice));
  grm_null_kernel<<<(n + 63) / 64, 64>>>(k, dr, n, dout);
  CK(cudaGetLastError());
  out.resize(n);
  CK(cudaMemcpy(out.data(), dout, n * sizeof(double), cudaMemcpyDeviceToHost));
  CK(cudaFree(dr)); CK(cudaFree(dout));
}

void radio_image(const Params& p, int ss, std::vector<double>& I, std::vector<double>& ab) {
  KP k = make_kp(p);
  const size_t n = size_t(p.w) * p.h;
  double *dI, *dab;
  CK(cudaMalloc(&dI, n * sizeof(double))); CK(cudaMalloc(&dab, 2 * n * sizeof(double)));
  const dim3 blk(8, 8), grd((p.w + 7) / 8, (p.h + 7) / 8);
  radio_image_kernel<<<grd, blk>>>(k, ss, dI, dab);
  CK(cudaGetLastError());
  I.resize(n); ab.resize(2 * n);
  CK(cudaMemcpy(I.data(), dI, n * sizeof(double), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(ab.data(), dab, 2 * n * sizeof(double), cudaMemcpyDeviceToHost));
  CK(cudaFree(dI)); CK(cudaFree(dab));
}

void radio_intensity_batch(const Params& p, const std::vector<double>& rays, std::vector<double>& out) {
  const int n = int(rays.size() / 3);
  KP k = make_kp(p);
  double *dr, *dout;
  CK(cudaMalloc(&dr, rays.size() * sizeof(double)));
  CK(cudaMalloc(&dout, n * sizeof(double)));
  CK(cudaMemcpy(dr, rays.data(), rays.size() * sizeof(double), cudaMemcpyHostToDevice));
  radio_kernel<<<(n + 63) / 64, 64>>>(k, dr, n, dout);
  CK(cudaGetLastError());
  out.resize(n);
  CK(cudaMemcpy(out.data(), dout, n * sizeof(double), cudaMemcpyDeviceToHost));
  CK(cudaFree(dr)); CK(cudaFree(dout));
}

__global__ void lp_kernel(KP k, const double* r, int n, double* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = lp_log10e(k, r[i]);
}
__global__ void lpg_kernel(KP k, const double* in, int n, double* out) {  // in: r, L, Q, vr per photon
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = redshift_plunge(k, in[4 * i], in[4 * i + 1], in[4 * i + 2], in[4 * i + 3]);
}
void lamppost_plunge_g_batch(const Params& p, const std::vector<double>& rLQv, std::vector<double>& g) {
  const int n = int(rLQv.size() / 4);
  KP k = make_kp(p);
  double *din, *dout;
  CK(cudaMalloc(&din, rLQv.size() * sizeof(double))); CK(cudaMalloc(&dout, n * sizeof(double)));
  CK(cudaMemcpy(din, rLQv.data(), rLQv.size() * sizeof(double), cudaMemcpyHostToDevice));
  lpg_kernel<<<(n + 63) / 64, 64>>>(k, din, n, dout);
  CK(cudaGetLastError());
  g.resize(n);
  CK(cudaMemcpy(g.data(), dout, n * sizeof(double), cudaMemcpyDeviceToHost));
  CK(cudaFree(din)); CK(cudaFree(dout));
}
void lamppost_batch(const Params& p, const std::vector<double>& r, std::vector<double>& log10e) {
  const int n = int(r.size());
  KP k = make_kp(p);
  double *din, *dout;
  CK(cudaMalloc(&din, n * sizeof(double))); CK(cudaMalloc(&dout, n * sizeof(double)));
  CK(cudaMemcpy(din, r.data(), n * sizeof(double), cudaMemcpyHostToDevice));
  lp_kernel<<<(n + 63) / 64, 64>>>(k, din, n, dout);
  CK(cudaGetLastError());
  log10e.resize(n);
  CK(cudaMemcpy(log10e.data(), dout, n * sizeof(double), cudaMemcpyDeviceToHost));
  CK(cudaFree(din)); CK(cudaFree(dout));
}

void torus_batch(const Params& p, const std::vector<double>& rmuL, std::vector<TorusPoint>& out, bool fast) {
  const int n = int(rmuL.size() / 3);
  KP k = make_kp(p);
  double* din; TorusPoint* dout;
  CK(cudaMalloc(&din, rmuL.size() * sizeof(double)));
  CK(cudaMalloc(&dout, n * sizeof(TorusPoint)));
  CK(cudaMemcpy(din, rmuL.data(), rmuL.size() * sizeof(double), cudaMemcpyHostToDevice));
  torus_kernel<<<(n + 63) / 64, 64>>>(k, din, n, dout, fast);
  CK(cudaGetLastError());
  out.resize(n);
  CK(cudaMemcpy(out.data(), dout, n * sizeof(TorusPoint), cudaMemcpyDeviceToHost));
  CK(cudaFree(din)); CK(cudaFree(dout));
}

void turb_batch(const Params& p, const std::vector<double>& rpmt, std::vector<double>& out) {
  const int n = int(rpmt.size() / 3);
  KP k = make_kp(p);
  double *din, *dout;
  CK(cudaMalloc(&din, rpmt.size() * sizeof(double)));
  CK(cudaMalloc(&dout, 2 * n * sizeof(double)));
  CK(cudaMemcpy(din, rpmt.data(), rpmt.size() * sizeof(double), cudaMemcpyHostToDevice));
  turb_kernel<<<(n + 63) / 64, 64>>>(k, din, n, dout);
  CK(cudaGetLastError());
  out.resize(2 * n);
  CK(cudaMemcpy(out.data(), dout, 2 * n * sizeof(double), cudaMemcpyDeviceToHost));
  CK(cudaFree(din)); CK(cudaFree(dout));
}

void ntkn_batch(const Params& p, const std::vector<double>& rL, std::vector<double>& out) {
  const int n = int(rL.size() / 2);
  KP k = make_kp(p);
  double *din, *dout;
  CK(cudaMalloc(&din, rL.size() * sizeof(double))); CK(cudaMalloc(&dout, 2 * n * sizeof(double)));
  CK(cudaMemcpy(din, rL.data(), rL.size() * sizeof(double), cudaMemcpyHostToDevice));
  ntkn_kernel<<<(n + 63) / 64, 64>>>(k, din, n, dout);
  CK(cudaGetLastError());
  out.resize(2 * n);
  CK(cudaMemcpy(out.data(), dout, 2 * n * sizeof(double), cudaMemcpyDeviceToHost));
  CK(cudaFree(din)); CK(cudaFree(dout));
}

void wh_batch(const Params& p, const std::vector<double>& alphas, std::vector<double>& psi, std::vector<int>& side) {
  const int n = int(alphas.size());
  KP k = make_kp(p);
  double *da, *dp; int* ds;
  CK(cudaMalloc(&da, n * sizeof(double))); CK(cudaMalloc(&dp, n * sizeof(double))); CK(cudaMalloc(&ds, n * sizeof(int)));
  CK(cudaMemcpy(da, alphas.data(), n * sizeof(double), cudaMemcpyHostToDevice));
  wh_kernel<<<(n + 63) / 64, 64>>>(k, da, n, dp, ds);
  CK(cudaGetLastError());
  psi.resize(n); side.resize(n);
  CK(cudaMemcpy(psi.data(), dp, n * sizeof(double), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(side.data(), ds, n * sizeof(int), cudaMemcpyDeviceToHost));
  CK(cudaFree(da)); CK(cudaFree(dp)); CK(cudaFree(ds));
}

void emitter_mu_batch(double a, const std::vector<double>& rLQ, std::vector<double>& out) {
  const int n = int(rLQ.size() / 3);
  double *dr, *dout;
  CK(cudaMalloc(&dr, rLQ.size() * sizeof(double)));
  CK(cudaMalloc(&dout, n * sizeof(double)));
  CK(cudaMemcpy(dr, rLQ.data(), rLQ.size() * sizeof(double), cudaMemcpyHostToDevice));
  mu_kernel<<<(n + 63) / 64, 64>>>(a, dr, n, dout);
  CK(cudaGetLastError());
  out.resize(n);
  CK(cudaMemcpy(out.data(), dout, n * sizeof(double), cudaMemcpyDeviceToHost));
  CK(cudaFree(dr)); CK(cudaFree(dout));
}

void boost_batch(const std::vector<double>& np, const double beta[3], std::vector<double>& out) {
  const int n = int(np.size() / 3);
  double *dn, *dout;
  CK(cudaMalloc(&dn, np.size() * sizeof(double)));
  CK(cudaMalloc(&dout, size_t(n) * 4 * sizeof(double)));
  CK(cudaMemcpy(dn, np.data(), np.size() * sizeof(double), cudaMemcpyHostToDevice));
  boost_kernel<<<(n + 63) / 64, 64>>>(dn, beta[0], beta[1], beta[2], n, dout);
  out.resize(size_t(n) * 4);
  CK(cudaMemcpy(out.data(), dout, out.size() * sizeof(double), cudaMemcpyDeviceToHost));
  CK(cudaFree(dn)); CK(cudaFree(dout));
}

void camera_rays(const Params& p, const std::vector<double>& xy, std::vector<double>& out) {
  const int n = int(xy.size() / 2);
  KP kp = make_kp(p);
  double *dxy, *dout;
  CK(cudaMalloc(&dxy, xy.size() * sizeof(double)));
  CK(cudaMalloc(&dout, size_t(n) * 3 * sizeof(double)));
  CK(cudaMemcpy(dxy, xy.data(), xy.size() * sizeof(double), cudaMemcpyHostToDevice));
  cam_kernel<<<(n + 63) / 64, 64>>>(kp, dxy, n, dout);
  out.resize(size_t(n) * 3);
  CK(cudaMemcpy(out.data(), dout, out.size() * sizeof(double), cudaMemcpyDeviceToHost));
  CK(cudaFree(dxy)); CK(cudaFree(dout));
}

bool accum_finite() {
  int* d; int bad = 0;
  CK(cudaMalloc(&d, sizeof(int)));
  CK(cudaMemset(d, 0, sizeof(int)));
  const int n = S.w * S.h;
  finite_kernel<<<(n + 255) / 256, 256>>>(S.accum, n, d);
  CK(cudaMemcpy(&bad, d, sizeof(int), cudaMemcpyDeviceToHost));
  CK(cudaFree(d));
  return bad == 0;
}

}  // namespace bhs::kcuda
