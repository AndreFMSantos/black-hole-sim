// v0.6b — Kerr render in CUDA. One thread per pixel runs exactly the shade() of v0.6a (CPU).
// Ver notes/physics/06b-kerr-cuda.md.
#include <cuda_fp16.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <bhs/exr_loader.hpp>
#include <bhs/kerr.hpp>
#include <bhs/lut.hpp>
#include <bhs/turbulence.hpp>
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
struct KP {
  int w, h, use_sky, sky_raw, nodisk, nodoppler, max_steps, prec;
  double phi_atol, l_min;
  double a, dist, theta, phi, tan_half, aspect, rin, rout, exposure, sky_exposure, tol, r_far;
  double tpeak, fnorm;
  // v0.7d
  int nt, nt_n, ldr, sample_k, limb;
  uint32_t seed;
  double nt_rin, nt_rout, bx, by, bz, clamp_lum, beaming;
  double turb_amp, turb_scale, disk_time, spot_r, spot_ph, spot_size, spot_amp, spot_om, fl_t, fl_w, fl_boost;  // v0.9c (spot_ph = phase at time disk_time, computed on the host; spot_amp = amp without flare)
  int slow;  // v0.9c2: slow light
  int vol, v_sub;                // v0.9b: thick disk (Gaussian layer + radiative transport)
  double v_hr, v_flare, v_tau;
  double edge_soft, edge_noise;  // v0.9a2: soft / irregular outer edge (0 = v0.8 hard cut)
  double fwd[3], right[3], up[3];  // v0.8b: camera basis in (e_r, e_theta, e_phi): d' = fwd + x right + y up, n' = -d'/|d'|
  const float* nt_tab;
  float4* accum;
  float4* alb;  // albedo accumulator (nullptr = off)
  const float* lut;
  cudaTextureObject_t sky;
  unsigned long long* stats;  // [0] sum of steps, [1] sum of per-warp maxima * 32, [2] maximum
};

constexpr double kT0 = lut::kT0, kT1 = lut::kT1;

__device__ double3 lut_lookup(const float* d, double T) {
  if (T <= kT0) return make_double3(0, 0, 0);
  const double x = log(fmin(T, kT1) / kT0) / log(kT1 / kT0) * (lut::kN - 1);
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
__device__ double redshift(double a, double r, double L, double r_obs, double th_obs) {
  const double s = sin(th_obs), mu = cos(th_obs);
  const double Sigma = r_obs * r_obs + a * a * mu * mu, Delta = bk::delta_d(r_obs, a);
  const double rr = r_obs * r_obs + a * a;
  const double A = rr * rr - a * a * Delta * s * s;
  const double alpha = sqrt(Sigma * Delta / A), omega = 2.0 * a * r_obs / A;
  const double E_obs = (1.0 - omega * L) / alpha;
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
  const double lmin = log(p.nt_rin), lmax = log(p.nt_rout);
  const double x = (log(r) - lmin) / (lmax - lmin) * (p.nt_n - 1);
  const int i = min(p.nt_n - 2, int(x));
  const double f = x - i;
  return (1 - f) * double(p.nt_tab[i]) + f * double(p.nt_tab[i + 1]);
}

// disk_emission.hpp: mu in the gas frame and Eddington limb darkening (flux conserved)
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
  const float inv = 1.f / (float(p.turb_scale) * h);
  const float n = fbm_f(cs * inv, sn * inv, (logf(r) + 1.7f * mu) * inv);
  const float a = float(p.turb_amp);
  return __expf(a * n - 0.5f * a * a * float(turb::kVar));
}
// hot-spot T factor at (r, phi, mu): (1 + amp G)^(1/4); G Gaussian in the Euclidean distance to the centre (phase and amp already at the right time)
__device__ float spot_factor_f(const KP& p, float r, float phi, float mu, float t) {
  if (p.spot_amp <= 0.0) return 1.f;
  const float rh = float(p.spot_r), z = r * mu;
  // phase and flare at time t (= disk_time without slow light): the phase at time disk_time is exact (host, double), only the difference t - disk_time is corrected
  const float ph = float(p.spot_ph) + float(p.spot_om) * (t - float(p.disk_time));
  float amp = float(p.spot_amp);
  if (p.fl_boost != 0.0) { const float d = (t - float(p.fl_t)) / float(p.fl_w); amp *= 1.f + float(p.fl_boost) * __expf(-0.5f * d * d); }
  const float d2 = r * r + rh * rh - 2.f * r * rh * cosf(phi - ph) + z * z;
  const float s = float(p.spot_size);
  return powf(1.f + amp * __expf(-0.5f * d2 / (s * s)), 0.25f);
}

struct VolAcc {
  static constexpr bool kActive = true;
  const KP& p;
  float dop, eobs = 1.f, L = 0.f, trans = 1.f;
  float T = 0.f, tref = 0.f, T_half = -1.f;  // v0.9c2: flight time accumulated since the camera; T_half = T when trans < 1/2 (test)
  bool slow = false;
  float3 I = make_float3(0.f, 0.f, 0.f);
  int nsub;
  __device__ VolAcc(const KP& pp, double d) : p(pp), dop(float(d)), nsub(pp.v_sub), slow(pp.slow != 0) {}
  // E_obs = (1 - omega L)/alpha (ZAMO observer) only depends on the radius, not on the point: g = E_obs dop / e_emit
  __device__ void begin(double L_, double Q_) {
    const double Lg = p.nodoppler ? 0.0 : L_;
    L = float(Lg);
    const double s = sin(p.theta), mu = cos(p.theta), Sigma = p.dist * p.dist + p.a * p.a * mu * mu, Delta = bk::delta_d(p.dist, p.a);
    const double rr = p.dist * p.dist + p.a * p.a, A = rr * rr - p.a * p.a * Delta * s * s;
    eobs = float(((1.0 - 2.0 * p.a * p.dist / A * Lg) / sqrt(Sigma * Delta / A)));
    (void)Q_;
    // time reference: the flight time of a radial ray from the camera to r = 10 in Schwarzschild, r*(D) - r*(10), r* = r + 2 ln(r/2 - 1)
    // (convention: t_obs = disk_time is the instant at which light emitted at r = 10 along the line of sight would arrive; it only fixes the origin)
    tref = float((p.dist + 2.0 * log(p.dist / 2.0 - 1.0)) - (10.0 + 2.0 * log(4.0)));
  }
  __device__ float sigma(float r) const { return p.v_flare == 0.0 ? float(p.v_hr) : float(p.v_hr) * powf(r * 0.1f, float(p.v_flare)); }
  __device__ bool opaque() const { return trans < 1e-3f; }
  __device__ void add(float r, float mu, float phi, float dM) {
    if (r <= float(p.rin)) return;
    const float rout = float(p.rout), es = float(p.edge_soft);
    float w = 1.f;
    if (es > 0.f) { const float t = fminf(1.f, fmaxf(0.f, (r - rout * (1.f - es)) / (rout * 2.f * es))); w = 1.f - t * t * (3.f - 2.f * t); }
    else if (r >= rout) return;
    if (w <= 0.f) return;
    const float H = sigma(r) * r, z = r * mu;
    float alpha = w * float(p.v_tau) / (2.50662827f * H) * __expf(-0.5f * z * z / (H * H));
    if (alpha < 1e-9f) return;
    const float rs = sqrtf(r), r32 = r * rs, r34 = rs * sqrtf(rs), af = float(p.a);
    const float phd = phi + float(p.phi);  // azimuth in the disk (the camera is at phi = p.phi)
    const float tem = slow ? float(p.disk_time) - (T - tref) : float(p.disk_time);  // emission time
    const float tdf = turb_density_f(p, r, phd, mu, sigma(r), 1.f / (r32 + af), tem);
    alpha *= tdf;
    const float ut = (r32 + af) / (r34 * sqrtf(r32 - 3.f * rs + 2.f * af));
    const float e_emit = fabsf(ut * (1.f - L / (r32 + af)));
    const float Sig = r * r + af * af * mu * mu;
    const float dtau = alpha * e_emit * Sig * dM;
    float g = eobs * dop / e_emit;
    if (p.beaming != 1.0) g = powf(g, float(p.beaming));
    // NT temperature profile (table in ln r) or Newtonian, in float
    float Tprof;
    if (p.nt) {
      if (r >= float(p.nt_rout)) Tprof = p.nt_tab[p.nt_n - 1] * powf(r / float(p.nt_rout), -0.75f);
      else {
        const float lmin = logf(float(p.nt_rin)), lmax = logf(float(p.nt_rout)), x = (logf(r) - lmin) / (lmax - lmin) * (p.nt_n - 1);
        const int i = min(p.nt_n - 2, int(x));
        const float f = x - i;
        Tprof = (1.f - f) * p.nt_tab[i] + f * p.nt_tab[i + 1];
      }
    } else {
      Tprof = powf(powf(r, -3.f) * (1.f - sqrtf(float(p.rin) / r)) / float(p.fnorm), 0.25f);
    }
    const float Tobs = g * float(p.tpeak) * Tprof * spot_factor_f(p, r, phd, mu, tem);  // (do not call it T: it shadows the accumulated flight time, member T)
    float3 c = make_float3(0.f, 0.f, 0.f);
    if (Tobs > float(kT0)) {  // colour LUT (linear interpolation in log T, like lut_lookup)
      const float xx = logf(fminf(Tobs, float(kT1)) / float(kT0)) / logf(float(kT1 / kT0)) * (lut::kN - 1);
      const int i = min(lut::kN - 2, int(xx));
      const float f = xx - i;
      c = make_float3((1.f - f) * p.lut[3 * i] + f * p.lut[3 * i + 3], (1.f - f) * p.lut[3 * i + 1] + f * p.lut[3 * i + 4], (1.f - f) * p.lut[3 * i + 2] + f * p.lut[3 * i + 5]);
    }
    const float wgt = trans * (-expm1f(-dtau)) * float(p.exposure);
    I.x += wgt * c.x; I.y += wgt * c.y; I.z += wgt * c.z;
    trans *= __expf(-dtau);
    if (T_half < 0.f && trans < 0.5f) T_half = T;
  }
};

__device__ __forceinline__ void trace_dispatch_vol(const KP& p, double nr, double nth, double nph, double rout, double atol, bk::Trace& tr, VolAcc& vol) {
  const double ap = p.phi_atol > 0 ? p.phi_atol : atol;
  if (p.prec == 0) bk::trace<double, double, VolAcc>(p.a, p.dist, p.theta, nr, nth, nph, p.rin, rout, p.tol, atol, ap, p.l_min, p.r_far, p.max_steps, tr, 0, &vol);
  else if (p.prec == 1) bk::trace<float, float, VolAcc>(p.a, p.dist, p.theta, nr, nth, nph, p.rin, rout, p.tol, atol, ap, p.l_min, p.r_far, p.max_steps, tr, 0, &vol);
  else bk::trace<double, float, VolAcc>(p.a, p.dist, p.theta, nr, nth, nph, p.rin, rout, p.tol, atol, ap, p.l_min, p.r_far, p.max_steps, tr, 0, &vol);
}

__device__ __forceinline__ void trace_dispatch(const KP& p, double nr, double nth, double nph, double rout, double atol, bk::Trace& tr, int skip = 0) {
  const double ap = p.phi_atol > 0 ? p.phi_atol : atol;
  if (p.prec == 0) bk::trace<double, double>(p.a, p.dist, p.theta, nr, nth, nph, p.rin, rout, p.tol, atol, ap, p.l_min, p.r_far, p.max_steps, tr, skip);
  else if (p.prec == 1) bk::trace<float, float>(p.a, p.dist, p.theta, nr, nth, nph, p.rin, rout, p.tol, atol, ap, p.l_min, p.r_far, p.max_steps, tr, skip);
  else bk::trace<double, float>(p.a, p.dist, p.theta, nr, nth, nph, p.rin, rout, p.tol, atol, ap, p.l_min, p.r_far, p.max_steps, tr, skip);
}

// Each thread = 1 pixel and 1 sub-pixel sample (R2 jitter per quad, sample index sample_k; sample 0 is the
// pixel centre). Adds the HDR colour into p.accum (RGBA32F); the resolve (average, bloom, tone mapping) is separate.
// 1D blocks; 4 consecutive threads form a 2x2 quad (horizontal neighbour = lane^1, vertical = lane^2) to
// derive the sky-uv gradient via __shfl_sync, like v0.5b's subgroup_quad. The jitter is the same in the 4
// threads of the quad (quad hash), so that the uv difference between neighbours remains the pixel's derivative.
// One warp (32) = 8 quads = an 8x4-pixel tile; a block of 128 = 4 warps = a 16x8 tile.
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
  int steps = 0;
  if (inside) {
    const double x = (2.0 * ((px + 0.5 + jx) / p.w) - 1.0) * p.tan_half * p.aspect;
    const double y = (1.0 - 2.0 * ((py + 0.5 + jy) / p.h)) * p.tan_half;
    const double rout = p.nodisk ? 0.0 : p.rout;
    const double atol = p.tol * 0.1;
    // moving observer: n' (camera basis) in the observer frame -> n in the ZAMO (aberration)
    double3 nz; double dop;
    observer_to_zamo(camera_ray(p, x, y), p.bx, p.by, p.bz, nz, dop);
    bk::Trace tr;
    // v0.9a2: soft/irregular edge. The disk becomes semi-transparent in the fringe: the ray continues behind it
    // (skip = layers already shaded) and the layers add up with transmittance Tr. No fringe = the v0.8 path.
    const bool soft = p.edge_soft > 0.0 || p.edge_noise > 0.0;
    const bool volmode = p.vol && rout > 0.0;
    const double rtrace = (soft && rout > 0.0) ? rout * (1.0 + p.edge_noise) * (1.0 + p.edge_soft) : rout;
    if (volmode) {
      VolAcc vol(p, dop);
      trace_dispatch_vol(p, nz.x, nz.y, nz.z, rout, atol, tr, vol);
      base = make_double3(vol.I.x, vol.I.y, vol.I.z); trans = vol.opaque() ? 0.0 : double(vol.trans); col = base;
    } else {
      trace_dispatch(p, nz.x, nz.y, nz.z, rtrace, atol, tr);
    }
    steps = tr.steps;
    for (int layer = 0; !volmode && tr.disk_hit; ++layer) {
      const double r = tr.disk_r;
      double wgt = 1.0;
      if (soft) {
        const double phe = tr.disk_phi + p.phi;
        const double ne = 0.6 * vnoise(3 * cos(phe), 3 * sin(phe), 1.7) + 0.4 * vnoise(8 * cos(phe), 8 * sin(phe), 5.3);
        const double re = rout * (1.0 + p.edge_noise * (2.0 * ne - 1.0));
        wgt = p.edge_soft > 0.0 ? 1.0 - ss(re * (1.0 - p.edge_soft), re * (1.0 + p.edge_soft), r) : (r < re ? 1.0 : 0.0);
      }
      double g = p.nodoppler ? redshift(p.a, r, 0.0, p.dist, p.theta) : redshift(p.a, r, tr.L, p.dist, p.theta) * dop;
      if (p.beaming != 1.0) g = pow(g, p.beaming);  // --beaming: aesthetic (not physical) choice, see the notes
      const double Tprof = p.nt ? nt_temp(p, r) : disk_temp(r, p.rin, p.fnorm);
      const double T_obs = g * p.tpeak * Tprof;
      const double psi = tr.disk_phi + p.phi - 2.2 * log(r / p.rin);
      const double lr = log(r);
      const double n = 0.55 * vnoise(3 * cos(psi), 3 * sin(psi), 9 * lr) + 0.45 * vnoise(9 * cos(psi), 9 * sin(psi), 28 * lr);
      double k = p.exposure * (0.55 + (1.15 - 0.55) * n);
      if (p.limb) k *= limb_eddington_d(emitter_mu_d(p.a, r, tr.L, tr.Q));
      const double3 c = lut_lookup(p.lut, T_obs);
      if (!soft) { col = make_double3(k * c.x, k * c.y, k * c.z); break; }
      base.x += trans * wgt * k * c.x; base.y += trans * wgt * k * c.y; base.z += trans * wgt * k * c.z;
      trans *= 1.0 - wgt;
      if (trans < 1e-3 || layer >= 2) { trans = 0.0; break; }
      trace_dispatch(p, nz.x, nz.y, nz.z, rtrace, atol, tr, layer + 1);  // continues behind the gas
      steps += tr.steps;
    }
    if (soft && !volmode) col = base;
    if (!tr.disk_hit && trans > 0.0 && tr.fate == 0 && p.use_sky) {
      const double ph = tr.phi_end + p.phi;  // axisymmetry: the camera azimuth only adds
      double uu = 0.5 - ph / (2.0 * M_PI);
      uu -= floor(uu);
      u = float(uu);
      v = float(acos(tr.mu_end) / M_PI);
      sky_ok = true;
    }
  }
  __syncwarp();
  if (p.use_sky) {  // uniform flow: all threads of the warp execute the shuffles
    const unsigned full = 0xffffffffu;
    const float uH = __shfl_xor_sync(full, u, 1), vH = __shfl_xor_sync(full, v, 1);
    const float uV = __shfl_xor_sync(full, u, 2), vV = __shfl_xor_sync(full, v, 2);
    const int okH = __shfl_xor_sync(full, int(sky_ok), 1), okV = __shfl_xor_sync(full, int(sky_ok), 2);
    if (sky_ok) {
      const float sx = lx ? -1.f : 1.f, sy = ly ? -1.f : 1.f;  // the neighbour is at +1 if lx == 0, at -1 if lx == 1
      auto wrap = [](float d) { return d - rintf(d); };
      float2 gx = okH ? make_float2(sx * wrap(uH - u), sx * (vH - v)) : make_float2(0.f, 0.f);
      float2 gy = okV ? make_float2(sy * wrap(uV - u), sy * (vV - v)) : make_float2(0.f, 0.f);
      const float4 s = p.sky_raw ? tex2DLod<float4>(p.sky, u, v, 0.f) : tex2DGrad<float4>(p.sky, u, v, gx, gy);
      col = trans == 1.0 ? make_double3(p.sky_exposure * s.x, p.sky_exposure * s.y, p.sky_exposure * s.z)
                         : make_double3(base.x + trans * p.sky_exposure * s.x, base.y + trans * p.sky_exposure * s.y, base.z + trans * p.sky_exposure * s.z);
    }
  }
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
__global__ void k_bright(const float4* hdr, int w, int h, float4* g0, int gw, int gh, float thr, const float4* coarse, int cw, int ch) {
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
      const float3 d = make_float3(fmaxf(0.f, c.x - m.x), fmaxf(0.f, c.y - m.y), fmaxf(0.f, c.z - m.z));
      const float L = 0.2126f * d.x + 0.7152f * d.y + 0.0722f * d.z;
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
};

__global__ void k_final(const float4* hdr, uchar4* out, int w, int h, BloomPtrs bp, int levels, float bloom, int ldr, int tonemap, Polish po) {
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
__global__ void turb_kernel(KP p, const double* in, int n, double* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float r = float(in[3 * i]), phi = float(in[3 * i + 1]), mu = float(in[3 * i + 2]);
  const float h = float(p.v_hr) * (p.v_flare == 0.0 ? 1.f : powf(r * 0.1f, float(p.v_flare)));
  const float r32 = r * sqrtf(r);
  out[2 * i] = turb_density_f(p, r, phi, mu, h, 1.f / (r32 + float(p.a)), float(p.disk_time));
  out[2 * i + 1] = spot_factor_f(p, r, phi, mu, float(p.disk_time));
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
struct State {
  float* lut = nullptr;
  std::vector<float> lutd, lut1;  // lutd: LUT of the current f_col; lut1: reference f = 1 (physical exposure)
  double lut_fcol = 1.0;
  cudaMipmappedArray_t mm = nullptr;
  cudaTextureObject_t sky = 0;
  bool has_sky = false;
  uchar4* out = nullptr;
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
  double nt_a = -1, nt_rin = 0, nt_rout = 0;
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
  if (!S.nt_tab || std::abs(p.a - S.nt_a) > 1e-9 || std::abs(rout_prof - S.nt_rout) > 1e-9) {
    const nt::Profile prof(p.a, rout_prof);
    std::vector<float> tab(prof.table().begin(), prof.table().end());
    if (!S.nt_tab) CK(cudaMalloc(&S.nt_tab, 2048 * sizeof(float)));
    CK(cudaMemcpy(S.nt_tab, tab.data(), tab.size() * sizeof(float), cudaMemcpyHostToDevice));
    S.nt_n = int(tab.size()); S.nt_a = p.a; S.nt_rout = rout_prof; S.nt_rin = prof.r_in();
  }
  k.nt_tab = S.nt_tab; k.nt_n = S.nt_n; k.nt_rin = S.nt_rin; k.nt_rout = S.nt_rout;
}

KP make_kp(const Params& p) {
  KP k{};
  k.w = p.w; k.h = p.h; k.use_sky = p.use_sky && S.has_sky; k.sky_raw = p.sky_raw; k.nodisk = p.nodisk; k.nodoppler = p.nodoppler;
  k.max_steps = p.max_steps; k.prec = int(p.prec);
  k.a = p.a; k.dist = p.dist; k.theta = p.theta_deg * M_PI / 180.0; k.phi = p.phi_deg * M_PI / 180.0;
  k.tan_half = std::tan(0.5 * p.fov_deg * M_PI / 180.0);
  k.aspect = double(p.w) / p.h;
  k.rin = p.rin < 0 ? kerr::isco(p.a, true) : p.rin;
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
  k.tol = p.tol; k.r_far = p.r_far; k.tpeak = p.tpeak;
  auto f = [&](double x) { return std::pow(x, -3.0) * (1.0 - std::sqrt(k.rin / x)); };
  k.fnorm = f(49.0 / 36.0 * k.rin);
  k.lut = S.lut; k.sky = S.sky;
  k.nt = p.nt; k.ldr = p.accum_ldr; k.seed = p.seed;
  k.bx = p.beta[0]; k.by = p.beta[1]; k.bz = p.beta[2];
  k.clamp_lum = p.clamp_lum; k.beaming = p.beaming; k.limb = p.limb;
  k.edge_soft = p.edge_soft; k.edge_noise = p.edge_noise;
  {  // v0.9c: spot phase and amplitude at the disk time (on the host, in double, reduced to [0, 2 pi))
    turb::Spot sp; sp.r = p.spot_r; sp.phi0 = p.spot_phi_deg * M_PI / 180.0; sp.amp = p.spot_amp; sp.fl_t = p.flare_t; sp.fl_w = p.flare_w; sp.fl_boost = p.flare_boost;
    k.spot_ph = std::fmod(turb::spot_phase(sp, p.a, p.disk_time), 2.0 * M_PI); k.spot_amp = p.spot_amp; k.spot_om = turb::omega(p.spot_r, p.a);
    k.fl_t = p.flare_t; k.fl_w = p.flare_w; k.fl_boost = p.flare_boost; k.slow = p.slowlight;
    k.spot_r = p.spot_r; k.spot_size = p.spot_size; k.turb_amp = p.turb; k.turb_scale = p.turb_scale; k.disk_time = p.disk_time;
  }
  k.vol = p.hr > 0.0; k.v_hr = p.hr; k.v_flare = p.flare; k.v_tau = p.tau; k.v_sub = std::max(1, p.vol_sub);
  for (int i = 0; i < 3; ++i) { k.fwd[i] = p.fwd[i]; k.right[i] = p.right[i]; k.up[i] = p.up[i]; }
  if (p.nt) ensure_nt(p, k);
  k.accum = S.accum;
  return k;
}

}  // namespace

bool init(const std::string& sky_path, std::string& err) {
  CK(cudaFree(nullptr));
  S.lutd = lut::build();
  S.lut1 = S.lutd; S.lut_fcol = 1.0;
  const std::vector<float>& lutd = S.lutd;
  CK(cudaMalloc(&S.lut, lutd.size() * sizeof(float)));
  CK(cudaMemcpy(S.lut, lutd.data(), lutd.size() * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&S.stats, 3 * sizeof(unsigned long long)));
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
  sky::Skymap m;
  if (!sky::load_exr_half(sky_path, m, &err)) return false;
  int levels = 1;
  for (int n = std::max(m.w, m.h); n > 1; n >>= 1) ++levels;
  cudaChannelFormatDesc cd = cudaCreateChannelDescHalf4();
  CK(cudaMallocMipmappedArray(&S.mm, &cd, make_cudaExtent(m.w, m.h, 0), levels));
  // level 0: RGB half -> RGBA half; following levels: 2x2 mean in float (linear HDR values)
  std::vector<uint16_t> cur(size_t(m.w) * m.h * 4), nxt;
  for (size_t i = 0; i < size_t(m.w) * m.h; ++i) {
    cur[4 * i] = m.rgb[3 * i]; cur[4 * i + 1] = m.rgb[3 * i + 1]; cur[4 * i + 2] = m.rgb[3 * i + 2]; cur[4 * i + 3] = f2h(1.f);
  }
  m.rgb.clear(); m.rgb.shrink_to_fit();
  int cw = m.w, ch = m.h;
  for (int l = 0; l < levels; ++l) {
    cudaArray_t arr;
    CK(cudaGetMipmappedArrayLevel(&arr, S.mm, l));
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
  rd.res.mipmap.mipmap = S.mm;
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
  CK(cudaCreateTextureObject(&S.sky, &rd, &td, nullptr));
  S.has_sky = true;
  std::printf("sky: %s (%dx%d, %d levels, aniso 16x)\n", sky_path.c_str(), m.w, m.h, levels);
  return true;
}

bool has_sky() { return S.has_sky; }
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
  if (S.out) { CK(cudaFree(S.out)); CK(cudaFree(S.accum)); CK(cudaFree(S.hdr)); CK(cudaFree(S.tmp)); CK(cudaFree(S.g0)); CK(cudaFree(S.g1)); CK(cudaFree(S.hdr2)); CK(cudaFree(S.alb)); CK(cudaFree(S.alb_avg)); for (auto* l : S.lv) CK(cudaFree(l)); }
  S.cw = std::max(w, S.cw); S.ch = std::max(h, S.ch);
  const size_t n = size_t(S.cw) * S.ch;
  CK(cudaMalloc(&S.out, n * sizeof(uchar4)));
  CK(cudaMemset(S.out, 0, n * sizeof(uchar4)));
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

float accum_sample(const Params& p, int k, Stats* stats) {
  set_size(p.w, p.h);
  KP kp = make_kp(p);
  kp.sample_k = k + p.sample_offset;
  kp.alb = p.denoise && p.denoise_albedo ? S.alb : nullptr;
  kp.stats = nullptr;
  if (stats) {
    CK(cudaMemset(S.stats, 0, 3 * sizeof(unsigned long long)));
    kp.stats = S.stats;
  }
  const int tilesx = (p.w + 15) / 16, tilesy = (p.h + 7) / 8;
  CK(cudaEventRecord(S.e0));
  kernel<<<tilesx * tilesy, 128>>>(kp);
  CK(cudaEventRecord(S.e1));
  CK(cudaEventSynchronize(S.e1));
  CK(cudaGetLastError());
  float ms = 0;
  CK(cudaEventElapsedTime(&ms, S.e0, S.e1));
  ++S.samples;
  if (stats) {
    unsigned long long h[3];
    CK(cudaMemcpy(h, S.stats, sizeof h, cudaMemcpyDeviceToHost));
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
    k_bright<<<grid, blk>>>(hsrc, w, h, S.g0, S.glw, S.glh, float(p.glare_thr), S.lv[2], S.lw[2], S.lh[2]);
    k_streaks<<<grid, blk>>>(S.g0, S.glw, S.glh, S.g1, float(p.glare_len), int(3 * p.glare_len));
  }
  CK(cudaEventRecord(S.e2));
  BloomPtrs bp{};
  for (int l = 0; l < kBloomLevels; ++l) { bp.lv[l] = S.lv[l]; bp.lw[l] = S.lw[l]; bp.lh[l] = S.lh[l]; }
  const dim3 blk(16, 16), grid((w + 15) / 16, (h + 15) / 16);
  Polish po{float(p.glare), float(p.grain), float(p.vignette), float(p.chroma), p.frame_index, 0xA5F1u, p.glare > 0 ? S.g1 : nullptr, S.glw, S.glh};
  k_final<<<grid, blk>>>(hsrc, S.out, w, h, bp, bloom ? kBloomLevels : 0, float(p.bloom), p.accum_ldr, int(p.tonemap), po);
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

void hud_draw(const std::string& text, int scale) {
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
  k_hud<<<grid2, blk>>>(S.out, S.w, S.h, S.hud_buf, int(cols), rows, scale, 12 * scale, 12 * scale, pad);
  CK(cudaGetLastError());
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
    out[i] = RayOut{h[i].fate, h[i].disk_hit, h[i].steps, h[i].mu_end, h[i].phi_end, h[i].disk_r, h[i].disk_phi, h[i].L, h[i].Q};
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
